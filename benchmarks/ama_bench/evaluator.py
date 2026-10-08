"""AMA-Bench evaluation module.

Follows the LoCoMoEvaluator pattern: memory injection via chunked sessions,
per-query evaluation using LLM-as-Judge with binary yes/no scoring matching
the official AMA-Bench implementation.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager
from src.result import EvaluationReport, ResultCollector
from benchmarks.ama_bench.dataset import AMABenchDataset, AMABenchQuery, AMABenchSession
from benchmarks.base import EvaluationUnit
from methods.base import MemoryBuildResult
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from utils.templates import get_prompt_manager
from utils.llm_client import get_usage_tracker


DEFAULT_MEMORY_CHUNK_SIZE = 32000


class AMABenchEvaluator:

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Path,
        dry_run: bool = False,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
        resume: bool = False,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = logger
        self.resume = resume

        self.prompt_manager = get_prompt_manager(
            dataset=dataset_config.dataset_name,
            method=method_config.method_name,
        )

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[AMABenchDataset] = None

        api_config = get_api_config()
        self.metrics_calculator = MetricsCalculator(
            dataset="ama_bench",
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            language="en",
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []

        eval_config = dataset_config.raw_config.get("evaluation", {})
        self.memory_chunk_size = eval_config.get("memory_chunk_size", DEFAULT_MEMORY_CHUNK_SIZE)

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    def _init_dataset(self) -> None:
        self._log(f"Loading dataset: {self.dataset_config.dataset_name}")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        eval_config = self.dataset_config.raw_config.get("evaluation", {})

        self.dataset = AMABenchDataset(
            data_dir=data_dir,
            config={
                "data_file": self.dataset_config.raw_config.get("data", {}).get("data_file", "open_end_qa_set.jsonl"),
                "max_samples": eval_config.get("max_samples"),
                "episode_ids": eval_config.get("episode_ids"),
                "domain_filter": eval_config.get("domain_filter"),
                "qa_type_filter": eval_config.get("qa_type_filter"),
            },
        )
        self.dataset.load()

        self._log(f"  Total Episodes: {len(self.dataset.get_episode_ids())}")
        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Total Queries: {self.dataset.get_total_queries()}")
        self._log(f"  Domain Distribution: {self.dataset.get_domain_distribution()}")
        self._log(f"  QA Type Distribution: {self.dataset.get_qa_type_distribution()}")
        self._log(f"  Memory Chunk Size: {self.memory_chunk_size:,} chars (~{self.memory_chunk_size // 4:,} tokens)")

    def _init_agent_for_context(self, context_id: Any, force_new: bool = True) -> None:
        if force_new or self.agent_manager is None:
            if self.agent_manager is not None:
                try:
                    self.agent_manager.reset()
                except Exception as e:
                    self._log(f"Warning: Failed to reset old agent: {e}", level="WARNING")

            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
            )

        self.agent_manager.set_context_id(context_id)

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()
        get_usage_tracker().reset()

        self._init_dataset()
        self._run_evaluation_loop()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)
        return report

    def _run_evaluation_loop(self) -> None:
        for unit in self.dataset.get_evaluation_units():
            episode_id = unit.context_id
            domain = unit.metadata.get("domain", "unknown")
            task_type = unit.metadata.get("task_type", "unknown")
            num_turns = unit.metadata.get("num_turns", 0)

            self._log(f"\n{'='*60}")
            self._log(f"Episode: {episode_id} (domain={domain}, task_type={task_type}, turns={num_turns})")
            self._log(f"  Sessions: {len(unit.sessions_to_inject)}")
            self._log(f"  Queries: {len(unit.queries_to_evaluate)}")

            if not self.dry_run:
                self._init_agent_for_context(context_id=episode_id, force_new=True)

            unit_results = self._evaluate_unit(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, episode_id)

    def _split_sessions_into_chunks(
        self,
        sessions: List[AMABenchSession],
    ) -> List[Tuple[List[AMABenchSession], str]]:
        """Split sessions into chunks for memory injection.

        For AMA-Bench, each episode typically has 1 large session (the full
        trajectory). When a single session exceeds memory_chunk_size, it is
        split at Step boundaries so that each chunk stays within the limit.
        """
        chunks = []

        for session in sessions:
            memory_text = session.to_memory_text()

            if len(memory_text) <= self.memory_chunk_size:
                chunks.append(([session], memory_text))
                continue

            # Split large session at Step boundaries
            sub_chunks = self._split_text_at_step_boundaries(memory_text)
            for idx, sub_text in enumerate(sub_chunks):
                sub_session = AMABenchSession(
                    session_id=f"{session.session_id}_chunk{idx}",
                    content=sub_text,
                    metadata={
                        **session.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(sub_chunks),
                    },
                    episode_id=session.episode_id,
                    turn_range=f"chunk_{idx}",
                    num_turns=session.num_turns,
                )
                chunks.append(([sub_session], sub_text))

        return chunks

    def _split_text_at_step_boundaries(self, text: str) -> List[str]:
        """Split trajectory text into chunks at 'Step N:' boundaries."""
        import re

        # Extract header (Task: ... before first Step)
        first_step_match = re.search(r'^Step \d+:', text, re.MULTILINE)
        header = ""
        body = text
        if first_step_match:
            header = text[:first_step_match.start()]
            body = text[first_step_match.start():]

        # Split body into step blocks
        step_positions = [m.start() for m in re.finditer(r'^Step \d+:', body, re.MULTILINE)]
        if not step_positions:
            return [text]

        step_blocks = []
        for i, pos in enumerate(step_positions):
            end = step_positions[i + 1] if i + 1 < len(step_positions) else len(body)
            step_blocks.append(body[pos:end])

        # Greedily group step blocks into chunks
        chunks = []
        current_parts = []
        current_size = len(header)

        for block in step_blocks:
            block_size = len(block)

            if current_size + block_size > self.memory_chunk_size and current_parts:
                chunk_text = header + "".join(current_parts)
                chunks.append(chunk_text)
                current_parts = []
                current_size = len(header)

            current_parts.append(block)
            current_size += block_size

        if current_parts:
            chunk_text = header + "".join(current_parts)
            chunks.append(chunk_text)

        return chunks

    def _evaluate_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        results = []

        self._log(f"  --- Memory Build Phase ---")

        if self.dry_run:
            self._log(f"  [Dry Run] Skipping memory build")
            total_memory_time = 0.0
            chunk_build_results = []
        else:
            chunks = self._split_sessions_into_chunks(unit.sessions_to_inject)
            total_chunks = len(chunks)

            self._log(f"  Split into {total_chunks} chunks (chunk_size={self.memory_chunk_size:,} chars)")

            total_memory_time = 0.0
            chunk_build_results = []
            all_session_ids = []

            for chunk_idx, (chunk_sessions, chunk_text) in enumerate(chunks):
                chunk_session_ids = [s.session_id for s in chunk_sessions]
                all_session_ids.extend(chunk_session_ids)

                chunk_chars = len(chunk_text)
                chunk_tokens_est = chunk_chars // 4

                self._log(f"    [Chunk {chunk_idx + 1}/{total_chunks}] "
                         f"Sessions: {chunk_session_ids} "
                         f"({len(chunk_sessions)} sessions, ~{chunk_tokens_est:,} tokens)")

                if self.prompt_manager.method_type == "agentic":
                    memorize_text = self.prompt_manager.format_memorize(
                        context=chunk_text,
                    )
                else:
                    memorize_text = chunk_text

                chunk_start_time = time.time()
                is_last_chunk = (chunk_idx == total_chunks - 1)

                try:
                    memory_result = self.agent_manager.send_message(
                        message=memorize_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_chunk,
                    )

                    chunk_time = time.time() - chunk_start_time
                    total_memory_time += chunk_time

                    if isinstance(memory_result, MemoryBuildResult):
                        entries_count = len(memory_result.memory_entries) if memory_result.memory_entries else 0
                        chunk_count = memory_result.chunk_count or 0

                        self._log(f"      -> Stored: entries={entries_count}, chunks={chunk_count}, "
                                 f"time={chunk_time:.2f}s")

                        chunk_build_results.append({
                            "chunk_index": chunk_idx,
                            "session_ids": chunk_session_ids,
                            "session_count": len(chunk_sessions),
                            "input_chars": chunk_chars,
                            "input_tokens_est": chunk_tokens_est,
                            "time_cost": chunk_time,
                            "build_result": memory_result.to_dict(),
                        })
                    else:
                        chunk_build_results.append({
                            "chunk_index": chunk_idx,
                            "session_ids": chunk_session_ids,
                            "session_count": len(chunk_sessions),
                            "input_chars": chunk_chars,
                            "input_tokens_est": chunk_tokens_est,
                            "time_cost": chunk_time,
                            "build_result": {"raw_result": str(memory_result)},
                        })

                except Exception as e:
                    chunk_time = time.time() - chunk_start_time
                    self._log(f"      [ERROR] Chunk {chunk_idx + 1} failed: {e}", level="ERROR")

                    chunk_build_results.append({
                        "chunk_index": chunk_idx,
                        "session_ids": chunk_session_ids,
                        "session_count": len(chunk_sessions),
                        "input_chars": chunk_chars,
                        "input_tokens_est": chunk_tokens_est,
                        "time_cost": chunk_time,
                        "error": str(e),
                    })

            total_entries = sum(
                len(r.get("build_result", {}).get("memory_entries", []))
                for r in chunk_build_results if "build_result" in r
            )
            total_stored_chunks = sum(
                r.get("build_result", {}).get("chunk_count", 0)
                for r in chunk_build_results if "build_result" in r
            )

            self._log(f"  [Summary] Total: {total_chunks} chunks, "
                     f"{len(all_session_ids)} sessions, "
                     f"entries={total_entries}, stored_chunks={total_stored_chunks}")

            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "episode_id": unit.metadata.get("episode_id"),
                "domain": unit.metadata.get("domain"),
                "task_type": unit.metadata.get("task_type"),
                "session_ids": all_session_ids,
                "session_count": len(unit.sessions_to_inject),
                "chunk_count": total_chunks,
                "chunk_size_config": self.memory_chunk_size,
                "total_time": total_memory_time,
                "total_entries": total_entries,
                "total_stored_chunks": total_stored_chunks,
                "chunk_builds": chunk_build_results,
            })

        self._log(f"  Memory Build Done, total_time={total_memory_time:.2f}s")

        self._log(f"  --- Query Evaluation Phase ---")

        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = total_memory_time / query_count if query_count > 0 else 0.0

        for query in unit.queries_to_evaluate:
            result = self._evaluate_query(query, unit.context_id)
            result.memory_construction_time = memory_time_per_query
            results.append(result)

            if "memrl" in self.agent_manager.method_name if self.agent_manager else False:
                self.agent_manager.send_query_feedback(
                    score=result.score, is_correct=result.is_correct
                )

            status = "+" if result.is_correct else "-"
            self._log(f"    [{status}] {query.query_id} ({query.query_type}/{query.qa_type_raw}): {result.score:.2f}")

        return results

    def _evaluate_query(self, query: AMABenchQuery, context_id: Any) -> MetricResult:
        if self.dry_run:
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output="[DRY RUN]",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={
                    "dry_run": True,
                    "qa_type_raw": query.qa_type_raw,
                    "question_uuid": query.question_uuid,
                },
            )

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )

        response = self.agent_manager.send_message(
            message=formatted_question,
            memorizing=False,
            context_id=context_id,
        )

        if isinstance(response, dict):
            model_output = response.get("output", "")
            query_time = response.get("query_time", 0.0)
            retrieved_memories = response.get("retrieved_memories", [])
            retrieved_count = response.get("retrieved_count", 0)
            extra = response.get("extra", {})
        else:
            model_output = str(response)
            query_time = 0.0
            retrieved_memories = []
            retrieved_count = 0
            extra = {}

        saved_phase = get_usage_tracker()._current_phase
        get_usage_tracker().set_phase("evaluation")
        result = self.metrics_calculator.compute(
            query_id=query.query_id,
            query_type=query.query_type,
            model_output=model_output,
            expected_answers=query.get_correct_answers(),
            question=query.question,
        )
        get_usage_tracker().set_phase(saved_phase)

        result.query_time = query_time
        result.retrieved_memories = retrieved_memories
        result.retrieved_count = retrieved_count
        result.extra = extra

        if "qa_type_raw" not in result.details:
            result.details["qa_type_raw"] = query.qa_type_raw
        if "question_uuid" not in result.details:
            result.details["question_uuid"] = query.question_uuid

        return result

    def _generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        summary = self.aggregator.get_summary()
        memory_build_summary = self._summarize_memory_builds()
        llm_usage = get_usage_tracker().get_stats()

        report = EvaluationReport(
            method_name=self.method_config.method_name,
            model_name=self.method_config.model.name,
            dataset_name=self.dataset_config.dataset_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            summary=summary,
            detailed_results=self.aggregator.get_detailed_results(),
            config={
                "method_config": self.method_config.raw_config,
                "dataset_config": self.dataset_config.raw_config,
                "dry_run": self.dry_run,
            },
            metadata={
                "total_episodes": len(self.dataset.get_episode_ids()),
                "domain_distribution": self.dataset.get_domain_distribution(),
                "qa_type_distribution": self.dataset.get_qa_type_distribution(),
                "memory_build_summary": memory_build_summary,
                "memory_chunk_size": self.memory_chunk_size,
                "llm_usage": llm_usage,
            },
        )

        result_path, memory_build_path, query_answer_path = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=self._memory_build_logs,
        )

        self._log(f"Results saved to: {result_path}")
        self._log(f"Memory build details saved to: {memory_build_path}")
        self._log(f"Query answer details saved to: {query_answer_path}")

        return report

    def _summarize_memory_builds(self) -> Dict[str, Any]:
        if not self._memory_build_logs:
            return {"total_builds": 0}

        total_units = len(self._memory_build_logs)
        total_sessions = sum(log.get("session_count", 0) for log in self._memory_build_logs)
        total_chunks = sum(log.get("chunk_count", 0) for log in self._memory_build_logs)
        total_time = sum(log.get("total_time", 0) for log in self._memory_build_logs)
        total_entries = sum(log.get("total_entries", 0) for log in self._memory_build_logs)
        total_stored_chunks = sum(log.get("total_stored_chunks", 0) for log in self._memory_build_logs)

        return {
            "total_units": total_units,
            "total_sessions": total_sessions,
            "total_memory_chunks": total_chunks,
            "total_time": total_time,
            "total_entries": total_entries,
            "total_stored_chunks": total_stored_chunks,
            "avg_time_per_unit": total_time / total_units if total_units > 0 else 0,
            "avg_chunks_per_unit": total_chunks / total_units if total_units > 0 else 0,
            "chunk_size_config": self.memory_chunk_size,
        }


@register_evaluator("ama_bench")
def evaluate_ama_bench(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    **kwargs,
) -> EvaluationReport:
    evaluator = AMABenchEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
    )
    return evaluator.evaluate()
