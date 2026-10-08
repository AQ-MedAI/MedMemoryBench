"""LongMemEval evaluation module."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager
from src.result import EvaluationReport, ResultCollector
from benchmarks.longmemeval.dataset import LongMemEvalDataset, LongMemEvalQuery, LongMemEvalSession
from benchmarks.base import EvaluationUnit
from methods.base import MemoryBuildResult
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from utils.templates import get_prompt_manager
from utils.llm_client import get_usage_tracker


DEFAULT_SESSIONS_PER_BATCH = 10


class LongMemEvalEvaluator:

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
        self.dataset: Optional[LongMemEvalDataset] = None

        api_config = get_api_config()
        self.metrics_calculator = MetricsCalculator(
            dataset="longmemeval",
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            language="en",
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []

        eval_config = dataset_config.raw_config.get("evaluation", {})
        self.sessions_per_batch = eval_config.get("sessions_per_batch", DEFAULT_SESSIONS_PER_BATCH)

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    def _init_dataset(self) -> None:
        self._log(f"Loading dataset: {self.dataset_config.dataset_name}")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        self.dataset = LongMemEvalDataset(
            data_dir=data_dir,
            config={
                "data_file": self.dataset_config.raw_config.get("data", {}).get("data_file", "longmemeval_s_cleaned.json"),
                "max_instances": self.dataset_config.raw_config.get("evaluation", {}).get("max_instances"),
                "max_sessions": self.dataset_config.raw_config.get("evaluation", {}).get("max_sessions"),
                "question_ids": self.dataset_config.raw_config.get("evaluation", {}).get("question_ids"),
            }
        )
        self.dataset.load()

        self._log(f"  Total Instances: {self.dataset.get_total_queries()}")
        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Sessions per Batch: {self.sessions_per_batch}")
        if self.dataset.max_sessions:
            self._log(f"  Max Sessions per Instance: {self.dataset.max_sessions}")
        self._log(f"  Type Distribution: {self.dataset.get_type_distribution()}")

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
        total_units = self.dataset.get_total_queries()
        for unit_idx, unit in enumerate(self.dataset.get_evaluation_units()):
            qid = unit.context_id

            self._log(f"\n{'='*60}")
            self._log(f"[{unit_idx + 1}/{total_units}] Instance: {qid}")
            self._log(f"  Type: {unit.metadata.get('question_type')}")
            self._log(f"  Sessions: {len(unit.sessions_to_inject)}")

            if not self.dry_run:
                self._init_agent_for_context(context_id=qid, force_new=True)

            unit_results = self._evaluate_unit(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, qid)

    def _batch_sessions(self, sessions: List[LongMemEvalSession]) -> List[List[LongMemEvalSession]]:
        batches = []
        for i in range(0, len(sessions), self.sessions_per_batch):
            batches.append(sessions[i:i + self.sessions_per_batch])
        return batches

    def _evaluate_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        results = []

        self._log(f"  --- Memory Build Phase ---")

        if self.dry_run:
            self._log(f"  [Dry Run] Skipping memory build")
            total_memory_time = 0.0
            batch_build_results = []
            failed_batches = 0
        else:
            batches = self._batch_sessions(unit.sessions_to_inject)
            total_batches = len(batches)

            self._log(f"  Split into {total_batches} batches "
                     f"(sessions_per_batch={self.sessions_per_batch})")

            total_memory_time = 0.0
            batch_build_results = []
            failed_batches = 0

            for batch_idx, batch_sessions in enumerate(batches):
                batch_text = self._format_batch_text(batch_sessions)
                batch_chars = len(batch_text)
                batch_tokens_est = batch_chars // 4
                is_last_batch = (batch_idx == total_batches - 1)

                self._log(f"    [Batch {batch_idx + 1}/{total_batches}] "
                         f"{len(batch_sessions)} sessions, ~{batch_tokens_est:,} tokens"
                         f"{' (LAST)' if is_last_batch else ''}")

                if self.prompt_manager.method_type == "agentic":
                    memorize_text = self.prompt_manager.format_memorize(
                        context=batch_text,
                        timestamp=None,
                    )
                else:
                    memorize_text = batch_text

                batch_start_time = time.time()

                try:
                    memory_result = self.agent_manager.send_message(
                        message=memorize_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_batch,
                    )

                    batch_time = time.time() - batch_start_time
                    total_memory_time += batch_time

                    if isinstance(memory_result, MemoryBuildResult):
                        entries_count = len(memory_result.memory_entries) if memory_result.memory_entries else 0
                        self._log(f"      -> entries={entries_count}, time={batch_time:.2f}s")

                        batch_build_results.append({
                            "batch_index": batch_idx,
                            "session_count": len(batch_sessions),
                            "input_chars": batch_chars,
                            "input_tokens_est": batch_tokens_est,
                            "time_cost": batch_time,
                            "build_result": memory_result.to_dict(),
                        })
                    else:
                        batch_build_results.append({
                            "batch_index": batch_idx,
                            "session_count": len(batch_sessions),
                            "input_chars": batch_chars,
                            "input_tokens_est": batch_tokens_est,
                            "time_cost": batch_time,
                            "build_result": {"raw_result": str(memory_result)},
                        })

                except Exception as e:
                    batch_time = time.time() - batch_start_time
                    failed_batches += 1
                    self._log(f"      [ERROR] Batch {batch_idx + 1} failed: {e}", level="ERROR")
                    batch_build_results.append({
                        "batch_index": batch_idx,
                        "session_count": len(batch_sessions),
                        "input_chars": batch_chars,
                        "input_tokens_est": batch_tokens_est,
                        "time_cost": batch_time,
                        "error": str(e),
                    })

            if failed_batches > 0:
                self._log(
                    f"  WARNING: {failed_batches}/{total_batches} batches failed — "
                    f"query results for {unit.context_id} may be unreliable",
                    level="WARNING",
                )

            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "session_count": len(unit.sessions_to_inject),
                "batch_count": total_batches,
                "sessions_per_batch": self.sessions_per_batch,
                "total_time": total_memory_time,
                "failed_batches": failed_batches,
                "batch_builds": batch_build_results,
            })

        self._log(f"  Memory Build Done, total_time={total_memory_time:.2f}s")
        self._log(f"  --- Query Evaluation Phase ---")

        for query in unit.queries_to_evaluate:
            result = self._evaluate_query(query, unit.context_id)
            result.memory_construction_time = total_memory_time
            if failed_batches > 0:
                result.details["memory_build_incomplete"] = True
                result.details["failed_batches"] = failed_batches
            results.append(result)

            if not self.dry_run and self.agent_manager is not None and "memrl" in self.agent_manager.method_name:
                self.agent_manager.send_query_feedback(
                    score=result.score, is_correct=result.is_correct
                )

            status = "v" if result.is_correct else "x"
            self._log(f"    [{status}] {query.query_id} ({query.query_type}): {result.score:.2f}")

        return results

    def _format_batch_text(self, sessions: List[LongMemEvalSession]) -> str:
        texts = [s.to_memory_text() for s in sessions]
        return "\n\n".join(texts)

    def _evaluate_query(self, query: LongMemEvalQuery, context_id: Any) -> MetricResult:
        if self.dry_run:
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output="[DRY RUN]",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={"dry_run": True},
            )

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
            question_date=query.question_date,
        )

        try:
            response = self.agent_manager.send_message(
                message=formatted_question,
                memorizing=False,
                context_id=context_id,
            )
        except Exception as e:
            self._log(f"Query {query.query_id} failed: {e}", level="WARNING")
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output=f"[ERROR: {type(e).__name__}]",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={"error": str(e)},
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
            is_abstention=query.is_abstention,
        )
        get_usage_tracker().set_phase(saved_phase)

        result.query_time = query_time
        result.retrieved_memories = retrieved_memories
        result.retrieved_count = retrieved_count
        result.extra = extra

        return result

    def _generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        summary = self.aggregator.get_summary()

        # Add LongMemEval-specific metrics: task-averaged accuracy, abstention accuracy
        by_type = summary.get("by_type", {})
        type_accuracies = [stats["accuracy"] for stats in by_type.values()]
        task_averaged_accuracy = sum(type_accuracies) / len(type_accuracies) if type_accuracies else 0.0

        abstention_results = [r for r in self.aggregator.results if r.query_id.endswith("_abs")]
        abstention_correct = sum(1 for r in abstention_results if r.is_correct)
        abstention_accuracy = abstention_correct / len(abstention_results) if abstention_results else 0.0

        summary["task_averaged_accuracy"] = task_averaged_accuracy
        summary["abstention_accuracy"] = abstention_accuracy
        summary["abstention_count"] = len(abstention_results)

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
                "total_instances": self.dataset.get_total_queries(),
                "type_distribution": self.dataset.get_type_distribution(),
                "memory_build_summary": memory_build_summary,
                "sessions_per_batch": self.sessions_per_batch,
                "llm_usage": llm_usage,
            }
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
        total_batches = sum(log.get("batch_count", 0) for log in self._memory_build_logs)
        total_time = sum(log.get("total_time", 0) for log in self._memory_build_logs)
        total_failed = sum(log.get("failed_batches", 0) for log in self._memory_build_logs)

        return {
            "total_units": total_units,
            "total_sessions": total_sessions,
            "total_batches": total_batches,
            "total_failed_batches": total_failed,
            "total_time": total_time,
            "avg_time_per_unit": total_time / total_units if total_units > 0 else 0,
            "avg_batches_per_unit": total_batches / total_units if total_units > 0 else 0,
            "sessions_per_batch": self.sessions_per_batch,
        }


@register_evaluator("longmemeval")
def evaluate_longmemeval(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    **kwargs
) -> EvaluationReport:
    evaluator = LongMemEvalEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
    )
    return evaluator.evaluate()
