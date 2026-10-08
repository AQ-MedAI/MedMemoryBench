"""MedMemoryBench evaluation module."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager, AgentResponse, MemoryBuildResult
from src.result import EvaluationReport, ResultCollector
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset, MedQuery, MedSession
from benchmarks.base import EvaluationUnit
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from metrics.base import MetricResult as BaseMetricResult
from utils.templates import get_prompt_manager
from utils.llm_client import get_usage_tracker, LLMRetryExhaustedError

from benchmarks.medmemorybench.checkpoint import (
    MedMemoryBenchCheckpointManager,
    compute_config_hash,
)


class MedMemoryBenchEvaluator:
    """MedMemoryBench evaluator with incremental evaluation and checkpoint support."""

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
            language=dataset_config.language,
        )

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[MedMemoryBenchDataset] = None

        api_config = get_api_config()
        self.metrics_calculator = MetricsCalculator(
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            language=dataset_config.language,
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []

        self._checkpoint_manager: Optional[MedMemoryBenchCheckpointManager] = None
        self._checkpoint_enabled = False

        if self._should_enable_checkpoint():
            self._init_checkpoint_manager()

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    def _init_dataset(self) -> None:
        self._log(f"Loading dataset: {self.dataset_config.dataset_name}")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        self.dataset = MedMemoryBenchDataset(
            data_dir=data_dir,
            config={
                "evaluation_mode": self.dataset_config.evaluation_mode,
                "persona_ids": self.dataset_config.persona_ids,
                "max_personas": self.dataset_config.max_personas,
                "max_sessions_per_persona": self.dataset_config.max_sessions_per_persona,
                "evaluation_interval": self.dataset_config.evaluation_interval,
                "inject_noise": self.dataset_config.inject_noise,
                "query_types": [qt.name for qt in self.dataset_config.query_types] if self.dataset_config.query_types else None,
            }
        )
        self.dataset.load()

        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Total Queries: {self.dataset.get_total_queries()}")
        self._log(f"  Evaluation Mode: {self.dataset_config.evaluation_mode}")
        self._log(f"  Evaluation Interval: {self.dataset_config.evaluation_interval}")

    def _should_enable_checkpoint(self) -> bool:
        return self.dataset_config.evaluation_mode == "independent"

    def _init_checkpoint_manager(self) -> None:
        config_hash = compute_config_hash(self.method_config, self.dataset_config)

        self._checkpoint_manager = MedMemoryBenchCheckpointManager(
            method_name=self.method_config.method_name,
            model_name=self.method_config.model.name,
            checkpoint_dir=self.output_dir / "checkpoints",
            config_hash=config_hash,
        )
        self._checkpoint_enabled = True

    def _try_resume_from_checkpoint(self) -> bool:
        if not self._checkpoint_manager or not self._checkpoint_manager.exists():
            self._log("No checkpoint found, starting from scratch")
            return False

        checkpoint = self._checkpoint_manager.load()
        if checkpoint is None:
            self._log("Checkpoint file corrupted, starting from scratch")
            self._checkpoint_manager.delete()
            return False

        if not self._checkpoint_manager.validate_config():
            self._log("Config changed, checkpoint invalid, starting from scratch")
            self._checkpoint_manager.delete()
            return False

        if not self._checkpoint_manager.is_independent_mode():
            self._log("Checkpoint not in independent mode, cannot resume")
            return False

        self._load_checkpoint_results()

        info = self._checkpoint_manager.get_resume_info()
        self._log("=" * 50)
        self._log("Resuming from checkpoint")
        self._log(f"  Checkpoint ID: {info['checkpoint_id'][:8]}...")
        self._log(f"  Completed Personas: {info['completed_personas']}/{info['total_personas']}")
        self._log(f"  Completed Queries: {info['completed_queries']}/{info['total_queries']}")
        if info['current_persona'] is not None:
            self._log(f"  Current Persona: {info['current_persona']} "
                     f"({info['current_persona_completed_queries']} queries done, rebuilding memory)")
        self._log("=" * 50)

        return True

    def _create_new_checkpoint(self) -> None:
        if not self._checkpoint_manager:
            return

        total_personas = len(self.dataset.get_persona_ids())
        total_queries = self.dataset.get_total_queries()

        self._checkpoint_manager.create(
            total_personas=total_personas,
            total_queries=total_queries,
            evaluation_mode=self.dataset_config.evaluation_mode,
        )
        self._log(f"Created checkpoint: {self._checkpoint_manager.checkpoint_path}")

    def _load_checkpoint_results(self) -> None:
        if not self._checkpoint_manager:
            return

        completed_results = self._checkpoint_manager.get_completed_results()

        for persona_id, results in completed_results.items():
            for result_dict in results:
                result = MetricResult(
                    query_id=result_dict.get("query_id", ""),
                    query_type=result_dict.get("query_type", ""),
                    score=result_dict.get("score", 0.0),
                    is_correct=result_dict.get("is_correct", False),
                    model_output=result_dict.get("model_output", ""),
                    expected_answer=result_dict.get("expected_answer", ""),
                    question=result_dict.get("question", ""),
                    details=result_dict.get("details", {}),
                )
                result.memory_construction_time = result_dict.get("memory_construction_time", 0.0)
                result.query_time = result_dict.get("query_time", 0.0)

                self.aggregator.add_result(result)
                self.result_collector.add_result(result, persona_id)

    def _init_agent_for_context(self, context_id: int, force_new: bool = True) -> None:
        if force_new or self.agent_manager is None:
            # Clean up old agent's resources before creating new one
            if self.agent_manager is not None:
                try:
                    self._log(f"[DEBUG] Starting reset of old agent...")
                    # Reset releases resources (closes Qdrant client, etc.)
                    self.agent_manager.reset()
                    self._log(f"[DEBUG] Reset completed")

                    # Explicitly delete old agent manager to release all references
                    old_manager = self.agent_manager
                    self.agent_manager = None
                    del old_manager

                    # Force garbage collection to release resources
                    import gc
                    gc.collect()

                    self._log(f"[DEBUG] Old agent deleted, sleeping 1s for resource cleanup...")
                    # Give time for resources to be fully released
                    import time
                    time.sleep(1.0)
                    self._log(f"[DEBUG] Sleep done, creating new agent...")
                except Exception as e:
                    self._log(f"Warning: Failed to reset old agent: {e}")

            self._log(f"[DEBUG] Creating AgentManager for context {context_id}...")
            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
            )
            self._log(f"[DEBUG] AgentManager created successfully")

        self.agent_manager.set_context_id(context_id)
        self._log(f"[DEBUG] Context ID set to {context_id}")

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()

        get_usage_tracker().reset()

        self._init_dataset()

        resumed = False
        if self.resume and self._checkpoint_enabled:
            resumed = self._try_resume_from_checkpoint()

        if not resumed and self._checkpoint_enabled:
            self._create_new_checkpoint()

        self._run_evaluation_loop()

        if self._checkpoint_manager:
            self._checkpoint_manager.mark_completed()
            self._checkpoint_manager.delete()
            self._log("Evaluation completed, checkpoint deleted")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self) -> None:
        current_context_id = None

        for unit in self.dataset.get_evaluation_units():
            persona_id = unit.context_id

            if self._checkpoint_manager and self._checkpoint_manager.is_persona_completed(persona_id):
                self._log(f"Skipping completed Persona: {persona_id}")
                continue

            if persona_id != current_context_id:
                if current_context_id is not None and self._checkpoint_manager:
                    self._checkpoint_manager.complete_persona(current_context_id)

                if current_context_id is not None:
                    self._log(f"\nSwitching Persona: {current_context_id} -> {persona_id}")

                if not self.dry_run:
                    self._init_agent_for_context(
                        context_id=persona_id,
                        force_new=(self.dataset_config.evaluation_mode == "independent")
                    )

                current_context_id = persona_id

                if self._checkpoint_manager:
                    self._checkpoint_manager.start_persona(persona_id)

            unit_results = self._evaluate_unit_with_checkpoint(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, persona_id)

        if current_context_id is not None and self._checkpoint_manager:
            self._checkpoint_manager.complete_persona(current_context_id)

    def _evaluate_unit_with_checkpoint(self, unit: EvaluationUnit) -> List[MetricResult]:
        """Evaluate unit with checkpoint support."""
        self._log(f"\n  Evaluation Unit {unit.unit_id}:")
        self._log(f"    Persona: {unit.context_id}")
        self._log(f"    Sessions to inject: {len(unit.sessions_to_inject)}")
        self._log(f"    Queries to evaluate: {len(unit.queries_to_evaluate)}")

        results = []

        self._log(f"    --- Memory Build Start ---")

        memory_build_failed = False
        total_memory_time = 0.0
        session_build_results = []  # Store per-session build results

        if self.dry_run:
            self._log(f"    [Dry Run] Skipping memory build")
        else:
            total_sessions = len(unit.sessions_to_inject)
            session_ids = []

            for idx, session in enumerate(unit.sessions_to_inject):
                session_ids.append(session.session_id)
                memory_text = session.to_memory_text()

                # Show progress
                self._log(f"      [Progress] Session {idx + 1}/{total_sessions} (ID: {session.session_id})")

                # Only agentic methods need instruction prompt for memorization;
                # RAG/long_context methods store raw text directly into their stores
                if self.prompt_manager.method_type == "agentic":
                    memorize_text = self.prompt_manager.format_memorize(
                        context=memory_text,
                        timestamp=None,
                    )
                else:
                    memorize_text = memory_text

                is_last_session = (idx == total_sessions - 1)

                try:
                    memory_result = self.agent_manager.send_message(
                        message=memorize_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_session,
                    )

                    if memory_result is not None and isinstance(memory_result, MemoryBuildResult):
                        total_memory_time += memory_result.time_cost

                        # Log brief progress info
                        passages_count = len(memory_result.all_passages) if memory_result.all_passages else memory_result.extra.get("inserted_count", 0)
                        self._log(f"        → Stored {passages_count} passages, time={memory_result.time_cost:.2f}s")

                        # Show extraction preview (first 100 chars)
                        if memory_result.extraction_result:
                            preview = memory_result.extraction_result[:150].replace('\n', ' ')
                            self._log(f"        → Extraction: {preview}...")

                        # Store detailed result for this session
                        session_build_results.append({
                            "session_id": session.session_id,
                            "session_index": idx,
                            "build_result": memory_result.to_dict(),
                        })

                except LLMRetryExhaustedError as e:
                    self._log(
                        f"        [API ERROR] Session {session.session_id} failed: {e}",
                        level="ERROR"
                    )
                    memory_build_failed = True
                    session_build_results.append({
                        "session_id": session.session_id,
                        "session_index": idx,
                        "error": str(e),
                    })

                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_session_injected(session.session_id)

            # Log summary
            total_passages = sum(
                len(r.get("build_result", {}).get("all_passages", []))
                for r in session_build_results if "build_result" in r
            )
            self._log(f"      [Summary] Total passages stored: {total_passages}")

            # Store all session build results
            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "session_ids": session_ids,
                "session_count": total_sessions,
                "total_time": total_memory_time,
                "total_passages": total_passages,
                "session_builds": session_build_results,
            })

        self._log(f"    --- Memory Build Done, time={total_memory_time:.2f}s ---")

        total_query_time = 0.0
        self._log(f"    --- Query Evaluation Start ---")

        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = total_memory_time / query_count if query_count > 0 else 0.0

        for query in unit.queries_to_evaluate:
            if self._checkpoint_manager and self._checkpoint_manager.is_query_completed(query.query_id):
                self._log(f"    [Skip] {query.query_id} (completed)")
                continue

            result = self._evaluate_query(query, unit.context_id)
            result.memory_construction_time = memory_time_per_query
            results.append(result)
            total_query_time += result.query_time

            # agent_manager is None during dry-run (no agent is constructed)
            if getattr(self.agent_manager, "method_name", "") and "memrl" in self.agent_manager.method_name:
                self.agent_manager.send_query_feedback(
                    score=result.score, is_correct=result.is_correct
                )

            status = "✓" if result.is_correct else "✗"
            self._log(f"    [{status}] {query.query_id} ({query.query_type}): {result.score:.2f}")

            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    query.query_id,
                    result.to_dict(),
                )

        self._log(f"    --- Query Evaluation Done, time={total_query_time:.2f}s ---")

        return results

    def _evaluate_query(self, query, context_id: int) -> MetricResult:
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

        answers_data = None
        if isinstance(query, MedQuery):
            answers_data = query.answers_data

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )

        try:
            response = self.agent_manager.send_message(
                message=formatted_question,
                memorizing=False,
                query_id=query.query_id,
                context_id=context_id,
            )
        except LLMRetryExhaustedError as e:
            # Log the error and return a failed result instead of crashing
            self._log(
                f"    [API ERROR] {query.query_id}: API call failed after retries, "
                f"skipping this query. Error: {e}",
                level="ERROR"
            )
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output="[API_ERROR] Connection failed after retries",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={
                    "api_error": True,
                    "error_type": "LLMRetryExhaustedError",
                    "error_message": str(e),
                },
            )

        if isinstance(response, dict):
            model_output = response.get("output", "")
            query_time = response.get("query_time", 0.0)
            retrieved_memories = response.get("retrieved_memories", [])
            retrieved_count = response.get("retrieved_count", 0)
            extra = response.get("extra", {})
        elif hasattr(response, "output"):
            model_output = response.output
            query_time = getattr(response, "query_time", 0.0)
            retrieved_memories = getattr(response, "retrieved_memories", [])
            retrieved_count = getattr(response, "retrieved_count", 0)
            extra = getattr(response, "extra", {})
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
            answers_data=answers_data,
            metadata=query.metadata,
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
                "evaluation_mode": self.dataset_config.evaluation_mode,
                "evaluation_interval": self.dataset_config.evaluation_interval,
                "total_personas": len(self.result_collector.get_context_ids()),
                "memory_build_summary": memory_build_summary,
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

        total_sessions = sum(
            log.get("session_count", 0)
            for log in self._memory_build_logs
        )

        # New structure: sum time from each unit's total_time
        total_time = sum(
            log.get("total_time", 0)
            for log in self._memory_build_logs
        )

        # Count total passages from session_builds
        total_passages = sum(
            log.get("total_passages", 0)
            for log in self._memory_build_logs
        )

        # Count memory entries from session builds
        total_memory_entries = 0
        methods = {}

        for log in self._memory_build_logs:
            session_builds = log.get("session_builds", [])
            for sb in session_builds:
                build_result = sb.get("build_result", {})
                total_memory_entries += len(build_result.get("memory_entries", []))

                method = build_result.get("method", "unknown")
                if method not in methods:
                    methods[method] = {"count": 0, "time_cost": 0, "passages": 0}
                methods[method]["count"] += 1
                methods[method]["time_cost"] += build_result.get("time_cost", 0)
                methods[method]["passages"] += len(build_result.get("all_passages", []))

        return {
            "total_units": total_units,
            "total_sessions": total_sessions,
            "total_time": total_time,
            "avg_time_per_session": total_time / total_sessions if total_sessions > 0 else 0,
            "total_passages": total_passages,
            "total_memory_entries": total_memory_entries,
            "by_method": methods,
        }


@register_evaluator("medmemorybench")
def evaluate_medmemorybench(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    **kwargs
) -> EvaluationReport:
    """MedMemoryBench evaluation entry point."""
    evaluator = MedMemoryBenchEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
    )
    return evaluator.evaluate()
