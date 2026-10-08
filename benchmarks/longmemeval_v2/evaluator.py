"""LongMemEval-V2 evaluation module.

Adapts the official LME-V2 three-phase pipeline (memory build → query →
score) to the MedMemoryBench evaluation framework.

Key design decisions:
  - One EvaluationUnit per domain (all trajectories share one haystack).
  - Trajectories are injected via **token-budget adaptive batching**: each
    batch is filled greedily until ``max_batch_tokens`` is reached.  This
    mirrors the official harness which inserts one trajectory at a time
    while respecting the model's context limit.
  - Questions include the ``\\boxed{}`` answer format instruction.
  - Metrics dispatch is handled by ``LongMemEvalV2Metric`` using the
    per-question ``eval_function`` spec from the official dataset.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager
from src.result import EvaluationReport, ResultCollector
from benchmarks.longmemeval_v2.dataset import (
    LongMemEvalV2Dataset,
    LongMemEvalV2Query,
    LongMemEvalV2Session,
    CATEGORY_MAP,
)
from benchmarks.base import EvaluationUnit
from methods.base import MemoryBuildResult
from metrics import MetricsAggregator, MetricResult
from metrics.longmemeval_v2_metrics import LongMemEvalV2Metric
from utils.templates import get_prompt_manager
from utils.llm_client import get_usage_tracker


DEFAULT_MAX_BATCH_TOKENS = 50_000
CHARS_PER_TOKEN_EST = 4

NON_ABSTENTION_CATEGORIES = {"static", "dynamic", "procedure", "gotchas"}
ABSTENTION_CATEGORIES = {"static-abs", "dynamic-abs", "procedure-abs"}


class LongMemEvalV2Evaluator:

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
            dataset="longmemeval_v2",
            method=method_config.method_name,
            language="en",
        )

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[LongMemEvalV2Dataset] = None

        api_config = get_api_config()
        self.metric = LongMemEvalV2Metric(
            dataset="longmemeval_v2",
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []

        eval_config = dataset_config.raw_config.get("evaluation", {})
        self.max_batch_tokens = eval_config.get(
            "max_batch_tokens", DEFAULT_MAX_BATCH_TOKENS
        )

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    def _init_dataset(self) -> None:
        self._log(f"Loading LongMemEval-V2 dataset")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        eval_cfg = self.dataset_config.raw_config.get("evaluation", {})

        self.dataset = LongMemEvalV2Dataset(
            data_dir=data_dir,
            config={
                "domain": eval_cfg.get("domain", "web"),
                "tier": eval_cfg.get("tier", "small"),
                "max_questions": eval_cfg.get("max_questions"),
                "question_ids": eval_cfg.get("question_ids"),
                "truncate_a11y": eval_cfg.get("truncate_a11y", 3000),
            },
        )
        self.dataset.load()

        self._log(f"  Domain: {self.dataset.domain}")
        self._log(f"  Tier: {self.dataset.tier}")
        self._log(f"  Trajectories: {self.dataset.get_total_sessions()}")
        self._log(f"  Questions: {self.dataset.get_total_queries()}")
        self._log(f"  Max batch tokens: {self.max_batch_tokens}")
        self._log(f"  Type distribution: {self.dataset.get_type_distribution()}")
        self._log(f"  Eval function distribution: {self.dataset.get_eval_function_distribution()}")

    def _init_agent_for_context(self, context_id: Any, force_new: bool = True) -> None:
        if force_new or self.agent_manager is None:
            if self.agent_manager is not None:
                try:
                    self.agent_manager.reset()
                except Exception as e:
                    self._log(f"Warning: agent reset failed: {e}", level="WARNING")

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
        return self._generate_report(start_time, end_time, duration)

    def _run_evaluation_loop(self) -> None:
        for unit in self.dataset.get_evaluation_units():
            self._log(f"\n{'=' * 60}")
            self._log(f"Domain: {unit.metadata.get('domain')}, Tier: {unit.metadata.get('tier')}")
            self._log(f"  Trajectories: {unit.metadata.get('trajectory_count')}")
            self._log(f"  Questions: {unit.metadata.get('question_count')}")

            if not self.dry_run:
                self._init_agent_for_context(context_id=unit.context_id, force_new=True)

            unit_results = self._evaluate_unit(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, unit.context_id)

    def _batch_sessions(
        self, sessions: List[LongMemEvalV2Session]
    ) -> List[List[LongMemEvalV2Session]]:
        max_chars = self.max_batch_tokens * CHARS_PER_TOKEN_EST
        batches: List[List[LongMemEvalV2Session]] = []
        current_batch: List[LongMemEvalV2Session] = []
        current_chars = 0

        for session in sessions:
            text_len = len(session.to_memory_text())
            if current_batch and current_chars + text_len > max_chars:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(session)
            current_chars += text_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def _evaluate_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        results: List[MetricResult] = []

        # ---- Phase 1: Memory Build ----
        self._log(f"  --- Memory Build Phase (trajectory injection) ---")

        if self.dry_run:
            self._log(f"  [Dry Run] Skipping memory build")
            total_memory_time = 0.0
            batch_build_results: List[Dict[str, Any]] = []
            failed_batches = 0
        else:
            batches = self._batch_sessions(unit.sessions_to_inject)
            total_batches = len(batches)
            self._log(
                f"  Split {len(unit.sessions_to_inject)} trajectories into "
                f"{total_batches} batches (max_batch_tokens={self.max_batch_tokens})"
            )

            total_memory_time = 0.0
            batch_build_results = []
            failed_batches = 0

            for batch_idx, batch_sessions in enumerate(batches):
                batch_text = self._format_batch_text(batch_sessions)
                batch_chars = len(batch_text)
                batch_tokens_est = batch_chars // 4
                is_last = batch_idx == total_batches - 1

                traj_ids = [s.trajectory_id for s in batch_sessions]
                self._log(
                    f"    [Batch {batch_idx + 1}/{total_batches}] "
                    f"{len(batch_sessions)} trajectories, ~{batch_tokens_est:,} tokens"
                    f"{' (LAST)' if is_last else ''}"
                )

                if self.prompt_manager.method_type == "agentic":
                    memorize_text = self.prompt_manager.format_memorize(
                        context=batch_text,
                        timestamp=None,
                    )
                else:
                    memorize_text = batch_text

                batch_start = time.time()
                try:
                    memory_result = self.agent_manager.send_message(
                        message=memorize_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last,
                    )
                    batch_time = time.time() - batch_start
                    total_memory_time += batch_time

                    if isinstance(memory_result, MemoryBuildResult):
                        entries = len(memory_result.memory_entries) if memory_result.memory_entries else 0
                        self._log(f"      -> entries={entries}, time={batch_time:.2f}s")
                        batch_build_results.append({
                            "batch_index": batch_idx,
                            "trajectory_ids": traj_ids,
                            "trajectory_count": len(batch_sessions),
                            "input_chars": batch_chars,
                            "input_tokens_est": batch_tokens_est,
                            "time_cost": batch_time,
                            "build_result": memory_result.to_dict(),
                        })
                    else:
                        batch_build_results.append({
                            "batch_index": batch_idx,
                            "trajectory_ids": traj_ids,
                            "trajectory_count": len(batch_sessions),
                            "input_chars": batch_chars,
                            "input_tokens_est": batch_tokens_est,
                            "time_cost": batch_time,
                            "build_result": {"raw_result": str(memory_result)[:500]},
                        })

                except Exception as e:
                    batch_time = time.time() - batch_start
                    failed_batches += 1
                    self._log(
                        f"      [ERROR] Batch {batch_idx + 1} failed: {e}", level="ERROR"
                    )
                    batch_build_results.append({
                        "batch_index": batch_idx,
                        "trajectory_ids": traj_ids,
                        "trajectory_count": len(batch_sessions),
                        "input_chars": batch_chars,
                        "input_tokens_est": batch_tokens_est,
                        "time_cost": batch_time,
                        "error": str(e),
                    })

            if failed_batches > 0:
                self._log(
                    f"  WARNING: {failed_batches}/{total_batches} batches failed",
                    level="WARNING",
                )

            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "domain": unit.metadata.get("domain"),
                "tier": unit.metadata.get("tier"),
                "trajectory_count": len(unit.sessions_to_inject),
                "batch_count": len(batches),
                "max_batch_tokens": self.max_batch_tokens,
                "total_time": total_memory_time,
                "failed_batches": failed_batches,
                "batch_builds": batch_build_results,
            })

        self._log(f"  Memory Build Done, total_time={total_memory_time:.2f}s")

        # ---- Phase 2 + 3: Query + Score ----
        self._log(f"  --- Query Evaluation Phase ({len(unit.queries_to_evaluate)} questions) ---")

        for qi, query in enumerate(unit.queries_to_evaluate):
            result = self._evaluate_query(query, unit.context_id)
            result.memory_construction_time = total_memory_time
            if failed_batches > 0:
                result.details["memory_build_incomplete"] = True
                result.details["failed_batches"] = failed_batches
            results.append(result)

            if (
                not self.dry_run
                and self.agent_manager is not None
                and "memrl" in self.agent_manager.method_name
            ):
                self.agent_manager.send_query_feedback(
                    score=result.score, is_correct=result.is_correct
                )

            status = "v" if result.is_correct else "x"
            if (qi + 1) % 20 == 0 or qi == len(unit.queries_to_evaluate) - 1:
                correct_so_far = sum(1 for r in results if r.is_correct)
                self._log(
                    f"    Progress: {qi + 1}/{len(unit.queries_to_evaluate)}, "
                    f"accuracy={correct_so_far}/{qi + 1} "
                    f"({correct_so_far / (qi + 1):.1%})"
                )

        return results

    def _format_batch_text(self, sessions: List[LongMemEvalV2Session]) -> str:
        return "\n\n".join(s.to_memory_text() for s in sessions)

    def _evaluate_query(
        self, query: LongMemEvalV2Query, context_id: Any
    ) -> MetricResult:
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
        result = self.metric.compute(
            query_id=query.query_id,
            query_type=query.query_type,
            model_output=model_output,
            expected_answers=query.get_correct_answers(),
            question=query.question,
            eval_function=query.eval_function,
            category=query.category,
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

        # --- LME-V2-specific aggregate metrics ---
        all_results = self.aggregator.results

        non_abs = [r for r in all_results if not r.details.get("is_abstention", False)]
        abs_only = [r for r in all_results if r.details.get("is_abstention", False)]

        summary["overall_full_set"] = (
            sum(r.score for r in all_results) / len(all_results) if all_results else 0.0
        )
        summary["overall_non_abstention"] = (
            sum(r.score for r in non_abs) / len(non_abs) if non_abs else 0.0
        )
        summary["overall_abstention"] = (
            sum(r.score for r in abs_only) / len(abs_only) if abs_only else 0.0
        )
        summary["count_non_abstention"] = len(non_abs)
        summary["count_abstention"] = len(abs_only)

        by_category: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            cat = r.details.get("category", r.query_type)
            if cat not in by_category:
                by_category[cat] = {"total": 0, "correct": 0}
            by_category[cat]["total"] += 1
            if r.is_correct:
                by_category[cat]["correct"] += 1
        for cat, stats in by_category.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        summary["by_category"] = by_category

        by_eval_method: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            em = r.details.get("eval_method", "unknown")
            if em not in by_eval_method:
                by_eval_method[em] = {"total": 0, "correct": 0}
            by_eval_method[em]["total"] += 1
            if r.is_correct:
                by_eval_method[em]["correct"] += 1
        for em, stats in by_eval_method.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        summary["by_eval_method"] = by_eval_method

        unknown_count = sum(1 for r in all_results if r.details.get("is_unknown", False))
        summary["unknown_count"] = unknown_count
        summary["unknown_rate"] = unknown_count / len(all_results) if all_results else 0.0

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
                "domain": self.dataset.domain,
                "tier": self.dataset.tier,
                "total_trajectories": self.dataset.get_total_sessions(),
                "total_questions": self.dataset.get_total_queries(),
                "type_distribution": self.dataset.get_type_distribution(),
                "category_distribution": self.dataset.get_category_distribution(),
                "eval_function_distribution": self.dataset.get_eval_function_distribution(),
                "memory_build_summary": memory_build_summary,
                "max_batch_tokens": self.max_batch_tokens,
                "llm_usage": llm_usage,
                "lme_v2_metrics": {
                    "overall_full_set": summary.get("overall_full_set", 0.0),
                    "overall_non_abstention": summary.get("overall_non_abstention", 0.0),
                    "overall_abstention": summary.get("overall_abstention", 0.0),
                    "count_non_abstention": summary.get("count_non_abstention", 0),
                    "count_abstention": summary.get("count_abstention", 0),
                    "by_category": summary.get("by_category", {}),
                    "by_eval_method": summary.get("by_eval_method", {}),
                    "unknown_count": summary.get("unknown_count", 0),
                    "unknown_rate": summary.get("unknown_rate", 0.0),
                },
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
        total_trajectories = sum(log.get("trajectory_count", 0) for log in self._memory_build_logs)
        total_batches = sum(log.get("batch_count", 0) for log in self._memory_build_logs)
        total_time = sum(log.get("total_time", 0) for log in self._memory_build_logs)
        total_failed = sum(log.get("failed_batches", 0) for log in self._memory_build_logs)

        return {
            "total_units": total_units,
            "total_trajectories": total_trajectories,
            "total_batches": total_batches,
            "total_failed_batches": total_failed,
            "total_time": total_time,
            "avg_time_per_trajectory": total_time / total_trajectories if total_trajectories > 0 else 0,
            "avg_time_per_batch": total_time / total_batches if total_batches > 0 else 0,
            "max_batch_tokens": self.max_batch_tokens,
        }


@register_evaluator("longmemeval_v2")
def evaluate_longmemeval_v2(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    **kwargs,
) -> EvaluationReport:
    evaluator = LongMemEvalV2Evaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
    )
    return evaluator.evaluate()
