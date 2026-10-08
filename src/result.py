"""Result collection and report generation module."""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

from metrics import MetricResult


@dataclass
class EvaluationReport:
    """Evaluation report data structure."""
    method_name: str
    model_name: str
    dataset_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    summary: Dict[str, Any]
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class ResultCollector:
    """Result collector for generating evaluation reports."""

    def __init__(self):
        self._results: List[MetricResult] = []
        self._results_by_context: Dict[Union[int, str], List[MetricResult]] = {}

    def add_result(self, result: MetricResult, context_id: Optional[Union[int, str]] = None) -> None:
        self._results.append(result)

        if context_id is not None:
            if context_id not in self._results_by_context:
                self._results_by_context[context_id] = []
            self._results_by_context[context_id].append(result)

    def get_all_results(self) -> List[MetricResult]:
        return self._results.copy()

    def get_results_by_context(self, context_id: Union[int, str]) -> List[MetricResult]:
        return self._results_by_context.get(context_id, []).copy()

    def get_context_ids(self) -> List[Union[int, str]]:
        return list(self._results_by_context.keys())

    def save_reports(
        self,
        report: EvaluationReport,
        output_dir: Path,
        memory_build_logs: List[Dict[str, Any]],
    ) -> Tuple[Path, Path, Path]:
        """Save three separate report files: result, memory_build, query_answer."""
        method_subdir = self._get_method_subdir(report.method_name, report.model_name)
        method_output_dir = output_dir / method_subdir
        method_output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{report.dataset_name}_{report.method_name}_{report.model_name}_{timestamp}"
        prefix = prefix.replace("/", "-").replace("\\", "-")

        result_path = self._save_result_json(report, method_output_dir, prefix)
        memory_build_path = self._save_memory_build_json(
            report, memory_build_logs, method_output_dir, prefix
        )
        query_answer_path = self._save_query_answer_json(report, method_output_dir, prefix)

        return result_path, memory_build_path, query_answer_path

    def _save_result_json(
        self,
        report: EvaluationReport,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save evaluation metrics file (result.json)."""
        filepath = output_dir / f"{prefix}_result.json"

        result_data = {
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "start_time": report.start_time,
            "end_time": report.end_time,
            "duration_seconds": report.duration_seconds,
            "summary": {
                "total_queries": report.summary.get("total", 0),
                "correct_count": report.summary.get("correct", 0),
                "overall_accuracy": report.summary.get("overall_accuracy", 0.0),
                "overall_avg_score": report.summary.get("overall_avg_score", 0.0),
                "by_type": report.summary.get("by_type", {}),
            },
            "efficiency": report.summary.get("efficiency", {}),
            "memory_build_summary": report.metadata.get("memory_build_summary", {}),
            "llm_usage": report.metadata.get("llm_usage", {}),
            "dataset_specific_metrics": report.metadata.get("lme_v2_metrics",
                                        report.metadata.get("dataset_specific_metrics", {})),
            "config": {
                "evaluation_mode": report.metadata.get("evaluation_mode", ""),
                "evaluation_interval": report.metadata.get("evaluation_interval", 0),
                "total_personas": report.metadata.get("total_personas", 0),
                "method_config": report.config.get("method_config", {}),
                "dataset_config": report.config.get("dataset_config", {}),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return filepath

    def _save_memory_build_json(
        self,
        report: EvaluationReport,
        memory_build_logs: List[Dict[str, Any]],
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save memory build details file (memory_build.json)."""
        filepath = output_dir / f"{prefix}_memory_build.json"

        processed_units = []
        for log in memory_build_logs:
            # Check if this is the new per-session format (MedMemoryBench)
            if "session_builds" in log:
                # New format: per-session builds
                processed_sessions = []
                for sb in log.get("session_builds", []):
                    build_result = sb.get("build_result", {})
                    processed_sessions.append({
                        "session_id": sb.get("session_id"),
                        "session_index": sb.get("session_index"),
                        "method": build_result.get("method", ""),
                        "action": build_result.get("action", ""),
                        "time_cost": build_result.get("time_cost", 0.0),
                        "input_content": build_result.get("input_content", ""),
                        "stored_content": build_result.get("stored_content", ""),
                        "extraction_result": build_result.get("extraction_result", ""),
                        "all_passages": build_result.get("all_passages", []),
                        "memory_entries": build_result.get("memory_entries", []),
                        "chunk_count": build_result.get("chunk_count", 0),
                        "extra": {
                            k: v for k, v in build_result.items()
                            if k not in ["method", "action", "time_cost", "input_content",
                                        "stored_content", "extraction_result", "all_passages",
                                        "memory_entries", "chunk_count", "success"]
                        },
                        "error": sb.get("error"),  # Include error if present
                    })

                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_ids": log.get("session_ids", []),
                    "session_count": log.get("session_count", 0),
                    "total_time": log.get("total_time", 0.0),
                    "total_passages": log.get("total_passages", 0),
                    "session_builds": processed_sessions,
                }

            # Check if this is the batch_builds format (LongMemEval)
            elif "batch_builds" in log:
                processed_batches = []
                for bb in log.get("batch_builds", []):
                    build_result = bb.get("build_result", {})
                    processed_batches.append({
                        "batch_index": bb.get("batch_index"),
                        "session_count": bb.get("session_count", 0),
                        "input_chars": bb.get("input_chars", 0),
                        "input_tokens_est": bb.get("input_tokens_est", 0),
                        "time_cost": bb.get("time_cost", 0.0),
                        "method": build_result.get("method", ""),
                        "action": build_result.get("action", ""),
                        "input_content": build_result.get("input_content", ""),
                        "stored_content": build_result.get("stored_content", ""),
                        "extraction_result": build_result.get("extraction_result", ""),
                        "all_passages": build_result.get("all_passages", []),
                        "memory_entries": build_result.get("memory_entries", []),
                        "chunk_count": build_result.get("chunk_count", 0),
                        "extra": {
                            k: v for k, v in build_result.items()
                            if k not in ["method", "action", "time_cost", "input_content",
                                        "stored_content", "extraction_result", "all_passages",
                                        "memory_entries", "chunk_count", "success"]
                        },
                        "error": bb.get("error"),
                    })

                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_count": log.get("session_count", 0),
                    "batch_count": log.get("batch_count", 0),
                    "sessions_per_batch": log.get("sessions_per_batch", 0),
                    "total_time": log.get("total_time", 0.0),
                    "batch_builds": processed_batches,
                }

            # Check if this is the new chunk_builds format (LoCoMo)
            elif "chunk_builds" in log:
                # LoCoMo format: per-chunk builds
                processed_chunks = []
                for cb in log.get("chunk_builds", []):
                    build_result = cb.get("build_result", {})
                    processed_chunks.append({
                        "chunk_index": cb.get("chunk_index"),
                        "session_ids": cb.get("session_ids", []),
                        "session_count": cb.get("session_count", 0),
                        "input_chars": cb.get("input_chars", 0),
                        "input_tokens_est": cb.get("input_tokens_est", 0),
                        "time_cost": cb.get("time_cost", 0.0),
                        "method": build_result.get("method", ""),
                        "action": build_result.get("action", ""),
                        "input_content": build_result.get("input_content", ""),
                        "stored_content": build_result.get("stored_content", ""),
                        "extraction_result": build_result.get("extraction_result", ""),
                        "all_passages": build_result.get("all_passages", []),
                        "memory_entries": build_result.get("memory_entries", []),
                        "chunk_count": build_result.get("chunk_count", 0),
                        "extra": {
                            k: v for k, v in build_result.items()
                            if k not in ["method", "action", "time_cost", "input_content",
                                        "stored_content", "extraction_result", "all_passages",
                                        "memory_entries", "chunk_count", "success"]
                        },
                        "error": cb.get("error"),  # Include error if present
                    })

                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_ids": log.get("session_ids", []),
                    "session_count": log.get("session_count", 0),
                    "chunk_count": log.get("chunk_count", 0),
                    "chunk_size_config": log.get("chunk_size_config", 0),
                    "total_time": log.get("total_time", 0.0),
                    "total_entries": log.get("total_entries", 0),
                    "total_stored_chunks": log.get("total_stored_chunks", 0),
                    "chunk_builds": processed_chunks,
                }

            else:
                # Legacy format: single build_result per unit
                build_result = log.get("build_result", {})
                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_ids": log.get("session_ids", []),
                    "session_count": log.get("session_count", 0),
                    "method": build_result.get("method", ""),
                    "action": build_result.get("action", ""),
                    "time_cost": build_result.get("time_cost", 0.0),
                    "input_content": build_result.get("input_content", ""),
                    "stored_content": build_result.get("stored_content", ""),
                    "extraction_result": build_result.get("extraction_result", ""),
                    "all_passages": build_result.get("all_passages", []),
                    "memory_entries": build_result.get("memory_entries", []),
                    "chunk_count": build_result.get("chunk_count", 0),
                    "extra": {
                        k: v for k, v in build_result.items()
                        if k not in ["method", "action", "time_cost", "input_content",
                                    "stored_content", "extraction_result", "all_passages",
                                    "memory_entries", "chunk_count", "success"]
                    },
                }

            processed_units.append(processed_unit)

        memory_build_data = {
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "summary": report.metadata.get("memory_build_summary", {}),
            "memory_chunk_size": report.metadata.get("memory_chunk_size"),
            "total_units": len(processed_units),
            "units": processed_units,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(memory_build_data, f, ensure_ascii=False, indent=2)

        return filepath

    def _save_query_answer_json(
        self,
        report: EvaluationReport,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save query answer details file (query_answer.json)."""
        filepath = output_dir / f"{prefix}_query_answer.json"

        query_details = []
        for result in report.detailed_results:
            query_detail = {
                "query_id": result.get("query_id", ""),
                "query_type": result.get("query_type", ""),
                "question": result.get("question", ""),
                "expected_answer": result.get("expected_answer", ""),
                "model_output": result.get("model_output", ""),
                "score": result.get("score", 0.0),
                "is_correct": result.get("is_correct", False),
                "retrieved_memories": result.get("retrieved_memories", []),
                "retrieved_count": result.get("retrieved_count", 0),
                "query_time": result.get("query_time", 0.0),
                "evaluation_details": result.get("details", {}),
                "extra": result.get("extra", {}),
            }
            query_details.append(query_detail)

        by_context = {}
        for result in self._results:
            for ctx_id, ctx_results in self._results_by_context.items():
                if result in ctx_results:
                    if ctx_id not in by_context:
                        by_context[ctx_id] = {
                            "total": 0,
                            "correct": 0,
                            "query_ids": [],
                        }
                    by_context[ctx_id]["total"] += 1
                    by_context[ctx_id]["correct"] += 1 if result.is_correct else 0
                    by_context[ctx_id]["query_ids"].append(result.query_id)
                    break

        query_answer_data = {
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "summary": {
                "total_queries": len(query_details),
                "correct_count": sum(1 for q in query_details if q["is_correct"]),
                "total_query_time": sum(q["query_time"] for q in query_details),
                "avg_query_time": (
                    sum(q["query_time"] for q in query_details) / len(query_details)
                    if query_details else 0.0
                ),
                "avg_retrieved_count": (
                    sum(q["retrieved_count"] for q in query_details) / len(query_details)
                    if query_details else 0.0
                ),
            },
            "by_context": by_context,
            "queries": query_details,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(query_answer_data, f, ensure_ascii=False, indent=2)

        return filepath

    @staticmethod
    def _get_method_subdir(method_name: str, model_name: str) -> str:
        """Generate method subdirectory name."""
        safe_method = method_name.replace("/", "-").replace("\\", "-")
        safe_model = model_name.replace("/", "-").replace("\\", "-")
        return f"{safe_method}_{safe_model}"

    def save_report(self, report: EvaluationReport, output_dir: Path) -> Path:
        """Save evaluation report (backward compatible method)."""
        memory_build_logs = report.metadata.get("memory_build_logs", [])
        result_path, _, _ = self.save_reports(report, output_dir, memory_build_logs)
        return result_path

    def clear(self) -> None:
        self._results.clear()
        self._results_by_context.clear()


def generate_comparison_report(
    reports: List[EvaluationReport],
    output_dir: Path,
) -> Path:
    """Generate multi-method comparison report."""
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "methods": [],
    }

    for report in reports:
        comparison["methods"].append({
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "duration_seconds": report.duration_seconds,
            "summary": report.summary,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"comparison_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(comparison, ensure_ascii=False, indent=2, fp=f)

    return filepath
