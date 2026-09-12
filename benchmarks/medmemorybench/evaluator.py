"""MedMemoryBench evaluation module."""

import hashlib
import copy
import gc
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import numpy as np
from tqdm.auto import tqdm

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager, AgentResponse, MemoryBuildResult
from methods.event_state.store import EventStateStore
from methods.event_state.subjects import normalize_scope
from src.result import EvaluationReport, ResultCollector
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset, MedQuery, MedSession
from benchmarks.base import EvaluationUnit
from benchmarks.snapshot_artifacts import (
    cleanup_memory_state_embedding_artifacts,
    file_sha256,
    fsync_directory,
    load_memory_state_embedding_artifacts,
    memory_state_embedding_artifact_paths,
    publish_memory_state_embedding_artifacts,
)
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from metrics.retrieval_quality import (
    RETRIEVAL_QUALITY_GROUP,
    compute_session_retrieval_quality,
    retrieved_session_ids,
)
from utils.templates import get_prompt_manager
from utils.logger import truncate_error_message
from utils.json_artifacts import dump_json_artifact
from utils.batch_client import create_batch_client
from utils.llm_client import (
    LLMAPIError,
    LLMRetryExhaustedError,
    TokenUsage,
    diff_usage_stats,
    get_usage_tracker,
    merge_usage_stats,
)
from utils.vertex_batch import (
    BatchChatRequest,
    VertexBatchClient,
    VertexBatchError,
    make_request_id,
    PREPARED_QUERY_METADATA_KEY,
    restore_prepared_query,
    scoped_manifest_path,
    snapshot_prepared_query,
)

from benchmarks.medmemorybench.checkpoint import (
    MedMemoryBenchCheckpointManager,
    compute_config_hash,
    compute_memory_query_compatibility_hash,
    compute_query_config_hash,
    derive_legacy_build_config_hash,
    is_manifest_build_compatible,
    is_manifest_query_compatible,
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
        force_resume: bool = False,
        execution_stage: str = "all",
        memory_run: Optional[str] = None,
        append: bool = False,
        append_persona: Optional[int] = None,
        append_unit: Optional[int] = None,
        memory_source_run_dir: Optional[Path] = None,
        memory_run_explicit: bool = False,
        run_scoped_output: bool = False,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
        workers: int = 1,
        query_workers: int = 5,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = logger
        self.resume = resume
        self.force_resume = force_resume
        if execution_stage not in {"all", "memory", "query"}:
            raise ValueError(f"Unsupported execution stage: {execution_stage}")
        self.execution_stage = execution_stage
        self.memory_run = memory_run
        self.append = bool(append)
        self.append_persona = append_persona
        self.append_unit = append_unit
        self.memory_source_run_dir = (
            Path(memory_source_run_dir).resolve()
            if memory_source_run_dir is not None
            else None
        )
        self.memory_run_explicit = memory_run_explicit
        self.run_scoped_output = run_scoped_output
        self.batch_api = batch_api
        self.batch_gcs_uri = batch_gcs_uri
        self.batch_wait = batch_wait
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if query_workers < 1:
            raise ValueError("query_workers must be at least 1")
        self.workers = workers
        self.query_workers = query_workers

        self.prompt_manager = get_prompt_manager(
            dataset=dataset_config.dataset_name,
            method=method_config.method_name,
            language=dataset_config.language,
        )
        self.prompt_protocol = dataset_config.prompt_protocol

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[MedMemoryBenchDataset] = None

        api_config = get_api_config()
        metric_mapping = {
            query_type.name: query_type.metric
            for query_type in dataset_config.query_types
            if query_type.name and query_type.metric
        }
        self.metrics_calculator = MetricsCalculator(
            custom_mapping=metric_mapping,
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            judge_temperature=getattr(api_config, "judge_temperature", 1.0),
            judge_client_max_tokens=getattr(api_config, "judge_client_max_tokens", 10000),
            judge_max_tokens=getattr(api_config, "judge_max_tokens", 500),
            judge_mcd_max_tokens=getattr(api_config, "judge_mcd_max_tokens", 2000),
            language=dataset_config.language,
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []
        self._source_memory_build_logs: List[Dict[str, Any]] = []
        self._batch_client: Optional[VertexBatchClient] = None
        self._judge_batch_client: Optional[VertexBatchClient] = None
        self._batch_fallback_logged = False
        self._judge_batch_fallback_logged = False
        self._deferred_judges: List[Dict[str, Any]] = []
        self._pending_batch_queries: List[Dict[str, Any]] = []
        self._pending_query_plan_requests: List[Dict[str, Any]] = []
        self._compiler_cache_hits = 0
        self._compiler_cache_misses = 0
        self._compiler_calls_this_run = 0
        self._api_failures: List[Dict[str, Any]] = []
        self._api_failure_duration_seconds = 0.0
        self._api_failure_lock = threading.Lock()
        self._memory_build_checkpoint_saved = False

        self._checkpoint_manager: Optional[MedMemoryBenchCheckpointManager] = None
        self._checkpoint_enabled = False
        self._force_resume_persona_id: Optional[int] = None
        self._memory_snapshot_manifest: Optional[Dict[str, Any]] = None
        self._memory_snapshot_run_dir: Optional[Path] = None
        self._evaluation_units: List[EvaluationUnit] = []
        self._memory_unit_ids: Optional[set[int]] = None
        self._append_target_index: Optional[int] = None
        self._append_target_unit_id: Optional[int] = None
        self._append_target_reached = False
        self._append_source_snapshot_keys: set[tuple[int, int]] = set()
        self._append_source_snapshot_records: Dict[tuple[int, int], Dict[str, Any]] = {}
        self._append_source_manifest: Optional[Dict[str, Any]] = None
        # Evaluator-only provenance. Memory methods receive source UIDs, never gold IDs.
        self._source_uid_to_benchmark_session_id: Dict[Any, Dict[str, Optional[Any]]] = {}

        if self._should_enable_checkpoint():
            self._init_checkpoint_manager()

    def _query_worker_count(self) -> int:
        """Use the query-specific limit, retaining compatibility with test fixtures."""
        return max(1, int(getattr(self, "query_workers", getattr(self, "workers", 1))))

    def _log(
        self,
        message: str,
        level: str = "INFO",
        *,
        terminal: bool = True,
    ) -> None:
        level = level.upper()
        if level in {"ERROR", "WARNING"}:
            message = truncate_error_message(message)
        if terminal and self.verbose:
            rendered = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}"
            if (
                getattr(self, "_memory_progress", None) is not None
                or getattr(self, "_query_progress", None) is not None
            ):
                tqdm.write(rendered, file=sys.stdout)
            else:
                print(rendered)
        if self.logger:
            self.logger.log(getattr(logging, level, logging.INFO), message)

    def _configure_query_progress(self, units: List[EvaluationUnit]) -> None:
        """Count pending questions without rendering a bar before query work begins."""
        self._query_progress_total = 0
        if getattr(self, "execution_stage", "all") == "memory":
            return
        total = 0
        checkpoint = getattr(self, "_checkpoint_manager", None)
        for unit in units:
            for query in unit.queries_to_evaluate:
                if not (
                    checkpoint
                    and checkpoint.is_query_completed(
                        query.query_id,
                        persona_id=unit.context_id,
                    )
                ) and not self._is_deferred_judge_query(query.query_id):
                    total += 1
        if not total:
            return
        self._query_progress_lock = threading.Lock()
        self._query_progress_completed = set()
        self._query_progress_total = total

    def _start_query_progress(self) -> None:
        """Render the run-wide counter only when query work starts."""
        if getattr(self, "_query_progress", None) is not None:
            return
        total = getattr(self, "_query_progress_total", 0)
        if not total:
            return
        self._query_progress = tqdm(
            total=total,
            desc="Query progress",
            unit="question",
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not getattr(self, "verbose", True),
        )

    def _start_memory_progress(self, total: int) -> None:
        """Render one per-unit counter while Event-State prepares sessions."""
        if total <= 0 or getattr(self, "_memory_progress", None) is not None:
            return
        self._memory_progress_lock = threading.Lock()
        self._memory_progress = tqdm(
            total=total,
            desc="Memory build",
            unit="session",
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not getattr(self, "verbose", True),
        )

    def _advance_memory_progress(self) -> None:
        progress = getattr(self, "_memory_progress", None)
        if progress is None:
            return
        with self._memory_progress_lock:
            progress.update(1)

    def _finish_memory_progress(self) -> None:
        progress = getattr(self, "_memory_progress", None)
        if progress is not None:
            progress.close()
            self._memory_progress = None

    def _advance_query_progress(
        self,
        query_id: str,
        *,
        context_id: Optional[int] = None,
    ) -> None:
        """Advance once after a question reaches a terminal run outcome."""
        progress = getattr(self, "_query_progress", None)
        if progress is None:
            return
        key = (context_id, query_id)
        with self._query_progress_lock:
            if key in self._query_progress_completed:
                return
            self._query_progress_completed.add(key)
            progress.update(1)

    def _finish_query_progress(self) -> None:
        progress = getattr(self, "_query_progress", None)
        if progress is not None:
            progress.close()
            self._query_progress = None

    def _amem_feature_configuration(self) -> Dict[str, Any]:
        """Describe the build feature combination represented by this run."""
        method_config = getattr(self, "method_config", None)
        method_name = str(getattr(method_config, "method_name", ""))
        build_config = getattr(method_config, "build_config", None)
        if not isinstance(build_config, dict):
            build_config = (
                method_config.snapshot_build_config()
                if hasattr(method_config, "snapshot_build_config")
                else {}
            )
        if method_name.lower() == "event_state":
            retrieval_config = getattr(method_config, "retrieval_config", {})
            if not isinstance(retrieval_config, dict):
                retrieval_config = {}
            features = {
                "episode_archive": bool(build_config.get("enable_episodes", True)),
                "claim_extraction": bool(build_config.get("enable_state_claims", True)),
                "state_compilation": bool(build_config.get("enable_state_compilation", True)),
                "bitemporal_state": bool(build_config.get("enable_bitemporal_time", True)),
                "provenance": bool(build_config.get("preserve_turn_evidence", True)),
            }
            enabled = [name for name, enabled_flag in features.items() if enabled_flag]
            return {
                "schema_version": 4,
                "method_name": method_name,
                "combination_id": "+".join(enabled) if enabled else "none",
                "enabled_features": enabled,
                "features": features,
                "dependencies": [],
                "build_config_hash": self._batch_config_hash() if hasattr(self, "dataset_config") else "",
                "build_config": copy.deepcopy(build_config),
                "retrieval_config": copy.deepcopy(retrieval_config),
            }
        is_experimental = method_name.lower().startswith("amem_test")
        features = {
            "base_memory": True,
            "original_evolution": bool(
                build_config.get("amem_original_evolution", not is_experimental)
            ),
            "typed_relations": bool(build_config.get("amem_typed_relations", False)),
            "temporal_state": bool(build_config.get("amem_temporal_state", False)),
            "provenance": bool(build_config.get("amem_provenance", False)),
        }
        enabled = [name for name, enabled_flag in features.items() if enabled_flag]
        combination = "+".join(enabled) if enabled else "none"
        dependencies = []
        if features["temporal_state"] and not features["typed_relations"]:
            dependencies.append(
                "temporal_state records timestamps, but inferred transitions require typed_relations"
            )
        return {
            "schema_version": 1,
            "method_name": method_name,
            "combination_id": combination,
            "enabled_features": enabled,
            "features": features,
            "dependencies": dependencies,
            "build_config_hash": (
                self._batch_config_hash()
                if method_config is not None and hasattr(self, "dataset_config")
                else ""
            ),
            "build_config": copy.deepcopy(build_config),
        }

    def _session_retrieval_quality_enabled(self) -> bool:
        """Return whether this run builds session-level experimental A-MEM notes."""
        method_config = getattr(self, "method_config", None)
        if str(getattr(method_config, "method_name", "")).lower() not in {"amem_test", "event_state"}:
            return False
        build_config = getattr(method_config, "build_config", {})
        if not isinstance(build_config, dict):
            build_config = {}
        note_level = str(build_config.get("amem_note_level", "turn"))
        if str(getattr(method_config, "method_name", "")).lower() == "event_state":
            return True
        return note_level.strip().lower() == "session"

    @staticmethod
    def _method_source_uid(session: MedSession) -> Any:
        return session.source_uid or session.session_id

    @staticmethod
    def _benchmark_session_id(session: MedSession) -> Optional[Any]:
        if session.is_noise:
            return None
        return (
            session.benchmark_session_id
            if session.benchmark_session_id is not None
            else session.session_id
        )

    def _validate_unit_source_integrity(self, unit: EvaluationUnit) -> Dict[str, Any]:
        """Validate source ingress and retain private source-to-gold provenance."""
        sessions = [item for item in unit.sessions_to_inject if isinstance(item, MedSession)]
        source_uids = [self._method_source_uid(session) for session in sessions]
        duplicate_uids = sorted({uid for uid in source_uids if source_uids.count(uid) > 1})
        if duplicate_uids or len(source_uids) != len(set(source_uids)):
            raise ValueError(
                f"MedMemoryBench source UID collision for persona {unit.context_id}, "
                f"unit {unit.unit_id}: {duplicate_uids}"
            )
        mapping = {
            self._method_source_uid(session): self._benchmark_session_id(session)
            for session in sessions
        }
        if any(
            session.is_noise and mapping[self._method_source_uid(session)] is not None
            for session in sessions
        ):
            raise ValueError(
                f"MedMemoryBench distractor has benchmark provenance for persona {unit.context_id}, unit {unit.unit_id}"
            )
        clean_ids = [value for value in mapping.values() if value is not None]
        if len(clean_ids) != len(set(clean_ids)):
            raise ValueError(
                f"MedMemoryBench clean benchmark provenance is non-unique for persona {unit.context_id}, unit {unit.unit_id}"
            )
        source_mappings = getattr(self, "_source_uid_to_benchmark_session_id", None)
        if source_mappings is None:
            source_mappings = self._source_uid_to_benchmark_session_id = {}
        context_mapping = source_mappings.setdefault(unit.context_id, {})
        collisions = [uid for uid, value in mapping.items() if uid in context_mapping and context_mapping[uid] != value]
        if collisions:
            raise ValueError(
                f"MedMemoryBench source provenance changed for persona {unit.context_id}: {collisions}"
            )
        context_mapping.update(mapping)
        counts = {
            "source_identity_schema_version": 1,
            "input_source_record_count": len(sessions),
            "unique_source_uid_count": len(set(source_uids)),
            "clean_source_record_count": sum(value is not None for value in mapping.values()),
            "noise_source_record_count": sum(value is None for value in mapping.values()),
            "source_uid_collision_count": len(source_uids) - len(set(source_uids)),
            "general_non_personal_source_count": sum(session.conversation_scope == "general_non_personal" for session in sessions),
            "third_party_source_count": sum(session.conversation_scope.startswith("third_party:") for session in sessions),
        }
        if counts["input_source_record_count"] != counts["unique_source_uid_count"]:
            raise ValueError(f"MedMemoryBench source UID invariant failed for persona {unit.context_id}, unit {unit.unit_id}")
        return counts

    def _noise_intrusion_diagnostics(
        self,
        retrieved_memories: List[Dict[str, Any]],
        context_id: Optional[Any],
    ) -> Dict[str, Any]:
        mapping = getattr(self, "_source_uid_to_benchmark_session_id", {}).get(context_id)
        source_uids = retrieved_session_ids(retrieved_memories)
        if mapping is None:
            return {
                "source_mapping_schema_version": 1,
                "selected_clean_source_count": len(source_uids),
                "selected_noise_source_count": 0,
                "item_level_noise_intrusion_rate": 0.0,
                "clean_evidence_rate": 1.0 if source_uids else 0.0,
            }
        unknown = [uid for uid in source_uids if uid not in mapping]
        if unknown:
            raise ValueError(
                f"MedMemoryBench retrieval references unknown source UIDs for persona {context_id}: {unknown}"
            )
        clean_count = sum(mapping[uid] is not None for uid in source_uids)
        noise_count = len(source_uids) - clean_count
        return {
            "source_mapping_schema_version": 1,
            "selected_clean_source_count": clean_count,
            "selected_noise_source_count": noise_count,
            "item_level_noise_intrusion_rate": noise_count / len(source_uids) if source_uids else 0.0,
            "clean_evidence_rate": clean_count / len(source_uids) if source_uids else 0.0,
        }

    def _session_retrieval_quality(
        self,
        query: Any,
        retrieved_memories: List[Dict[str, Any]],
        context_id: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._session_retrieval_quality_enabled():
            return None
        quality = compute_session_retrieval_quality(
            retrieved_memories=retrieved_memories,
            source_key_points=getattr(query, "source_key_points", []),
            metadata=getattr(query, "metadata", {}),
            source_uid_to_benchmark_session_id=getattr(self, "_source_uid_to_benchmark_session_id", {}).get(context_id),
        )
        if str(getattr(self.method_config, "method_name", "")).lower() != "event_state":
            return quality
        visible_records = []
        for record in retrieved_memories:
            copied = copy.deepcopy(record)
            if copied.get("type") == "state_claim":
                included_in_context = bool(copied.get("included_in_context", False))
                included_provenance = copied.get("included_provenance_evidence", [])
                if not isinstance(included_provenance, list):
                    included_provenance = []
                if not included_in_context and not included_provenance:
                    continue
                if not included_in_context:
                    copied["source_session_id"] = None
                # Answer visibility is limited to prompt-visible state or excerpts.
                copied.pop("provenance", None)
                copied.pop("raw_evidence_added", None)
                copied.pop("retrieval_audit", None)
                copied["provenance_evidence"] = included_provenance
            elif not copied.get("included_in_context", False):
                continue
            visible_records.append(copied)
        lineage_records = []
        for record in retrieved_memories:
            copied = copy.deepcopy(record)
            if copied.get("type") == "state_claim":
                copied["provenance_evidence"] = copied.get("all_provenance_evidence", [])
            lineage_records.append(copied)
        quality["claim_lineage"] = compute_session_retrieval_quality(
            retrieved_memories=lineage_records,
            source_key_points=getattr(query, "source_key_points", []),
            metadata=getattr(query, "metadata", {}),
            source_uid_to_benchmark_session_id=getattr(self, "_source_uid_to_benchmark_session_id", {}).get(context_id),
        )
        quality["answer_visible"] = compute_session_retrieval_quality(
            retrieved_memories=visible_records,
            source_key_points=getattr(query, "source_key_points", []),
            metadata=getattr(query, "metadata", {}),
            source_uid_to_benchmark_session_id=getattr(self, "_source_uid_to_benchmark_session_id", {}).get(context_id),
        )
        quality["retrieval_quality_claim_lineage"] = quality["claim_lineage"]
        quality["retrieval_quality_answer_visible"] = quality["answer_visible"]
        quality["provenance_semantics"] = "selected claim origin plus selected episodes; answer_visible uses included evidence only"
        quality["noise_intrusion"] = self._noise_intrusion_diagnostics(
            visible_records, context_id
        )
        return quality

    @staticmethod
    def _attach_session_retrieval_quality(
        result: MetricResult,
        retrieval_quality: Optional[Dict[str, Any]],
    ) -> None:
        if retrieval_quality is None:
            return
        metric_groups = result.details.get("metric_groups")
        if not isinstance(metric_groups, dict):
            metric_groups = {}
            result.details["metric_groups"] = metric_groups
        metric_groups[RETRIEVAL_QUALITY_GROUP] = retrieval_quality

    @staticmethod
    def _feature_for_operation(operation: str) -> str:
        parts = str(operation).split(".")
        if len(parts) >= 2 and parts[0] == "amem":
            return parts[1]
        if len(parts) >= 2 and parts[0] == "event_state":
            return {"extract": "claim_extraction", "extract_repair": "claim_extraction", "update_classify": "state_compilation", "embedding": "embedding"}.get(parts[1], parts[1])
        return "unscoped"

    def _build_metrics_report(
        self,
        logs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-unit telemetry into a report-ready feature breakdown."""
        selected_logs = list(
            logs
            if logs is not None
            else self._combined_memory_build_logs()
        )
        usage_values = []
        units = []
        total_wall_time = 0.0
        total_sessions = 0
        complete_telemetry = True
        measured_memory_units = 0
        memory_snapshot_bytes = 0
        previous_memory_bytes: Dict[Any, int] = {}
        latest_memory_by_context: Dict[Any, Dict[str, Any]] = {}
        for log in selected_logs:
            metrics = log.get("build_metrics") or {}
            usage = metrics.get("usage") or {}
            if usage:
                usage_values.append(usage)
            wall_time = float(
                metrics.get("wall_time_seconds", log.get("total_time", 0.0)) or 0.0
            )
            total_wall_time += wall_time
            total_sessions += int(log.get("session_count", 0) or 0)
            legacy_metrics_unavailable = bool(
                metrics.get("legacy_metrics_unavailable", False)
            )
            complete_telemetry = complete_telemetry and not legacy_metrics_unavailable
            context_id = log.get("context_id")
            memory_size = log.get("memory_size") or metrics.get("memory_size")
            unit_memory_size: Optional[Dict[str, Any]] = None
            if isinstance(memory_size, dict) and memory_size.get("bytes") is not None:
                unit_memory_size = copy.deepcopy(memory_size)
                memory_bytes = int(unit_memory_size["bytes"])
                prior_bytes = previous_memory_bytes.get(context_id, 0)
                delta_bytes = memory_bytes - prior_bytes
                unit_memory_size.update({
                    "delta_from_previous_unit_bytes": delta_bytes,
                    "delta_from_previous_unit_mib": round(
                        delta_bytes / (1024 ** 2), 6
                    ),
                })
                previous_memory_bytes[context_id] = memory_bytes
                measured_memory_units += 1
                memory_snapshot_bytes += memory_bytes
                latest_memory_by_context[context_id] = {
                    "unit_id": log.get("unit_id"),
                    "bytes": memory_bytes,
                    "mib": round(memory_bytes / (1024 ** 2), 6),
                }
            units.append({
                "unit_id": log.get("unit_id"),
                "context_id": context_id,
                "session_count": log.get("session_count", 0),
                "wall_time_seconds": round(wall_time, 6),
                "restored_from_snapshot": bool(log.get("restored_from_snapshot", False)),
                "legacy_metrics_unavailable": legacy_metrics_unavailable,
                "memory_size": unit_memory_size,
                "usage": usage,
            })

        merged_usage = merge_usage_stats(*usage_values)
        operations = (
            merged_usage.get("operations", {}).get("memorize", {})
        )
        feature_configuration = self._amem_feature_configuration()
        feature_totals: Dict[str, TokenUsage] = {
            "base": TokenUsage(),
            "embedding": TokenUsage(),
        }
        for feature, enabled in feature_configuration["features"].items():
            if enabled and feature != "base_memory":
                feature_totals.setdefault(feature, TokenUsage())
        for operation, usage in operations.items():
            feature = self._feature_for_operation(operation)
            feature_totals.setdefault(feature, TokenUsage()).merge(
                TokenUsage.from_dict(usage)
            )
        scoped_wall_time = sum(
            float(usage.get("wall_time", 0.0) or 0.0)
            for usage in operations.values()
        )
        memorize_totals = dict(merged_usage.get("memorize_phase", {}))
        memorize_totals.pop("wall_time", None)
        memorize_totals.update({
            "wall_time_seconds": round(total_wall_time, 6),
            "scoped_operation_wall_time_seconds": round(scoped_wall_time, 6),
            "unattributed_wall_time_seconds": round(
                max(total_wall_time - scoped_wall_time, 0.0), 6
            ),
        })
        overall_memory_bytes = sum(
            item["bytes"] for item in latest_memory_by_context.values()
        )
        memory_size_report = {
            "schema_version": 1,
            "measurement": "serialized_memory_state",
            "unit": "bytes",
            "available": measured_memory_units > 0,
            "complete": bool(units) and measured_memory_units == len(units),
            "overall_bytes": overall_memory_bytes,
            "overall_mib": round(overall_memory_bytes / (1024 ** 2), 6),
            "unit_snapshot_bytes": memory_snapshot_bytes,
            "unit_snapshot_mib": round(memory_snapshot_bytes / (1024 ** 2), 6),
            "context_count": len({unit.get("context_id") for unit in units}),
            "measured_unit_count": measured_memory_units,
            "by_context": {
                str(context_id): value
                for context_id, value in sorted(
                    latest_memory_by_context.items(),
                    key=lambda item: str(item[0]),
                )
            },
        }
        memorize_totals.update({
            "memory_size_bytes": overall_memory_bytes,
            "memory_size_mib": memory_size_report["overall_mib"],
        })
        return {
            "schema_version": 2,
            "feature_configuration": feature_configuration,
            "unit_count": len(units),
            "session_count": total_sessions,
            "complete_telemetry": complete_telemetry,
            "totals": memorize_totals,
            "memory_size": memory_size_report,
            "by_feature": {
                feature: usage.to_dict()
                for feature, usage in sorted(feature_totals.items())
            },
            "by_operation": operations,
            "units": units,
            "usage": merged_usage,
        }

    def _combined_memory_build_logs(self) -> List[Dict[str, Any]]:
        """Return one cumulative build record per evaluation unit."""
        combined: Dict[tuple[Any, Any], Dict[str, Any]] = {}
        for log in getattr(self, "_source_memory_build_logs", []):
            combined[(log.get("context_id"), log.get("unit_id"))] = log
        for log in getattr(self, "_memory_build_logs", []):
            combined[(log.get("context_id"), log.get("unit_id"))] = log
        unit_order = {
            (unit.context_id, unit.unit_id): index
            for index, unit in enumerate(self._get_evaluation_units())
        }
        return sorted(
            combined.values(),
            key=lambda log: unit_order.get(
                (log.get("context_id"), log.get("unit_id")),
                len(unit_order),
            ),
        )

    def _record_source_build_metrics(
        self,
        unit: EvaluationUnit,
        payload: Dict[str, Any],
    ) -> None:
        metrics = payload.get("memory_build_metrics")
        if not isinstance(metrics, dict):
            metrics = {
                "wall_time_seconds": float(
                    payload.get("memory_build_time", 0.0) or 0.0
                ),
                "usage": {},
                "legacy_metrics_unavailable": True,
            }
        logs = getattr(self, "_source_memory_build_logs", None)
        if logs is None:
            self._source_memory_build_logs = []
            logs = self._source_memory_build_logs
        logs[:] = [item for item in logs if item.get("unit_id") != unit.unit_id]
        logs.append({
            "unit_id": unit.unit_id,
            "context_id": unit.context_id,
            "session_count": len(unit.sessions_to_inject),
            "total_time": payload.get("memory_build_time", 0.0),
            "build_metrics": copy.deepcopy(metrics),
            "memory_size": copy.deepcopy(
                payload.get("memory_size") or metrics.get("memory_size")
            ),
            "restored_from_snapshot": True,
        })

    def _record_api_failure(
        self,
        phase: str,
        error: Any,
        failure_duration_seconds: Optional[float] = None,
        **context: Any,
    ) -> None:
        """Record provider failures without constructing a metric result."""
        if failure_duration_seconds is None:
            failure_duration_seconds = getattr(
                error,
                "_medmemorybench_api_duration_seconds",
                0.0,
            )
        try:
            failure_duration_seconds = max(float(failure_duration_seconds), 0.0)
        except (TypeError, ValueError):
            failure_duration_seconds = 0.0

        structured = error if isinstance(error, dict) else None
        root_error = (
            error.last_exception
            if isinstance(error, LLMRetryExhaustedError)
            else error
        )
        if structured is not None:
            root_error = None
        failure = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "error_type": (
                structured.get("error_type", "UnknownError")
                if structured is not None
                else type(error).__name__
            ),
            "error_message": (
                truncate_error_message(structured.get("error_message", ""))
                if structured is not None
                else truncate_error_message(error)
            ),
            **{key: value for key, value in context.items() if value is not None},
        }
        if structured is not None:
            for key in (
                "operation",
                "memory_id",
                "root_error_type",
                "root_error_message",
                "attempts",
                "failure_type",
                "retry_counts",
                "duration_seconds",
            ):
                if structured.get(key) is not None:
                    failure[key] = structured[key]
            for key in ("error_message", "root_error_message"):
                if key in failure:
                    failure[key] = truncate_error_message(failure[key])
        else:
            failure["root_error_type"] = type(root_error).__name__
            failure["root_error_message"] = truncate_error_message(root_error)
        if isinstance(error, LLMRetryExhaustedError):
            failure["attempts"] = error.attempts
            if error.failure_type:
                failure["failure_type"] = error.failure_type
            if error.retry_counts:
                failure["retry_counts"] = error.retry_counts

        def store_failure() -> None:
            if failure_duration_seconds:
                failure["duration_seconds"] = failure_duration_seconds
                self._api_failure_duration_seconds = (
                    getattr(self, "_api_failure_duration_seconds", 0.0)
                    + failure_duration_seconds
                )
            self._api_failures.append(failure)

        lock = getattr(self, "_api_failure_lock", None)
        if lock is None:
            store_failure()
        else:
            with lock:
                store_failure()

    @staticmethod
    def _run_api_call(operation, *args, **kwargs):
        """Attach elapsed time to an API exception without changing its type."""
        tracker = get_usage_tracker()
        failure_duration_before = tracker.current_failure_duration()
        started_at = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        except LLMAPIError as error:
            elapsed = time.perf_counter() - started_at
            tracked_failure_duration = max(
                tracker.current_failure_duration() - failure_duration_before,
                0.0,
            )
            previous_duration = getattr(
                error,
                "_medmemorybench_api_duration_seconds",
                0.0,
            )
            error._medmemorybench_api_duration_seconds = (
                max(float(previous_duration), 0.0)
                + max(elapsed - tracked_failure_duration, 0.0)
            )
            raise

    def _drain_memory_api_failures(self, context_id: Any) -> List[Dict[str, Any]]:
        """Collect retry-exhausted A-MEM build calls even when a build aborts."""
        manager = getattr(self, "agent_manager", None)
        agent = getattr(manager, "_agent", None)
        getter = getattr(agent, "_get_memory_system", None)
        if not callable(getter):
            return []
        system = getter(context_id)
        drain = getattr(system, "drain_api_failure_events", None)
        return list(drain()) if callable(drain) else []

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
        self._source_uid_to_benchmark_session_id = {
            persona_id: self.dataset.get_source_identity_mapping(persona_id)
            for persona_id in self.dataset.get_persona_ids()
        }
        if self.dataset_config.evaluation_mode == "merged":
            self._source_uid_to_benchmark_session_id[0] = (
                self.dataset.get_all_source_identity_mappings()
            )
        self._evaluation_units = list(self.dataset.get_evaluation_units())
        self._evaluation_total_sessions = sum(
            len(unit.sessions_to_inject) for unit in self._evaluation_units
        )
        self._evaluation_total_queries = sum(
            len(unit.queries_to_evaluate) for unit in self._evaluation_units
        )

        self._log(
            f"Dataset ready | sessions={self._evaluation_total_sessions} | "
            f"queries={self._evaluation_total_queries} | "
            f"mode={self.dataset_config.evaluation_mode} | "
            f"interval={self.dataset_config.evaluation_interval}"
        )

    def _should_enable_checkpoint(self) -> bool:
        return self.dataset_config.evaluation_mode == "independent"

    def _session_level_force_resume(self) -> bool:
        """Use forced session skipping only for backends without A-MEM snapshots."""
        method_name = str(
            getattr(getattr(self, "method_config", None), "method_name", "")
        ).lower()
        return bool(getattr(self, "force_resume", False) and not (method_name.startswith("amem") or method_name == "event_state"))

    def _get_evaluation_units(self) -> List[EvaluationUnit]:
        """Return the loaded units without consuming the dataset iterator twice."""
        cached = getattr(self, "_evaluation_units", None)
        if cached:
            return list(cached)
        if self.dataset is None:
            return []
        return list(self.dataset.get_evaluation_units())

    def _prepare_append_plan(self) -> None:
        """Resolve the append target and validate the source snapshot prefix."""
        if not getattr(self, "append", False):
            return
        if self.dataset_config.evaluation_mode != "independent":
            raise ValueError("--append currently requires independent evaluation mode")
        if self.memory_source_run_dir is None:
            raise ValueError("--append requires a memory source run")
        if getattr(self, "append_persona", None) is None or getattr(self, "append_unit", None) is None:
            raise ValueError("--append requires both --persona and --unit")

        units = self._get_evaluation_units()
        target_matches = [
            (index, unit)
            for index, unit in enumerate(units)
            if unit.context_id == getattr(self, "append_persona", None)
            and unit.unit_id == getattr(self, "append_unit", None)
        ]
        if len(target_matches) != 1:
            raise ValueError(
                f"Append target persona={getattr(self, 'append_persona', None)}, "
                f"unit={getattr(self, 'append_unit', None)} "
                "does not identify exactly one evaluation unit in the stored dataset"
            )
        target_index, target_unit = target_matches[0]
        self._append_target_index = target_index
        self._append_target_unit_id = target_unit.unit_id
        target_key = (target_unit.context_id, target_unit.unit_id)

        manifest_path = self.memory_source_run_dir / "memory" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read append source manifest {manifest_path}: {truncate_error_message(exc)}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != "medmemorybench.memory_manifest"
            or manifest.get("version") != 1
            or manifest.get("status") not in {"building", "complete"}
        ):
            raise ValueError(f"Append source manifest is unavailable or invalid: {manifest_path}")

        unit_lookup = {(unit.context_id, unit.unit_id): index for index, unit in enumerate(units)}
        records: Dict[tuple[int, int], Dict[str, Any]] = {}
        processed_indices: List[int] = []
        for record in manifest.get("snapshots", []):
            if not isinstance(record, dict):
                raise ValueError(f"Append source manifest contains an invalid snapshot record: {manifest_path}")
            try:
                key = (int(record["context_id"]), int(record["unit_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Append source snapshot record is missing its unit identity: {manifest_path}") from exc
            if key not in unit_lookup:
                raise ValueError(
                    f"Append source snapshot {key} does not belong to the stored dataset configuration"
                )
            if key in records:
                raise ValueError(f"Append source contains duplicate snapshot record {key}")
            records[key] = record
            processed_indices.append(unit_lookup[key])

        processed_indices.sort()
        if processed_indices:
            expected_prefix = list(range(processed_indices[-1] + 1))
            if processed_indices != expected_prefix:
                raise ValueError(
                    "Append source snapshots are not a contiguous evaluation prefix; "
                    "repair or rebuild the source run before appending"
                )
        self._append_target_reached = target_key in records

        self._append_source_manifest = manifest
        self._append_source_snapshot_records = records
        self._append_source_snapshot_keys = set(records)

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
            self._log(
                "Checkpoint and previous-good backup are unavailable or corrupted; "
                "starting from scratch",
                level="WARNING",
            )
            self._checkpoint_manager.delete()
            return False

        if self._checkpoint_manager.recovered_from_backup:
            self._log(
                "Primary checkpoint was corrupted; restored the previous-good backup.",
                level="WARNING",
            )

        config_matches = self._checkpoint_manager.validate_config()
        force_requested = bool(getattr(self, "force_resume", False))
        force_resume = self._session_level_force_resume()
        method_name = str(
            getattr(getattr(self, "method_config", None), "method_name", "")
        ).lower()
        if (
            not config_matches
            and force_requested
            and (method_name.startswith("amem") or method_name == "event_state")
        ):
            raise ValueError(
                f"Cannot force-resume a {self._memory_method_label()} run after "
                "memory-build configuration changes. Resume with the stored "
                "configuration or start a new run."
            )
        if not config_matches and not force_requested:
            self._log("Config changed, checkpoint invalid, starting from scratch")
            self._checkpoint_manager.delete()
            return False

        if not config_matches:
            self._log(
                "Config changed; --force is keeping the existing checkpoint and "
                "continuing with the new configuration.",
                level="WARNING",
            )
            self._checkpoint_manager.adopt_config_hash()

        if not self._checkpoint_manager.is_independent_mode():
            self._log("Checkpoint not in independent mode, cannot resume")
            return False

        resume_info = self._checkpoint_manager.get_resume_info()
        if force_resume and resume_info.get("active_session_id") is not None:
            raise RuntimeError(
                "Cannot safely force-resume because the last session may already have "
                "committed to the persistent memory backend. Resolve or rebuild that "
                "backend before resuming."
            )

        rollback = self._checkpoint_manager.rollback_incomplete_session()
        if rollback is not None:
            self._log(
                f"Rolled back unfinished session {rollback['session_id']} "
                f"from evaluation unit {rollback['unit_id']}; it will be retried.",
                level="WARNING",
            )

        self._load_checkpoint_results()

        info = self._checkpoint_manager.get_resume_info()
        if force_resume:
            self._force_resume_persona_id = info["current_persona"]
            self._log(
                "Forced session-level resume trusts the memory backend to retain all "
                "sessions recorded by the checkpoint.",
                level="WARNING",
            )
        self._log("=" * 50)
        self._log("Resuming from checkpoint")
        self._log(f"  Checkpoint ID: {info['checkpoint_id'][:8]}...")
        self._log(f"  Completed Personas: {info['completed_personas']}/{info['total_personas']}")
        self._log(f"  Completed Queries: {info['completed_queries']}/{info['total_queries']}")
        if info['current_persona'] is not None:
            resume_action = "continuing stored memory" if force_resume else "rebuilding memory"
            self._log(
                f"  Current Persona: {info['current_persona']} "
                f"({info['current_persona_completed_queries']} queries done, "
                f"{info['current_persona_injected_sessions']} sessions injected, "
                f"{resume_action})"
            )
        self._log("=" * 50)

        return True

    def _create_new_checkpoint(self) -> None:
        if not self._checkpoint_manager:
            return

        total_personas = len(self.dataset.get_persona_ids())
        total_queries = getattr(
            self,
            "_evaluation_total_queries",
            self.dataset.get_total_queries(),
        )

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
                    # Reset releases resources (closes Qdrant client, etc.)
                    self.agent_manager.reset()

                    # Explicitly delete old agent manager to release all references
                    old_manager = self.agent_manager
                    self.agent_manager = None
                    del old_manager

                    # Force garbage collection to release resources
                    import gc
                    gc.collect()

                    # Give time for resources to be fully released
                    import time
                    time.sleep(1.0)
                except Exception as e:
                    self._log(
                        f"Failed to reset previous agent: {truncate_error_message(e)}",
                        level="WARNING",
                    )

            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=self.batch_api,
                batch_gcs_uri=getattr(self, "batch_gcs_uri", None),
                batch_wait=getattr(self, "batch_wait", False),
                workers=self.workers,
                batch_manifest_dir=self.output_dir / "batch",
                batch_config_hash=self._batch_manifest_config_hash(),
                batch_progress_callback=self._log,
                query_only=self.execution_stage == "query",
            )

        self.agent_manager.set_context_id(context_id)

    def _batch_config_hash(self) -> str:
        if getattr(self, "execution_stage", "all") == "query":
            return compute_query_config_hash(self.method_config, self.dataset_config)
        return compute_config_hash(self.method_config, self.dataset_config)

    def _query_compiler_cache_config(self) -> Dict[str, Any]:
        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
        configured_path = retrieval.get("query_compiler_plan_cache_path")
        return {
            "enabled": bool(retrieval.get("query_compiler_plan_cache_enabled", True)),
            "path": Path(configured_path) if configured_path else self.output_dir / "query_compiler_plans.jsonl",
        }

    def _query_compiler_cache_fingerprint_payload(
        self, question: str, reference_time: Optional[str] = None, *, include_max_tokens: bool = True,
    ) -> Dict[str, Any]:
        """Return compiler-only cache inputs, optionally in the legacy shape."""
        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
        model = self.method_config.model
        payload = {
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "reference_time": reference_time,
            "provider": getattr(model, "provider", None), "model": getattr(model, "name", None),
            "prompt_version": "event_state_query_compiler_v2", "schema_version": 2,
            "max_searches": retrieval.get("query_compiler_max_searches", 3),
            "temperature": retrieval.get("planner_temperature", 0.0),
        }
        if include_max_tokens:
            payload["query_compiler_max_tokens"] = retrieval.get("query_compiler_max_tokens", 256)
        return payload

    def _query_compiler_cache_fingerprint(self, question: str, reference_time: Optional[str] = None) -> str:
        """Fingerprint compiler inputs only; retrieval changes reuse plans."""
        payload = self._query_compiler_cache_fingerprint_payload(question, reference_time)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    def _legacy_query_compiler_cache_entry(
        self, plan_cache: Dict[str, Dict[str, Any]], question: str, reference_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reuse pre-token-limit v2 rows only at their known default of 256."""
        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
        if retrieval.get("query_compiler_max_tokens", 256) != 256:
            return None
        payload = self._query_compiler_cache_fingerprint_payload(
            question, reference_time, include_max_tokens=False,
        )
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return plan_cache.get(fingerprint)

    def _load_query_compiler_plan_cache(self) -> Dict[str, Dict[str, Any]]:
        config = self._query_compiler_cache_config()
        if not config["enabled"] or not config["path"].is_file():
            return {}
        try:
            with config["path"].open("r", encoding="utf-8") as handle:
                return {
                    entry["fingerprint"]: entry for entry in map(json.loads, handle)
                    if isinstance(entry, dict) and isinstance(entry.get("fingerprint"), str)
                    and isinstance(entry.get("raw_model_output"), str)
                }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._log(f"Ignoring unreadable query compiler plan cache: {exc}", level="WARNING")
            return {}

    def _append_query_compiler_plan_cache(self, entry: Dict[str, Any]) -> None:
        config = self._query_compiler_cache_config()
        if not config["enabled"]:
            return
        try:
            config["path"].parent.mkdir(parents=True, exist_ok=True)
            with config["path"].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            self._log(f"Could not persist query compiler plan cache: {exc}", level="WARNING")

    @staticmethod
    def _export_event_state_for_transfer(manager: AgentManager, context_id: Any) -> Dict[str, Any]:
        """Clone immutable snapshot state for stage-two local retrieval."""
        state = manager.export_memory_state(context_id=context_id)
        exporter = getattr(manager, "export_memory_binary_artifacts", None)
        if callable(exporter):
            artifacts = exporter(context_id=context_id)
            if artifacts:
                state["embedding_artifacts"] = artifacts
        return state

    def _answer_query_kwargs(self, query: MedQuery) -> Dict[str, Any]:
        """Keep benchmark type metadata out of neutral agent-facing requests."""
        kwargs: Dict[str, Any] = {"raw_question": query.question}
        prompt_protocol = getattr(self, "prompt_protocol", "type_aware")
        if prompt_protocol == "type_aware":
            kwargs["query_type"] = query.query_type
        else:
            kwargs["query_system_prompt"] = self.prompt_manager.get_query_system_prompt(
                prompt_protocol=prompt_protocol,
            )
        return kwargs

    def _judge_batch_config_hash(self) -> str:
        """Include `.env` judge settings in judge submit/resume identity."""
        api_config = get_api_config()
        payload = {
            "method_config_hash": self._batch_manifest_config_hash(),
            "provider": api_config.get_judge_provider(),
            "model": api_config.get_judge_model(),
            "temperature": getattr(api_config, "judge_temperature", 1.0),
            "reasoning_effort": getattr(api_config, "judge_reasoning_effort", None),
            "client_max_tokens": getattr(api_config, "judge_client_max_tokens", 10000),
            "max_tokens": getattr(api_config, "judge_max_tokens", 500),
            "mcd_max_tokens": getattr(api_config, "judge_mcd_max_tokens", 2000),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:48]

    def _supports_memory_snapshots(self) -> bool:
        support = getattr(self.agent_manager, "supports_memory_snapshots", None)
        return bool(callable(support) and support())

    def _supports_configured_memory_snapshots(self) -> bool:
        """Check stage eligibility before constructing a mutable adapter."""
        return AgentManager.supports_configured_memory_snapshots(self.method_config)

    def _supports_staged_queries(self) -> bool:
        manager = getattr(self, "agent_manager", None)
        if manager is None:
            # Lightweight test doubles and legacy callers only expose the
            # snapshot capability; preserve their historical dispatch.
            return self._supports_memory_snapshots()
        support = getattr(manager, "supports_staged_queries", None)
        return bool(callable(support) and support())

    def _memory_snapshot_root(self) -> Path:
        if getattr(self, "run_scoped_output", False):
            return self.output_dir / "memory"
        safe_method = self.method_config.method_name.replace("/", "-").replace("\\", "-")
        safe_model = self.method_config.model.name.replace("/", "-").replace("\\", "-")
        return self.output_dir / f"{safe_method}_{safe_model}" / "memory"

    def _run_scoped_experiment_dir(self) -> Path:
        memory_source_run_dir = getattr(self, "memory_source_run_dir", None)
        if memory_source_run_dir is not None:
            return memory_source_run_dir.parent
        if self.output_dir.parent.name == "query_runs":
            return self.output_dir.parent.parent.parent
        return self.output_dir.parent

    def _legacy_memory_snapshot_root(self) -> Path:
        """Return the pre-run-directory memory root for compatibility."""
        if getattr(self, "run_scoped_output", False):
            return self._run_scoped_experiment_dir() / "memory"
        return self._memory_snapshot_root()

    def _memory_snapshot_dir(self) -> Path:
        if self._memory_snapshot_run_dir is None:
            raise RuntimeError(
                f"{self._memory_method_label()} memory snapshot run has not been selected"
            )
        return self._memory_snapshot_run_dir

    def _memory_method_label(self) -> str:
        """Return the user-facing name for the active snapshot-backed method."""
        method_name = str(getattr(self.method_config, "method_name", "")).lower()
        labels = {
            "long_context": "Long Context",
            "embedding_rag": "Embedding RAG",
            "bm25_rag": "BM25 RAG",
            "event_state": "Event-State",
        }
        return labels.get(method_name, "A-MEM")

    def _memory_snapshot_path(self, unit: EvaluationUnit) -> Path:
        return self._memory_snapshot_dir() / (
            f"persona_{unit.context_id}_unit_{unit.unit_id}.json"
        )

    def _memory_snapshot_manifest_path(self) -> Path:
        return self._memory_snapshot_dir() / "manifest.json"

    def _memory_snapshot_embeddings_path(self, unit: EvaluationUnit) -> Path:
        return self._memory_snapshot_path(unit).with_suffix(".embeddings.npy")

    def _artifact_relative_path(self, path: Optional[Path]) -> Optional[str]:
        """Return an artifact path relative to this run when possible."""
        if path is None or not hasattr(self, "output_dir"):
            return None
        return os.path.relpath(Path(path), self.output_dir)

    def _query_artifact_references(
        self,
        *,
        context_id: Optional[int],
        unit_id: Optional[int],
        answer_request_id: Optional[str] = None,
        answer_batch_client: Optional[Any] = None,
        answer_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build compact, joinable provenance for one scored query."""
        references: Dict[str, Any] = {}
        manifest = getattr(self, "_memory_snapshot_manifest", None) or {}
        if context_id is not None and unit_id is not None and manifest:
            snapshot = next(
                (
                    item for item in manifest.get("snapshots", [])
                    if item.get("context_id") == context_id
                    and item.get("unit_id") == unit_id
                ),
                {},
            )
            if snapshot:
                snapshot_path = self._memory_snapshot_dir() / snapshot.get("path", "")
                references["memory_snapshot"] = {
                    "build_id": manifest.get("build_id", ""),
                    "build_config_hash": manifest.get("build_config_hash", ""),
                    "retrieval_config_hash": manifest.get("retrieval_config_hash", ""),
                    "manifest_path": self._artifact_relative_path(
                        self._memory_snapshot_manifest_path()
                    ),
                    "snapshot_path": self._artifact_relative_path(snapshot_path),
                    "integrity_hash": snapshot.get("integrity_hash", ""),
                    "embedding_sha256": snapshot.get("embedding_sha256", ""),
                }
        if answer_request_id:
            answer_reference: Dict[str, Any] = {
                "transport": "batch" if answer_batch_client is not None else "realtime",
            }
            answer_reference[
                "request_id" if answer_batch_client is not None else "correlation_id"
            ] = answer_request_id
            relative_path = self._artifact_relative_path(
                getattr(answer_batch_client, "manifest_path", None)
            )
            if relative_path is not None:
                answer_reference["manifest_path"] = relative_path
            if answer_messages is not None:
                answer_reference["prompt_sha256"] = self._artifact_content_hash(
                    answer_messages
                )
            references["answer"] = answer_reference
        return references

    def _batch_artifact_reference(
        self,
        batch_client: Any,
        request_id: str,
    ) -> Dict[str, Any]:
        """Describe a persisted batch request without copying its raw payload."""
        reference: Dict[str, Any] = {
            "transport": "batch",
            "request_id": request_id,
        }
        relative_path = self._artifact_relative_path(
            getattr(batch_client, "manifest_path", None)
        )
        if relative_path is not None:
            reference["manifest_path"] = relative_path
        return reference

    @staticmethod
    def _artifact_content_hash(value: Any) -> str:
        """Return a stable digest for a JSON artifact fragment."""
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _batch_response_usage(response: Any, *, transport: str) -> Dict[str, Any]:
        """Normalize provider usage without treating unavailable timing as zero."""
        return {
            "transport": transport,
            "input_tokens": int(getattr(response, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(response, "output_tokens", 0) or 0),
            "visible_output_tokens": int(
                getattr(response, "visible_output_tokens", 0) or 0
            ),
            "thinking_tokens": int(getattr(response, "thinking_tokens", 0) or 0),
            "duration_seconds": getattr(response, "duration_seconds", None),
        }

    @staticmethod
    def _usage_total(*usage_values: Dict[str, Any]) -> Dict[str, Any]:
        """Merge tracker usage objects while preserving derived averages."""
        total = TokenUsage()
        for usage in usage_values:
            total.merge(TokenUsage.from_dict(usage))
        return total.to_dict()

    @staticmethod
    def _model_usage_identity(model_config: Any, *, transport: str) -> Dict[str, Any]:
        """Describe the configured model used by a stage, never a guessed route."""
        return {
            "provider": str(getattr(model_config, "provider", "") or ""),
            "model": str(getattr(model_config, "name", "") or ""),
            "transport": transport,
            "service_tier": getattr(model_config, "openrouter_service_tier", None),
        }

    def _batch_manifest_stage_summaries(self, batch_client: Any) -> List[Dict[str, Any]]:
        """Expose compact batch lifecycle and provider usage from a raw manifest."""
        manifest_path = getattr(batch_client, "manifest_path", None)
        if manifest_path is None:
            return []
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        summaries = []
        for stage_name, stage in (manifest.get("jobs") or {}).items():
            if not isinstance(stage, dict):
                continue
            responses = stage.get("responses") or {}
            response_values = list(responses.values()) if isinstance(responses, dict) else []
            submitted_at = stage.get("submitted_at")
            running_at = stage.get("running_at")
            completed_at = stage.get("completed_at")
            queue_inclusive_elapsed_seconds = None
            overall_latency_seconds = None
            queue_wait_seconds = None
            if isinstance(submitted_at, str) and isinstance(completed_at, str):
                try:
                    queue_inclusive_elapsed_seconds = (
                        datetime.fromisoformat(completed_at)
                        - datetime.fromisoformat(submitted_at)
                    ).total_seconds()
                except ValueError:
                    pass
            if isinstance(running_at, str) and isinstance(completed_at, str):
                try:
                    overall_latency_seconds = (
                        datetime.fromisoformat(completed_at)
                        - datetime.fromisoformat(running_at)
                    ).total_seconds()
                except ValueError:
                    pass
            if isinstance(submitted_at, str) and isinstance(running_at, str):
                try:
                    queue_wait_seconds = (
                        datetime.fromisoformat(running_at)
                        - datetime.fromisoformat(submitted_at)
                    ).total_seconds()
                except ValueError:
                    pass
            fallback_reason = stage.get("fallback_reason")
            summaries.append({
                "stage": stage_name,
                "manifest_path": self._artifact_relative_path(Path(manifest_path)),
                "transport": "realtime_fallback" if fallback_reason else "batch",
                "provider": manifest.get("provider") or (
                    "openrouter" if manifest.get("project") == "openrouter" else "vertex"
                ),
                "model": manifest.get("model", ""),
                "state": stage.get("state", ""),
                "request_count": len(stage.get("requests") or []),
                "response_count": len(response_values),
                "successful_response_count": sum(
                    not item.get("status") for item in response_values if isinstance(item, dict)
                ),
                "failed_response_count": sum(
                    bool(item.get("status")) for item in response_values if isinstance(item, dict)
                ),
                "retry_count": len(stage.get("retries") or []),
                "submitted_at": submitted_at,
                "running_at": running_at,
                "completed_at": completed_at,
                "overall_latency_seconds": overall_latency_seconds,
                "queue_wait_seconds": queue_wait_seconds,
                "queue_inclusive_elapsed_seconds": queue_inclusive_elapsed_seconds,
                # Retain the legacy name for readers that interpret it as
                # submission-to-completion rather than execution-only time.
                "remote_elapsed_seconds": queue_inclusive_elapsed_seconds,
                "token_usage": {
                    field: sum(
                        int(item.get(field, 0) or 0)
                        for item in response_values if isinstance(item, dict)
                    )
                    for field in (
                        "input_tokens", "output_tokens", "visible_output_tokens",
                        "thinking_tokens",
                    )
                },
                "fallback_reason": fallback_reason,
            })
        return summaries

    def _stage_usage_report(self, llm_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Publish stage attribution, batch lifecycle, and explicit cost status."""
        operations = llm_usage.get("operations") or {}
        query_operations = operations.get("query", {})
        judge_operations = operations.get("judge", {})
        query_total = llm_usage.get("query_phase", {})
        judge_total = llm_usage.get("judge_phase", {})
        answer_batch = query_operations.get("query.answer_batch", {})
        answer_realtime = query_operations.get("query.answer_realtime", {})
        planner_controller = query_operations.get("event_state.plan_or_answer", {})
        event_state_final = query_operations.get("event_state.final_answer", {})
        # Read legacy operation names so older in-memory snapshots remain reportable.
        judge_batch = judge_operations.get("judge.batch") or query_operations.get(
            "query.judge_batch", {}
        )
        judge_realtime = judge_operations.get("judge.realtime") or query_operations.get(
            "query.judge_realtime", {}
        )
        retrieval = query_operations.get("query.retrieval_preparation", {})
        attributed_query = self._usage_total(
            answer_batch, answer_realtime, retrieval, planner_controller, event_state_final
        )
        attributed_judge = self._usage_total(judge_batch, judge_realtime)
        unattributed_query = TokenUsage.from_dict(query_total).subtract(
            TokenUsage.from_dict(attributed_query)
        ).to_dict()
        unattributed_judge = TokenUsage.from_dict(judge_total).subtract(
            TokenUsage.from_dict(attributed_judge)
        ).to_dict()
        api_config = get_api_config()
        answer_model = self._model_usage_identity(
            self.method_config.model,
            transport="batch_or_realtime",
        )
        judge_model = {
            "provider": api_config.get_judge_provider(),
            "model": api_config.get_judge_model(),
            "transport": "batch_or_realtime",
            "service_tier": None,
        }
        cost_unavailable = {
            "available": False,
            "currency": "USD",
            "reason": (
                "No versioned provider price schedule is configured; token totals "
                "are recorded without an estimated monetary cost."
            ),
        }
        batch_stages = (
            self._batch_manifest_stage_summaries(getattr(self, "_batch_client", None))
            + self._batch_manifest_stage_summaries(
                getattr(self, "_judge_batch_client", None)
            )
        )
        def batch_latency_for(*stage_prefixes: str) -> Optional[float]:
            values = [
                stage.get("overall_latency_seconds")
                for stage in batch_stages
                if stage.get("transport") == "batch"
                and str(stage.get("stage", "")).startswith(stage_prefixes)
                and stage.get("overall_latency_seconds") is not None
            ]
            return sum(values) if values else None

        answer_usage = self._usage_total(
            answer_batch, answer_realtime, event_state_final
        )
        answer_usage["batch_overall_latency_seconds"] = batch_latency_for("query")
        judge_usage = self._usage_total(judge_batch, judge_realtime)
        judge_usage["batch_overall_latency_seconds"] = batch_latency_for(
            "judge", "rejudge"
        )
        return {
            "schema_version": 1,
            "memory_build": {
                "usage": llm_usage.get("memorize_phase", {}),
                "models": [],
                "cost": cost_unavailable,
            },
            "retrieval_preparation": {
                "usage": retrieval,
                "models": [answer_model],
                "cost": cost_unavailable,
            },
            "answer": {
                "usage": answer_usage,
                "batch_overall_latency_seconds": answer_usage[
                    "batch_overall_latency_seconds"
                ],
                "models": [answer_model],
                "cost": cost_unavailable,
            },
            "planner_controller": {
                "usage": planner_controller,
                "models": [answer_model],
                "cost": cost_unavailable,
            },
            "judge": {
                "usage": judge_usage,
                "batch_overall_latency_seconds": judge_usage[
                    "batch_overall_latency_seconds"
                ],
                "models": [judge_model],
                "cost": cost_unavailable,
            },
            "unattributed_query_usage": unattributed_query,
            "unattributed_judge_usage": unattributed_judge,
            "batch_stages": batch_stages,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return file_sha256(path)

    def _write_memory_snapshot_manifest(self) -> None:
        if self._memory_snapshot_manifest is None:
            return
        path = self._memory_snapshot_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            dump_json_artifact(self._memory_snapshot_manifest, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        self._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Make a completed atomic replacement durable on supported filesystems."""
        fsync_directory(path)

    @staticmethod
    def _new_memory_run_name() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _memory_run_candidates(self) -> List[Path]:
        if getattr(self, "run_scoped_output", False):
            candidates = []
            experiment_dir = self._run_scoped_experiment_dir()
            for run_dir in sorted(
                [path for path in experiment_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            ):
                memory_dir = run_dir / "memory"
                if memory_dir != self._memory_snapshot_root():
                    candidates.append(memory_dir)

            legacy_root = self._legacy_memory_snapshot_root()
            if legacy_root.exists():
                candidates.extend(sorted(
                    [path for path in legacy_root.iterdir() if path.is_dir()],
                    key=lambda path: path.name,
                    reverse=True,
                ))
                if (legacy_root / "manifest.json").exists():
                    candidates.append(legacy_root)
            return candidates

        root = self._memory_snapshot_root()
        if not root.exists():
            return []
        candidates = sorted(
            [path for path in root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        # Runs produced before timestamped directories stored their manifest
        # and unit snapshots directly in the memory root.
        if (root / "manifest.json").exists():
            candidates.append(root)
        return candidates

    def _saved_memory_source_candidate(self) -> Optional[Path]:
        if not getattr(self, "run_scoped_output", False):
            return None
        source_path = self.output_dir / "memory_source.json"
        if not source_path.exists():
            return None
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read saved memory source {source_path}: {truncate_error_message(exc)}"
            ) from exc
        manifest_value = payload.get("manifest_path")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError(f"Saved memory source is invalid: {source_path}")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = self.output_dir / manifest_path
        return manifest_path.resolve().parent

    def _explicit_memory_run_candidates(self, requested_run: str) -> List[Path]:
        if requested_run == "legacy":
            return [self._legacy_memory_snapshot_root()]
        if Path(requested_run).name != requested_run:
            raise ValueError("--memory-run must be a run timestamp directory name")
        if getattr(self, "run_scoped_output", False):
            candidates = []
            memory_source_run_dir = getattr(self, "memory_source_run_dir", None)
            if memory_source_run_dir is not None:
                candidates.append(memory_source_run_dir / "memory")
            candidates.extend([
                self._run_scoped_experiment_dir() / requested_run / "memory",
                self._legacy_memory_snapshot_root() / requested_run,
            ])
            return list(dict.fromkeys(candidates))
        return [self._memory_snapshot_root() / requested_run]

    def _write_memory_source(self, run_dir: Path, manifest: Dict[str, Any]) -> None:
        if not getattr(self, "run_scoped_output", False):
            return
        manifest_path = run_dir / "manifest.json"
        relative_manifest = os.path.relpath(manifest_path, self.output_dir)
        source_run_id = (
            run_dir.parent.name
            if (
                run_dir.name == "memory"
                and run_dir.parent.parent == self._run_scoped_experiment_dir()
            )
            else f"legacy:{run_dir.name}"
        )
        payload = {
            "format": "medmemorybench.memory_source",
            "version": 2,
            "selected_at": datetime.now().isoformat(),
            "selection": (
                "explicit"
                if getattr(
                    self,
                    "memory_run_explicit",
                    bool(getattr(self, "memory_run", None)),
                )
                else "newest_compatible"
            ),
            "source_run_id": source_run_id,
            "manifest_path": relative_manifest,
            # The manifest owns build telemetry; this is only the identity we
            # selected so a resumed query run can audit the same source.
            "memory_identity": {
                "build_id": manifest.get("build_id"),
                "status": manifest.get("status"),
                "config_hash": manifest.get("config_hash"),
                "build_config_hash": derive_legacy_build_config_hash(
                    manifest, manifest_path
                ),
            },
        }
        path = self.output_dir / "memory_source.json"
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            dump_json_artifact(payload, handle)
        temporary_path.replace(path)

    def _select_memory_snapshot_run(
        self,
        *,
        require_complete: bool,
        resume_current: bool = False,
        allow_building: bool = False,
    ) -> Path:
        requested_run = getattr(self, "memory_run", None)
        saved_source = (
            self._saved_memory_source_candidate()
            if require_complete and not requested_run
            else None
        )
        if resume_current and getattr(self, "run_scoped_output", False):
            candidates = [self._memory_snapshot_root()]
        elif saved_source is not None:
            candidates = [saved_source]
        elif requested_run:
            candidates = self._explicit_memory_run_candidates(requested_run)
        else:
            candidates = self._memory_run_candidates()

        for run_dir in candidates:
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            accepted_statuses = {"complete"} if require_complete else {"building"}
            if allow_building:
                accepted_statuses.add("building")
            if (
                manifest.get("format") != "medmemorybench.memory_manifest"
                or manifest.get("version") != 1
                or not (
                    is_manifest_query_compatible(
                        manifest,
                        self.method_config,
                        self.dataset_config,
                        manifest_path,
                    )
                    if self.execution_stage == "query"
                    else is_manifest_build_compatible(
                        manifest, self._batch_config_hash(), manifest_path
                    )
                )
                or manifest.get("method_name") != self.method_config.method_name
                or manifest.get("status") not in accepted_statuses
            ):
                continue
            self._memory_snapshot_run_dir = run_dir
            self._memory_snapshot_manifest = manifest
            if require_complete or allow_building:
                self._write_memory_source(run_dir, manifest)
            return run_dir

        selection = f" '{requested_run}'" if requested_run else ""
        status = "complete" if require_complete else "in-progress"
        raise FileNotFoundError(
            f"No compatible {status} {self._memory_method_label()} memory run{selection} for experiment "
            f"{self._run_scoped_experiment_dir() if getattr(self, 'run_scoped_output', False) else self._memory_snapshot_root()}"
        )

    def _copy_append_source_snapshots(self) -> None:
        """Materialize source snapshots into the derived append run."""
        if self._append_source_manifest is None or self.memory_source_run_dir is None:
            return
        units = self._get_evaluation_units()
        unit_lookup = {(unit.context_id, unit.unit_id): unit for unit in units}
        source_memory_dir = self.memory_source_run_dir / "memory"
        snapshots = self._memory_snapshot_manifest["snapshots"]
        for key, record in sorted(
            self._append_source_snapshot_records.items(),
            key=lambda item: unit_lookup[item[0]].unit_id,
        ):
            unit = unit_lookup[key]
            source_name = record.get("path")
            if not isinstance(source_name, str) or Path(source_name).name != source_name:
                raise ValueError(f"Append source snapshot path is invalid for unit {key}")
            source_path = source_memory_dir / source_name
            source_payload = self._read_snapshot_payload(
                source_path,
                self._append_source_manifest,
                unit,
                require_current_config=True,
            )
            child_payload = copy.deepcopy(source_payload)
            child_payload["build_id"] = self._memory_snapshot_manifest["build_id"]
            child_payload["append_source"] = {
                "run_id": self.memory_source_run_dir.name,
                "build_id": self._append_source_manifest.get("build_id"),
            }
            target_path = self._memory_snapshot_path(unit)
            child_payload = self._write_snapshot_payload(child_payload, target_path)
            snapshots[:] = [
                item
                for item in snapshots
                if not (
                    item.get("context_id") == unit.context_id
                    and item.get("unit_id") == unit.unit_id
                )
            ]
            snapshots.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "path": target_path.name,
                "integrity_hash": child_payload.get("integrity_hash", ""),
                "embedding_sha256": self._snapshot_embedding_sha256(child_payload),
                "source": self.memory_source_run_dir.name,
                "session_count": len(child_payload.get("session_ids", [])),
                "memory_build_time": child_payload.get("memory_build_time", 0.0),
                "memory_build_metrics": copy.deepcopy(
                    child_payload.get("memory_build_metrics") or {}
                ),
                "memory_size": copy.deepcopy(child_payload.get("memory_size")),
            })
        snapshots.sort(key=lambda item: item["unit_id"])
        self._write_memory_snapshot_manifest()

    def _start_append_memory_snapshot_manifest(self) -> None:
        """Create a derived memory run seeded with the source prefix."""
        self._prepare_append_plan()
        units = self._get_evaluation_units()
        if self._append_target_index is None:
            raise RuntimeError("Append target was not resolved")
        selected_units = units[: self._append_target_index + 1]
        run_name = self.output_dir.name if self.run_scoped_output else self._new_memory_run_name()
        run_dir = self._memory_snapshot_root()
        run_dir.mkdir(parents=True, exist_ok=False)
        self._memory_snapshot_run_dir = run_dir
        source_manifest = self._append_source_manifest or {}
        self._memory_snapshot_manifest = {
            "format": "medmemorybench.memory_manifest",
            "version": 1,
            "build_id": str(uuid.uuid4()),
            "run_name": run_name,
            "status": "building",
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": self._batch_config_hash(),
            "build_config_hash": self._batch_config_hash(),
            "retrieval_config_hash": compute_query_config_hash(
                self.method_config, self.dataset_config
            ),
            "retrieval_compatibility_hash": compute_memory_query_compatibility_hash(
                self.method_config, self.dataset_config
            ),
            "feature_configuration": self._amem_feature_configuration(),
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "unit_ids": [unit.unit_id for unit in selected_units],
            "dataset_unit_ids": [unit.unit_id for unit in units],
            "append": {
                "source_run_id": self.memory_source_run_dir.name,
                "source_build_id": source_manifest.get("build_id"),
                "target_persona": getattr(self, "append_persona", None),
                "target_unit": getattr(self, "append_unit", None),
                "target_index": self._append_target_index,
                "source_snapshot_count": len(self._append_source_snapshot_records),
            },
            "snapshots": [],
        }
        self._write_memory_snapshot_manifest()
        self._copy_append_source_snapshots()
        self._log(f"Created append {self._memory_method_label()} memory run: {run_dir}")

    def _start_memory_snapshot_manifest(self, *, resume_existing: bool = False) -> None:
        units = self._get_evaluation_units()
        # A process can stop after publishing the memory manifest but before the
        # checkpoint is created. Reuse that current run on the next --resume.
        if (
            not resume_existing
            and getattr(self, "resume", False)
            and getattr(self, "run_scoped_output", False)
            and (self._memory_snapshot_root() / "manifest.json").exists()
        ):
            resume_existing = True
        if resume_existing:
            run_dir = self._select_memory_snapshot_run(
                require_complete=False,
                resume_current=True,
            )
            expected_unit_ids = [unit.unit_id for unit in units]
            stored_unit_ids = self._memory_snapshot_manifest.get("unit_ids", [])
            if getattr(self, "append", False):
                if not self._memory_snapshot_manifest.get("append"):
                    raise ValueError(
                        f"{self._memory_method_label()} resume target is not an append run: {run_dir}"
                    )
                if self._append_target_index is None:
                    self._prepare_append_plan()
                if stored_unit_ids != expected_unit_ids[: self._append_target_index + 1]:
                    raise ValueError(
                        f"{self._memory_method_label()} append run does not match current evaluation units: {run_dir}"
                    )
                # Reconcile source snapshots after an interruption during the
                # initial copy, before building any missing evaluation units.
                self._copy_append_source_snapshots()
            elif stored_unit_ids != expected_unit_ids:
                raise ValueError(
                    f"{self._memory_method_label()} memory run does not match current evaluation units: {run_dir}"
                )
            run_label = run_dir.name if run_dir != self._memory_snapshot_root() else "legacy"
            self._log(f"Resuming {self._memory_method_label()} memory run: {run_label}")
            return

        if getattr(self, "append", False):
            self._start_append_memory_snapshot_manifest()
            return

        if getattr(self, "run_scoped_output", False):
            run_name = self.output_dir.name
            run_dir = self._memory_snapshot_root()
        else:
            run_name = self._new_memory_run_name()
            run_dir = self._memory_snapshot_root() / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        self._memory_snapshot_run_dir = run_dir
        self._memory_snapshot_manifest = {
            "format": "medmemorybench.memory_manifest",
            "version": 1,
            "build_id": str(uuid.uuid4()),
            "run_name": run_name,
            "status": "building",
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": self._batch_config_hash(),
            "build_config_hash": self._batch_config_hash(),
            "retrieval_config_hash": compute_query_config_hash(
                self.method_config, self.dataset_config
            ),
            "retrieval_compatibility_hash": compute_memory_query_compatibility_hash(
                self.method_config, self.dataset_config
            ),
            "feature_configuration": self._amem_feature_configuration(),
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "unit_ids": [unit.unit_id for unit in units],
            "dataset_unit_ids": [unit.unit_id for unit in units],
            "snapshots": [],
        }
        self._write_memory_snapshot_manifest()
        self._log(f"Created {self._memory_method_label()} memory run: {run_dir}")

    def _load_memory_snapshot_manifest(self) -> None:
        allow_building = (
            self.method_config.method_name.lower() == "event_state"
            and bool(getattr(self, "memory_run", None))
        )
        run_dir = self._select_memory_snapshot_run(
            require_complete=True,
            allow_building=allow_building,
        )
        manifest = self._memory_snapshot_manifest
        expected_unit_ids = [unit.unit_id for unit in self._get_evaluation_units()]
        stored_unit_ids = manifest.get("unit_ids", [])
        if manifest.get("append"):
            if not set(stored_unit_ids).issubset(set(expected_unit_ids)):
                raise ValueError(
                    f"{self._memory_method_label()} append manifest contains unknown evaluation units: {run_dir}"
                )
        elif stored_unit_ids != expected_unit_ids:
            raise ValueError(
                f"{self._memory_method_label()} memory manifest does not match current evaluation units: {run_dir}"
            )
        snapshot_unit_ids = sorted(item.get("unit_id") for item in manifest.get("snapshots", []))
        if snapshot_unit_ids != sorted(stored_unit_ids):
            raise ValueError(
                f"{self._memory_method_label()} memory manifest snapshot list is incomplete: {run_dir}"
            )
        self._memory_unit_ids = set(stored_unit_ids)
        run_label = run_dir.name if run_dir != self._memory_snapshot_root() else "legacy"
        self._log(f"Using {self._memory_method_label()} memory run: {run_label}")

    def _complete_memory_snapshot_manifest(self) -> None:
        if self._memory_snapshot_manifest is None:
            return
        expected = self._memory_snapshot_manifest.get("unit_ids", [])
        completed = sorted(item["unit_id"] for item in self._memory_snapshot_manifest["snapshots"])
        expected = sorted(expected)
        if completed != expected:
            raise RuntimeError(
                f"{self._memory_method_label()} snapshot build is incomplete: expected units {expected}, got {completed}"
            )
        snapshot_logs = [
            {
                "unit_id": item.get("unit_id"),
                "context_id": item.get("context_id"),
                "session_count": item.get("session_count", 0),
                "total_time": item.get("memory_build_time", 0.0),
                "build_metrics": item.get("memory_build_metrics") or {},
                "memory_size": item.get("memory_size"),
                "restored_from_snapshot": bool(item.get("source")),
            }
            for item in self._memory_snapshot_manifest.get("snapshots", [])
        ]
        self._memory_snapshot_manifest["build_metrics"] = self._build_metrics_report(
            snapshot_logs
        )
        self._memory_snapshot_manifest["memory_size"] = copy.deepcopy(
            self._memory_snapshot_manifest["build_metrics"]["memory_size"]
        )
        self._memory_snapshot_manifest["status"] = "complete"
        self._memory_snapshot_manifest["completed_at"] = datetime.now().isoformat()
        self._write_memory_snapshot_manifest()

    @staticmethod
    def _snapshot_integrity_hash(payload: Dict[str, Any]) -> str:
        hash_payload = dict(payload)
        hash_payload.pop("integrity_hash", None)
        content = json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _snapshot_embedding_sha256(payload: Dict[str, Any]) -> str:
        """Return the published embedding sidecar digest, when present."""
        embeddings = (
            payload.get("memory_state", {})
            .get("system_state", {})
            .get("retriever", {})
            .get("embeddings")
        )
        return str(embeddings.get("sha256", "")) if isinstance(embeddings, dict) else ""

    @staticmethod
    def _measure_snapshot_memory_size(
        payload: Dict[str, Any],
        snapshot_path: Path,
    ) -> Dict[str, Any]:
        """Measure the serialized retrieval state without snapshot metadata."""
        memory_state = payload.get("memory_state") or {}
        json_bytes = len(json.dumps(
            memory_state,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"))
        system_state = memory_state.get("system_state") or {}
        embedding_bytes = sum(
            artifact_path.stat().st_size
            for artifact_path in memory_state_embedding_artifact_paths(
                memory_state, snapshot_path
            )
            if artifact_path.is_file()
        )
        memory_entries = system_state.get("memories") or []
        memory_chunks = (memory_state.get("agent_state") or {}).get(
            "memory_chunks"
        ) or []
        total_bytes = json_bytes + embedding_bytes
        return {
            "measurement": "serialized_memory_state",
            "bytes": total_bytes,
            "mib": round(total_bytes / (1024 ** 2), 6),
            "json_bytes": json_bytes,
            "embedding_bytes": embedding_bytes,
            "memory_entry_count": len(memory_entries),
            "memory_chunk_count": len(memory_chunks),
        }

    def _write_snapshot_payload(
        self,
        payload: Dict[str, Any],
        path: Path,
    ) -> Dict[str, Any]:
        """Publish a snapshot without invalidating the previous generation."""
        payload = copy.deepcopy(payload)
        memory_state = payload.get("memory_state")
        if not isinstance(memory_state, dict):
            raise ValueError("Memory snapshot state is invalid")
        publish_memory_state_embedding_artifacts(memory_state, path)

        payload["memory_size"] = self._measure_snapshot_memory_size(payload, path)
        payload["integrity_hash"] = self._snapshot_integrity_hash(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            dump_json_artifact(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        self._fsync_directory(path.parent)
        cleanup_memory_state_embedding_artifacts(memory_state, path)
        return payload

    def _read_snapshot_payload(
        self,
        path: Path,
        manifest: Dict[str, Any],
        unit: EvaluationUnit,
        *,
        require_current_config: bool = True,
    ) -> Dict[str, Any]:
        """Read and validate a snapshot from any memory-run directory."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read {self._memory_method_label()} snapshot {path}: {truncate_error_message(exc)}"
            ) from exc
        if (
            payload.get("format") != "medmemorybench.memory_snapshot"
            or payload.get("version") != 1
            or payload.get("context_id") != unit.context_id
            or payload.get("unit_id") != unit.unit_id
            or payload.get("build_id") != manifest.get("build_id")
        ):
            raise ValueError(f"{self._memory_method_label()} snapshot metadata does not match unit {unit.unit_id}: {path}")
        expected_hash = self._snapshot_integrity_hash(payload)
        if payload.get("integrity_hash") != expected_hash:
            raise ValueError(f"{self._memory_method_label()} snapshot integrity check failed: {path}")
        if require_current_config:
            manifest_path = path.parent / "manifest.json"
            if self.execution_stage == "query":
                config_compatible = is_manifest_query_compatible(
                    manifest,
                    self.method_config,
                    self.dataset_config,
                    manifest_path,
                )
                mismatch_message = "query configuration"
            else:
                config_compatible = is_manifest_build_compatible(
                    manifest,
                    self._batch_config_hash(),
                    manifest_path,
                )
                mismatch_message = "build configuration"
            if not config_compatible:
                raise ValueError(
                    f"{self._memory_method_label()} snapshot {mismatch_message} does not match: {path}"
                )
        payload_build_hash = str(payload.get("build_config_hash") or "")
        if (
            require_current_config
            and self.execution_stage != "query"
            and payload_build_hash
            and payload_build_hash != self._batch_config_hash()
        ):
            raise ValueError(
                f"{self._memory_method_label()} snapshot payload build configuration does not match: {path}"
            )
        payload_query_hash = str(payload.get("retrieval_compatibility_hash") or "")
        if (
            require_current_config
            and self.execution_stage == "query"
            and payload_query_hash
            and payload_query_hash
            != compute_memory_query_compatibility_hash(
                self.method_config, self.dataset_config
            )
        ):
            raise ValueError(
                f"{self._memory_method_label()} snapshot payload query configuration does not match: {path}"
            )
        payload["memory_size"] = self._measure_snapshot_memory_size(payload, path)
        memory_state = payload.get("memory_state")
        if not isinstance(memory_state, dict):
            raise ValueError(f"{self._memory_method_label()} snapshot state is invalid: {path}")
        load_memory_state_embedding_artifacts(memory_state, path)
        if (
            memory_state.get("method") == "event_state"
            and memory_state.get("schema_version") == EventStateStore.SCHEMA_VERSION
        ):
            # Validate Event-State's semantic ID mappings before append/resume
            # can copy an otherwise syntactically valid sidecar set.
            EventStateStore.from_export(memory_state)
        return payload

    def _write_memory_snapshot(
        self,
        unit: EvaluationUnit,
        memory_build_time: float = 0.0,
        memory_build_metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist the exact post-build state, then atomically publish it."""
        path = self._memory_snapshot_path(unit)
        state = self.agent_manager.export_memory_state(context_id=unit.context_id)
        binary_artifact_exporter = getattr(
            self.agent_manager, "export_memory_binary_artifacts", None
        )
        if callable(binary_artifact_exporter):
            binary_artifacts = binary_artifact_exporter(context_id=unit.context_id)
            if binary_artifacts:
                state["embedding_artifacts"] = binary_artifacts
        payload = {
            "format": "medmemorybench.memory_snapshot",
            "version": 1,
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": self._batch_config_hash(),
            "build_config_hash": self._batch_config_hash(),
            "retrieval_config_hash": compute_query_config_hash(
                self.method_config, self.dataset_config
            ),
            "retrieval_compatibility_hash": compute_memory_query_compatibility_hash(
                self.method_config, self.dataset_config
            ),
            "build_id": self._memory_snapshot_manifest["build_id"],
            "context_id": unit.context_id,
            "unit_id": unit.unit_id,
            "evaluation_session_id": unit.metadata.get("eval_session_id"),
            "source_identity_schema_version": 1,
            "source_uids": [
                self._method_source_uid(session)
                for session in unit.sessions_to_inject
            ],
            # Retained for readers of version-1 snapshots; values are source UIDs.
            "session_ids": [
                self._method_source_uid(session)
                for session in unit.sessions_to_inject
            ],
            "memory_build_time": memory_build_time,
            "memory_build_metrics": copy.deepcopy(memory_build_metrics or {}),
            "feature_configuration": self._amem_feature_configuration(),
            "created_at": datetime.now().isoformat(),
            "memory_state": state,
        }
        payload = self._write_snapshot_payload(payload, path)
        memory_size = copy.deepcopy(payload.get("memory_size"))
        snapshot_record = {
            "unit_id": unit.unit_id,
            "context_id": unit.context_id,
            "path": path.name,
            "integrity_hash": payload["integrity_hash"],
            "embedding_sha256": self._snapshot_embedding_sha256(payload),
            "session_count": len(unit.sessions_to_inject),
            "memory_build_time": memory_build_time,
            "memory_build_metrics": copy.deepcopy(memory_build_metrics or {}),
            "memory_size": memory_size,
        }
        snapshots = self._memory_snapshot_manifest["snapshots"]
        snapshots[:] = [item for item in snapshots if item["unit_id"] != unit.unit_id]
        snapshots.append(snapshot_record)
        snapshots.sort(key=lambda item: item["unit_id"])
        for log in reversed(getattr(self, "_memory_build_logs", [])):
            if (
                log.get("unit_id") == unit.unit_id
                and log.get("context_id") == unit.context_id
            ):
                log["memory_size"] = copy.deepcopy(memory_size)
                break
        self._write_memory_snapshot_manifest()
        return path

    def _read_memory_snapshot(
        self,
        unit: EvaluationUnit,
        *,
        require_current_config: bool = True,
    ) -> Optional[Dict[str, Any]]:
        path = self._memory_snapshot_path(unit)
        if not path.exists():
            return None
        if self._memory_snapshot_manifest is None:
            raise RuntimeError(
                f"{self._memory_method_label()} memory snapshot manifest has not been loaded"
            )
        payload = self._read_snapshot_payload(
            path,
            self._memory_snapshot_manifest,
            unit,
            require_current_config=require_current_config,
        )
        return payload

    def _restore_memory_snapshot(self, unit: EvaluationUnit) -> Path:
        """Reload the just-written state before retrieval starts."""
        payload = self._read_memory_snapshot(unit)
        if payload is None:
            raise RuntimeError(
                f"{self._memory_method_label()} snapshot is unavailable for unit {unit.unit_id}"
            )
        self.agent_manager.import_memory_state(
            payload["memory_state"],
            context_id=unit.context_id,
        )
        return self._memory_snapshot_path(unit)

    def _save_and_restore_unit_memory(
        self,
        unit: EvaluationUnit,
        memory_build_time: float = 0.0,
        memory_build_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if (
            self.dry_run
            or getattr(self, "_memory_snapshot_manifest", None) is None
            or not self._supports_memory_snapshots()
        ):
            return None
        self._log("    --- Memory Snapshot Start ---")
        path = self._write_memory_snapshot(
            unit,
            memory_build_time=memory_build_time,
            memory_build_metrics=memory_build_metrics,
        )
        self._restore_memory_snapshot(unit)
        self._log(f"      Saved and restored retrieval state: {path}")
        self._log("    --- Memory Snapshot Done ---")
        return path

    def _restore_completed_unit_for_resume(
        self,
        unit: EvaluationUnit,
    ) -> Optional[Dict[str, Any]]:
        if (
            not (
                getattr(self, "resume", False)
                or getattr(self, "force_resume", False)
            )
            or self.dry_run
            or getattr(self, "_memory_snapshot_manifest", None) is None
            or not self._supports_memory_snapshots()
        ):
            return None
        payload = self._read_memory_snapshot(unit)
        if payload is None:
            return None
        self.agent_manager.import_memory_state(
            payload["memory_state"],
            context_id=unit.context_id,
        )
        self._log(
            f"      [Resume] Restored completed unit state from "
            f"{self._memory_snapshot_path(unit)}"
        )
        return payload

    def _batch_manifest_config_hash(self) -> str:
        """Namespace manifests by checkpoint so only --resume reuses a run."""
        config_hash = self._batch_config_hash()
        checkpoint_manager = getattr(self, "_checkpoint_manager", None)
        scope = (
            checkpoint_manager.get_batch_manifest_scope()
            if checkpoint_manager
            else ""
        )
        # Older checkpoints have no scope and retain their legacy manifest path.
        return f"{config_hash}{scope.replace('-', '')}" if scope else config_hash

    def _batch_manifest_path(self, stem: str, model: str) -> Path:
        """Keep manifests isolated when different methods share one output directory."""
        return scoped_manifest_path(
            self.output_dir / "batch",
            stem,
            model=model,
            config_hash=self._batch_manifest_config_hash(),
        )

    def _deferred_judge_state_paths(self) -> List[Path]:
        batch_dir = self.output_dir / "batch"
        return sorted(batch_dir.glob(
            f"medmemorybench_deferred_judges-*-{self._judge_batch_config_hash()}.json"
        ))

    def _save_deferred_judges(self, judge_client) -> None:
        """Persist judge inputs before submitting so resume never regenerates them."""
        path = scoped_manifest_path(
            self.output_dir / "batch",
            "medmemorybench_deferred_judges",
            model=judge_client.model,
            config_hash=self._judge_batch_config_hash(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config_hash": self._judge_batch_config_hash(),
            "judge_model": judge_client.model,
            "items": self._deferred_judges,
        }
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            dump_json_artifact(payload, handle)
        temporary_path.replace(path)

    def _load_deferred_judges(self) -> None:
        """Restore submitted-or-ready judge work before rebuilding evaluation units."""
        paths = self._deferred_judge_state_paths()
        if not paths:
            return
        if len(paths) != 1:
            raise VertexBatchError(
                "Found multiple deferred-judge state files for this configuration. "
                "Use a separate output directory to resume safely."
            )
        path = paths[0]
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise VertexBatchError(
                f"Cannot read deferred judge state {path}: {truncate_error_message(exc)}"
            ) from exc
        if (
            payload.get("version") != 1
            or payload.get("config_hash") != self._judge_batch_config_hash()
        ):
            raise VertexBatchError(f"Deferred judge state does not match this evaluator: {path}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise VertexBatchError(f"Deferred judge state is invalid: {path}")
        self._deferred_judges = items
        self._log(f"[Batch] Restored {len(items):,} deferred LLM-judge request(s) from {path}.")

    def _clear_deferred_judges(self) -> None:
        for path in self._deferred_judge_state_paths():
            path.unlink(missing_ok=True)
        self._deferred_judges = []

    def _is_deferred_judge_query(self, query_id: str) -> bool:
        return any(
            item.get("query_id") == query_id
            for item in getattr(self, "_deferred_judges", [])
        )

    def _flush_query_checkpoint(self) -> None:
        """Flush supported checkpoint managers without constraining test adapters."""
        checkpoint_manager = getattr(self, "_checkpoint_manager", None)
        flush = getattr(checkpoint_manager, "flush_query_progress", None)
        if callable(flush):
            flush(force=True)

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()

        get_usage_tracker().reset()

        self._init_dataset()

        if self.execution_stage != "all" and not self._supports_configured_memory_snapshots():
            raise ValueError(
                "Separated memory/query execution requires a snapshot-backed method"
            )
        if getattr(self, "append", False):
            if not self._supports_configured_memory_snapshots():
                raise ValueError("--append requires a snapshot-backed method")
            self._prepare_append_plan()

        resumed = False
        if self.resume and self._checkpoint_enabled:
            resumed = self._try_resume_from_checkpoint()

        if self.execution_stage == "query":
            self._load_memory_snapshot_manifest()
        elif not self.dry_run and (
            self.method_config.method_name.lower().startswith("amem")
            or self.method_config.method_name.lower() == "event_state"
            or self.execution_stage == "memory"
            or getattr(self, "append", False)
        ) and self._supports_configured_memory_snapshots():
            self._start_memory_snapshot_manifest(resume_existing=resumed)

        # Deferred judge inputs belong to the loaded checkpoint namespace.
        # Do not attach a fresh run to an older namespace when no checkpoint
        # was successfully resumed.
        if resumed and self.batch_api and not self.dry_run:
            self._load_deferred_judges()

        # Query-only runs still submit batch work, so they need the same unique
        # checkpoint-backed manifest scope as full runs. Without this, a fresh
        # query invocation falls back to the config-only legacy manifest name.
        if not resumed and self._checkpoint_enabled:
            self._create_new_checkpoint()

        try:
            try:
                self._run_evaluation_loop()
            finally:
                self._finish_query_progress()

            self._flush_query_checkpoint()

            if (
                self.execution_stage in {"all", "memory"}
                and not self.dry_run
                and (self.method_config.method_name.lower().startswith("amem") or self.method_config.method_name.lower() == "event_state")
            ):
                self._complete_memory_snapshot_manifest()

            if self._checkpoint_manager and self.execution_stage in {"all", "query"}:
                self._checkpoint_manager.mark_completed()
                self._checkpoint_manager.delete()
                self._log("Evaluation completed, checkpoint deleted")
            elif self._checkpoint_manager and self.execution_stage == "memory":
                self._checkpoint_manager.delete()
                self._log("Memory build completed, checkpoint deleted")
        except BaseException:
            try:
                self._flush_query_checkpoint()
            except Exception as checkpoint_error:
                self._log(
                    "Unable to save MedMemoryBench query checkpoint: "
                    f"{truncate_error_message(checkpoint_error)}",
                    level="WARNING",
                    terminal=False,
                )
            try:
                self._persist_memory_build_checkpoint(start_time)
            except Exception as checkpoint_error:
                self._log(
                    "Unable to save memory build checkpoint: "
                    f"{truncate_error_message(checkpoint_error)}",
                    level="WARNING",
                    terminal=False,
                )
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self) -> None:
        current_context_id = None

        units = self._get_evaluation_units()
        if getattr(self, "append", False) and self._append_target_index is not None:
            units = units[: self._append_target_index + 1]
        elif self.execution_stage == "query" and self._memory_unit_ids is not None:
            units = [unit for unit in units if unit.unit_id in self._memory_unit_ids]

        self._configure_query_progress(units)

        if self.execution_stage == "query" and self._query_worker_count() > 1 and len(units) > 1:
            try:
                self._start_query_progress()
                self._run_query_stage_parallel(units)
            finally:
                self._finish_query_progress()
            return

        for unit in units:
            persona_id = unit.context_id

            if self._checkpoint_manager and self._checkpoint_manager.is_persona_completed(persona_id):
                self._log(f"Skipping completed Persona: {persona_id}")
                continue

            if persona_id != current_context_id:
                if current_context_id is not None and self._checkpoint_manager and not self.batch_api:
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
                    preserve_progress = (
                        self._session_level_force_resume()
                        and persona_id == getattr(self, "_force_resume_persona_id", None)
                    )
                    self._checkpoint_manager.start_persona(
                        persona_id,
                        preserve_progress=preserve_progress,
                    )

            if getattr(self, "execution_stage", "all") == "query":
                snapshot_payload = self._read_memory_snapshot(unit)
                if snapshot_payload is None:
                    raise FileNotFoundError(
                        f"No compatible {self._memory_method_label()} snapshot for evaluation unit {unit.unit_id}: "
                        f"{self._memory_snapshot_path(unit)}"
                    )
                if not self._supports_memory_snapshots():
                    raise RuntimeError(
                        f"Method {self.method_config.method_name} does not support query-only snapshots"
                    )
                self.agent_manager.import_memory_state(
                    snapshot_payload["memory_state"],
                    context_id=unit.context_id,
                )
                self._record_source_build_metrics(unit, snapshot_payload)
                self._log(
                    f"\n  Evaluation Unit {unit.unit_id}: restored query-stage memory "
                    f"from {self._memory_snapshot_path(unit)}"
                )
                unit_results = self._evaluate_unit_queries(
                    unit,
                    total_memory_time=float(snapshot_payload.get("memory_build_time", 0.0)),
                )
            elif getattr(self, "append", False):
                unit_results = self._evaluate_append_unit(unit)
            else:
                unit_results = self._evaluate_unit_with_checkpoint(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, persona_id)

        if getattr(self, "execution_stage", "all") == "memory":
            self._finish_query_progress()
            return

        for item in self._complete_combined_batch_queries():
            result = item["result"]
            persona_id = item["persona_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, persona_id)
            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    result.query_id,
                    result.to_dict(),
                    persona_id=persona_id,
                )

        # Query-answer work is complete. Judge batches are a separate stage and
        # should not leave a completed query bar mounted while they run.
        self._finish_query_progress()

        for item in self._complete_deferred_judges():
            result = item["result"]
            persona_id = item["persona_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, persona_id)
            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    result.query_id,
                    result.to_dict(),
                    persona_id=persona_id,
                )

        if self._checkpoint_manager and self.batch_api:
            for persona_id in self.dataset.get_persona_ids():
                self._checkpoint_manager.complete_persona(persona_id)
        elif current_context_id is not None and self._checkpoint_manager:
            self._checkpoint_manager.complete_persona(current_context_id)
        self._finish_query_progress()

    def _run_query_stage_parallel(self, units: List[EvaluationUnit]) -> None:
        """Evaluate snapshot-backed units concurrently with isolated agents.

        Query-stage snapshots are immutable and independent, but ``AgentManager``
        keeps mutable active-memory state. Each task therefore owns a manager;
        shared report/checkpoint state is merged by this coordinator in dataset
        order after the workers finish.
        """
        if not hasattr(self, "_pending_batch_queries"):
            self._pending_batch_queries = []
        if not hasattr(self, "_api_failures"):
            self._api_failures = []
        pending_units: List[tuple[EvaluationUnit, Dict[str, Any]]] = []
        deferred_query_ids = {
            item.get("query_id") for item in self._deferred_judges
        }
        for unit in units:
            if (
                self._checkpoint_manager
                and self._checkpoint_manager.is_persona_completed(unit.context_id)
            ):
                self._log(f"Skipping completed Persona: {unit.context_id}")
                continue
            payload = self._read_memory_snapshot(unit)
            if payload is None:
                raise FileNotFoundError(
                    f"No compatible {self._memory_method_label()} snapshot for evaluation unit {unit.unit_id}: "
                    f"{self._memory_snapshot_path(unit)}"
                )
            self._record_source_build_metrics(unit, payload)
            queries = [
                query
                for query in unit.queries_to_evaluate
                if not (
                    self._checkpoint_manager
                    and self._checkpoint_manager.is_query_completed(
                        query.query_id,
                        persona_id=unit.context_id,
                    )
                )
                and query.query_id not in deferred_query_ids
            ]
            pending_units.append((
                EvaluationUnit(
                    unit_id=unit.unit_id,
                    sessions_to_inject=unit.sessions_to_inject,
                    queries_to_evaluate=queries,
                    context_id=unit.context_id,
                    metadata={
                        **unit.metadata,
                        # Resume may omit completed queries from this worker,
                        # but memory-build time must retain the serial run's
                        # original per-query attribution.
                        "_query_stage_original_query_count": len(
                            unit.queries_to_evaluate
                        ),
                    },
                ),
                payload,
            ))

        if not pending_units:
            return

        # Batch completion/finalization uses the coordinator's manager. Worker
        # managers are independent and never mutate this active manager state.
        if self.batch_api:
            self._init_agent_for_context(
                context_id=pending_units[0][0].context_id,
                force_new=True,
            )
            if self._supports_batch_queries():
                self._get_batch_client()

        def create_worker(item: tuple[EvaluationUnit, Dict[str, Any]]) -> Dict[str, Any]:
            unit, payload = item
            worker = copy.copy(self)
            # The outer executor owns the complete worker budget. Never let an
            # isolated unit create a nested per-query executor.
            worker.workers = 1
            worker.query_workers = 1
            worker.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=self.batch_api,
                batch_gcs_uri=getattr(self, "batch_gcs_uri", None),
                batch_wait=getattr(self, "batch_wait", False),
                workers=1,
                batch_manifest_dir=self.output_dir / "batch",
                batch_config_hash=self._batch_manifest_config_hash(),
                batch_progress_callback=self._log,
                query_only=self.execution_stage == "query",
            )
            worker._checkpoint_manager = None
            worker._memory_build_logs = []
            worker._source_memory_build_logs = []
            worker._pending_batch_queries = []
            worker._deferred_judges = []
            worker._api_failures = []
            worker._api_failure_duration_seconds = 0.0
            worker._api_failure_lock = threading.Lock()
            worker._batch_fallback_logged = True
            worker._judge_batch_fallback_logged = True
            worker._batch_client = getattr(self, "_batch_client", None)
            worker._judge_batch_client = getattr(self, "_judge_batch_client", None)
            worker.agent_manager.import_memory_state(
                payload["memory_state"],
                context_id=unit.context_id,
            )
            return {
                "unit": unit,
                "payload": payload,
                "worker": worker,
                "results": [],
            }

        context_worker_count = min(self._query_worker_count(), len(pending_units))
        self._log(
            f"[Workers] Initializing {len(pending_units):,} isolated query-unit "
            f"contexts with up to {context_worker_count} workers."
        )
        with ThreadPoolExecutor(max_workers=context_worker_count) as executor:
            worker_results = list(executor.map(create_worker, pending_units))

        try:
            if self._supports_batch_queries():
                # Batch final-answer transport has one combined dispatch below.
                # Unit preparation is bounded by the same global worker budget.
                with ThreadPoolExecutor(max_workers=context_worker_count) as executor:
                    for item, results in zip(
                        worker_results,
                        executor.map(
                            lambda item: item["worker"]._evaluate_unit_queries(
                                item["unit"],
                                total_memory_time=float(
                                    item["payload"].get("memory_build_time", 0.0)
                                ),
                            ),
                            worker_results,
                        ),
                    ):
                        item["results"] = results
            else:
                query_jobs = [
                    (item, query)
                    for item in worker_results
                    for query in item["unit"].queries_to_evaluate
                ]
                query_worker_count = min(self._query_worker_count(), len(query_jobs))
                self._log(
                    f"[Workers] Running {len(query_jobs):,} real-time queries across "
                    f"{len(worker_results):,} query units with {query_worker_count} "
                    "workers total."
                )

                def evaluate_query_job(job):
                    item, query = job
                    unit = item["unit"]
                    worker = item["worker"]
                    try:
                        memory_time = float(
                            item["payload"].get("memory_build_time", 0.0)
                        ) / int(
                            unit.metadata.get(
                                "_query_stage_original_query_count",
                                len(unit.queries_to_evaluate),
                            )
                        )
                        if worker._supports_staged_queries():
                            result = worker._evaluate_query_staged(
                                query,
                                unit.context_id,
                                memory_time_per_query=memory_time,
                                unit_id=unit.unit_id,
                            )
                        else:
                            result = worker._evaluate_query(
                                query,
                                unit.context_id,
                                memory_time_per_query=memory_time,
                                unit_id=unit.unit_id,
                            )
                        return item, result
                    finally:
                        self._advance_query_progress(
                            query.query_id,
                            context_id=unit.context_id,
                        )

                if query_jobs:
                    with ThreadPoolExecutor(max_workers=query_worker_count) as executor:
                        # map preserves unit/query ordering for report aggregation.
                        evaluated_queries = list(executor.map(evaluate_query_job, query_jobs))
                    for item, result in evaluated_queries:
                        if result is not None:
                            item["results"].append(result)
        finally:
            for item in worker_results:
                item["worker"].agent_manager.reset()

        for item in worker_results:
            worker = item["worker"]
            self._api_failures.extend(worker._api_failures)
            self._api_failure_duration_seconds = (
                getattr(self, "_api_failure_duration_seconds", 0.0)
                + worker._api_failure_duration_seconds
            )
            self._pending_batch_queries.extend(worker._pending_batch_queries)
            self._deferred_judges.extend(worker._deferred_judges)
            for result in item["results"]:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, item["unit"].context_id)
                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        result.query_id,
                        result.to_dict(),
                        persona_id=item["unit"].context_id,
                    )

        for item in self._complete_combined_batch_queries():
            result = item["result"]
            persona_id = item["persona_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, persona_id)
            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    result.query_id,
                    result.to_dict(),
                    persona_id=persona_id,
                )

        self._finish_query_progress()

        for item in self._complete_deferred_judges():
            result = item["result"]
            persona_id = item["persona_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, persona_id)
            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    result.query_id,
                    result.to_dict(),
                    persona_id=persona_id,
                )

        if self._checkpoint_manager and self.batch_api:
            for persona_id in self.dataset.get_persona_ids():
                self._checkpoint_manager.complete_persona(persona_id)
        elif self._checkpoint_manager:
            for unit, _ in pending_units:
                self._checkpoint_manager.complete_persona(unit.context_id)

    def _evaluate_append_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        """Reuse a carried-forward snapshot or build the next missing unit."""
        snapshot_payload = self._read_memory_snapshot(unit)
        if snapshot_payload is not None:
            if not self._supports_memory_snapshots():
                raise RuntimeError(
                    f"Method {self.method_config.method_name} does not support append snapshots"
                )
            self.agent_manager.import_memory_state(
                snapshot_payload["memory_state"],
                context_id=unit.context_id,
            )
            self._record_source_build_metrics(unit, snapshot_payload)
            self._log(
                f"\n  Evaluation Unit {unit.unit_id}: restored append memory from "
                f"{self._memory_snapshot_path(unit)}"
            )
            if self.execution_stage == "memory":
                return []
            return self._evaluate_unit_queries(
                unit,
                total_memory_time=float(snapshot_payload.get("memory_build_time", 0.0)),
            )
        return self._evaluate_unit_with_checkpoint(unit)

    def _evaluate_unit_with_checkpoint(self, unit: EvaluationUnit) -> List[MetricResult]:
        """Evaluate unit with checkpoint support."""
        source_integrity = self._validate_unit_source_integrity(unit)
        self._log(
            f"\nUnit {unit.unit_id} | persona={unit.context_id} | "
            f"sessions={len(unit.sessions_to_inject)} | queries={len(unit.queries_to_evaluate)}"
        )

        results = []

        self._log(f"  Memory build started ({len(unit.sessions_to_inject)} sessions)")

        memory_build_failed = False
        total_memory_time = 0.0
        staged_build_wall_time = None
        session_build_results = []  # Store per-session build results
        restored_payload = self._restore_completed_unit_for_resume(unit)
        restored_completed_unit = restored_payload is not None
        if restored_payload is not None:
            total_memory_time = float(restored_payload.get("memory_build_time", 0.0))

        if self.dry_run:
            self._log(f"    [Dry Run] Skipping memory build")
        else:
            total_sessions = len(unit.sessions_to_inject)
            session_ids = []

            method_name = str(getattr(getattr(self, "method_config", None), "method_name", "")).lower()
            pending_sessions = list(unit.sessions_to_inject)
            if self._session_level_force_resume() and not self._supports_memory_snapshots() and self._checkpoint_manager:
                pending_sessions = [
                    session for session in pending_sessions
                    if not self._checkpoint_manager.is_session_injected(self._method_source_uid(session), persona_id=unit.context_id)
                ]
            supports_staged_memory = getattr(self.agent_manager, "supports_staged_memory", None)
            staged_event_state = "event_state" in method_name and len(pending_sessions) > 1 and callable(supports_staged_memory) and supports_staged_memory()
            sessions_to_process = unit.sessions_to_inject
            if staged_event_state and not restored_completed_unit:
                # Extraction and embedding are independent per source session;
                # the returned preparations are committed below in source order.
                staged_items = []
                pending_ids = {self._method_source_uid(session) for session in pending_sessions}
                for idx, session in enumerate(unit.sessions_to_inject):
                    source_uid = self._method_source_uid(session)
                    if source_uid not in pending_ids:
                        continue
                    session_ids.append(source_uid)
                    session_scope = normalize_scope(explicit=session.conversation_scope)
                    source_messages = session.metadata.get("messages", []) or [{"speaker": "Unknown", "content": session.content, "source_turn_id": None}]
                    for turn_index, item in enumerate(source_messages):
                        staged_item = dict(item)
                        staged_item.setdefault("source_session_id", source_uid)
                        staged_item.setdefault("source_uid", source_uid)
                        staged_item.setdefault("source_session_index", idx)
                        staged_item.setdefault("source_turn_id", item.get("turn_id", item.get("dia_id", turn_index)))
                        staged_item.setdefault("source_event_id", session.metadata.get("event_id"))
                        staged_item.setdefault("timestamp", session.timestamp)
                        staged_item.setdefault("conversation_scope", session_scope)
                        staged_items.append(staged_item)
                    marker = getattr(self._checkpoint_manager, "mark_session_started", None)
                    if callable(marker):
                        marker(source_uid, unit.unit_id)
                staged_start = time.perf_counter()
                staged_usage_before = get_usage_tracker().get_stats()
                try:
                    # Each session has an independent preparation step and a
                    # stateful ordered commit step.
                    self._start_memory_progress(len(pending_sessions) * 2)
                    staged_text = "\n\n".join(session.to_memory_text() for session in pending_sessions)
                    prepared = self.agent_manager.prepare_memory_sessions(
                        message=staged_text,
                        context_id=unit.context_id,
                        memory_items=staged_items,
                        timestamp=None,
                        progress_callback=self._advance_memory_progress,
                    )
                    total_memory_time += time.perf_counter() - staged_start
                    for prepared_session in prepared:
                        session_start = time.perf_counter()
                        memory_result = self.agent_manager.commit_prepared_memory(
                            [prepared_session],
                            text=prepared_session.session.get("text", ""),
                            context_id=unit.context_id,
                        )
                        session_id = prepared_session.session.get("source_session_id")
                        session_index = prepared_session.session.get("source_session_index")
                        session_build_results.append({
                            "source_uid": session_id,
                            "session_index": session_index,
                            "build_result": memory_result.to_dict(),
                            "wall_time_seconds": time.perf_counter() - session_start,
                            "build_metrics": {"wall_time_seconds": time.perf_counter() - session_start, "usage": {}},
                        })
                        total_memory_time += time.perf_counter() - session_start
                        if self._checkpoint_manager:
                            self._checkpoint_manager.mark_session_injected(session_id)
                        self._advance_memory_progress()
                    if session_build_results:
                        session_build_results[0]["build_metrics"]["usage"] = diff_usage_stats(
                            get_usage_tracker().get_stats(), staged_usage_before
                        )
                    staged_build_wall_time = time.perf_counter() - staged_start
                    sessions_to_process = []
                except Exception as exc:
                    memory_build_failed = True
                    self._log(
                        f"        [ERROR] Event-State staged preparation failed: {truncate_error_message(exc)}",
                        level="ERROR",
                    )
                    rollback_incomplete = getattr(
                        self._checkpoint_manager,
                        "rollback_incomplete_session",
                        None,
                    )
                    if callable(rollback_incomplete):
                        rollback_incomplete()
                    if not isinstance(exc, LLMAPIError):
                        raise
                    sessions_to_process = []
                finally:
                    self._finish_memory_progress()

            for idx, session in enumerate(sessions_to_process):
                source_uid = self._method_source_uid(session)
                session_ids.append(source_uid)
                if restored_completed_unit:
                    self._log(
                        f"      [Resume] Session {source_uid} restored from snapshot; skipping"
                    )
                    continue
                if (
                    self._session_level_force_resume()
                    and not self._supports_memory_snapshots()
                    and self._checkpoint_manager
                    and self._checkpoint_manager.is_session_injected(
                        source_uid,
                        persona_id=unit.context_id,
                    )
                ):
                    self._log(
                        f"      [Resume] Session {source_uid} already injected; skipping"
                    )
                    continue
                session_succeeded = False
                memory_text = session.to_memory_text()

                # Show progress
                self._log(f"      [Progress] Session {idx + 1}/{total_sessions} (source UID: {source_uid})")

                formatted_text = self.prompt_manager.format_memorize(
                    context=memory_text,
                    timestamp=None,
                )

                is_last_session = (idx == total_sessions - 1)

                mark_session_started = getattr(
                    self._checkpoint_manager,
                    "mark_session_started",
                    None,
                )
                if callable(mark_session_started):
                    mark_session_started(source_uid, unit.unit_id)

                usage_tracker = get_usage_tracker()
                usage_before = usage_tracker.get_stats()
                session_started_at = time.perf_counter()
                session_build_record: Dict[str, Any] = {
                    "source_uid": source_uid,
                    "session_index": idx,
                }
                try:
                    # Generic source semantics are available to every adapter;
                    # benchmark gold IDs and distractor labels remain private.
                    source_metadata = {
                        "source_session_id": source_uid,
                        "source_uid": source_uid,
                        "source_session_index": idx,
                        "source_event_id": session.metadata.get("event_id"),
                        "conversation_scope": session.conversation_scope,
                    }
                    memory_result = self._run_api_call(
                        self.agent_manager.send_message,
                        message=formatted_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_session,
                        timestamp=session.timestamp,
                        memory_items=session.metadata.get("messages", []),
                        **source_metadata,
                    )

                    if memory_result is not None and isinstance(memory_result, MemoryBuildResult):
                        session_succeeded = True
                        total_memory_time += memory_result.time_cost

                        for failure in memory_result.extra.get("api_failures", []):
                            self._record_api_failure(
                                "memory_build",
                                failure,
                                unit_id=unit.unit_id,
                                context_id=unit.context_id,
                                session_id=source_uid,
                                session_index=idx,
                            )

                        passages_count = len(memory_result.all_passages) if memory_result.all_passages else memory_result.extra.get("inserted_count", 0)
                        self._log(
                            f"  Session {source_uid} complete | "
                            f"passages={passages_count} | time={memory_result.time_cost:.2f}s"
                        )

                        session_build_record["build_result"] = memory_result.to_dict()

                except LLMAPIError as e:
                    session_build_record["error"] = truncate_error_message(e)
                    self._log(
                        f"        [API ERROR] Session {source_uid} failed: {truncate_error_message(e)}",
                        level="ERROR"
                    )
                    memory_build_failed = True
                    memory_failures = self._drain_memory_api_failures(unit.context_id)
                    if not memory_failures:
                        memory_failures = [{}]
                    for failure in memory_failures:
                        self._record_api_failure(
                            "memory_build",
                            failure if failure else e,
                            unit_id=unit.unit_id,
                            context_id=unit.context_id,
                            session_id=source_uid,
                            session_index=idx,
                            affected_query_ids=[
                                query.query_id for query in unit.queries_to_evaluate
                            ],
                        )
                    rollback_incomplete = getattr(
                        self._checkpoint_manager,
                        "rollback_incomplete_session",
                        None,
                    )
                    if callable(rollback_incomplete):
                        rollback_incomplete()
                finally:
                    session_wall_time = time.perf_counter() - session_started_at
                    session_build_record["wall_time_seconds"] = session_wall_time
                    session_build_record["build_metrics"] = {
                        "wall_time_seconds": session_wall_time,
                        "usage": diff_usage_stats(
                            usage_tracker.get_stats(), usage_before
                        ),
                    }
                    session_build_results.append(session_build_record)

                if self._checkpoint_manager and session_succeeded:
                    self._checkpoint_manager.mark_session_injected(source_uid)
                if memory_build_failed:
                    break

            # Log summary
            total_passages = sum(
                len(r.get("build_result", {}).get("all_passages", []))
                for r in session_build_results if "build_result" in r
            )
            self._log(f"      [Summary] Total passages stored: {total_passages}")

            if restored_payload is not None and not session_build_results:
                unit_build_metrics = copy.deepcopy(
                    restored_payload.get("memory_build_metrics") or {
                        "wall_time_seconds": total_memory_time,
                        "usage": {},
                    }
                )
            else:
                unit_usage = merge_usage_stats(*[
                    record.get("build_metrics", {}).get("usage", {})
                    for record in session_build_results
                ])
                unit_build_metrics = {
                    "wall_time_seconds": staged_build_wall_time if staged_build_wall_time is not None else sum(
                        float(record.get("wall_time_seconds", 0.0) or 0.0)
                        for record in session_build_results
                    ),
                    "usage": unit_usage,
                }

            # Store all session build results
            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "session_ids": session_ids,
                "mixed_source_integrity": {
                    **source_integrity,
                    "source_records_ingested_count": len(session_build_results),
                },
                "session_count": total_sessions,
                "total_time": total_memory_time,
                "total_passages": total_passages,
                "session_builds": session_build_results,
                "build_metrics": unit_build_metrics,
                "feature_configuration": self._amem_feature_configuration(),
                "restored_from_snapshot": bool(
                    restored_payload is not None and not session_build_results
                ),
            })

        self._log(
            f"  Memory build complete | passages={total_passages if not self.dry_run else 0} | "
            f"time={total_memory_time:.2f}s"
        )

        if memory_build_failed:
            self._log(
                "    [API ERROR] Skipping this unit's queries because memory construction "
                "did not complete; no metric results will be recorded.",
                level="ERROR",
            )
            return results

        # Snapshot boundaries are part of the staged AMem execution, but their
        # disk I/O is intentionally excluded from method construction/query time.
        snapshot_path = self._save_and_restore_unit_memory(
            unit,
            memory_build_time=total_memory_time,
            memory_build_metrics=(
                self._memory_build_logs[-1].get("build_metrics", {})
                if self._memory_build_logs else {}
            ),
        )
        if snapshot_path is not None and self._memory_build_logs:
            self._memory_build_logs[-1]["memory_snapshot"] = str(snapshot_path)

        if getattr(self, "execution_stage", "all") == "memory":
            self._log("  Query evaluation skipped (memory stage only)")
            return results

        return self._evaluate_unit_queries(unit, total_memory_time)

    def _evaluate_unit_queries(
        self,
        unit: EvaluationUnit,
        total_memory_time: float,
    ) -> List[MetricResult]:
        """Retrieve, answer, and score one unit from its active memory state."""
        results: List[MetricResult] = []

        total_query_time = 0.0
        self._log("  Query evaluation started")

        query_count = int(
            getattr(unit, "metadata", {}).get(
                "_query_stage_original_query_count",
                len(unit.queries_to_evaluate),
            )
        )
        memory_time_per_query = total_memory_time / query_count if query_count > 0 else 0.0

        pending_query_count = sum(
            1
            for query in unit.queries_to_evaluate
            if not (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(
                    query.query_id,
                    persona_id=unit.context_id,
                )
            )
            and not self._is_deferred_judge_query(query.query_id)
        )
        if not pending_query_count:
            self._log("  Query evaluation complete | no pending questions")
            return results

        self._start_query_progress()

        if pending_query_count and self._supports_batch_queries():
            batch_client = self._get_batch_client()
            combined_stage = "query-final"
            legacy_stage = f"query-unit-{unit.unit_id}"
            uses_query_compiler = bool(
                getattr(self, "agent_manager", None)
                and getattr(self.agent_manager, "uses_query_compiler", lambda: False)()
            )
            if (
                not uses_query_compiler
                and not batch_client.has_stage(combined_stage)
                and batch_client.has_stage(legacy_stage)
            ):
                query_results = self._evaluate_batch_queries(unit, memory_time_per_query)
            else:
                prepared_count = self._prepare_combined_batch_queries(
                    unit,
                    memory_time_per_query,
                )
                query_results = []
                self._log(
                    f"    [Batch] Prepared {prepared_count:,} request(s) for "
                    f"combined stage '{combined_stage}'."
                )
            for result in query_results:
                results.append(result)
                total_query_time += result.query_time
                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        result.query_id,
                        result.to_dict(),
                        persona_id=unit.context_id,
                    )
        else:
            if self.batch_api and not self.dry_run and not self._batch_fallback_logged:
                self._log(
                    "Batch API is unavailable for this adapter; "
                    "using real-time query execution.",
                    level="WARNING",
                )
                self._batch_fallback_logged = True

            pending_queries = []
            for query in unit.queries_to_evaluate:
                if (
                    self._checkpoint_manager
                    and self._checkpoint_manager.is_query_completed(
                        query.query_id,
                        persona_id=unit.context_id,
                    )
                ):
                    self._log(f"    [Skip] {query.query_id} (completed)")
                    continue
                if self._is_deferred_judge_query(query.query_id):
                    self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                    continue
                pending_queries.append(query)

            for query, result in self._evaluate_realtime_queries(
                pending_queries,
                context_id=unit.context_id,
                memory_time_per_query=memory_time_per_query,
                unit_id=unit.unit_id,
            ):
                if result is None:
                    continue
                results.append(result)
                total_query_time += result.query_time

                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        query.query_id,
                        result.to_dict(),
                        persona_id=unit.context_id,
                    )

        self._log(f"  Query evaluation complete | time={total_query_time:.2f}s")

        return results

    def _evaluate_realtime_queries(
        self,
        queries: List[MedQuery],
        *,
        context_id: int,
        memory_time_per_query: float,
        unit_id: int,
    ) -> List[tuple[MedQuery, Optional[MetricResult]]]:
        """Run independent answer-and-score pairs with bounded concurrency."""
        def evaluate_one(query: MedQuery) -> tuple[MedQuery, Optional[MetricResult]]:
            try:
                if self._supports_staged_queries():
                    result = self._evaluate_query_staged(
                        query,
                        context_id,
                        memory_time_per_query=memory_time_per_query,
                        unit_id=unit_id,
                    )
                else:
                    result = self._evaluate_query(
                        query,
                        context_id,
                        memory_time_per_query=memory_time_per_query,
                        unit_id=unit_id,
                    )
                return query, result
            finally:
                self._advance_query_progress(
                    query.query_id,
                    context_id=context_id,
                )

        query_workers = self._query_worker_count()
        if len(queries) < 2 or query_workers == 1:
            return [evaluate_one(query) for query in queries]

        worker_count = min(query_workers, len(queries))
        self._log(f"    [Workers] Running {len(queries):,} real-time queries with {worker_count} workers.")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves dataset order for deterministic reports/checkpoints.
            return list(executor.map(evaluate_one, queries))

    def _supports_batch_queries(self) -> bool:
        return bool(
            self.batch_api
            and self.agent_manager
            and self.agent_manager.supports_batch_queries()
        )

    def _get_batch_client(self) -> VertexBatchClient:
        if self._batch_client is None:
            if self.agent_manager is None:
                raise VertexBatchError("Agent manager is not initialized for batch execution.")
            llm_client = self.agent_manager.get_batch_llm_client()
            if llm_client is None:
                raise VertexBatchError("This method does not expose a managed batch client.")
            self._batch_client = create_batch_client(
                llm_client,
                gcs_uri=self.batch_gcs_uri,
                manifest_path=self._batch_manifest_path("medmemorybench_batch_manifest", llm_client.model),
                wait=self.batch_wait,
                config_hash=self._batch_manifest_config_hash(),
                progress_callback=self._log,
                vertex_batch_class=VertexBatchClient,
            )
        return self._batch_client

    def _get_judge_batch_client(self, judge_client) -> VertexBatchClient:
        """Create an independent manifest because judge and agent models may differ."""
        if self._judge_batch_client is None:
            self._judge_batch_client = create_batch_client(
                judge_client,
                gcs_uri=self.batch_gcs_uri,
                manifest_path=scoped_manifest_path(
                    self.output_dir / "batch",
                    "medmemorybench_judge_batch_manifest",
                    model=judge_client.model,
                    config_hash=self._judge_batch_config_hash(),
                ),
                wait=self.batch_wait,
                config_hash=self._judge_batch_config_hash(),
                progress_callback=self._log,
                vertex_batch_class=VertexBatchClient,
            )
        return self._judge_batch_client

    def _evaluate_batch_queries(
        self,
        unit: EvaluationUnit,
        memory_time_per_query: float,
    ) -> List[MetricResult]:
        """Prepare local retrieval for a unit, then batch its final generations."""
        prepared_by_id: Dict[str, tuple[Any, Dict[str, Any]]] = {}
        requests: List[BatchChatRequest] = []
        stage = f"query-unit-{unit.unit_id}"
        batch_client = self._get_batch_client()

        for query in unit.queries_to_evaluate:
            if (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(
                    query.query_id,
                    persona_id=unit.context_id,
                )
            ):
                self._log(f"    [Skip] {query.query_id} (completed)")
                continue
            if self._is_deferred_judge_query(query.query_id):
                self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                continue
            request_id = make_request_id(
                "query",
                f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}",
            )
            saved_request = batch_client.get_saved_request(stage, request_id)
            prepared = restore_prepared_query(saved_request) if saved_request else None
            if prepared is None:
                # Reuse a persisted display timestamp for legacy manifests that
                # predate prepared-query snapshots. New manifests do not rerun
                # retrieval at all when collecting a completed stage.
                batch_request_time = (
                    saved_request.metadata.get("batch_request_time")
                    if saved_request is not None
                    else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                formatted_question = self._format_query_prompt(
                    question=query.question,
                    query_type=query.query_type,
                )
                try:
                    prepared = self._run_api_call(
                        self.agent_manager.prepare_batch_query,
                        formatted_question,
                        query_id=query.query_id,
                        context_id=unit.context_id,
                        batch_request_time=batch_request_time,
                        **self._answer_query_kwargs(query),
                    )
                except LLMAPIError as e:
                    self._log(
                        f"    [API ERROR] {query.query_id}: query preparation failed; "
                        f"skipping this query. Error: {truncate_error_message(e)}",
                        level="ERROR",
                    )
                    self._record_api_failure(
                        "query_preparation",
                        e,
                        query_id=query.query_id,
                        query_type=query.query_type,
                        context_id=unit.context_id,
                        unit_id=unit.unit_id,
                    )
                    self._advance_query_progress(
                        query.query_id,
                        context_id=unit.context_id,
                    )
                    continue

            if saved_request is not None:
                # Reuse the exact staged request. This protects a resume from
                # both timestamp changes and non-deterministic retrieval.
                requests.append(saved_request)
            else:
                requests.append(
                    BatchChatRequest(
                        request_id=request_id,
                        messages=prepared["messages"],
                        temperature=self.method_config.model.temperature,
                        max_tokens=(
                            self.method_config.model.max_completion_tokens
                            or self.method_config.model.max_tokens
                        ),
                        phase="query",
                        metadata={
                            "query_id": query.query_id,
                            "unit_id": unit.unit_id,
                            "batch_request_time": batch_request_time,
                            PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared),
                        },
                    )
                )
            prepared_by_id[request_id] = (query, prepared)

        if not requests:
            return []

        self._log(
            f"    [Batch] Stage '{stage}': local preparation complete; "
            f"submitting {len(requests):,} final-answer request(s)."
        )
        get_saved_requests = getattr(batch_client, "get_saved_requests", None)
        saved_requests = get_saved_requests(stage) if callable(get_saved_requests) else []
        submitted_requests = saved_requests or requests
        tracker = get_usage_tracker()
        tracker.set_phase("query")
        with tracker.scope("query.answer_batch"):
            responses = batch_client.run_stage(stage, submitted_requests)
        results: List[MetricResult] = []
        for request_id, (query, prepared) in prepared_by_id.items():
            batch_response = responses.get(request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                self._record_batch_api_failure(
                    query,
                    f"Batch request failed: {truncate_error_message(error)}",
                    failure_duration_seconds=(
                        getattr(batch_response, "duration_seconds", 0.0)
                        if batch_response is not None else 0.0
                    ),
                    context_id=unit.context_id,
                    unit_id=unit.unit_id,
                )
                self._advance_query_progress(
                    query.query_id,
                    context_id=unit.context_id,
                )
                continue
            try:
                response = self._run_api_call(
                    self.agent_manager.finalize_batch_query,
                    prepared,
                    batch_response.content,
                    input_tokens=batch_response.input_tokens,
                    output_tokens=batch_response.output_tokens,
                )
                result = self._run_api_call(
                    self._score_agent_response,
                    query,
                    response,
                    context_id=unit.context_id,
                    memory_construction_time=memory_time_per_query,
                    artifact_references=self._query_artifact_references(
                        context_id=unit.context_id,
                        unit_id=unit.unit_id,
                        answer_request_id=request_id,
                        answer_batch_client=batch_client,
                        answer_messages=prepared.get("messages"),
                    ),
                    execution_usage={
                        "answer": self._batch_response_usage(
                            batch_response, transport="batch"
                        ),
                    },
                )
            except LLMAPIError as e:
                self._log(
                    f"    [API ERROR] {query.query_id}: judge API call failed; "
                    f"skipping this query. Error: {truncate_error_message(e)}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "judge",
                    e,
                    query_id=query.query_id,
                    query_type=query.query_type,
                    context_id=unit.context_id,
                    unit_id=unit.unit_id,
                )
                self._advance_query_progress(
                    query.query_id,
                    context_id=unit.context_id,
                )
                continue
            # This bar tracks final-answer completion, not asynchronous judge
            # completion. A deferred judge already has its answer and belongs
            # to the subsequent judge stage.
            self._advance_query_progress(
                query.query_id,
                context_id=unit.context_id,
            )
            if result is not None:
                results.append(result)
        return results

    def _prepare_combined_batch_queries(
        self,
        unit: EvaluationUnit,
        memory_time_per_query: float,
    ) -> int:
        """Freeze this unit's prompts while its current memory snapshot is active."""
        stage = "query-final"
        batch_client = self._get_batch_client()
        prepared_count = 0

        # Query-compiler plans are a dependency of local retrieval, so final
        # prompts cannot be frozen until the shared query-plan stage completes.
        if self.agent_manager and getattr(self.agent_manager, "uses_query_compiler", lambda: False)():
            event_state_snapshot = self._export_event_state_for_transfer(
                self.agent_manager, unit.context_id,
            )
            plan_cache = self._load_query_compiler_plan_cache()
            for query in unit.queries_to_evaluate:
                if (
                    self._checkpoint_manager
                    and self._checkpoint_manager.is_query_completed(
                        query.query_id, persona_id=unit.context_id
                    )
                ):
                    self._log(f"    [Skip] {query.query_id} (completed)")
                    continue
                if self._is_deferred_judge_query(query.query_id):
                    self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                    continue
                final_id = make_request_id(
                    "query", f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}",
                )
                saved_final = batch_client.get_saved_request(stage, final_id)
                restored_final = restore_prepared_query(saved_final) if saved_final else None
                if restored_final is not None:
                    self._pending_batch_queries.append({
                        "request": saved_final, "query": query, "prepared": restored_final,
                        "persona_id": unit.context_id, "unit_id": unit.unit_id,
                        "memory_time_per_query": memory_time_per_query,
                        "compiler_usage": {"transport": "resume", "call_count": 0, "cache_hit": False},
                    })
                    prepared_count += 1
                    continue
                compiler = self.agent_manager.prepare_query_compiler(query.question)
                reference_time = None
                fingerprint = self._query_compiler_cache_fingerprint(query.question, reference_time)
                cached = plan_cache.get(fingerprint) or self._legacy_query_compiler_cache_entry(
                    plan_cache, query.question, reference_time,
                )
                self._compiler_cache_hits += int(cached is not None)
                self._compiler_cache_misses += int(cached is None)
                self._pending_query_plan_requests.append({
                    "request": BatchChatRequest(
                        request_id=make_request_id(
                            "query-plan", f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}",
                        ),
                        messages=compiler["messages"], temperature=compiler["temperature"],
                        max_tokens=compiler["max_tokens"], response_format=compiler.get("response_format"),
                        phase="query-plan",
                        metadata={"query_id": query.query_id, "unit_id": unit.unit_id, "context_id": unit.context_id},
                    ),
                    "query": query,
                    "question": self._format_query_prompt(query.question, query.query_type),
                    "raw_question": query.question,
                    "persona_id": unit.context_id,
                    "unit_id": unit.unit_id,
                    "memory_state": event_state_snapshot,
                    "memory_time_per_query": memory_time_per_query,
                    "reference_time": reference_time,
                    "compiler_fingerprint": fingerprint,
                    "cached_plan": cached,
                })
                prepared_count += 1
            return prepared_count

        query_items = []
        for query in unit.queries_to_evaluate:
            if (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(
                    query.query_id, persona_id=unit.context_id
                )
            ):
                self._log(f"    [Skip] {query.query_id} (completed)")
                continue
            if self._is_deferred_judge_query(query.query_id):
                self._log(
                    f"    [Skip] {query.query_id} "
                    "(awaiting saved Vertex judge result)"
                )
                continue
            request_id = make_request_id(
                "query",
                f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}",
            )
            saved_request = batch_client.get_saved_request(stage, request_id)
            batch_request_time = (
                saved_request.metadata.get("batch_request_time")
                if saved_request is not None
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            prepared = restore_prepared_query(saved_request) if saved_request else None
            query_items.append((query, request_id, saved_request, batch_request_time, prepared))

        def prepare_item(item):
            query, _, _, batch_request_time, prepared = item
            if prepared is not None:
                return prepared, None
            formatted_question = self._format_query_prompt(
                question=query.question,
                query_type=query.query_type,
            )
            try:
                return self._run_api_call(
                    self.agent_manager.prepare_batch_query,
                    formatted_question,
                    query_id=query.query_id,
                    context_id=unit.context_id,
                    batch_request_time=batch_request_time,
                    **self._answer_query_kwargs(query),
                ), None
            except LLMAPIError as exc:
                return None, exc

        worker_count = self._query_worker_count()
        if worker_count > 1 and len(query_items) > 1:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(query_items))) as executor:
                prepared_items = list(executor.map(prepare_item, query_items))
        else:
            prepared_items = [prepare_item(item) for item in query_items]

        for item, (prepared, preparation_error) in zip(query_items, prepared_items):
            query, request_id, saved_request, batch_request_time, _ = item
            if preparation_error is not None:
                self._log(
                    f"    [API ERROR] {query.query_id}: query preparation failed; "
                    f"skipping this query. Error: {preparation_error}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "query_preparation",
                    preparation_error,
                    query_id=query.query_id,
                    query_type=query.query_type,
                    context_id=unit.context_id,
                    unit_id=unit.unit_id,
                )
                self._advance_query_progress(
                    query.query_id,
                    context_id=unit.context_id,
                )
                continue

            request = saved_request or BatchChatRequest(
                request_id=request_id,
                messages=prepared["messages"],
                temperature=self.method_config.model.temperature,
                max_tokens=(
                    self.method_config.model.max_completion_tokens
                    or self.method_config.model.max_tokens
                ),
                phase="query",
                metadata={
                    "query_id": query.query_id,
                    "unit_id": unit.unit_id,
                    "context_id": unit.context_id,
                    "memory_time_per_query": memory_time_per_query,
                    "batch_request_time": batch_request_time,
                    PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared),
                },
            )
            self._pending_batch_queries.append({
                "request": request,
                "query": query,
                "prepared": prepared,
                "persona_id": unit.context_id,
                "unit_id": unit.unit_id,
                "memory_time_per_query": request.metadata.get(
                    "memory_time_per_query",
                    memory_time_per_query,
                ),
            })
            prepared_count += 1

        return prepared_count

    def _complete_combined_batch_queries(self) -> List[Dict[str, Any]]:
        """Submit all frozen final-answer prompts in one provider batch job."""
        if getattr(self, "_pending_query_plan_requests", []):
            batch_client = self._get_batch_client()
            pending_plans = self._pending_query_plan_requests
            self._pending_query_plan_requests = []
            misses = [item for item in pending_plans if item["cached_plan"] is None]
            saved_requests = batch_client.get_saved_requests("query-plan") if misses else []
            expected_ids = {item["request"].request_id for item in misses}
            requests = (
                saved_requests
                if {request.request_id for request in saved_requests} == expected_ids
                else [item["request"] for item in misses]
            )
            self._log(f"[Vertex] Stage 'query-plan': dispatching {len(requests):,} combined compiler request(s).")
            tracker = get_usage_tracker()
            tracker.set_phase("query")
            with tracker.scope("query.compiler_batch"):
                responses = batch_client.run_stage("query-plan", requests) if requests else {}
            self._compiler_calls_this_run += len(requests)
            managers: Dict[Any, AgentManager] = {}
            try:
                for item in pending_plans:
                    cached = item["cached_plan"]
                    batch_response = responses.get(item["request"].request_id)
                    content = cached["raw_model_output"] if cached is not None else (
                        batch_response.content if batch_response is not None and not batch_response.status else ""
                    )
                    context_id = item["persona_id"]
                    manager = managers.get(context_id)
                    if manager is None:
                        manager = AgentManager(
                            method_config=self.method_config, dataset_config=self.dataset_config,
                            batch_api=self.batch_api, batch_gcs_uri=self.batch_gcs_uri,
                            batch_wait=self.batch_wait, workers=1,
                        )
                        manager.import_memory_state(item["memory_state"], context_id=context_id)
                        managers[context_id] = manager
                    item["memory_state"] = None
                    prepared = manager.prepare_query_compiler_result(
                        item["question"], content, context_id=context_id,
                        **self._answer_query_kwargs(item["query"]),
                    )
                    diagnostics = prepared.get("extra", {}).get("query_compiler", {})
                    if cached is None and diagnostics.get("compiler_validation_success"):
                        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
                        self._append_query_compiler_plan_cache({
                            "fingerprint": item["compiler_fingerprint"],
                            "question_sha256": hashlib.sha256(item["raw_question"].encode("utf-8")).hexdigest(),
                            "reference_time": item["reference_time"],
                            "provider": getattr(self.method_config.model, "provider", None),
                            "model": getattr(self.method_config.model, "name", None),
                            "prompt_version": "event_state_query_compiler_v2", "schema_version": 2,
                            "max_searches": retrieval.get("query_compiler_max_searches", 3),
                            "query_compiler_max_tokens": retrieval.get("query_compiler_max_tokens", 256),
                            "temperature": retrieval.get("planner_temperature", 0.0),
                            "raw_model_output": content, "validated_plan": diagnostics.get("validated_plan"),
                            "parse_success": diagnostics.get("compiler_parse_success"),
                            "salvage_used": diagnostics.get("compiler_salvage_used"),
                            "warning_codes": diagnostics.get("compiler_warning_codes", []),
                        })
                    final_request = BatchChatRequest(
                        request_id=make_request_id(
                            "query", f"{self.method_config.method_name}:{item['unit_id']}:{item['query'].query_id}",
                        ),
                        messages=prepared["messages"], temperature=self.method_config.model.temperature,
                        max_tokens=(self.method_config.model.max_completion_tokens or self.method_config.model.max_tokens),
                        phase="query",
                        metadata={
                            "query_id": item["query"].query_id, "unit_id": item["unit_id"],
                            "context_id": context_id,
                            PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared),
                        },
                    )
                    self._pending_batch_queries.append({
                        "request": final_request, "query": item["query"], "prepared": prepared,
                        "persona_id": context_id, "unit_id": item["unit_id"],
                        "memory_time_per_query": item["memory_time_per_query"],
                        "compiler_usage": {
                            "transport": "cache" if cached is not None else "batch",
                            "input_tokens": 0 if cached is not None else getattr(batch_response, "input_tokens", 0),
                            "output_tokens": 0 if cached is not None else getattr(batch_response, "output_tokens", 0),
                            "call_count": 0 if cached is not None else 1,
                            "cache_hit": cached is not None,
                        },
                    })
            finally:
                managers.clear()
                pending_plans.clear()
                if isinstance(responses, dict):
                    responses.clear()
                gc.collect()
            return self._complete_combined_batch_queries()

        if not self._pending_batch_queries:
            return []

        stage = "query-final"
        batch_client = self._get_batch_client()
        saved_requests = batch_client.get_saved_requests(stage)
        requests = saved_requests or [
            item["request"] for item in self._pending_batch_queries
        ]
        self._log(
            f"[Vertex] Stage '{stage}': dispatching {len(requests):,} combined "
            "final-answer request(s) from all prepared units."
        )
        tracker = get_usage_tracker()
        tracker.set_phase("query")
        with tracker.scope("query.answer_batch"):
            responses = batch_client.run_stage(stage, requests)
        finalized: List[Dict[str, Any]] = []

        for item in self._pending_batch_queries:
            request = item["request"]
            query = item["query"]
            batch_response = responses.get(request.request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                self._record_batch_api_failure(
                    query,
                    f"Batch request failed: {truncate_error_message(error)}",
                    failure_duration_seconds=(
                        getattr(batch_response, "duration_seconds", 0.0)
                        if batch_response is not None else 0.0
                    ),
                    context_id=item["persona_id"],
                    unit_id=item["unit_id"],
                )
                self._advance_query_progress(
                    query.query_id,
                    context_id=item["persona_id"],
                )
                continue
            try:
                response = self._run_api_call(
                    self.agent_manager.finalize_batch_query,
                    item["prepared"],
                    batch_response.content,
                    input_tokens=batch_response.input_tokens,
                    output_tokens=batch_response.output_tokens,
                )
                result = self._run_api_call(
                    self._score_agent_response,
                    query,
                    response,
                    context_id=item["persona_id"],
                    memory_construction_time=item["memory_time_per_query"],
                    artifact_references=self._query_artifact_references(
                        context_id=item["persona_id"],
                        unit_id=item["unit_id"],
                        answer_request_id=request.request_id,
                        answer_batch_client=batch_client,
                        answer_messages=item["prepared"].get("messages"),
                    ),
                    execution_usage={
                        "answer": self._batch_response_usage(batch_response, transport="batch"),
                        **({"query_compiler": item["compiler_usage"]} if item.get("compiler_usage") else {}),
                    },
                )
            except LLMAPIError as e:
                self._log(
                    f"[API ERROR] {query.query_id}: judge API call failed; "
                    f"skipping this query. Error: {truncate_error_message(e)}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "judge",
                    e,
                    query_id=query.query_id,
                    query_type=query.query_type,
                    context_id=item["persona_id"],
                    unit_id=item["unit_id"],
                )
                self._advance_query_progress(
                    query.query_id,
                    context_id=item["persona_id"],
                )
                continue
            # The answer batch is complete even when metric scoring is queued
            # for the next judge batch.
            self._advance_query_progress(
                query.query_id,
                context_id=item["persona_id"],
            )
            if result is not None:
                finalized.append({
                    "persona_id": item["persona_id"],
                    "result": result,
                })

        self._pending_batch_queries = []
        return finalized

    def _format_query_prompt(
        self,
        question: str,
        query_type: Optional[str] = None,
    ) -> str:
        prompt_protocol = getattr(self, "prompt_protocol", "type_aware")
        try:
            return self.prompt_manager.format_query(
                question=question,
                query_type=query_type,
                prompt_protocol=prompt_protocol,
            )
        except TypeError:
            return self.prompt_manager.format_query(
                question=question,
                query_type=query_type,
            )

    def _evaluate_query(
        self,
        query,
        context_id: int,
        memory_time_per_query: float = 0.0,
        unit_id: Optional[int] = None,
    ) -> Optional[MetricResult]:
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

        formatted_question = self._format_query_prompt(
            question=query.question,
            query_type=query.query_type,
        )

        try:
            response = self._run_api_call(
                self.agent_manager.send_message,
                message=formatted_question,
                memorizing=False,
                query_id=query.query_id,
                context_id=context_id,
                **self._answer_query_kwargs(query),
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: answer API call failed, "
                f"skipping this query. Error: {truncate_error_message(e)}",
                level="ERROR"
            )
            self._record_api_failure(
                "query",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

        try:
            return self._run_api_call(
                self._score_agent_response,
                query,
                response,
                context_id=context_id,
                memory_construction_time=memory_time_per_query,
                artifact_references=self._query_artifact_references(
                    context_id=context_id,
                    unit_id=unit_id,
                    answer_request_id=make_request_id(
                        "query", f"{self.method_config.method_name}:{unit_id}:{query.query_id}"
                    ) if unit_id is not None else None,
                    answer_messages=[{"role": "user", "content": formatted_question}],
                ),
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: judge API call failed, "
                f"skipping this query. Error: {truncate_error_message(e)}",
                level="ERROR"
            )
            self._record_api_failure(
                "judge",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

    def _evaluate_query_staged(
        self,
        query,
        context_id: int,
        memory_time_per_query: float = 0.0,
        unit_id: Optional[int] = None,
    ) -> Optional[MetricResult]:
        """Run retrieval, final answering, and scoring as explicit stages."""
        formatted_question = self._format_query_prompt(
            question=query.question,
            query_type=query.query_type,
        )
        try:
            staged_query = self._run_api_call(
                self.agent_manager.prepare_query,
                message=formatted_question,
                query_id=query.query_id,
                context_id=context_id,
                **self._answer_query_kwargs(query),
            )
            response = self._run_api_call(
                self.agent_manager.answer_prepared_query,
                staged_query,
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: answer API call failed, "
                f"skipping this query. Error: {truncate_error_message(e)}",
                level="ERROR",
            )
            self._record_api_failure(
                "query",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

        try:
            return self._run_api_call(
                self._score_agent_response,
                query,
                response,
                context_id=context_id,
                memory_construction_time=memory_time_per_query,
                artifact_references=self._query_artifact_references(
                    context_id=context_id,
                    unit_id=unit_id,
                    answer_request_id=make_request_id(
                        "query", f"{self.method_config.method_name}:{unit_id}:{query.query_id}"
                    ) if unit_id is not None else None,
                    answer_messages=staged_query.get("messages") if isinstance(staged_query, dict) else None,
                ),
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: judge API call failed, "
                f"skipping this query. Error: {truncate_error_message(e)}",
                level="ERROR",
            )
            self._record_api_failure(
                "judge",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

    def _record_batch_api_failure(
        self,
        query,
        error_message: str,
        **context: Any,
    ) -> None:
        failure_duration_seconds = context.pop("failure_duration_seconds", None)
        self._log(
            f"    [API ERROR] {query.query_id}: {error_message}; "
            "skipping this query.",
            level="ERROR",
        )
        self._record_api_failure(
            "batch_query",
            VertexBatchError(error_message),
            query_id=query.query_id,
            query_type=query.query_type,
            failure_duration_seconds=failure_duration_seconds,
            **context,
        )

    def _score_agent_response(
        self,
        query,
        response: Any,
        context_id: Optional[int] = None,
        memory_construction_time: float = 0.0,
        artifact_references: Optional[Dict[str, Any]] = None,
        execution_usage: Optional[Dict[str, Any]] = None,
    ) -> Optional[MetricResult]:
        """Score a normal or batch-finalized agent response with unchanged metrics."""
        answers_data = query.answers_data if isinstance(query, MedQuery) else None

        if isinstance(response, dict):
            model_output = response.get("output", "")
            query_time = response.get("query_time", 0.0)
            retrieved_memories = response.get("retrieved_memories", [])
            retrieved_count = response.get("retrieved_count", 0)
            response_usage = response.get("answer_usage")
            response_extra = response.get("extra", {})
        elif hasattr(response, "output"):
            model_output = response.output
            query_time = getattr(response, "query_time", 0.0)
            retrieved_memories = getattr(response, "retrieved_memories", [])
            retrieved_count = getattr(response, "retrieved_count", 0)
            response_usage = None
            response_extra = getattr(response, "extra", {})
        else:
            model_output = str(response)
            query_time = 0.0
            retrieved_memories = []
            retrieved_count = 0
            response_usage = None
            response_extra = {}

        resolved_execution_usage = copy.deepcopy(execution_usage or {})
        if isinstance(response_usage, dict) and "answer" not in resolved_execution_usage:
            resolved_execution_usage["answer"] = copy.deepcopy(response_usage)

        metric_kwargs = {
            "answers_data": answers_data,
            "metadata": query.metadata,
        }
        prepared_metric = None
        judge_client = None
        if self.batch_api and not self.dry_run:
            prepared_metric = self.metrics_calculator.prepare_batch(
                query_id=query.query_id,
                query_type=query.query_type,
                model_output=model_output,
                expected_answers=query.get_correct_answers(),
                question=query.question,
                **metric_kwargs,
            )
            if prepared_metric is not None:
                judge_client = self.metrics_calculator.get_batch_judge_client(query.query_type)

        if prepared_metric is not None and judge_client is not None:
            judge_payload = prepared_metric["prepared"]["judge_payload"]
            if "immediate" not in judge_payload:
                deferred_references = copy.deepcopy(artifact_references or {})
                deferred_references["retrieval"] = {
                    "payload_sha256": self._artifact_content_hash(retrieved_memories),
                    "memory_ids_sha256": self._artifact_content_hash([
                        memory.get("memory_id", memory.get("id"))
                        for memory in retrieved_memories
                        if isinstance(memory, dict)
                    ]),
                }
                deferred_judge = {
                    "query_id": query.query_id,
                    "persona_id": context_id,
                    "prepared_metric": prepared_metric,
                    "expected_answers": query.get_correct_answers(),
                    "metric_kwargs": metric_kwargs,
                    "query_time": query_time,
                    "question": query.question,
                    "model_output": model_output,
                    "response_extra": copy.deepcopy(response_extra) if isinstance(response_extra, dict) else {},
                    "memory_construction_time": memory_construction_time,
                    "retrieved_memories": retrieved_memories,
                    "retrieved_count": retrieved_count,
                    "retrieval_quality": self._session_retrieval_quality(
                        query, retrieved_memories, context_id
                    ),
                    "artifact_references": deferred_references,
                    "execution_usage": resolved_execution_usage,
                }
                if isinstance(response_extra, dict) and isinstance(response_extra.get("planner"), dict):
                    deferred_judge["planner"] = copy.deepcopy(response_extra["planner"])
                self._deferred_judges.append(deferred_judge)
                return None
            result = self.metrics_calculator.finalize_batch(prepared_metric, "")
        else:
            if prepared_metric is not None and not self._judge_batch_fallback_logged:
                self._log(
                    "Batch API cannot batch the configured LLM judge; "
                    "using its existing real-time client.",
                    level="WARNING",
                )
                self._judge_batch_fallback_logged = True
            tracker = get_usage_tracker()
            tracker.set_phase("judge")
            with tracker.scope("judge.realtime"):
                result = self.metrics_calculator.compute(
                    query_id=query.query_id,
                    query_type=query.query_type,
                    model_output=model_output,
                    expected_answers=query.get_correct_answers(),
                    question=query.question,
                    **metric_kwargs,
                )

        result.query_time = query_time
        result.memory_construction_time = memory_construction_time
        result.retrieved_memories = retrieved_memories
        result.retrieved_count = retrieved_count
        references = copy.deepcopy(artifact_references or {})
        references["retrieval"] = {
            "payload_sha256": self._artifact_content_hash(retrieved_memories),
            "memory_ids_sha256": self._artifact_content_hash([
                memory.get("memory_id", memory.get("id"))
                for memory in retrieved_memories
                if isinstance(memory, dict)
            ]),
        }
        answer_reference = references.get("answer")
        if isinstance(answer_reference, dict):
            answer_reference["response_sha256"] = self._artifact_content_hash(
                model_output
            )
        references.setdefault("judge", {"transport": "realtime"})
        self._attach_session_retrieval_quality(
            result,
            self._session_retrieval_quality(query, retrieved_memories, context_id),
        )
        result.details["artifact_references"] = references
        result.details["execution_usage"] = resolved_execution_usage
        if isinstance(response_extra, dict) and isinstance(response_extra.get("planner"), dict):
            result.details["planner"] = copy.deepcopy(response_extra["planner"])

        return result

    def _complete_deferred_judges(self) -> List[Dict[str, Any]]:
        """Submit all independent MedMemoryBench judge prompts as one final stage."""
        if not self._deferred_judges:
            return []

        first_prepared = self._deferred_judges[0]["prepared_metric"]["prepared"]
        judge_client = self.metrics_calculator.get_batch_judge_client(first_prepared["query_type"])
        if judge_client is None:
            raise VertexBatchError(
                "Deferred LLM-judge work requires the original Gemini judge configuration."
            )
        # This is written before the remote submission.  If the process exits
        # afterward, resume can collect the same job without regenerating answers.
        self._save_deferred_judges(judge_client)
        requests: List[BatchChatRequest] = []
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in self._deferred_judges:
            payload = item["prepared_metric"]["prepared"]["judge_payload"]
            request_id = make_request_id(
                "judge",
                f"{self.method_config.method_name}:{item['persona_id']}:{item['query_id']}",
            )
            requests.append(
                BatchChatRequest(
                    request_id=request_id,
                    messages=[{"role": "user", "content": payload["prompt"]}],
                    temperature=payload.get(
                        "temperature", getattr(get_api_config(), "judge_temperature", 1.0)
                    ),
                    max_tokens=payload["max_tokens"],
                    reasoning_effort=payload.get("reasoning_effort"),
                    response_format={"type": "json_object"},
                    phase="judge",
                    metadata={"query_id": item["query_id"], "phase": "judge"},
                )
            )
            by_id[request_id] = item

        self._log(
            f"[Batch] Stage 'judge-final': prepared {len(requests):,} LLM-judge request(s)."
        )
        judge_batch_client = self._get_judge_batch_client(judge_client)
        tracker = get_usage_tracker()
        tracker.set_phase("judge")
        with tracker.scope("judge.batch"):
            responses = judge_batch_client.run_stage("judge-final", requests)
        finalized: List[Dict[str, Any]] = []
        for request_id, item in by_id.items():
            response = responses.get(request_id)
            prepared = item["prepared_metric"]["prepared"]
            if response is None or response.status:
                error = response.status if response else "No output row returned"
                self._log(
                    f"[API ERROR] Batch judge failed for {item['query_id']}; "
                    f"no metric result recorded. Error: {truncate_error_message(error)}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "batch_judge",
                    VertexBatchError(truncate_error_message(error)),
                    failure_duration_seconds=(
                        getattr(response, "duration_seconds", 0.0)
                        if response is not None else 0.0
                    ),
                    query_id=item["query_id"],
                    query_type=prepared["query_type"],
                    context_id=item["persona_id"],
                )
                finalized.append({"persona_id": item["persona_id"], "result": self._make_unscored_execution_result(item, error)})
                self._advance_query_progress(
                    item["query_id"],
                    context_id=item["persona_id"],
                )
                continue
            try:
                result = self._run_api_call(
                    self.metrics_calculator.finalize_batch,
                    item["prepared_metric"],
                    response.content,
                )
            except LLMAPIError as e:
                self._log(
                    f"[API ERROR] Batch judge returned an unusable response for "
                    f"{item['query_id']}; no metric result recorded. Error: {truncate_error_message(e)}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "batch_judge",
                    e,
                    query_id=item["query_id"],
                    query_type=prepared["query_type"],
                    context_id=item["persona_id"],
                )
                finalized.append({"persona_id": item["persona_id"], "result": self._make_unscored_execution_result(item, str(e))})
                self._advance_query_progress(
                    item["query_id"],
                    context_id=item["persona_id"],
                )
                continue
            result.query_time = item["query_time"]
            result.memory_construction_time = item["memory_construction_time"]
            result.retrieved_memories = item["retrieved_memories"]
            result.retrieved_count = item["retrieved_count"]
            self._attach_session_retrieval_quality(
                result,
                item.get("retrieval_quality"),
            )
            references = copy.deepcopy(item.get("artifact_references") or {})
            references["judge"] = self._batch_artifact_reference(
                judge_batch_client, request_id
            )
            references["judge"]["prompt_sha256"] = self._artifact_content_hash(
                payload["prompt"]
            )
            references["judge"]["response_sha256"] = self._artifact_content_hash(
                response.content
            )
            result.details["artifact_references"] = references
            execution_usage = copy.deepcopy(item.get("execution_usage") or {})
            execution_usage["judge"] = self._batch_response_usage(
                response, transport="batch"
            )
            result.details["execution_usage"] = execution_usage
            if isinstance(item.get("planner"), dict):
                result.details["planner"] = copy.deepcopy(item["planner"])
            finalized.append({"persona_id": item["persona_id"], "result": result})
            self._advance_query_progress(
                item["query_id"],
                context_id=item["persona_id"],
            )
        self._clear_deferred_judges()
        return finalized

    def _make_unscored_execution_result(self, item: Dict[str, Any], judge_error: str) -> MetricResult:
        """Preserve a completed answer/retrieval when only the judge failed."""
        details: Dict[str, Any] = {
            "evaluation_status": "judge_failed",
            "judge_failure": {"error": truncate_error_message(judge_error)},
            "artifact_references": copy.deepcopy(item.get("artifact_references") or {}),
            "execution_usage": copy.deepcopy(item.get("execution_usage") or {}),
        }
        result = MetricResult(
            query_id=item["query_id"], query_type=item["prepared_metric"]["prepared"]["query_type"],
            score=None, is_correct=None, model_output=item.get("model_output", ""),
            expected_answer="\n".join(item.get("expected_answers", [])), question=item.get("question", ""),
            details=details, query_time=item.get("query_time", 0.0), memory_construction_time=item.get("memory_construction_time", 0.0),
            retrieved_memories=item.get("retrieved_memories", []), retrieved_count=item.get("retrieved_count", 0),
        )
        self._attach_session_retrieval_quality(result, item.get("retrieval_quality"))
        if isinstance(item.get("planner"), dict):
            result.details["planner"] = copy.deepcopy(item["planner"])
        return result

    def _build_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        summary = self.aggregator.get_summary()

        memory_build_summary = self._summarize_memory_builds()
        build_metrics = self._build_metrics_report()

        llm_usage = get_usage_tracker().get_stats()
        stage_usage = self._stage_usage_report(llm_usage)
        retry_failure_duration = self._retry_failure_duration(llm_usage)
        failed_query_ids = set()
        for failure in self._api_failures:
            if failure.get("query_id"):
                failed_query_ids.add(failure["query_id"])
            failed_query_ids.update(failure.get("affected_query_ids", []))
        expected_queries = self._expected_query_count()
        executed_queries = len(self.aggregator.results)
        scored_queries = sum(
            1 for result in self.aggregator.results
            if result.score is not None and result.is_correct is not None
        )
        retrieval_evaluated_queries = sum(
            1 for result in self.aggregator.results
            if isinstance(result.details.get("metric_groups"), dict)
            and RETRIEVAL_QUALITY_GROUP in result.details.get("metric_groups", {})
        )
        evaluation_coverage = {
            "expected_queries": expected_queries,
            "scored_queries": scored_queries,
            "executed_queries": executed_queries,
            "query_execution_coverage": executed_queries / expected_queries if expected_queries else 1.0,
            "retrieval_evaluation_queries": retrieval_evaluated_queries,
            "retrieval_evaluation_coverage": retrieval_evaluated_queries / expected_queries if expected_queries else 1.0,
            "omitted_queries": max(expected_queries - scored_queries, 0),
            "api_failed_queries": len(failed_query_ids),
            "api_failure_events": len(self._api_failures),
            "coverage": (
                scored_queries / expected_queries if expected_queries > 0 else 1.0
            ),
            "complete": scored_queries == expected_queries,
        }

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
                "judge_config": {
                    "provider": get_api_config().get_judge_provider(),
                    "model": get_api_config().get_judge_model(),
                    "temperature": getattr(get_api_config(), "judge_temperature", 1.0),
                    "reasoning_effort": getattr(
                        get_api_config(), "judge_reasoning_effort", None
                    ),
                    "client_max_tokens": getattr(
                        get_api_config(), "judge_client_max_tokens", 10000
                    ),
                    "max_tokens": getattr(get_api_config(), "judge_max_tokens", 500),
                    "mcd_max_tokens": getattr(
                        get_api_config(), "judge_mcd_max_tokens", 2000
                    ),
                },
                "dry_run": self.dry_run,
                "prompt_protocol": self.prompt_protocol,
            },
            metadata={
                "prompt_protocol": self.prompt_protocol,
                "query_type_aware_prompting": self.prompt_protocol == "type_aware",
                "evaluation_mode": self.dataset_config.evaluation_mode,
                "evaluation_interval": self.dataset_config.evaluation_interval,
                "total_personas": len(self.result_collector.get_context_ids()),
                "memory_build_summary": memory_build_summary,
                "build_metrics": build_metrics,
                "memory_size": build_metrics["memory_size"],
                "feature_configuration": build_metrics["feature_configuration"],
                "llm_usage": llm_usage,
                "stage_usage": stage_usage,
                "evaluation_coverage": evaluation_coverage,
                "api_failures": self._api_failures,
                "api_failure_duration_seconds": retry_failure_duration,
            }
        )

        return report

    def _persist_memory_build_checkpoint(self, start_time: datetime) -> None:
        """Preserve completed memory telemetry when a later stage aborts the run."""
        build_logs = self._combined_memory_build_logs()
        if (
            getattr(self, "_memory_build_checkpoint_saved", False)
            or self.execution_stage == "query"
            or not build_logs
        ):
            return
        end_time = datetime.now()
        report = self._build_report(
            start_time,
            end_time,
            (end_time - start_time).total_seconds(),
        )
        report.metadata["memory_build_artifact_status"] = "checkpoint"
        _, memory_build_path, _ = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=build_logs,
            include_result=False,
            include_memory_build=True,
            include_query_answer=False,
            include_api_failures=False,
            use_method_subdir=not getattr(self, "run_scoped_output", False),
        )
        self._memory_build_checkpoint_saved = True
        self._log(
            f"Memory build checkpoint saved to: {memory_build_path}",
            terminal=False,
        )

    def _generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        report = self._build_report(start_time, end_time, duration)
        execution_stage = getattr(self, "execution_stage", "all")
        result_path, memory_build_path, query_answer_path = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=self._combined_memory_build_logs(),
            include_result=execution_stage in {"all", "query"},
            include_memory_build=execution_stage in {"all", "memory"},
            include_query_answer=execution_stage in {"all", "query"},
            include_api_failures=True,
            use_method_subdir=not getattr(self, "run_scoped_output", False),
        )

        if result_path is not None:
            self._log(f"Results saved to: {result_path}", terminal=False)
        if memory_build_path is not None:
            self._log(
                f"Memory build details saved to: {memory_build_path}",
                terminal=False,
            )
        if query_answer_path is not None:
            self._log(
                f"Query answer details saved to: {query_answer_path}",
                terminal=False,
            )
        if self.result_collector.last_api_failure_path is not None:
            self._log(
                f"API failure details saved separately to: "
                f"{self.result_collector.last_api_failure_path}"
            )

        return report

    def _retry_failure_duration(self, llm_usage: Dict[str, Any]) -> float:
        """Return recovered retry time plus terminal API failure time."""
        recovered_retry_duration = float(
            llm_usage.get("total", {}).get("failure_duration_seconds", 0.0) or 0.0
        )
        return recovered_retry_duration + self._api_failure_duration_seconds

    def _expected_query_count(self) -> int:
        """Return coverage denominator for the units selected by this run."""
        units = self._get_evaluation_units()
        if getattr(self, "append", False) and self._append_target_index is not None:
            units = units[: self._append_target_index + 1]
            return sum(len(unit.queries_to_evaluate) for unit in units)
        if self.execution_stage == "query" and self._memory_unit_ids is not None:
            units = [unit for unit in units if unit.unit_id in self._memory_unit_ids]
        return sum(len(unit.queries_to_evaluate) for unit in units)

    def _summarize_memory_builds(self) -> Dict[str, Any]:
        build_logs = self._combined_memory_build_logs()
        if not build_logs:
            return {"total_builds": 0}

        total_units = len(build_logs)

        total_sessions = sum(
            log.get("session_count", 0)
            for log in build_logs
        )

        # New structure: sum time from each unit's total_time
        total_time = sum(
            log.get("total_time", 0)
            for log in build_logs
        )

        # Count total passages from session_builds
        total_passages = sum(
            log.get("total_passages", 0)
            for log in build_logs
        )

        # Count memory entries from session builds
        total_memory_entries = 0
        methods = {}

        for log in build_logs:
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
    force_resume: bool = False,
    execution_stage: str = "all",
    memory_run: Optional[str] = None,
    append: bool = False,
    append_persona: Optional[int] = None,
    append_unit: Optional[int] = None,
    memory_source_run_dir: Optional[Path] = None,
    memory_run_explicit: bool = False,
    run_scoped_output: bool = False,
    batch_api: bool = False,
    batch_gcs_uri: Optional[str] = None,
    batch_wait: bool = False,
    workers: int = 1,
    query_workers: int = 5,
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
        force_resume=force_resume,
        execution_stage=execution_stage,
        memory_run=memory_run,
        append=append,
        append_persona=append_persona,
        append_unit=append_unit,
        memory_source_run_dir=memory_source_run_dir,
        memory_run_explicit=memory_run_explicit,
        run_scoped_output=run_scoped_output,
        batch_api=batch_api,
        batch_gcs_uri=batch_gcs_uri,
        batch_wait=batch_wait,
        workers=workers,
        query_workers=query_workers,
    )
    return evaluator.evaluate()
