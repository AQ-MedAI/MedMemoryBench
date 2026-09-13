"""LoCoMo evaluation module."""

import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple
import logging

from tqdm.auto import tqdm

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager
from src.result import EvaluationReport, ResultCollector
from benchmarks.locomo.dataset import LoCoMoDataset, LoCoMoQuery, LoCoMoSession
from benchmarks.base import EvaluationUnit
from benchmarks.snapshot_artifacts import (
    cleanup_memory_state_embedding_artifacts,
    fsync_directory,
    load_memory_state_embedding_artifacts,
    memory_state_embedding_artifact_paths,
    publish_memory_state_embedding_artifacts,
)
from methods.base import MemoryBuildResult
from methods.event_state.store import EventStateStore
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from metrics.retrieval_quality import compute_session_retrieval_quality
from utils.templates import get_prompt_manager
from utils.logger import truncate_error_message
from utils.batch_client import create_batch_client
from utils.llm_client import LLMResponse, get_usage_tracker
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
    compute_build_config_hash,
    compute_memory_query_compatibility_hash,
    compute_query_config_hash,
    is_manifest_build_compatible,
    is_manifest_query_compatible,
)


# Default chunk size for memory injection (in characters)
# ~32K chars ≈ 8K tokens, safe for GPT-5.1/Qwen3-235B (128K context)
DEFAULT_MEMORY_CHUNK_SIZE = 32000

# A LoCoMo result can retain substantial retrieval diagnostics. Writing the
# entire resumable checkpoint for every completed answer is needlessly costly.
LOCOMO_QUERY_CHECKPOINT_FLUSH_INTERVAL = 25
LOCOMO_RESULT_JOURNAL_VERSION = 1


class LoCoMoEvaluator:

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Path,
        dry_run: bool = False,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
        resume: bool = False,
        execution_stage: str = "all",
        memory_run: Optional[str] = None,
        memory_source_run_dir: Optional[Path] = None,
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
        if execution_stage not in {"all", "memory", "query"}:
            raise ValueError(f"Unsupported execution stage: {execution_stage}")
        self.execution_stage = execution_stage
        self.memory_run = memory_run
        self.memory_source_run_dir = (
            Path(memory_source_run_dir).resolve()
            if memory_source_run_dir is not None else None
        )
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
        )
        self.prompt_protocol = dataset_config.prompt_protocol

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[LoCoMoDataset] = None

        api_config = get_api_config()
        self.metrics_calculator = MetricsCalculator(
            dataset="locomo",
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            judge_temperature=getattr(api_config, "judge_temperature", 1.0),
            judge_reasoning_effort=getattr(api_config, "judge_reasoning_effort", None),
            judge_client_max_tokens=getattr(api_config, "judge_client_max_tokens", 10000),
            judge_max_tokens=getattr(api_config, "judge_max_tokens", 500),
            judge_mcd_max_tokens=getattr(api_config, "judge_mcd_max_tokens", 2000),
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []
        self._batch_client: Optional[VertexBatchClient] = None
        self._batch_fallback_logged = False
        self._pending_batch_queries: List[Dict[str, Any]] = []
        self._pending_query_plan_requests: List[Dict[str, Any]] = []
        self._memory_snapshot_manifest: Optional[Dict[str, Any]] = None
        self._memory_snapshot_dir_path: Optional[Path] = None
        self._query_checkpoint: Dict[str, Any] = {
            "results": {},
            "result_journal": {
                "version": LOCOMO_RESULT_JOURNAL_VERSION,
                "path": "locomo_query_results.jsonl",
            },
        }
        self._query_checkpoint_pending_writes = 0
        self._query_checkpoint_pending_records: List[Dict[str, Any]] = []
        self._recovered_batch_usage_query_ids: set[Tuple[str, str]] = set()
        self._batch_retrieval_preparation_wall_time = 0.0
        self._batch_manager_creation_count = 0
        self._batch_memory_import_count = 0
        self._compiler_cache_hits = 0
        self._compiler_cache_misses = 0
        self._compiler_calls_this_run = 0
        self._memory_build_checkpoint_saved = False

        # Memory chunk configuration
        # Get from dataset config or use default
        eval_config = dataset_config.raw_config.get("evaluation", {})
        self.memory_chunk_size = eval_config.get("memory_chunk_size", DEFAULT_MEMORY_CHUNK_SIZE)

    def _query_worker_count(self) -> int:
        """Use the query-specific limit, retaining compatibility with test fixtures."""
        return max(1, int(getattr(self, "query_workers", getattr(self, "workers", 1))))

    @staticmethod
    def _available_memory_mb() -> Optional[float]:
        """Best-effort available memory without making psutil a dependency."""
        try:
            import psutil  # type: ignore
            return float(psutil.virtual_memory().available) / (1024 * 1024)
        except Exception:
            pass
        try:
            values = {}
            with open("/proc/meminfo", "r", encoding="ascii") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0])
            return float(values["MemAvailable"]) / 1024.0
        except Exception:
            return None

    def _event_state_query_execution(self, requested: int) -> Dict[str, Any]:
        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
        mode = retrieval.get("query_manager_reuse_mode", "auto")
        available = self._available_memory_mb()
        guard = bool(retrieval.get("query_memory_guard_enabled", True))
        reserve = max(0.0, float(retrieval.get("query_memory_reserve_mb", 1024)))
        minimum = max(0.0, float(retrieval.get("query_memory_min_available_mb", 512)))
        fraction = float(retrieval.get("query_memory_max_fraction", .5))
        estimate = retrieval.get("query_worker_memory_estimate_mb")
        estimate = float(estimate) if estimate is not None else max(128.0, reserve * .25)
        effective, reason = max(1, requested), None
        resolved = "shared" if mode == "auto" else mode
        if resolved == "worker_local" and guard:
            if available is None:
                effective, reason = min(effective, 1), "memory_telemetry_unavailable"
            else:
                usable = max(0.0, min(available - reserve, available * fraction))
                cap = max(1, int(usable // max(1.0, estimate)))
                effective = min(effective, cap)
                if effective < requested:
                    reason = "available_memory_budget"
                if available < minimum:
                    effective, reason = 1, "below_minimum_available_memory"
        return {"requested_query_workers": requested, "effective_query_workers": effective,
                "query_manager_reuse_mode": resolved, "available_memory_before_mb": available,
                "estimated_worker_memory_mb": estimate, "memory_guard_triggered": reason is not None,
                "memory_guard_reason": reason, "imported_manager_count": 0, "reused_query_count": 0}

    def _batch_config_hash(self) -> str:
        """Bind a resumable batch manifest to the evaluated configuration."""
        payload = {
            "method": self.method_config.raw_config,
            "dataset": self.dataset_config.raw_config,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _query_compiler_cache_config(self) -> Dict[str, Any]:
        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
        enabled = bool(retrieval.get("query_compiler_plan_cache_enabled", True))
        configured_path = retrieval.get("query_compiler_plan_cache_path")
        path = Path(configured_path) if configured_path else self.output_dir / "query_compiler_plans.jsonl"
        return {"enabled": enabled, "path": path}

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
            # v2 changes the compiler temporal contract.  Cached raw output
            # from v1 is intentionally not assumed equivalent.
            "prompt_version": "event_state_query_compiler_v2", "schema_version": 2,
            "max_searches": retrieval.get("query_compiler_max_searches", 3),
            "temperature": retrieval.get("planner_temperature", 0.0),
        }
        if include_max_tokens:
            payload["query_compiler_max_tokens"] = retrieval.get("query_compiler_max_tokens", 256)
        return payload

    def _query_compiler_cache_fingerprint(self, question: str, reference_time: Optional[str] = None) -> str:
        """Fingerprint compiler inputs only; retrieval ablations deliberately reuse plans."""
        payload = self._query_compiler_cache_fingerprint_payload(question, reference_time)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    def _legacy_query_compiler_cache_entry(
        self, plan_cache: Dict[str, Dict[str, Any]], question: str, reference_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read v2 entries made before token limits joined the fingerprint.

        Those entries omitted the token limit entirely, so only the historical
        default of 256 can be identified without risking reuse under a changed
        compiler limit. New entries always use the complete fingerprint.
        """
        retrieval = (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {})
        if retrieval.get("query_compiler_max_tokens", 256) != 256:
            return None
        payload = self._query_compiler_cache_fingerprint_payload(
            question, reference_time, include_max_tokens=False,
        )
        return plan_cache.get(
            hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
        )

    def _load_query_compiler_plan_cache(self) -> Dict[str, Dict[str, Any]]:
        config = self._query_compiler_cache_config()
        if not config["enabled"] or not config["path"].is_file():
            return {}
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            with config["path"].open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = json.loads(line)
                    if isinstance(entry, dict) and isinstance(entry.get("fingerprint"), str) and isinstance(entry.get("raw_model_output"), str):
                        rows[entry["fingerprint"]] = entry
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._log(f"Ignoring unreadable query compiler plan cache: {exc}", level="WARNING")
        return rows

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

    def _answer_query_kwargs(self, query: LoCoMoQuery) -> Dict[str, Any]:
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

    def _log(self, message: str, level: str = "INFO") -> None:
        if level.upper() in {"ERROR", "WARNING"}:
            message = truncate_error_message(message)
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    @staticmethod
    def _git_metadata() -> Dict[str, Any]:
        """Record local repository identity without making Git a runtime dependency."""
        def run(*args: str) -> Optional[str]:
            try:
                value = subprocess.run(
                    ["git", *args], cwd=PROJECT_ROOT, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
                ).stdout.strip()
                return value or None
            except (OSError, subprocess.CalledProcessError):
                return None
        status = run("status", "--porcelain")
        commit_sha = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        available = status is not None and commit_sha is not None
        return {
            "commit_sha": commit_sha,
            "branch": branch,
            "dirty": None if status is None else bool(status),
            "git_metadata_available": available,
            # Legacy fields remain readable for older result consumers.
            "git_commit_sha": commit_sha,
            "git_branch": branch,
            "git_dirty": None if status is None else bool(status),
        }

    def _init_dataset(self) -> None:
        self._log(f"Loading dataset: {self.dataset_config.dataset_name}")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        self.dataset = LoCoMoDataset(
            data_dir=data_dir,
            config={
                "data_file": self.dataset_config.raw_config.get("data", {}).get("data_file", "locomo10.json"),
                "sample_ids": self.dataset_config.raw_config.get("evaluation", {}).get("sample_ids"),
                "max_samples": self.dataset_config.raw_config.get("evaluation", {}).get("max_samples"),
                "category_filter": self.dataset_config.raw_config.get("evaluation", {}).get("category_filter"),
                "include_images": self.dataset_config.raw_config.get("evaluation", {}).get("include_images", True),
            }
        )
        self.dataset.load()

        self._log(f"  Total Samples: {len(self.dataset.get_sample_ids())}")
        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Total Queries: {self.dataset.get_total_queries()}")
        self._log(f"  Category Distribution: {self.dataset.get_category_distribution()}")
        self._log(f"  Memory Chunk Size: {self.memory_chunk_size:,} chars (~{self.memory_chunk_size // 4:,} tokens)")

    def _init_agent_for_context(self, context_id: Any, force_new: bool = True) -> None:
        if force_new or self.agent_manager is None:
            # Clean up old agent if exists
            if self.agent_manager is not None:
                try:
                    self.agent_manager.reset()
                except Exception as e:
                    self._log(
                        f"Warning: Failed to reset old agent: {truncate_error_message(e)}",
                        level="WARNING",
                    )

            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=self.batch_api,
                batch_gcs_uri=self.batch_gcs_uri,
                batch_wait=self.batch_wait,
                workers=self.workers,
                batch_manifest_dir=self.output_dir / "batch",
                batch_config_hash=self._batch_config_hash(),
                batch_progress_callback=self._log,
            )

        self.agent_manager.set_context_id(context_id)

    def _supports_event_state_snapshots(self) -> bool:
        return bool(
            self.method_config.method_name.lower() == "event_state"
            and self.agent_manager
            and self.agent_manager.supports_memory_snapshots()
        )

    @staticmethod
    def _export_event_state_for_transfer(manager, context_id: Any) -> Dict[str, Any]:
        """Keep derived vectors available when cloning an in-process store."""
        state = manager.export_memory_state(context_id=context_id)
        binary_artifact_exporter = getattr(
            manager, "export_memory_binary_artifacts", None
        )
        if callable(binary_artifact_exporter):
            binary_artifacts = binary_artifact_exporter(context_id=context_id)
            if binary_artifacts:
                state["embedding_artifacts"] = binary_artifacts
        return state

    def _memory_snapshot_root(self) -> Path:
        if self.memory_source_run_dir is not None:
            return self.memory_source_run_dir / "memory"
        return self.output_dir / "memory"

    @staticmethod
    def _snapshot_integrity_hash(payload: Dict[str, Any]) -> str:
        content = dict(payload)
        content.pop("integrity_hash", None)
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)

    def _snapshot_path(self, unit: EvaluationUnit) -> Path:
        if self._memory_snapshot_dir_path is None:
            raise RuntimeError("LoCoMo memory snapshot run has not been selected")
        safe_sample = str(unit.context_id).replace("/", "-").replace("\\", "-")
        return self._memory_snapshot_dir_path / f"sample_{unit.unit_id}_{safe_sample}.json"

    @staticmethod
    def _measure_snapshot_memory_size(
        payload: Dict[str, Any],
        snapshot_path: Path,
    ) -> Dict[str, Any]:
        """Measure one Event-State JSON state plus its dense sidecars."""
        memory_state = payload.get("memory_state") or {}
        if not isinstance(memory_state, dict):
            raise ValueError("LoCoMo snapshot memory state is invalid")
        json_bytes = len(json.dumps(
            memory_state,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"))
        embedding_bytes = sum(
            artifact_path.stat().st_size
            for artifact_path in memory_state_embedding_artifact_paths(
                memory_state, snapshot_path
            )
            if artifact_path.is_file()
        )
        total_bytes = json_bytes + embedding_bytes
        return {
            "measurement": "serialized_memory_state",
            "bytes": total_bytes,
            "mib": round(total_bytes / (1024 ** 2), 6),
            "json_bytes": json_bytes,
            "embedding_bytes": embedding_bytes,
        }

    def _start_memory_snapshot_manifest(self, units: List[EvaluationUnit]) -> None:
        path = self._memory_snapshot_root()
        manifest_path = path / "manifest.json"
        if self.resume and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot resume LoCoMo memory manifest: {exc}") from exc
            if (
                manifest.get("format") != "locomo.event_state_memory_manifest"
                or manifest.get("version") not in {1, 2}
                or not is_manifest_build_compatible(
                    manifest, compute_build_config_hash(self.method_config, self.dataset_config), manifest_path
                )
            ):
                raise ValueError("LoCoMo memory manifest is incompatible with the current Event-State build configuration")
            self._memory_snapshot_manifest = manifest
            self._memory_snapshot_dir_path = path
            return
        path.mkdir(parents=True, exist_ok=False)
        self._memory_snapshot_dir_path = path
        self._memory_snapshot_manifest = {
            "format": "locomo.event_state_memory_manifest",
            "version": 2,
            "build_id": str(uuid.uuid4()),
            "status": "building",
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": compute_build_config_hash(self.method_config, self.dataset_config),
            "build_config_hash": compute_build_config_hash(self.method_config, self.dataset_config),
            "retrieval_config_hash": compute_query_config_hash(self.method_config, self.dataset_config),
            "retrieval_compatibility_hash": compute_memory_query_compatibility_hash(self.method_config, self.dataset_config),
            "sample_ids": [str(unit.context_id) for unit in units],
            "snapshots": [],
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self._write_json_atomic(path / "manifest.json", self._memory_snapshot_manifest)

    def _load_memory_snapshot_manifest(self, units: List[EvaluationUnit]) -> None:
        path = self._memory_snapshot_root()
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(f"Cannot read LoCoMo memory manifest: {manifest_path}") from exc
        if (
            manifest.get("format") != "locomo.event_state_memory_manifest"
            or manifest.get("version") not in {1, 2}
            # A full snapshot set can be safely queried after an answer-stage
            # failure, before the source memory run reaches its final marker.
            or manifest.get("status") not in {"complete", "building"}
            or manifest.get("sample_ids") != [str(unit.context_id) for unit in units]
            or not is_manifest_query_compatible(manifest, self.method_config, self.dataset_config, manifest_path)
        ):
            raise ValueError("LoCoMo Event-State memory manifest is incomplete or incompatible with the query configuration")
        records = manifest.get("snapshots", [])
        if {str(record.get("sample_id")) for record in records} != set(manifest["sample_ids"]):
            raise ValueError("LoCoMo Event-State memory manifest has an incomplete sample snapshot list")
        self._memory_snapshot_manifest = manifest
        self._memory_snapshot_dir_path = path

    def _write_memory_snapshot(self, unit: EvaluationUnit, build_time: float, build_metrics: Dict[str, Any]) -> Path:
        path = self._snapshot_path(unit)
        memory_state = self.agent_manager.export_memory_state(context_id=unit.context_id)
        binary_artifact_exporter = getattr(
            self.agent_manager, "export_memory_binary_artifacts", None
        )
        if callable(binary_artifact_exporter):
            binary_artifacts = binary_artifact_exporter(context_id=unit.context_id)
            if binary_artifacts:
                memory_state["embedding_artifacts"] = binary_artifacts
        payload = {
            "format": "locomo.event_state_memory_snapshot",
            "version": 2,
            "build_id": self._memory_snapshot_manifest["build_id"],
            "sample_id": str(unit.context_id),
            "unit_id": unit.unit_id,
            "session_ids": [session.session_id for session in unit.sessions_to_inject],
            "session_timestamps": [session.date_time for session in unit.sessions_to_inject],
            "memory_build_time": build_time,
            "memory_build_metrics": build_metrics,
            "created_at": datetime.now().isoformat(),
            "memory_state": memory_state,
        }
        publish_memory_state_embedding_artifacts(memory_state, path)
        payload["memory_size"] = self._measure_snapshot_memory_size(payload, path)
        payload["integrity_hash"] = self._snapshot_integrity_hash(payload)
        self._write_json_atomic(path, payload)
        cleanup_memory_state_embedding_artifacts(memory_state, path)
        record = {
            "sample_id": str(unit.context_id), "unit_id": unit.unit_id,
            "path": path.name, "integrity_hash": payload["integrity_hash"],
            "session_count": len(unit.sessions_to_inject), "memory_build_time": build_time,
            "memory_build_metrics": build_metrics,
            "memory_size": payload["memory_size"],
        }
        records = self._memory_snapshot_manifest["snapshots"]
        records[:] = [item for item in records if str(item.get("sample_id")) != str(unit.context_id)]
        records.append(record)
        records.sort(key=lambda item: item["unit_id"])
        for log in reversed(getattr(self, "_memory_build_logs", [])):
            if (
                log.get("unit_id") == unit.unit_id
                and log.get("context_id") == unit.context_id
            ):
                log["memory_size"] = dict(payload["memory_size"])
                break
        self._write_json_atomic(self._memory_snapshot_dir_path / "manifest.json", self._memory_snapshot_manifest)
        return path

    def _read_memory_snapshot(self, unit: EvaluationUnit) -> Dict[str, Any]:
        record = next((item for item in self._memory_snapshot_manifest["snapshots"] if str(item.get("sample_id")) == str(unit.context_id)), None)
        if record is None:
            raise ValueError(f"No LoCoMo snapshot exists for sample {unit.context_id}")
        path = self._memory_snapshot_dir_path / record["path"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read LoCoMo snapshot {path}") from exc
        if (
            payload.get("format") != "locomo.event_state_memory_snapshot"
            or payload.get("version") not in {1, 2}
            or payload.get("build_id") != self._memory_snapshot_manifest.get("build_id")
            or str(payload.get("sample_id")) != str(unit.context_id)
            or payload.get("integrity_hash") != self._snapshot_integrity_hash(payload)
        ):
            raise ValueError(f"LoCoMo snapshot integrity or identity check failed: {path}")
        payload["memory_size"] = self._measure_snapshot_memory_size(payload, path)
        memory_state = payload.get("memory_state")
        if not isinstance(memory_state, dict):
            raise ValueError(f"LoCoMo snapshot state is invalid: {path}")
        load_memory_state_embedding_artifacts(memory_state, path)
        if memory_state.get("schema_version") == EventStateStore.SCHEMA_VERSION:
            EventStateStore.from_export(memory_state)
        return payload

    def _complete_memory_snapshot_manifest(self) -> None:
        expected = self._memory_snapshot_manifest.get("sample_ids", [])
        actual = [str(item.get("sample_id")) for item in self._memory_snapshot_manifest.get("snapshots", [])]
        if sorted(actual) != sorted(expected):
            raise RuntimeError("LoCoMo Event-State memory build is incomplete")
        self._memory_snapshot_manifest["status"] = "complete"
        self._memory_snapshot_manifest["completed_at"] = datetime.now().isoformat()
        self._write_json_atomic(self._memory_snapshot_dir_path / "manifest.json", self._memory_snapshot_manifest)

    def _query_checkpoint_path(self) -> Path:
        return self.output_dir / "locomo_query_checkpoint.json"

    def _query_result_journal_path(self) -> Path:
        return self.output_dir / "locomo_query_results.jsonl"

    @staticmethod
    def _query_result_journal_digest(sample_id: str, result: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"sample_id": sample_id, "result": result},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _uses_query_result_journal(self) -> bool:
        journal = self._query_checkpoint.get("result_journal", {})
        return isinstance(journal, dict) and journal.get("version") == LOCOMO_RESULT_JOURNAL_VERSION

    def _append_query_result_journal(self) -> None:
        records = getattr(self, "_query_checkpoint_pending_records", [])
        if not records:
            return
        path = self._query_result_journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._query_checkpoint_pending_records = []

    def _restore_query_result_journal(self) -> None:
        if not self._uses_query_result_journal():
            return
        path = self._query_result_journal_path()
        if not path.exists():
            return
        results = self._query_checkpoint.setdefault("results", {})
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        sample_id = str(record["sample_id"])
                        result = record["result"]
                        if (
                            not isinstance(result, dict)
                            or record.get("digest") != self._query_result_journal_digest(sample_id, result)
                        ):
                            break
                        query_id = result.get("query_id")
                        if query_id is None:
                            break
                    except (KeyError, TypeError, json.JSONDecodeError):
                        # Preserve all complete fsynced JSONL entries before a torn tail.
                        break
                    results.setdefault(sample_id, {}).setdefault(str(query_id), result)
        except OSError:
            return

    def _load_query_checkpoint(self) -> None:
        if not self.resume or not self._query_checkpoint_path().exists():
            return
        try:
            payload = json.loads(self._query_checkpoint_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read the LoCoMo query resume checkpoint") from exc
        if (
            payload.get("format") != "locomo.query_checkpoint"
            or payload.get("version") not in {1, 2}
            or payload.get("query_config_hash") != compute_query_config_hash(self.method_config, self.dataset_config)
            or payload.get("integrity_hash") != self._snapshot_integrity_hash(payload)
            or not isinstance(payload.get("results"), dict)
        ):
            raise ValueError("LoCoMo query resume checkpoint is incompatible or corrupt")
        self._query_checkpoint = payload
        self._restore_query_result_journal()

    def _completed_query_results(self, unit: EvaluationUnit) -> List[MetricResult]:
        saved = self._query_checkpoint.get("results", {}).get(str(unit.context_id), {})
        if not isinstance(saved, dict):
            return []
        results = [
            MetricResult(**saved[query.query_id])
            for query in unit.queries_to_evaluate
            if query.query_id in saved
        ]
        self._restore_batch_usage_from_checkpoint(unit.context_id, results)
        return results

    def _restore_batch_usage_from_checkpoint(
        self,
        context_id: Any,
        results: List[MetricResult],
    ) -> None:
        """Rehydrate batch usage when a resume skips already-completed queries."""
        recovered_ids = getattr(self, "_recovered_batch_usage_query_ids", None)
        if recovered_ids is None:
            recovered_ids = set()
            self._recovered_batch_usage_query_ids = recovered_ids

        tracker = get_usage_tracker()
        model = getattr(getattr(self, "method_config", None), "model", None)
        model_name = getattr(model, "name", "")
        for result in results:
            details = result.details if isinstance(result.details, dict) else {}
            execution_usage = details.get("execution_usage", {})
            answer_usage = (
                execution_usage.get("answer", {})
                if isinstance(execution_usage, dict) else {}
            )
            if (
                not isinstance(answer_usage, dict)
                or answer_usage.get("transport") != "batch"
            ):
                continue
            key = (str(context_id), str(result.query_id))
            if key in recovered_ids:
                continue
            recovered_ids.add(key)
            tracker.set_phase("query")
            tracker.record(LLMResponse(
                content="",
                input_tokens=answer_usage.get("input_tokens", 0),
                output_tokens=answer_usage.get("output_tokens", 0),
                visible_output_tokens=answer_usage.get("visible_output_tokens"),
                thinking_tokens=answer_usage.get("thinking_tokens", 0),
                model=model_name,
            ))

    @staticmethod
    def _batch_answer_execution_usage(
        batch_response: Any,
    ) -> Dict[str, Any]:
        """Preserve all provider token fields in resumable query results."""
        return {
            "transport": "batch",
            "input_tokens": batch_response.input_tokens,
            "output_tokens": batch_response.output_tokens,
            "visible_output_tokens": batch_response.visible_output_tokens,
            "thinking_tokens": batch_response.thinking_tokens,
        }

    def _pending_query_unit(self, unit: EvaluationUnit) -> EvaluationUnit:
        saved = self._query_checkpoint.get("results", {}).get(str(unit.context_id), {})
        pending = [query for query in unit.queries_to_evaluate if query.query_id not in saved]
        return replace(unit, queries_to_evaluate=pending)

    def _record_completed_query(self, sample_id: Any, result: MetricResult) -> None:
        results = self._query_checkpoint.setdefault("results", {}).setdefault(str(sample_id), {})
        if result.query_id in results:
            return
        results[result.query_id] = result.to_dict()
        if self._uses_query_result_journal():
            result_dict = results[result.query_id]
            sample_key = str(sample_id)
            self._query_checkpoint_pending_records.append({
                "sample_id": sample_key,
                "result": result_dict,
                "digest": self._query_result_journal_digest(sample_key, result_dict),
            })
        self._query_checkpoint_pending_writes = (
            getattr(self, "_query_checkpoint_pending_writes", 0) + 1
        )
        self._flush_query_checkpoint()

    def _flush_query_checkpoint(self, *, force: bool = False) -> None:
        """Durably save newly completed queries in bounded batches."""
        pending_writes = getattr(self, "_query_checkpoint_pending_writes", 0)
        if not pending_writes or (
            not force and pending_writes < LOCOMO_QUERY_CHECKPOINT_FLUSH_INTERVAL
        ):
            return
        self._append_query_result_journal()
        self._query_checkpoint.update({
            "format": "locomo.query_checkpoint",
            "version": 2,
            "query_config_hash": compute_query_config_hash(self.method_config, self.dataset_config),
        })
        payload = dict(self._query_checkpoint)
        if self._uses_query_result_journal():
            payload["results"] = {}
        payload["integrity_hash"] = self._snapshot_integrity_hash(payload)
        self._query_checkpoint["integrity_hash"] = payload["integrity_hash"]
        self._write_json_atomic(self._query_checkpoint_path(), payload)
        self._query_checkpoint_pending_writes = 0

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()

        get_usage_tracker().reset()

        try:
            self._init_dataset()
            self._load_query_checkpoint()
            units = list(self.dataset.get_evaluation_units())
            if self.execution_stage != "all" and self.method_config.method_name.lower() != "event_state":
                raise ValueError("LoCoMo --stage memory/query requires Event-State memory snapshots")
            if self.method_config.method_name.lower() == "event_state" and not self.dry_run:
                if self.execution_stage == "query":
                    self._load_memory_snapshot_manifest(units)
                else:
                    self._start_memory_snapshot_manifest(units)

            self._run_evaluation_loop(units)
            self._flush_query_checkpoint(force=True)
            if (
                self.method_config.method_name.lower() == "event_state"
                and not self.dry_run
                and self.execution_stage != "query"
            ):
                self._complete_memory_snapshot_manifest()
        except BaseException:
            try:
                self._flush_query_checkpoint(force=True)
            except Exception as checkpoint_error:
                self._log(
                    "Unable to save LoCoMo query checkpoint: "
                    f"{truncate_error_message(checkpoint_error)}",
                    level="WARNING",
                )
            try:
                self._persist_memory_build_checkpoint(start_time)
            except Exception as checkpoint_error:
                self._log(
                    "Unable to save memory build checkpoint: "
                    f"{truncate_error_message(checkpoint_error)}",
                    level="WARNING",
                )
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        gc.collect()
        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self, units: List[EvaluationUnit]) -> None:
        if self.execution_stage == "query":
            self._run_snapshot_queries(units)
            return
        for unit in units:
            sample_id = unit.context_id

            self._log(f"\n{'='*60}")
            self._log(f"Sample: {sample_id}")
            self._log(f"  Sessions: {len(unit.sessions_to_inject)}")
            self._log(f"  Queries: {len(unit.queries_to_evaluate)}")

            for result in self._completed_query_results(unit):
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, sample_id)
            unit = self._pending_query_unit(unit)

            if not self.dry_run:
                self._init_agent_for_context(context_id=sample_id, force_new=True)

            unit_results = self._evaluate_unit(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, sample_id)
                self._record_completed_query(sample_id, result)

        for item in self._complete_combined_batch_queries():
            result = item["result"]
            sample_id = item["sample_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, sample_id)
            self._record_completed_query(sample_id, result)

    def _run_snapshot_queries(self, units: List[EvaluationUnit]) -> None:
        """Restore each complete conversation into an isolated Event-State store."""
        if self.batch_api:
            # The run-wide manifest stays single-writer, while each prompt is
            # frozen immediately after restoring its own sample snapshot.
            for unit in units:
                for result in self._completed_query_results(unit):
                    self.aggregator.add_result(result)
                    self.result_collector.add_result(result, unit.context_id)
                unit = self._pending_query_unit(unit)
                self._init_agent_for_context(unit.context_id, force_new=True)
                self.agent_manager.import_memory_state(
                    self._read_memory_snapshot(unit)["memory_state"], context_id=unit.context_id
                )
                for result in self._evaluate_unit_queries(unit, memory_time=0.0):
                    self.aggregator.add_result(result)
                    self.result_collector.add_result(result, unit.context_id)
                    self._record_completed_query(unit.context_id, result)
            for item in self._complete_combined_batch_queries():
                self.aggregator.add_result(item["result"])
                self.result_collector.add_result(item["result"], item["sample_id"])
                self._record_completed_query(item["sample_id"], item["result"])
            return

        completed_prior = []
        pending_units = []
        for unit in units:
            completed_prior.extend((unit.context_id, result) for result in self._completed_query_results(unit))
            pending_units.append(self._pending_query_unit(unit))
        for sample_id, result in completed_prior:
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, sample_id)

        units_with_queries = [u for u in pending_units if u.queries_to_evaluate]
        query_jobs = [
            (unit, query)
            for unit in units_with_queries
            for query in unit.queries_to_evaluate
        ]
        if not query_jobs:
            return

        query_worker_count = min(self._query_worker_count(), len(query_jobs))
        if len(units_with_queries) > 1:
            self._log(
                f"  [Workers] Running {len(query_jobs):,} real-time queries across "
                f"{len(units_with_queries):,} query units with {query_worker_count} "
                "workers total."
            )
        else:
            self._log(
                f"  [Workers] Running {len(query_jobs):,} real-time queries with "
                f"{query_worker_count} workers."
            )

        unit_states: Dict[Any, Dict[str, Any]] = {}
        unit_state_lock = threading.Lock()

        def get_unit_state(unit: EvaluationUnit) -> Any:
            with unit_state_lock:
                state = unit_states.get(unit.context_id)
                if state is None:
                    if hasattr(self, "_read_memory_snapshot"):
                        payload = self._read_memory_snapshot(unit)
                        state = payload.get("memory_state", payload) if isinstance(payload, dict) else payload
                    else:
                        state = {}
                    unit_states[unit.context_id] = state
                return state

        worker_local = threading.local()

        def get_worker_manager(unit: EvaluationUnit) -> Optional[AgentManager]:
            if not hasattr(worker_local, "managers"):
                worker_local.managers = {}
            manager = worker_local.managers.get(unit.context_id)
            if manager is None:
                state = get_unit_state(unit)
                try:
                    manager = AgentManager(
                        method_config=getattr(self, "method_config", None),
                        dataset_config=getattr(self, "dataset_config", None),
                        batch_api=False,
                        workers=1,
                    )
                    if hasattr(manager, "import_memory_state") and isinstance(state, dict) and state:
                        manager.import_memory_state(
                            state,
                            context_id=unit.context_id,
                        )
                except Exception:
                    manager = getattr(self, "agent_manager", None)
                worker_local.managers[unit.context_id] = manager
            return manager

        def evaluate_query_job(job: Tuple[EvaluationUnit, LoCoMoQuery]) -> Tuple[EvaluationUnit, MetricResult]:
            unit, query = job
            manager = get_worker_manager(unit)
            try:
                result = self._evaluate_query(query, unit.context_id, manager=manager)
            except TypeError:
                result = self._evaluate_query(query, unit.context_id)
            if hasattr(result, "memory_construction_time"):
                result.memory_construction_time = 0.0
            return unit, result

        if query_worker_count > 1:
            with ThreadPoolExecutor(max_workers=query_worker_count) as executor:
                for unit, result in executor.map(evaluate_query_job, query_jobs):
                    if hasattr(self, "aggregator") and hasattr(self.aggregator, "add_result"):
                        self.aggregator.add_result(result)
                    if hasattr(self, "result_collector") and hasattr(self.result_collector, "add_result"):
                        self.result_collector.add_result(result, unit.context_id)
                    if hasattr(self, "_record_completed_query"):
                        self._record_completed_query(unit.context_id, result)
        else:
            for job in query_jobs:
                unit, result = evaluate_query_job(job)
                if hasattr(self, "aggregator") and hasattr(self.aggregator, "add_result"):
                    self.aggregator.add_result(result)
                if hasattr(self, "result_collector") and hasattr(self.result_collector, "add_result"):
                    self.result_collector.add_result(result, unit.context_id)
                if hasattr(self, "_record_completed_query"):
                    self._record_completed_query(unit.context_id, result)

    def _split_sessions_into_chunks(
        self,
        sessions: List[LoCoMoSession],
    ) -> List[Tuple[List[LoCoMoSession], str]]:
        """Split sessions into chunks based on character size limit.

        Returns:
            List of tuples: (sessions_in_chunk, combined_memory_text)
        """
        chunks = []
        current_chunk_sessions = []
        current_chunk_texts = []
        current_chunk_size = 0

        for session in sessions:
            memory_text = session.to_memory_text()
            text_size = len(memory_text)

            # If adding this session exceeds the limit and we have content, start a new chunk
            if current_chunk_size + text_size > self.memory_chunk_size and current_chunk_sessions:
                # Save current chunk
                combined_text = "\n\n".join(current_chunk_texts)
                chunks.append((current_chunk_sessions.copy(), combined_text))

                # Start new chunk
                current_chunk_sessions = []
                current_chunk_texts = []
                current_chunk_size = 0

            # Add session to current chunk
            current_chunk_sessions.append(session)
            current_chunk_texts.append(memory_text)
            current_chunk_size += text_size

        # Don't forget the last chunk
        if current_chunk_sessions:
            combined_text = "\n\n".join(current_chunk_texts)
            chunks.append((current_chunk_sessions, combined_text))

        return chunks

    def _evaluate_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        if self.method_config.method_name.lower() == "event_state":
            return self._evaluate_event_state_unit(unit)
        results = []

        self._log(f"  --- Memory Build Phase ---")

        if self.dry_run:
            self._log(f"  [Dry Run] Skipping memory build")
            total_memory_time = 0.0
            chunk_build_results = []
        else:
            # Split sessions into chunks
            chunks = self._split_sessions_into_chunks(unit.sessions_to_inject)
            total_chunks = len(chunks)

            self._log(f"  Split into {total_chunks} chunks (chunk_size={self.memory_chunk_size:,} chars)")

            total_memory_time = 0.0
            chunk_build_results = []
            all_session_ids = []

            speakers = unit.metadata.get("speaker_a", "A") + " and " + unit.metadata.get("speaker_b", "B")

            for chunk_idx, (chunk_sessions, chunk_text) in enumerate(chunks):
                chunk_session_ids = [s.session_id for s in chunk_sessions]
                all_session_ids.extend(chunk_session_ids)

                chunk_chars = len(chunk_text)
                chunk_tokens_est = chunk_chars // 4

                self._log(f"    [Chunk {chunk_idx + 1}/{total_chunks}] "
                         f"Sessions: {chunk_session_ids[0]}-{chunk_session_ids[-1]} "
                         f"({len(chunk_sessions)} sessions, ~{chunk_tokens_est:,} tokens)")

                # Format the chunk text with prompt template
                formatted_text = self.prompt_manager.format_memorize(
                    context=chunk_text,
                    timestamp=speakers,
                )

                chunk_start_time = time.time()

                try:
                    memory_items = []
                    for session_index, session in enumerate(chunk_sessions):
                        for turn in session.dialogues:
                            memory_item = {
                                "speaker": turn.get("speaker", "Unknown"),
                                "content": turn.get("text", ""),
                                "blip_caption": turn.get("blip_caption", ""),
                                "timestamp": session.date_time,
                            }
                            if self.method_config.method_name in {"amem_test", "event_state"}:
                                memory_item.update({
                                    "source_session_id": session.session_id,
                                    "source_session_index": session_index if self.method_config.method_name == "event_state" else session.session_id,
                                    "source_turn_id": turn.get("dia_id"),
                                    "source_event_id": session.metadata.get("session_key"),
                                })
                            memory_items.append(memory_item)
                    is_last_session = (chunk_idx == total_chunks - 1)
                    memory_result = self.agent_manager.send_message(
                        message=formatted_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_session,
                        memory_items=memory_items,
                    )

                    chunk_time = time.time() - chunk_start_time
                    total_memory_time += chunk_time

                    if isinstance(memory_result, MemoryBuildResult):
                        # Log brief info
                        entries_count = len(memory_result.memory_entries) if memory_result.memory_entries else 0
                        chunk_count = memory_result.chunk_count or 0

                        self._log(f"      → Stored: entries={entries_count}, chunks={chunk_count}, "
                                 f"time={chunk_time:.2f}s")

                        # Store detailed result for this chunk
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
                        # Fallback for non-standard result
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
                    self._log(
                        f"      [ERROR] Chunk {chunk_idx + 1} failed: {truncate_error_message(e)}",
                        level="ERROR",
                    )

                    chunk_build_results.append({
                        "chunk_index": chunk_idx,
                        "session_ids": chunk_session_ids,
                        "session_count": len(chunk_sessions),
                        "input_chars": chunk_chars,
                        "input_tokens_est": chunk_tokens_est,
                        "time_cost": chunk_time,
                        "error": truncate_error_message(e),
                    })

            # Calculate summary stats
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

            # Store all chunk build results
            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
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

        return self._evaluate_unit_queries(unit, memory_time=total_memory_time)

    def _event_state_memory_items(self, unit: EvaluationUnit) -> List[Dict[str, Any]]:
        """Keep LoCoMo's source sessions intact, with sample-global chronology."""
        items: List[Dict[str, Any]] = []
        for session_index, session in enumerate(unit.sessions_to_inject):
            for turn in session.dialogues:
                items.append({
                    "speaker": turn.get("speaker", "Unknown"),
                    "content": turn.get("text", ""),
                    "blip_caption": turn.get("blip_caption", ""),
                    # The adapter owns LoCoMo-specific timestamp parsing; the
                    # memory method receives a generic canonical record time.
                    "timestamp": session.metadata.get("recorded_at", session.date_time),
                    "recorded_at_raw": session.metadata.get("recorded_at_raw"),
                    "source_session_id": session.session_id,
                    "source_session_index": session_index,
                    "source_turn_id": turn.get("dia_id"),
                    "source_event_id": session.metadata.get("session_key"),
                })
        return items

    @staticmethod
    def _event_state_store_diagnostics(
        state: Dict[str, Any], input_session_count: int,
    ) -> Dict[str, Any]:
        episodes = state.get("episodes", []) if isinstance(state, dict) else []
        claims = state.get("claims", []) if isinstance(state, dict) else []
        source_ids = [item.get("source_session_id") for item in episodes if item.get("source_session_id") is not None]
        episode_count = len(episodes)
        unique_source_count = len(set(source_ids))
        duplicate_source_count = len(source_ids) - unique_source_count
        return {
            "input_session_count": input_session_count,
            "unique_source_session_id_count": unique_source_count,
            "final_episode_count": episode_count,
            "final_claim_count": len(claims),
            "final_memory_object_count": episode_count + len(claims),
            "duplicate_episode_source_id_count": duplicate_source_count,
            "source_identity_integrity": {
                "expected": input_session_count,
                "actual_unique_source_session_ids": unique_source_count,
                "actual_episodes": episode_count,
                "valid": (
                    input_session_count == unique_source_count == episode_count
                ),
            },
        }

    def _evaluate_event_state_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        if self.dry_run:
            return [] if self.execution_stage == "memory" else self._evaluate_unit_queries(unit, memory_time=0.0)
        existing = None
        if self.resume:
            try:
                existing = self._read_memory_snapshot(unit)
            except ValueError:
                existing = None
        if existing is not None:
            self.agent_manager.import_memory_state(existing["memory_state"], context_id=unit.context_id)
            build_time = float(existing.get("memory_build_time", 0.0) or 0.0)
            snapshot_state = existing.get("memory_state", {})
            build_metrics = dict(existing.get("memory_build_metrics") or {})
            build_metrics.pop("chunk_count", None)
            inserted_record_count = int(
                build_metrics.pop(
                    "inserted_record_count",
                    build_metrics.pop("inserted_count", 0),
                )
                or 0
            )
            build_metrics["inserted_record_count"] = inserted_record_count
            build_metrics.update(self._event_state_store_diagnostics(
                snapshot_state, len(unit.sessions_to_inject)
            ))
            self._memory_build_logs.append({
                "unit_id": unit.unit_id, "context_id": unit.context_id,
                "session_ids": existing.get("session_ids", []),
                "session_count": len(unit.sessions_to_inject),
                "total_time": build_time,
                "inserted_record_count": inserted_record_count,
                "build_metrics": build_metrics, "reporting_kind": "event_state",
                "final_store": self._event_state_store_diagnostics(
                    snapshot_state, len(unit.sessions_to_inject)
                ), "restored_from_snapshot": True,
                "memory_size": dict(existing.get("memory_size") or {}),
            })
        else:
            started = time.time()
            items = self._event_state_memory_items(unit)
            prepared = self.agent_manager.prepare_memory_sessions(
                "", context_id=unit.context_id, memory_items=items
            )
            # Event-State preparation can run concurrently internally; this
            # commit is deliberately single-threaded and source ordered.
            memory_result = self.agent_manager.commit_prepared_memory(
                prepared, context_id=unit.context_id
            )
            build_time = time.time() - started
            if isinstance(memory_result, MemoryBuildResult):
                # Keep compact build telemetry, not a second copy of raw memory.
                build_metrics = {
                    key: value for key, value in memory_result.to_dict().items()
                    if key not in {
                        "memory_entries", "all_passages", "input_content",
                        "stored_content", "extraction_result", "chunk_count",
                    }
                }
                inserted_record_count = int(
                    build_metrics.pop(
                        "inserted_count", len(memory_result.memory_entries or [])
                    )
                    or 0
                )
                build_metrics["inserted_record_count"] = inserted_record_count
            else:
                build_metrics = {"raw_result": str(memory_result)}
                inserted_record_count = 0
            final_store = self._event_state_store_diagnostics(
                self.agent_manager.export_memory_state(context_id=unit.context_id),
                len(unit.sessions_to_inject),
            )
            build_metrics.update(final_store)
            if not final_store["source_identity_integrity"]["valid"]:
                raise RuntimeError(
                    "LoCoMo Event-State ingestion lost or duplicated source sessions: "
                    f"{final_store['source_identity_integrity']}"
                )
            self._memory_build_logs.append({
                "unit_id": unit.unit_id, "context_id": unit.context_id,
                "session_ids": [session.session_id for session in unit.sessions_to_inject],
                "session_count": len(unit.sessions_to_inject),
                "total_time": build_time,
                "inserted_record_count": inserted_record_count,
                "build_metrics": build_metrics, "reporting_kind": "event_state",
                "final_store": final_store, "staged_session_count": len(prepared),
            })
            self._write_memory_snapshot(unit, build_time, build_metrics)
        if self.execution_stage == "memory":
            return []
        return self._evaluate_unit_queries(unit, memory_time=build_time)

    def _evaluate_unit_queries(self, unit: EvaluationUnit, memory_time: float) -> List[MetricResult]:
        self._log("  --- Query Evaluation Phase ---")
        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = memory_time / query_count if query_count else 0.0
        if unit.queries_to_evaluate and self._supports_batch_queries():
            batch_client = self._get_batch_client()
            if not batch_client.has_stage("query-final") and batch_client.has_stage(f"query-unit-{unit.unit_id}"):
                query_results = self._evaluate_batch_queries(unit)
            else:
                self._prepare_combined_batch_queries(unit, memory_time_per_query)
                query_results = []
        else:
            if self.batch_api and not self.dry_run and not self._batch_fallback_logged:
                self._log("Batch API is unavailable for this adapter; using real-time final-answer generation.", level="WARNING")
                self._batch_fallback_logged = True
            query_results = self._evaluate_realtime_queries(unit.queries_to_evaluate, unit.context_id)
        for result in query_results:
            result.memory_construction_time = memory_time_per_query
        return query_results

    def _supports_batch_queries(self) -> bool:
        return bool(
            self.batch_api
            and self.agent_manager
            and self.agent_manager.supports_batch_queries()
        )

    def _evaluate_realtime_queries(
        self,
        queries: List[LoCoMoQuery],
        context_id: Any,
        manager: Optional[AgentManager] = None,
    ) -> List[MetricResult]:
        """Run independent real-time query evaluations with bounded concurrency."""
        explicit_manager = manager is not None
        manager = manager or getattr(self, "agent_manager", None)
        query_workers = self._query_worker_count()
        if len(queries) < 2 or query_workers == 1:
            if not explicit_manager:
                return [self._evaluate_query(query, context_id) for query in queries]
            return [self._evaluate_query(query, context_id, manager=manager) for query in queries]

        worker_count = min(query_workers, len(queries))
        self._log(f"  [Workers] Running {len(queries):,} real-time queries with {worker_count} workers.")
        if getattr(getattr(self, "method_config", None), "method_name", "").lower() == "event_state":
            # Event-State retrieval is logically read-only, but separate stores
            # make that guarantee explicit and keep future adapter changes safe.
            state = self._export_event_state_for_transfer(manager, context_id)

            def evaluate_isolated(query: LoCoMoQuery) -> MetricResult:
                isolated = AgentManager(
                    method_config=self.method_config,
                    dataset_config=self.dataset_config,
                    batch_api=False,
                    workers=1,
                )
                isolated.import_memory_state(state, context_id=context_id)
                return self._evaluate_query(query, context_id, manager=isolated)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                return list(executor.map(evaluate_isolated, queries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves dataset order for deterministic reports.
            if not explicit_manager:
                return list(executor.map(
                    lambda query: self._evaluate_query(query, context_id), queries
                ))
            return list(executor.map(
                # Every worker receives an isolated manager for snapshot-only
                # execution; legacy in-sample calls retain their current agent.
                lambda query: self._evaluate_query(query, context_id, manager=manager),
                queries,
            ))

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
                manifest_path=scoped_manifest_path(
                    self.output_dir / "batch",
                    "locomo_batch_manifest",
                    model=llm_client.model,
                    config_hash=self._batch_config_hash(),
                ),
                wait=self.batch_wait,
                config_hash=self._batch_config_hash(),
                progress_callback=self._log,
                vertex_batch_class=VertexBatchClient,
            )
        return self._batch_client

    def _evaluate_batch_queries(self, unit: EvaluationUnit) -> List[MetricResult]:
        prepared_by_id: Dict[str, tuple[LoCoMoQuery, Dict[str, Any]]] = {}
        requests: List[BatchChatRequest] = []
        stage = f"query-unit-{unit.unit_id}"
        batch_client = self._get_batch_client()
        for query in unit.queries_to_evaluate:
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
            if prepared is None:
                formatted_question = self.prompt_manager.format_query(
                    question=query.question,
                    query_type=query.query_type,
                    prompt_protocol=self.prompt_protocol,
                )
                prepared = self.agent_manager.prepare_batch_query(
                    formatted_question,
                    query_id=query.query_id,
                    context_id=unit.context_id,
                    batch_request_time=batch_request_time,
                    **self._answer_query_kwargs(query),
                )

            if saved_request is not None:
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

        get_saved_requests = getattr(batch_client, "get_saved_requests", None)
        saved_requests = get_saved_requests(stage) if callable(get_saved_requests) else []
        submitted_requests = saved_requests or requests
        responses = batch_client.run_stage(stage, submitted_requests)
        results: List[MetricResult] = []
        for request_id, (query, prepared) in prepared_by_id.items():
            batch_response = responses.get(request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                results.append(
                    self._api_error_result(
                        query, f"Batch request failed: {truncate_error_message(error)}"
                    )
                )
                continue
            response = self.agent_manager.finalize_batch_query(
                prepared,
                batch_response.content,
                input_tokens=batch_response.input_tokens,
                output_tokens=batch_response.output_tokens,
            )
            result = self._score_agent_response(query, response)
            result.details.setdefault("execution_usage", {})["answer"] = (
                self._batch_answer_execution_usage(batch_response)
            )
            results.append(result)
        return results

    def _prepare_combined_batch_queries(
        self,
        unit: EvaluationUnit,
        memory_time_per_query: float,
    ) -> int:
        """Freeze a sample's prompts before its agent is reset for the next sample."""
        stage = "query-final"
        batch_client = self._get_batch_client()
        prepared_count = 0

        # Query-compiler mode is deliberately two-stage: all memory-free plans
        # first, then local retrieval, then one shared final-answer stage.
        if self.agent_manager and getattr(self.agent_manager, "uses_query_compiler", lambda: False)():
            event_state_snapshot = self._export_event_state_for_transfer(self.agent_manager, unit.context_id)
            plan_cache = self._load_query_compiler_plan_cache()
            for query in unit.queries_to_evaluate:
                formatted_question = self.prompt_manager.format_query(
                    question=query.question, query_type=query.query_type,
                    prompt_protocol=self.prompt_protocol,
                )
                final_id = make_request_id("query", f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}")
                saved_final = batch_client.get_saved_request("query-final", final_id)
                restored_final = restore_prepared_query(saved_final) if saved_final else None
                if restored_final is not None:
                    # The answer manifest is the authoritative frozen request
                    # during resume; do not recompile or rerun local retrieval.
                    self._pending_batch_queries.append({
                        "request": saved_final, "query": query, "prepared": restored_final,
                        "sample_id": unit.context_id, "memory_time_per_query": memory_time_per_query,
                        "compiler_usage": {"transport": "resume", "input_tokens": 0, "output_tokens": 0, "call_count": 0},
                    })
                    prepared_count += 1
                    continue
                request_id = make_request_id("query-plan", f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}")
                # The compiler intentionally receives no benchmark query type
                # or evaluator prompt wrapper; only deployable query text.
                compiler = self.agent_manager.prepare_query_compiler(query.question)
                reference_time = None
                fingerprint = self._query_compiler_cache_fingerprint(query.question, reference_time)
                cached = plan_cache.get(fingerprint) or self._legacy_query_compiler_cache_entry(
                    plan_cache, query.question, reference_time,
                )
                if cached is not None:
                    self._compiler_cache_hits += 1
                else:
                    self._compiler_cache_misses += 1
                self._pending_query_plan_requests.append({
                    "request": BatchChatRequest(request_id=request_id, messages=compiler["messages"],
                        temperature=compiler["temperature"], max_tokens=compiler["max_tokens"],
                        response_format=compiler.get("response_format"), phase="query-plan",
                        metadata={"query_id": query.query_id, "unit_id": unit.unit_id, "context_id": unit.context_id}),
                    "query": query, "question": formatted_question, "raw_question": query.question, "sample_id": unit.context_id,
                    "unit_id": unit.unit_id, "memory_state": event_state_snapshot,
                    "memory_time_per_query": memory_time_per_query, "reference_time": reference_time,
                    "compiler_fingerprint": fingerprint, "cached_plan": cached,
                })
                prepared_count += 1
            return prepared_count

        query_items = []
        for query in unit.queries_to_evaluate:
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

        event_state_snapshot = None
        if (
            getattr(self.method_config, "method_name", "").lower() == "event_state"
            and self.agent_manager is not None
        ):
            event_state_snapshot = self._export_event_state_for_transfer(
                self.agent_manager, unit.context_id
            )

        execution = self._event_state_query_execution(self._query_worker_count()) if event_state_snapshot is not None else None
        shared_manager = None
        worker_local = threading.local()
        if event_state_snapshot is not None and execution and execution["query_manager_reuse_mode"] == "shared":
            shared_manager = AgentManager(
                method_config=self.method_config, dataset_config=self.dataset_config,
                batch_api=self.batch_api, batch_gcs_uri=self.batch_gcs_uri,
                batch_wait=self.batch_wait, workers=1,
            )
            shared_manager.import_memory_state(event_state_snapshot, context_id=unit.context_id)
            execution["imported_manager_count"] = 1

        def prepare_item(item):
            query, _, _, batch_request_time, prepared = item
            if prepared is not None:
                return prepared
            formatted_question = self.prompt_manager.format_query(
                question=query.question,
                query_type=query.query_type,
                prompt_protocol=self.prompt_protocol,
            )
            manager = self.agent_manager
            if event_state_snapshot is not None:
                mode = execution["query_manager_reuse_mode"]
                if mode == "shared":
                    manager = shared_manager
                elif mode == "worker_local":
                    manager = getattr(worker_local, "manager", None)
                    if manager is None:
                        manager = AgentManager(method_config=self.method_config, dataset_config=self.dataset_config,
                            batch_api=self.batch_api, batch_gcs_uri=self.batch_gcs_uri, batch_wait=self.batch_wait, workers=1)
                        manager.import_memory_state(event_state_snapshot, context_id=unit.context_id)
                        worker_local.manager = manager
                        execution["imported_manager_count"] += 1
                else:  # Explicit legacy mode intentionally retains one import per query.
                    manager = AgentManager(method_config=self.method_config, dataset_config=self.dataset_config,
                        batch_api=self.batch_api, batch_gcs_uri=self.batch_gcs_uri, batch_wait=self.batch_wait, workers=1)
                    manager.import_memory_state(event_state_snapshot, context_id=unit.context_id)
                    execution["imported_manager_count"] += 1
            prepared_result = manager.prepare_batch_query(
                formatted_question,
                query_id=query.query_id,
                context_id=unit.context_id,
                batch_request_time=batch_request_time,
                **self._answer_query_kwargs(query),
            )
            if execution is not None:
                execution["reused_query_count"] += 1 if execution["query_manager_reuse_mode"] != "legacy" else 0
                prepared_result.setdefault("extra", {}).setdefault("query_execution", dict(execution))
            return prepared_result

        worker_count = execution["effective_query_workers"] if execution is not None else self._query_worker_count()
        preparation_started = time.perf_counter()
        if worker_count > 1 and len(query_items) > 1:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(query_items))) as executor:
                prepared_items = list(executor.map(prepare_item, query_items))
        else:
            prepared_items = [prepare_item(item) for item in query_items]
        self._batch_retrieval_preparation_wall_time = (
            getattr(self, "_batch_retrieval_preparation_wall_time", 0.0)
            + time.perf_counter() - preparation_started
        )

        for item, prepared in zip(query_items, prepared_items):
            query, request_id, saved_request, batch_request_time, _ = item

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
                "sample_id": unit.context_id,
                "memory_time_per_query": request.metadata.get(
                    "memory_time_per_query",
                    memory_time_per_query,
                ),
            })
            prepared_count += 1

        return prepared_count

    def _complete_combined_batch_queries(self) -> Iterator[Dict[str, Any]]:
        """Submit one Vertex stage and stream local finalization results."""
        if getattr(self, "_pending_query_plan_requests", None):
            stage = "query-plan"
            batch_client = self._get_batch_client()
            pending_plans = self._pending_query_plan_requests
            self._pending_query_plan_requests = []
            misses = [item for item in pending_plans if item.get("cached_plan") is None]
            saved_plan_requests = batch_client.get_saved_requests(stage) if misses else []
            expected_plan_ids = {item["request"].request_id for item in misses}
            requests = (saved_plan_requests if {request.request_id for request in saved_plan_requests} == expected_plan_ids
                        else [item["request"] for item in misses])
            self._log(f"[Vertex] Stage '{stage}': dispatching {len(requests):,} combined compiler request(s).")
            planner_started = time.perf_counter()
            responses = batch_client.run_stage(stage, requests) if requests else {}
            planner_wall_time = time.perf_counter() - planner_started
            self._compiler_calls_this_run += len(requests)
            retrieval_started = time.perf_counter()
            managers_by_context: Dict[Any, AgentManager] = {}
            self._log(
                "[Vertex] Stage 'query-plan': preparing "
                f"{len(pending_plans):,} local retrieval context(s) for the final-answer batch."
            )
            preparation_progress = tqdm(
                total=len(pending_plans),
                desc="Preparing final batch requests",
                unit="plan",
                dynamic_ncols=True,
                file=sys.stdout,
                disable=not getattr(self, "verbose", True),
            )
            try:
                for item in pending_plans:
                    cached = item.get("cached_plan")
                    batch_response = responses.get(item["request"].request_id)
                    content = cached["raw_model_output"] if cached is not None else (
                        batch_response.content if batch_response is not None and not batch_response.status else ""
                    )
                    context_id = item["sample_id"]
                    manager = managers_by_context.get(context_id)
                    if manager is None:
                        manager = AgentManager(method_config=self.method_config, dataset_config=self.dataset_config,
                            batch_api=self.batch_api, batch_gcs_uri=self.batch_gcs_uri, batch_wait=self.batch_wait, workers=1)
                        manager.import_memory_state(item["memory_state"], context_id=context_id)
                        managers_by_context[context_id] = manager
                        self._batch_manager_creation_count += 1
                        self._batch_memory_import_count += 1
                    item["memory_state"] = None
                    prepared = manager.prepare_query_compiler_result(item["question"], content,
                        context_id=item["sample_id"], **self._answer_query_kwargs(item["query"]))
                    compiler_diagnostics = prepared.get("extra", {}).get("query_compiler", {})
                    if cached is None and compiler_diagnostics.get("compiler_validation_success"):
                        self._append_query_compiler_plan_cache({
                            "fingerprint": item["compiler_fingerprint"], "question_sha256": hashlib.sha256(item["raw_question"].encode("utf-8")).hexdigest(),
                            "reference_time": item["reference_time"],
                            "provider": getattr(self.method_config.model, "provider", None), "model": getattr(self.method_config.model, "name", None),
                            "prompt_version": "event_state_query_compiler_v2", "schema_version": 2,
                            "max_searches": (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {}).get("query_compiler_max_searches", 3),
                            "query_compiler_max_tokens": (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {}).get("query_compiler_max_tokens", 256),
                            "temperature": (getattr(self.method_config, "raw_config", {}) or {}).get("retrieval_config", {}).get("planner_temperature", 0.0),
                            "raw_model_output": content, "validated_plan": compiler_diagnostics.get("validated_plan"),
                            "parse_success": compiler_diagnostics.get("compiler_parse_success"),
                            "salvage_used": compiler_diagnostics.get("compiler_salvage_used"),
                            "warning_codes": compiler_diagnostics.get("compiler_warning_codes", []),
                        })
                    final_id = make_request_id("query", f"{self.method_config.method_name}:{item['unit_id']}:{item['query'].query_id}")
                    final_request = BatchChatRequest(request_id=final_id, messages=prepared["messages"],
                        temperature=self.method_config.model.temperature,
                        max_tokens=(self.method_config.model.max_completion_tokens or self.method_config.model.max_tokens),
                        phase="query", metadata={"query_id": item["query"].query_id, "unit_id": item["unit_id"],
                        "context_id": item["sample_id"], PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared)})
                    self._pending_batch_queries.append({"request": final_request, "query": item["query"], "prepared": prepared,
                        "sample_id": item["sample_id"], "memory_time_per_query": item["memory_time_per_query"],
                        "compiler_usage": {"transport": "cache" if cached is not None else "batch", "input_tokens": 0 if cached is not None else getattr(batch_response, "input_tokens", 0),
                                           "output_tokens": 0 if cached is not None else getattr(batch_response, "output_tokens", 0), "call_count": 0 if cached is not None else 1,
                                           "cache_hit": cached is not None},
                        "batch_retrieval_diagnostics": {"manager_creation_count": self._batch_manager_creation_count,
                            "memory_import_count": self._batch_memory_import_count,
                            "planner_batch_wall_time": planner_wall_time,
                            "compiler_cache_hits": self._compiler_cache_hits, "compiler_cache_misses": self._compiler_cache_misses,
                            "compiler_calls_this_run": self._compiler_calls_this_run}})
                    preparation_progress.update(1)
            finally:
                preparation_progress.close()
            self._batch_retrieval_preparation_wall_time = getattr(self, "_batch_retrieval_preparation_wall_time", 0.0) + time.perf_counter() - retrieval_started
            # Release heavy stage-1 resources and run garbage collection before stage 2
            managers_by_context.clear()
            del managers_by_context
            del pending_plans
            del responses
            del requests
            del misses
            gc.collect()
            yield from self._complete_combined_batch_queries()
            return
        if not self._pending_batch_queries:
            return

        stage = "query-final"
        batch_client = self._get_batch_client()
        saved_requests = batch_client.get_saved_requests(stage)
        requests = saved_requests or [
            item["request"] for item in self._pending_batch_queries
        ]
        self._log(
            f"[Vertex] Stage '{stage}': dispatching {len(requests):,} combined "
            "final-answer request(s) from all prepared samples."
        )
        answer_started = time.perf_counter()
        responses = batch_client.run_stage(stage, requests)
        answer_wall_time = time.perf_counter() - answer_started
        pending_items = self._pending_batch_queries
        self._pending_batch_queries = []
        self._log(
            f"[Vertex] Stage '{stage}': finalizing {len(pending_items):,} response(s) "
            f"locally; checkpoint flush interval is "
            f"{LOCOMO_QUERY_CHECKPOINT_FLUSH_INTERVAL:,} answer(s)."
        )
        progress = tqdm(
            total=len(pending_items),
            desc="Finalizing batch answers",
            unit="answer",
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not getattr(self, "verbose", True),
        )
        try:
            for item in pending_items:
                request = item["request"]
                query = item["query"]
                batch_response = responses.get(request.request_id)
                if batch_response is None or batch_response.status:
                    error = batch_response.status if batch_response else "No output row returned"
                    result = self._api_error_result(
                        query,
                        f"Batch request failed: {truncate_error_message(error)}",
                    )
                else:
                    response = self.agent_manager.finalize_batch_query(
                        item["prepared"],
                        batch_response.content,
                        input_tokens=batch_response.input_tokens,
                        output_tokens=batch_response.output_tokens,
                    )
                    result = self._score_agent_response(query, response)
                    if hasattr(result, "details"):
                        result.details.setdefault("execution_usage", {})["answer"] = (
                            self._batch_answer_execution_usage(batch_response)
                        )
                        if item.get("compiler_usage"):
                            result.details.setdefault("execution_usage", {})["query_compiler"] = item["compiler_usage"]
                        result.details.setdefault("event_state_batch", {}).update({
                            **item.get("batch_retrieval_diagnostics", {}),
                            "local_retrieval_preparation_wall_time": getattr(self, "_batch_retrieval_preparation_wall_time", 0.0),
                            "final_answer_batch_wall_time": answer_wall_time,
                        })

                result.memory_construction_time = item["memory_time_per_query"]
                # Progressively free prompt and request payload memory
                item["prepared"] = None
                item["request"] = None
                progress.update(1)
                yield {
                    "sample_id": item["sample_id"],
                    "result": result,
                }
        finally:
            progress.close()
            pending_items.clear()
            if isinstance(responses, dict):
                responses.clear()
            gc.collect()

    def _evaluate_query(
        self,
        query: LoCoMoQuery,
        context_id: Any,
        manager: Optional[AgentManager] = None,
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
                details={"dry_run": True, "category": query.category},
            )

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
            prompt_protocol=getattr(self, "prompt_protocol", "type_aware"),
        )

        response = (manager or self.agent_manager).send_message(
            message=formatted_question,
            memorizing=False,
            context_id=context_id,
            **self._answer_query_kwargs(query),
        )

        return self._score_agent_response(query, response)

    def _api_error_result(self, query: LoCoMoQuery, error_message: str) -> MetricResult:
        return MetricResult(
            query_id=query.query_id,
            query_type=query.query_type,
            score=0.0,
            is_correct=False,
            model_output="[API_ERROR] Batch request failed",
            expected_answer=", ".join(query.get_correct_answers()),
            question=query.question,
            details={
                "api_error": True,
                "error_message": truncate_error_message(error_message),
            },
        )

    @staticmethod
    def _gold_evidence_turns(evidence: List[str]) -> List[str]:
        """Extract canonical LoCoMo evidence IDs for evaluator-only diagnostics."""
        values, seen = [], set()
        for value in evidence or []:
            # An annotation can pack multiple IDs into one field (for example,
            # ``D8:6; D9:17``). Only standalone canonical tokens are accepted.
            for match in re.finditer(r"(?<![^\s;])(D\d+:[^\s:;]+)(?=$|[\s;])", str(value)):
                normalized = match.group(1)
                if normalized in seen:
                    continue
                values.append(normalized)
                seen.add(normalized)
        return values

    @staticmethod
    def _turn_quality(gold_turn_ids: List[str], visible_turn_ids: List[str]) -> Dict[str, Any]:
        """Score exact evidence turns rendered in final answer context."""
        predicted = list(dict.fromkeys(str(value) for value in visible_turn_ids))
        gold = list(dict.fromkeys(gold_turn_ids))
        predicted_set, gold_set = set(predicted), set(gold)
        matched = [value for value in predicted if value in gold_set]
        precision = len(matched) / len(predicted) if predicted else 0.0
        recall = len(matched) / len(gold) if gold else None
        return {
            "stage": "answer_visible_exact_source_evidence",
            "unit": "locomo_turn_id",
            "available": bool(gold),
            "gold_turn_ids": gold,
            "answer_visible_turn_ids": predicted,
            "matched_turn_ids": matched,
            "precision": precision if gold else None,
            "recall": recall,
            "hit": bool(matched) if gold else None,
            "true_positive_count": len(matched),
            "false_positive_count": len(predicted_set - gold_set),
            "false_negative_count": len(gold_set - predicted_set),
        }

    @staticmethod
    def _locomo_turn_id(source_session_id: Any, turn_id: Any) -> Optional[str]:
        value = str(turn_id).strip()
        if re.fullmatch(r"D\d+:[^\s:]+", value):
            return value
        if source_session_id is None or not value:
            return None
        return f"D{source_session_id}:{value}"

    def _locomo_retrieval_quality(
        self,
        query: LoCoMoQuery,
        retrieved_memories: List[Dict[str, Any]],
        method_retrieval: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Derive evaluator-private gold diagnostics after retrieval completes."""
        gold_turn_ids = self._gold_evidence_turns(query.evidence)
        gold_session_ids = [match.group(1) for item in gold_turn_ids if (match := re.fullmatch(r"D(\d+):[^\s:]+", item))]
        gold_metadata = {"gold_session_ids": gold_session_ids}

        # These method traces contain only generic source IDs and ranking
        # outcomes. Gold annotations are joined here, after the agent has
        # completed selection and answer construction.
        stage_candidates = (
            method_retrieval.get("retrieval_stage_candidates", {})
            if isinstance(method_retrieval, dict) else {}
        )

        def candidate_quality(stage: str, candidates: Any) -> Dict[str, Any]:
            records = []
            for candidate in candidates if isinstance(candidates, list) else []:
                if not isinstance(candidate, dict):
                    continue
                for source_id in candidate.get("source_session_ids", []):
                    records.append({"source_session_id": source_id})
            quality = compute_session_retrieval_quality(records, [], gold_metadata)
            quality["stage"] = stage
            return quality

        session_quality = compute_session_retrieval_quality(
            retrieved_memories, [], gold_metadata,
        )
        session_quality["stage"] = "selected_memory_objects"
        archive_turn_ids = []
        for record in retrieved_memories or []:
            if not isinstance(record, dict) or record.get("type") != "episode":
                continue
            for turn_id in record.get("episode_archive_turn_ids", []):
                normalized = self._locomo_turn_id(record.get("source_session_id"), turn_id)
                if normalized:
                    archive_turn_ids.append(normalized)
        archive_quality = self._turn_quality(gold_turn_ids, archive_turn_ids)
        archive_quality["stage"] = "selected_episode_archive"
        archive_quality["selected_episode_archive_turn_ids"] = archive_quality.pop(
            "answer_visible_turn_ids"
        )
        visible_turn_ids = []
        visible_routes: Dict[str, set[str]] = {}

        def add_visible(turn_id: Optional[str], route: str) -> None:
            if not turn_id:
                return
            visible_turn_ids.append(turn_id)
            visible_routes.setdefault(turn_id, set()).add(route)

        for record in retrieved_memories or []:
            if not isinstance(record, dict):
                continue
            source_session = record.get("source_session_id")
            route = (
                "direct_immutable_turn" if record.get("type") == "immutable_turn"
                else "episode_excerpt"
            )
            for turn_id in record.get("episode_evidence_turn_ids", []) if record.get("included_in_context") else []:
                normalized = self._locomo_turn_id(source_session, turn_id)
                add_visible(normalized, route)
            for item in record.get("included_provenance_evidence", []):
                evidence = item.get("evidence", item) if isinstance(item, dict) else {}
                if not isinstance(evidence, dict):
                    continue
                source_session = evidence.get("source_session_id")
                for turn_id in evidence.get("source_turn_ids", []):
                    normalized = self._locomo_turn_id(source_session, turn_id)
                    add_visible(normalized, "claim_provenance")
        route_counts = {
            route: sum(route in visible_routes.get(turn_id, set()) for turn_id in gold_turn_ids)
            for route in ("claim_provenance", "episode_excerpt", "direct_immutable_turn")
        }
        compiler_stage_names = (
            "channel_semantic_candidates", "merged_semantic_union",
            "temporally_reranked_union", "final_memory_object_selection",
        )
        compiler_stages = {
            name: candidate_quality(name, stage_candidates[name])
            for name in compiler_stage_names if name in stage_candidates
        }
        result = {
            "gold_evidence_turn_ids": gold_turn_ids,
            "gold_evidence_session_ids": list(dict.fromkeys(gold_session_ids)),
            "selected_memory_session": session_quality,
            "selected_episode_archive_exact_turn": archive_quality,
            "answer_visible_exact_turn": self._turn_quality(gold_turn_ids, visible_turn_ids),
            # Routes are additive: one visible gold turn can have more than
            # one rendering route, while answer_visible_exact_turn is their union.
            "answer_visible_gold_turn_routes": {
                turn_id: sorted(visible_routes[turn_id])
                for turn_id in gold_turn_ids if turn_id in visible_routes
            },
            "answer_visible_gold_turn_via_claim_count": route_counts["claim_provenance"],
            "answer_visible_gold_turn_via_episode_excerpt_count": route_counts["episode_excerpt"],
            "answer_visible_gold_turn_via_direct_turn_count": route_counts["direct_immutable_turn"],
        }
        # Do not claim compiler stages are legacy truncation stages. Retain
        # old labels only when the method actually supplied those traces.
        for key in ("pre_candidate_truncation_fused", "post_candidate_count"):
            if key in stage_candidates:
                result[f"{key}_session"] = candidate_quality(key, stage_candidates[key])
        result.update({f"{key}_session": value for key, value in compiler_stages.items()})
        return result

    def _score_agent_response(self, query: LoCoMoQuery, response: Any) -> MetricResult:
        if isinstance(response, dict):
            model_output = response.get("output", "")
            query_time = response.get("query_time", 0.0)
            retrieved_memories = response.get("retrieved_memories", [])
            retrieved_count = response.get("retrieved_count", 0)
            response_extra = response.get("extra", {})
        elif hasattr(response, "output"):
            model_output = response.output
            query_time = getattr(response, "query_time", 0.0)
            retrieved_memories = getattr(response, "retrieved_memories", [])
            retrieved_count = getattr(response, "retrieved_count", 0)
            response_extra = getattr(response, "extra", {})
        else:
            model_output = str(response)
            query_time = 0.0
            retrieved_memories = []
            retrieved_count = 0
            response_extra = {}

        tracker = get_usage_tracker()
        tracker.set_phase("judge")
        with tracker.scope("judge.realtime"):
            result = self.metrics_calculator.compute(
                query_id=query.query_id,
                query_type=query.query_type,
                model_output=model_output,
                expected_answers=query.get_correct_answers(),
                question=query.question,
                category=query.category,
                evidence=query.evidence,
                adversarial_answer=query.adversarial_answer,
                metadata=query.metadata,
            )

        result.query_time = query_time
        result.retrieved_memories = retrieved_memories
        result.retrieved_count = retrieved_count
        if isinstance(response_extra, dict):
            result.details["method_retrieval"] = response_extra

        if "category" not in result.details:
            result.details["category"] = query.category
        if "evidence" not in result.details:
            result.details["evidence"] = query.evidence
        result.details["locomo_retrieval_quality"] = self._locomo_retrieval_quality(
            query, retrieved_memories, response_extra if isinstance(response_extra, dict) else None,
        )
        if isinstance(result.details.get("method_retrieval"), dict):
            mr = result.details["method_retrieval"]
            for key in (
                "retrieval_stage_candidates",
                "channel_semantic_candidates",
                "merged_semantic_union",
                "temporally_reranked_union",
            ):
                mr.pop(key, None)

        return result

    def _build_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        summary = self.aggregator.get_summary()
        self._apply_locomo_summary(summary)

        memory_build_summary = self._summarize_memory_builds()
        build_metrics = self._compact_build_metrics()
        is_event_state = self.method_config.method_name.lower() == "event_state"
        memory_size = self._event_state_memory_size() if is_event_state else {}
        feature_configuration = (
            self._event_state_feature_configuration() if is_event_state else {}
        )

        llm_usage = get_usage_tracker().get_stats()
        stage_usage = self._stage_usage_report(llm_usage)
        results = self.aggregator.results
        expected_queries = self.dataset.get_total_queries()
        scored_queries = sum(result.score is not None for result in results)
        api_error_count = sum(
            bool(result.details.get("api_error")) for result in results
        )
        category_filter = self.dataset.category_filter
        evaluation_coverage = {
            "expected_query_count": expected_queries,
            "scored_query_count": scored_queries,
            "api_error_count": api_error_count,
            "skipped_query_count": max(expected_queries - len(results), 0),
            "unscored_query_count": max(expected_queries - scored_queries, 0),
            "category_filter": category_filter,
            "category_coverage": self.dataset.get_category_distribution(),
            "complete": scored_queries == expected_queries,
        }
        run_metadata = self._git_metadata()
        reproducibility_warning = None
        if not run_metadata["git_metadata_available"]:
            reproducibility_warning = "Canonical benchmark run has unavailable Git metadata; commit identity and dirty state are uncertain."
        elif run_metadata["dirty"] is True:
            reproducibility_warning = "Canonical benchmark run uses a dirty worktree."
        elif run_metadata["dirty"] is None:
            reproducibility_warning = "Canonical benchmark run has unknown worktree cleanliness."
        if reproducibility_warning:
            self._log(reproducibility_warning, level="WARNING")
            run_metadata["reproducibility_warning"] = reproducibility_warning

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
                "locomo_scoring": {
                    "primary_metric": "official_token_stem_f1",
                    "official_locomo": "canonical_score",
                    "legacy_medmemorybench": "historical_compatibility_score",
                    "conservative_enhanced_f1": "diagnostic_only",
                },
            },
            metadata={
                "prompt_protocol": self.prompt_protocol,
                "query_type_aware_prompting": self.prompt_protocol == "type_aware",
                "run_metadata": run_metadata,
                "dataset_coverage": {
                    "available_sample_count": self.dataset.get_available_sample_count(),
                    "evaluated_sample_count": len(self.dataset.get_sample_ids()),
                    "configured_max_samples": self.dataset.max_samples,
                    "sample_ids": self.dataset.get_sample_ids(),
                },
                "category_distribution": self.dataset.get_category_distribution(),
                "input_modality": {
                    "image_input_mode": "caption_only" if self.dataset.include_images else "disabled",
                    "image_caption_field": "blip_caption" if self.dataset.include_images else None,
                },
                "memory_build_summary": memory_build_summary,
                "build_metrics": build_metrics,
                "memory_size": memory_size,
                "feature_configuration": feature_configuration,
                "memory_chunk_size": self.memory_chunk_size,
                "llm_usage": llm_usage,
                "stage_usage": stage_usage,
                "evaluation_coverage": evaluation_coverage,
            }
        )

        return report

    def _persist_memory_build_checkpoint(self, start_time: datetime) -> None:
        """Preserve completed build telemetry when a later stage aborts the run."""
        if (
            getattr(self, "_memory_build_checkpoint_saved", False)
            or self.execution_stage == "query"
            or not self._memory_build_logs
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
            memory_build_logs=self._memory_build_logs,
            include_result=False,
            include_memory_build=True,
            include_query_answer=False,
            include_api_failures=False,
            use_method_subdir=not self.run_scoped_output,
        )
        self._memory_build_checkpoint_saved = True
        self._log(f"Memory build checkpoint saved to: {memory_build_path}")

    def _generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        report = self._build_report(start_time, end_time, duration)
        result_path, memory_build_path, query_answer_path = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=self._memory_build_logs,
            use_method_subdir=not self.run_scoped_output,
        )

        self._log(f"Results saved to: {result_path}")
        self._log(f"Memory build details saved to: {memory_build_path}")
        self._log(f"Query answer details saved to: {query_answer_path}")

        return report

    @staticmethod
    def _aggregate_locomo_retrieval(results: List[MetricResult]) -> Dict[str, Any]:
        """Macro aggregate evaluator-only selected-session and visible-turn metrics."""
        stages = {
            "pre_candidate_truncation_fused_session": "pre_candidate_truncation_fused_session",
            "post_candidate_count_session": "post_candidate_count_session",
            "channel_semantic_candidates_session": "channel_semantic_candidates_session",
            "merged_semantic_union_session": "merged_semantic_union_session",
            "temporally_reranked_union_session": "temporally_reranked_union_session",
            "final_memory_object_selection_session": "final_memory_object_selection_session",
            "selected_memory_session": "selected_memory_session",
            "selected_episode_archive_exact_turn": "selected_episode_archive_exact_turn",
            "answer_visible_exact_turn": "answer_visible_exact_turn",
        }
        aggregated: Dict[str, Any] = {}
        for output_name, source_name in stages.items():
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for result in results:
                quality = result.details.get("locomo_retrieval_quality", {})
                value = quality.get(source_name) if isinstance(quality, dict) else None
                if isinstance(value, dict):
                    groups.setdefault(result.query_type, []).append(value)

            def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
                available = [item for item in items if item.get("available")]
                def mean(field: str) -> Optional[float]:
                    values = [float(item[field]) for item in available if item.get(field) is not None]
                    return sum(values) / len(values) if values else None
                return {
                    "queries_with_gold_evidence": len(available),
                    "queries_without_gold_evidence": len(items) - len(available),
                    "hit_rate": mean("hit"), "recall": mean("recall"),
                    "precision": mean("precision"),
                    "mean_average_precision": mean("average_precision"),
                    "mean_reciprocal_rank": mean("reciprocal_rank"),
                }
            all_items = [item for items in groups.values() for item in items]
            aggregated[output_name] = {
                "stage": source_name,
                **summarize(all_items),
                "by_query_type": {
                    query_type: summarize(items) for query_type, items in sorted(groups.items())
                },
            }
        route_fields = (
            "answer_visible_gold_turn_via_claim_count",
            "answer_visible_gold_turn_via_episode_excerpt_count",
            "answer_visible_gold_turn_via_direct_turn_count",
        )
        for field in route_fields:
            values = [
                int(quality.get(field, 0))
                for result in results
                if isinstance((quality := result.details.get("locomo_retrieval_quality")), dict)
            ]
            aggregated[field] = sum(values)
        return aggregated

    def _apply_locomo_summary(self, summary: Dict[str, Any]) -> None:
        """Make official F1 primary while retaining legacy compatibility totals."""
        results = [
            result for result in self.aggregator.results
            if result.score is not None and result.details.get("metric") == "locomo_f1"
        ]
        summary["f1_query_count"] = len(results)
        summary["mean_f1"] = (
            sum(float(result.score) for result in results) / len(results)
            if results else 0.0
        )
        summary["queries_f1_ge_0_5"] = sum(
            float(result.score) >= 0.5 for result in results
        )
        summary["fraction_f1_ge_0_5"] = (
            summary["queries_f1_ge_0_5"] / len(results) if results else 0.0
        )
        legacy_scores = [
            float(result.details["legacy_medmemorybench_score"])
            for result in results
            if result.details.get("legacy_medmemorybench_score") is not None
        ]
        legacy_correct_count = sum(score >= 0.5 for score in legacy_scores)
        summary["metric_variants"] = {
            "official_locomo": {
                "f1_query_count": len(results),
                "mean_f1": summary["mean_f1"],
                "queries_f1_ge_0_5": summary["queries_f1_ge_0_5"],
                "fraction_f1_ge_0_5": summary["fraction_f1_ge_0_5"],
            },
            "legacy_medmemorybench": {
                "f1_query_count": len(legacy_scores),
                "mean_f1": (
                    sum(legacy_scores) / len(legacy_scores)
                    if legacy_scores else 0.0
                ),
                "queries_f1_ge_0_5": legacy_correct_count,
                "fraction_f1_ge_0_5": (
                    legacy_correct_count / len(legacy_scores)
                    if legacy_scores else 0.0
                ),
            },
        }
        for query_type, stats in summary.get("by_type", {}).items():
            if query_type == "adversarial":
                continue
            stats["mean_f1"] = stats.get("avg_score", 0.0)
            stats["queries_f1_ge_0_5"] = stats.pop("correct", 0)
            stats["fraction_f1_ge_0_5"] = stats.pop("accuracy", 0.0)
        locomo_f1 = summary.get("by_metric", {}).get("locomo_f1")
        if isinstance(locomo_f1, dict):
            locomo_f1["mean_f1"] = locomo_f1.pop("avg_score", 0.0)
            locomo_f1["queries_f1_ge_0_5"] = locomo_f1.pop("correct", 0)
            locomo_f1["fraction_f1_ge_0_5"] = locomo_f1.pop("accuracy", 0.0)
        summary["retrieval_quality"] = self._aggregate_locomo_retrieval(results)

    @staticmethod
    def _batch_elapsed_seconds(
        stage: Dict[str, Any],
        started_at_key: str,
    ) -> Optional[float]:
        """Return elapsed batch time only when the recorded timestamps agree."""
        started_at = stage.get(started_at_key)
        completed_at = stage.get("completed_at")
        if not isinstance(started_at, str) or not isinstance(completed_at, str):
            return None
        try:
            elapsed = (
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds()
        except (TypeError, ValueError):
            return None
        return elapsed if elapsed >= 0 else None

    @classmethod
    def _batch_wall_time_seconds(cls, stage: Dict[str, Any]) -> Optional[float]:
        """Return queue-inclusive elapsed time retained for legacy reports."""
        return cls._batch_elapsed_seconds(stage, "submitted_at")

    @classmethod
    def _batch_overall_latency_seconds(cls, stage: Dict[str, Any]) -> Optional[float]:
        """Return batch execution time from RUNNING to terminal completion."""
        return cls._batch_elapsed_seconds(stage, "running_at")

    @staticmethod
    def _aggregate_batch_answer_usage(results: List[MetricResult]) -> Dict[str, Any]:
        """Aggregate per-result batch usage without re-counting global trackers."""
        input_tokens = output_tokens = request_count = 0
        for result in results:
            details = result.details if isinstance(result.details, dict) else {}
            execution_usage = details.get("execution_usage", {})
            answer_usage = (
                execution_usage.get("answer", {})
                if isinstance(execution_usage, dict) else {}
            )
            if not isinstance(answer_usage, dict) or answer_usage.get("transport") != "batch":
                continue
            input_tokens += int(answer_usage.get("input_tokens", 0) or 0)
            output_tokens += int(answer_usage.get("output_tokens", 0) or 0)
            request_count += 1
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "request_count": request_count,
            "successful_requests": request_count,
            "transport": "batch",
        }

    def _batch_manifest_paths_for_report(self) -> List[Path]:
        """Find the current run's manifest even when resume skipped batch setup."""
        paths: List[Path] = []
        manifest_path = getattr(getattr(self, "_batch_client", None), "manifest_path", None)
        if manifest_path:
            paths.append(Path(manifest_path))

        model = getattr(getattr(self, "method_config", None), "model", None)
        model_name = getattr(model, "name", None)
        if model_name:
            paths.append(scoped_manifest_path(
                self.output_dir / "batch",
                "locomo_batch_manifest",
                model=model_name,
                config_hash=self._batch_config_hash(),
            ))

        unique_paths: List[Path] = []
        seen = set()
        for path in paths:
            key = str(path.resolve())
            if key not in seen and path.exists():
                seen.add(key)
                unique_paths.append(path)
        return unique_paths

    @classmethod
    def _stream_manifest_stages(cls, path: Path) -> List[Dict[str, Any]]:
        """Extract stage lifecycle metadata from large manifests without full JSON decoding."""
        stages: List[Dict[str, Any]] = []
        current_stage: Optional[Dict[str, Any]] = None
        in_jobs = False
        in_requests = False

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    indent = len(line) - len(line.lstrip(" "))
                    stripped = line.strip()

                    if not in_jobs:
                        if stripped == '"jobs": {':
                            in_jobs = True
                        continue

                    if indent <= 2 and stripped.startswith("}"):
                        break

                    if indent == 4 and stripped.endswith("{"):
                        m = re.match(r'^"([^"]+)":\s*\{', stripped)
                        if m:
                            current_stage = {
                                "stage": m.group(1),
                                "request_count": 0,
                                "state": None,
                                "submitted_at": None,
                                "running_at": None,
                                "completed_at": None,
                            }
                            stages.append(current_stage)
                            in_requests = False
                            continue

                    if current_stage is None:
                        continue

                    if indent == 6 and stripped == '"requests": [':
                        in_requests = True
                        continue
                    elif indent == 6 and stripped in ("]", "],"):
                        in_requests = False
                        continue

                    if in_requests:
                        if indent == 10 and stripped.startswith('"request_id":'):
                            current_stage["request_count"] += 1
                    elif indent == 6:
                        if stripped.startswith('"request_count":'):
                            val_m = re.search(r'"request_count":\s*(\d+)', stripped)
                            if val_m:
                                current_stage["request_count"] = int(val_m.group(1))
                        for key in ("state", "submitted_at", "running_at", "completed_at"):
                            if stripped.startswith(f'"{key}":'):
                                val_m = re.search(rf'"{key}":\s*"([^"]*)"', stripped)
                                if val_m:
                                    current_stage[key] = val_m.group(1)
        except OSError:
            return []

        for stage in stages:
            stage["overall_latency_seconds"] = cls._batch_overall_latency_seconds(stage)
            stage["queue_inclusive_elapsed_seconds"] = cls._batch_wall_time_seconds(stage)
        return stages

    @classmethod
    def _extract_batch_stages_from_manifest(cls, manifest_path: Path) -> List[Dict[str, Any]]:
        path = Path(manifest_path)
        if not path.exists():
            return []
        try:
            if path.stat().st_size <= 5 * 1024 * 1024:
                with path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                stages = []
                for name, stage in (manifest.get("jobs") or {}).items():
                    if not isinstance(stage, dict):
                        continue
                    request_count = stage.get("request_count")
                    if request_count is None:
                        requests = stage.get("requests")
                        request_count = len(requests) if isinstance(requests, list) else 0
                    stages.append({
                        "stage": name,
                        "state": stage.get("state"),
                        "request_count": request_count,
                        "submitted_at": stage.get("submitted_at"),
                        "running_at": stage.get("running_at"),
                        "completed_at": stage.get("completed_at"),
                        "overall_latency_seconds": cls._batch_overall_latency_seconds(stage),
                        "queue_inclusive_elapsed_seconds": cls._batch_wall_time_seconds(stage),
                    })
                return stages
        except (OSError, json.JSONDecodeError):
            return []

        return cls._stream_manifest_stages(path)

    def _stage_usage_report(self, llm_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Expose local phase accounting and batch lifecycle without fake latency."""
        operations = llm_usage.get("operations", {})
        query_operations = operations.get("query", {})
        batch_stages: List[Dict[str, Any]] = []

        batch_client = getattr(self, "_batch_client", None)
        if batch_client is not None and hasattr(batch_client, "get_stage_summaries"):
            try:
                for summary in batch_client.get_stage_summaries():
                    batch_stages.append({
                        "stage": summary.get("stage"),
                        "state": summary.get("state"),
                        "request_count": summary.get("request_count", 0),
                        "submitted_at": summary.get("submitted_at"),
                        "running_at": summary.get("running_at"),
                        "completed_at": summary.get("completed_at"),
                        "overall_latency_seconds": self._batch_overall_latency_seconds(summary),
                        "queue_inclusive_elapsed_seconds": self._batch_wall_time_seconds(summary),
                    })
            except Exception:
                pass

        if not batch_stages:
            for manifest_path in self._batch_manifest_paths_for_report():
                batch_stages.extend(self._extract_batch_stages_from_manifest(manifest_path))
        retrieval_usage = query_operations.get("query.retrieval_preparation", {})
        retrieval_end_to_end = getattr(
            self, "_batch_retrieval_preparation_wall_time", 0.0
        )
        batch_usage = self._aggregate_batch_answer_usage(
            getattr(getattr(self, "aggregator", None), "results", [])
        )
        realtime_usage = query_operations.get("query.answer_realtime", {})
        answer_batch_stages = [
            stage for stage in batch_stages if stage.get("stage") == "query-final"
        ]
        has_batch_usage = batch_usage["request_count"] > 0
        has_realtime_usage = bool(realtime_usage.get("call_count", 0))
        if answer_batch_stages and has_realtime_usage:
            answer_generation = {
                "transport": "mixed",
                "usage": {"batch": batch_usage, "realtime": realtime_usage},
                "batch_usage": batch_usage,
                "realtime_usage": realtime_usage,
            }
        elif answer_batch_stages or has_batch_usage:
            answer_generation = {"transport": "batch", "usage": batch_usage}
        else:
            answer_generation = {"transport": "realtime", "usage": realtime_usage}
        if answer_batch_stages:
            answer_generation["batch_overall_latency_seconds"] = answer_batch_stages[-1][
                "overall_latency_seconds"
            ]
        return {
            "schema_version": 2,
            "memory": {"usage": llm_usage.get("memorize_phase", {})},
            "retrieval_preparation": {
                "usage": retrieval_usage,
                "operation_wall_time_seconds": retrieval_usage.get("wall_time"),
                "end_to_end_wall_time_seconds": retrieval_end_to_end,
            },
            "answer_generation": answer_generation,
            "judge": {"usage": llm_usage.get("judge_phase", {})},
            "batch_stages": batch_stages,
        }

    def _summarize_memory_builds(self) -> Dict[str, Any]:
        if not self._memory_build_logs:
            return {"total_builds": 0}

        total_units = len(self._memory_build_logs)

        total_sessions = sum(
            log.get("session_count", 0)
            for log in self._memory_build_logs
        )

        total_time = sum(
            log.get("total_time", 0)
            for log in self._memory_build_logs
        )

        inserted_record_count = sum(
            log.get("inserted_record_count", log.get("total_entries", 0))
            for log in self._memory_build_logs
        )

        summary = {
            "total_units": total_units,
            "total_sessions": total_sessions,
            "total_time": total_time,
            "inserted_record_count": inserted_record_count,
            "avg_time_per_unit": total_time / total_units if total_units > 0 else 0,
        }
        if self.method_config.method_name.lower() == "event_state":
            return summary

        total_chunks = sum(
            log.get("chunk_count", 0)
            for log in self._memory_build_logs
        )
        total_stored_chunks = sum(
            log.get("total_stored_chunks", 0)
            for log in self._memory_build_logs
        )
        summary.update({
            "total_memory_chunks": total_chunks,
            "total_stored_chunks": total_stored_chunks,
            "avg_chunks_per_unit": total_chunks / total_units if total_units > 0 else 0,
            "chunk_size_config": self.memory_chunk_size,
        })
        return summary

    def _event_state_memory_size(self) -> Dict[str, Any]:
        """Report final-store cardinalities and latest persisted snapshot bytes."""
        totals = {
            "final_episode_count": 0,
            "final_claim_count": 0,
            "final_memory_object_count": 0,
        }
        found_diagnostics = False
        latest_serialized_sizes: Dict[str, Dict[str, Any]] = {}
        for log in self._memory_build_logs:
            diagnostics = log.get("final_store") or log.get("build_metrics")
            if isinstance(diagnostics, dict) and any(
                field in diagnostics for field in totals
            ):
                found_diagnostics = True
                for field in totals:
                    totals[field] += int(diagnostics.get(field, 0) or 0)
            serialized_size = log.get("memory_size")
            if (
                isinstance(serialized_size, dict)
                and serialized_size.get("measurement") == "serialized_memory_state"
            ):
                latest_serialized_sizes[str(log.get("context_id"))] = serialized_size

        if not found_diagnostics and not latest_serialized_sizes:
            return {}
        result: Dict[str, Any] = totals if found_diagnostics else {}
        if latest_serialized_sizes:
            result.update({
                "serialized_memory_bytes": sum(
                    int(item.get("bytes", 0) or 0)
                    for item in latest_serialized_sizes.values()
                ),
                "serialized_memory_json_bytes": sum(
                    int(item.get("json_bytes", 0) or 0)
                    for item in latest_serialized_sizes.values()
                ),
                "serialized_memory_embedding_bytes": sum(
                    int(item.get("embedding_bytes", 0) or 0)
                    for item in latest_serialized_sizes.values()
                ),
                "serialized_memory_mib": round(
                    sum(
                        int(item.get("bytes", 0) or 0)
                        for item in latest_serialized_sizes.values()
                    ) / (1024 ** 2),
                    6,
                ),
                "serialized_memory_context_count": len(latest_serialized_sizes),
            })
        return result

    def _event_state_feature_configuration(self) -> Dict[str, Any]:
        """Expose the compact effective settings needed to interpret an artifact."""
        build_config = dict(getattr(self.method_config, "build_config", {}) or {})
        retrieval_config = dict(
            getattr(self.method_config, "retrieval_config", {}) or {}
        )
        embedding = getattr(self.method_config, "embedding", None)
        return {
            "semantic_version": build_config.get("event_state_semantic_version"),
            "planner_enabled": int(retrieval_config.get("planner_rounds", 0) or 0) > 0,
            "ppr_enabled": retrieval_config.get("ppr_enabled", False),
            "temporal_retrieval_enabled": retrieval_config.get(
                "temporal_retrieval_enabled", True
            ),
            "selector_mode": retrieval_config.get("selector_mode", "state_mmr"),
            "evidence_count": retrieval_config.get("evidence_count", 8),
            "turn_evidence_count": retrieval_config.get("turn_evidence_count", 0),
            "claim_top_k": retrieval_config.get("claim_top_k", 30),
            "episode_top_k": retrieval_config.get("episode_top_k", 20),
            "retrieve_turns": retrieval_config.get("retrieve_turns", True),
            "turn_top_k": retrieval_config.get("turn_top_k", 8),
            "turn_retrieval_weight": retrieval_config.get("turn_retrieval_weight", 1.0),
            "turn_lexical_retrieval_enabled": retrieval_config.get("turn_lexical_retrieval_enabled", False),
            "max_episode_source_excerpts_total": retrieval_config.get("max_episode_source_excerpts_total", 2),
            "episode_retrieval_sampling": "evenly_spaced_source_order",
            "episode_retrieval_max_turns": 8,
            "candidate_count": retrieval_config.get("candidate_count", 40),
            "embedding_model": getattr(embedding, "model", None),
        }

    def _compact_build_metrics(self) -> Dict[str, Any]:
        """Keep useful Event-State diagnostics in result.json without raw memory."""
        units = {}
        excluded = {
            "memory_entries", "all_passages", "input_content", "stored_content",
            "extraction_result", "raw_result",
        }
        if self.method_config.method_name.lower() == "event_state":
            excluded.add("chunk_count")
        for log in self._memory_build_logs:
            metrics = log.get("build_metrics", {})
            if not isinstance(metrics, dict):
                continue
            units[str(log.get("context_id"))] = {
                key: value for key, value in metrics.items() if key not in excluded
            }
        return {
            "schema_version": (
                2 if self.method_config.method_name.lower() == "event_state" else 1
            ),
            "units": units,
        }


@register_evaluator("locomo")
def evaluate_locomo(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    execution_stage: str = "all",
    memory_run: Optional[str] = None,
    memory_source_run_dir: Optional[Path] = None,
    run_scoped_output: bool = False,
    batch_api: bool = False,
    batch_gcs_uri: Optional[str] = None,
    batch_wait: bool = False,
    workers: int = 1,
    query_workers: int = 5,
    **kwargs
) -> EvaluationReport:
    evaluator = LoCoMoEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
        execution_stage=execution_stage,
        memory_run=memory_run,
        memory_source_run_dir=memory_source_run_dir,
        run_scoped_output=run_scoped_output,
        batch_api=batch_api,
        batch_gcs_uri=batch_gcs_uri,
        batch_wait=batch_wait,
        workers=workers,
        query_workers=query_workers,
    )
    return evaluator.evaluate()
