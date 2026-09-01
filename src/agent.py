import os
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

from src.config import MethodConfig, DatasetConfig, get_api_config
from methods.base import MemoryBuildResult, AgentResponse
from utils.llm_client import (
    get_usage_tracker,
    is_batch_provider,
    is_gemini_provider,
)


class AgentManager:

    SUPPORTED_METHODS = {
        "long_context": ("methods.long_context", "LongContextAgent"),
        "embedding_rag": ("methods.embedding_rag", "EmbeddingRAGAgent"),
        "bm25_rag": ("methods.bm25_rag", "BM25RAGAgent"),
        "amem_fix": ("methods.amem_fix_agent", "AMemFixAgent"),
        "amem_test": ("methods.amem_test_agent", "AMemTestAgent"),
        "event_state": ("methods.event_state_agent", "EventStateAgent"),
        "amem": ("methods.amem_agent", "AMemAgent"),
        "letta": ("methods.letta_agent", "LettaAgent"),
        "memos": ("methods.memos_agent", "MemOSAgent"),
        "mirix": ("methods.mirix_agent", "MIRIXAgent"),
        "mem0": ("methods.mem0_agent", "Mem0Agent"),
        "smart_mem0": ("methods.smart_mem0_agent", "SmartMem0Agent"),
        "mem1": ("methods.mem1_agent", "Mem1Agent"),
        "memrl": ("methods.memrl_agent", "MemRLAgent"),
        "zep": ("methods.zep_agent", "ZepAgent"),
        "graph_rag": ("methods.graph_rag", "GraphRAGAgent"),
        "lightmem": ("methods.lightmem_agent", "LightMemAgent"),
        "remem": ("methods.remem_agent", "RememAgent"),
        "hipporag": ("methods.hipporag_agent", "HippoRAGAgent"),
        "q2q": ("methods.q2q_agent", "Q2QBenchAgent"),
    }

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        *,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
        batch_manifest_dir: Optional[Path] = None,
        batch_config_hash: str = "",
        batch_progress_callback: Optional[Callable[[str], None]] = None,
        workers: int = 1,
        query_only: bool = False,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.method_name = method_config.method_name.lower()
        self._batch_api = batch_api
        self._batch_gcs_uri = batch_gcs_uri
        self._batch_wait = batch_wait
        self._batch_manifest_dir = batch_manifest_dir
        self._batch_config_hash = batch_config_hash
        self._batch_progress_callback = batch_progress_callback
        self._workers = max(1, int(workers))
        self._query_only = bool(query_only)

        self._api_config = get_api_config()

        self._agent = None
        self._context_id: Optional[int] = None
        self._agent_start_time = time.time()

        self._initialize_agent()

    def _initialize_agent(self) -> None:
        matched_method = self._matched_method_key(self.method_name)

        module_path, class_name = self.SUPPORTED_METHODS[matched_method]
        self._agent = self._create_agent_instance(module_path, class_name, matched_method)

    @classmethod
    def _matched_method_key(cls, method_name: str) -> str:
        """Return the adapter key selected for a configured method name."""
        normalized_name = method_name.lower()
        for method_key in cls.SUPPORTED_METHODS:
            if method_key in normalized_name:
                return method_key
        return "long_context"

    @classmethod
    def resolve_effective_init_params(
        cls,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        *,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
        batch_manifest_dir: Optional[Path] = None,
        batch_config_hash: str = "",
        query_only: bool = False,
        workers: int = 1,
    ) -> tuple[str, Dict[str, Any]]:
        """Resolve adapter kwargs without constructing an agent or making provider calls."""
        manager = cls.__new__(cls)
        manager.method_config = method_config
        manager.dataset_config = dataset_config
        manager.method_name = method_config.method_name.lower()
        manager._batch_api = batch_api
        manager._batch_gcs_uri = batch_gcs_uri
        manager._batch_wait = batch_wait
        manager._batch_manifest_dir = batch_manifest_dir
        manager._batch_config_hash = batch_config_hash
        manager._batch_progress_callback = None
        manager._workers = max(1, int(workers))
        manager._query_only = bool(query_only)
        manager._api_config = get_api_config()
        method_key = cls._matched_method_key(manager.method_name)
        return method_key, manager._build_agent_params(method_key)

    def _create_agent_instance(self, module_path: str, class_name: str, method_key: str):
        import importlib
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        init_params = self._build_agent_params(method_key)
        return agent_class(**init_params)

    def _resolve_model_runtime(
        self,
        model_config,
    ) -> tuple[Optional[str], str, Dict[str, Any]]:
        """Resolve credentials, endpoint, and provider options for one model block."""
        api_key = model_config.api_key
        provider = model_config.provider.lower()
        if not is_gemini_provider(provider):
            if provider == "modal":
                api_key = api_key or getattr(self._api_config, "modal_api_key", "")
            elif provider == "openrouter":
                api_key = api_key or getattr(
                    self._api_config, "openrouter_api_key", ""
                )
            else:
                api_key = api_key or getattr(self._api_config, "openai_api_key", "")

        configured_base_url = getattr(self._api_config, "openai_base_url", "")
        if provider == "modal":
            configured_base_url = getattr(self._api_config, "modal_base_url", "")
        elif provider == "openrouter":
            configured_base_url = getattr(self._api_config, "openrouter_base_url", "")

        client_kwargs: Dict[str, Any] = {}
        reasoning_effort = getattr(model_config, "reasoning_effort", None)
        if reasoning_effort is not None:
            client_kwargs["reasoning_effort"] = reasoning_effort
        nim_thinking_enabled = getattr(model_config, "nim_thinking_enabled", None)
        if nim_thinking_enabled is not None:
            client_kwargs["nim_thinking_enabled"] = nim_thinking_enabled
        nim_reasoning_budget = getattr(model_config, "nim_reasoning_budget", None)
        if nim_reasoning_budget is not None:
            client_kwargs["nim_reasoning_budget"] = nim_reasoning_budget
        if provider == "openrouter":
            if model_config.openrouter_provider is not None:
                client_kwargs["provider_routing"] = dict(
                    model_config.openrouter_provider
                )
            if model_config.openrouter_service_tier is not None:
                client_kwargs["service_tier"] = model_config.openrouter_service_tier

        return api_key, model_config.base_url or configured_base_url, client_kwargs

    def _build_agent_params(self, method_key: str) -> Dict[str, Any]:
        model_config = self.method_config.model
        # Merge sections only at the adapter boundary; snapshot identity keeps
        # build and retrieval ownership separate.
        agent_params = {
            **(self.method_config.build_config or {}),
            **(self.method_config.retrieval_config or {}),
        }
        if not agent_params:
            agent_params = self.method_config.agent_params or {}
        env_embedding_model = os.environ.get("DEFAULT_EMBEDDING_MODEL")
        env_embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
        use_env_embedding = bool(
            env_embedding_model
            and env_embedding_provider in ("local", "huggingface")
        )
        amem_embedding_model = agent_params.get("amem_embedding_model", "all-MiniLM-L6-v2")
        use_env_amem_embedding = (
            use_env_embedding
            and amem_embedding_model.startswith(("models/", "./", "/"))
            and not Path(amem_embedding_model).exists()
        )

        effective_max_tokens = model_config.max_completion_tokens or model_config.max_tokens
        api_key, base_url, llm_client_kwargs = self._resolve_model_runtime(model_config)

        params = {
            "model": model_config.name,
            "temperature": model_config.temperature,
            "max_tokens": effective_max_tokens,
            "provider": model_config.provider,
            "api_key": api_key,
            "base_url": base_url,
        }
        if llm_client_kwargs:
            params["llm_client_kwargs"] = llm_client_kwargs

        build_model_config = self.method_config.memorize_model
        if build_model_config is not None:
            build_api_key, build_base_url, build_client_kwargs = (
                self._resolve_model_runtime(build_model_config)
            )
            build_max_tokens = (
                build_model_config.max_completion_tokens
                or build_model_config.max_tokens
            )
        else:
            build_api_key = build_base_url = None
            build_client_kwargs = {}
            build_max_tokens = None

        if (
            self._batch_api
            and is_batch_provider(model_config.provider)
            and method_key in {"lightmem", "remem", "graph_rag", "memrl"}
        ):
            params.update({
                "vertex_batch_enabled": True,
                "vertex_batch_gcs_uri": self._batch_gcs_uri,
                "vertex_batch_wait": self._batch_wait,
                "vertex_batch_manifest_dir": str(self._batch_manifest_dir) if self._batch_manifest_dir else None,
                "vertex_batch_config_hash": self._batch_config_hash,
                "vertex_batch_progress_callback": self._batch_progress_callback,
            })

        if method_key == "long_context":
            params.update({
                "max_context_tokens": agent_params.get("max_context_tokens", 100000),
                "truncation_strategy": agent_params.get("truncation_strategy", "oldest_first"),
            })

        elif method_key == "embedding_rag":
            params.update({
                "top_k": agent_params.get("top_k", 5),
                "chunk_size": agent_params.get("chunk_size", 512),
                "chunk_overlap": agent_params.get("chunk_overlap", 50),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "bm25_rag":
            params.update({
                "top_k": agent_params.get("top_k", 5),
                "k1": agent_params.get("k1", 1.5),
                "b": agent_params.get("b", 0.75),
                "language": agent_params.get("language", "auto"),
                "chunk_size": agent_params.get("chunk_size", 512),
                "chunk_overlap": agent_params.get("chunk_overlap", 50),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })

        elif method_key == "mem0":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "chunk_size_tokens": agent_params.get("chunk_size_tokens"),
                "max_context_tokens": agent_params.get("max_context_tokens"),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "amem_fix":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 10),
                "amem_backend": agent_params.get(
                    "amem_backend",
                    build_model_config.provider if build_model_config else "openai",
                ),
                "amem_model": agent_params.get(
                    "amem_model",
                    build_model_config.name if build_model_config else model_config.name,
                ),
                "amem_embedding_model": agent_params.get(
                    "amem_embedding_model", "all-MiniLM-L6-v2"
                ),
                "amem_evo_threshold": agent_params.get("amem_evo_threshold", 100),
                "amem_max_tokens": agent_params.get(
                    "amem_max_tokens", build_max_tokens or 1000
                ),
                "amem_temperature": agent_params.get(
                    "amem_temperature",
                    build_model_config.temperature if build_model_config else 0.7,
                ),
                "amem_retry_temperature": agent_params.get("amem_retry_temperature", 0.3),
                "amem_connectivity_temperature": agent_params.get("amem_connectivity_temperature", 0.0),
                "amem_build_max_context_tokens": agent_params.get(
                    "amem_build_max_context_tokens", 200000
                ),
                "amem_max_context_tokens": agent_params.get("amem_max_context_tokens", 200000),
                "amem_chunk_size_tokens": agent_params.get("amem_chunk_size_tokens"),
                "amem_query_keywords": agent_params.get("amem_query_keywords", True),
                "amem_expand_links": agent_params.get("amem_expand_links", True),
            })

        elif method_key == "amem_test":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 10),
                "amem_backend": agent_params.get(
                    "amem_backend",
                    build_model_config.provider if build_model_config else "openai",
                ),
                "amem_model": agent_params.get(
                    "amem_model",
                    build_model_config.name if build_model_config else model_config.name,
                ),
                "amem_embedding_model": agent_params.get(
                    "amem_embedding_model", "all-MiniLM-L6-v2"
                ),
                "amem_evo_threshold": agent_params.get("amem_evo_threshold", 100),
                "amem_max_tokens": agent_params.get(
                    "amem_max_tokens", build_max_tokens or 1000
                ),
                "amem_temperature": agent_params.get(
                    "amem_temperature",
                    build_model_config.temperature if build_model_config else 0.7,
                ),
                "amem_retry_temperature": agent_params.get("amem_retry_temperature", 0.3),
                "amem_connectivity_temperature": agent_params.get("amem_connectivity_temperature", 0.0),
                "amem_build_max_context_tokens": agent_params.get(
                    "amem_build_max_context_tokens", 200000
                ),
                "amem_max_context_tokens": agent_params.get("amem_max_context_tokens", 200000),
                "amem_chunk_size_tokens": agent_params.get("amem_chunk_size_tokens"),
                "amem_query_keywords": agent_params.get("amem_query_keywords", True),
                "amem_regex_intent_conditioning": agent_params.get(
                    "amem_regex_intent_conditioning", True
                ),
                "amem_expand_links": agent_params.get("amem_expand_links", True),
                "amem_note_level": agent_params.get("amem_note_level", "turn"),
                "amem_original_evolution": agent_params.get(
                    "amem_original_evolution", True
                ),
                "amem_typed_relations": agent_params.get("amem_typed_relations", True),
                "amem_workers": getattr(self, "_workers", 1),
                "amem_query_only": getattr(self, "_query_only", False),
                "amem_typed_retrieval": agent_params.get(
                    "amem_typed_retrieval",
                    bool(agent_params.get("amem_typed_relations", True)),
                ),
                "amem_relation_candidate_count": agent_params.get(
                    "amem_relation_candidate_count", 5
                ),
                "amem_typed_expansion_count": agent_params.get(
                    "amem_typed_expansion_count", 5
                ),
                "amem_expand_related": agent_params.get("amem_expand_related", False),
                "amem_relation_min_confidence": agent_params.get(
                    "amem_relation_min_confidence", 0.5
                ),
                "amem_relation_temperature": agent_params.get("amem_relation_temperature", 0.2),
                "amem_temporal_transition_min_confidence": agent_params.get(
                    "amem_temporal_transition_min_confidence", 0.5
                ),
                "amem_temporal_state": agent_params.get("amem_temporal_state", False),
                "amem_temporal_retrieval": agent_params.get(
                    "amem_temporal_retrieval",
                    bool(agent_params.get("amem_temporal_state", False)),
                ),
                "amem_temporal_ordering": agent_params.get(
                    "amem_temporal_ordering", True
                ),
                "amem_temporal_expansion_count": agent_params.get(
                    "amem_temporal_expansion_count", 5
                ),
                "amem_provenance": agent_params.get("amem_provenance", False),
                "amem_provenance_retrieval": agent_params.get(
                    "amem_provenance_retrieval",
                    bool(agent_params.get("amem_provenance", False)),
                ),
                "amem_provenance_max_evidence": agent_params.get(
                    "amem_provenance_max_evidence", 10
                ),
                "amem_provenance_inject_raw_text": agent_params.get(
                    "amem_provenance_inject_raw_text", False
                ),
                "amem_hybrid_retrieval": agent_params.get(
                    "amem_hybrid_retrieval", False
                ),
                "amem_hybrid_candidate_count": agent_params.get(
                    "amem_hybrid_candidate_count", 50
                ),
                "amem_hybrid_rrf_k": agent_params.get("amem_hybrid_rrf_k", 60.0),
                "amem_hybrid_dense_weight": agent_params.get(
                    "amem_hybrid_dense_weight", 1.0
                ),
                "amem_hybrid_bm25_weight": agent_params.get(
                    "amem_hybrid_bm25_weight", 1.0
                ),
                "amem_hybrid_entity_weight": agent_params.get(
                    "amem_hybrid_entity_weight", 1.0
                ),
                "amem_hybrid_timestamp_weight": agent_params.get(
                    "amem_hybrid_timestamp_weight", 1.0
                ),
                "amem_hybrid_state_weight": agent_params.get(
                    "amem_hybrid_state_weight", 1.0
                ),
                "amem_hybrid_graph_weight": agent_params.get(
                    "amem_hybrid_graph_weight", 1.0
                ),
                "amem_graph_ranking_mode": agent_params.get(
                    "amem_graph_ranking_mode", "fixed_bfs"
                ),
                "amem_graph_alpha": agent_params.get("amem_graph_alpha", 0.85),
                "amem_graph_iterations": agent_params.get(
                    "amem_graph_iterations", 20
                ),
                "amem_graph_tolerance": agent_params.get(
                    "amem_graph_tolerance", 1e-6
                ),
                "amem_graph_supersede_weight": agent_params.get(
                    "amem_graph_supersede_weight", 1.25
                ),
                "amem_graph_conflict_weight": agent_params.get(
                    "amem_graph_conflict_weight", 0.75
                ),
                "amem_graph_refine_weight": agent_params.get(
                    "amem_graph_refine_weight", 1.15
                ),
                "amem_graph_support_weight": agent_params.get(
                    "amem_graph_support_weight", 1.0
                ),
                "amem_graph_related_weight": agent_params.get(
                    "amem_graph_related_weight", 0.5
                ),
                "amem_chain_selection": agent_params.get(
                    "amem_chain_selection", False
                ),
                "amem_chain_candidate_count": agent_params.get(
                    "amem_chain_candidate_count", 50
                ),
                "amem_chain_evidence_count": agent_params.get(
                    "amem_chain_evidence_count", 30
                ),
                "amem_chain_max_hops": agent_params.get(
                    "amem_chain_max_hops", 2
                ),
                "amem_chain_max_groups": agent_params.get(
                    "amem_chain_max_groups", 3
                ),
                "amem_chain_relevance_weight": agent_params.get(
                    "amem_chain_relevance_weight", 1.0
                ),
                "amem_chain_coverage_weight": agent_params.get(
                    "amem_chain_coverage_weight", 1.0
                ),
                "amem_chain_connectivity_weight": agent_params.get(
                    "amem_chain_connectivity_weight", 0.35
                ),
                "amem_chain_path_weight": agent_params.get(
                    "amem_chain_path_weight", 0.75
                ),
                "amem_chain_temporal_weight": agent_params.get(
                    "amem_chain_temporal_weight", 0.5
                ),
                "amem_chain_redundancy_weight": agent_params.get(
                    "amem_chain_redundancy_weight", 0.25
                ),
            })

        elif method_key == "event_state":
            params.update(agent_params)
            params.update({
                "event_state_workers": getattr(self, "_workers", 1),
                "memory_model": build_model_config.name if build_model_config else model_config.name,
                "memory_provider": build_model_config.provider if build_model_config else model_config.provider,
                "memory_temperature": build_model_config.temperature if build_model_config else model_config.temperature,
                "memory_max_tokens": build_max_tokens or effective_max_tokens,
                "memory_api_key": build_api_key if build_model_config else api_key,
                "memory_base_url": build_base_url if build_model_config else base_url,
                "memory_llm_client_kwargs": build_client_kwargs if build_model_config else llm_client_kwargs,
            })
            if self.method_config.embedding:
                params.update({
                    "embedding_model": self.method_config.embedding.model,
                    "embedding_provider": self.method_config.embedding.provider,
                    "embedding_model_path": self.method_config.embedding.model_path,
                    "embedding_api_key": self.method_config.embedding.api_key,
                    "embedding_base_url": self.method_config.embedding.base_url,
                })

        elif method_key == "amem":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "amem_backend": agent_params.get(
                    "amem_backend",
                    build_model_config.provider if build_model_config else "openai",
                ),
                "amem_model": agent_params.get(
                    "amem_model",
                    build_model_config.name if build_model_config else model_config.name,
                ),
                "amem_embedding_model": (
                    env_embedding_model
                    if use_env_amem_embedding
                    else amem_embedding_model
                ),
                "amem_evo_threshold": agent_params.get("amem_evo_threshold", 100),
                "amem_max_tokens": agent_params.get(
                    "amem_max_tokens", build_max_tokens
                ),
                "amem_temperature": agent_params.get(
                    "amem_temperature",
                    build_model_config.temperature if build_model_config else 0.7,
                ),
                "amem_retry_temperature": agent_params.get("amem_retry_temperature", 0.3),
                "amem_connectivity_temperature": agent_params.get("amem_connectivity_temperature", 0.0),
                "amem_build_max_context_tokens": agent_params.get(
                    "amem_build_max_context_tokens", 200000
                ),
                "amem_max_context_tokens": agent_params.get("amem_max_context_tokens", 200000),
                "amem_chunk_size_tokens": agent_params.get("amem_chunk_size_tokens"),
            })

        elif method_key == "letta":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "context_window": agent_params.get("context_window", 128000),
                "embedding_model": agent_params.get("embedding_model", "embedding-3"),
                "embedding_provider": agent_params.get("embedding_provider", "openai"),
                "embedding_dim": agent_params.get("embedding_dim", 2048),
                "embedding_chunk_size": agent_params.get("embedding_chunk_size", 300),
                "memory_persona": agent_params.get(
                    "memory_persona",
                    "I am an assistant helping with medical QA while preserving long-term memory.",
                ),
                "memory_human": agent_params.get(
                    "memory_human",
                    "The user is a patient in a longitudinal medical dialogue setting.",
                ),
                # Memorize chunking parameters
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 6000),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
                # Query truncation parameters
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim

        elif method_key == "memos":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "memos_backend": agent_params.get("memos_backend", "openai"),
                "memos_model": agent_params.get("memos_model", model_config.name),
                "memos_temperature": agent_params.get("memos_temperature", 0.0),
                "memos_max_tokens": agent_params.get("memos_max_tokens", 4096),
                "text_mem_type": agent_params.get("text_mem_type", "general_text"),
                "embedding_dim": agent_params.get("embedding_dim"),
                "observability_search": agent_params.get("observability_search", True),
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "mirix":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 2500),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
                "query_memory_item_tokens": agent_params.get("query_memory_item_tokens", 300),
                "query_memory_context_tokens": agent_params.get("query_memory_context_tokens", 1800),
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Query mode: use MIRIX native send_message for memory-aware responses
                "use_native_query": agent_params.get("use_native_query", True),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # For local embedding: device specification (cuda, cpu, mps)
                if self.method_config.embedding.provider == "local":
                    params["embedding_device"] = agent_params.get("embedding_device", None)

        elif method_key == "mem1":
            # MEM1 uses vLLM for model serving
            params.update({
                "vllm_url": agent_params.get("vllm_url", "http://localhost:8014"),
                "model_path": agent_params.get("model_path", model_config.name),
                "max_state_tokens": agent_params.get("max_state_tokens", 4096),
                "max_input_tokens": agent_params.get("max_input_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 8192),
                "use_chat_api": agent_params.get("use_chat_api", True),
            })

        elif method_key == "memrl":
            params.update({
                # Retrieval configuration
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "candidate_top_k": agent_params.get("candidate_top_k", 12),
                "similarity_threshold": agent_params.get("similarity_threshold", 0.2),
                # Memory building configuration
                "max_new_items_per_memorize": agent_params.get("max_new_items_per_memorize", 8),
                "max_experience_tokens": agent_params.get("max_experience_tokens", 700),
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 6000),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
                "query_memory_item_tokens": agent_params.get("query_memory_item_tokens", 300),
                "query_memory_context_tokens": agent_params.get("query_memory_context_tokens", 1800),
                "max_concurrent_api_calls": agent_params.get("max_concurrent_api_calls", 4),
                "memrl_keyword_temperature": agent_params.get("memrl_keyword_temperature", 0.0),
                "memrl_keyword_max_tokens": agent_params.get("memrl_keyword_max_tokens", 100),
                "memrl_script_temperature": agent_params.get("memrl_script_temperature", 0.7),
                "memrl_script_max_tokens": agent_params.get("memrl_script_max_tokens", 500),
                "memrl_reflection_temperature": agent_params.get("memrl_reflection_temperature", 0.3),
                "memrl_reflection_max_tokens": agent_params.get("memrl_reflection_max_tokens"),
                "memrl_extractor_temperature": agent_params.get("memrl_extractor_temperature", 0.0),
                "memrl_extractor_max_tokens": agent_params.get("memrl_extractor_max_tokens", 4096),
                # Strategy configuration
                "build_strategy": agent_params.get("build_strategy", "proceduralization"),
                "retrieve_strategy": agent_params.get("retrieve_strategy", "query"),
                "update_strategy": agent_params.get("update_strategy", "adjustment"),
                # Q-learning / RL configuration
                "epsilon": agent_params.get("epsilon", 0.1),
                "gamma": agent_params.get("gamma", 0.0),
                "learning_rate": agent_params.get("learning_rate", 0.2),
                "initial_q": agent_params.get("initial_q", 0.5),
                "q_init_pos": agent_params.get("q_init_pos", 0.5),
                "q_init_neg": agent_params.get("q_init_neg", 0.0),
                "success_reward": agent_params.get("success_reward", 1.0),
                "failure_reward": agent_params.get("failure_reward", -1.0),
                "weight_sim": agent_params.get("weight_sim", 0.5),
                "weight_q": agent_params.get("weight_q", 0.5),
                "utility_mix_lambda": agent_params.get("utility_mix_lambda", 0.6),
                # Token limits
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                # Embedding (default, may be overridden below)
                "embedding_model": agent_params.get("embedding_model", "all-MiniLM-L6-v2"),
                "embedding_provider": agent_params.get("embedding_provider", "local"),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "zep":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "chunk_size": agent_params.get("chunk_size", 512),
            })
            # Zep API Key
            zep_api_key = agent_params.get("zep_api_key") or os.environ.get("ZEP_API_KEY")
            if zep_api_key:
                params["zep_api_key"] = zep_api_key

            # Azure OpenAI
            if self._api_config.use_azure:
                params.update({
                    "use_azure": True,
                    "azure_endpoint": self._api_config.azure_endpoint,
                    "azure_api_key": self._api_config.azure_api_key,
                    "azure_api_version": self._api_config.azure_api_version,
                })

        elif method_key == "graph_rag":
            params.update({
                "top_k": agent_params.get("top_k", 5),
                "chunk_size": agent_params.get("chunk_size", 4096),
                "chunk_overlap": agent_params.get("chunk_overlap", 200),
                "edges_threshold": agent_params.get("edges_threshold", 0.8),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "lightmem":
            params.update({
                # Retrieval configuration
                "retrieve_num": agent_params.get("retrieve_num", 5),
                # LightMem core feature switches
                "pre_compress": agent_params.get("pre_compress", False),
                "topic_segment": agent_params.get("topic_segment", True),
                # Strategy configuration
                "index_strategy": agent_params.get("index_strategy", "embedding"),
                "retrieve_strategy": agent_params.get("retrieve_strategy", "embedding"),
                "update_mode": agent_params.get("update_mode", "offline"),
                "extraction_mode": agent_params.get("extraction_mode", "flat"),
                "messages_use": agent_params.get("messages_use", "user_only"),
                # LightMem internal LLM settings
                "lightmem_temperature": agent_params.get("lightmem_temperature", 0.1),
                "lightmem_max_tokens": agent_params.get("lightmem_max_tokens", 2000),
                "lightmem_top_p": agent_params.get("lightmem_top_p", 0.1),
                "lightmem_buffer_max_tokens": agent_params.get("lightmem_buffer_max_tokens", 4096),
                # Token limits
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            # Embedding configuration
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # Embedding API configuration (for openai/huggingface_api providers)
                if self.method_config.embedding.api_key:
                    params["embedding_api_key"] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params["embedding_base_url"] = self.method_config.embedding.base_url

        elif method_key == "remem":
            # ReMem: Reasoning with Episodic Memory
            params.update({
                # Information extraction method
                "extract_method": agent_params.get("extract_method", "episodic_gist"),
                # Graph configuration
                "is_directed_graph": agent_params.get("is_directed_graph", False),
                "synonymy_edge_sim_threshold": agent_params.get("synonymy_edge_sim_threshold", 0.8),
                "synonymy_edge_topk": agent_params.get("synonymy_edge_topk", 10),
                # Retrieval configuration
                "retrieval_top_k": agent_params.get("retrieval_top_k", 20),
                "qa_top_k": agent_params.get("qa_top_k", 10),
                "linking_top_k": agent_params.get("linking_top_k", 5),
                "damping": agent_params.get("damping", 0.5),
                "passage_node_weight": agent_params.get("passage_node_weight", 0.05),
                # Agent configuration
                "use_agent": agent_params.get("use_agent", False),
                "agent_fixed_tools": agent_params.get("agent_fixed_tools", False),
                "agent_max_steps": agent_params.get("agent_max_steps", 5),
                # Cache configuration
                "use_cache": agent_params.get("use_cache", True),
                "force_index_from_scratch": agent_params.get("force_index_from_scratch", False),
                "save_openie": agent_params.get("save_openie", True),
                # API concurrency configuration
                "extraction_max_workers": agent_params.get("extraction_max_workers", 5),
                # Text preprocessing
                "text_preprocessor_class_name": agent_params.get(
                    "text_preprocessor_class_name", "SentenceWindowPreprocessor"
                ),
                # Chunking configuration
                "chunk_size_tokens": agent_params.get("chunk_size_tokens", 8000),
                "chunk_overlap_tokens": agent_params.get("chunk_overlap_tokens", 200),
                # Embedding settings
                "embedding_batch_size": agent_params.get("embedding_batch_size", 16),
                "embedding_max_seq_len": agent_params.get("embedding_max_seq_len", 512),
                # Token limits
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Working directory (optional)
                "working_dir": agent_params.get("working_dir", None),
            })
            # Embedding configuration
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # Embedding API configuration (for openai/api providers)
                if self.method_config.embedding.api_key:
                    params["embedding_api_key"] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params["embedding_base_url"] = self.method_config.embedding.base_url

        elif method_key == "hipporag":
            # HippoRAG 2: Graph-Based RAG Framework
            params.update({
                # OpenIE mode
                "openie_mode": agent_params.get("openie_mode", "online"),
                # Graph configuration
                "is_directed_graph": agent_params.get("is_directed_graph", False),
                "synonymy_edge_sim_threshold": agent_params.get("synonymy_edge_sim_threshold", 0.8),
                "synonymy_edge_topk": agent_params.get("synonymy_edge_topk", 2047),
                # Retrieval configuration
                "linking_top_k": agent_params.get("linking_top_k", 5),
                "retrieval_top_k": agent_params.get("retrieval_top_k", 200),
                "qa_top_k": agent_params.get("qa_top_k", 5),
                "damping": agent_params.get("damping", 0.5),
                "passage_node_weight": agent_params.get("passage_node_weight", 0.05),
                # Cache configuration
                "force_index_from_scratch": agent_params.get("force_index_from_scratch", False),
                "force_openie_from_scratch": agent_params.get("force_openie_from_scratch", False),
                "save_openie": agent_params.get("save_openie", True),
                # Chunking configuration
                "chunk_size_tokens": agent_params.get("chunk_size_tokens", 8000),
                "chunk_overlap_tokens": agent_params.get("chunk_overlap_tokens", 200),
                # Embedding settings
                "embedding_batch_size": agent_params.get("embedding_batch_size", 16),
                "embedding_max_seq_len": agent_params.get("embedding_max_seq_len", 512),
                # Token limits
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Working directory (optional)
                "working_dir": agent_params.get("working_dir", None),
            })
            # Embedding configuration
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # Embedding API configuration (for openai/api providers)
                if self.method_config.embedding.api_key:
                    params["embedding_api_key"] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params["embedding_base_url"] = self.method_config.embedding.base_url

        elif method_key == "q2q":
            params.update({
                "q2q_project_path": agent_params.get("q2q_project_path", ""),
                "alpha": agent_params.get("alpha", 0.7),
                "top_k_per_sub": agent_params.get("top_k_per_sub", 20),
                "top_n": agent_params.get("top_n", 5),
                "top_k_q2c": agent_params.get("top_k_q2c", 3),
                "num_fake_queries": agent_params.get("num_fake_queries", 10),
                "storage_backend": agent_params.get("storage_backend", "chromadb"),
                "language": agent_params.get("language", "zh"),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                "embedding_device": agent_params.get("embedding_device", "cpu"),
                "version_threshold": agent_params.get("version_threshold", 0.80),
                "version_chain_depth": agent_params.get("version_chain_depth", 3),
                "fq_confidence_threshold": agent_params.get("fq_confidence_threshold", 0.80),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        if build_model_config is not None and method_key in {
            "amem", "amem_fix", "amem_test"
        }:
            params.update({
                "amem_api_key": build_api_key,
                "amem_base_url": build_base_url,
                "amem_llm_client_kwargs": build_client_kwargs,
            })

        return params

    def send_message(
        self,
        message: str,
        memorizing: bool = False,
        context_id: Optional[int] = None,
        is_last_session: bool = False,
        **kwargs,
    ) -> Any:
        if context_id is not None and context_id != self._context_id:
            self._context_id = context_id
            self._agent.set_context_id(context_id)

        if memorizing:
            return self._handle_memorize(
                message,
                is_last_session=is_last_session,
                **kwargs,
            )
        else:
            return self._handle_query(message, **kwargs)

    def supports_batch_queries(self) -> bool:
        """Return whether this adapter can separate retrieval from final generation."""
        adapter_support = getattr(self._agent, "supports_batch_queries", None)
        if callable(adapter_support) and not adapter_support():
            return False
        return bool(
            is_batch_provider(self.method_config.model.provider)
            and hasattr(self._agent, "prepare_batch_query")
            and hasattr(self._agent, "finalize_batch_query")
        )

    def supports_memory_snapshots(self) -> bool:
        """Return whether the adapter can persist exact retrieval state."""
        support = getattr(self._agent, "supports_memory_snapshots", None)
        return bool(callable(support) and support())

    def export_memory_state(self, context_id: Optional[int] = None) -> Dict[str, Any]:
        """Export the active adapter's retrieval state."""
        if not self.supports_memory_snapshots():
            raise RuntimeError(f"{self.method_name} does not support memory snapshots")
        if context_id is not None:
            self.set_context_id(context_id)
        return self._agent.export_memory_state(context_id=context_id)

    def import_memory_state(
        self,
        state: Dict[str, Any],
        context_id: Optional[int] = None,
    ) -> None:
        """Restore the active adapter's retrieval state."""
        if not self.supports_memory_snapshots():
            raise RuntimeError(f"{self.method_name} does not support memory snapshots")
        self._agent.import_memory_state(state, context_id=context_id)
        resolved_context_id = state.get("context_id") if context_id is None else context_id
        if resolved_context_id is not None:
            self._context_id = int(resolved_context_id)

    def get_batch_llm_client(self):
        """Expose the managed client only for the evaluator's batch transport."""
        if not self.supports_batch_queries():
            return None
        return getattr(self._agent, "_llm_client", None)

    def prepare_batch_query(self, message: str, **kwargs) -> Dict[str, Any]:
        """Prepare local retrieval and an immutable final-answer request."""
        if not self.supports_batch_queries():
            raise RuntimeError(f"{self.method_name} does not support batch queries")
        context_id = kwargs.pop("context_id", None)
        if context_id is not None and context_id != self._context_id:
            self._context_id = context_id
            self._agent.set_context_id(context_id)
        tracker = get_usage_tracker()
        tracker.set_phase("query")
        with tracker.scope("query.retrieval_preparation"):
            return self._agent.prepare_batch_query(message, **kwargs)

    def supports_staged_queries(self) -> bool:
        """Return whether retrieval and final generation can run separately."""
        return bool(
            hasattr(self._agent, "prepare_batch_query")
            and hasattr(self._agent, "finalize_batch_query")
            and getattr(self._agent, "_llm_client", None) is not None
        )

    def prepare_query(self, message: str, **kwargs) -> Dict[str, Any]:
        """Run retrieval and freeze the exact final-answer request."""
        if not self.supports_staged_queries():
            raise RuntimeError(f"{self.method_name} does not support staged queries")
        context_id = kwargs.pop("context_id", None)
        if context_id is not None and context_id != self._context_id:
            self.set_context_id(context_id)
        tracker = get_usage_tracker()
        tracker.set_phase("query")
        from utils.templates import get_template_manager
        template_manager = get_template_manager(self.dataset_config.dataset_name)
        system_message = template_manager.get_system_message()
        started_at = time.time()
        with tracker.scope("query.retrieval_preparation"):
            prepared = self._agent.prepare_batch_query(
                message,
                system_message=system_message,
                **kwargs,
            )
        return {"prepared": prepared, "started_at": started_at}

    def answer_prepared_query(self, staged_query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate from a frozen retrieval request and preserve legacy timing."""
        if not self.supports_staged_queries():
            raise RuntimeError(f"{self.method_name} does not support staged queries")
        prepared = staged_query["prepared"]
        tracker = get_usage_tracker()
        tracker.set_phase("query")
        with tracker.scope("query.answer_realtime"):
            response = self._agent._llm_client.chat(prepared["messages"])
        finalized = self._agent.finalize_batch_query(prepared, response.content)
        return {
            "output": finalized.output,
            "query_time": time.time() - staged_query["started_at"],
            "retrieved_count": finalized.retrieved_count,
            "retrieved_memories": finalized.retrieved_memories,
            "answer_usage": {
                "transport": "realtime",
                "input_tokens": getattr(response, "input_tokens", 0),
                "output_tokens": getattr(response, "output_tokens", 0),
                "visible_output_tokens": getattr(response, "visible_output_tokens", 0),
                "thinking_tokens": getattr(response, "thinking_tokens", 0),
                "finish_reason": getattr(response, "finish_reason", None),
                "duration_seconds": getattr(response, "latency", None),
            },
        }

    def finalize_batch_query(
        self,
        prepared: Dict[str, Any],
        content: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> AgentResponse:
        """Convert one Vertex output row to the method's normal response shape."""
        response = self._agent.finalize_batch_query(prepared, content)
        record_usage = getattr(self._agent, "record_batch_query_usage", None)
        if callable(record_usage) and input_tokens is not None and output_tokens is not None:
            record_usage(response, input_tokens, output_tokens)
        return response

    def _handle_memorize(
        self,
        message: str,
        is_last_session: bool = False,
        **kwargs,
    ) -> MemoryBuildResult:
        get_usage_tracker().set_phase("memorize")
        start_time = time.time()

        result = self._agent.memorize(
            message,
            is_last_session=is_last_session,
            **kwargs,
        )

        memory_time = time.time() - start_time

        if isinstance(result, MemoryBuildResult):
            result.time_cost = memory_time
            return result

        if isinstance(result, dict):
            return MemoryBuildResult(
                success=result.get("success", True),
                method=result.get("method", self.method_name),
                action=result.get("action", "add_to_memory"),
                input_content=message,
                stored_content=result.get("stored_content", message),
                memory_entries=result.get("memory_entries", []),
                chunk_count=result.get("chunk_count", self._agent.memory_size),
                time_cost=memory_time,
                extra={}
            )

        return MemoryBuildResult(
            success=True,
            method=self.method_name,
            action="add_to_memory",
            input_content=message,
            stored_content=message,
            memory_entries=[],
            chunk_count=self._agent.memory_size,
            time_cost=memory_time,
        )

    def supports_staged_memory(self) -> bool:
        """Return whether memory construction exposes preparation and commit stages."""
        return bool(
            hasattr(self._agent, "prepare_memory_sessions")
            and hasattr(self._agent, "commit_prepared_memory")
        )

    def prepare_memory_sessions(self, message: str, **kwargs):
        if not self.supports_staged_memory():
            raise RuntimeError(f"{self.method_name} does not support staged memory")
        context_id = kwargs.get("context_id")
        if context_id is not None and context_id != self._context_id:
            self._context_id = context_id
            self._agent.set_context_id(context_id)
        get_usage_tracker().set_phase("memorize")
        return self._agent.prepare_memory_sessions(message, **kwargs)

    def commit_prepared_memory(self, prepared_sessions, **kwargs):
        if not self.supports_staged_memory():
            raise RuntimeError(f"{self.method_name} does not support staged memory")
        get_usage_tracker().set_phase("memorize")
        return self._agent.commit_prepared_memory(prepared_sessions, **kwargs)

    def _handle_query(self, message: str, **kwargs) -> Dict[str, Any]:
        get_usage_tracker().set_phase("query")
        from utils.templates import get_template_manager
        template_manager = get_template_manager(self.dataset_config.dataset_name)
        system_message = template_manager.get_system_message()

        start_time = time.time()

        with get_usage_tracker().scope("query.answer_realtime"):
            response = self._agent.query(
                message,
                system_message=system_message,
                **kwargs,
            )

        query_time = time.time() - start_time

        return {
            "output": response.output,
            "query_time": query_time,
            "retrieved_count": response.retrieved_count,
            "retrieved_memories": response.retrieved_memories, 
            "extra": response.extra,
        }

    def reset(self) -> None:
        if self._agent:
            self._agent.reset()
        self._context_id = None
        self._agent_start_time = time.time()

    def set_context_id(self, context_id: int) -> None:
        self._context_id = context_id
        if self._agent:
            self._agent.set_context_id(context_id)

    def get_info(self) -> Dict[str, Any]:
        info = {
            "method_name": self.method_name,
            "context_id": self._context_id,
        }
        if self._agent:
            info.update(self._agent.get_info())
        return info


def create_agent_manager(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
) -> AgentManager:
    return AgentManager(method_config, dataset_config)


def list_available_methods() -> List[str]:
    return list(AgentManager.SUPPORTED_METHODS.keys())
