import time
from typing import Dict, Any, Optional, List

from src.config import MethodConfig, DatasetConfig, get_api_config
from methods.base import MemoryBuildResult, AgentResponse
from utils.llm_client import get_usage_tracker


class AgentManager:

    SUPPORTED_METHODS = {
        "long_context": ("methods.long_context", "LongContextAgent"),
        "embedding_rag": ("methods.embedding_rag", "EmbeddingRAGAgent"),
        "bm25_rag": ("methods.bm25_rag", "BM25RAGAgent"),
        "amem": ("methods.amem_agent", "AMemAgent"),
        "letta": ("methods.letta_agent", "LettaAgent"),
        "memos": ("methods.memos_agent", "MemOSAgent"),
        "mirix": ("methods.mirix_agent", "MIRIXAgent"),
        "mem0": ("methods.mem0_agent", "Mem0Agent"),
        "mem1": ("methods.mem1_agent", "Mem1Agent"),
        "memrl": ("methods.memrl_agent", "MemRLAgent"),
        "zep": ("methods.zep_agent", "ZepAgent"),
        "graph_rag": ("methods.graph_rag", "GraphRAGAgent"),
        "lightmem": ("methods.lightmem_agent", "LightMemAgent"),
        "remem": ("methods.remem_agent", "RememAgent"),
        "hipporag": ("methods.hipporag_agent", "HippoRAGAgent"),
        "q2q_v1": ("methods.q2q_v1_agent", "Q2QV1BenchAgent"),
        "q2q": ("methods.q2q_agent", "Q2QBenchAgent"),
    }

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.method_name = method_config.method_name.lower()

        self._api_config = get_api_config()

        self._agent = None
        self._context_id: Optional[int] = None
        self._agent_start_time = time.time()

        self._initialize_agent()

    def _initialize_agent(self) -> None:
        matched_method = None
        for method_key in self.SUPPORTED_METHODS:
            if method_key in self.method_name:
                matched_method = method_key
                break

        if matched_method is None:
            matched_method = "long_context"

        module_path, class_name = self.SUPPORTED_METHODS[matched_method]
        self._agent = self._create_agent_instance(module_path, class_name, matched_method)

    def _create_agent_instance(self, module_path: str, class_name: str, method_key: str):
        import importlib
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        init_params = self._build_agent_params(method_key)
        return agent_class(**init_params)

    def _build_agent_params(self, method_key: str) -> Dict[str, Any]:
        model_config = self.method_config.model
        agent_params = self.method_config.agent_params or {}

        effective_max_tokens = model_config.max_completion_tokens or model_config.max_tokens

        params = {
            "model": model_config.name,
            "temperature": model_config.temperature,
            "max_tokens": effective_max_tokens,
            "provider": model_config.provider,
            "api_key": (
                model_config.api_key
                or self._api_config.openai_api_key
            ),
            "base_url": (
                model_config.base_url
                or self._api_config.openai_base_url
            ),
        }

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
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim

        elif method_key == "amem":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "amem_backend": agent_params.get("amem_backend", "openai"),
                "amem_model": agent_params.get("amem_model", model_config.name),
                "amem_embedding_model": agent_params.get("amem_embedding_model", "all-MiniLM-L6-v2"),
                "amem_evo_threshold": agent_params.get("amem_evo_threshold", 100),
                "amem_max_tokens": agent_params.get("amem_max_tokens", 2000),
                "max_context_tokens": agent_params.get("max_context_tokens", 100000),
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
                "text_mem_type": agent_params.get("text_mem_type", "general_text"),
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if agent_params.get("max_memory_tokens") is not None:
                params["max_memory_tokens"] = agent_params["max_memory_tokens"]
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim

        elif method_key == "mirix":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 2500),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
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
                # Dataset awareness for evaluation mode adaptation
                "dataset_name": self.dataset_config.dataset_name,
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim

        elif method_key == "zep":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "chunk_size": agent_params.get("chunk_size", 512),
            })
            # Zep API Key
            import os
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
                # Dataset awareness for prompt template selection
                "dataset_name": self.dataset_config.dataset_name,
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
                # Dataset awareness for prompt template routing
                "dataset_name": self.dataset_config.dataset_name,
                # Information extraction method
                "extract_method": agent_params.get("extract_method", "episodic_gist"),
                # Graph configuration
                "is_directed_graph": agent_params.get("is_directed_graph", False),
                "synonymy_edge_sim_threshold": agent_params.get("synonymy_edge_sim_threshold", 0.8),
                "synonymy_edge_topk": agent_params.get("synonymy_edge_topk", 2047),
                # Retrieval configuration
                "retrieval_top_k": agent_params.get("retrieval_top_k", 200),
                "qa_top_k": agent_params.get("qa_top_k", 5),
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
                    "text_preprocessor_class_name", "TextPreprocessor"
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

        elif method_key == "q2q_v1":
            params.update({
                "q2q_project_path": agent_params.get("q2q_project_path", ""),
                # Retrieval (v1 params)
                "alpha": agent_params.get("alpha", 0.7),
                "top_k_per_sub": agent_params.get("top_k_per_sub", 20),
                "top_n": agent_params.get("top_n", 5),
                "top_k_q2c": agent_params.get("top_k_q2c", 3),
                "num_fake_queries": agent_params.get("num_fake_queries", 10),
                "version_threshold": agent_params.get("version_threshold", 0.80),
                "version_chain_depth": agent_params.get("version_chain_depth", 3),
                "fq_confidence_threshold": agent_params.get("fq_confidence_threshold", 0.80),
                "paragraph_top_k": agent_params.get("paragraph_top_k", 3),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # General
                "storage_backend": agent_params.get("storage_backend", "chromadb"),
                "language": agent_params.get("language", "zh"),
                "embedding_device": agent_params.get("embedding_device", "cpu"),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "q2q":
            params.update({
                "q2q_project_path": agent_params.get("q2q_project_path", ""),
                # Retrieval
                "alpha": agent_params.get("alpha", 0.7),
                "top_k_per_sub": agent_params.get("top_k_per_sub", 20),
                "top_n": agent_params.get("top_n", 5),
                "top_k_q2c": agent_params.get("top_k_q2c", 3),
                "num_fake_queries": agent_params.get("num_fake_queries", 10),
                "chain_depth": agent_params.get("chain_depth", 3),
                "prop_top_k_per_fq": agent_params.get("prop_top_k_per_fq", 3),
                "binding_verify_threshold": agent_params.get("binding_verify_threshold", 0.8),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Preprocessing
                "proposition_mode": agent_params.get("proposition_mode", "llm"),
                "compressor_model": agent_params.get("compressor_model", ""),
                "compressor_device": agent_params.get("compressor_device", "cpu"),
                "perplexity_threshold": agent_params.get("perplexity_threshold", 5.0),
                "nlp_inference_model": agent_params.get("nlp_inference_model", ""),
                # Graph
                "evolve_threshold": agent_params.get("evolve_threshold", 0.6),
                "diverge_low": agent_params.get("diverge_low", 0.2),
                "diverge_high": agent_params.get("diverge_high", 0.6),
                "max_parents": agent_params.get("max_parents", 3),
                "max_chain_length": agent_params.get("max_chain_length", 6),
                # General
                "storage_backend": agent_params.get("storage_backend", "chromadb"),
                "language": agent_params.get("language", "zh"),
                "embedding_device": agent_params.get("embedding_device", "cpu"),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

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
            return self._handle_memorize(message, is_last_session=is_last_session)
        else:
            return self._handle_query(message)

    def _handle_memorize(self, message: str, is_last_session: bool = False) -> MemoryBuildResult:
        get_usage_tracker().set_phase("memorize")
        start_time = time.time()

        result = self._agent.memorize(message, is_last_session=is_last_session)

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

    def _handle_query(self, message: str) -> Dict[str, Any]:
        get_usage_tracker().set_phase("query")
        from utils.templates import get_template_manager
        template_manager = get_template_manager(self.dataset_config.dataset_name)
        system_message = template_manager.get_system_message()

        start_time = time.time()

        response = self._agent.query(message, system_message=system_message)

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

    def send_query_feedback(self, score: float, is_correct: bool, **kwargs) -> None:
        if self._agent:
            self._agent.on_query_feedback(score=score, is_correct=is_correct, **kwargs)

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
