"""MemOS agent adapter for MedMemoryBench - tree_text mode.

Uses official TreeTextMemory + SimpleStructMemReader to ensure
full alignment with memOS's tree-based memory system.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    get_usage_tracker,
)

logger = logging.getLogger(__name__)


def _ensure_memos_path():
    """Add memOS source to sys.path."""
    memos_src = Path(__file__).resolve().parent / "memOS" / "MemOS" / "src"
    if not memos_src.exists():
        raise ImportError("MemOS source folder not found at methods/memOS/MemOS/src")
    memos_src_str = str(memos_src)
    if memos_src_str not in sys.path:
        sys.path.insert(0, memos_src_str)
    # Also ensure project root utils is importable from within memOS
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


class MemOSAgent(BaseAgent):
    """Adapter using memOS tree_text mode with full official pipeline.

    Ingestion: SimpleStructMemReader.get_memory() -> TreeTextMemory.add()
    Retrieval: TreeTextMemory.search() (Searcher pipeline with BM25+reranker)
    """

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_num: int = 5,
        memos_backend: str = "openai",
        memos_model: Optional[str] = None,
        text_mem_type: str = "tree_text",
        embedding_model: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        embedding_dim: int = 512,
        embedding_provider: str = "local",
        # Neo4j config
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "memos_benchmark",
        neo4j_db_name: str = "memos_eval",
        # Search config
        search_mode: str = "fast",
        search_strategy: Optional[Dict[str, Any]] = None,
        reorganize: bool = False,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.retrieve_num = retrieve_num
        self.memos_backend = memos_backend
        self.memos_model = memos_model or model
        self.text_mem_type = text_mem_type
        self.search_mode = search_mode
        self.search_strategy = search_strategy or {"bm25": True, "cot": False}
        self.reorganize = reorganize

        # Token limits
        self.max_input_tokens = int(kwargs.get("max_input_tokens", 8000))
        self.max_context_tokens = int(kwargs.get("max_context_tokens", 120000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))
        default_memory_tokens = self.max_context_tokens - self.max_question_tokens - max_tokens - 500
        self.max_memory_tokens = max(0, int(kwargs.get("max_memory_tokens", default_memory_tokens)))

        # Embedding config
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider

        # Neo4j config
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_db_name = neo4j_db_name

        # API config for memOS LLM calls
        self._memos_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._memos_api_base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # LLM client for final QA response generation
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Memory system (lazily initialized per context)
        self._tree_memory = None
        self._mem_reader = None
        self._stored_memory_count: int = 0

        # Load memOS
        _ensure_memos_path()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _get_embedder_config(self) -> Dict[str, Any]:
        if self.embedding_provider == "local" and self.embedding_model_path:
            return {
                "backend": "sentence_transformer",
                "config": {
                    "model_name_or_path": self.embedding_model_path,
                    "embedding_dims": self.embedding_dim,
                    "trust_remote_code": True,
                },
            }
        else:
            return {
                "backend": "universal_api",
                "config": {
                    "provider": "openai",
                    "model_name_or_path": self.embedding_model or "text-embedding-3-small",
                    "api_key": self._memos_api_key,
                    "base_url": self._memos_api_base,
                    "embedding_dims": self.embedding_dim,
                },
            }

    def _get_llm_config(self) -> Dict[str, Any]:
        return {
            "backend": self.memos_backend,
            "config": {
                "model_name_or_path": self.memos_model,
                "temperature": 0,
                "max_tokens": 4096,
                "api_key": self._memos_api_key,
                "api_base": self._memos_api_base,
            },
        }

    def _init_tree_memory(self):
        """Initialize TreeTextMemory with Neo4j backend (shared-db mode)."""
        if self._tree_memory is not None:
            return

        from memos.configs.memory import MemoryConfigFactory
        from memos.memories.factory import MemoryFactory

        context_id = self._get_context_id()
        user_name = f"ctx_{context_id}"

        tree_config = {
            "extractor_llm": self._get_llm_config(),
            "dispatcher_llm": self._get_llm_config(),
            "embedder": self._get_embedder_config(),
            "graph_db": {
                "backend": "neo4j",
                "config": {
                    "uri": self.neo4j_uri,
                    "user": self.neo4j_user,
                    "password": self.neo4j_password,
                    "db_name": "neo4j",
                    "auto_create": False,
                    "use_multi_db": False,
                    "user_name": user_name,
                    "embedding_dimension": self.embedding_dim,
                },
            },
            "search_strategy": self.search_strategy,
            "reorganize": self.reorganize,
            "mode": "sync",
        }

        config_factory = MemoryConfigFactory(
            backend="tree_text",
            config=tree_config,
        )
        self._tree_memory = MemoryFactory.from_config(config_factory)
        logger.info(f"TreeTextMemory initialized for context {context_id}, user_name={user_name}")

    def _init_mem_reader(self):
        """Initialize SimpleStructMemReader for extraction."""
        if self._mem_reader is not None:
            return

        from memos.configs.mem_reader import MemReaderConfigFactory
        from memos.mem_reader.factory import MemReaderFactory

        reader_config = MemReaderConfigFactory(
            backend="simple_struct",
            config={
                "llm": self._get_llm_config(),
                "general_llm": self._get_llm_config(),
                "embedder": self._get_embedder_config(),
                "chunker": {
                    "backend": "sentence",
                    "config": {
                        "chunk_size": 512,
                        "chunk_overlap": 128,
                        "save_rawfile": False,
                    },
                },
                "chat_window_max_tokens": 2048,
                "remove_prompt_example": False,
            },
        )
        self._mem_reader = MemReaderFactory.from_config(reader_config)
        logger.info("SimpleStructMemReader initialized")

    def _get_context_id(self):
        return self._context_id if self._context_id is not None else 0

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store memory using official memOS pipeline.

        Flow: SimpleStructMemReader.get_memory() -> TreeTextMemory.add()
        """
        self._init_tree_memory()
        self._init_mem_reader()

        context_id = self._get_context_id()
        user_name = f"ctx_{context_id}"

        bounded_text = self._truncate_to_tokens(text, self.max_input_tokens)
        messages = [{"role": "user", "content": bounded_text}]

        start_time = time.time()

        try:
            # Phase tracking: memOS internal LLM calls will be recorded
            get_usage_tracker().set_phase("memorize")

            # Step 1: Extract memories using official mem_reader
            extracted_results = self._mem_reader.get_memory(
                scene_data=[messages],
                type="chat",
                info={"user_id": user_name, "session_id": ""},
                mode="fine",
                user_name=user_name,
            )

            extraction_time = time.time() - start_time

            # Flatten results (list[list[TextualMemoryItem]] -> list[TextualMemoryItem])
            all_memories = []
            for scene_memories in extracted_results:
                all_memories.extend(scene_memories)

            # Step 2: Add extracted memories to TreeTextMemory (Neo4j)
            add_start = time.time()
            if all_memories:
                added_ids = self._tree_memory.add(all_memories, user_name=user_name)
                self._stored_memory_count += len(added_ids) if added_ids else len(all_memories)
            add_time = time.time() - add_start

            total_time = time.time() - start_time

        except Exception as e:
            logger.error(f"MemOS tree_text memorize failed: {e}", exc_info=True)
            return MemoryBuildResult(
                success=False,
                method="memos",
                action="tree_text_extract_and_add",
                input_content=text,
                stored_content="",
                extraction_result=f"[Error: {e}]",
                all_passages=[],
                memory_entries=[],
                chunk_count=0,
                extra={"context_id": context_id, "error": str(e)},
            )

        self._memory_chunks.append(text)
        self._is_initialized = True

        # Build summary
        memory_entries = []
        all_stored = []
        for item in all_memories[:10]:
            mem_text = getattr(item, "memory", str(item))
            metadata = getattr(item, "metadata", {})
            if hasattr(metadata, "model_dump"):
                metadata_dict = metadata.model_dump(exclude={"embedding"})
            else:
                metadata_dict = {}
            memory_entries.append({
                "event": "ADD",
                "memory": mem_text[:200],
                "metadata": metadata_dict,
            })
            all_stored.append({
                "memory": mem_text,
                "memory_type": getattr(metadata, "memory_type", "unknown"),
            })

        extraction_summary = f"Extracted {len(all_memories)} memories via SimpleStructMemReader (fine mode)\n"
        for i, item in enumerate(all_memories[:5]):
            extraction_summary += f"  [{i+1}] [{getattr(item.metadata, 'memory_type', '?')}] {item.memory[:100]}...\n"

        return MemoryBuildResult(
            success=True,
            method="memos",
            action="tree_text_extract_and_add",
            input_content=text,
            stored_content=bounded_text,
            extraction_result=extraction_summary,
            all_passages=all_stored,
            memory_entries=memory_entries,
            chunk_count=self._stored_memory_count,
            time_cost=total_time,
            extra={
                "context_id": context_id,
                "text_mem_type": self.text_mem_type,
                "extracted_count": len(all_memories),
                "extraction_time": extraction_time,
                "add_time": add_time,
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query using official TreeTextMemory.search() pipeline.

        Uses full Searcher pipeline: TaskGoalParser -> GraphMemoryRetriever -> Reranker
        """
        self._init_tree_memory()

        context_id = self._get_context_id()
        user_name = f"ctx_{context_id}"

        bounded_question = self._truncate_to_tokens(question, self.max_question_tokens)

        start_time = time.time()

        # Phase tracking for dispatcher_llm calls during search
        get_usage_tracker().set_phase("query")

        # Use official TreeTextMemory.search()
        memory_items = self._tree_memory.search(
            query=bounded_question,
            top_k=self.retrieve_num,
            mode=self.search_mode,
            manual_close_internet=True,
            user_name=user_name,
            info={"user_id": user_name, "session_id": "eval"},
        )

        search_time = time.time() - start_time

        # Build memory context with token budget
        system_tokens = self._llm_client.count_tokens(system_message) if system_message else 0
        reserved_tokens = self.max_tokens + 500
        available_tokens = max(self.max_context_tokens - reserved_tokens - system_tokens, 0)
        question_tokens = self._llm_client.count_tokens(bounded_question)
        memory_budget = min(max(available_tokens - question_tokens, 0), self.max_memory_tokens)

        retrieved_memories: List[Dict[str, Any]] = []
        memory_blocks: List[str] = []
        used_tokens = 0

        for item in memory_items:
            mem_text = str(getattr(item, "memory", "") or "").strip()
            if not mem_text:
                continue

            mem_tokens = self._llm_client.count_tokens(mem_text)
            if used_tokens + mem_tokens > memory_budget:
                remaining = memory_budget - used_tokens
                if remaining > 100:
                    truncated = self._truncate_to_tokens(mem_text, remaining)
                    memory_blocks.append(truncated)
                    retrieved_memories.append({
                        "memory": truncated[:2000],
                        "type": "tree_text_search",
                        "truncated": True,
                    })
                break

            memory_blocks.append(mem_text)
            used_tokens += mem_tokens
            metadata = getattr(item, "metadata", None)
            retrieved_memories.append({
                "memory": mem_text[:2000],
                "type": "tree_text_search",
                "memory_type": getattr(metadata, "memory_type", "unknown") if metadata else "unknown",
            })

        # Build final prompt
        full_question = bounded_question
        if memory_blocks:
            memory_context = "\n\n".join(
                [f"[Memory {idx + 1}]\n{block}" for idx, block in enumerate(memory_blocks)]
            )
            full_question = f"[Retrieved MemOS Memories]\n{memory_context}\n\n[Question]\n{bounded_question}"

        # Generate response (tracked by llm_client)
        messages = format_messages(full_question, system_message)
        response = self._llm_client.chat(messages)

        total_time = time.time() - start_time

        return AgentResponse(
            output=response.content,
            query_time=total_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "memos",
                "text_mem_type": self.text_mem_type,
                "search_mode": self.search_mode,
                "search_time": search_time,
                "tokens_used": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
            },
        )

    def reset(self) -> None:
        """Reset agent state. Clears Neo4j data for current context."""
        super().reset()

        if self._tree_memory is not None:
            try:
                context_id = self._get_context_id()
                user_name = f"ctx_{context_id}"
                self._tree_memory.delete_all(user_name=user_name)
            except Exception as e:
                logger.warning(f"Failed to clear Neo4j data: {e}")

        self._tree_memory = None
        self._mem_reader = None
        self._stored_memory_count = 0

    def set_context_id(self, context_id: int) -> None:
        """Set context ID. Re-initializes memory system for new context."""
        old_id = self._context_id
        super().set_context_id(context_id)

        # Re-init memory system if context changed
        if old_id != context_id:
            self._tree_memory = None
            self._mem_reader = None

    @property
    def memory_size(self) -> int:
        return self._stored_memory_count
