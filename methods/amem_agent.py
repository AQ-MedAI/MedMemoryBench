"""A-Mem agent adapter for MedMemoryBench.

Directly uses the original AgenticMemorySystem from memory_layer.py,
which implements the full A-Mem pipeline:
  1. MemoryNote with analyze_content (keyword/context/tag extraction via LLM)
  2. process_memory (single-call evolution with JSON schema)
  3. SimpleEmbeddingRetriever for vector search
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, format_messages, BaseLLMClient, get_usage_tracker

logger = logging.getLogger(__name__)


class AMemAgent(BaseAgent):
    """Adapter bridging native A-Mem AgenticMemorySystem to BaseAgent interface."""

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
        amem_backend: str = "openai",
        amem_model: Optional[str] = None,
        amem_embedding_model: str = "all-MiniLM-L6-v2",
        amem_evo_threshold: int = 100,
        amem_max_tokens: int = 2000,
        max_context_tokens: int = 100000,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.retrieve_num = retrieve_num
        self.amem_backend = amem_backend
        self.amem_model = amem_model or model
        self.amem_embedding_model = amem_embedding_model
        self.amem_evo_threshold = amem_evo_threshold
        # amem_max_tokens: controls A-Mem internal LLM (analyze_content + process_memory)
        # query-time answer generation uses self.max_tokens (from framework config)
        self.amem_max_tokens = amem_max_tokens
        self.max_context_tokens = max_context_tokens

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._api_base = base_url or os.environ.get("OPENAI_BASE_URL")

        # LLM client for query-time response generation
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Per-context memory systems
        self._amem_systems: Dict[int, Any] = {}
        self._amem_class = self._load_amem_class()

    def _load_amem_class(self):
        """Load AgenticMemorySystem from methods/amem/A-mem/memory_layer.py."""
        amem_dir = Path(__file__).resolve().parent / "amem" / "A-mem"
        if not amem_dir.exists():
            raise ImportError(f"A-mem source folder not found at {amem_dir}")

        amem_dir_str = str(amem_dir)
        if amem_dir_str not in sys.path:
            sys.path.insert(0, amem_dir_str)

        module = importlib.import_module("memory_layer")
        return getattr(module, "AgenticMemorySystem")

    def _get_context_id(self) -> int:
        return self._context_id if self._context_id is not None else 0

    def _get_memory_system(self, context_id: int):
        """Get or create AgenticMemorySystem for the given context."""
        system = self._amem_systems.get(context_id)
        if system is not None:
            return system

        system = self._amem_class(
            model_name=self.amem_embedding_model,
            llm_backend=self.amem_backend,
            llm_model=self.amem_model,
            evo_threshold=self.amem_evo_threshold,
            api_key=self._api_key,
            api_base=self._api_base,
            max_tokens=self.amem_max_tokens,
            usage_tracker=get_usage_tracker(),
        )
        self._amem_systems[context_id] = system
        logger.info(
            "Created A-Mem system for context %d: model=%s, embedding=%s, evo_threshold=%d",
            context_id, self.amem_model, self.amem_embedding_model, self.amem_evo_threshold,
        )
        return system

    @staticmethod
    def _parse_turns(text: str):
        """Split session text into (turn_lines, timestamp).

        Session format produced by LongMemEvalSession.to_memory_text():
            [2024-03-15]
            user: ...
            assistant: ...
        Returns the list of individual turn strings and the date header (may be None).
        """
        import re
        timestamp = None
        turns = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            date_match = re.match(r'^\[(.+)\]$', line)
            if date_match:
                timestamp = date_match.group(1)
                continue
            if re.match(r'^(user|assistant|human|system)\s*:', line, re.IGNORECASE):
                turns.append(line)
        return turns, timestamp

    def _expand_query_to_keywords(self, memory_system, question: str) -> str:
        """Extract retrieval keywords from question via A-Mem's LLM controller.

        Mirrors generate_query_llm() from the official test_advanced.py.
        Falls back to the original question on any error.
        """
        import json
        prompt = (
            "Given the following question, generate several keywords, "
            "using 'cosmos' as the separator.\n\n"
            f"Question: {question}\n\n"
            "Format your response as a JSON object with a \"keywords\" field.\n"
            "Example: {\"keywords\": \"keyword1, keyword2, keyword3\"}"
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"keywords": {"type": "string"}},
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        try:
            raw = memory_system.llm_controller.llm.get_completion(
                prompt, response_format=response_format
            )
            keywords = json.loads(raw).get("keywords", "").strip()
            return keywords if keywords else question
        except Exception:
            return question

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store a session as per-turn MemoryNotes via AgenticMemorySystem.add_note.

        Official A-Mem granularity (per test_advanced.py): one note per conversation
        turn, not one note per session.  The session text is split into individual
        turn lines before storage so that analyze_content / process_memory operate
        on short, focused units and embedding retrieval stays precise.
        """
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)

        turns, timestamp = self._parse_turns(text)
        if not turns:
            turns = [text]  # fallback: store whole text as one note

        note_ids = []
        for turn_text in turns:
            note_id = memory_system.add_note(content=turn_text, time=timestamp)
            note_ids.append(str(note_id))

        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="amem",
            action="add_note",
            input_content=text,
            stored_content="\n".join(turns),
            extraction_result=f"split into {len(turns)} turns",
            memory_entries=[
                {"event": "ADD", "memory": t[:400], "id": nid}
                for t, nid in zip(turns, note_ids)
            ],
            all_passages=[
                {"event": "ADD", "memory": t[:400]}
                for t in turns
            ],
            chunk_count=len(self._memory_chunks),
            extra={
                "context_id": context_id,
                "note_ids": note_ids,
                "inserted_count": len(note_ids),
                "total_memories": len(memory_system.memories),
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query using official A-Mem retrieval pipeline + LLM response generation.

        Mirrors the official test_advanced.py answer_question():
          1. LLM extracts retrieval keywords from the question (query expansion)
          2. find_related_memories_raw() retrieves notes + neighbor link expansion
          3. LLM generates the answer from the retrieved context
        """
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)

        # Step 1: query expansion via LLM keyword extraction (official A-Mem behavior)
        retrieval_query = self._expand_query_to_keywords(memory_system, question)

        # Step 2: retrieve with neighbor link expansion (find_related_memories_raw)
        memory_str = memory_system.find_related_memories_raw(retrieval_query, k=self.retrieve_num)

        # Truncate retrieved context to max_context_tokens
        if memory_str and memory_str.strip():
            token_count = len(self._tokenizer.encode(memory_str))
            if token_count > self.max_context_tokens:
                encoded = self._tokenizer.encode(memory_str)[:self.max_context_tokens]
                memory_str = self._tokenizer.decode(encoded)

        # Build full question with retrieved context
        if memory_str and memory_str.strip():
            full_question = f"[Retrieved Memories]\n{memory_str}\n\n{question}"
        else:
            full_question = question

        # Step 3: generate response
        messages = format_messages(full_question, system_message)
        response = self._llm_client.chat(messages)

        retrieved_memories: List[Dict[str, Any]] = []
        if memory_str and memory_str.strip():
            retrieved_memories.append({
                "memory": memory_str,
                "type": "amem_retrieval_raw",
                "retrieval_query": retrieval_query,
            })

        return AgentResponse(
            output=response.content,
            retrieved_count=1 if retrieved_memories else 0,
            retrieved_memories=retrieved_memories,
            extra={"method": "amem", "context_id": context_id, "retrieval_query": retrieval_query},
        )

    def reset(self) -> None:
        """Reset agent state including all A-Mem systems."""
        super().reset()
        self._amem_systems = {}

    def set_context_id(self, context_id: int) -> None:
        super().set_context_id(context_id)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "retrieve_num": self.retrieve_num,
            "amem_backend": self.amem_backend,
            "amem_model": self.amem_model,
            "amem_embedding_model": self.amem_embedding_model,
            "amem_evo_threshold": self.amem_evo_threshold,
            "amem_max_tokens": self.amem_max_tokens,
            "max_context_tokens": self.max_context_tokens,
            "active_contexts": list(self._amem_systems.keys()),
        })
        return info
