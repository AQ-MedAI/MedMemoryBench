"""MemRL Agent - Thin adapter delegating to official MemoryService (HLE runner pattern)."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    LLMResponse,
)

logger = logging.getLogger(__name__)


def _ensure_memrl_path():
    """Ensure MemRL and MemOS are importable."""
    memrl_root = Path(__file__).resolve().parent / "MemRL"
    if not memrl_root.exists():
        raise ImportError("MemRL source folder not found at methods/MemRL")
    if str(memrl_root) not in sys.path:
        sys.path.insert(0, str(memrl_root))

    memos_src = Path(__file__).resolve().parent / "memOS" / "MemOS" / "src"
    if memos_src.exists() and str(memos_src) not in sys.path:
        sys.path.insert(0, str(memos_src))


class TrackedLLMProvider:
    """Bridge: MedMemoryBench BaseLLMClient -> MemRL BaseLLM interface (duck-typed)."""

    def __init__(self, llm_client: BaseLLMClient, model_name: str):
        self._client = llm_client
        self._model_name = model_name
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency = 0.0

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        response = self._client.chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        self._call_count += 1
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens
        self._total_latency += response.latency
        return response.content

    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        prompt = (
            f"Extract up to {max_keywords} key concepts from the following text. "
            f"Return only the keywords separated by commas.\n\nText: {text}\n\nKeywords:"
        )
        response = self.generate([{"role": "user", "content": prompt}], temperature=0, max_tokens=100)
        keywords = [k.strip().strip('"\'').lower() for k in response.split(',')]
        return [k for k in keywords if k and len(k) > 1][:max_keywords]

    def generate_script(self, trajectory: str) -> str:
        # Only called if build_strategy != trajectory; kept for interface completeness
        prompt = (
            "Create a concise high-level script (3-5 steps) from this trajectory:\n\n"
            f"{trajectory}\n\nScript:"
        )
        return self.generate([{"role": "user", "content": prompt}], temperature=0, max_tokens=500)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_latency": round(self._total_latency, 3),
        }

    def reset_stats(self) -> None:
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency = 0.0


class MemRLAgent(BaseAgent):
    """MemRL agent using official MemoryService with trajectory build strategy.

    All memory logic (build/retrieve/update) is delegated to MemRL's MemoryService.
    This class is purely an adapter for the MedMemoryBench evaluation framework.
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
        # Retrieval params
        retrieve_num: int = 8,
        candidate_top_k: int = 20,
        similarity_threshold: float = 0.1,
        # Chunk params
        memorize_chunk_tokens: int = 500,
        memorize_chunk_overlap_tokens: int = 50,
        max_task_desc_chars: int = 512,
        query_memory_context_tokens: int = 6000,
        # Strategy
        build_strategy: str = "trajectory",
        retrieve_strategy: str = "query",
        update_strategy: str = "adjustment",
        # Q-learning
        epsilon: float = 0.1,
        gamma: float = 0.0,
        learning_rate: float = 0.2,
        initial_q: float = 0.5,
        q_init_pos: float = 0.5,
        q_init_neg: float = 0.0,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        weight_sim: float = 0.6,
        weight_q: float = 0.4,
        # Embedding
        embedding_model: str = "bge-small-en-v1.5",
        embedding_provider: str = "local",
        embedding_model_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        # Dataset awareness
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # Store config
        self.retrieve_num = retrieve_num
        self.candidate_top_k = candidate_top_k
        self.similarity_threshold = similarity_threshold
        self.memorize_chunk_tokens = memorize_chunk_tokens
        self.memorize_chunk_overlap_tokens = memorize_chunk_overlap_tokens
        self.max_task_desc_chars = max_task_desc_chars
        self.query_memory_context_tokens = query_memory_context_tokens
        self.max_context_tokens = int(kwargs.get("max_context_tokens", 120000))

        # Strategy
        self.build_strategy = build_strategy
        self.retrieve_strategy = retrieve_strategy
        self.update_strategy = update_strategy

        # RL params
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.initial_q = initial_q
        self.q_init_pos = q_init_pos
        self.q_init_neg = q_init_neg
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.weight_sim = weight_sim
        self.weight_q = weight_q

        # Embedding
        self.embedding_model = embedding_model
        self.embedding_provider_name = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim

        # Dataset awareness: disable Q-weight for per-question evaluation modes
        # where Q-learning cannot accumulate across multiple queries
        self.dataset_name = dataset_name
        if dataset_name and "longmemeval" in dataset_name.lower():
            if weight_q > 0:
                logger.info(
                    f"[MemRLAgent] LongMemEval independent mode detected: "
                    f"overriding weight_q={weight_q}->0.0, weight_sim={weight_sim}->1.0 "
                    f"(Q-learning ineffective with 1 query per agent lifecycle)"
                )
                self.weight_q = 0.0
                self.weight_sim = 1.0

        # API config
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._provider = provider

        # LLM client
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key, base_url=base_url,
        )

        # State
        self._last_retrieved_ids: List[str] = []
        self._memory_service = None
        self._tracked_llm = None
        self._temp_dir = None

        self._init_memrl()

    def _init_memrl(self) -> None:
        """Initialize MemoryService using official MemRL components."""
        _ensure_memrl_path()

        from memrl.providers.embedding import LocalEmbedder
        from memrl.service.memory_service import MemoryService
        from memrl.service.strategies import StrategyConfiguration
        from memrl.service.value_driven import RLConfig

        # Temp dir for MOS artifacts
        self._temp_dir = tempfile.mkdtemp(prefix="memrl_agent_")

        # MOS config
        mos_config_path = self._create_mos_config()

        # LLM bridge
        self._tracked_llm = TrackedLLMProvider(self._llm_client, self.model)

        # Embedding provider
        model_path = self.embedding_model_path or self.embedding_model
        self._memrl_embedder = LocalEmbedder(model_name=model_path)

        # Strategy config
        strategy_config = StrategyConfiguration.from_strings(
            build=self.build_strategy,
            retrieve=self.retrieve_strategy,
            update=self.update_strategy,
        )

        # RL config
        rl_config = RLConfig(
            epsilon=self.epsilon,
            alpha=self.learning_rate,
            gamma=self.gamma,
            q_init_pos=self.q_init_pos,
            q_init_neg=self.q_init_neg,
            success_reward=self.success_reward,
            failure_reward=self.failure_reward,
            sim_threshold=self.similarity_threshold,
            topk=self.candidate_top_k,
            weight_sim=self.weight_sim,
            weight_q=self.weight_q,
        )

        # User ID
        user_id = (
            f"memrl_{self._context_id}" if self._context_id is not None
            else f"memrl_{os.getpid()}"
        )

        # Initialize MemoryService
        init_kwargs = dict(
            mos_config_path=mos_config_path,
            llm_provider=self._tracked_llm,
            embedding_provider=self._memrl_embedder,
            strategy_config=strategy_config,
            user_id=user_id,
            num_workers=4,
            enable_value_driven=True,
            rl_config=rl_config,
            add_similarity_threshold=0.92,
        )
        if self.embedding_dim:
            init_kwargs["vector_dimension"] = self.embedding_dim
        self._memory_service = MemoryService(**init_kwargs)

        logger.info(
            f"[MemRLAgent] Initialized: strategy={self.build_strategy}/{self.retrieve_strategy}, "
            f"embedding={model_path}, user={user_id}"
        )

    def _create_mos_config(self) -> str:
        """Create temporary MOS config JSON (following run_llb.py pattern)."""
        embedder_config = {
            "backend": "sentence_transformer",
            "config": {
                "model_name_or_path": self.embedding_model_path or self.embedding_model,
            },
        }

        config = {
            "user_manager": {
                "backend": "sqlite",
                "config": {"db_path": os.path.join(self._temp_dir, "users.db")},
            },
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": self.model,
                    "api_key": self._api_key,
                    "api_base": self._base_url or "https://api.openai.com/v1",
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": self.model,
                            "api_key": self._api_key,
                            "api_base": self._base_url or "https://api.openai.com/v1",
                        },
                    },
                    "embedder": embedder_config,
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
        }

        config_path = os.path.join(self._temp_dir, "mos_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return config_path

    def _split_text_into_chunks(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Split text into token-bounded chunks."""
        if not text.strip():
            return []
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        step = max(1, max_tokens - overlap_tokens)
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start += step
        return chunks

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store text using MemRL's build_memory (trajectory strategy = no LLM call)."""
        start_time = time.time()

        chunks = self._split_text_into_chunks(
            text,
            max_tokens=self.memorize_chunk_tokens,
            overlap_tokens=self.memorize_chunk_overlap_tokens,
        )

        if not chunks:
            return MemoryBuildResult(
                success=False, method="memrl", action="memorize",
                input_content=text, stored_content="", memory_entries=[], chunk_count=0,
            )

        memory_ids = []
        for i, chunk in enumerate(chunks):
            try:
                # task_description: short text for embedding retrieval key
                task_desc = chunk[:self.max_task_desc_chars]
                # trajectory: full chunk content stored as full_content
                memory_id = self._memory_service.build_memory(
                    task_description=task_desc,
                    trajectory=chunk,
                    metadata={"source_benchmark": "medmemorybench", "chunk_index": i},
                )
                memory_ids.append(memory_id)
            except Exception as e:
                logger.warning(f"[MemRL] build_memory failed for chunk {i}: {e}")

        total_time = time.time() - start_time
        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=len(memory_ids) > 0,
            method="memrl",
            action="build_memory",
            input_content=text,
            stored_content=text,
            memory_entries=[{"memory_id": mid, "index": i} for i, mid in enumerate(memory_ids)],
            chunk_count=len(chunks),
            time_cost=total_time,
            extraction_result=f"Built {len(memory_ids)} memories from {len(chunks)} chunks (strategy={self.build_strategy})",
            all_passages=[{"memory_id": mid} for mid in memory_ids],
            extra={
                "total_memories": len(memory_ids),
                "strategy": self.build_strategy,
                "llm_stats": self._tracked_llm.get_stats() if self._tracked_llm else {},
            },
        )

    def query(self, question: str, system_message: Optional[str] = None, **kwargs) -> AgentResponse:
        """Retrieve memories and answer using HLE runner context-building pattern."""
        start_time = time.time()

        # Retrieve via official MemoryService
        retrieved_memories = []
        retrieval_result = {}
        try:
            result = self._memory_service.retrieve_query(
                task_description=question,
                k=self.candidate_top_k,
                threshold=self.similarity_threshold,
            )
            # Returns (dict, sim_list) tuple
            if isinstance(result, tuple):
                retrieval_result, _ = result
            else:
                retrieval_result = result

            selected = retrieval_result.get("selected", [])
            for mem in selected[:self.retrieve_num]:
                content = mem.get("content") or ""
                if not content:
                    md = mem.get("metadata")
                    if hasattr(md, "model_extra"):
                        content = md.model_extra.get("full_content", "")
                    elif isinstance(md, dict):
                        content = md.get("full_content", "")
                retrieved_memories.append({
                    "memory_id": mem.get("memory_id"),
                    "content": content,
                    "similarity": mem.get("similarity", 0.0),
                    "q_value": mem.get("q_estimate", self.initial_q),
                    "score": mem.get("score", 0.0),
                })
        except Exception as e:
            logger.warning(f"[MemRL] Retrieval failed: {e}")

        # Save IDs for feedback
        self._last_retrieved_ids = [m["memory_id"] for m in retrieved_memories if m.get("memory_id")]

        # Build context (HLE pattern: categorize by success/failure)
        memory_context = self._build_memory_context(retrieved_memories)

        # Construct prompt
        if memory_context:
            full_question = f"[Retrieved Memories]\n{memory_context}\n\n[Question]\n{question}"
        else:
            full_question = question

        # Generate answer (graceful fallback on API failure)
        messages = format_messages(full_question, system_message)
        try:
            response = self._llm_client.chat(messages)
        except Exception as e:
            logger.error(f"[MemRL] LLM call failed after retries, skipping: {e}")
            total_time = time.time() - start_time
            return AgentResponse(
                output="",
                query_time=total_time,
                retrieved_count=len(retrieved_memories),
                retrieved_memories=[],
                extra={"method": "memrl", "error": str(e)},
            )

        total_time = time.time() - start_time

        return AgentResponse(
            output=response.content,
            query_time=total_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=[
                {
                    "memory": m.get("content", "")[:500],
                    "memory_id": m.get("memory_id"),
                    "similarity": m.get("similarity", 0),
                    "q_value": m.get("q_value", 0),
                    "score": m.get("score", 0),
                }
                for m in retrieved_memories
            ],
            extra={
                "method": "memrl",
                "strategy": self.retrieve_strategy,
                "candidates_count": len(retrieval_result.get("candidates", [])),
                "simmax": retrieval_result.get("simmax", 0),
                "tokens_used": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
            },
        )

    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build memory context string with token budget (HLE pattern)."""
        if not memories:
            return ""

        blocks = []
        used_tokens = 0
        max_per_item = self.query_memory_context_tokens // max(len(memories), 1)

        for i, mem in enumerate(memories):
            content = str(mem.get("content", "")).strip()
            if not content:
                continue

            # Truncate individual memory to budget
            content_tokens = self.count_tokens(content)
            if content_tokens > max_per_item:
                tokens = self._tokenizer.encode(content)[:max_per_item]
                content = self._tokenizer.decode(tokens)

            if used_tokens + self.count_tokens(content) > self.query_memory_context_tokens:
                break

            q_val = mem.get("q_value", self.initial_q)
            sim = mem.get("similarity", 0)
            blocks.append(f"[Memory {i+1} | Q={q_val:.2f}, sim={sim:.3f}]\n{content}")
            used_tokens += self.count_tokens(content)

        return "\n\n".join(blocks)

    def on_query_feedback(self, score: float, is_correct: bool, **kwargs) -> None:
        """Update Q-values via official MemoryService.update_values()."""
        if not self._memory_service or not self._last_retrieved_ids:
            return
        try:
            self._memory_service.update_values(
                successes=[1.0 if is_correct else 0.0],
                retrieved_ids_list=[self._last_retrieved_ids],
            )
            logger.debug(
                f"[MemRL] Q-update: {len(self._last_retrieved_ids)} memories, correct={is_correct}"
            )
        except Exception as e:
            logger.warning(f"[MemRL] Q-value update failed: {e}")
        self._last_retrieved_ids = []

    def reset(self) -> None:
        """Reset agent and reinitialize MemoryService for next persona."""
        import gc
        super().reset()

        self._memory_service = None
        if self._tracked_llm:
            self._tracked_llm.reset_stats()
        self._tracked_llm = None
        self._last_retrieved_ids = []

        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"[MemRLAgent] Failed to clean temp dir: {e}")
            self._temp_dir = None

        gc.collect()
        self._context_id = None

    def set_context_id(self, context_id: int) -> None:
        """Set context and reinitialize MemoryService for new persona."""
        old_context = self._context_id
        super().set_context_id(context_id)
        if old_context != context_id and self._memory_service is not None:
            logger.info(f"[MemRLAgent] Context {old_context} -> {context_id}, reinitializing")
            self.reset()
            self._context_id = context_id
            self._init_memrl()

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "memrl_config": {
                "build_strategy": self.build_strategy,
                "retrieve_strategy": self.retrieve_strategy,
                "retrieve_num": self.retrieve_num,
                "candidate_top_k": self.candidate_top_k,
                "memorize_chunk_tokens": self.memorize_chunk_tokens,
                "epsilon": self.epsilon,
                "weight_sim": self.weight_sim,
                "weight_q": self.weight_q,
            },
            "embedding": {
                "model": self.embedding_model,
                "provider": self.embedding_provider_name,
            },
        })
        if self._tracked_llm:
            info["memrl_llm_stats"] = self._tracked_llm.get_stats()
        return info
