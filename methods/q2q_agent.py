"""Q2Q Agent - Proposition-based dual-path memory retrieval adapter for MedMemoryBench.

Imports and delegates to the Q2Q framework at ../Agent_Memory/Q2Q.
Handles namespace isolation (Q2Q uses 'src/' which conflicts with
MedMemoryBench's 'src/'), model caching, and temp storage management.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    get_usage_tracker,
    LLMRetryExhaustedError,
)

logger = logging.getLogger(__name__)

# ============================================================
# Q2Q Module Isolation
# ============================================================

_q2q_loaded = False
_Q2QAgent = None
_Q2QConfig = None
_q2q_llm_module = None
_q2q_LLMRetryExhaustedError = None
_q2q_embedding_factory = None
_q2q_kept_refs = {}


def _load_q2q_modules(q2q_root: str):
    global _q2q_loaded, _Q2QAgent, _Q2QConfig, _q2q_llm_module
    global _q2q_LLMRetryExhaustedError, _q2q_embedding_factory, _q2q_kept_refs

    if _q2q_loaded:
        return

    q2q_root = str(Path(q2q_root).resolve())
    if not os.path.isdir(q2q_root):
        raise ImportError(f"Q2Q project not found: {q2q_root}")

    saved_src = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src."):
            saved_src[key] = sys.modules.pop(key)

    sys.path.insert(0, q2q_root)

    try:
        import agent as _q2q_agent_mod
        import src.utils.llm_client as _q2q_llm_mod
        import src.embedding.factory as _q2q_emb_factory
        import src.preprocessing.compressor as _q2q_comp_mod

        _Q2QAgent = _q2q_agent_mod.Q2QAgent
        _Q2QConfig = _q2q_agent_mod.Q2QConfig
        _q2q_llm_module = _q2q_llm_mod
        _q2q_LLMRetryExhaustedError = _q2q_llm_mod.LLMRetryExhaustedError
        _q2q_embedding_factory = _q2q_emb_factory

    finally:
        for key in list(sys.modules.keys()):
            if key == "src" or key.startswith("src.") or key == "agent":
                _q2q_kept_refs[key] = sys.modules.pop(key)

        sys.modules.update(saved_src)

        if q2q_root in sys.path:
            sys.path.remove(q2q_root)

    _q2q_loaded = True
    logger.info(f"Q2Q modules loaded from {q2q_root}")


# ============================================================
# Module-level caches (survive agent resets per persona)
# ============================================================

_cached_embedding_provider = None
_cached_embedding_key = None


def _get_or_create_embedding_provider(provider: str, model_name: str, device: str):
    global _cached_embedding_provider, _cached_embedding_key
    key = (provider, model_name, device)
    if _cached_embedding_key == key and _cached_embedding_provider is not None:
        return _cached_embedding_provider
    _cached_embedding_provider = _q2q_embedding_factory.create_embedding_provider(
        provider=provider, model_name=model_name, device=device,
    )
    _cached_embedding_key = key
    logger.info("Embedding provider created: %s / %s / %s", provider, model_name, device)
    return _cached_embedding_provider


_cached_compressor = None
_cached_compressor_key = None


def _get_or_create_compressor(model_path: str, device: str, threshold: float):
    global _cached_compressor, _cached_compressor_key
    key = (model_path, device, threshold)
    if _cached_compressor_key == key and _cached_compressor is not None:
        return _cached_compressor
    comp_mod = _q2q_kept_refs.get("src.preprocessing.compressor")
    if comp_mod is None:
        return None
    _cached_compressor = comp_mod.PerplexityCompressor(
        model_path=model_path, device=device, threshold=threshold,
    )
    _cached_compressor_key = key
    logger.info(
        "Compressor created: %s / %s / thr=%.1f", model_path, device, threshold,
    )
    return _cached_compressor


# ============================================================
# Answer instruction constants
# ============================================================

Q2Q_ANSWER_INSTRUCTION_ZH = """【回复风格要求】
- 给出完整的回答，包含记忆上下文中的相关细节和背景信息
- 用完整句子回答，不要只给出片段或关键词
- 重点体现从记忆中检索到的具体信息（如特定名称、日期、数值、地点等细节）
- 如果记忆中有相关信息但不完全匹配问题，仍应基于可用信息给出最佳回答
- 仅当检索到的记忆中完全没有任何相关信息时，才回答"未提及"
"""

Q2Q_ANSWER_INSTRUCTION_EN = """[RESPONSE STYLE]
- Give COMPLETE answers using full sentences that include relevant context and details from the retrieved memory
- Do NOT give minimal or fragment answers — include the surrounding context, reasons, and specifics
- For example, if asked "What does X like?", answer "X likes Y because Z" rather than just "Y"
- If asked "When did X happen?", answer "X happened on [date] when [context]" rather than just the date
- Include specific details from the retrieved memory (names, dates, values, locations, reasons)
- If the retrieved memory contains relevant information that partially matches the question, still provide the best answer based on available evidence
- Only respond with "Not mentioned" when the retrieved memories contain absolutely NO relevant information whatsoever"""


# ============================================================
# Q2QBenchAgent
# ============================================================

class Q2QBenchAgent(BaseAgent):

    METHOD_TYPE = "agentic_memory"
    DEFAULT_MAX_CONTEXT_TOKENS = 120000

    def _get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _run_async(self, coro):
        loop = self._get_event_loop()
        return loop.run_until_complete(coro)

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        q2q_project_path: str = "",
        # Retrieval
        top_k_per_sub: int = 20,
        top_n: int = 5,
        top_k_q2c: int = 3,
        num_fake_queries: int = 10,
        chain_depth: int = 3,
        prop_top_k_per_fq: int = 3,
        binding_verify_threshold: float = 0.8,
        max_context_tokens: Optional[int] = None,
        # Preprocessing
        proposition_mode: str = "llm",
        compressor_model: str = "",
        compressor_device: str = "cpu",
        perplexity_threshold: float = 5.0,
        nlp_inference_model: str = "",
        # Graph
        evolve_threshold: float = 0.6,
        diverge_low: float = 0.2,
        diverge_high: float = 0.6,
        max_parents: int = 3,
        max_chain_length: int = 6,
        # Embedding
        embedding_model: str = "",
        embedding_provider: str = "local",
        embedding_model_path: Optional[str] = None,
        embedding_device: str = "cpu",
        # Storage
        storage_backend: str = "chromadb",
        language: str = "zh",
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        _load_q2q_modules(q2q_project_path)

        _q2q_llm_module._usage_tracker = get_usage_tracker()

        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url
        # Retrieval
        self._top_k_per_sub = top_k_per_sub
        self._top_n = top_n
        self._top_k_q2c = top_k_q2c
        self._num_fake_queries = num_fake_queries
        self._chain_depth = chain_depth
        self._prop_top_k_per_fq = prop_top_k_per_fq
        self._binding_verify_threshold = binding_verify_threshold
        self._max_context_tokens = max_context_tokens or self.DEFAULT_MAX_CONTEXT_TOKENS
        # Preprocessing
        self._proposition_mode = proposition_mode
        self._compressor_model = compressor_model
        self._compressor_device = compressor_device
        self._perplexity_threshold = perplexity_threshold
        self._nlp_inference_model = nlp_inference_model
        # Graph
        self._evolve_threshold = evolve_threshold
        self._diverge_low = diverge_low
        self._diverge_high = diverge_high
        self._max_parents = max_parents
        self._max_chain_length = max_chain_length
        self._max_chain_nodes = int(kwargs.get("max_chain_nodes", 60))
        # Embedding
        self._embedding_model = embedding_model_path or embedding_model
        self._embedding_provider = embedding_provider
        self._embedding_device = embedding_device
        # Storage
        self._storage_backend = storage_backend
        self._language = language
        # DAG/Q2C context participation (default ON; ablation configs may disable)
        self._include_chain_texts = bool(kwargs.get("include_chain_texts", True))
        self._include_q2c_props = bool(kwargs.get("include_q2c_props", True))
        # Dynamic context tier caps (participation degree, tunable)
        self._max_chain_sessions = int(kwargs.get("max_chain_sessions", 2))
        self._max_chain_props = int(kwargs.get("max_chain_props", 30))
        self._max_q2c_props = int(kwargs.get("max_q2c_props", 30))
        self._max_q2c_sessions = int(kwargs.get("max_q2c_sessions", 3))
        self._max_fqs_per_session = int(kwargs.get("max_fqs_per_session", 3))

        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._temp_dir: Optional[str] = None
        self._q2q_agent = None
        self._create_q2q_agent()

    # ------------------------------------------------------------------
    # Q2Q agent lifecycle
    # ------------------------------------------------------------------

    def _build_q2q_config(self, storage_path: str):
        config = _Q2QConfig()

        config.llm.provider = self._provider
        config.llm.model = self.model
        config.llm.temperature = self.temperature
        config.llm.max_tokens = self.max_tokens
        config.llm.api_key = self._api_key or ""
        config.llm.base_url = self._base_url or ""

        config.embedding.provider = self._embedding_provider
        config.embedding.model_name = self._embedding_model
        config.embedding.device = self._embedding_device

        config.preprocessing.proposition_mode = self._proposition_mode
        config.preprocessing.compressor_model = self._compressor_model
        config.preprocessing.compressor_device = self._compressor_device
        config.preprocessing.perplexity_threshold = self._perplexity_threshold
        config.preprocessing.nlp_inference_model = self._nlp_inference_model

        config.graph.evolve_threshold = self._evolve_threshold
        config.graph.diverge_low = self._diverge_low
        config.graph.diverge_high = self._diverge_high
        config.graph.max_parents = self._max_parents
        config.graph.max_chain_length = self._max_chain_length
        config.graph.max_chain_nodes = self._max_chain_nodes

        config.retrieval.top_k_per_sub = self._top_k_per_sub
        config.retrieval.top_n = self._top_n
        config.retrieval.top_k_q2c = self._top_k_q2c
        config.retrieval.num_fake_queries = self._num_fake_queries
        config.retrieval.chain_depth = self._chain_depth
        config.retrieval.prop_top_k_per_fq = self._prop_top_k_per_fq
        config.retrieval.max_fqs_per_session = self._max_fqs_per_session
        config.retrieval.binding_verify_threshold = self._binding_verify_threshold
        config.retrieval.max_context_tokens = self._max_context_tokens

        config.storage.backend = self._storage_backend
        config.storage.chromadb_path = os.path.join(storage_path, "chromadb")
        config.storage.chromadb_collection = f"q2q_{os.getpid()}"

        config.language = self._language
        config.log.level = "WARNING"

        return config

    def _create_q2q_agent(self):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        self._temp_dir = tempfile.mkdtemp(prefix="q2q_bench_")
        _q2q_llm_module._usage_tracker = get_usage_tracker()

        config = self._build_q2q_config(self._temp_dir)

        cached_emb = _get_or_create_embedding_provider(
            provider=self._embedding_provider,
            model_name=self._embedding_model,
            device=self._embedding_device,
        )

        agent_module = _q2q_kept_refs.get("agent")
        original_fn = getattr(agent_module, "create_embedding_provider", None) if agent_module else None
        if agent_module:
            agent_module.create_embedding_provider = lambda *a, **kw: cached_emb

        try:
            self._q2q_agent = _Q2QAgent(config)
        finally:
            if agent_module and original_fn is not None:
                agent_module.create_embedding_provider = original_fn

        # Inject cached compressor to avoid reloading model on reset
        if self._compressor_model:
            cached_comp = _get_or_create_compressor(
                self._compressor_model, self._compressor_device,
                self._perplexity_threshold,
            )
            if cached_comp is not None:
                self._q2q_agent.compressor = cached_comp
                self._q2q_agent.memory_constructor.compressor = cached_comp

        logger.info(
            "Q2Q agent created: model=%s, fq=%d, prop_mode=%s",
            self.model, self._num_fake_queries, self._proposition_mode,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface: memorize
    # ------------------------------------------------------------------

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        try:
            entry = self._run_async(self._q2q_agent.memorize(text))
        except Exception as e:
            if _q2q_LLMRetryExhaustedError and isinstance(e, _q2q_LLMRetryExhaustedError):
                raise LLMRetryExhaustedError(str(e), e, getattr(e, 'attempts', 0)) from e
            raise

        self._memory_chunks.append(text)
        self._is_initialized = True

        # Build memory_entries: one entry per FQ with answer_source binding
        memory_entries = []
        for fq in entry.fake_queries:
            memory_entries.append({
                "event": "Q2Q_MEMORIZE",
                "session_id": entry.session_id,
                "fake_query_id": fq.query_id,
                "fake_query_text": fq.text,
                "answer_source": fq.answer_source,
            })

        # Build extraction_result: formatted propositions
        extraction_lines = []
        for prop in entry.propositions:
            line = f"[{prop.prop_id}] {prop.text}"
            if prop.source:
                line += f" (source: {prop.source})"
            if prop.time:
                line += f" (time: {prop.time})"
            if prop.entities:
                line += f" (entities: {', '.join(prop.entities)})"
            extraction_lines.append(line)
        extraction_result = "\n".join(extraction_lines)

        # Build all_passages: per-FQ passage info with bound propositions
        prop_map = {p.prop_id: p for p in entry.propositions}
        all_passages = []
        for fq in entry.fake_queries:
            bound_props = []
            for pid in fq.answer_source:
                if pid in prop_map:
                    bound_props.append({
                        "prop_id": pid,
                        "text": prop_map[pid].text,
                    })
            all_passages.append({
                "fake_query_id": fq.query_id,
                "fake_query_text": fq.text,
                "session_id": entry.session_id,
                "bound_propositions": bound_props,
            })

        # Collect graph stats for this session's FQs
        graph_stats = self._run_async(self._get_graph_stats(entry))

        return MemoryBuildResult(
            success=True,
            method="q2q",
            action="q2q_dual_memorize",
            input_content=text,
            stored_content=entry.pre_session_text or text,
            memory_entries=memory_entries,
            chunk_count=self._run_async(self._q2q_agent.memory_store.count()),
            extraction_result=extraction_result,
            all_passages=all_passages,
            extra={
                "session_id": entry.session_id,
                "num_propositions": len(entry.propositions),
                "num_fake_queries": len(entry.fake_queries),
                "num_content_chunks": len(entry.content_embeddings),
                "compression_ratio": (
                    len(entry.pre_session_text) / max(len(entry.session_text), 1)
                    if entry.pre_session_text else 1.0
                ),
                "compression_stats": {
                    "input_chars": len(entry.session_text),
                    "compressed_chars": len(entry.pre_session_text) if entry.pre_session_text else 0,
                    "ratio": round(
                        len(entry.pre_session_text) / max(len(entry.session_text), 1), 4
                    ) if entry.pre_session_text else 1.0,
                    "input_tokens_est": len(entry.session_text) // 4,
                    "compressed_tokens_est": (
                        len(entry.pre_session_text) // 4 if entry.pre_session_text else 0
                    ),
                    "compressed_preview": (
                        entry.pre_session_text[:500] if entry.pre_session_text else ""
                    ),
                },
                "graph_stats": graph_stats,
            },
        )

    # ------------------------------------------------------------------
    # Graph stats helper
    # ------------------------------------------------------------------

    async def _get_graph_stats(self, entry) -> dict:
        store = self._q2q_agent.memory_store

        fqs_with_parents = 0
        fqs_with_children = 0
        parent_details = []

        for fq in entry.fake_queries:
            node = await store.get_fake_query_by_id(fq.query_id)
            if not node:
                continue
            parents = node.get("parent_ids", [])
            children = node.get("child_ids", [])
            if parents:
                fqs_with_parents += 1
                parent_details.append({
                    "fq_id": fq.query_id,
                    "fq_text": fq.text[:80],
                    "parent_ids": parents,
                })
            if children:
                fqs_with_children += 1

        all_edges = await store.get_all_edges()
        session_fq_ids = {fq.query_id for fq in entry.fake_queries}
        relevant_edges = [
            e for e in all_edges
            if e.dst_id in session_fq_ids or e.src_id in session_fq_ids
        ]

        evolves_count = sum(1 for e in relevant_edges if e.edge_type == "EVOLVES_TO")
        diverges_count = len(relevant_edges) - evolves_count

        return {
            "total_fqs": len(entry.fake_queries),
            "fqs_with_parents": fqs_with_parents,
            "fqs_with_children": fqs_with_children,
            "relevant_edges": len(relevant_edges),
            "evolves_to_count": evolves_count,
            "diverges_to_count": diverges_count,
            "total_graph_edges": len(all_edges),
            "parent_details": parent_details[:10],
        }

    # ------------------------------------------------------------------
    # BaseAgent interface: query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        agent = self._q2q_agent
        raw_question = self._extract_raw_question(question)

        try:
            sub_queries = self._run_async(agent.query_decomposer.decompose(raw_question))
            retrieval_result = self._run_async(agent.dual_retriever.retrieve(sub_queries))
        except Exception as e:
            if _q2q_LLMRetryExhaustedError and isinstance(e, _q2q_LLMRetryExhaustedError):
                raise LLMRetryExhaustedError(str(e), e, getattr(e, 'attempts', 0)) from e
            raise

        # Budget-aware tiered filling: no fixed percentage split. Budget is
        # claimed in order of information density per token, then whatever
        # remains expands into full-fidelity text, so a small retrieval never
        # wastes the window and a large one degrades gracefully.
        #
        # Tier 1: DAG chain + Q2C propositions (distilled anchors)
        # Tier 2: direct Q2Q session texts     (primary evidence, full fidelity)
        # Tier 3: DAG chain session texts      (expansion if budget remains)
        # Tier 4: Q2C session texts            (expansion if budget remains)
        budget = self._context_budget(question, final_system_preview=None)
        used = 0
        session_context_str = ""
        propositions_context_str = ""

        def _fits(text: str) -> bool:
            nonlocal used
            need = self.count_tokens(text) + 4
            if used + need > budget:
                return False
            used += need
            return True

        # Tier 1: propositions claim budget before raw text. They carry the
        # distilled, query-aligned anchors (step indices, element IDs, dates)
        # that precise-recall questions hinge on, at a fraction of the tokens
        # per fact. Letting session texts fill first starved this tier to zero
        # bytes on every long-session query.
        prop_lines: list[str] = []
        seen_prop_lines: set[str] = set()

        def _add_props(props, limit):
            for p in props[:limit]:
                line = f"- {p.text}" + (f" (time: {p.time})" if p.time else "")
                if line in seen_prop_lines:
                    continue
                if not _fits(line):
                    return
                seen_prop_lines.add(line)
                prop_lines.append(line)

        if self._include_chain_texts:
            _add_props(retrieval_result.chain_propositions, self._max_chain_props)
        if self._include_q2c_props:
            _add_props(retrieval_result.q2c_propositions, self._max_q2c_props)

        if prop_lines:
            propositions_context_str = (
                "--- Supplementary Facts (from memory graph, may be relevant) ---\n"
                + "\n".join(prop_lines)
            )

        # Tier 2: direct Q2Q session texts
        session_parts = []
        for i, st in enumerate(retrieval_result.direct_session_texts, 1):
            block = f"--- Conversation Segment {i} ---\n{st}"
            if not _fits(block):
                break
            session_parts.append(block)
        if session_parts:
            session_context_str = "\n\n".join(session_parts)

        # Full-fidelity evidence leads the prompt, distilled facts follow it as
        # anchors: budget priority and presentation order are independent.
        context_parts: list[str] = []
        if session_context_str:
            context_parts.append(session_context_str)
        if propositions_context_str:
            context_parts.append(propositions_context_str)

        # Tier 3: DAG chain session texts (dedup vs direct already done upstream)
        if self._include_chain_texts:
            for st in retrieval_result.chain_session_texts[:self._max_chain_sessions]:
                block = f"--- Related Context (DAG Chain) ---\n{st}"
                if not _fits(block):
                    break
                context_parts.append(block)

        # Tier 4: Q2C session texts
        if self._include_q2c_props:
            for st in retrieval_result.q2c_session_texts[:self._max_q2c_sessions]:
                block = f"--- Related Context (Content Match) ---\n{st}"
                if not _fits(block):
                    break
                context_parts.append(block)

        memories_context = "\n\n".join(context_parts) if context_parts else ""

        # Assemble final prompt with context
        if memories_context:
            full_message = f"{memories_context}\n\n{question}"
        else:
            full_message = question

        answer_instruction = (
            Q2Q_ANSWER_INSTRUCTION_ZH if self._language == "zh" else Q2Q_ANSWER_INSTRUCTION_EN
        )
        if system_message:
            final_system = f"{system_message}\n\n{answer_instruction}"
        else:
            final_system = answer_instruction

        # Truncate to fit context window
        truncated_docs = self.truncate_docs_to_context(
            [memories_context] if memories_context else [],
            question,
            system_message=final_system,
            max_context_tokens=self._max_context_tokens,
        )
        if truncated_docs:
            full_message = f"{truncated_docs[0]}\n\n{question}"

        messages = format_messages(full_message, final_system)
        response = self._llm_client.chat(messages)

        # Build retrieved_memories for logging: matched FQs
        retrieved_memories = []
        for fq in retrieval_result.matched_fqs:
            retrieved_memories.append({
                "type": "q2q_matched_fq",
                "fake_query_id": fq.query_id,
                "fake_query_text": fq.text,
                "answer_source": fq.answer_source,
                "session_id": fq.session_id,
                "score": round(retrieval_result.q2q_scores.get(fq.query_id, 0.0), 4),
            })

        # Top propositions for logging
        prop_details = []
        for prop in retrieval_result.ranked_propositions[:10]:
            prop_details.append({
                "prop_id": prop.prop_id,
                "text": prop.text,
                "session_id": prop.session_id,
                "source": prop.source,
            })

        return AgentResponse(
            output=response.content,
            retrieved_count=len(retrieval_result.matched_fqs),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "q2q",
                "sub_queries": [
                    {"query": sq.text, "keywords": sq.keywords}
                    for sq in sub_queries
                ],
                "num_chain_fqs": len(retrieval_result.chain_fqs),
                "num_propositions": len(retrieval_result.ranked_propositions),
                "num_sessions": len(retrieval_result.direct_session_texts),
                "num_chain_sessions": len(retrieval_result.chain_session_texts),
                "num_q2c_sessions": len(retrieval_result.q2c_session_texts),
                "top_propositions": prop_details,
                "session_text_included": bool(retrieval_result.direct_session_texts),
                "session_text_total_chars": sum(
                    len(s) for s in retrieval_result.direct_session_texts
                ),
                "context_budget_tokens": budget,
                "context_used_tokens": used,
                "context_composition": {
                    "propositions_chars": len(propositions_context_str),
                    "session_texts_chars": len(session_context_str),
                    "total_context_chars": len(memories_context),
                },
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Reserve for system message, answer instruction and prompt scaffolding.
    _CONTEXT_RESERVE_TOKENS = 2000

    def _context_budget(self, question: str, final_system_preview: Optional[str] = None) -> int:
        """Tokens available for retrieved context after question/system overhead."""
        overhead = self.count_tokens(question) + self._CONTEXT_RESERVE_TOKENS
        if final_system_preview:
            overhead += self.count_tokens(final_system_preview)
        return max(self._max_context_tokens - overhead, 0)

    @staticmethod
    def _extract_raw_question(formatted_text: str) -> str:
        match = re.search(r'问题[：:]\s*(.+?)(?:\n\n【|$)', formatted_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r'Question[：:]\s*(.+?)(?:\n\n\[ANSWER|$)', formatted_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = formatted_text.split('\n\n')
        if len(lines) >= 2:
            middle_parts = []
            for part in lines[1:]:
                if part.startswith('【') or part.startswith('[ANSWER') or part.strip() in ('答案：', 'Answer:'):
                    break
                middle_parts.append(part)
            if middle_parts:
                return '\n\n'.join(middle_parts).strip()

        return formatted_text

    def reset(self) -> None:
        super().reset()
        self._create_q2q_agent()
        logger.info("Q2Q agent reset with fresh storage")

    def set_context_id(self, context_id: int) -> None:
        super().set_context_id(context_id)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "top_k_per_sub": self._top_k_per_sub,
            "top_n": self._top_n,
            "top_k_q2c": self._top_k_q2c,
            "num_fake_queries": self._num_fake_queries,
            "chain_depth": self._chain_depth,
            "prop_top_k_per_fq": self._prop_top_k_per_fq,
            "binding_verify_threshold": self._binding_verify_threshold,
            "proposition_mode": self._proposition_mode,
            "embedding_model": self._embedding_model,
            "embedding_provider": self._embedding_provider,
            "storage_backend": self._storage_backend,
            "language": self._language,
            "max_context_tokens": self._max_context_tokens,
        })
        if self._q2q_agent:
            info["memory_count"] = self._run_async(self._q2q_agent.memory_store.count())
        return info

    def __del__(self):
        if hasattr(self, "_temp_dir") and self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
