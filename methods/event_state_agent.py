"""BaseAgent adapter for the independent Event-State Hybrid memory method."""

from __future__ import annotations

import json
import hashlib
import logging
import re
import contextvars
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from methods.base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import BaseLLMClient, LLMAPIError, create_llm_client, format_messages, get_usage_tracker

from .event_state.compiler import StateCompiler, parse_json
from .event_state.context import evidence_identity, episode_turn_embedding_text, fit_context, render_claim, render_episode, render_episode_evidence, render_selected_claim_evidence, select_claim_evidence, selected_claim_evidence_turn_keys, select_global_episode_evidence
from .event_state.embeddings import DenseEmbedder
from .event_state.prompts import (
    ANSWER_DATA_BOUNDARY_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    QUERY_COMPILER_SYSTEM_PROMPT,
    QUERY_PLANNER_SYSTEM_PROMPT,
    STRUCTURED_OUTPUT_REPAIR_SYSTEM_PROMPT,
)
from .event_state.planner import QueryPlan, fallback_query_plan, query_compiler_json_schema, salvage_query_compiler_output, validate_planner_output
from .event_state.retrieval import EventStateRetriever
from .event_state.schemas import Claim, Episode, EvidenceRef, NormalizedTurn, TurnEvidence
from .event_state.store import EventStateStore
from .event_state.subjects import display_subject, is_canonical_subject_id, is_valid_canonical_subject_proposal, is_visible_subject_identity, normalize_name, normalize_scope, resolve_subject_id
from .event_state.validation import canonical_turn_id, claim_repair_identity, claim_semantic_fingerprint, is_meta_claim, is_no_information_value, normalize_ingress_temporal_bounds, normalize_state_slot, resolve_model_turn_reference_with_form, validated_claim

logger = logging.getLogger(__name__)

SELF_REFERENCES = {"i", "me", "my", "myself", "self"}


def _balanced_json_end(text: str, start: int) -> Optional[int]:
    """Return the exclusive end of one complete JSON object or array."""
    if start >= len(text) or text[start] not in "[{":
        return None
    stack = [text[start]]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            expected = "[" if character == "]" else "{"
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def _json_string_value(text: str, start: int) -> tuple[Optional[Any], Optional[int]]:
    try:
        value, consumed = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None, None
    return value, start + consumed


def salvage_extraction_fragments(content: str) -> Dict[str, Any]:
    """Recover literal complete extraction fragments without repairing content."""
    summary = None
    claims: List[Dict[str, Any]] = []
    claims_array_start = None
    index = 0
    while index < len(content):
        if content[index] != '"':
            index += 1
            continue
        key, end = _json_string_value(content, index)
        if not isinstance(key, str) or end is None:
            index += 1
            continue
        cursor = end
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor >= len(content) or content[cursor] != ":":
            index = end
            continue
        cursor += 1
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if key == "episode_summary":
            value, value_end = _json_string_value(content, cursor)
            if isinstance(value, str):
                summary = value
            index = value_end or end
            continue
        if key == "claims" and cursor < len(content) and content[cursor] == "[":
            claims_array_start = cursor + 1
            break
        index = end

    if claims_array_start is not None:
        cursor = claims_array_start
        while cursor < len(content):
            while cursor < len(content) and (content[cursor].isspace() or content[cursor] == ","):
                cursor += 1
            if cursor >= len(content) or content[cursor] == "]":
                break
            end = _balanced_json_end(content, cursor)
            if end is None:
                break
            try:
                value = json.loads(content[cursor:end])
            except json.JSONDecodeError:
                cursor = end
                continue
            if isinstance(value, dict):
                claims.append(value)
            cursor = end
    return {"episode_summary": summary, "claims": claims}


@dataclass
class SubjectResolution:
    subject_id: str
    subject_display: str
    source_speaker: Optional[str]
    resolution_source: str
    conflict_detected: bool = False


@dataclass
class PreparedMemorySession:
    """State-independent Event-State work ready for ordered compilation."""

    context_id: Any
    session: Dict[str, Any]
    normalized_turns: List[NormalizedTurn]
    extracted: Dict[str, Any]
    episode: Episode
    episode_embedding: List[float]
    turn_embeddings: List[List[float]]
    claims: List[Claim]
    claim_embeddings: List[List[float]]
    claim_slot_embeddings: List[List[float]]


class EventStateAgent(BaseAgent):
    """Lossless episodes plus conservative, versioned semantic claims."""

    METHOD_TYPE = "agentic_memory"

    def __init__(self, model="gpt-4o-mini", temperature=1.0, max_tokens=2000, provider="openai", api_key=None, base_url=None, llm_client_kwargs=None, memory_model=None, memory_provider=None, memory_temperature=0.0, memory_max_tokens=1800, memory_api_key=None, memory_base_url=None, memory_llm_client_kwargs=None, llm_client=None, memory_llm_client=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_provider="local", embedding_model_path=None, embedding_api_key=None, embedding_base_url=None, embedding_client=None, enable_episodes=True, enable_state_claims=True, enable_state_compilation=True, extraction_max_tokens=1800, extraction_temperature=0.0, max_claims_per_episode=20, state_candidate_top_k=5, state_current_candidate_top_k=3, state_candidate_min_similarity=0.45, update_min_confidence=0.55, update_temperature=0.0, update_max_tokens=800, store_raw_episode_text=True, enable_bitemporal_time=True, event_time_normalization_enabled=True, preserve_turn_evidence=True, max_context_tokens=120000, retrieve_claims=True, retrieve_episodes=True, retrieve_turns=True, claim_top_k=30, episode_top_k=20, turn_top_k=8, candidate_count=40, fusion_mode="rrf", rrf_k=60.0, claim_retrieval_weight=1.0, episode_retrieval_weight=1.0, turn_retrieval_weight=1.0, turn_lexical_retrieval_enabled=False, turn_lexical_retrieval_weight=1.0, temporal_retrieval_enabled=True, temporal_retrieval_weight=1.0, ppr_enabled=False, ppr_alpha=0.85, ppr_max_iterations=20, ppr_tolerance=1e-6, ppr_expand_hops=2, ppr_mix_weight=0.35, ppr_weight_supersedes=1.2, ppr_weight_refines=1.0, ppr_weight_conflict=0.8, ppr_weight_evidence=0.7, selector_mode="state_mmr", evidence_count=8, turn_evidence_count=0, mmr_lambda=0.7, state_relation_bonus=0.05, source_diversity_bonus=0.02, representation_balance_bonus=0.02, inject_source_evidence=True, max_source_excerpts_per_claim=2, max_episode_source_excerpts_total=2, event_state_workers=1, planner_rounds=0, planner_max_requests=3, planner_mode=None, query_compiler_max_searches=3, query_compiler_max_tokens=256, planner_temperature=0.0, planner_max_tokens=1200, planner_merge_mode="coverage_interleave", query_manager_reuse_mode="auto", query_memory_guard_enabled=True, query_memory_reserve_mb=1024, query_memory_max_fraction=.5, query_memory_min_available_mb=512, query_worker_memory_estimate_mb=None, score_calibration_mode="minmax", turn_lexical_mode="overlap", turn_sparse_unit="turn", bm25_k1=1.2, bm25_b=.75, turn_chunk_min_chars=24, turn_chunk_max_per_turn=8, episode_relevance_mode="summary", episode_summary_weight=.5, episode_best_turn_weight=.5, episode_excerpt_mode="global", local_reranker_mode="off", local_reranker_model=None, local_reranker_top_k=20, local_reranker_batch_size=8, local_reranker_weight=.35, local_reranker_max_length=512, local_reranker_min_available_mb=1024, local_reranker_auto_disable_on_low_memory=True, temporal_query_mode="legacy", evidence_budget_mode="fixed", adaptive_evidence_min=None, adaptive_evidence_max=None, source_coherence_max_sources=None, source_coherence_score_ratio=.7, source_coherence_score_mass=.85, source_coherence_additional_candidates=2, source_coherence_additional_weight=.35, source_coherence_bonus=.08, source_coherence_penalty=.04, **kwargs):
        super().__init__(model, temperature, max_tokens, **kwargs)
        # YAML 1.1 loaders parse bare ``off`` as False. Older snapshot configs
        # can also contain null for options that did not exist at build time.
        local_reranker_mode = "off" if local_reranker_mode in {None, False, ""} else local_reranker_mode
        temporal_query_mode = "legacy" if temporal_query_mode in {None, ""} else temporal_query_mode
        if fusion_mode not in {"rrf", "weighted_score"}:
            raise ValueError("fusion_mode must be rrf or weighted_score")
        if selector_mode not in {"topk", "mmr", "state_mmr", "source_coherent_mmr"}:
            raise ValueError("selector_mode must be topk, mmr, state_mmr, or source_coherent_mmr")
        if query_manager_reuse_mode not in {"legacy", "shared", "worker_local", "auto"}:
            raise ValueError("query_manager_reuse_mode must be legacy, shared, worker_local, or auto")
        if score_calibration_mode not in {"none", "minmax", "sigmoid_zscore"}:
            raise ValueError("score_calibration_mode must be none, minmax, or sigmoid_zscore")
        if turn_lexical_mode not in {"overlap", "bm25"} or turn_sparse_unit not in {"turn", "sentence_chunk"}:
            raise ValueError("turn_lexical_mode must be overlap or bm25 and turn_sparse_unit must be turn or sentence_chunk")
        if episode_relevance_mode not in {"summary", "summary_plus_best_turn"} or episode_excerpt_mode not in {"global", "joint"}:
            raise ValueError("unsupported episode relevance or excerpt mode")
        if local_reranker_mode not in {"off", "cross_encoder"} or temporal_query_mode not in {"legacy", "dual_axis_semantic"}:
            raise ValueError("unsupported local reranker or temporal query mode")
        if evidence_budget_mode not in {"fixed", "adaptive"}:
            raise ValueError("evidence_budget_mode must be fixed or adaptive")
        if float(query_memory_reserve_mb) < 0 or float(query_memory_min_available_mb) < 0 or not 0 < float(query_memory_max_fraction) <= 1:
            raise ValueError("query memory reserve/minimum must be non-negative and max fraction must be in (0, 1]")
        if float(bm25_k1) <= 0 or not 0 <= float(bm25_b) <= 1:
            raise ValueError("bm25_k1 must be positive and bm25_b must be in [0, 1]")
        if not 0 <= float(local_reranker_weight) <= 1:
            raise ValueError("local_reranker_weight must be in [0, 1]")
        if float(episode_summary_weight) < 0 or float(episode_best_turn_weight) < 0:
            raise ValueError("episode relevance weights must be non-negative")
        if not 0 <= float(source_coherence_score_ratio) <= 1 or not 0 <= float(source_coherence_score_mass) <= 1:
            raise ValueError("source coherence score ratio and mass must be in [0, 1]")
        if int(turn_chunk_min_chars) < 0 or int(turn_chunk_max_per_turn) < 1:
            raise ValueError("turn chunk minimum chars must be non-negative and max per turn must be positive")
        if int(local_reranker_top_k) < 1 or int(local_reranker_batch_size) < 1 or int(local_reranker_max_length) < 1:
            raise ValueError("local reranker top-k, batch size, and max length must be positive")
        if adaptive_evidence_min is not None and adaptive_evidence_max is not None and int(adaptive_evidence_min) > int(adaptive_evidence_max):
            raise ValueError("adaptive_evidence_min must be <= adaptive_evidence_max")
        if enable_state_claims and not enable_episodes:
            raise ValueError("enable_state_claims requires enable_episodes for provenance")
        if not 1 <= int(evidence_count) <= int(candidate_count):
            raise ValueError("evidence_count must be >= 1 and <= candidate_count")
        try:
            normalized_turn_evidence_count = int(turn_evidence_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("turn_evidence_count must be an integer >= 0") from exc
        if (
            isinstance(turn_evidence_count, bool)
            or isinstance(turn_evidence_count, float)
            or normalized_turn_evidence_count < 0
        ):
            raise ValueError("turn_evidence_count must be an integer >= 0")
        if not 0 <= float(mmr_lambda) <= 1 or not 0 <= float(ppr_alpha) <= 1 or not 0 <= float(ppr_mix_weight) <= 1 or not 0 <= float(update_min_confidence) <= 1:
            raise ValueError("mmr_lambda, ppr_alpha, ppr_mix_weight, and update_min_confidence must be between 0 and 1")
        if any(int(value) < 0 for value in (claim_top_k, episode_top_k, turn_top_k, candidate_count)):
            raise ValueError("retrieval top-k values must be non-negative")
        if int(planner_rounds) < 0 or (int(planner_rounds) > 0 and int(planner_max_requests) < 1) or int(planner_max_tokens) <= 0:
            raise ValueError("planner_rounds must be >= 0, planner_max_requests >= 1 when enabled, and planner_max_tokens > 0")
        planner_mode = planner_mode or ("iterative" if int(planner_rounds) else "off")
        if planner_mode not in {"off", "iterative", "query_compiler"}:
            raise ValueError("planner_mode must be off, iterative, or query_compiler")
        if planner_mode == "query_compiler" and int(planner_rounds):
            raise ValueError("query_compiler mode requires planner_rounds=0")
        if planner_mode == "iterative" and not int(planner_rounds):
            raise ValueError("iterative planner_mode requires planner_rounds > 0")
        if not 0 <= int(query_compiler_max_searches) <= 8:
            raise ValueError("query_compiler_max_searches must be between 0 and 8")
        if int(query_compiler_max_tokens) <= 0:
            raise ValueError("query_compiler_max_tokens must be positive")
        self.max_context_tokens = int(max_context_tokens)
        self.max_claims_per_episode = max(0, int(max_claims_per_episode))
        self.extraction_max_tokens, self.extraction_temperature = int(extraction_max_tokens), float(extraction_temperature)
        self.update_temperature, self.update_max_tokens = float(update_temperature), int(update_max_tokens)
        self.enable_episodes, self.enable_state_claims, self.enable_state_compilation = bool(enable_episodes), bool(enable_state_claims), bool(enable_state_compilation)
        self.store_raw_episode_text, self.enable_bitemporal_time, self.preserve_turn_evidence = bool(store_raw_episode_text), bool(enable_bitemporal_time), bool(preserve_turn_evidence)
        self.event_time_normalization_enabled = bool(event_time_normalization_enabled)
        self.inject_source_evidence, self.max_source_excerpts_per_claim = bool(inject_source_evidence), max(0, int(max_source_excerpts_per_claim))
        self.max_episode_source_excerpts_total = max(0, int(max_episode_source_excerpts_total))
        self.event_state_workers = max(1, int(event_state_workers))
        self.planner_rounds = int(planner_rounds)
        self.planner_mode = planner_mode
        self.query_compiler_max_searches = int(query_compiler_max_searches)
        self.query_compiler_max_tokens = int(query_compiler_max_tokens)
        self.planner_max_requests = int(planner_max_requests)
        self.planner_temperature = float(planner_temperature)
        self.planner_max_tokens = int(planner_max_tokens)
        if planner_merge_mode not in {"coverage_interleave", "sum_rrf"}:
            raise ValueError("planner_merge_mode must be coverage_interleave or sum_rrf")
        self.planner_merge_mode = planner_merge_mode
        self._llm_client: BaseLLMClient = llm_client or create_llm_client(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, base_url=base_url, **(llm_client_kwargs or {}))
        self._memory_llm_client: BaseLLMClient = memory_llm_client or create_llm_client(provider=memory_provider or provider, model=memory_model or model, temperature=memory_temperature if memory_model else temperature, max_tokens=memory_max_tokens if memory_model else max_tokens, api_key=memory_api_key if memory_model else api_key, base_url=memory_base_url if memory_model else base_url, **(memory_llm_client_kwargs or {}))
        self._embedder = embedding_client or DenseEmbedder(embedding_provider, embedding_model, embedding_model_path, embedding_api_key or api_key, embedding_base_url or base_url)
        self._stores: Dict[Any, EventStateStore] = {}
        self._context_id = None
        self._build_config = {"enable_episodes": self.enable_episodes, "enable_state_claims": self.enable_state_claims, "enable_state_compilation": self.enable_state_compilation, "extraction_max_tokens": self.extraction_max_tokens, "extraction_temperature": self.extraction_temperature, "max_claims_per_episode": self.max_claims_per_episode, "state_candidate_top_k": int(state_candidate_top_k), "state_current_candidate_top_k": max(1, int(state_current_candidate_top_k)), "state_candidate_min_similarity": float(state_candidate_min_similarity), "update_min_confidence": float(update_min_confidence), "update_temperature": self.update_temperature, "update_max_tokens": self.update_max_tokens, "store_raw_episode_text": self.store_raw_episode_text, "enable_bitemporal_time": self.enable_bitemporal_time, "event_time_normalization_enabled": self.event_time_normalization_enabled, "preserve_turn_evidence": self.preserve_turn_evidence}
        self._retrieval_config = {"retrieve_claims": bool(retrieve_claims), "retrieve_episodes": bool(retrieve_episodes), "retrieve_turns": bool(retrieve_turns), "claim_top_k": int(claim_top_k), "episode_top_k": int(episode_top_k), "turn_top_k": int(turn_top_k), "candidate_count": int(candidate_count), "fusion_mode": fusion_mode, "rrf_k": float(rrf_k), "claim_retrieval_weight": float(claim_retrieval_weight), "episode_retrieval_weight": float(episode_retrieval_weight), "turn_retrieval_weight": float(turn_retrieval_weight), "turn_lexical_retrieval_enabled": bool(turn_lexical_retrieval_enabled), "turn_lexical_retrieval_weight": float(turn_lexical_retrieval_weight), "temporal_retrieval_enabled": bool(temporal_retrieval_enabled), "temporal_retrieval_weight": float(temporal_retrieval_weight), "ppr_enabled": bool(ppr_enabled), "ppr_alpha": float(ppr_alpha), "ppr_max_iterations": int(ppr_max_iterations), "ppr_tolerance": float(ppr_tolerance), "ppr_expand_hops": int(ppr_expand_hops), "ppr_mix_weight": float(ppr_mix_weight), "ppr_weight_supersedes": float(ppr_weight_supersedes), "ppr_weight_refines": float(ppr_weight_refines), "ppr_weight_conflict": float(ppr_weight_conflict), "ppr_weight_evidence": float(ppr_weight_evidence), "selector_mode": selector_mode, "evidence_count": int(evidence_count), "turn_evidence_count": normalized_turn_evidence_count, "mmr_lambda": float(mmr_lambda), "state_relation_bonus": float(state_relation_bonus), "source_diversity_bonus": float(source_diversity_bonus), "representation_balance_bonus": float(representation_balance_bonus), "max_episode_source_excerpts_total": self.max_episode_source_excerpts_total, "planner_rounds": self.planner_rounds, "planner_mode": self.planner_mode, "query_compiler_max_searches": self.query_compiler_max_searches, "query_compiler_max_tokens": self.query_compiler_max_tokens, "planner_max_requests": self.planner_max_requests, "planner_temperature": self.planner_temperature, "planner_max_tokens": self.planner_max_tokens, "planner_merge_mode": self.planner_merge_mode,
            "query_manager_reuse_mode": query_manager_reuse_mode, "query_memory_guard_enabled": bool(query_memory_guard_enabled), "query_memory_reserve_mb": float(query_memory_reserve_mb), "query_memory_max_fraction": float(query_memory_max_fraction), "query_memory_min_available_mb": float(query_memory_min_available_mb), "query_worker_memory_estimate_mb": query_worker_memory_estimate_mb,
            "score_calibration_mode": score_calibration_mode, "turn_lexical_mode": turn_lexical_mode, "turn_sparse_unit": turn_sparse_unit, "bm25_k1": float(bm25_k1), "bm25_b": float(bm25_b), "turn_chunk_min_chars": int(turn_chunk_min_chars), "turn_chunk_max_per_turn": int(turn_chunk_max_per_turn),
            "episode_relevance_mode": episode_relevance_mode, "episode_summary_weight": float(episode_summary_weight), "episode_best_turn_weight": float(episode_best_turn_weight), "episode_excerpt_mode": episode_excerpt_mode,
            "local_reranker_mode": local_reranker_mode, "local_reranker_model": local_reranker_model, "local_reranker_top_k": int(local_reranker_top_k), "local_reranker_batch_size": int(local_reranker_batch_size), "local_reranker_weight": float(local_reranker_weight), "local_reranker_max_length": int(local_reranker_max_length), "local_reranker_min_available_mb": float(local_reranker_min_available_mb), "local_reranker_auto_disable_on_low_memory": bool(local_reranker_auto_disable_on_low_memory),
            "temporal_query_mode": temporal_query_mode, "evidence_budget_mode": evidence_budget_mode, "adaptive_evidence_min": adaptive_evidence_min, "adaptive_evidence_max": adaptive_evidence_max,
            "source_coherence_max_sources": source_coherence_max_sources, "source_coherence_score_ratio": float(source_coherence_score_ratio), "source_coherence_score_mass": float(source_coherence_score_mass), "source_coherence_additional_candidates": int(source_coherence_additional_candidates), "source_coherence_additional_weight": float(source_coherence_additional_weight), "source_coherence_bonus": float(source_coherence_bonus), "source_coherence_penalty": float(source_coherence_penalty)}

    def _store(self, context_id=None) -> EventStateStore:
        key = self._context_id if context_id is None else context_id
        if key not in self._stores:
            self._stores[key] = EventStateStore(key)
        return self._stores[key]

    def set_context_id(self, context_id):
        self._context_id = context_id
        self._store(context_id)

    def reset(self):
        super().reset()
        self._stores.clear()

    @staticmethod
    def normalize_turn(item: Dict[str, Any], fallback_index: int, session: Dict[str, Any]) -> NormalizedTurn:
        role = str(item.get("role")).casefold() if item.get("role") else None
        speaker = item.get("speaker") or {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, "Unknown")
        return NormalizedTurn(canonical_turn_id(item.get("source_turn_id", item.get("turn_id", item.get("dia_id"))), fallback_index), session.get("source_session_id"), session.get("source_session_index"), item.get("source_event_id", session.get("source_event_id")), role, str(speaker), "speaker:" + normalize_name(str(speaker)), str(item.get("text", item.get("content", "")) or ""), item.get("image_caption", item.get("blip_caption")), item.get("timestamp", session.get("timestamp")))

    @classmethod
    def normalize_turns(cls, session: Dict[str, Any]) -> List[NormalizedTurn]:
        """Normalize IDs once and namespace accidental per-session collisions."""
        turns = [cls.normalize_turn(item, index, session) for index, item in enumerate(session["turns"])]
        seen: Dict[str, int] = {}
        used = set()
        for index, turn in enumerate(turns):
            count = seen.get(turn.source_turn_id, 0)
            seen[turn.source_turn_id] = count + 1
            if count:
                base = turn.source_turn_id
                candidate = f"{base}__turn_{index}"
                suffix = 1
                while candidate in used:
                    candidate = f"{base}__turn_{index}__dup_{suffix}"
                    suffix += 1
                turn.source_turn_id = candidate
            used.add(turn.source_turn_id)
        return turns

    @staticmethod
    def fallback_episode_summary(normalized: List[NormalizedTurn], limit: int = 1000) -> str:
        """Build a bounded chronological summary when extraction cannot be trusted."""
        if not normalized or limit <= 0:
            return ""
        labels = [f"{turn.speaker} (turn {turn.source_turn_id}): " for turn in normalized]
        separators = max(0, len(normalized) - 1) * 2
        body_budget = max(0, limit - separators - sum(len(label) for label in labels))
        per_turn = body_budget // len(normalized)
        remainder = body_budget % len(normalized)
        parts = []
        for index, (turn, label) in enumerate(zip(normalized, labels)):
            chars = per_turn + (1 if index < remainder else 0)
            text = turn.text.strip()
            if turn.image_caption:
                text = f"{text} [Shared image: {turn.image_caption}]".strip()
            if len(text) > chars:
                # Keep both ends so late-turn conclusions survive truncation.
                head = max(1, (chars - 3) // 2)
                tail = max(0, chars - head - 3)
                text = text[:head] + "..." + text[-tail:] if tail else text[:chars]
            parts.append(label + text)
        return "\n\n".join(parts)[:limit]

    @staticmethod
    def resolve_claim_source_speaker(raw_claim: Dict[str, Any], normalized_turns: List[NormalizedTurn]) -> Optional[str]:
        source_ids = set(raw_claim.get("source_turn_ids") or [])
        source_speakers = []
        for turn in normalized_turns:
            if turn.source_turn_id in source_ids and turn.speaker not in source_speakers:
                source_speakers.append(turn.speaker)
        if len(source_speakers) == 1:
            return source_speakers[0]
        if not source_ids and len({turn.speaker for turn in normalized_turns}) == 1:
            return normalized_turns[0].speaker
        return None

    @classmethod
    def resolve_claim_subject(cls, raw_claim: Dict[str, Any], normalized_turns: List[NormalizedTurn], conversation_scope: str) -> Optional[SubjectResolution]:
        """Resolve a proposed LLM subject ID against visible conversational evidence."""
        source_speaker = cls.resolve_claim_source_speaker(raw_claim, normalized_turns)
        participants = [turn.speaker for turn in normalized_turns]
        raw_subject = str(raw_claim.get("subject") or "").strip()
        raw_key = normalize_name(raw_subject)
        is_self_reference = raw_key in SELF_REFERENCES
        named_speakers = {normalize_name(name) for name in participants if normalize_name(name) not in {"user", "assistant", "system", "unknown"}}
        if is_self_reference and source_speaker is None and len(named_speakers) > 1:
            return None

        derived = resolve_subject_id(raw_subject, conversation_scope, participants, source_speaker)
        proposed = raw_claim.get("subject_id")
        proposal_is_canonical = is_canonical_subject_id(proposed)
        proposal_is_valid = proposal_is_canonical and is_valid_canonical_subject_proposal(proposed, conversation_scope, participants)
        proposed_id = resolve_subject_id(proposed, conversation_scope, participants, source_speaker) if proposal_is_valid else None
        compatible = proposed_id == derived
        if conversation_scope.startswith("third_party:") and (is_self_reference or raw_key in {"she", "he", "patient", "the_patient", "user", "the_user"}):
            derived = conversation_scope
            compatible = proposed_id == derived
        elif conversation_scope == "general_non_personal" and proposed_id == "primary_user":
            compatible = False
        display = raw_subject if is_visible_subject_identity(raw_subject, participants) else display_subject(derived)
        return SubjectResolution(derived, display, source_speaker, "visible_evidence", bool((proposal_is_canonical and not proposal_is_valid) or (proposed_id and not compatible)))

    @staticmethod
    def _normalize_source_sessions(text: str, memory_items: Optional[List[Dict[str, Any]]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        top_session = metadata.get("source_session_id")
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        order: List[Any] = []
        for index, raw_item in enumerate(memory_items or []):
            item = dict(raw_item)
            session_id = item.get("source_session_id", top_session)
            if session_id not in groups:
                groups[session_id], order = [], order + [session_id]
            item.setdefault("source_turn_id", item.get("turn_id", item.get("dia_id", index)))
            item.setdefault("text", item.get("content", ""))
            groups[session_id].append(item)
        if not groups:
            groups[top_session] = [{"speaker": "User", "text": text, "source_turn_id": 0, "timestamp": metadata.get("timestamp")}]
            order = [top_session]
        sessions = []
        for index, session_id in enumerate(order):
            turns = groups[session_id]
            session_text = "\n".join(str(item.get("text", item.get("content", "")) or "") for item in turns)
            explicit_scope = turns[0].get("conversation_scope")
            if explicit_scope is None and len(order) == 1:
                explicit_scope = metadata.get("conversation_scope")
            scope = normalize_scope(text if len(order) == 1 else session_text, explicit_scope)
            sessions.append({"source_session_id": session_id, "source_session_index": turns[0].get("source_session_index", metadata.get("source_session_index", index)), "source_event_id": turns[0].get("source_event_id", metadata.get("source_event_id")), "timestamp": turns[0].get("timestamp", metadata.get("timestamp")), "recorded_at_raw": turns[0].get("recorded_at_raw", metadata.get("recorded_at_raw")), "conversation_scope": scope, "turns": turns})
        return sessions

    def _extract(self, session: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_turns(session)
        participants = sorted({turn.speaker for turn in normalized})
        turn_text = "\n".join(f"[turn_id={turn.source_turn_id}] [role={turn.role or 'unknown'}] [speaker={turn.speaker}]\n{turn.text}{(' [Shared image: ' + turn.image_caption + ']') if turn.image_caption else ''}" for turn in normalized)
        allowed_subjects = ["primary_user", "general_non_personal"] + ["speaker:" + normalize_name(name) for name in participants]
        if session.get("conversation_scope", "").startswith("third_party:"):
            allowed_subjects.append(session["conversation_scope"])
        allowed_ids = {turn.source_turn_id for turn in normalized}
        prompt = f"Known session timestamp: {session.get('timestamp')}\nConversation scope: {session.get('conversation_scope')}\nParticipants: {participants}\nAllowed canonical subject IDs: {allowed_subjects}\nAllowed source_turn_ids: {sorted(allowed_ids)}\nUse only these canonical source_turn_ids exactly as listed; display wrappers such as [turn_id=...] are not IDs.\nExtract at most {self.max_claims_per_episode} claims. Prioritize distinct high-value long-term propositions; do not create low-value claims to fill the limit.\n{turn_text}"
        repaired = False
        parse_failure = False
        structure_failure = False
        validation_failures = 0
        missing_source_turn_id_claim_count = 0
        unknown_source_turn_id_claim_count = 0
        repaired_source_turn_id_claim_count = 0
        rejected_ungrounded_claim_count = 0
        meta_claim_rejected_count = 0
        no_information_claim_rejected_count = 0
        normalized_source_turn_reference_count = 0
        normalized_source_turn_reference_form_counts: Dict[str, int] = {}
        claim_level_grounding_failure_count = 0
        valid_claims_preserved_despite_other_claim_errors = 0
        invalid_claim_subset_repair_calls = 0
        invalid_claim_subset_repair_successes = 0
        invalid_claim_subset_repair_failures = 0
        grounding_repair_semantic_change_rejected_count = 0
        grounding_invalid_keys = set()
        invalid_source_turn_id_samples: List[Dict[str, Any]] = []
        invalid_source_turn_id_sample_types: List[str] = []
        invalid_extraction_output_previews: List[str] = []
        invalid_extraction_output_sha256: List[str] = []
        excess_claim_count = 0
        raw_turn_ids = [canonical_turn_id(item.get("source_turn_id", item.get("turn_id", item.get("dia_id"))), index) for index, item in enumerate(session["turns"])]
        turn_id_collision_count = len(raw_turn_ids) - len(set(raw_turn_ids))

        def extraction_messages(extraction_prompt: str) -> List[Dict[str, str]]:
            return format_messages(extraction_prompt, EXTRACTION_SYSTEM_PROMPT)

        def request_json(messages: List[Dict[str, str]]):
            try:
                return self._memory_llm_client.chat(
                    messages,
                    temperature=self.extraction_temperature,
                    max_tokens=self.extraction_max_tokens,
                    response_format={"type": "json_object"},
                )
            except TypeError as exc:
                message = str(exc).casefold()
                if "response_format" not in message and "unexpected keyword" not in message and "positional" not in message:
                    raise
                # Keep extraction decoding settings when a client lacks JSON mode.
                return self._memory_llm_client.chat(
                    messages,
                    temperature=self.extraction_temperature,
                    max_tokens=self.extraction_max_tokens,
                )

        def record_invalid_output(content: str) -> None:
            invalid_extraction_output_previews.append(content[:1000])
            invalid_extraction_output_sha256.append(hashlib.sha256(content.encode("utf-8", "replace")).hexdigest())

        def parse_and_validate(content: str, record_telemetry: bool = True) -> Dict[str, Any]:
            nonlocal structure_failure, validation_failures, missing_source_turn_id_claim_count, unknown_source_turn_id_claim_count, repaired_source_turn_id_claim_count, rejected_ungrounded_claim_count, meta_claim_rejected_count, no_information_claim_rejected_count, grounding_invalid_keys, invalid_source_turn_id_samples, invalid_source_turn_id_sample_types, excess_claim_count, normalized_source_turn_reference_count, claim_level_grounding_failure_count, valid_claims_preserved_despite_other_claim_errors
            value = parse_json(content)
            raw_claims = value.get("claims")
            if not isinstance(raw_claims, list):
                if record_telemetry:
                    structure_failure = True
                raise ValueError("claims must be a JSON array")
            summary = value.get("episode_summary")
            if summary is not None and not isinstance(summary, str):
                if record_telemetry:
                    structure_failure = True
                raise ValueError("episode_summary must be a string")
            if record_telemetry:
                excess_claim_count += max(0, len(raw_claims) - self.max_claims_per_episode)
            claims_to_validate = raw_claims[:self.max_claims_per_episode]
            claims = []
            invalid_claims = []
            invalid_grounding = 0
            valid_grounding_keys = set()
            for raw in claims_to_validate:
                claim_key = tuple(str(raw.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value")) if isinstance(raw, dict) else ()
                normalized_raw = raw
                grounding_bad = False
                if not isinstance(raw, dict):
                    invalid_claims.append({"raw_claim": raw, "reason": "schema_invalid"})
                    if record_telemetry:
                        validation_failures += 1
                    continue
                supplied = raw.get("source_turn_ids")
                if not isinstance(supplied, list) or not supplied:
                    if record_telemetry:
                        missing_source_turn_id_claim_count += 1
                    if len(allowed_ids) == 1:
                        normalized_raw = {**raw, "source_turn_ids": [next(iter(allowed_ids))]}
                        if record_telemetry:
                            repaired_source_turn_id_claim_count += 1
                    else:
                        grounding_bad = True
                else:
                    resolved_ids = []
                    unknown_reference_seen = False
                    for reference in supplied:
                        resolved, form = resolve_model_turn_reference_with_form(reference, allowed_ids)
                        if resolved is None:
                            grounding_bad = True
                            if record_telemetry and not unknown_reference_seen:
                                unknown_source_turn_id_claim_count += 1
                                unknown_reference_seen = True
                            if record_telemetry:
                                if len(invalid_source_turn_id_samples) < 5:
                                    invalid_source_turn_id_samples.append({"value": reference, "type": type(reference).__name__})
                                    invalid_source_turn_id_sample_types.append(type(reference).__name__)
                        else:
                            resolved_ids.append(resolved)
                            if record_telemetry and form:
                                normalized_source_turn_reference_count += 1
                                normalized_source_turn_reference_form_counts[form] = normalized_source_turn_reference_form_counts.get(form, 0) + 1
                    if not grounding_bad:
                        normalized_raw = {**raw, "source_turn_ids": resolved_ids}
                if grounding_bad:
                    invalid_claims.append({"raw_claim": raw, "reason": "grounding_invalid"})
                    invalid_grounding += 1
                    if record_telemetry:
                        grounding_invalid_keys.add(claim_key)
                    continue
                if is_meta_claim(normalized_raw):
                    if record_telemetry:
                        meta_claim_rejected_count += 1
                    continue
                if is_no_information_value(normalized_raw.get("value")):
                    if record_telemetry:
                        no_information_claim_rejected_count += 1
                    continue
                item = validated_claim(normalized_raw, allowed_ids, allowed_ids)
                if item is None:
                    invalid_claims.append({"raw_claim": raw, "reason": "schema_invalid"})
                    if record_telemetry:
                        validation_failures += 1
                    continue
                if not self.event_time_normalization_enabled:
                    item.update({"event_time_start": None, "event_time_end": None, "event_time_precision": "unknown"})
                claims.append(item)
                valid_grounding_keys.add(tuple(str(item.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value")))
            if record_telemetry:
                claim_level_grounding_failure_count += invalid_grounding
                if invalid_claims and claims:
                    valid_claims_preserved_despite_other_claim_errors += len(claims)
                rejected_ungrounded_claim_count += invalid_grounding
            accepted = [item for item in claims if item is not None]
            return {"episode_summary": summary, "claims": accepted, "invalid_claims": invalid_claims, "valid_grounding_keys": valid_grounding_keys, "invalid_grounding_count": invalid_grounding}

        subset_repair_failure = False
        repair_attempted = False
        structural_repair_success = 0
        structural_repair_failure = 0
        fragment_salvage_attempt_count = 0
        fragment_salvage_success_count = 0
        fragment_salvaged_claim_count = 0
        recovery_calls = 0
        recovery_successes = 0
        recovery_failures = 0
        semantic_extraction_unavailable_count = 0
        response_content = ""
        try:
            with get_usage_tracker().scope("event_state.extract"):
                response = request_json(extraction_messages(prompt))
            response_content = str(getattr(response, "content", "") or "")
            parsed = parse_and_validate(response_content)
        except ValueError as exc:
            parse_failure = not structure_failure
            record_invalid_output(response_content)
            fragment_salvage_attempt_count = 1
            fragments = salvage_extraction_fragments(response_content)
            salvaged = parse_and_validate(json.dumps(fragments), record_telemetry=True)
            fragment_salvaged_claim_count = len(salvaged["claims"])
            fragment_salvage_success_count = int(bool(salvaged["claims"]) or fragments["episode_summary"] is not None)
            try:
                repair_attempted = True
                with get_usage_tracker().scope("event_state.extract_repair"):
                    repair_prompt = (
                        "Perform JSON / structure repair only. Return a JSON object with an "
                        "episode_summary string or null and a claims array. Preserve only complete "
                        "claims literally present in the previous output; do not add or reinterpret "
                        f"claims. Claims may cite only these source_turn_ids: {sorted(allowed_ids)}. "
                        f"Structural issue: {exc}\nPrevious output:\n{response_content}"
                    )
                    response = request_json(format_messages(repair_prompt, STRUCTURED_OUTPUT_REPAIR_SYSTEM_PROMPT))
                repaired = True
                repaired_content = str(getattr(response, "content", "") or "")
                parsed = parse_and_validate(repaired_content, record_telemetry=False)
                structural_repair_success = 1
            except ValueError:
                structural_repair_failure = 1
                record_invalid_output(str(getattr(locals().get("response", None), "content", "") or ""))
                if salvaged["claims"]:
                    parsed = salvaged
                else:
                    recovery_calls = 1
                    try:
                        with get_usage_tracker().scope("event_state.extract_recovery"):
                            response = request_json(extraction_messages(prompt))
                        recovery_content = str(getattr(response, "content", "") or "")
                        parsed = parse_and_validate(recovery_content, record_telemetry=False)
                        recovery_successes = 1
                    except ValueError:
                        recovery_failures = 1
                        record_invalid_output(str(getattr(locals().get("response", None), "content", "") or ""))
                        parsed = {"episode_summary": salvaged.get("episode_summary"), "claims": [], "invalid_claims": []}
                        semantic_extraction_unavailable_count = 1
        if parsed.get("invalid_claims") and not repaired:
            invalid_claim_subset_repair_calls = 1
            try:
                subset_json = json.dumps([item["raw_claim"] for item in parsed["invalid_claims"]], ensure_ascii=True, default=str)
                repair_prompt = f"Repair this extraction claim subset as valid JSON only. Return an object with a claims array containing only these claims, preserving their semantic propositions. Do not create new claims, echo valid claims, or guess provenance. Correct source_turn_ids only when supported by visible turns and use only these bare allowed IDs: {sorted(allowed_ids)}. Invalid claims:\n{subset_json}\nVisible turns:\n{turn_text}"
                with get_usage_tracker().scope("event_state.extract_repair"):
                    messages = format_messages(repair_prompt, STRUCTURED_OUTPUT_REPAIR_SYSTEM_PROMPT)
                    response = request_json(messages)
                subset = parse_and_validate(response.content, record_telemetry=False)
                existing = {
                    claim_repair_identity(item)
                    for item in parsed["claims"]
                }
                repairable_keys = {
                    tuple(str(item["raw_claim"].get(key) or "").strip().casefold() for key in ("subject", "predicate", "value"))
                    for item in parsed["invalid_claims"]
                    if item["reason"] == "schema_invalid" and isinstance(item["raw_claim"], dict)
                }
                grounding_by_key = {}
                for invalid in parsed["invalid_claims"]:
                    raw_claim = invalid["raw_claim"]
                    if invalid["reason"] == "grounding_invalid" and isinstance(raw_claim, dict):
                        key = tuple(str(raw_claim.get(field) or "").strip().casefold() for field in ("subject", "predicate", "value"))
                        grounding_by_key.setdefault(key, []).append(invalid)
                repaired_items = []
                for item in subset["claims"]:
                    item_key = tuple(str(item.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value"))
                    grounding_matches = grounding_by_key.get(item_key, [])
                    if grounding_matches and not any(claim_semantic_fingerprint(item) == claim_semantic_fingerprint(invalid["raw_claim"]) for invalid in grounding_matches):
                        grounding_repair_semantic_change_rejected_count += 1
                        continue
                    if not grounding_matches and item_key not in repairable_keys:
                        if grounding_by_key:
                            grounding_repair_semantic_change_rejected_count += 1
                        continue
                    identity = claim_repair_identity(item)
                    if identity not in existing:
                        existing.add(identity)
                        repaired_items.append(item)
                repaired_count = sum(tuple(str(item.get(key) or "").strip().casefold() for key in ("subject", "predicate", "value")) in grounding_invalid_keys for item in repaired_items)
                parsed["claims"].extend(repaired_items)
                repaired_source_turn_id_claim_count += repaired_count
                rejected_ungrounded_claim_count = max(0, rejected_ungrounded_claim_count - repaired_count)
                if repaired_items:
                    invalid_claim_subset_repair_successes = 1
                else:
                    invalid_claim_subset_repair_failures = 1
                    subset_repair_failure = True
                if subset.get("invalid_claims"):
                    invalid_claim_subset_repair_failures = 1
                    subset_repair_failure = True
            except Exception as repair_exc:
                if self._is_provider_error(repair_exc):
                    raise
                invalid_claim_subset_repair_failures = 1
                subset_repair_failure = True
        claims = parsed.get("claims", [])[:self.max_claims_per_episode]
        repair_failed = bool(structural_repair_failure) or subset_repair_failure or (bool(parsed.get("invalid_claims")) and repaired)
        return {"episode_summary": parsed.get("episode_summary") or (self.fallback_episode_summary(normalized) if not claims else ""), "claims": claims, "parse_failure": parse_failure, "json_parse_failure": parse_failure, "structure_failure": structure_failure, "claim_validation_failures": validation_failures, "repair_failure": repair_failed, "repair_calls": int(repair_attempted) + invalid_claim_subset_repair_calls, "structural_repair_success_count": structural_repair_success, "structural_repair_failure_count": structural_repair_failure, "fragment_salvage_attempt_count": fragment_salvage_attempt_count, "fragment_salvage_success_count": fragment_salvage_success_count, "fragment_salvaged_claim_count": fragment_salvaged_claim_count, "recovery_calls": recovery_calls, "recovery_successes": recovery_successes, "recovery_failures": recovery_failures, "semantic_extraction_unavailable_count": semantic_extraction_unavailable_count, "invalid_extraction_output_previews": invalid_extraction_output_previews, "invalid_extraction_output_sha256": invalid_extraction_output_sha256, "missing_source_turn_id_claim_count": missing_source_turn_id_claim_count, "unknown_source_turn_id_claim_count": unknown_source_turn_id_claim_count, "repaired_source_turn_id_claim_count": repaired_source_turn_id_claim_count, "rejected_ungrounded_claim_count": rejected_ungrounded_claim_count, "meta_claim_rejected_count": meta_claim_rejected_count, "no_information_claim_rejected_count": no_information_claim_rejected_count, "normalized_source_turn_reference_count": normalized_source_turn_reference_count, "normalized_source_turn_reference_form_counts": normalized_source_turn_reference_form_counts, "claim_level_grounding_failure_count": claim_level_grounding_failure_count, "valid_claims_preserved_despite_other_claim_errors": valid_claims_preserved_despite_other_claim_errors, "invalid_claim_subset_repair_calls": invalid_claim_subset_repair_calls, "invalid_claim_subset_repair_successes": invalid_claim_subset_repair_successes, "invalid_claim_subset_repair_failures": invalid_claim_subset_repair_failures, "grounding_repair_semantic_change_rejected_count": grounding_repair_semantic_change_rejected_count, "invalid_source_turn_id_samples": invalid_source_turn_id_samples, "invalid_source_turn_id_sample_types": invalid_source_turn_id_sample_types, "allowed_source_turn_ids": sorted(allowed_ids), "excess_claim_count": excess_claim_count, "turn_id_collision_count": turn_id_collision_count}

    @staticmethod
    def _is_provider_error(exc: Exception) -> bool:
        if isinstance(exc, LLMAPIError):
            return True
        name = type(exc).__name__.casefold()
        return any(token in name for token in ("api", "provider", "timeout", "connection", "retry"))

    @staticmethod
    def _valid_from(value: Any, recorded_at: Any, valid_time_text: Any, enabled: bool) -> Any:
        """Retain only extractor-established lifecycle bounds.

        Relative occurrence wording belongs to normalized event time, not the
        lifecycle of a state/version.  The extra arguments remain for callers
        and old integrations, but intentionally do not infer a date.
        """
        del recorded_at, valid_time_text
        return value if enabled else None

    def truncate_to_tokens(self, text: str, limit: int) -> str:
        return self._tokenizer.decode(self._tokenizer.encode(text)[:max(0, int(limit))])

    def _prepare_session(self, session: Dict[str, Any], context_id: Any) -> PreparedMemorySession:
        extracted = self._extract(session)
        normalized = self.normalize_turns(session)
        raw_lines = [f"[turn_id={turn.source_turn_id}] [role={turn.role or 'unknown'}] [speaker={turn.speaker}]\n{turn.text}" + (f"\n[Shared image: {turn.image_caption}]" if turn.image_caption else "") for turn in normalized]
        raw = f"[conversation_scope={session['conversation_scope']}]\n[recorded_at={session.get('timestamp') or 'unknown'}]\n\n" + "\n\n".join(raw_lines)
        episode_id = self._store(context_id).stable_id("E", [context_id, session["source_session_id"], raw])
        episode = Episode(episode_id, context_id, session["source_session_id"], session["source_session_index"], session["source_event_id"], session.get("timestamp"), sorted({turn.speaker for turn in normalized}), session["conversation_scope"], raw if self.store_raw_episode_text else "", extracted["episode_summary"], [TurnEvidence(turn.source_turn_id, turn.speaker, turn.role, turn.text, turn.timestamp, turn.image_caption, turn.source_session_id, turn.source_session_index, turn.source_event_id) for turn in normalized] if self.preserve_turn_evidence else [], session.get("recorded_at_raw"))
        claims: List[Claim] = []
        if self.enable_state_claims:
            for index, raw_claim in enumerate(extracted["claims"]):
                resolution = self.resolve_claim_subject(raw_claim, normalized, session["conversation_scope"])
                if resolution is None:
                    continue
                extracted.setdefault("subject_resolution_counts", {})[resolution.resolution_source] = extracted.setdefault("subject_resolution_counts", {}).get(resolution.resolution_source, 0) + 1
                if resolution.conflict_detected:
                    extracted["invalid_proposed_subject_id_count"] = extracted.get("invalid_proposed_subject_id_count", 0) + 1
                    extracted["subject_resolution_override_count"] = extracted.get("subject_resolution_override_count", 0) + 1
                persistence = raw_claim["persistence"]
                temporal_raw = dict(raw_claim)
                temporal_raw["valid_from"] = self._valid_from(raw_claim.get("valid_from"), session.get("timestamp"), raw_claim.get("valid_time_text"), self.enable_bitemporal_time)
                temporal_raw, temporal_telemetry = normalize_ingress_temporal_bounds(
                    temporal_raw, session.get("timestamp"), self.enable_bitemporal_time
                )
                for key, value in temporal_telemetry.items():
                    extracted[key] = extracted.get(key, 0) + value
                valid_from = temporal_raw["valid_from"]
                valid_to = temporal_raw["valid_to"]
                state_slot = raw_claim.get("state_slot") if persistence == "state" else None
                if persistence == "state" and not state_slot:
                    state_slot = normalize_state_slot(raw_claim["predicate"])
                    extracted["state_claim_slot_fallback_count"] = extracted.get("state_claim_slot_fallback_count", 0) + 1
                claims.append(Claim(self._store(context_id).stable_id("C", [episode_id, index, resolution.subject_id, raw_claim["predicate"], raw_claim["value"]]), resolution.subject_display, resolution.subject_id, raw_claim["predicate"], raw_claim["value"], raw_claim["qualifiers"], raw_claim["polarity"], raw_claim["modality"], persistence, session.get("timestamp"), valid_from, valid_to, raw_claim.get("valid_time_text"), "standalone" if persistence == "history" else "active", [EvidenceRef(episode_id, session["source_session_id"], raw_claim["source_turn_ids"], "origin")], raw_claim["confidence"], resolution.subject_id, state_slot, event_time_start=raw_claim.get("event_time_start"), event_time_end=raw_claim.get("event_time_end"), event_time_precision=raw_claim.get("event_time_precision", "unknown")))
        turn_embeddings: List[List[float]] = []
        if self.enable_episodes:
            # Build the immutable-turn index with the episode, so restored
            # query runs never need to re-embed an already indexed snapshot.
            embedding_texts = [episode.retrieval_text()] + [
                episode_turn_embedding_text(turn) for turn in episode.turn_evidence
            ]
            with get_usage_tracker().scope("event_state.embedding"):
                embeddings = self._embedder.embed_documents(embedding_texts)
            episode_embedding = embeddings[0]
            turn_embeddings = [list(item) for item in embeddings[1:]]
        else:
            episode_embedding = []
        claim_embeddings, claim_slot_embeddings = [], [[] for _ in claims]
        if self.enable_state_claims:
            with get_usage_tracker().scope("event_state.embedding"):
                claim_embeddings = self._embedder.embed_documents([claim.semantic_text() for claim in claims])
                state_slot_items = [(index, StateCompiler.slot_text(claim)) for index, claim in enumerate(claims) if claim.persistence == "state"]
                if state_slot_items:
                    state_slot_embeddings = self._embedder.embed_documents([text for _, text in state_slot_items])
                    for (index, _), slot_embedding in zip(state_slot_items, state_slot_embeddings):
                        claim_slot_embeddings[index] = list(slot_embedding)
        return PreparedMemorySession(context_id, session, normalized, extracted, episode, list(episode_embedding), turn_embeddings, claims, [list(item) for item in claim_embeddings], [list(item) for item in claim_slot_embeddings])

    def prepare_memory_sessions(
        self,
        text: str,
        *,
        progress_callback: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> List[PreparedMemorySession]:
        """Prepare independent source sessions concurrently, preserving source order."""
        get_usage_tracker().set_phase("memorize")
        context_id = kwargs.get("context_id", self._context_id)
        sessions = self._normalize_source_sessions(text, kwargs.get("memory_items"), kwargs)
        if self.event_state_workers == 1 or len(sessions) < 2:
            prepared_sessions = []
            for session in sessions:
                prepared_sessions.append(self._prepare_session(session, context_id))
                if progress_callback is not None:
                    progress_callback()
            return prepared_sessions
        worker_count = min(self.event_state_workers, len(sessions))
        def prepare_in_worker(session: Dict[str, Any]) -> PreparedMemorySession:
            worker_context = contextvars.copy_context()

            def run() -> PreparedMemorySession:
                get_usage_tracker().set_phase("memorize")
                return self._prepare_session(session, context_id)

            return worker_context.run(run)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_indexes = {
                executor.submit(prepare_in_worker, session): index
                for index, session in enumerate(sessions)
            }
            prepared_sessions: List[Optional[PreparedMemorySession]] = [None] * len(sessions)
            for future in as_completed(future_indexes):
                prepared_sessions[future_indexes[future]] = future.result()
                if progress_callback is not None:
                    progress_callback()

        # Workers finish out of order, but stateful commits consume source order.
        return [session for session in prepared_sessions if session is not None]

    def _commit_prepared(self, prepared: PreparedMemorySession, store: EventStateStore, counts: Dict[str, Any], added_records: List[Dict[str, Any]]) -> None:
        extracted = prepared.extracted
        counts["extract_parse_failures"] += int(extracted.get("parse_failure", False))
        counts["extract_json_parse_failures"] += int(extracted.get("json_parse_failure", False))
        counts["extract_structure_failures"] += int(extracted.get("structure_failure", False))
        counts["extract_claim_validation_failures"] += int(extracted.get("claim_validation_failures", 0))
        counts["extract_repair_failures"] += int(extracted.get("repair_failure", False))
        counts["extract_repair_calls"] += int(extracted.get("repair_calls", 0))
        counts["extracted_state_claim_count"] += sum(item.get("persistence") == "state" for item in extracted.get("claims", []))
        counts["extracted_history_claim_count"] += sum(item.get("persistence") == "history" for item in extracted.get("claims", []))
        counts["extracted_episode_claim_count"] += sum(item.get("persistence") == "episode" for item in extracted.get("claims", []))
        for key in ("structural_repair_success_count", "structural_repair_failure_count", "fragment_salvage_attempt_count", "fragment_salvage_success_count", "fragment_salvaged_claim_count", "recovery_calls", "recovery_successes", "recovery_failures", "semantic_extraction_unavailable_count", "temporal_ingress_guard_count", "temporal_ingress_future_valid_from_cleared_count", "temporal_ingress_state_valid_to_cleared_count", "temporal_ingress_invalid_interval_count", "missing_source_turn_id_claim_count", "unknown_source_turn_id_claim_count", "repaired_source_turn_id_claim_count", "rejected_ungrounded_claim_count", "meta_claim_rejected_count", "no_information_claim_rejected_count", "normalized_source_turn_reference_count", "claim_level_grounding_failure_count", "valid_claims_preserved_despite_other_claim_errors", "invalid_claim_subset_repair_calls", "invalid_claim_subset_repair_successes", "invalid_claim_subset_repair_failures", "grounding_repair_semantic_change_rejected_count", "state_claim_slot_fallback_count", "invalid_proposed_subject_id_count", "subject_resolution_override_count", "excess_claim_count", "turn_id_collision_count"):
            counts[key] += int(extracted.get(key, 0))
        for form, count in extracted.get("normalized_source_turn_reference_form_counts", {}).items():
            counts["normalized_source_turn_reference_form_counts"][form] = counts["normalized_source_turn_reference_form_counts"].get(form, 0) + int(count)
        counts["invalid_source_turn_id_samples"].extend(extracted.get("invalid_source_turn_id_samples", []))
        counts["invalid_source_turn_id_sample_types"].extend(extracted.get("invalid_source_turn_id_sample_types", []))
        counts["invalid_extraction_output_previews"].extend(extracted.get("invalid_extraction_output_previews", []))
        counts["invalid_extraction_output_sha256"].extend(extracted.get("invalid_extraction_output_sha256", []))
        counts["allowed_source_turn_ids"].extend(extracted.get("allowed_source_turn_ids", []))
        episode = prepared.episode
        if self.enable_episodes:
            store.add_episode(episode, prepared.episode_embedding, prepared.turn_embeddings)
            counts["episodes_added"] += 1
            added_records.append({"id": episode.episode_id, "type": "episode", "memory": episode.summary, "source_uid": episode.source_session_id, "source_session_id": episode.source_session_id})
        if not self.enable_state_claims:
            return
        counts["ambiguous_subject_claim_count"] += max(0, len(prepared.extracted.get("claims", [])) - len(prepared.claims))
        counts["claims_extracted"] += len(prepared.extracted.get("claims", []))
        counts["accepted_claim_count"] += len(prepared.claims)
        counts["rejected_claim_count"] += max(0, len(prepared.extracted.get("claims", [])) - len(prepared.claims))
        counts["primary_user_claim_count"] += sum(claim.subject_id == "primary_user" for claim in prepared.claims)
        counts["third_party_claim_count"] += sum(claim.subject_id.startswith("third_party:") for claim in prepared.claims)
        counts["general_non_personal_claim_count"] += sum(claim.subject_id == "general_non_personal" for claim in prepared.claims)
        counts["real_speaker_claim_count"] += sum(claim.subject_id.startswith("speaker:") for claim in prepared.claims)
        compiler = StateCompiler(store, self._embedder, self._memory_llm_client, candidate_top_k=self._build_config["state_candidate_top_k"], current_candidate_top_k=self._build_config["state_current_candidate_top_k"], min_similarity=self._build_config["state_candidate_min_similarity"], min_confidence=self._build_config["update_min_confidence"], update_temperature=self._build_config["update_temperature"], update_max_tokens=self._build_config["update_max_tokens"])
        for claim, claim_embedding, slot_embedding in zip(prepared.claims, prepared.claim_embeddings, prepared.claim_slot_embeddings):
            if not self.enable_state_compilation:
                store.add_claim(claim, claim_embedding, slot_embedding)
                counts["uncompiled_claim_count"] += 1
                counts["new_count"] += 1
                added_records.append({"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "source_uid": episode.source_session_id, "source_session_id": episode.source_session_id})
                continue
            compile_result = compiler.apply(claim, episode.episode_id, claim_embedding, slot_embedding)
            operation = compile_result.operation
            counts["update_llm_calls"] += compiler.update_llm_calls
            counts["update_parse_failures"] += compiler.update_parse_failures
            counts["update_repair_calls"] += compiler.update_repair_calls
            counts["update_repair_successes"] += compiler.update_repair_successes
            counts["update_repair_failures"] += compiler.update_repair_failures
            counts["invalid_update_output_previews"].extend(compiler.invalid_update_output_previews)
            counts["invalid_update_output_sha256"].extend(compiler.invalid_update_output_sha256)
            counts["state_candidate_queries"] += compiler.state_candidate_queries
            counts["state_candidate_no_match_count"] += compiler.state_candidate_no_match_count
            counts["state_candidate_exact_slot_match_count"] += compiler.state_candidate_exact_slot_match_count
            counts["state_candidate_semantic_slot_match_count"] += compiler.state_candidate_semantic_slot_match_count
            counts["state_candidates_rejected_below_threshold"] += compiler.state_candidates_rejected_below_threshold
            counts["different_state_dimension_guard_count"] += compiler.different_state_dimension_guard_count
            counts["supersede_temporal_guard_count"] += compiler.supersede_temporal_guard_count
            counts["supersede_record_time_fallback_count"] += compiler.supersede_record_time_fallback_count
            counts["retroactive_correction_applied_count"] += compiler.retroactive_correction_applied_count
            counts["state_value_relation_equivalent_count"] += compiler.state_value_relation_equivalent_count
            counts["state_value_relation_refinement_count"] += compiler.state_value_relation_refinement_count
            counts["state_value_relation_changed_count"] += compiler.state_value_relation_changed_count
            counts["state_value_relation_contradictory_count"] += compiler.state_value_relation_contradictory_count
            counts["state_value_relation_uncertain_count"] += compiler.state_value_relation_uncertain_count
            counts["state_value_relation_guard_count"] += compiler.state_value_relation_guard_count
            counts[f"{operation.casefold()}_count"] = counts.get(f"{operation.casefold()}_count", 0) + 1
            if compile_result.same_session:
                counts[f"same_session_{operation.casefold()}_count"] = counts.get(f"same_session_{operation.casefold()}_count", 0) + 1
                if operation == "CORROBORATE":
                    counts["same_session_corrob_count"] += 1
            elif compile_result.matched_claim_id:
                counts[f"cross_session_{operation.casefold()}_count"] = counts.get(f"cross_session_{operation.casefold()}_count", 0) + 1
                if operation == "CORROBORATE":
                    counts["cross_session_corrob_count"] += 1
            if compile_result.transition_guard:
                counts["same_session_transition_guard_count"] += 1
                if compile_result.transition_guard == "same_session_supersede_guard":
                    counts["same_session_supersede_guard_count"] += 1
                elif compile_result.transition_guard == "same_session_refine_guard":
                    counts["same_session_refine_guard_count"] += 1
                elif compile_result.transition_guard == "same_session_conflict_guard":
                    counts["same_session_conflict_guard_count"] += 1
            if compile_result.fallback_reason == "same_session_corrob_downgraded":
                counts["same_session_corrob_downgraded_to_duplicate_count"] += 1
            if operation in {"NEW", "REFINE", "SUPERSEDE", "CONFLICT"}:
                counts["canonical_claims_added"] += 1
                added_records.append({"id": claim.claim_id, "type": "state_claim", "memory": claim.semantic_text(), "source_uid": episode.source_session_id, "source_session_id": episode.source_session_id})
            if operation == "EPISODIC":
                counts["episodic_claims_added"] += 1
            if operation == "NEW" and compile_result.fallback_reason == "below_confidence_threshold":
                counts["low_confidence_new_count"] += 1
            compiler.update_llm_calls = 0
            compiler.update_parse_failures = 0
            compiler.update_repair_calls = compiler.update_repair_successes = compiler.update_repair_failures = 0
            compiler.invalid_update_output_previews.clear()
            compiler.invalid_update_output_sha256.clear()
            compiler.state_candidate_queries = compiler.state_candidate_no_match_count = 0
            compiler.state_candidate_exact_slot_match_count = compiler.state_candidate_semantic_slot_match_count = 0
            compiler.state_candidates_rejected_below_threshold = compiler.different_state_dimension_guard_count = 0
            compiler.supersede_temporal_guard_count = compiler.supersede_record_time_fallback_count = compiler.retroactive_correction_applied_count = 0
            compiler.state_value_relation_equivalent_count = compiler.state_value_relation_refinement_count = 0
            compiler.state_value_relation_changed_count = compiler.state_value_relation_contradictory_count = 0
            compiler.state_value_relation_uncertain_count = compiler.state_value_relation_guard_count = 0

    def commit_prepared_memory(self, prepared_sessions: List[PreparedMemorySession], text: str = "", context_id: Any = None) -> MemoryBuildResult:
        context_id = self._context_id if context_id is None else context_id
        store = self._store(context_id)
        counts = {"episodes_added": 0, "claims_extracted": 0, "accepted_claim_count": 0, "rejected_claim_count": 0, "canonical_claims_added": 0, "episodic_claims_added": 0, "uncompiled_claim_count": 0, "extracted_state_claim_count": 0, "extracted_history_claim_count": 0, "extracted_episode_claim_count": 0, "extract_parse_failures": 0, "extract_json_parse_failures": 0, "extract_structure_failures": 0, "extract_claim_validation_failures": 0, "extract_repair_calls": 0, "extract_repair_failures": 0, "ambiguous_subject_claim_count": 0, "missing_source_turn_id_claim_count": 0, "unknown_source_turn_id_claim_count": 0, "repaired_source_turn_id_claim_count": 0, "rejected_ungrounded_claim_count": 0, "meta_claim_rejected_count": 0, "no_information_claim_rejected_count": 0, "state_claim_slot_fallback_count": 0, "invalid_proposed_subject_id_count": 0, "subject_resolution_override_count": 0, "excess_claim_count": 0, "turn_id_collision_count": 0, "invalid_source_turn_id_samples": [], "invalid_source_turn_id_sample_types": [], "allowed_source_turn_ids": [], "invalid_update_output_previews": [], "invalid_update_output_sha256": [], "same_session_transition_guard_count": 0, "same_session_supersede_guard_count": 0, "same_session_refine_guard_count": 0, "same_session_conflict_guard_count": 0, "same_session_corrob_downgraded_to_duplicate_count": 0, "same_session_duplicate_count": 0, "same_session_corroborate_count": 0, "same_session_corrob_count": 0, "same_session_refine_count": 0, "same_session_supersede_count": 0, "same_session_conflict_count": 0, "cross_session_duplicate_count": 0, "cross_session_corroborate_count": 0, "cross_session_corrob_count": 0, "cross_session_refine_count": 0, "cross_session_supersede_count": 0, "cross_session_conflict_count": 0, "primary_user_claim_count": 0, "third_party_claim_count": 0, "general_non_personal_claim_count": 0, "real_speaker_claim_count": 0, "update_llm_calls": 0, "update_parse_failures": 0, "update_repair_calls": 0, "update_repair_successes": 0, "update_repair_failures": 0, "state_candidate_queries": 0, "state_candidate_no_match_count": 0, "state_candidate_exact_slot_match_count": 0, "state_candidate_semantic_slot_match_count": 0, "state_candidates_rejected_below_threshold": 0, "different_state_dimension_guard_count": 0, "supersede_temporal_guard_count": 0, "supersede_record_time_fallback_count": 0, "retroactive_correction_applied_count": 0, "state_value_relation_equivalent_count": 0, "state_value_relation_refinement_count": 0, "state_value_relation_changed_count": 0, "state_value_relation_contradictory_count": 0, "state_value_relation_uncertain_count": 0, "state_value_relation_guard_count": 0, "low_confidence_new_count": 0, "low_extraction_confidence_count": 0, "new_count": 0, "duplicate_count": 0, "corroborate_count": 0, "refine_count": 0, "supersede_count": 0, "conflict_count": 0, "episodic_count": 0}
        counts.update({"normalized_source_turn_reference_count": 0, "normalized_source_turn_reference_form_counts": {}, "claim_level_grounding_failure_count": 0, "valid_claims_preserved_despite_other_claim_errors": 0, "invalid_claim_subset_repair_calls": 0, "invalid_claim_subset_repair_successes": 0, "invalid_claim_subset_repair_failures": 0, "grounding_repair_semantic_change_rejected_count": 0, "structural_repair_success_count": 0, "structural_repair_failure_count": 0, "fragment_salvage_attempt_count": 0, "fragment_salvage_success_count": 0, "fragment_salvaged_claim_count": 0, "recovery_calls": 0, "recovery_successes": 0, "recovery_failures": 0, "semantic_extraction_unavailable_count": 0, "invalid_extraction_output_previews": [], "invalid_extraction_output_sha256": [], "temporal_ingress_guard_count": 0, "temporal_ingress_future_valid_from_cleared_count": 0, "temporal_ingress_state_valid_to_cleared_count": 0, "temporal_ingress_invalid_interval_count": 0})
        added_records: List[Dict[str, Any]] = []
        for prepared in prepared_sessions:
            self._commit_prepared(prepared, store, counts, added_records)
        self._memory_chunks = [episode.summary for episode in store.episodes.values()]
        self._is_initialized = bool(store.episodes or store.claims)
        extra = {**counts, **store.claim_counts(), "invalid_update_output_previews": counts["invalid_update_output_previews"][:3], "invalid_update_output_preview": counts["invalid_update_output_previews"][:3], "invalid_update_output_sha256": counts["invalid_update_output_sha256"][:3], "invalid_extraction_output_previews": counts["invalid_extraction_output_previews"][:3], "invalid_extraction_output_preview": counts["invalid_extraction_output_previews"][:3], "invalid_extraction_output_sha256": counts["invalid_extraction_output_sha256"][:3], "state_claim_count_with_slot": sum(claim.persistence == "state" and bool(claim.state_slot) for claim in store.claims.values()), "distinct_active_state_slot_count": len({claim.state_slot for claim in store.claims.values() if claim.persistence == "state" and claim.status in {"active", "contested"} and claim.state_slot}), "invalid_source_turn_id_samples": counts["invalid_source_turn_id_samples"][:5], "invalid_source_turn_id_sample_types": sorted(set(counts["invalid_source_turn_id_sample_types"][:5])), "allowed_source_turn_ids": sorted(set(counts["allowed_source_turn_ids"])), "inserted_count": len(added_records), "active_state_count": sum(claim.status == "active" for claim in store.claims.values()), "superseded_state_count": sum(claim.status == "superseded" for claim in store.claims.values()), "refined_state_count": sum(claim.status == "refined" for claim in store.claims.values()), "standalone_claim_count": sum(claim.status == "standalone" for claim in store.claims.values()), "standalone_history_count": sum(claim.status == "standalone" and claim.persistence == "history" for claim in store.claims.values()), "standalone_episode_count": sum(claim.status == "standalone" and claim.persistence == "episode" for claim in store.claims.values()), "contested_state_count": sum(claim.status == "contested" for claim in store.claims.values()), "claims_with_multiple_provenance_references": sum(len(claim.evidence) > 1 for claim in store.claims.values()), "claims_with_evidence_from_multiple_sessions": sum(len({ref.source_session_id for ref in claim.evidence}) > 1 for claim in store.claims.values()), "claims_with_multi_turn_evidence": sum(any(len(ref.source_turn_ids) > 1 for ref in claim.evidence) for claim in store.claims.values()), "edge_count": len(store.edges), "raw_evidence_turn_count": sum(len(episode.turn_evidence) for episode in store.episodes.values()), "distinct_persistent_subject_id_count": len({claim.subject_id for claim in store.claims.values() if claim.persistence == "state"})}
        if extra["duplicate_episode_source_id_count"]:
            raise RuntimeError("Event-State episode/source UID invariant failed after memory ingestion")
        return MemoryBuildResult(success=True, method="event_state", action="compile", input_content=text, stored_content="\n".join(item["memory"] for item in added_records), memory_entries=added_records, chunk_count=len(added_records), extra=extra, all_passages=added_records)

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        context_id = kwargs.get("context_id", self._context_id)
        if context_id != self._context_id:
            self.set_context_id(context_id)
        prepared_sessions = self.prepare_memory_sessions(text, **kwargs)
        return self.commit_prepared_memory(prepared_sessions, text=text, context_id=context_id)

    def _record(self, store: EventStateStore, item: Dict[str, Any], episode_evidence: Optional[List[TurnEvidence]] = None) -> Dict[str, Any]:
        planner_channel_retrieval = bool(item.get("planner_request_indices"))
        planner_added_to_final = bool(item.get("planner_added_to_final"))
        if item["type"] == "episode":
            episode = store.episodes[item["id"]]
            return {"id": episode.episode_id, "type": "episode", "memory": render_episode(episode), "source_uid": episode.source_session_id, "source_session_id": episode.source_session_id, "timestamp": episode.recorded_at, "dense_score": item.get("dense_score", 0.0), "fusion_score": item.get("fusion_score", item.get("score", 0.0)), "ppr_score": item.get("ppr_score", 0.0), "final_score": item.get("final_score", item.get("score", 0.0)), "selection_score": item.get("selection_score", item.get("score", 0.0)), "temporal_score": item.get("temporal_score", 0.0), "temporal_match_type": item.get("temporal_match_type"), "selected_rank": item.get("selected_rank"), "episode_evidence_turn_ids": [turn.turn_id for turn in episode_evidence or ()], "episode_archive_turn_ids": [turn.turn_id for turn in episode.turn_evidence], "planner_channel_retrieval": planner_channel_retrieval, "planner_added_to_final": planner_added_to_final, "planner_retrieval": planner_channel_retrieval, **{key: item[key] for key in ("planner_request_indices", "planner_channel_support_count", "planner_fusion_score") if key in item}}
        claim = store.claims[item["id"]]
        evidence = [{"evidence": {"source_uid": ref.source_session_id, "source_session_id": ref.source_session_id, "episode_id": ref.episode_id, "source_turn_ids": ref.source_turn_ids, "support_type": ref.support_type}} for ref in claim.evidence]
        source_uid = claim.evidence[0].source_session_id if claim.evidence else None
        return {"id": claim.claim_id, "type": "state_claim", "memory": render_claim(claim, store.edges, store.claims), "subject": claim.subject, "subject_id": claim.subject_id or claim.subject_key, "status": claim.status, "source_uid": source_uid, "source_session_id": source_uid, "all_provenance_evidence": evidence, "provenance_evidence": evidence[:1], "dense_score": item.get("dense_score", 0.0), "fusion_score": item.get("fusion_score", item.get("score", 0.0)), "ppr_score": item.get("ppr_score", 0.0), "final_score": item.get("final_score", item.get("score", 0.0)), "selection_score": item.get("selection_score", item.get("score", 0.0)), "temporal_score": item.get("temporal_score", 0.0), "temporal_match_type": item.get("temporal_match_type"), "selected_rank": item.get("selected_rank"), "planner_channel_retrieval": planner_channel_retrieval, "planner_added_to_final": planner_added_to_final, "planner_retrieval": planner_channel_retrieval, **{key: item[key] for key in ("planner_request_indices", "planner_channel_support_count", "planner_fusion_score") if key in item}}

    @staticmethod
    def _turn_record(store: EventStateStore, item: Dict[str, Any]) -> Dict[str, Any]:
        """Expose a selected immutable turn without representing it as a claim."""
        episode, turn = store.turn_for_key(item["id"])
        if episode is None or turn is None:
            raise ValueError(f"Selected Event-State turn is missing from the archive: {item['id']}")
        return {
            "id": item["id"], "type": "immutable_turn",
            "source_uid": episode.source_session_id,
            "source_session_id": episode.source_session_id,
            "episode_id": episode.episode_id,
            "source_turn_id": turn.turn_id,
            "episode_evidence_turn_ids": [turn.turn_id],
            "dense_score": item.get("dense_score", 0.0),
            "fusion_score": item.get("fusion_score", item.get("score", 0.0)),
            "final_score": item.get("final_score", item.get("score", 0.0)),
            "selection_score": item.get("selection_score", item.get("score", 0.0)),
            "selected_rank": item.get("selected_rank"),
        }

    def _effective_evidence_selection(
        self,
        store: EventStateStore,
        selection_order: List[Dict[str, Any]],
        query_vectors: List[List[float]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fill the evidence budget after removing redundant direct turns.

        MMR has already produced ``selection_order``. This pass only applies
        immutable provenance identity and never changes retrieval scores.
        """
        target = self._effective_evidence_budget(selection_order)
        selected: List[Dict[str, Any]] = []
        selected_claim_evidence: Dict[str, Any] = {}
        claimed_turns = set()
        turn_vector_cache = self._persisted_turn_vector_cache(store)
        considered_turn_ids: List[str] = []
        deduplicated_turn_ids: List[str] = []
        considered_count = 0

        def remove_redundant_turns() -> None:
            nonlocal selected
            retained = []
            for candidate in selected:
                if candidate["type"] != "turn":
                    retained.append(candidate)
                    continue
                episode, turn = store.turn_for_key(candidate["id"])
                if episode is None or turn is None:
                    continue
                identity = evidence_identity("turn", episode.episode_id, turn.turn_id)
                if identity in claimed_turns:
                    deduplicated_turn_ids.append(candidate["id"])
                    continue
                retained.append(candidate)
            selected = retained

        for candidate in selection_order:
            if len(selected) >= target:
                break
            considered_count += 1
            if candidate["type"] == "turn":
                considered_turn_ids.append(candidate["id"])
                episode, turn = store.turn_for_key(candidate["id"])
                if episode is None or turn is None:
                    continue
                if evidence_identity("turn", episode.episode_id, turn.turn_id) in claimed_turns:
                    deduplicated_turn_ids.append(candidate["id"])
                    continue
            selected.append(candidate)
            if candidate["type"] != "state_claim" or not self.inject_source_evidence:
                continue
            claim = store.claims[candidate["id"]]
            selections = select_claim_evidence(
                claim,
                store.episodes,
                query_vectors[0],
                self._embedder,
                self.max_source_excerpts_per_claim,
                turn_vector_cache=turn_vector_cache,
                query_vectors=query_vectors,
            )
            selected_claim_evidence[claim.claim_id] = selections
            claimed_turns.update(
                evidence_identity("turn", episode_id, turn_id)
                for episode_id, turn_id in selected_claim_evidence_turn_keys(selections)
            )
            remove_redundant_turns()

        for rank, candidate in enumerate(selected, 1):
            candidate["selected_rank"] = rank
        selected_turn_ids = [item["id"] for item in selected if item["type"] == "turn"]
        backfilled = max(0, considered_count - target)
        return selected, {
            "selected_candidate_count": considered_count,
            "deduplicated_turn_count": len(deduplicated_turn_ids),
            "deduplicated_selected_turn_count": len(deduplicated_turn_ids),
            "backfilled_candidate_count": backfilled,
            "backfilled_after_turn_dedup_count": backfilled,
            "selected_turn_ids_before_dedup": considered_turn_ids,
            "selected_turn_ids_after_dedup": selected_turn_ids,
            "effective_selected_turn_count": len(selected_turn_ids),
            "effective_retrieved_record_count": len(selected),
            "selected_claim_evidence": selected_claim_evidence,
            "effective_evidence_budget": target,
        }

    def _effective_evidence_budget(self, selection_order: Sequence[Dict[str, Any]]) -> int:
        """Use a bounded score-shape heuristic only when adaptive mode is selected."""
        fixed = int(self._retrieval_config["evidence_count"])
        if self._retrieval_config.get("evidence_budget_mode", "fixed") != "adaptive":
            return fixed
        minimum = self._retrieval_config.get("adaptive_evidence_min")
        maximum = self._retrieval_config.get("adaptive_evidence_max")
        minimum = fixed if minimum is None else max(1, int(minimum))
        maximum = fixed if maximum is None else max(minimum, int(maximum))
        scores = sorted((max(0.0, float(item.get("final_score", item.get("score", 0.0)))) for item in selection_order), reverse=True)
        if len(scores) < 2 or scores[0] <= 0:
            return minimum
        gap = (scores[0] - scores[1]) / scores[0]
        strong = sum(score >= scores[0] * .75 for score in scores)
        # Flat score distributions and several strong candidates get a larger
        # budget; focused queries remain close to the configured minimum.
        fraction = min(1.0, max(0.0, (1.0 - gap) * min(1.0, strong / 4.0)))
        return min(maximum, max(minimum, round(minimum + (maximum - minimum) * fraction)))

    def _effective_separate_evidence_selection(
        self,
        store: EventStateStore,
        non_turn_selection_order: List[Dict[str, Any]],
        turn_selection_order: List[Dict[str, Any]],
        query_vectors: List[List[float]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply independent structured and direct-turn budgets before rendering."""
        selected_non_turn, non_turn_telemetry = self._effective_evidence_selection(
            store, non_turn_selection_order, query_vectors,
        )
        selected_claim_evidence = non_turn_telemetry["selected_claim_evidence"]
        claimed_turns = {
            evidence_identity("turn", episode_id, turn_id)
            for selections in selected_claim_evidence.values()
            for episode_id, turn_id in selected_claim_evidence_turn_keys(selections)
        }
        direct_budget = int(self._retrieval_config["turn_evidence_count"])
        selected_turns: List[Dict[str, Any]] = []
        considered_turn_ids: List[str] = []
        deduplicated_turn_ids: List[str] = []
        for candidate in turn_selection_order:
            if len(selected_turns) >= direct_budget:
                break
            considered_turn_ids.append(candidate["id"])
            episode, turn = store.turn_for_key(candidate["id"])
            if episode is None or turn is None:
                continue
            if evidence_identity("turn", episode.episode_id, turn.turn_id) in claimed_turns:
                deduplicated_turn_ids.append(candidate["id"])
                continue
            selected_turns.append(candidate)

        selected = selected_non_turn + selected_turns
        for rank, candidate in enumerate(selected, 1):
            candidate["selected_rank"] = rank
        selected_turn_ids = [candidate["id"] for candidate in selected_turns]
        selected_non_turn_count = len(selected_non_turn)
        selected_direct_turn_count = len(selected_turns)
        return selected, {
            **non_turn_telemetry,
            "selected_candidate_count": (
                non_turn_telemetry["selected_candidate_count"] + len(considered_turn_ids)
            ),
            "deduplicated_turn_count": (
                non_turn_telemetry["deduplicated_turn_count"] + len(deduplicated_turn_ids)
            ),
            "deduplicated_selected_turn_count": (
                non_turn_telemetry["deduplicated_selected_turn_count"] + len(deduplicated_turn_ids)
            ),
            "selected_turn_ids_before_dedup": considered_turn_ids,
            "selected_turn_ids_after_dedup": selected_turn_ids,
            "effective_selected_turn_count": selected_direct_turn_count,
            "effective_retrieved_record_count": len(selected),
            "evidence_count": int(self._retrieval_config["evidence_count"]),
            "turn_evidence_count": direct_budget,
            "separate_turn_evidence_budget": True,
            "selected_non_turn_count": selected_non_turn_count,
            "selected_direct_turn_count": selected_direct_turn_count,
            "selected_total_count": len(selected),
        }

    def _select_evidence_pools(
        self,
        retriever: EventStateRetriever,
        non_turn_candidates: List[Dict[str, Any]],
        turn_candidates: List[Dict[str, Any]],
        retrieval_extra: Dict[str, Any],
        query_vectors: List[List[float]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run selectors before provenance-aware rendering, retaining legacy mode."""
        non_turn_order, retrieval_extra = retriever.select_candidates(
            non_turn_candidates, retrieval_extra, count=len(non_turn_candidates),
        )
        if int(self._retrieval_config["turn_evidence_count"]) <= 0:
            selected, effective_extra = self._effective_evidence_selection(
                retriever.store, non_turn_order, query_vectors,
            )
        else:
            turn_order, _turn_extra = retriever.select_candidates(
                turn_candidates, count=len(turn_candidates),
            )
            selected, effective_extra = self._effective_separate_evidence_selection(
                retriever.store, non_turn_order, turn_order, query_vectors,
            )
        retrieval_extra.update(effective_extra)
        return selected, retrieval_extra

    def supports_batch_queries(self) -> bool:
        return self.planner_mode != "iterative"

    def supports_staged_queries(self) -> bool:
        return self.planner_mode != "iterative"

    def prepare_query_compiler(self, question: str, **kwargs) -> Dict[str, Any]:
        """Build the memory-free one-shot compiler request used by both transports."""
        reference_time = kwargs.get("reference_time")
        if reference_time is not None and not isinstance(reference_time, str):
            reference_time = None
        user = f"[QUESTION]\n{question}\n\nREFERENCE_TIME: {reference_time or 'null'}\nMAX_SEARCHES: {self.query_compiler_max_searches}"
        schema = query_compiler_json_schema()
        return {"messages": format_messages(user, QUERY_COMPILER_SYSTEM_PROMPT),
                "temperature": self.planner_temperature, "max_tokens": self.query_compiler_max_tokens,
                "response_format": {"type": "json_object", "response_json_schema": schema},
                "response_json_schema": schema}

    def validate_query_compiler_response(self, question: str, content: str) -> tuple[QueryPlan, Dict[str, Any]]:
        diagnostics = {"enabled": True, "planner_mode": "query_compiler", "compiler_parse_success": False,
                       "compiler_validation_success": False, "compiler_salvage_used": False,
                       "compiler_fallback_used": False, "compiler_warning_codes": [],
                       "compiler_failure_reason": None}
        try:
            value = parse_json(content)
            diagnostics["compiler_parse_success"] = True
            plan, warnings, salvage_used = salvage_query_compiler_output(value, self.query_compiler_max_searches)
            diagnostics["compiler_validation_success"] = True
            diagnostics["compiler_salvage_used"] = salvage_used
            diagnostics["compiler_warning_codes"] = warnings[:16]
        except Exception as exc:
            plan = fallback_query_plan(question)
            diagnostics.update({"compiler_fallback_used": True, "compiler_failure_reason": str(exc)[:300]})
        diagnostics.update({"search_count": len(plan.searches), "normalized_searches": [item.query for item in plan.searches],
                            "state_view": plan.state_view, "temporal_axis": plan.temporal.axis,
                            "temporal_relation": plan.temporal.relation,
                            "temporal_start": plan.temporal.start.isoformat() if plan.temporal.start else None,
                            "temporal_end": plan.temporal.end.isoformat() if plan.temporal.end else None,
                            "temporal_precision": plan.temporal.precision, "anchor_search": plan.temporal.anchor_search,
                            "duplicate_search_count": plan.duplicate_search_count})
        return plan, diagnostics

    def prepare_query_compiler_result(self, question: str, content: str, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        plan, diagnostics = self.validate_query_compiler_response(question, content)
        store = self._store(kwargs.get("context_id"))
        retriever = EventStateRetriever(store, self._embedder, **self._retrieval_config)
        channels = [kwargs.get("raw_question", question)] + [item.query for item in plan.searches]
        vectors = [self._embedder.embed_query(item) for item in channels]
        with get_usage_tracker().scope("event_state.retrieval"):
            ranked, turns, retrieval_extra = retriever.rank_query_plan(channels[0], plan, vectors)
        if plan.temporal.axis != "none":
            if plan.temporal.start and plan.temporal.end:
                retrieval_extra["retrieval_interpretation"] = f"{plan.temporal.axis.title()}-time window: {plan.temporal.start} through {plan.temporal.end}"
            elif plan.temporal.relation == "as_of" and plan.temporal.end:
                retrieval_extra["retrieval_interpretation"] = f"Historical state as of: {plan.temporal.end}"
        selected, retrieval_extra = self._select_evidence_pools(retriever, ranked, turns, retrieval_extra, vectors)
        prepared = self._compile_query_context(question, kwargs.get("answer_system_prompt", system_message), store, selected,
                                               retrieval_extra, vectors,
                                               neutral_answer_contract="answer_system_prompt" in kwargs)
        prepared["extra"].update({"query_compiler": {**diagnostics, "validated_plan": plan.to_dict()},
                                  "planner_mode": "query_compiler"})
        return prepared

    @staticmethod
    def _persisted_turn_vector_cache(store: EventStateStore) -> Dict[Tuple[str, Any], Sequence[float]]:
        """Expose persisted immutable-turn vectors under evidence-selection keys."""
        vectors: Dict[Tuple[str, Any], Sequence[float]] = {}
        for store_key, metadata in store.turn_metadata.items():
            vector = store.turn_embeddings.get(store_key)
            episode_id = metadata.get("episode_id") if isinstance(metadata, dict) else None
            turn_index = metadata.get("source_turn_index") if isinstance(metadata, dict) else None
            episode = store.episodes.get(episode_id)
            if (
                vector is None
                or episode is None
                or not isinstance(turn_index, int)
                or not 0 <= turn_index < len(episode.turn_evidence)
            ):
                continue
            vectors[(episode_id, episode.turn_evidence[turn_index].turn_id)] = vector
        return vectors

    def _compile_query_context(
        self,
        question,
        system_message,
        store,
        selected,
        retrieval_extra,
        query_vectors,
        neutral_answer_contract=False,
    ):
        query_vector = query_vectors[0]
        selected_memory_items = [item for item in selected if item["type"] != "turn"]
        selected_turn_items = [item for item in selected if item["type"] == "turn"]
        selected_claims = [store.claims[item["id"]] for item in selected_memory_items if item["type"] == "state_claim"]
        selected_claim_evidence = dict(retrieval_extra.pop("selected_claim_evidence", {}))
        claimed_turns = set()
        # Immutable-turn vectors are persisted with the store.  Key them by the
        # renderer's source identity so evidence selection avoids re-embedding.
        turn_vector_cache = self._persisted_turn_vector_cache(store)
        if self.inject_source_evidence:
            for claim in selected_claims:
                selections = selected_claim_evidence.get(claim.claim_id)
                if selections is None:
                    selections = select_claim_evidence(
                        claim,
                        store.episodes,
                        query_vector,
                        self._embedder,
                        self.max_source_excerpts_per_claim,
                        turn_vector_cache=turn_vector_cache,
                        query_vectors=query_vectors,
                    )
                    selected_claim_evidence[claim.claim_id] = selections
                claimed_turns.update(selected_claim_evidence_turn_keys(selections))
        selected_episodes = [
            (index, store.episodes[item["id"]])
            for index, item in enumerate(selected_memory_items)
            if item["type"] == "episode"
        ]
        indexed_turns = []
        indexed_turn_keys = set()
        for item in selected_turn_items:
            episode, turn = store.turn_for_key(item["id"])
            if episode is None or turn is None:
                continue
            key = (episode.episode_id, turn.turn_id)
            if key in claimed_turns or key in indexed_turn_keys:
                continue
            indexed_turn_keys.add(key)
            indexed_turns.append((item, episode, turn))
        excluded_episode_turns = claimed_turns | indexed_turn_keys
        episode_evidence_by_id, episode_evidence_candidate_turn_count, episode_evidence_deduplicated_against_claim_count = select_global_episode_evidence(
            selected_episodes, query_vector, self._embedder, self.max_episode_source_excerpts_total,
            excluded_episode_turns, query_vectors=query_vectors, turn_vector_cache=turn_vector_cache,
        )
        if self._retrieval_config.get("episode_excerpt_mode", "global") == "joint":
            # Reserve existing excerpt slots for selected episode support turns,
            # then use the unchanged global allocator for the remaining slots.
            preferred = {}
            selected_by_id = {item["id"]: item for item in selected_memory_items if item["type"] == "episode"}
            remaining = self.max_episode_source_excerpts_total
            for _rank, episode in selected_episodes:
                item = selected_by_id[episode.episode_id]
                support_key = item.get("episode_support_turn_id")
                support_episode, support_turn = store.turn_for_key(support_key) if support_key else (None, None)
                if remaining and support_episode is episode and support_turn is not None and (episode.episode_id, support_turn.turn_id) not in excluded_episode_turns:
                    preferred.setdefault(episode.episode_id, []).append(support_turn)
                    remaining -= 1
            if preferred:
                preferred_keys = {(episode_id, turn.turn_id) for episode_id, turns in preferred.items() for turn in turns}
                global_rest, _, _ = select_global_episode_evidence(
                    selected_episodes, query_vector, self._embedder, remaining,
                    excluded_episode_turns | preferred_keys, query_vectors=query_vectors,
                    turn_vector_cache=turn_vector_cache,
                )
                episode_evidence_by_id = {**preferred}
                for episode_id, turns in global_rest.items():
                    episode_evidence_by_id.setdefault(episode_id, []).extend(turns)
        records = [
            self._record(store, item, episode_evidence_by_id.get(item["id"]))
            for item in selected_memory_items
        ]
        records.extend(self._turn_record(store, item) for item, _episode, _turn in indexed_turns)
        blocks = [
            {"text": record["memory"], "kind": "state" if record["type"] == "state_claim" else "episode", "record_id": record["id"]}
            for record in records if record["type"] != "immutable_turn"
        ]
        if self.inject_source_evidence:
            rendered_turns = set()
            for record in records:
                if record["type"] != "state_claim":
                    continue
                rendered = render_selected_claim_evidence(selected_claim_evidence.get(record["id"], []), rendered_turns)
                record["selected_provenance_evidence"] = [item.metadata() for item in rendered]
                for item in rendered:
                    episode = item.selection.episode
                    excerpt = "\n".join(
                        f"  {turn.speaker} ({turn.timestamp or episode.recorded_at or 'unknown'}): {turn.text}"
                        + (f" [Shared image: {turn.image_caption}]" if turn.image_caption else "")
                        for turn in item.turns
                    )
                    blocks.append({"text": f"[Supporting Evidence {episode.episode_id} / session {episode.source_session_id}]\n{excerpt}", "kind": "source", "record_id": None, "provenance_record_id": record["id"], "provenance_evidence": item.metadata()})
        blocks.extend(
            {"text": render_episode_evidence(store.episodes[record["id"]], episode_evidence_by_id[record["id"]]), "kind": "source", "record_id": record["id"]}
            for record in records if record["type"] == "episode" and episode_evidence_by_id.get(record["id"])
        )
        for item, episode, turn in indexed_turns:
            text = (
                f"[Retrieved Immutable Evidence {episode.episode_id} / session {episode.source_session_id}]\n"
                f"  {turn.speaker} ({turn.timestamp or episode.recorded_at or 'unknown'}): {turn.text}"
                + (f" [Shared image: {turn.image_caption}]" if turn.image_caption else "")
            )
            blocks.append({"text": text, "kind": "source", "record_id": item["id"]})
        if neutral_answer_contract:
            instruction = "Retrieved conversational evidence:"
            answer_base_system = ANSWER_DATA_BOUNDARY_SYSTEM_PROMPT
        else:
            instruction = "The retrieved memory contains conversational evidence. Ground personalized facts in it; use general domain knowledge only for reasoning, and say when personalized evidence is insufficient."
            answer_base_system = ANSWER_SYSTEM_PROMPT
        answer_system = "\n\n".join(item for item in (answer_base_system, (system_message or "").strip()) if item)
        included_blocks, included_tokens = fit_context(blocks, answer_system, instruction, question, self.max_context_tokens, self.max_tokens, self.count_tokens, self.truncate_to_tokens)
        included_counts = Counter(included_blocks)
        included_ids = []
        for block in blocks:
            record_id = block.get("record_id")
            if record_id and included_counts[block["text"]] > 0:
                included_counts[block["text"]] -= 1
                included_ids.append(record_id)
        included_ids = list(dict.fromkeys(included_ids))
        for record in records:
            record["included_in_context"] = record["id"] in included_ids
        included_source_counts = Counter(included_blocks)
        included_provenance_by_claim = {}
        for block in blocks:
            if not block.get("provenance_evidence") or included_source_counts[block["text"]] <= 0:
                continue
            included_source_counts[block["text"]] -= 1
            included_provenance_by_claim.setdefault(block["provenance_record_id"], []).append(block["provenance_evidence"])
        for record in records:
            if record["type"] != "state_claim":
                continue
            record["included_provenance_evidence"] = included_provenance_by_claim.get(record["id"], [])
        context = "\n\n".join(included_blocks)
        interpretation = retrieval_extra.get("retrieval_interpretation")
        interpretation_block = f"\n\n[Retrieval interpretation]\n{interpretation}" if interpretation else ""
        user_content = (f"{instruction}\n\n{context}{interpretation_block}\n\n{question}"
                        if context else f"{instruction}{interpretation_block}\n\n{question}")
        candidate_stages = {
            key: self._candidate_trace(rows, store, len(rows))
            for key in ("channel_semantic_candidates", "merged_semantic_union", "temporally_reranked_union")
            if isinstance((rows := retrieval_extra.get(key)), list)
        }
        # Legacy paths genuinely expose these truncation stages. Compiler
        # paths instead retain their semantic-union stage names above.
        for key, source_key in (
            ("pre_candidate_truncation_fused", "pre_candidate_truncation_candidates"),
            ("post_candidate_count", "post_candidate_truncation_candidates"),
        ):
            if isinstance((rows := retrieval_extra.get(source_key)), list):
                candidate_stages[key] = self._candidate_trace(rows, store, len(rows))
        candidate_stages["final_memory_object_selection"] = self._candidate_trace(selected, store, len(selected))
        base_extra = {
            key: value for key, value in retrieval_extra.items()
            if key not in {
                "pre_candidate_truncation_candidates",
                "post_candidate_truncation_candidates",
                "channel_semantic_candidates",
                "merged_semantic_union",
                "temporally_reranked_union",
            }
        }
        selected_context_tokens = sum(self.count_tokens(block["text"]) for block in blocks)
        source_turn_count = len({
            (episode_id, turn_id)
            for record in records if record.get("included_in_context")
            for episode_id, turn_id in (
                [(record.get("episode_id"), record.get("source_turn_id"))]
                if record["type"] == "immutable_turn" else []
            )
        })
        source_turn_count += len({
            (record["id"], turn_id)
            for record in records if record["type"] == "episode" and record.get("included_in_context")
            for turn_id in record.get("episode_evidence_turn_ids", [])
        })
        source_turn_count += len({
            (item["evidence"].get("episode_id"), turn_id)
            for item in (item for record in records for item in record.get("included_provenance_evidence", []))
            for turn_id in item.get("evidence", {}).get("source_turn_ids", [])
        })
        candidate_stage_for_sessions = candidate_stages.get("temporally_reranked_union", candidate_stages.get("post_candidate_count", []))
        candidate_episode_sessions = {
            source_id for item in candidate_stage_for_sessions
            if item.get("type") == "episode" for source_id in item.get("source_session_ids", [])
        }
        selected_episode_sessions = {
            str(record["source_session_id"]) for record in records
            if record["type"] == "episode" and record.get("source_session_id") is not None
        }
        extra = {**base_extra, "retrieval_record_schema_version": 3, "retrieval_stage_candidates": candidate_stages, "claim_candidate_count": retrieval_extra.get("claim_candidates", 0), "episode_candidate_count": retrieval_extra.get("episode_candidates", 0), "turn_candidate_count": retrieval_extra.get("turn_candidates", 0), "selected_ids": [record["id"] for record in records], "included_ids": included_ids, "selected_context_tokens": selected_context_tokens, "included_context_tokens": included_tokens, "context_truncated": included_tokens < selected_context_tokens, "selected_candidate_count": retrieval_extra.get("selected_candidate_count", len(selected)), "selected_claim_count": sum(record["type"] == "state_claim" for record in records), "selected_episode_count": sum(record["type"] == "episode" for record in records), "selected_memory_object_count": sum(record["type"] != "immutable_turn" for record in records), "selected_turn_count": sum(record["type"] == "immutable_turn" for record in records), "selected_raw_turn_count": sum(record["type"] == "immutable_turn" for record in records), "deduplicated_turn_count": retrieval_extra.get("deduplicated_turn_count", 0), "deduplicated_selected_turn_count": retrieval_extra.get("deduplicated_selected_turn_count", 0), "backfilled_candidate_count": retrieval_extra.get("backfilled_candidate_count", 0), "backfilled_after_turn_dedup_count": retrieval_extra.get("backfilled_after_turn_dedup_count", 0), "effective_selected_turn_count": retrieval_extra.get("effective_selected_turn_count", sum(record["type"] == "immutable_turn" for record in records)), "effective_retrieved_record_count": len(records), "selected_turn_ids_before_dedup": retrieval_extra.get("selected_turn_ids_before_dedup", []), "selected_turn_ids_after_dedup": retrieval_extra.get("selected_turn_ids_after_dedup", []), "selected_episode_evidence_excerpt_count": sum(len(record.get("episode_evidence_turn_ids", [])) for record in records if record["type"] == "episode"), "episode_evidence_candidate_turn_count": episode_evidence_candidate_turn_count, "raw_episode_turn_candidate_count": episode_evidence_candidate_turn_count + len(indexed_turns), "episode_evidence_deduplicated_against_claim_count": episode_evidence_deduplicated_against_claim_count, "selected_episode_count_with_evidence": sum(bool(record.get("episode_evidence_turn_ids")) for record in records if record["type"] == "episode"), "configured_global_raw_excerpt_budget": self.max_episode_source_excerpts_total, "source_evidence_turns_contributed_by_claims": sum(len(item.get("evidence", {}).get("source_turn_ids", [])) for item in (item for record in records for item in record.get("included_provenance_evidence", []))), "answer_visible_distinct_source_turn_count": source_turn_count, "total_answer_visible_distinct_source_turns": source_turn_count, "candidate_episode_source_session_count": len(candidate_episode_sessions), "selected_episode_source_session_count": len(selected_episode_sessions), "included_claim_count": sum(record["type"] == "state_claim" for record in records if record["id"] in included_ids), "included_episode_count": sum(record["type"] == "episode" for record in records if record["id"] in included_ids), "included_provenance_evidence": [item for record in records for item in record.get("included_provenance_evidence", [])]}
        selected_claims = [store.claims[record["id"]] for record in records
                           if record["type"] == "state_claim" and record["id"] in store.claims]
        extra["selected_claim_status_counts"] = dict(sorted(Counter(claim.status for claim in selected_claims).items()))
        extra["selected_claim_persistence_counts"] = dict(sorted(Counter(claim.persistence for claim in selected_claims).items()))
        selected_by_id = {item["id"]: item for item in selected}
        extra["selected_temporal_claim_count"] = sum(
            bool(selected_by_id[claim.claim_id].get("temporal_score", 0.0)) for claim in selected_claims
        )
        extra["selected_temporal_episode_count"] = sum(
            bool(selected_by_id[record["id"]].get("temporal_score", 0.0))
            for record in records if record["type"] == "episode" and record["id"] in selected_by_id
        )
        if sum(extra["selected_claim_status_counts"].values()) != extra["selected_claim_count"]:
            raise AssertionError("selected claim status diagnostics must describe final evidence")
        if sum(extra["selected_claim_persistence_counts"].values()) != extra["selected_claim_count"]:
            raise AssertionError("selected claim persistence diagnostics must describe final evidence")
        if extra["selected_temporal_claim_count"] > extra["selected_claim_count"] or extra["selected_temporal_episode_count"] > extra["selected_episode_count"]:
            raise AssertionError("selected temporal diagnostics must describe final evidence")
        return {"messages": format_messages(user_content, answer_system), "context": context, "retrieved_count": len(records), "retrieved_memories": records, "extra": extra}

    def _initial_query_context(self, question: str, system_message: Optional[str], **kwargs):
        store = self._store(kwargs.get("context_id"))
        retrieval_question = kwargs.get("raw_question", question)
        query_vector = self._embedder.embed_query(retrieval_question)
        retriever = EventStateRetriever(store, self._embedder, **self._retrieval_config)
        with get_usage_tracker().scope("event_state.retrieval"):
            if self.planner_rounds:
                ranked, turn_ranked, retrieval_extra = retriever.rank_candidate_pools(
                    retrieval_question, query_vector=query_vector, parse_temporal_query=False
                )
            else:
                ranked, turn_ranked, retrieval_extra = retriever.rank_candidate_pools(
                    retrieval_question, query_vector=query_vector,
                )
            selected, retrieval_extra = self._select_evidence_pools(
                retriever, ranked, turn_ranked, retrieval_extra, [query_vector],
            )
        neutral_answer_contract = "answer_system_prompt" in kwargs
        answer_system_prompt = kwargs.get("answer_system_prompt", system_message)
        channels = None
        if self.planner_rounds:
            channels = ([ranked], [turn_ranked]) if int(self._retrieval_config["turn_evidence_count"]) > 0 else [ranked]
        return self._compile_query_context(question, answer_system_prompt, store, selected, retrieval_extra, [query_vector], neutral_answer_contract=neutral_answer_contract), retriever, store, channels, query_vector

    def prepare_batch_query(self, question: str, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if self.planner_mode == "iterative":
            raise RuntimeError("Event-State planner queries require realtime execution")
        if self.planner_mode == "query_compiler":
            content = kwargs.pop("query_compiler_content", None)
            if content is None:
                raise RuntimeError("query_compiler batch queries require a completed query-plan stage")
            return self.prepare_query_compiler_result(question, content, system_message, **kwargs)
        prepared, _retriever, _store, _channels, _query_vector = self._initial_query_context(question, system_message, **kwargs)
        prepared.pop("context", None)
        return prepared

    @staticmethod
    def finalize_batch_query(prepared: Dict[str, Any], content: str) -> AgentResponse:
        return AgentResponse(content, retrieved_count=prepared["retrieved_count"], retrieved_memories=prepared["retrieved_memories"], extra=prepared["extra"])

    def query(self, question: str, system_message: Optional[str] = None, **kwargs) -> AgentResponse:
        if self.planner_mode == "query_compiler":
            compiler_question = kwargs.get("raw_question", question)
            compiler = self.prepare_query_compiler(compiler_question, **kwargs)
            with get_usage_tracker().scope("event_state.query_compiler"):
                try:
                    response = self._llm_client.chat(compiler["messages"], temperature=compiler["temperature"],
                                                     max_tokens=compiler["max_tokens"], response_format=compiler["response_format"],
                                                     response_json_schema=compiler["response_json_schema"])
                except TypeError:
                    response = self._llm_client.chat(compiler["messages"], temperature=compiler["temperature"], max_tokens=compiler["max_tokens"])
            prepared = self.prepare_query_compiler_result(question, str(getattr(response, "content", "")), system_message, **kwargs)
            with get_usage_tracker().scope("event_state.answer"):
                answer = self._llm_client.chat(prepared["messages"])
            return self.finalize_batch_query(prepared, answer.content)
        if not self.planner_rounds:
            prepared = self.prepare_batch_query(question, system_message=system_message, **kwargs)
            with get_usage_tracker().scope("event_state.answer"):
                response = self._llm_client.chat(prepared["messages"])
            return self.finalize_batch_query(prepared, response.content)
        prepared, retriever, store, channels, query_vector = self._initial_query_context(question, system_message, **kwargs)
        answer_system_prompt = kwargs.get("answer_system_prompt", system_message)
        neutral_answer_contract = "answer_system_prompt" in kwargs
        query_vectors = [query_vector]
        base_selected_ids = [item["id"] for item in prepared["retrieved_memories"]]
        telemetry = {"planner_rounds_configured": self.planner_rounds, "planner_rounds_used": 0, "planner_decision_call_count": 0, "planner_retrieval_round_count": 0, "planner_request_count": 0, "planner_valid_request_count": 0, "planner_invalid_request_count": 0, "planner_duplicate_request_count": 0, "planner_parse_failure_count": 0, "planner_early_answer": False, "planner_forced_final_answer": False, "planner_requests": [], "planner_invalid_output_previews": [], "planner_invalid_output_sha256": [], "base_selected_ids": base_selected_ids}
        trace_limit = int(self._retrieval_config.get("candidate_count", 40))
        separate_turn_budget = int(self._retrieval_config["turn_evidence_count"]) > 0
        if separate_turn_budget:
            non_turn_channels, turn_channels = channels
            base_ranked = non_turn_channels[0] + turn_channels[0]
        else:
            base_ranked = channels[0] if channels else []
        telemetry.update({"planner_json_parse_failure_count": 0, "planner_schema_validation_failure_count": 0, "planner_failure_diagnostics": [], "candidate_trace": {"base_ranked": self._candidate_trace(base_ranked, store, trace_limit), "planner_channels": [], "merged_preselect": []}})
        attempted = set()
        for round_index in range(self.planner_rounds):
            telemetry["planner_rounds_used"] = round_index + 1
            previous = json.dumps(telemetry["planner_requests"], ensure_ascii=True)
            planner_user = f"[ORIGINAL QUESTION]\n{question}\n\n[CURRENT PERSONALIZED MEMORY EVIDENCE]\n{prepared.get('context', '') or 'none'}\n\n[PREVIOUS RETRIEVAL REQUESTS]\n{previous if telemetry['planner_requests'] else 'none'}\n\n[PLANNER BUDGET]\nretrieval rounds available including this decision: {self.planner_rounds - round_index}\nadditional retrieval rounds after this decision: {self.planner_rounds - round_index - 1}\nmaximum requests this round: {self.planner_max_requests}"
            planner_system = "\n\n".join(item for item in (QUERY_PLANNER_SYSTEM_PROMPT, (system_message or "").strip()) if item)
            try:
                with get_usage_tracker().scope("event_state.plan_or_answer"):
                    response = self._llm_client.chat(format_messages(planner_user, planner_system), temperature=self.planner_temperature, max_tokens=self.planner_max_tokens, response_format={"type": "json_object"})
            except TypeError as exc:
                if "response_format" not in str(exc).lower():
                    raise
                with get_usage_tracker().scope("event_state.plan_or_answer"):
                    response = self._llm_client.chat(format_messages(planner_user, planner_system), temperature=self.planner_temperature, max_tokens=self.planner_max_tokens)
            telemetry["planner_decision_call_count"] += 1
            raw_output = getattr(response, "content", "")
            raw_text = raw_output if isinstance(raw_output, str) else str(raw_output or "")
            output_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            try:
                parsed_output = json.loads(raw_output)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                telemetry["planner_json_parse_failure_count"] += 1
                telemetry["planner_parse_failure_count"] += 1
                telemetry["planner_invalid_output_previews"].append(raw_text[:500])
                telemetry["planner_invalid_output_sha256"].append(output_hash)
                telemetry["planner_failure_diagnostics"].append({"failure_stage": "json_parse", "failure_reason": str(exc)[:300], "invalid_output_sha256": output_hash, "invalid_output_preview": raw_text[:500]})
                telemetry["planner_forced_final_answer"] = True
                break
            try:
                decision = validate_planner_output(parsed_output, self.planner_max_requests)
            except ValueError as exc:
                telemetry["planner_schema_validation_failure_count"] += 1
                telemetry["planner_parse_failure_count"] += 1
                telemetry["planner_invalid_output_previews"].append(raw_text[:500])
                telemetry["planner_invalid_output_sha256"].append(output_hash)
                telemetry["planner_failure_diagnostics"].append({"failure_stage": "schema_validation", "failure_reason": str(exc)[:300], "invalid_output_sha256": output_hash, "invalid_output_preview": raw_text[:500]})
                telemetry["planner_forced_final_answer"] = True
                break
            telemetry["planner_invalid_request_count"] += decision.invalid_request_count
            telemetry["planner_duplicate_request_count"] += decision.duplicate_request_count
            if decision.action == "answer":
                telemetry["planner_early_answer"] = True
                final_ids = [item["id"] for item in prepared["retrieved_memories"]]
                telemetry.update({"final_selected_ids": final_ids, "planner_added_ids": [item for item in final_ids if item not in base_selected_ids], "planner_removed_ids": [item for item in base_selected_ids if item not in final_ids]})
                return AgentResponse(decision.answer or "", retrieved_count=prepared["retrieved_count"], retrieved_memories=prepared["retrieved_memories"], extra={**prepared["extra"], **telemetry, "planner": self._planner_artifact(telemetry)})
            new_requests = [request for request in decision.requests if request.key() not in attempted]
            telemetry["planner_duplicate_request_count"] += len(decision.requests) - len(new_requests)
            telemetry["planner_request_count"] += len(decision.requests)
            telemetry["planner_valid_request_count"] += len(new_requests)
            if not new_requests:
                telemetry["planner_forced_final_answer"] = True
                break
            telemetry["planner_retrieval_round_count"] += 1
            for request in new_requests:
                attempted.add(request.key())
                telemetry["planner_requests"].append(request.to_dict())
                vector = self._embedder.embed_query(request.query)
                query_vectors.append(vector)
                ranked_request, turn_ranked_request, _request_extra = retriever.rank_candidate_pools(request.query, query_vector=vector, temporal_constraint=request.temporal_constraint, parse_temporal_query=False, retrieve_claims_override=request.sources in {"claims", "both"}, retrieve_episodes_override=request.sources in {"episodes", "both"}, retrieve_turns_override=request.sources in {"episodes", "both"}, state_view=request.state_view)
                if separate_turn_budget:
                    non_turn_channels.append(ranked_request)
                    turn_channels.append(turn_ranked_request)
                else:
                    channels.append(ranked_request)
                telemetry["candidate_trace"]["planner_channels"].append({"request_index": len(telemetry["candidate_trace"]["planner_channels"]), "request": request.to_dict(), "ranked": self._candidate_trace(ranked_request + turn_ranked_request, store, trace_limit)})
            if separate_turn_budget:
                merged = retriever.merge_rank_channels(non_turn_channels)
                merged_turns = retriever.merge_rank_channels(turn_channels)
                retrieval_extra = {
                    "pre_candidate_truncation_candidates": merged + merged_turns,
                    "post_candidate_truncation_candidates": merged + merged_turns,
                }
                telemetry["candidate_trace"]["merged_preselect"] = self._candidate_trace(merged + merged_turns, store, trace_limit)
            else:
                merged = retriever.merge_rank_channels(channels)
                merged_turns = []
                retrieval_extra = {}
                telemetry["candidate_trace"]["merged_preselect"] = self._candidate_trace(merged, store, trace_limit)
            selected, retrieval_extra = self._select_evidence_pools(
                retriever, merged, merged_turns, retrieval_extra, query_vectors,
            )
            base_selected_id_set = set(base_selected_ids)
            for item in selected:
                item["planner_added_to_final"] = item["id"] not in base_selected_id_set
            prepared = self._compile_query_context(question, answer_system_prompt, store, selected, {**retrieval_extra, "planner_expanded": True}, query_vectors, neutral_answer_contract=neutral_answer_contract)
        telemetry["planner_forced_final_answer"] = True
        with get_usage_tracker().scope("event_state.final_answer"):
            final_response = self._llm_client.chat(prepared["messages"])
        final_ids = [item["id"] for item in prepared["retrieved_memories"]]
        telemetry.update({"final_selected_ids": final_ids, "planner_added_ids": [item for item in final_ids if item not in base_selected_ids], "planner_removed_ids": [item for item in base_selected_ids if item not in final_ids]})
        return AgentResponse(final_response.content, retrieved_count=prepared["retrieved_count"], retrieved_memories=prepared["retrieved_memories"], extra={**prepared["extra"], **telemetry, "planner": self._planner_artifact(telemetry)})

    @staticmethod
    def _planner_artifact(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Return the bounded planner diagnostics persisted in query artifacts."""
        mapping = {
            "rounds_configured": "planner_rounds_configured",
            "rounds_used": "planner_rounds_used",
            "decision_call_count": "planner_decision_call_count",
            "retrieval_round_count": "planner_retrieval_round_count",
            "request_count": "planner_request_count",
            "valid_request_count": "planner_valid_request_count",
            "invalid_request_count": "planner_invalid_request_count",
            "duplicate_request_count": "planner_duplicate_request_count",
            "parse_failure_count": "planner_parse_failure_count",
            "json_parse_failure_count": "planner_json_parse_failure_count",
            "schema_validation_failure_count": "planner_schema_validation_failure_count",
            "failure_diagnostics": "planner_failure_diagnostics",
            "early_answer": "planner_early_answer",
            "forced_final_answer": "planner_forced_final_answer",
            "requests": "planner_requests",
            "base_selected_ids": "base_selected_ids",
            "final_selected_ids": "final_selected_ids",
            "added_ids": "planner_added_ids",
            "removed_ids": "planner_removed_ids",
            "candidate_trace": "candidate_trace",
        }
        return {name: telemetry.get(source, [] if name in {"requests", "base_selected_ids", "final_selected_ids", "added_ids", "removed_ids", "failure_diagnostics", "candidate_trace"} else 0) for name, source in mapping.items()}

    @staticmethod
    def _candidate_trace(candidates: List[Dict[str, Any]], store: EventStateStore, limit: int) -> List[Dict[str, Any]]:
        rows = []
        for rank, item in enumerate(list(candidates)[:max(0, limit)], 1):
            identifier = item.get("id")
            source_ids = set()
            if item.get("type") == "episode" and identifier in store.episodes:
                source_ids.add(store.episodes[identifier].source_session_id)
            elif item.get("type") == "state_claim" and identifier in store.claims:
                source_ids.update(ref.source_session_id for ref in store.claims[identifier].evidence)
            elif item.get("type") == "turn":
                source_ids.add(store.turn_metadata.get(identifier, {}).get("source_session_id"))
            rows.append({"id": identifier, "type": item.get("type"), "rank": rank, "source_session_ids": sorted(str(value) for value in source_ids if value is not None), "final_score": float(item.get("final_score", item.get("score", 0.0)) or 0.0), **({"planner_fusion_score": float(item["planner_fusion_score"])} if "planner_fusion_score" in item else {}), "temporal_score": float(item.get("temporal_score", 0.0) or 0.0), "temporal_match_type": item.get("temporal_match_type")})
        return rows

    def supports_memory_snapshots(self) -> bool:
        return True

    def export_memory_state(self, context_id=None):
        return self._store(context_id).export()

    def export_memory_binary_artifacts(self, context_id=None):
        """Return derived dense indexes for snapshot-sidecar publication."""
        return self._store(context_id).export_embedding_artifacts()

    def import_memory_state(self, state, context_id=None):
        stored_context = state.get("context_id")
        key = context_id if context_id is not None else (self._context_id if self._context_id is not None else stored_context)
        restored = EventStateStore.from_export(state)
        missing_turn_keys = (
            [
                turn_key for turn_key in restored.turn_metadata
                if turn_key not in restored.turn_embeddings
            ]
            if state.get("schema_version") == 4
            else []
        )
        if missing_turn_keys:
            # Schema v4 retains immutable evidence but predates the turn-vector
            # index. Schema v6 must always load its persisted sidecars instead.
            texts = []
            valid_keys = []
            for key in missing_turn_keys:
                _episode, turn = restored.turn_for_key(key)
                if turn is not None:
                    valid_keys.append(key)
                    texts.append(episode_turn_embedding_text(turn))
            if texts:
                with get_usage_tracker().scope("event_state.embedding"):
                    vectors = self._embedder.embed_documents(texts)
                restored.turn_embeddings.update(
                    {key: list(vector) for key, vector in zip(valid_keys, vectors)}
                )
        restored.context_id = key
        self._stores[key] = restored
        self._context_id = key
        self._memory_chunks = [episode.summary for episode in restored.episodes.values()]
        self._is_initialized = bool(restored.episodes or restored.claims)

    def get_info(self):
        info = super().get_info()
        store = self._store()
        info.update(store.claim_counts())
        info.update({"episode_count": len(store.episodes), "claim_count": len(store.claims), "edge_count": len(store.edges)})
        return info
