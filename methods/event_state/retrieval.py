"""Dual dense retrieval, bounded optional PPR, and deterministic MMR selection."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
import re
import threading
import time
from datetime import date
from typing import Any, Dict, List, Sequence, Tuple

from .embeddings import cosine
from .store import EventStateStore
from .planner import QueryPlan
from .temporal import (
    TemporalQueryConstraint,
    claim_temporal_match,
    claim_visible_as_of,
    episode_temporal_match,
    parse_temporal_query as parse_temporal_query_fn,
    parse_stored_date,
)
from utils.llm_client import get_usage_tracker

# Keep the historical module-level hook patchable for compatibility tests.
parse_temporal_query = parse_temporal_query_fn
_LOCAL_RERANKERS: Dict[str, Any] = {}
_LOCAL_RERANKER_LOCK = threading.Lock()


def dense_rank(query: Sequence[float], vectors: Dict[str, Sequence[float]], top_k: int) -> List[Tuple[str, float]]:
    rows = [(key, cosine(query, value)) for key, value in vectors.items()]
    return sorted(rows, key=lambda item: (-item[1], item[0]))[:max(0, int(top_k))]


def normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def calibrate_scores(values: Sequence[float], mode: str) -> List[float]:
    """Calibrate one query-local representation list without NaNs."""
    if not values:
        return []
    if mode == "none":
        return [float(value) for value in values]
    if mode == "minmax":
        return normalize_scores(values)
    median = sorted(values)[len(values) // 2]
    deviations = sorted(abs(value - median) for value in values)
    scale = deviations[len(deviations) // 2] * 1.4826
    if scale <= 1e-12:
        return [0.5] * len(values)
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, (value - median) / scale)))) for value in values]


def available_memory_mb() -> float | None:
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().available) / (1024 * 1024)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


class EventStateRetriever:
    def __init__(self, store: EventStateStore, embedder: Any, **config: Any) -> None:
        self.store, self.embedder, self.config = store, embedder, config

    def rank_candidates(
        self,
        question: str,
        query_vector: Sequence[float] | None = None,
        *,
        temporal_constraint: TemporalQueryConstraint | None = None,
        parse_temporal_query: bool = True,
        retrieve_claims_override: bool | None = None,
        retrieve_episodes_override: bool | None = None,
        retrieve_turns_override: bool | None = None,
        state_view: str = "current",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_vector = list(query_vector) if query_vector is not None else self.embedder.embed_query(question)
        temporal = temporal_constraint
        if temporal is None and parse_temporal_query and self.config.get("temporal_retrieval_enabled", True):
            temporal = globals()["parse_temporal_query"](question)
        retrieve_claims = self.config.get("retrieve_claims", True) if retrieve_claims_override is None else retrieve_claims_override
        retrieve_episodes = self.config.get("retrieve_episodes", True) if retrieve_episodes_override is None else retrieve_episodes_override
        retrieve_turns = self.config.get("retrieve_turns", True) if retrieve_turns_override is None else retrieve_turns_override
        claim_vectors, hidden_prior_state_count = self._visible_claim_vectors(temporal, state_view)
        claim_rank = dense_rank(query_vector, claim_vectors, self.config.get("claim_top_k", 30)) if retrieve_claims else []
        episode_rank = self._episode_rank(query_vector) if retrieve_episodes else []
        turn_rank = dense_rank(query_vector, self.store.turn_embeddings, self.config.get("turn_top_k", 8)) if retrieve_turns else []
        lexical_turn_rank = self._lexical_turn_rank(question) if retrieve_turns and self.config.get("turn_lexical_retrieval_enabled", False) else []
        temporal_claim_rank, temporal_episode_rank = [], []
        if temporal is not None and self.config.get("temporal_query_mode", "legacy") == "legacy":
            temporal_claim_rank = self._temporal_claim_rank(query_vector, temporal, state_view) if retrieve_claims else []
            temporal_episode_rank = self._temporal_episode_rank(query_vector, temporal) if retrieve_episodes else []
        candidates = self._fuse(
            claim_rank, episode_rank, turn_rank, lexical_turn_rank,
            temporal_claim_rank, temporal_episode_rank,
        )
        for item in candidates:
            if item["type"] == "episode" and hasattr(self, "_episode_support") and item["id"] in self._episode_support:
                turn_key, support_score, summary_score = self._episode_support[item["id"]]
                item["episode_support_turn_id"] = turn_key
                item["episode_support_turn_score"] = support_score
                item["episode_summary_score"] = summary_score
        if temporal is not None and self.config.get("temporal_query_mode", "legacy") == "dual_axis_semantic":
            candidates = self._dual_axis_temporal(candidates, temporal)
        candidates = self._rerank_local(question, candidates)
        if self.config.get("ppr_enabled", False):
            graph_candidates = [item for item in candidates if item["type"] != "turn"]
            turn_candidates = [item for item in candidates if item["type"] == "turn"]
            candidates = self._ppr(graph_candidates) + turn_candidates
        candidates = [item for item in candidates if item["type"] != "state_claim" or self._claim_is_directly_visible(item["id"], temporal, state_view)]
        values = normalize_scores([item.get("score", 0.0) for item in candidates])
        for item, final_score in zip(candidates, values):
            item["final_score"] = final_score
        candidates.sort(key=lambda item: (-item["final_score"], item["id"]))
        pre_candidate_truncation = [dict(item) for item in candidates]
        candidate_count = int(self.config.get("candidate_count", 40))
        candidates = candidates[:candidate_count]
        claim_candidate_statuses = Counter(self.store.claims[identifier].status for identifier, _ in claim_rank)
        claim_candidate_persistence = Counter(self.store.claims[identifier].persistence for identifier, _ in claim_rank)
        temporal_historical = sum(
            1 for identifier, _score, _match in temporal_claim_rank
            if self.store.claims[identifier].status in {"superseded", "refined"}
        )
        future_filtered = 0
        if temporal is not None and temporal.kind == "as_of":
            target = temporal.target_date
            future_filtered = sum(
                1
                for claim in self.store.claims.values()
                if claim.persistence == "state"
                and parse_stored_date(claim.recorded_at) is not None
                and parse_stored_date(claim.recorded_at) > target
            )
        return candidates, {
            "claim_candidates": len(claim_rank),
            "episode_candidates": len(episode_rank),
            "turn_candidates": len(turn_rank),
            "lexical_turn_candidates": len(lexical_turn_rank),
            "pre_candidate_truncation_fused_candidate_count": len(pre_candidate_truncation),
            "candidate_count": len(candidates),
            "post_candidate_truncation_candidate_count": len(candidates),
            "pre_candidate_truncation_candidates": pre_candidate_truncation,
            "post_candidate_truncation_candidates": [dict(item) for item in candidates],
            "ppr_enabled": bool(self.config.get("ppr_enabled", False)),
            "selector_mode": self.config.get("selector_mode", "state_mmr"),
            "selected_ids": [],
            "claim_candidate_status_counts": dict(sorted(claim_candidate_statuses.items())),
            "candidate_claim_status_counts": dict(sorted(claim_candidate_statuses.items())),
            "candidate_claim_persistence_counts": dict(sorted(claim_candidate_persistence.items())),
            "selected_claim_status_counts": {},
            "selected_claim_persistence_counts": {},
            "hidden_prior_state_candidate_count": hidden_prior_state_count,
            "temporal_constraint_detected": temporal is not None,
            "temporal_constraint_kind": temporal.kind if temporal else None,
            "temporal_target_date": temporal.target_date.isoformat() if temporal and temporal.target_date else None,
            "temporal_start_date": temporal.start_date.isoformat() if temporal and temporal.start_date else None,
            "temporal_end_date": temporal.end_date.isoformat() if temporal and temporal.end_date else None,
            "temporal_claim_candidate_count": len(temporal_claim_rank),
            "temporal_episode_candidate_count": len(temporal_episode_rank),
            "temporal_historical_state_candidate_count": temporal_historical,
            "temporal_future_state_filtered_count": future_filtered,
            "selected_temporal_claim_count": 0,
            "selected_temporal_episode_count": 0,
            "fusion_mode": self.config.get("fusion_mode", "rrf"),
            "score_calibration_mode": self.config.get("score_calibration_mode", "minmax"),
            "turn_lexical_mode": self.config.get("turn_lexical_mode", "overlap"),
            "turn_sparse_unit": self.config.get("turn_sparse_unit", "turn"),
            "bm25_cache_built": bool(getattr(self.store, "_turn_bm25_caches", {})),
            "episode_relevance_mode": self.config.get("episode_relevance_mode", "summary"),
            "temporal_query_mode": self.config.get("temporal_query_mode", "legacy"),
            "local_reranker_mode": self.config.get("local_reranker_mode", "off"),
            "local_reranker_disabled_reason": getattr(self, "_reranker_disabled_reason", None),
            "bm25_index_seconds": getattr(self, "_last_bm25_build_seconds", 0.0),
        }

    def rank_candidate_pools(
        self,
        question: str,
        query_vector: Sequence[float] | None = None,
        *,
        temporal_constraint: TemporalQueryConstraint | None = None,
        parse_temporal_query: bool = True,
        retrieve_claims_override: bool | None = None,
        retrieve_episodes_override: bool | None = None,
        retrieve_turns_override: bool | None = None,
        state_view: str = "current",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Rank structured and direct-turn pools independently when configured.

        The zero-budget path intentionally delegates to ``rank_candidates``
        unchanged.  A positive direct-turn budget gives claims/episodes and
        turns independent fusion normalization and candidate truncation before
        their respective selectors run.
        """
        if int(self.config.get("turn_evidence_count", 0)) <= 0:
            candidates, diagnostics = self.rank_candidates(
                question,
                query_vector=query_vector,
                temporal_constraint=temporal_constraint,
                parse_temporal_query=parse_temporal_query,
                retrieve_claims_override=retrieve_claims_override,
                retrieve_episodes_override=retrieve_episodes_override,
                retrieve_turns_override=retrieve_turns_override,
                state_view=state_view,
            )
            return candidates, [], diagnostics

        structured, structured_diagnostics = self.rank_candidates(
            question,
            query_vector=query_vector,
            temporal_constraint=temporal_constraint,
            parse_temporal_query=parse_temporal_query,
            retrieve_claims_override=retrieve_claims_override,
            retrieve_episodes_override=retrieve_episodes_override,
            retrieve_turns_override=False,
            state_view=state_view,
        )
        turns, turn_diagnostics = self.rank_candidates(
            question,
            query_vector=query_vector,
            temporal_constraint=temporal_constraint,
            parse_temporal_query=parse_temporal_query,
            retrieve_claims_override=False,
            retrieve_episodes_override=False,
            retrieve_turns_override=retrieve_turns_override,
            state_view=state_view,
        )
        diagnostics = dict(structured_diagnostics)
        diagnostics.update({
            "turn_candidates": turn_diagnostics["turn_candidates"],
            "lexical_turn_candidates": turn_diagnostics["lexical_turn_candidates"],
            "pre_candidate_truncation_fused_candidate_count": (
                structured_diagnostics["pre_candidate_truncation_fused_candidate_count"]
                + turn_diagnostics["pre_candidate_truncation_fused_candidate_count"]
            ),
            "candidate_count": len(structured) + len(turns),
            "post_candidate_truncation_candidate_count": len(structured) + len(turns),
            "pre_candidate_truncation_candidates": (
                structured_diagnostics["pre_candidate_truncation_candidates"]
                + turn_diagnostics["pre_candidate_truncation_candidates"]
            ),
            "post_candidate_truncation_candidates": (
                structured_diagnostics["post_candidate_truncation_candidates"]
                + turn_diagnostics["post_candidate_truncation_candidates"]
            ),
            "non_turn_candidate_count": len(structured),
            "direct_turn_candidate_count": len(turns),
        })
        return structured, turns, diagnostics

    def select_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        extra: Dict[str, Any] | None = None,
        *,
        count: int | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        source_info = self._query_support_info(candidates, query_vector)
        selected = self._select(
            candidates,
            int(self.config.get("evidence_count", 8)) if count is None else max(0, int(count)),
            source_info=source_info,
        )
        diagnostics = dict(extra or {})
        selected_claims = [self.store.claims[item["id"]] for item in selected if item["type"] == "state_claim"]
        diagnostics.update({
            "selected_ids": [item["id"] for item in selected],
            "selected_claim_status_counts": dict(sorted(Counter(claim.status for claim in selected_claims).items())),
            "selected_claim_persistence_counts": dict(sorted(Counter(claim.persistence for claim in selected_claims).items())),
            "selected_temporal_claim_count": sum(1 for item in selected if item["type"] == "state_claim" and item.get("temporal_score", 0.0)),
            "selected_temporal_episode_count": sum(1 for item in selected if item["type"] == "episode" and item.get("temporal_score", 0.0)),
            "selected_memory_object_count": sum(1 for item in selected if item["type"] != "turn"),
            "selected_turn_count": sum(1 for item in selected if item["type"] == "turn"),
            "selected_claim_count": sum(1 for item in selected if item["type"] == "state_claim"),
            "selected_episode_count": sum(1 for item in selected if item["type"] == "episode"),
        })
        diagnostics.update(self.source_coherence_diagnostics(candidates, selected, source_info))
        return selected, diagnostics

    def retrieve(self, question: str, query_vector: Sequence[float] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidates, extra = self.rank_candidates(question, query_vector=query_vector)
        vector = list(query_vector) if query_vector is not None else self.embedder.embed_query(question)
        return self.select_candidates(candidates, extra, query_vector=vector)

    def rank_query_plan(
        self, question: str, plan: QueryPlan, query_vectors: Sequence[Sequence[float]] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Run compiler channels locally, retaining the raw query as channel zero.

        Unlike legacy temporal retrieval this never adds a broad date-only
        candidate list. Temporal evidence only changes the score of semantic
        candidates that survived coverage-aware fusion.
        """
        channels = [question] + [search.query for search in plan.searches]
        vectors = list(query_vectors or [self.embedder.embed_query(item) for item in channels])
        no_temporal_constraint = plan.temporal.axis == "none" or plan.temporal.relation == "none"
        if not plan.searches and no_temporal_constraint and plan.state_view == "current":
            # Preserve the normal single-channel geometry exactly. In particular,
            # do not normalize a second time through planner-level fusion.
            structured, turns, details = self.rank_candidate_pools(
                question, query_vector=vectors[0], parse_temporal_query=False, state_view="current",
            )
            details.update({
                "query_compiler_channels": [{"channel_index": 0, "query": question, "role": "original",
                                             "structured_candidates": len(structured), "turn_candidates": len(turns)}],
                "channel_semantic_candidates": list(structured) + list(turns),
                "merged_semantic_union": list(structured) + list(turns),
                "temporally_reranked_union": list(structured) + list(turns),
                "original_query_channel": question, "coverage_merge_mode": "single_channel",
                "anchor_resolution_status": "none", "anchor_candidate_count": 0,
                "anchor_temporal_candidate_count": 0, "resolved_anchor_spans": [],
                "resolved_anchor_source_types": [], "resolved_anchor_source_ids": [],
                "temporalized_candidate_count": sum(
                    bool(item.get("temporal_score")) for item in structured + turns
                ),
            })
            return structured, turns, details
        structured_channels, turn_channels, channel_details = [], [], []
        as_of = None
        if plan.temporal.axis == "knowledge" and plan.temporal.end:
            as_of = TemporalQueryConstraint("as_of", target_date=plan.temporal.end, intent="knowledge")
        for index, (channel, vector) in enumerate(zip(channels, vectors)):
            structured, turns, details = self.rank_candidate_pools(
                channel, query_vector=vector, temporal_constraint=as_of,
                parse_temporal_query=False, state_view=plan.state_view,
            )
            structured_channels.append(structured)
            if int(self.config.get("turn_evidence_count", 0)) > 0:
                turn_channels.append(turns)
            channel_details.append({"channel_index": index, "query": channel,
                                    "role": "original" if index == 0 else plan.searches[index - 1].role,
                                    "structured_candidates": len(structured), "turn_candidates": len(turns)})
        structured = self.merge_rank_channels(structured_channels)
        turns = self.merge_rank_channels(turn_channels) if turn_channels else []
        channel_semantic = [item for channel in structured_channels for item in channel]
        channel_semantic.extend(item for channel in turn_channels for item in channel)
        merged_semantic = list(structured) + list(turns)
        anchor_spans, anchor_diagnostics = self._resolve_anchor_spans(plan, structured_channels, turn_channels)
        structured = self._rerank_query_plan_temporal(structured, plan, anchor_spans)
        turns = self._rerank_query_plan_temporal(turns, plan, anchor_spans)
        return structured, turns, {
            "query_compiler_channels": channel_details,
            "original_query_channel": question,
            "coverage_merge_mode": self.config.get("planner_merge_mode", "coverage_interleave"),
            "anchor_temporal_resolution_success": anchor_diagnostics["anchor_resolution_status"] == "resolved",
            "resolved_anchor_spans_count": len(anchor_spans),
            "channel_semantic_candidates": channel_semantic,
            "merged_semantic_union": merged_semantic,
            "temporally_reranked_union": list(structured) + list(turns),
            **anchor_diagnostics,
            "temporalized_candidate_count": sum(bool(item.get("temporal_score")) for item in structured + turns),
        }

    @staticmethod
    def _span_dates(span: Dict[str, Any]) -> tuple[date | None, date | None]:
        return parse_stored_date(span.get("start")), parse_stored_date(span.get("end"))

    def _candidate_spans(self, item: Dict[str, Any], axis: str) -> List[Dict[str, str]]:
        if axis == "record":
            if item["type"] == "turn":
                episode, _turn = self.store.turn_for_key(item["id"])
                recorded = episode.recorded_at if episode else None
            elif item["type"] == "episode":
                recorded = self.store.episodes[item["id"]].recorded_at
            else:
                recorded = self.store.claims[item["id"]].recorded_at
            return [{"start": recorded, "end": recorded, "precision": "exact"}] if parse_stored_date(recorded) else []
        if item["type"] == "turn":
            episode, turn = self.store.turn_for_key(item["id"])
            return list(self.store.turn_temporal_spans.get((episode.episode_id, turn.turn_id), [])) if episode and turn else []
        if item["type"] == "episode":
            return list(self.store.episode_temporal_spans.get(item["id"], []))
        claim = self.store.claims[item["id"]]
        return ([{"start": claim.event_time_start, "end": claim.event_time_end,
                  "precision": claim.event_time_precision}]
                if claim.event_time_start and claim.event_time_end else [])

    def _anchor_candidate_spans(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return event time only when it is attributable to anchor evidence."""
        if item["type"] == "state_claim":
            claim = self.store.claims.get(item["id"])
            if claim and claim.event_time_start and claim.event_time_end:
                return [{"start": claim.event_time_start, "end": claim.event_time_end,
                         "precision": claim.event_time_precision, "source_type": "claim",
                         "source_id": claim.claim_id}]
            return []
        if item["type"] != "turn":
            # An episode is a container, not an assertion that all of its
            # claims occurred at every date mentioned inside it.
            return []
        episode, turn = self.store.turn_for_key(item["id"])
        if not episode or not turn:
            return []
        spans = []
        for claim in self.store.claims.values():
            if not claim.event_time_start or not claim.event_time_end:
                continue
            if any(ref.episode_id == episode.episode_id and turn.turn_id in ref.source_turn_ids for ref in claim.evidence):
                spans.append({"start": claim.event_time_start, "end": claim.event_time_end,
                              "precision": claim.event_time_precision, "source_type": "turn_claim",
                              "source_id": f"{episode.episode_id}:{turn.turn_id}:{claim.claim_id}"})
        return spans

    def _resolve_anchor_spans(self, plan: QueryPlan, structured_channels: Sequence[Sequence[Dict[str, Any]]], turn_channels: Sequence[Sequence[Dict[str, Any]]]) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        anchor = plan.temporal.anchor_search
        if anchor is None:
            return [], {"anchor_resolution_status": "none", "anchor_candidate_count": 0,
                        "anchor_temporal_candidate_count": 0, "resolved_anchor_spans": [],
                        "resolved_anchor_source_types": [], "resolved_anchor_source_ids": []}
        # +1 accounts for the always-on original-query channel.
        channel_index = anchor + 1
        if channel_index >= len(structured_channels):
            return [], {"anchor_resolution_status": "missing_metadata", "anchor_candidate_count": 0,
                        "anchor_temporal_candidate_count": 0, "resolved_anchor_spans": [],
                        "resolved_anchor_source_types": [], "resolved_anchor_source_ids": []}
        rows = list(structured_channels[channel_index]) + (list(turn_channels[channel_index]) if channel_index < len(turn_channels) else [])
        temporal_rows = []
        for item in rows:
            for span in self._anchor_candidate_spans(item):
                start, end = self._span_dates(span)
                if start and end:
                    temporal_rows.append((span, max(0.0, float(item.get("final_score", item.get("score", 0.0))))))
        if not temporal_rows:
            return [], {"anchor_resolution_status": "missing_metadata", "anchor_candidate_count": len(rows),
                        "anchor_temporal_candidate_count": 0, "resolved_anchor_spans": [],
                        "resolved_anchor_source_types": [], "resolved_anchor_source_ids": []}
        # Greedily group overlapping event intervals. This intentionally does
        # not union disjoint dates merely because they were retrieved together.
        clusters: List[Dict[str, Any]] = []
        for span, relevance in sorted(temporal_rows, key=lambda row: (row[0]["start"], row[0]["end"])):
            start, end = self._span_dates(span)
            compatible = next((cluster for cluster in clusters if start <= cluster["end"] and end >= cluster["start"]), None)
            precision = {"exact": 1.0, "bounded": .8, "approximate": .55, "unknown": .3}.get(span.get("precision"), .3)
            if compatible is None:
                compatible = {"start": start, "end": end, "rows": [], "score": 0.0}
                clusters.append(compatible)
            compatible["start"] = min(compatible["start"], start)
            compatible["end"] = max(compatible["end"], end)
            compatible["rows"].append(span)
            compatible["score"] += relevance * precision
        clusters.sort(key=lambda cluster: (-cluster["score"], cluster["start"], cluster["end"]))
        if len(clusters) > 1 and clusters[1]["score"] >= clusters[0]["score"] * .8:
            return [], {"anchor_resolution_status": "ambiguous", "anchor_candidate_count": len(rows),
                        "anchor_temporal_candidate_count": len(temporal_rows), "resolved_anchor_spans": [],
                        "resolved_anchor_source_types": [], "resolved_anchor_source_ids": []}
        resolved = [{"start": clusters[0]["start"].isoformat(), "end": clusters[0]["end"].isoformat(),
                     "precision": "bounded" if clusters[0]["start"] != clusters[0]["end"] else "exact"}]
        source_types = sorted({str(row.get("source_type", "unknown")) for row in clusters[0]["rows"]})
        source_ids = sorted({str(row.get("source_id")) for row in clusters[0]["rows"] if row.get("source_id")})
        return resolved, {"anchor_resolution_status": "resolved", "anchor_candidate_count": len(rows),
                          "anchor_temporal_candidate_count": len(temporal_rows), "resolved_anchor_spans": resolved,
                          "resolved_anchor_source_types": source_types, "resolved_anchor_source_ids": source_ids}

    def _rerank_query_plan_temporal(self, candidates: List[Dict[str, Any]], plan: QueryPlan, anchors: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
        temporal = plan.temporal
        # ``event/none`` retains event-time answer intent in the plan, but it
        # does not impose a retrieval-time temporal constraint.
        if temporal.axis == "none" or temporal.axis == "knowledge" or temporal.relation == "none":
            return candidates
        scores = []
        for item in candidates:
            score, match = self._temporal_compatibility(self._candidate_spans(item, temporal.axis), temporal, anchors)
            item["semantic_score"] = float(item.get("final_score", item.get("score", 0.0)))
            item["temporal_score"] = score
            item["temporal_match_type"] = match
            scores.append(score)
        if temporal.relation in {"latest", "earliest"}:
            dated = []
            for item in candidates:
                spans = self._candidate_spans(item, temporal.axis)
                points = [self._span_dates(span)[1 if temporal.relation == "latest" else 0] for span in spans]
                points = [point for point in points if point is not None]
                if points:
                    dated.append((item, max(points) if temporal.relation == "latest" else min(points)))
            if dated:
                low, high = min(point for _item, point in dated), max(point for _item, point in dated)
                for item, point in dated:
                    order = 1.0 if high == low else (
                        (point - low).days / (high - low).days if temporal.relation == "latest"
                        else (high - point).days / (high - low).days
                    )
                    # Chronology is a bounded secondary preference inside the
                    # semantic pool: compatible spans contribute .25-.50.
                    item["temporal_score"] = min(1.0, float(item["temporal_score"]) * (.25 + .25 * order))
        # Missing annotations are neutral. Temporal compatibility can only
        # amplify semantic relevance; it cannot manufacture it.
        weight = float(self.config.get("temporal_retrieval_weight", 1.0))
        for item in candidates:
            semantic = max(0.0, float(item.get("semantic_score", 0.0)))
            item["joint_score"] = semantic * (1.0 + weight * min(1.0, float(item.get("temporal_score", 0.0))))
            item["final_score"] = item["joint_score"]
        return sorted(candidates, key=lambda item: (-item.get("final_score", 0.0), item["id"]))

    def _temporal_compatibility(self, spans: Sequence[Dict[str, str]], temporal: Any, anchors: Sequence[Dict[str, str]]) -> tuple[float, str | None]:
        if temporal.relation == "none":
            return 0.0, None
        if not spans:
            return 0.0, None
        requested = [(temporal.start, temporal.end)] if temporal.start or temporal.end else []
        if temporal.relation in {"before", "after"} and anchors:
            requested = [self._span_dates(span) for span in anchors]
        best = 0.0
        for span in spans:
            start, end = self._span_dates(span)
            if not start or not end:
                continue
            precision = {"exact": 1.0, "bounded": .8, "approximate": .55, "unknown": .3}.get(span.get("precision"), .3)
            for target_start, target_end in requested or [(None, None)]:
                match = temporal.relation == "none"
                if temporal.relation == "overlap":
                    match = target_start is not None and target_end is not None and start <= target_end and end >= target_start
                elif temporal.relation == "before":
                    match = target_start is not None and end < target_start
                elif temporal.relation == "after":
                    match = target_end is not None and start > target_end
                elif temporal.relation == "latest":
                    match = True
                elif temporal.relation == "earliest":
                    match = True
                if match:
                    # Ordering is local to the semantic pool; date ordinal only breaks ties.
                    best = max(best, min(1.0, precision))
        return best, temporal.relation if best else None

    def merge_rank_channels(self, channels: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Union bounded channel pools while retaining semantic relevance.

        Planner channels widen recall. They do not replace channel-zero scores
        with an outer rank-only score or reduce the union to one channel's cap.
        """
        values: Dict[str, Dict[str, Any]] = {}
        rrf_k = float(self.config.get("rrf_k", 60.0))
        merge_mode = self.config.get("planner_merge_mode", "coverage_interleave")
        for channel_index, channel in enumerate(channels):
            for rank, item in enumerate(channel, 1):
                identifier = item["id"]
                if identifier not in values:
                    merged = deepcopy(item)
                    # Prior selection fields do not describe this union.
                    for key in (
                        "final_score",
                        "selection_score",
                        "selected_rank",
                        "base_rank",
                        "planner_request_indices",
                        "planner_channel_support_count",
                        "planner_fusion_score",
                        "_planner_merged_final_score",
                    ):
                        merged.pop(key, None)
                    values[identifier] = merged
                merged = values[identifier]
                if channel_index:
                    merged.setdefault("planner_request_indices", []).append(channel_index - 1)
                merged["planner_request_indices"] = sorted(set(merged.get("planner_request_indices", [])))
                merged["planner_channel_support_count"] = int(merged.get("planner_channel_support_count", 0)) + 1
                relevance = max(0.0, float(item.get("final_score", item.get("score", 0.0))))
                supports = merged.setdefault("planner_channel_support", [])
                supports.append({"channel": channel_index, "rank": rank, "relevance": relevance})
                merged["planner_channel_support"] = sorted(supports, key=lambda support: support["channel"])
                merged["planner_fusion_score"] = max(float(merged.get("planner_fusion_score", 0.0)), relevance)
                merged["max_channel_relevance"] = max(float(merged.get("max_channel_relevance", 0.0)), relevance)
                if channel_index == 0:
                    merged["base_rank"] = rank
                    merged["original_query_relevance"] = relevance
        for merged in values.values():
            merged["planner_channel_support_count"] = len(merged.get("planner_channel_support", []))
            if merge_mode == "sum_rrf":
                merged["planner_rrf_score"] = sum(1.0 / (rrf_k + item["rank"]) for item in merged["planner_channel_support"])
            # MMR consumes semantic relevance; channel coverage remains explicit
            # metadata rather than a replacement relevance metric.
            merged["score"] = float(merged["max_channel_relevance"])
            merged["fusion_score"] = merged["score"]
            merged["final_score"] = merged["score"]
        if merge_mode == "sum_rrf":
            rows = list(values.values())
            return sorted(rows, key=lambda item: (-float(item.get("planner_rrf_score", 0.0)), -float(item["final_score"]), item["id"]))
        return sorted(values.values(), key=lambda item: (-float(item["final_score"]), item["id"]))

    @staticmethod
    def _set_merged_final_scores(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize outer-fusion relevance once for the final planner pool."""
        rows = list(candidates)
        final_scores = normalize_scores([float(item.get("planner_fusion_score", item.get("score", 0.0))) for item in rows])
        for item, final_score in zip(rows, final_scores):
            item["final_score"] = final_score
            item["_planner_merged_final_score"] = True
        return rows

    def _claim_is_directly_visible(self, claim_id: str, temporal: TemporalQueryConstraint | None = None, state_view: str = "current") -> bool:
        claim = self.store.claims.get(claim_id)
        if not claim:
            return False
        if temporal is not None and temporal.kind == "as_of":
            if claim.persistence == "state":
                return claim_visible_as_of(claim, temporal.target_date) is not None
            recorded = parse_stored_date(claim.recorded_at)
            return recorded is not None and recorded <= temporal.target_date
        if temporal is not None and temporal.kind == "before" and claim.persistence == "state":
            recorded = parse_stored_date(claim.recorded_at)
            if recorded is not None and recorded >= temporal.target_date:
                return False
            valid_from = parse_stored_date(claim.valid_from)
            if valid_from is not None and valid_from >= temporal.target_date:
                return False
        return claim.persistence != "state" or state_view == "all_versions" or claim.status in {"active", "contested"}

    def _visible_claim_vectors(self, temporal: TemporalQueryConstraint | None = None, state_view: str = "current") -> Tuple[Dict[str, Sequence[float]], int]:
        vectors = {}
        hidden = 0
        for claim_id, embedding in self.store.claim_embeddings.items():
            if self._claim_is_directly_visible(claim_id, temporal, state_view):
                vectors[claim_id] = embedding
            elif self.store.claims.get(claim_id) and self.store.claims[claim_id].status in {"superseded", "refined"}:
                hidden += 1
        return vectors, hidden

    def _temporal_episode_rank(self, query_vector: Sequence[float], temporal: TemporalQueryConstraint) -> List[Tuple[str, float, str]]:
        rows = []
        for identifier, episode in self.store.episodes.items():
            match = episode_temporal_match(episode, temporal)
            if match is not None:
                rows.append((identifier, cosine(query_vector, self.store.episode_embeddings.get(identifier, [])), match.match_type))
        rows.sort(key=lambda item: (-item[1], self.store.episodes[item[0]].source_session_index is None, self.store.episodes[item[0]].source_session_index or 0, item[0]))
        return rows[: max(0, int(self.config.get("episode_top_k", 20)))]

    def _temporal_claim_rank(self, query_vector: Sequence[float], temporal: TemporalQueryConstraint, state_view: str = "current") -> List[Tuple[str, float, str]]:
        rows = []
        for identifier, claim in self.store.claims.items():
            match = claim_temporal_match(claim, self.store.episodes, temporal)
            if match is not None and self._claim_is_directly_visible(identifier, temporal, state_view):
                rows.append((identifier, cosine(query_vector, self.store.claim_embeddings.get(identifier, [])), match.match_type))
        rows.sort(key=lambda item: (-item[1], self.store.claims[item[0]].recorded_at or "", item[0]))
        return rows[: max(0, int(self.config.get("claim_top_k", 30)))]

    def _episode_rank(self, query_vector: Sequence[float]) -> List[Tuple[str, float]]:
        """Rank summaries, optionally blending a supporting immutable turn."""
        mode = self.config.get("episode_relevance_mode", "summary")
        if mode == "summary":
            return dense_rank(query_vector, self.store.episode_embeddings, self.config.get("episode_top_k", 20))
        summary_weight = float(self.config.get("episode_summary_weight", 0.5))
        turn_weight = float(self.config.get("episode_best_turn_weight", 0.5))
        normalizer = summary_weight + turn_weight
        if normalizer <= 0:
            summary_weight, turn_weight, normalizer = 1.0, 0.0, 1.0
        rows = []
        for episode_id, vector in self.store.episode_embeddings.items():
            summary_score = cosine(query_vector, vector)
            support_key, support_score = None, None
            for key in self.store.episode_turn_keys.get(episode_id, ()):
                if key not in self.store.turn_embeddings:
                    continue
                value = cosine(query_vector, self.store.turn_embeddings[key])
                if support_score is None or value > support_score:
                    support_key, support_score = key, value
            support_score = support_score if support_score is not None else summary_score
            rows.append((episode_id, (summary_weight * summary_score + turn_weight * support_score) / normalizer, summary_score, support_key, support_score))
        rows.sort(key=lambda row: (-row[1], row[0]))
        self._episode_support = {identifier: (key, score, summary) for identifier, _value, summary, key, score in rows}
        return [(identifier, score) for identifier, score, _summary, _key, _support in rows[:max(0, int(self.config.get("episode_top_k", 20)))]]

    def _fuse(
        self,
        claims: Sequence[Tuple[str, float]], episodes: Sequence[Tuple[str, float]],
        turns: Sequence[Tuple[str, float]] = (), lexical_turns: Sequence[Tuple[str, float]] = (),
        temporal_claims: Sequence[Tuple[str, float, str]] = (), temporal_episodes: Sequence[Tuple[str, float, str]] = (),
    ) -> List[Dict[str, Any]]:
        if self.config.get("fusion_mode", "rrf") == "rrf":
            return self._rrf(claims, episodes, turns, lexical_turns, temporal_claims, temporal_episodes)
        return self._score_fusion(claims, episodes, turns, lexical_turns, temporal_claims, temporal_episodes)

    def _rrf(
        self,
        claims: Sequence[Tuple[str, float]],
        episodes: Sequence[Tuple[str, float]],
        turns: Sequence[Tuple[str, float]] = (),
        lexical_turns: Sequence[Tuple[str, float]] = (),
        temporal_claims: Sequence[Tuple[str, float, str]] = (),
        temporal_episodes: Sequence[Tuple[str, float, str]] = (),
    ) -> List[Dict[str, Any]]:
        values: Dict[str, Dict[str, Any]] = {}
        rrf_k = float(self.config.get("rrf_k", 60.0))
        channels = (
            ("state_claim", claims, float(self.config.get("claim_retrieval_weight", 1.0))),
            ("episode", episodes, float(self.config.get("episode_retrieval_weight", 1.0))),
            ("turn", turns, float(self.config.get("turn_retrieval_weight", 1.0))),
            ("turn", lexical_turns, float(self.config.get("turn_lexical_retrieval_weight", 1.0))),
        )
        for record_type, rows, weight in channels:
            for rank, (identifier, dense_score) in enumerate(rows, 1):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0, "dense_score": dense_score, "fusion_score": 0.0, "ppr_score": 0.0})
                item["score"] += weight / (rrf_k + rank)
                item["fusion_score"] = item["score"]
                item["dense_score"] = max(item["dense_score"], dense_score)
        temporal_weight = float(self.config.get("temporal_retrieval_weight", 1.0))
        for record_type, rows in (("state_claim", temporal_claims), ("episode", temporal_episodes)):
            for rank, (identifier, _dense_score, match_type) in enumerate(rows, 1):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0, "dense_score": 0.0, "fusion_score": 0.0, "ppr_score": 0.0})
                item["score"] += temporal_weight / (rrf_k + rank)
                item["fusion_score"] = item["score"]
                match_score = 0.5 if match_type == "as_of_fallback" else 1.0
                item["temporal_score"] = max(float(item.get("temporal_score", 0.0)), match_score)
                item["temporal_match_type"] = item.get("temporal_match_type") or match_type
        return list(values.values())

    def _score_fusion(
        self,
        claims: Sequence[Tuple[str, float]], episodes: Sequence[Tuple[str, float]],
        turns: Sequence[Tuple[str, float]] = (), lexical_turns: Sequence[Tuple[str, float]] = (),
        temporal_claims: Sequence[Tuple[str, float, str]] = (), temporal_episodes: Sequence[Tuple[str, float, str]] = (),
    ) -> List[Dict[str, Any]]:
        values: Dict[str, Dict[str, Any]] = {}
        calibration = self.config.get("score_calibration_mode", "minmax")
        channels = (
            ("state_claim", claims, float(self.config.get("claim_retrieval_weight", 1.0)), "dense"),
            ("episode", episodes, float(self.config.get("episode_retrieval_weight", 1.0)), "dense"),
            ("turn", turns, float(self.config.get("turn_retrieval_weight", 1.0)), "dense"),
            ("turn", lexical_turns, float(self.config.get("turn_lexical_retrieval_weight", 1.0)), "sparse"),
        )
        # Cosine scores share a representation scale across claims, episodes,
        # and immutable turns, so calibrate their pooled values.  This retains
        # magnitude differences even when two separate lists both have rank 1.
        dense_calibrated = iter(calibrate_scores(
            [float(score) for _type, rows, _weight, channel in channels if channel == "dense" for _identifier, score in rows], calibration,
        ))
        for record_type, rows, weight, channel in channels:
            if channel == "dense":
                calibrated = [next(dense_calibrated) for _identifier, _score in rows]
            else:
                calibrated = calibrate_scores([float(score) for _identifier, score in rows], calibration)
            for (identifier, raw), score in zip(rows, calibrated):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0,
                    "dense_raw_score": None, "dense_calibrated_score": None, "sparse_raw_score": None,
                    "sparse_calibrated_score": None, "temporal_score": 0.0, "ppr_score": 0.0})
                raw_key, calibrated_key = ("dense_raw_score", "dense_calibrated_score") if channel == "dense" else ("sparse_raw_score", "sparse_calibrated_score")
                item[raw_key] = max(float(raw), item[raw_key]) if item[raw_key] is not None else float(raw)
                item[calibrated_key] = max(float(score), item[calibrated_key]) if item[calibrated_key] is not None else float(score)
                item["score"] += weight * float(score)
        temporal_weight = float(self.config.get("temporal_retrieval_weight", 1.0))
        for record_type, rows in (("state_claim", temporal_claims), ("episode", temporal_episodes)):
            for identifier, raw, match_type in rows:
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0,
                    "dense_raw_score": None, "dense_calibrated_score": None, "sparse_raw_score": None,
                    "sparse_calibrated_score": None, "temporal_score": 0.0, "ppr_score": 0.0})
                temporal = .5 if match_type == "as_of_fallback" else 1.0
                item["temporal_score"] = max(item["temporal_score"], temporal)
                item["score"] += temporal_weight * temporal
        for item in values.values():
            item["fusion_score"] = item["score"]
            item["fused_score"] = item["score"]
            item["dense_score"] = item["dense_raw_score"] if item["dense_raw_score"] is not None else 0.0
        return list(values.values())

    def _lexical_turn_rank(self, question: str) -> List[Tuple[str, float]]:
        """Return dependency-free literal-term rankings for archived turns."""
        if self.config.get("turn_lexical_mode", "overlap") == "bm25":
            return self._bm25_turn_rank(question)
        query_tokens = set(re.findall(r"[a-z0-9]+", question.casefold()))
        if not query_tokens:
            return []
        rows = []
        for key in self.store.turn_embeddings:
            episode, turn = self.store.turn_for_key(key)
            if episode is None or turn is None:
                continue
            tokens = set(re.findall(r"[a-z0-9]+", f"{turn.speaker} {turn.text} {turn.image_caption or ''}".casefold()))
            overlap = len(query_tokens & tokens)
            if overlap:
                rows.append((key, overlap / len(query_tokens)))
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows[:max(0, int(self.config.get("turn_top_k", 8)))]

    def _bm25_turn_rank(self, question: str) -> List[Tuple[str, float]]:
        query_tokens = re.findall(r"[a-z0-9]+", question.casefold())
        if not query_tokens:
            return []
        started = time.perf_counter()
        cache = self.store.turn_bm25_cache(
            unit=self.config.get("turn_sparse_unit", "turn"),
            min_chars=int(self.config.get("turn_chunk_min_chars", 24)),
            max_chunks_per_turn=int(self.config.get("turn_chunk_max_per_turn", 8)),
        )
        k1, b = float(self.config.get("bm25_k1", 1.2)), float(self.config.get("bm25_b", .75))
        corpus_size, average = len(cache["documents"]), cache["average_length"] or 1.0
        scores: Dict[str, float] = {}
        for parent, _chunk_index, tokens in cache["documents"]:
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = cache["document_frequency"].get(token, 0)
                idf = math.log(1.0 + (corpus_size - df + .5) / (df + .5))
                score += idf * frequency * (k1 + 1.0) / (frequency + k1 * (1.0 - b + b * len(tokens) / average))
            if score:
                # Chunks resolve to their immutable parent; max avoids a long
                # turn winning merely because it contains many matching chunks.
                scores[parent] = max(scores.get(parent, 0.0), score)
        self._last_bm25_build_seconds = time.perf_counter() - started
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max(0, int(self.config.get("turn_top_k", 8)))]

    def _candidate_event_compatibility(self, item: Dict[str, Any], temporal: TemporalQueryConstraint) -> float:
        if item["type"] == "state_claim":
            claim = self.store.claims.get(item["id"])
            spans = ([{"start": claim.event_time_start, "end": claim.event_time_end}]
                     if claim and claim.event_time_start and claim.event_time_end else [])
        elif item["type"] == "episode":
            spans = self.store.episode_temporal_spans.get(item["id"], [])
        else:
            episode, turn = self.store.turn_for_key(item["id"])
            spans = self.store.turn_temporal_spans.get((episode.episode_id, turn.turn_id), []) if episode and turn else []
        target = temporal.target_date
        for span in spans:
            start, end = parse_stored_date(span.get("start")), parse_stored_date(span.get("end"))
            if not start or not end:
                continue
            if ((temporal.kind == "exact_record_time" and target is not None and start <= target <= end)
                    or (temporal.kind == "interval" and temporal.start_date and temporal.end_date and start <= temporal.end_date and end >= temporal.start_date)
                    or (temporal.kind == "before" and target is not None and end < target)
                    or (temporal.kind == "after" and target is not None and start > target)):
                return 1.0
        return 0.0

    def _candidate_record_compatibility(self, item: Dict[str, Any], temporal: TemporalQueryConstraint) -> float:
        if item["type"] == "episode":
            return float(episode_temporal_match(self.store.episodes[item["id"]], temporal) is not None)
        if item["type"] == "turn":
            episode, _turn = self.store.turn_for_key(item["id"])
            return float(episode is not None and episode_temporal_match(episode, temporal) is not None)
        claim = self.store.claims.get(item["id"])
        return float(claim is not None and claim_temporal_match(claim, self.store.episodes, temporal) is not None)

    def _dual_axis_temporal(self, candidates: List[Dict[str, Any]], temporal: TemporalQueryConstraint) -> List[Dict[str, Any]]:
        """Rerank the semantic pool only; dates never create new candidates."""
        weight = float(self.config.get("temporal_retrieval_weight", 1.0))
        for item in candidates:
            event = self._candidate_event_compatibility(item, temporal)
            record = self._candidate_record_compatibility(item, temporal)
            compatibility = min(1.0, max(event, record) + (.25 if event and record else 0.0))
            semantic = max(0.0, float(item.get("score", 0.0)))
            item["semantic_score"] = semantic
            item["event_time_compatibility"] = event
            item["record_time_compatibility"] = record
            item["temporal_score"] = compatibility
            item["joint_score"] = semantic * (1.0 + weight * compatibility)
            item["score"] = item["joint_score"]
            item["fusion_score"] = item.get("fusion_score", semantic)
        return candidates

    def _candidate_text(self, item: Dict[str, Any]) -> str:
        if item["type"] == "state_claim":
            claim = self.store.claims[item["id"]]
            return f"{claim.subject} {claim.predicate} {claim.value} {claim.qualifiers}"
        if item["type"] == "episode":
            return self.store.episodes[item["id"]].summary
        _episode, turn = self.store.turn_for_key(item["id"])
        return f"{turn.speaker} {turn.text} {turn.image_caption or ''}" if turn else ""

    def _rerank_local(self, question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.config.get("local_reranker_mode", "off") != "cross_encoder" or not candidates:
            return candidates
        available = self.config.get("local_reranker_available_memory_mb", available_memory_mb())
        minimum = float(self.config.get("local_reranker_min_available_mb", 1024))
        if available is None or float(available) < minimum:
            if self.config.get("local_reranker_auto_disable_on_low_memory", True):
                self._reranker_disabled_reason = "memory_telemetry_unavailable" if available is None else "low_available_memory"
                return candidates
            raise RuntimeError("local cross-encoder reranker requires more available memory")
        reranker = self.config.get("local_reranker")
        if reranker is None:
            try:
                model = self.config.get("local_reranker_model")
                if not model:
                    raise ValueError("local_reranker_model is required")
                with _LOCAL_RERANKER_LOCK:
                    reranker = _LOCAL_RERANKERS.get(str(model))
                    if reranker is None:
                        from sentence_transformers import CrossEncoder
                        reranker = CrossEncoder(model, max_length=int(self.config.get("local_reranker_max_length", 512)))
                        _LOCAL_RERANKERS[str(model)] = reranker
            except Exception as exc:
                if self.config.get("local_reranker_auto_disable_on_error", False):
                    self._reranker_disabled_reason = f"unavailable:{type(exc).__name__}"
                    return candidates
                raise RuntimeError("local_reranker_mode=cross_encoder requires sentence-transformers and its configured model") from exc
        base_all = normalize_scores([float(item.get("score", 0.0)) for item in candidates])
        for item, base_value in zip(candidates, base_all):
            item["score"] = base_value
        pool = sorted(candidates, key=lambda item: (-float(item.get("score", 0.0)), item["id"]))[:max(1, int(self.config.get("local_reranker_top_k", 20)))]
        raw = list(reranker.predict([(question, self._candidate_text(item)) for item in pool], batch_size=int(self.config.get("local_reranker_batch_size", 8))))
        calibrated = calibrate_scores([float(value) for value in raw], self.config.get("local_reranker_calibration_mode", "minmax"))
        alpha = float(self.config.get("local_reranker_weight", .35))
        for item, raw_value, calibrated_value in zip(pool, raw, calibrated):
            base_value = float(item["score"])
            item["reranker_raw_score"] = float(raw_value)
            item["reranker_calibrated_score"] = float(calibrated_value)
            item["score"] = 0.0 if base_value <= 1e-12 else (1.0 - alpha) * base_value + alpha * float(calibrated_value)
            item["fused_score"] = item["score"]
        return candidates

    def _ppr(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with get_usage_tracker().scope("event_state.ppr"):
            return self._ppr_impl(candidates)

    def _ppr_impl(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seeds = [item["id"] for item in candidates]
        if not seeds:
            return candidates
        hops = max(0, int(self.config.get("ppr_expand_hops", 2)))
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        relation_weights = {"SUPERSEDES": self.config.get("ppr_weight_supersedes", 1.2), "SUPERSEDED_BY": self.config.get("ppr_weight_supersedes", 1.2), "REFINES": self.config.get("ppr_weight_refines", 1.0), "REFINED_BY": self.config.get("ppr_weight_refines", 1.0), "CONFLICTS_WITH": self.config.get("ppr_weight_conflict", 0.8), "CLAIM_SUPPORTED_BY_EPISODE": self.config.get("ppr_weight_evidence", 0.7), "EPISODE_SUPPORTS_CLAIM": self.config.get("ppr_weight_evidence", 0.7)}
        for edge in self.store.edges:
            weight = float(relation_weights.get(edge["relation_type"], 0.0))
            if weight > 0:
                adjacency[edge["source_id"]].append((edge["target_id"], weight))
        identifiers = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            frontier = {target for source in frontier for target, _ in adjacency.get(source, []) if target in self.store.claims or target in self.store.episodes} - identifiers
            identifiers.update(frontier)
        identifiers = sorted(identifiers)
        initial = {identifier: next((item["score"] for item in candidates if item["id"] == identifier), 0.0) for identifier in identifiers}
        total = sum(initial.values()) or 1.0
        personalization = {key: value / total for key, value in initial.items()}
        scores = dict(personalization)
        alpha = float(self.config.get("ppr_alpha", .85))
        tolerance = float(self.config.get("ppr_tolerance", 1e-6))
        for _ in range(max(1, int(self.config.get("ppr_max_iterations", 20)))):
            updated = {key: (1 - alpha) * personalization[key] for key in identifiers}
            dangling = 0.0
            for source in identifiers:
                neighbors = [(target, weight) for target, weight in adjacency.get(source, []) if target in scores]
                normalizer = sum(weight for _, weight in neighbors)
                if not normalizer:
                    dangling += scores[source]
                    continue
                for target, weight in neighbors:
                    updated[target] += alpha * scores[source] * weight / normalizer
            for target in identifiers:
                updated[target] += alpha * dangling * personalization[target]
            delta = sum(abs(updated[key] - scores[key]) for key in identifiers)
            scores = updated
            if delta < tolerance:
                break
        known = {item["id"] for item in candidates}
        base_values = normalize_scores([item["score"] for item in candidates])
        gamma = float(self.config.get("ppr_mix_weight", .35))
        for item, base in zip(candidates, base_values):
            item["ppr_score"] = scores[item["id"]]
            item["base_score"] = base
            item["score"] = (1 - gamma) * base + gamma * scores[item["id"]]
            item["final_score"] = item["score"]
        for identifier in identifiers:
            if identifier not in known:
                record_type = "state_claim" if identifier in self.store.claims else "episode"
                candidates.append({"id": identifier, "type": record_type, "score": gamma * scores[identifier], "fusion_score": 0.0, "dense_score": 0.0, "ppr_score": scores[identifier], "base_score": 0.0, "final_score": gamma * scores[identifier]})
        return candidates

    def _select(
        self,
        candidates: Sequence[Dict[str, Any]],
        count: int,
        *,
        source_info: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        with get_usage_tracker().scope("event_state.selector"):
            # Preserve the private selector call contract used by legacy
            # regression oracles when source-coherent attribution is inactive.
            if self.config.get("selector_mode", "state_mmr") != "source_coherent_mmr":
                return self._select_impl(candidates, count)
            return self._select_impl(candidates, count, source_info=source_info)

    def _select_impl(
        self,
        candidates: Sequence[Dict[str, Any]],
        count: int,
        *,
        source_info: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        mode, selected = self.config.get("selector_mode", "state_mmr"), []
        remaining = list(candidates)
        if remaining and all(item.get("_planner_merged_final_score") for item in remaining):
            relevance = [float(item["final_score"]) for item in remaining]
        else:
            relevance = normalize_scores([float(item.get("final_score", item.get("score", 0.0))) for item in remaining])
        relevance_by_id = {item["id"]: value for item, value in zip(remaining, relevance)}

        # Selection is read-only. Cache only values that the original scoring
        # expression recomputed for every remaining candidate and MMR round.
        vector_cache: Dict[int, Sequence[float]] = {}
        similarity_cache: Dict[Tuple[int, int], float] = {}
        source_ids_cache: Dict[int, set[Any]] = {}
        related_ids: Dict[str, set[str]] | None = None
        selected_ids: set[str] = set()
        selected_source_ids: set[Any] = set()
        selected_sources_ready = False

        def vector(item: Dict[str, Any]) -> Sequence[float]:
            key = id(item)
            if key not in vector_cache:
                vector_cache[key] = self._vector(item["id"], item["type"])
            return vector_cache[key]

        def similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
            key = (id(left), id(right))
            if key not in similarity_cache:
                # Keep the existing cosine implementation and operand order so
                # cached values are bit-for-bit the original score values.
                similarity_cache[key] = cosine(vector(left), vector(right))
            return similarity_cache[key]

        def source_ids(item: Dict[str, Any]) -> set[Any]:
            key = id(item)
            if key not in source_ids_cache:
                # State MMR deliberately keeps its historical full-provenance
                # source-diversity behavior. Query-local attribution belongs
                # exclusively to source-coherent selection.
                source_ids_cache[key] = set(
                    (source_info or {}).get("support_sources", {}).get(key, self._source_ids(item))
                    if mode == "source_coherent_mmr" else self._source_ids(item)
                )
            return source_ids_cache[key]

        source_scores: Dict[Any, float] = {}
        preferred_sources: set[Any] = set()
        if mode == "source_coherent_mmr":
            source_scores = dict((source_info or {}).get("source_scores", {}))
            if not source_scores:
                grouped: Dict[Any, List[float]] = defaultdict(list)
                for item in remaining:
                    for source in source_ids(item):
                        if source is not None:
                            grouped[source].append(relevance_by_id[item["id"]])
                additional = max(0, int(self.config.get("source_coherence_additional_candidates", 2)))
                for source, scores in grouped.items():
                    ranked = sorted(scores, reverse=True)
                    source_scores[source] = ranked[0] + sum(ranked[1:additional + 1]) * float(self.config.get("source_coherence_additional_weight", .35))
            preferred_sources = set((source_info or {}).get("preferred_sources", self._preferred_sources(source_scores)))

        while remaining and len(selected) < count:
            # Preserve the established semantic/episode evidence path when it
            # exists; immutable turns complement it rather than replacing the
            # only selected memory object under a small evidence budget.
            semantic_remaining = [item for item in remaining if item["type"] != "turn"]
            choices = semantic_remaining if semantic_remaining and not selected else remaining
            if mode == "topk":
                choice, choice_score = max(((item, item.get("final_score", item.get("score", 0.0))) for item in choices), key=lambda pair: (pair[1], pair[0]["id"] ))
            else:
                weight = float(self.config.get("mmr_lambda", .7))

                use_state_relation = mode == "state_mmr" or (
                    mode == "source_coherent_mmr"
                    and self.config.get("source_coherent_use_state_relation_bonus", True)
                )
                use_representation_balance = mode == "state_mmr" or (
                    mode == "source_coherent_mmr"
                    and self.config.get("source_coherent_use_representation_balance_bonus", True)
                )
                use_source_diversity = mode == "state_mmr" or (
                    mode == "source_coherent_mmr"
                    and self.config.get("source_coherent_use_source_diversity_bonus", False)
                )
                if use_state_relation and selected and related_ids is None:
                    related_ids = {}
                    for edge in self.store.edges:
                        source_id, target_id = edge["source_id"], edge["target_id"]
                        related_ids.setdefault(source_id, set()).add(target_id)
                        related_ids.setdefault(target_id, set()).add(source_id)
                if use_source_diversity and selected and not selected_sources_ready:
                    for prior in selected:
                        selected_source_ids.update(source_ids(prior))
                    selected_sources_ready = True

                def score(item: Dict[str, Any]) -> float:
                    redundancy = max((similarity(item, other) for other in selected), default=0.0)
                    value = weight * relevance_by_id[item["id"]] - (1 - weight) * redundancy
                    if selected:
                        if use_state_relation and related_ids and related_ids.get(item["id"], set()) & selected_ids:
                            value += float(self.config.get("state_relation_bonus", .05))
                        if use_representation_balance and item["type"] != selected[-1]["type"]:
                            value += float(self.config.get("representation_balance_bonus", .02))
                        if use_source_diversity and source_ids(item) - selected_source_ids:
                            value += float(self.config.get("source_diversity_bonus", .02))
                    if mode == "source_coherent_mmr":
                        sources = source_ids(item)
                        if sources & preferred_sources:
                            value += float(self.config.get("source_coherence_bonus", .08))
                        elif sources:
                            # A soft penalty retains genuinely strong evidence
                            # from an otherwise less competitive source.
                            value -= float(self.config.get("source_coherence_penalty", .04))
                    return value

                scores = {id(item): score(item) for item in choices}
                choice = max(choices, key=lambda item: (scores[id(item)], item["id"]))
                choice_score = scores[id(choice)]
            choice["selection_score"] = choice_score
            selected.append(choice)
            remaining.remove(choice)
            selected_ids.add(choice["id"])
            if (mode == "state_mmr" or self.config.get("source_coherent_use_source_diversity_bonus", False)) and selected_sources_ready:
                selected_source_ids.update(source_ids(choice))
        for rank, item in enumerate(selected, 1):
            item["selected_rank"] = rank
        return selected

    def source_coherence_diagnostics(
        self,
        candidates: Sequence[Dict[str, Any]],
        selected: Sequence[Dict[str, Any]],
        source_info: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return bounded source provenance diagnostics for final selection."""
        source_info = source_info or self._query_support_info(candidates, None)
        support_by_item = source_info.get("support_sources", {})
        sources = set()
        query_sources = set()
        for item in candidates:
            sources.update(source for source in self._source_ids(item) if source is not None)
            query_sources.update(source for source in support_by_item.get(id(item), self._source_ids(item)) if source is not None)
        selected_sources = set()
        selected_query_sources = set()
        for item in selected:
            selected_sources.update(source for source in self._source_ids(item) if source is not None)
            selected_query_sources.update(source for source in support_by_item.get(id(item), self._source_ids(item)) if source is not None)
        scores: Dict[str, float] = {}
        for item in candidates:
            for source in support_by_item.get(id(item), self._source_ids(item)):
                if source is not None:
                    scores[str(source)] = max(scores.get(str(source), 0.0), float(item.get("final_score", item.get("score", 0.0))))
        preferred = set(source_info.get("preferred_sources", set()))
        preferred_selected = sum(bool(support_by_item.get(id(item), self._source_ids(item)) & preferred) for item in selected)
        nonpreferred_selected = sum(
            bool(support_by_item.get(id(item), self._source_ids(item)))
            and not bool(support_by_item.get(id(item), self._source_ids(item)) & preferred)
            for item in selected
        )
        return {
            "source_coherence_mode": self.config.get("selector_mode", "state_mmr"),
            "candidate_full_provenance_source_count": len(sources),
            "selected_full_provenance_source_count": len(selected_sources),
            "candidate_query_support_source_count": len(query_sources),
            "selected_query_support_source_count": len(selected_query_sources),
            "selected_query_support_source_ids": sorted(str(source) for source in selected_query_sources)[:32],
            "preferred_query_support_source_ids": sorted(str(source) for source in preferred)[:32],
            "top_query_support_source_scores": [
                {"source_session_id": source, "score": score}
                for source, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:16]
            ],
            "claims_using_query_local_source_attribution": source_info.get("query_local_claim_count", 0),
            "claims_falling_back_to_full_provenance": source_info.get("full_provenance_fallback_count", 0),
            "source_coherence_preferred_selected_count": preferred_selected,
            "source_coherence_nonpreferred_selected_count": nonpreferred_selected,
            "candidate_source_session_count": len(sources),
            "selected_source_session_count": len(selected_sources),
            "selected_source_session_ids": sorted(str(source) for source in selected_sources)[:32],
            "top_source_scores": [
                {"source_session_id": source, "score": score}
                for source, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:16]
            ],
            "source_coherence_filter_count": nonpreferred_selected,
        }

    def _query_support_info(
        self, candidates: Sequence[Dict[str, Any]], query_vector: Sequence[float] | None,
    ) -> Dict[str, Any]:
        """Resolve query-local claim support without changing canonical provenance."""
        support_sources: Dict[int, set[Any]] = {}
        query_local_claim_count = full_provenance_fallback_count = 0
        similarity_cache: Dict[str, float] = {}
        mode = self.config.get("source_coherence_claim_source_mode", "query_relevant")
        max_sources = max(1, int(self.config.get("source_coherence_claim_max_sources", 2)))
        for item in candidates:
            if item["type"] != "state_claim" or mode == "all_provenance":
                support_sources[id(item)] = self._source_ids(item)
                continue
            claim = self.store.claims.get(item["id"])
            if claim is None or query_vector is None:
                support_sources[id(item)] = self._source_ids(item)
                full_provenance_fallback_count += int(claim is not None)
                continue
            source_scores: Dict[Any, float] = {}
            fallback_refs = []
            for ref_index, ref in enumerate(claim.evidence):
                episode = self.store.episodes.get(ref.episode_id)
                if episode is None:
                    continue
                ref_keys = [
                    key for key in self.store.episode_turn_keys.get(episode.episode_id, ())
                    if not ref.source_turn_ids
                    or self.store.turn_metadata.get(key, {}).get("source_turn_id") in ref.source_turn_ids
                ]
                fallback_refs.append((0 if ref.support_type == "origin" else 1, ref_index, str(ref.source_session_id), ref.source_session_id))
                values = []
                for key in ref_keys:
                    vector = self.store.turn_embeddings.get(key)
                    if not vector:
                        continue
                    if key not in similarity_cache:
                        similarity_cache[key] = cosine(query_vector, vector)
                    values.append(similarity_cache[key])
                if values and ref.source_session_id is not None:
                    source_scores[ref.source_session_id] = max(source_scores.get(ref.source_session_id, -math.inf), max(values))
            if source_scores:
                support_sources[id(item)] = {
                    source for source, _score in sorted(source_scores.items(), key=lambda row: (-row[1], str(row[0])))[:max_sources]
                }
                query_local_claim_count += 1
            elif fallback_refs:
                fallback_refs.sort()
                support_sources[id(item)] = {fallback_refs[0][3]} if fallback_refs[0][3] is not None else self._source_ids(item)
            else:
                support_sources[id(item)] = self._source_ids(item)
                full_provenance_fallback_count += 1

        source_scores: Dict[Any, float] = {}
        if self.config.get("selector_mode", "state_mmr") == "source_coherent_mmr":
            grouped: Dict[Any, List[float]] = defaultdict(list)
            relevance = (
                [float(item["final_score"]) for item in candidates]
                if candidates and all(item.get("_planner_merged_final_score") for item in candidates)
                else normalize_scores([float(item.get("final_score", item.get("score", 0.0))) for item in candidates])
            )
            for item, value in zip(candidates, relevance):
                for source in support_sources.get(id(item), set()):
                    if source is not None:
                        grouped[source].append(value)
            additional = max(0, int(self.config.get("source_coherence_additional_candidates", 2)))
            for source, values in grouped.items():
                ranked = sorted(values, reverse=True)
                source_scores[source] = ranked[0] + sum(ranked[1:additional + 1]) * float(self.config.get("source_coherence_additional_weight", .35))
        preferred_sources = self._preferred_sources(source_scores)
        return {
            "support_sources": support_sources,
            "source_scores": source_scores,
            "preferred_sources": preferred_sources,
            "query_local_claim_count": query_local_claim_count,
            "full_provenance_fallback_count": full_provenance_fallback_count,
        }

    def _preferred_sources(self, source_scores: Dict[Any, float]) -> set[Any]:
        """Choose competitive sources until the configured relevance mass is covered."""
        ordered = sorted(source_scores, key=lambda source: (-source_scores[source], str(source)))
        if not ordered:
            return set()
        maximum = self.config.get("source_coherence_max_sources", None)
        limit = len(ordered) if maximum is None else max(1, int(maximum))
        preferred = {ordered[0]}
        best = source_scores[ordered[0]]
        total = sum(source_scores.values()) or 1.0
        accumulated = best
        ratio = max(0.0, float(self.config.get("source_coherence_score_ratio", .7)))
        mass = max(0.0, min(1.0, float(self.config.get("source_coherence_score_mass", .85))))
        for source in ordered[1:limit]:
            competitive = source_scores[source] >= best * ratio
            needs_mass = accumulated / total < mass
            if not competitive and not needs_mass:
                continue
            preferred.add(source)
            accumulated += source_scores[source]
        return preferred

    def _vector(self, identifier: str, record_type: str) -> Sequence[float]:
        if record_type == "state_claim":
            return self.store.claim_embeddings.get(identifier, [])
        if record_type == "turn":
            return self.store.turn_embeddings.get(identifier, [])
        return self.store.episode_embeddings.get(identifier, [])

    def _source_ids(self, item: Dict[str, Any]) -> set[Any]:
        if item["type"] == "episode":
            episode = self.store.episodes.get(item["id"])
            return {episode.source_session_id} if episode is not None else set()
        if item["type"] == "turn":
            return {self.store.turn_metadata.get(item["id"], {}).get("source_session_id")}
        claim = self.store.claims.get(item["id"])
        return {ref.source_session_id for ref in claim.evidence} if claim is not None else set()
