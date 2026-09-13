"""Deterministic rendering and token budgeting for final answer context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .embeddings import cosine
from .schemas import Claim, Episode, EvidenceRef
from .subjects import display_subject


def relation_lines(claim_id: str, edges: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for edge in edges:
        if edge["source_id"] == claim_id:
            lines.append(f"{edge['relation_type']} {edge['target_id']}")
    return lines


def prior_state_lines(claim: Claim, edges: Sequence[Dict[str, Any]], claims: Dict[str, Claim], limit: int = 3) -> List[str]:
    """Follow direct version predecessors from current to older states."""
    lines, visited, frontier = [], {claim.claim_id}, [claim.claim_id]
    while frontier and len(lines) < max(0, limit):
        current = frontier.pop(0)
        predecessors = sorted(
            edge["target_id"]
            for edge in edges
            if edge["source_id"] == current and edge["relation_type"] in {"SUPERSEDES", "REFINES"}
        )
        for claim_id in predecessors:
            if claim_id in visited:
                continue
            visited.add(claim_id)
            prior = claims.get(claim_id)
            if not prior:
                continue
            lines.append(f"  {prior.predicate} = {prior.value}; recorded {prior.recorded_at or 'unknown'}; valid to {prior.valid_to or 'unknown'}")
            frontier.append(claim_id)
            if len(lines) == max(0, limit):
                break
    return lines


def render_claim(claim: Claim, edges: Sequence[Dict[str, Any]], claims: Dict[str, Claim] | None = None) -> str:
    relations = relation_lines(claim.claim_id, edges)
    evidence = "\n".join(f"  session {ref.source_session_id} ({ref.support_type})" for ref in claim.evidence)
    prior_states = prior_state_lines(claim, edges, claims or {})
    event_time = []
    if claim.event_time_start and claim.event_time_end:
        event_time.append(
            f"Event time: {claim.event_time_start}"
            + (f" through {claim.event_time_end}" if claim.event_time_end != claim.event_time_start else "")
        )
        event_time.append(f"Event-time precision: {claim.event_time_precision}")
    if claim.valid_time_text:
        event_time.append(f"Source temporal wording: {claim.valid_time_text}")
    lines = [
        f"[State {claim.claim_id}]",
        f"Subject: {display_subject(claim.subject_id or claim.subject_key, claim.subject)}",
        f"Subject ID: {claim.subject_id or claim.subject_key}",
        f"Status: {claim.status}",
        f"Persistence: {claim.persistence}",
        f"Modality: {claim.modality}",
        f"Polarity: {claim.polarity}",
        f"Recorded: {claim.recorded_at or 'unknown'}",
        f"Valid from: {claim.valid_from or 'unknown'}",
        f"Valid to: {claim.valid_to or 'unknown'}",
        *event_time,
        f"Claim: {claim.predicate} = {claim.value}",
        f"Qualifiers: {claim.qualifiers or 'none'}",
        "Relations:\n" + ("\n".join(f"  {item}" for item in relations) if relations else "  none"),
        "Prior states:\n" + ("\n".join(prior_states) if prior_states else "  none"),
        "Evidence:\n" + (evidence or "  none"),
    ]
    return "\n".join(lines)


def _turn_text(turn: Any, fallback_timestamp: str | None = None) -> str:
    text = f"{turn.speaker} ({turn.timestamp or fallback_timestamp or 'unknown'}): {turn.text}"
    if turn.image_caption:
        text += f" [Shared image: {turn.image_caption}]"
    return text


def episode_turn_embedding_text(turn: Any) -> str:
    """Build the small immutable representation used for query-time scoring."""
    role = f" {turn.role}" if turn.role else ""
    caption = f" {turn.image_caption}" if turn.image_caption else ""
    return f"{turn.speaker}{role}: {turn.text}{caption}".strip()


def evidence_identity(
    record_type: str, episode_id: Any, source_turn_id: Any | None = None,
) -> Tuple[Any, ...]:
    """Return a provenance-aware identity for retrieval evidence."""
    if record_type in {"turn", "immutable_turn"}:
        if source_turn_id is None:
            raise ValueError("Immutable-turn evidence requires a source turn ID")
        return ("turn", episode_id, source_turn_id)
    if record_type == "episode":
        return ("episode", episode_id)
    return (record_type, episode_id)


@dataclass(frozen=True)
class SelectedClaimEvidence:
    """One query-selected immutable provenance reference and its turns."""

    ref_index: int
    ref: EvidenceRef
    episode: Episode
    turns: Tuple[Any, ...]


@dataclass(frozen=True)
class RenderedClaimEvidence:
    """The exact selected provenance turns rendered into one source block."""

    selection: SelectedClaimEvidence
    turns: Tuple[Any, ...]

    def metadata(self) -> Dict[str, Any]:
        ref = self.selection.ref
        return {
            "evidence": {
                "source_session_id": ref.source_session_id,
                "episode_id": ref.episode_id,
                "source_turn_ids": [turn.turn_id for turn in self.turns],
                "support_type": ref.support_type,
            }
        }


def _source_order(episode: Episode) -> Tuple[int, Any, str]:
    """Order selected evidence by immutable session order when it is available."""
    index = episode.source_session_index
    if isinstance(index, int):
        return 0, index, episode.episode_id
    return 1, str(index if index is not None else episode.source_session_id), episode.episode_id


def _reference_turns(ref: EvidenceRef, episode: Episode) -> List[Tuple[int, Any]]:
    """Resolve cited immutable turns, retaining the legacy first-turn fallback."""
    turns = [
        (index, turn)
        for index, turn in enumerate(episode.turn_evidence)
        if not ref.source_turn_ids or turn.turn_id in ref.source_turn_ids
    ]
    if not turns and episode.turn_evidence:
        turns = [(0, episode.turn_evidence[0])]
    return turns


def select_claim_evidence(
    claim: Claim,
    episodes: Dict[str, Episode],
    query_vector: Sequence[float],
    embedder: Any,
    ref_limit: int,
    turn_limit: int = 3,
    turn_vector_cache: Dict[Tuple[str, Any], Sequence[float]] | None = None,
    query_vectors: Sequence[Sequence[float]] | None = None,
) -> List[SelectedClaimEvidence]:
    """Select bounded claim provenance by query similarity without mutating memory."""
    if ref_limit <= 0 or turn_limit <= 0:
        return []
    candidates: List[Tuple[int, EvidenceRef, Episode, List[Tuple[int, Any]]]] = []
    vectors = turn_vector_cache if turn_vector_cache is not None else {}
    query_vectors = [list(vector) for vector in (query_vectors or [query_vector])]
    missing: List[Tuple[Tuple[str, Any], Any]] = []
    for ref_index, ref in enumerate(claim.evidence):
        episode = episodes.get(ref.episode_id)
        if episode is None:
            continue
        turns = _reference_turns(ref, episode)
        if not turns:
            continue
        candidates.append((ref_index, ref, episode, turns))
        for _, turn in turns:
            key = (episode.episode_id, turn.turn_id)
            if key not in vectors:
                vectors[key] = ()
                missing.append((key, turn))
    if not candidates:
        return []
    if missing:
        texts = [episode_turn_embedding_text(turn) for _, turn in missing]
        try:
            embedded = embedder.embed_documents(texts)
        except AttributeError:
            embedded = [embedder.embed_query(text) for text in texts]
        for (key, _), vector in zip(missing, embedded):
            vectors[key] = vector

    scored_refs = []
    for ref_index, ref, episode, turns in candidates:
        scored_turns = [
            (max(cosine(query, vectors[(episode.episode_id, turn.turn_id)]) for query in query_vectors), turn_index, str(turn.turn_id), turn)
            for turn_index, turn in turns
        ]
        score = max(item[0] for item in scored_turns)
        scored_refs.append((score, ref_index, ref, episode, scored_turns))
    selected_refs = sorted(
        scored_refs,
        key=lambda item: (-item[0], item[1], *_source_order(item[3]), str(item[2].source_session_id)),
    )[:ref_limit]

    selected = []
    for _, ref_index, ref, episode, scored_turns in selected_refs:
        chosen = sorted(scored_turns, key=lambda item: (-item[0], item[1], item[2]))[:turn_limit]
        turns = tuple(item[3] for item in sorted(chosen, key=lambda item: (item[1], item[2])))
        selected.append(SelectedClaimEvidence(ref_index, ref, episode, turns))
    return sorted(selected, key=lambda item: (*_source_order(item.episode), item.ref_index))


def selected_claim_evidence_turn_keys(selections: Iterable[SelectedClaimEvidence]) -> set[Tuple[str, Any]]:
    """Return the exact selected immutable turns used for rendering and deduplication."""
    return {
        (selection.episode.episode_id, turn.turn_id)
        for selection in selections
        for turn in selection.turns
    }


def render_selected_claim_evidence(
    selections: Iterable[SelectedClaimEvidence], already: set[Tuple[str, Any]]
) -> List[RenderedClaimEvidence]:
    """Render only selected claim provenance turns that have not already appeared."""
    rendered = []
    for selection in selections:
        turns = tuple(
            turn
            for turn in selection.turns
            if (selection.episode.episode_id, turn.turn_id) not in already
        )
        if not turns:
            continue
        already.update((selection.episode.episode_id, turn.turn_id) for turn in turns)
        rendered.append(RenderedClaimEvidence(selection, turns))
    return rendered


def select_episode_evidence(
    episode: Episode,
    query_vector: Sequence[float],
    embedder: Any,
    limit: int = 2,
    turn_vector_cache: Dict[Tuple[str, Any], Sequence[float]] | None = None,
) -> List[Any]:
    """Select query-relevant turns from one episode in source order.

    This compatibility helper is deliberately local; query assembly uses the
    global selector below so its source-excerpt budget is not per episode.
    """
    selected, _, _ = select_global_episode_evidence(
        [(0, episode)], query_vector, embedder, limit,
        turn_vector_cache=turn_vector_cache,
    )
    return selected.get(episode.episode_id, [])


def select_global_episode_evidence(
    selected_episodes: Sequence[Tuple[int, Episode]],
    query_vector: Sequence[float],
    embedder: Any,
    limit: int = 2,
    excluded_turns: set[Tuple[str, Any]] | None = None,
    query_vectors: Sequence[Sequence[float]] | None = None,
    turn_vector_cache: Dict[Tuple[str, Any], Sequence[float]] | None = None,
) -> Tuple[Dict[str, List[Any]], int, int]:
    """Select a small non-duplicate source-evidence set across episodes.

    Similarity decides which turns survive. Rendering order is restored to the
    selected-memory order and each episode's immutable conversation order.
    """
    excluded = excluded_turns or set()
    candidates: List[Tuple[int, Episode, int, Any]] = []
    deduplicated = 0
    for episode_rank, episode in selected_episodes:
        for turn_index, turn in enumerate(episode.turn_evidence):
            if (episode.episode_id, turn.turn_id) in excluded:
                deduplicated += 1
                continue
            candidates.append((episode_rank, episode, turn_index, turn))
    if limit <= 0 or not candidates:
        return {}, len(candidates), deduplicated
    vectors_by_key = turn_vector_cache if turn_vector_cache is not None else {}
    missing: List[Tuple[Tuple[str, Any], Any]] = []
    for _, episode, _, turn in candidates:
        key = (episode.episode_id, turn.turn_id)
        if key not in vectors_by_key:
            vectors_by_key[key] = ()
            missing.append((key, turn))
    if missing:
        texts = [episode_turn_embedding_text(turn) for _, turn in missing]
        try:
            embedded = embedder.embed_documents(texts)
        except AttributeError:
            embedded = [embedder.embed_query(text) for text in texts]
        for (key, _), vector in zip(missing, embedded):
            vectors_by_key[key] = vector
    vectors = [vectors_by_key[(episode.episode_id, turn.turn_id)] for _, episode, _, turn in candidates]
    query_vectors = [list(vector) for vector in (query_vectors or [query_vector])]
    scored = sorted(
        (
            (
                max(cosine(query, vector) for query in query_vectors),
                episode_rank,
                turn_index,
                episode.episode_id,
                str(turn.turn_id),
                index,
            )
            for index, ((episode_rank, episode, turn_index, turn), vector) in enumerate(zip(candidates, vectors))
        ),
        key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
    )[:limit]
    chosen_indexes = {item[-1] for item in scored}
    selected: Dict[str, List[Any]] = {}
    for index, (episode_rank, episode, turn_index, turn) in enumerate(candidates):
        if index not in chosen_indexes:
            continue
        selected.setdefault(episode.episode_id, []).append(turn)
    return selected, len(candidates), deduplicated


def select_joint_episode_support_evidence(
    support_candidates: Sequence[Tuple[int, Episode, Any, float]],
    limit: int,
    excluded_turns: set[Tuple[str, Any]] | None = None,
) -> Tuple[Dict[str, List[Any]], int, int, List[Tuple[str, Any]]]:
    """Globally reserve the strongest selected-episode support turns.

    One preferred support turn may be supplied per episode.  This first pass
    deliberately does not promise every episode a slot; the normal global
    allocator fills any remaining budget afterward.
    """
    excluded = excluded_turns or set()
    eligible = []
    deduplicated = 0
    seen = set()
    for _rank, episode, turn, score in support_candidates:
        key = (episode.episode_id, turn.turn_id)
        if key in excluded or key in seen:
            deduplicated += 1
            continue
        seen.add(key)
        eligible.append((float(score), str(episode.source_session_id), episode.episode_id, str(turn.turn_id), turn))
    chosen = sorted(eligible, key=lambda row: (-row[0], row[1], row[2], row[3]))[:max(0, int(limit))]
    selected: Dict[str, List[Any]] = {}
    selected_keys = []
    for _score, _source, episode_id, _turn_id, turn in chosen:
        selected.setdefault(episode_id, []).append(turn)
        selected_keys.append((episode_id, turn.turn_id))
    return selected, len(eligible), deduplicated, selected_keys


def select_joint_episode_evidence(
    selected_episodes: Sequence[Tuple[int, Episode]],
    support_candidates: Sequence[Tuple[int, Episode, Any, float]],
    query_vector: Sequence[float],
    embedder: Any,
    limit: int,
    excluded_turns: set[Tuple[str, Any]] | None = None,
    query_vectors: Sequence[Sequence[float]] | None = None,
    turn_vector_cache: Dict[Tuple[str, Any], Sequence[float]] | None = None,
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """Reserve globally ranked support turns, then globally fill spare slots."""
    selected, support_count, support_deduplicated, support_keys = select_joint_episode_support_evidence(
        support_candidates, limit, excluded_turns,
    )
    global_fill, global_candidates, global_deduplicated = select_global_episode_evidence(
        selected_episodes,
        query_vector,
        embedder,
        max(0, int(limit) - len(support_keys)),
        (excluded_turns or set()) | set(support_keys),
        query_vectors=query_vectors,
        turn_vector_cache=turn_vector_cache,
    )
    for episode_id, turns in global_fill.items():
        selected.setdefault(episode_id, []).extend(turns)
    return selected, {
        "support_candidate_count": support_count,
        "support_reserved_count": len(support_keys),
        "support_deduplicated_count": support_deduplicated,
        "support_turn_keys": support_keys,
        "global_fill_count": sum(len(turns) for turns in global_fill.values()),
        "global_candidate_count": global_candidates,
        "global_deduplicated_count": global_deduplicated,
    }


def claim_evidence_turn_keys(
    claim: Claim,
    episodes: Dict[str, Episode],
    limit: int,
    query_vector: Sequence[float] | None = None,
    embedder: Any | None = None,
) -> set[Tuple[str, Any]]:
    """Return exact immutable turns for query-selected or legacy evidence."""
    if query_vector is not None and embedder is not None:
        return selected_claim_evidence_turn_keys(
            select_claim_evidence(claim, episodes, query_vector, embedder, limit)
        )
    keys: set[Tuple[str, Any]] = set()
    for ref in claim.evidence[:max(0, limit)]:
        episode = episodes.get(ref.episode_id)
        if not episode:
            continue
        turns = [turn for turn in episode.turn_evidence if not ref.source_turn_ids or turn.turn_id in ref.source_turn_ids]
        if not turns:
            turns = episode.turn_evidence[:1]
        keys.update((episode.episode_id, turn.turn_id) for turn in turns[:3])
    return keys


def render_episode(episode: Episode, evidence_turns: Sequence[Any] = ()) -> str:
    lines = [
        f"[Episode {episode.episode_id}]",
        f"Recorded: {episode.recorded_at or 'unknown'}",
        f"Participants: {', '.join(episode.participants) or 'unknown'}",
        f"Scope: {episode.conversation_scope or 'unknown'}",
        f"Summary: {episode.summary}",
    ]
    if evidence_turns:
        lines.extend(["", "Relevant source evidence:"])
        lines.extend(f"  {_turn_text(turn, episode.recorded_at)}" for turn in evidence_turns)
    return "\n".join(lines)


def render_episode_evidence(episode: Episode, evidence_turns: Sequence[Any]) -> str:
    """Render selected immutable turns as a separately budgetable source block."""
    excerpt = "\n".join(f"  {_turn_text(turn, episode.recorded_at)}" for turn in evidence_turns)
    return f"[Episode Evidence {episode.episode_id} / session {episode.source_session_id}]\n{excerpt}"


def expand_claim_evidence(
    claim: Claim,
    episodes: Dict[str, Episode],
    already: set[Any],
    limit: int,
    query_vector: Sequence[float] | None = None,
    embedder: Any | None = None,
) -> List[str]:
    """Render selected claim evidence, retaining a legacy compatibility path."""
    if query_vector is not None and embedder is not None:
        rendered = render_selected_claim_evidence(
            select_claim_evidence(claim, episodes, query_vector, embedder, limit),
            already,
        )
        return [
            f"[Supporting Evidence {item.selection.episode.episode_id} / session {item.selection.episode.source_session_id}]\n"
            + "\n".join(f"  {_turn_text(turn, item.selection.episode.recorded_at)}" for turn in item.turns)
            for item in rendered
        ]
    blocks = []
    for ref in claim.evidence[:max(0, limit)]:
        episode = episodes.get(ref.episode_id)
        if not episode:
            continue
        turns = [turn for turn in episode.turn_evidence if not ref.source_turn_ids or turn.turn_id in ref.source_turn_ids]
        if not turns:
            turns = episode.turn_evidence[:1]
        unseen = [
            turn for turn in turns
            if episode.episode_id not in already and (episode.episode_id, turn.turn_id) not in already
        ]
        if not unseen:
            continue
        excerpt_turns = unseen[:3]
        already.update((episode.episode_id, turn.turn_id) for turn in excerpt_turns)
        excerpt = "\n".join(f"  {_turn_text(turn, episode.recorded_at)}" for turn in excerpt_turns)
        blocks.append(f"[Supporting Evidence {episode.episode_id} / session {episode.source_session_id}]\n{excerpt}")
    return blocks


def fit_context(
    blocks: Sequence[Dict[str, str]],
    system_message: str,
    instruction: str,
    question: str,
    max_context_tokens: int,
    answer_reserve: int,
    count_tokens: Any,
    truncate_to_tokens: Any,
) -> Tuple[List[str], int]:
    """Fit complete state/episode blocks and truncate only source excerpts."""
    included: List[str] = []

    def total(texts: Sequence[str]) -> int:
        user_content = instruction + ("\n\n" + "\n\n".join(texts) if texts else "") + "\n\n" + question
        return count_tokens(system_message or "") + count_tokens(user_content) + max(0, int(answer_reserve))

    for block in blocks:
        text, kind = block["text"], block.get("kind", "state")
        if total(included + [text]) <= int(max_context_tokens):
            included.append(text)
            continue
        if kind != "source":
            continue
        remaining = int(max_context_tokens) - total(included)
        if remaining <= 0:
            continue
        low, high, best = 0, max(1, count_tokens(text)), ""
        while low <= high:
            mid = (low + high) // 2
            candidate = truncate_to_tokens(text, mid)
            if candidate and total(included + [candidate]) <= int(max_context_tokens):
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        if best:
            included.append(best)
    return included, sum(count_tokens(block) for block in included)
