"""In-memory immutable evidence archive and versioned semantic state store."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import Claim, Episode, StateOperation, claim_from_dict, episode_from_dict
from .validation import normalize_state_slot


class EventStateStore:
    """Keeps raw episodes immutable while allowing state metadata to evolve."""

    SCHEMA_VERSION = 6
    SEMANTIC_VERSION = "2.9"
    LEGACY_INLINE_SCHEMA_VERSIONS = frozenset({4, 5})
    EMBEDDING_ARTIFACT_NAMES = (
        "episode_embeddings",
        "turn_embeddings",
        "claim_embeddings",
        "claim_slot_embeddings",
    )
    EMBEDDING_ARTIFACT_DTYPE = np.dtype(np.float64)

    def __init__(self, context_id: Optional[Any] = None) -> None:
        self.context_id = context_id
        self.episodes: Dict[str, Episode] = {}
        self.claims: Dict[str, Claim] = {}
        self.operations: List[StateOperation] = []
        self.edges: List[Dict[str, Any]] = []
        self.episode_embeddings: Dict[str, List[float]] = {}
        # This is an index over immutable episode evidence, not a semantic
        # memory layer. Keys are stable across exported snapshots.
        self.turn_embeddings: Dict[str, List[float]] = {}
        self.turn_metadata: Dict[str, Dict[str, Any]] = {}
        self.claim_embeddings: Dict[str, List[float]] = {}
        self.claim_slot_embeddings: Dict[str, List[float]] = {}
        # Rebuildable provenance index: a raw turn can support several events.
        self.turn_temporal_spans: Dict[tuple[str, Any], List[Dict[str, str]]] = {}
        self.episode_temporal_spans: Dict[str, List[Dict[str, str]]] = {}
        # Query-only indexes are deliberately absent from exported snapshots.
        # They are immutable after lazy construction and safe to share between
        # query workers that use the same restored store.
        self._turn_bm25_caches: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        self._turn_bm25_lock = threading.Lock()

    @staticmethod
    def stable_id(prefix: str, value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        return f"{prefix}{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"

    def add_episode(
        self,
        episode: Episode,
        embedding: List[float],
        turn_embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        conflicting_episode = next(
            (
                existing.episode_id
                for existing in self.episodes.values()
                if episode.source_session_id is not None
                and existing.source_session_id == episode.source_session_id
            ),
            None,
        )
        if conflicting_episode is not None:
            raise ValueError(
                "Event-State source identity collision: source UID "
                f"{episode.source_session_id!r} already belongs to episode "
                f"{conflicting_episode!r}"
            )
        if episode.episode_id not in self.episodes:
            self.episodes[episode.episode_id] = episode
            self.episode_embeddings[episode.episode_id] = list(embedding)
            self._index_episode_turns(episode, turn_embeddings)

    def turn_key(self, episode_id: str, turn_index: int) -> str:
        """Return the stable identifier for one archived source turn."""
        return self.stable_id("T", [episode_id, int(turn_index)])

    def _index_episode_turns(
        self,
        episode: Episode,
        vectors: Optional[List[List[float]]] = None,
    ) -> None:
        for turn_index, turn in enumerate(episode.turn_evidence):
            key = self.turn_key(episode.episode_id, turn_index)
            self.turn_metadata[key] = {
                "episode_id": episode.episode_id,
                "source_session_id": turn.source_session_id
                if turn.source_session_id is not None else episode.source_session_id,
                "source_turn_id": turn.turn_id,
                "source_session_index": turn.source_session_index
                if turn.source_session_index is not None else episode.source_session_index,
                "source_turn_index": turn_index,
            }
            if vectors is not None and turn_index < len(vectors):
                self.turn_embeddings[key] = list(vectors[turn_index])

    def rebuild_turn_metadata(self) -> None:
        """Reconstruct index metadata for snapshots that predate turn search."""
        for episode in self.episodes.values():
            self._index_episode_turns(episode)
        self.rebuild_temporal_indexes()

    def rebuild_temporal_indexes(self) -> None:
        """Derive event-time evidence from claim provenance, never record time."""
        self.turn_temporal_spans, self.episode_temporal_spans = {}, {}
        for claim in self.claims.values():
            if not claim.event_time_start or not claim.event_time_end:
                continue
            span = {"start": claim.event_time_start, "end": claim.event_time_end,
                    "precision": claim.event_time_precision}
            for ref in claim.evidence:
                self.episode_temporal_spans.setdefault(ref.episode_id, []).append(span)
                for turn_id in ref.source_turn_ids:
                    self.turn_temporal_spans.setdefault((ref.episode_id, turn_id), []).append(span)

    def turn_for_key(self, key: str) -> tuple[Optional[Episode], Optional[Any]]:
        metadata = self.turn_metadata.get(key, {})
        episode = self.episodes.get(metadata.get("episode_id"))
        index = metadata.get("source_turn_index")
        if episode is None or not isinstance(index, int) or not 0 <= index < len(episode.turn_evidence):
            return None, None
        return episode, episode.turn_evidence[index]

    def turn_bm25_cache(
        self, *, unit: str = "turn", min_chars: int = 24, max_chunks_per_turn: int = 8,
    ) -> Dict[str, Any]:
        """Build a compact, runtime-only sparse index over immutable turns.

        Index rows retain only parent turn provenance and tokens.  In
        particular, sentence chunks never become Event-State objects or dense
        embeddings, and raw source text is not copied into the cache.
        """
        key = (unit, max(0, int(min_chars)), max(1, int(max_chunks_per_turn)))
        cached = self._turn_bm25_caches.get(key)
        if cached is not None:
            return cached
        with self._turn_bm25_lock:
            cached = self._turn_bm25_caches.get(key)
            if cached is not None:
                return cached
            documents, document_frequency, total_length = [], {}, 0
            for turn_key in sorted(self.turn_embeddings):
                episode, turn = self.turn_for_key(turn_key)
                if episode is None or turn is None:
                    continue
                text = f"{turn.speaker} {turn.text} {turn.image_caption or ''}"
                chunks = [text]
                if unit == "sentence_chunk":
                    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]
                    chunks = [piece for piece in pieces if len(piece) >= max(0, int(min_chars))]
                    if not chunks:
                        chunks = [text]
                    chunks = chunks[:max(1, int(max_chunks_per_turn))]
                for chunk_index, chunk in enumerate(chunks):
                    tokens = tuple(re.findall(r"[a-z0-9]+", chunk.casefold()))
                    if not tokens:
                        continue
                    documents.append((turn_key, chunk_index, tokens))
                    total_length += len(tokens)
                    for token in set(tokens):
                        document_frequency[token] = document_frequency.get(token, 0) + 1
            cached = {
                "documents": tuple(documents), "document_frequency": document_frequency,
                "average_length": total_length / len(documents) if documents else 0.0,
            }
            self._turn_bm25_caches[key] = cached
            return cached

    def add_claim(self, claim: Claim, embedding: List[float], slot_embedding: Optional[List[float]] = None) -> None:
        if claim.persistence == "history":
            claim.state_slot = None
        elif claim.persistence == "state" and not claim.state_slot:
            claim.state_slot = normalize_state_slot(claim.predicate)
        self.claims[claim.claim_id] = claim
        self.claim_embeddings[claim.claim_id] = list(embedding)
        if claim.persistence == "state":
            self.claim_slot_embeddings[claim.claim_id] = list(slot_embedding if slot_embedding is not None else embedding)
        for evidence in claim.evidence:
            self.attach_claim_evidence(claim.claim_id, evidence, evidence.support_type)
        self.rebuild_temporal_indexes()

    def attach_claim_evidence(self, claim_id: str, evidence: Any, support_type: str = "origin") -> bool:
        """Attach one provenance reference and keep the heterogeneous graph in sync."""
        claim = self.claims[claim_id]
        normalized = replace(evidence, support_type=support_type)
        for existing in claim.evidence:
            if existing.episode_id == normalized.episode_id and existing.source_turn_ids == normalized.source_turn_ids:
                self.add_edge(claim_id, normalized.episode_id, "CLAIM_SUPPORTED_BY_EPISODE")
                self.add_edge(normalized.episode_id, claim_id, "EPISODE_SUPPORTS_CLAIM")
                return False
        claim.evidence.append(normalized)
        self.add_edge(claim_id, normalized.episode_id, "CLAIM_SUPPORTED_BY_EPISODE")
        self.add_edge(normalized.episode_id, claim_id, "EPISODE_SUPPORTS_CLAIM")
        return True

    def validate_state_invariants(self) -> List[str]:
        errors: List[str] = []
        node_ids = set(self.claims) | set(self.episodes)
        seen_edges = set()
        version_edges = []
        for edge in self.edges:
            key = tuple(edge.get(field) for field in ("source_id", "target_id", "relation_type", "weight"))
            if key in seen_edges:
                errors.append(f"duplicate edge: {key}")
            seen_edges.add(key)
            if edge.get("source_id") not in node_ids or edge.get("target_id") not in node_ids:
                errors.append(f"edge references missing node: {edge}")
            if edge.get("relation_type") in {"SUPERSEDES", "REFINES"}:
                version_edges.append(edge)
        for claim_id, claim in self.claims.items():
            if claim_id not in self.claim_embeddings:
                errors.append(f"missing claim embedding: {claim_id}")
            if claim.persistence == "state" and claim_id not in self.claim_slot_embeddings:
                errors.append(f"missing claim slot embedding: {claim_id}")
            for ref in claim.evidence:
                if ref.episode_id not in self.episodes:
                    errors.append(f"missing evidence episode: {claim_id}->{ref.episode_id}")
        for episode_id in self.episodes:
            if episode_id not in self.episode_embeddings:
                errors.append(f"missing episode embedding: {episode_id}")
        for key, metadata in self.turn_metadata.items():
            if metadata.get("episode_id") not in self.episodes:
                errors.append(f"turn index references missing episode: {key}")
        adjacency: Dict[str, List[str]] = {}
        for edge in version_edges:
            adjacency.setdefault(edge["source_id"], []).append(edge["target_id"])
        visiting, visited = set(), set()

        def visit(node: str) -> None:
            if node in visiting:
                errors.append(f"version cycle detected at: {node}")
                return
            if node in visited:
                return
            visiting.add(node)
            for target in adjacency.get(node, []):
                visit(target)
            visiting.remove(node)
            visited.add(node)

        for node in adjacency:
            visit(node)

        components: Dict[str, set[str]] = {}
        for edge in version_edges:
            components.setdefault(edge["source_id"], set()).update((edge["source_id"], edge["target_id"]))
            components.setdefault(edge["target_id"], set()).update((edge["source_id"], edge["target_id"]))
        for members in components.values():
            active = [claim_id for claim_id in members if claim_id in self.claims and self.claims[claim_id].persistence == "state" and self.claims[claim_id].status == "active"]
            if len(active) > 1 and not any(self.claims[claim_id].status == "contested" for claim_id in members if claim_id in self.claims):
                errors.append(f"version chain has multiple active terminals: {sorted(active)}")

        for operation in self.operations:
            if operation.operation != "CORROBORATE":
                continue
            matched = self.claims.get(operation.matched_claim_id)
            prior_episode_ids = {ref.episode_id for ref in matched.evidence if ref.episode_id != operation.episode_id} if matched else set()
            if matched and not prior_episode_ids:
                errors.append(f"same-session corroboration remains: {operation.operation_id}")

        for edge in self.edges:
            if edge.get("relation_type") in {"SUPERSEDES", "REFINES"}:
                newer = self.claims.get(edge.get("source_id"))
                older = self.claims.get(edge.get("target_id"))
                if newer and older and newer.status == "active" and older.status == "active":
                    errors.append(f"version target remains active: {edge}")
        return errors

    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0) -> None:
        edge = {"source_id": source_id, "target_id": target_id, "relation_type": relation_type, "weight": weight}
        if edge not in self.edges:
            self.edges.append(edge)

    def add_relation_pair(self, newer: str, older: str, relation: str) -> None:
        inverse = {"SUPERSEDES": "SUPERSEDED_BY", "REFINES": "REFINED_BY"}.get(relation)
        self.add_edge(newer, older, relation)
        if inverse:
            self.add_edge(older, newer, inverse)
        elif relation == "CONFLICTS_WITH":
            self.add_edge(older, newer, relation)

    def claim_counts(self) -> Dict[str, int]:
        return {
            "active_claim_count": sum(item.status == "active" for item in self.claims.values()),
            "historical_claim_count": sum(item.status in {"superseded", "historical"} for item in self.claims.values()),
            "superseded_claim_count": sum(item.status == "superseded" for item in self.claims.values()),
            "refined_claim_count": sum(item.status == "refined" for item in self.claims.values()),
            "contested_claim_count": sum(item.status == "contested" for item in self.claims.values()),
            "standalone_claim_count": sum(item.status == "standalone" for item in self.claims.values()),
            "total_claim_count": len(self.claims),
            "total_episode_count": len(self.episodes),
            "distinct_episode_source_id_count": len(
                {
                    episode.source_session_id
                    for episode in self.episodes.values()
                    if episode.source_session_id is not None
                }
            ),
            "duplicate_episode_source_id_count": sum(
                episode.source_session_id is not None for episode in self.episodes.values()
            ) - len(
                {
                    episode.source_session_id
                    for episode in self.episodes.values()
                    if episode.source_session_id is not None
                }
            ),
            "claims_with_valid_time_text": sum(bool(claim.valid_time_text) for claim in self.claims.values()),
            "claims_with_normalized_event_time": sum(bool(claim.event_time_start and claim.event_time_end) for claim in self.claims.values()),
            "event_time_exact_count": sum(claim.event_time_precision == "exact" and bool(claim.event_time_start) for claim in self.claims.values()),
            "event_time_bounded_count": sum(claim.event_time_precision == "bounded" and bool(claim.event_time_start) for claim in self.claims.values()),
            "event_time_approximate_count": sum(claim.event_time_precision == "approximate" and bool(claim.event_time_start) for claim in self.claims.values()),
            "event_time_unknown_count": sum(claim.event_time_precision == "unknown" or not claim.event_time_start for claim in self.claims.values()),
            "turns_with_event_time_spans": len(self.turn_temporal_spans),
            "episodes_with_event_time_spans": len(self.episode_temporal_spans),
        }

    @classmethod
    def _embedding_artifact_metadata(
        cls,
        name: str,
        embeddings: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Describe a deterministic dense index without materializing its matrix."""
        if any(not isinstance(identifier, str) for identifier in embeddings):
            raise ValueError(f"Event-State {name} has a non-string embedding ID")
        ids = sorted(embeddings)
        if not ids:
            return {
                "ids": ids,
                "dtype": str(cls.EMBEDDING_ARTIFACT_DTYPE),
                "shape": [0, 0],
            }

        width: Optional[int] = None
        for identifier in ids:
            try:
                vector = np.asarray(
                    embeddings[identifier], dtype=cls.EMBEDDING_ARTIFACT_DTYPE
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Event-State {name} cannot be represented as a dense matrix"
                ) from exc
            if vector.ndim != 1:
                raise ValueError(
                    f"Event-State {name} cannot be represented as a dense matrix"
                )
            if width is None:
                width = vector.shape[0]
            elif vector.shape[0] != width:
                raise ValueError(
                    f"Event-State {name} cannot be represented as a dense matrix"
                )
        return {
            "ids": ids,
            "dtype": str(cls.EMBEDDING_ARTIFACT_DTYPE),
            "shape": [len(ids), width or 0],
        }

    @classmethod
    def _embedding_artifact_matrix(
        cls,
        name: str,
        embeddings: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Return one dense index with a deterministic matrix and row mapping."""
        metadata = cls._embedding_artifact_metadata(name, embeddings)
        ids = metadata["ids"]
        if ids:
            values = np.asarray(
                [embeddings[identifier] for identifier in ids],
                dtype=cls.EMBEDDING_ARTIFACT_DTYPE,
            )
        else:
            values = np.empty(tuple(metadata["shape"]), dtype=cls.EMBEDDING_ARTIFACT_DTYPE)
        return {**metadata, "values": values}

    def export_embedding_artifacts(self) -> Dict[str, Dict[str, Any]]:
        """Export derived dense indexes separately from canonical JSON state."""
        return {
            "episode_embeddings": self._embedding_artifact_matrix(
                "episode_embeddings", self.episode_embeddings
            ),
            "turn_embeddings": self._embedding_artifact_matrix(
                "turn_embeddings", self.turn_embeddings
            ),
            "claim_embeddings": self._embedding_artifact_matrix(
                "claim_embeddings", self.claim_embeddings
            ),
            "claim_slot_embeddings": self._embedding_artifact_matrix(
                "claim_slot_embeddings", self.claim_slot_embeddings
            ),
        }

    def export(self) -> Dict[str, Any]:
        """Export canonical Event-State memory without inline dense vectors."""
        artifact_metadata = {
            "episode_embeddings": self._embedding_artifact_metadata(
                "episode_embeddings", self.episode_embeddings
            ),
            "turn_embeddings": self._embedding_artifact_metadata(
                "turn_embeddings", self.turn_embeddings
            ),
            "claim_embeddings": self._embedding_artifact_metadata(
                "claim_embeddings", self.claim_embeddings
            ),
            "claim_slot_embeddings": self._embedding_artifact_metadata(
                "claim_slot_embeddings", self.claim_slot_embeddings
            ),
        }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "semantic_version": self.SEMANTIC_VERSION,
            "method": "event_state",
            "context_id": self.context_id,
            "episodes": [asdict(item) for item in self.episodes.values()],
            "claims": [asdict(item) for item in self.claims.values()],
            "state_operations": [asdict(item) for item in self.operations],
            "edges": self.edges,
            "turn_metadata": self.turn_metadata,
            "embedding_artifacts": {
                name: {
                    "ids": descriptor["ids"],
                    "dtype": descriptor["dtype"],
                    "shape": descriptor["shape"],
                }
                for name, descriptor in artifact_metadata.items()
            },
        }

    @classmethod
    def _restore_embedding_artifacts(
        cls,
        store: "EventStateStore",
        state: Dict[str, Any],
    ) -> None:
        """Restore schema-v6 dense indexes after sidecars have been validated."""
        artifacts = state.get("embedding_artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(
            cls.EMBEDDING_ARTIFACT_NAMES
        ):
            raise ValueError("Event-State schema v6 embedding artifact metadata is incomplete")

        expected_ids = {
            "episode_embeddings": set(store.episodes),
            "turn_embeddings": set(store.turn_metadata),
            "claim_embeddings": set(store.claims),
            "claim_slot_embeddings": {
                claim_id
                for claim_id, claim in store.claims.items()
                if claim.persistence == "state"
            },
        }
        restored: Dict[str, Dict[str, List[float]]] = {}
        for name in cls.EMBEDDING_ARTIFACT_NAMES:
            descriptor = artifacts.get(name)
            if not isinstance(descriptor, dict):
                raise ValueError(
                    f"Event-State schema v6 embedding artifact {name} is invalid"
                )
            ids = descriptor.get("ids")
            values = descriptor.get("values")
            if (
                not isinstance(ids, list)
                or any(not isinstance(identifier, str) for identifier in ids)
                or len(set(ids)) != len(ids)
                or ids != sorted(ids)
                or values is None
            ):
                raise ValueError(
                    f"Event-State schema v6 embedding artifact {name} is invalid"
                )
            try:
                matrix = np.asarray(values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Event-State schema v6 embedding artifact {name} is invalid"
                ) from exc
            if (
                matrix.ndim != 2
                or matrix.dtype != cls.EMBEDDING_ARTIFACT_DTYPE
                or descriptor.get("dtype") != str(cls.EMBEDDING_ARTIFACT_DTYPE)
                or str(matrix.dtype) != descriptor.get("dtype")
                or list(matrix.shape) != descriptor.get("shape")
                or matrix.shape[0] != len(ids)
                or set(ids) != expected_ids[name]
            ):
                raise ValueError(
                    f"Event-State schema v6 embedding artifact {name} is inconsistent"
                )
            restored[name] = {
                identifier: list(matrix[row].tolist())
                for row, identifier in enumerate(ids)
            }

        store.episode_embeddings = restored["episode_embeddings"]
        store.turn_embeddings = restored["turn_embeddings"]
        store.claim_embeddings = restored["claim_embeddings"]
        store.claim_slot_embeddings = restored["claim_slot_embeddings"]

    @classmethod
    def from_export(cls, state: Dict[str, Any]) -> "EventStateStore":
        if state.get("method") != "event_state":
            raise ValueError("Not an Event-State Hybrid Memory snapshot")
        schema_version = state.get("schema_version")
        if schema_version not in {*cls.LEGACY_INLINE_SCHEMA_VERSIONS, cls.SCHEMA_VERSION}:
            raise ValueError(
                f"Event-State snapshot schema v{state.get('schema_version')} is incompatible with schema v{cls.SCHEMA_VERSION}; rebuild the memory snapshot."
            )
        if state.get("semantic_version") != cls.SEMANTIC_VERSION:
            raise ValueError("Event-State snapshot semantic version is incompatible; rebuild the memory snapshot.")
        if schema_version == cls.SCHEMA_VERSION and any(
            name in state for name in cls.EMBEDDING_ARTIFACT_NAMES
        ):
            raise ValueError(
                "Event-State schema v6 snapshots must not include inline embedding maps"
            )
        store = cls(state.get("context_id"))
        store.episodes = {item["episode_id"]: episode_from_dict(item) for item in state.get("episodes", [])}
        store.claims = {item["claim_id"]: claim_from_dict(item) for item in state.get("claims", [])}
        store.operations = [StateOperation(**item) for item in state.get("state_operations", [])]
        store.edges = list(state.get("edges", []))
        store.episode_embeddings = {key: list(value) for key, value in state.get("episode_embeddings", {}).items()}
        store.turn_embeddings = {key: list(value) for key, value in state.get("turn_embeddings", {}).items()}
        store.turn_metadata = {
            key: dict(value) for key, value in state.get("turn_metadata", {}).items()
            if isinstance(value, dict)
        }
        store.rebuild_turn_metadata()
        if schema_version == cls.SCHEMA_VERSION:
            cls._restore_embedding_artifacts(store, state)
        else:
            store.claim_embeddings = {key: list(value) for key, value in state.get("claim_embeddings", {}).items()}
            store.claim_slot_embeddings = {key: list(value) for key, value in state.get("claim_slot_embeddings", {}).items()}
        return store
