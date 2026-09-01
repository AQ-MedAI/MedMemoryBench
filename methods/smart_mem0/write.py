"""Transactional write lifecycle and frozen-store loading."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from methods.base import MemoryBuildResult

from .contracts import MEMORY_WRITE_SCHEMA_VERSION, MemoryWriteContext, VALID_RELATIONS


class WriteLifecycleMixin:
    """Coordinates capture and commit with session-level rollback."""

    MEMORY_SNAPSHOT_VERSION = MEMORY_WRITE_SCHEMA_VERSION

    def export_memory_state(self, context_id: Optional[int] = None) -> Dict[str, Any]:
        """Return the complete durable state needed to reproduce retrieval."""
        del context_id
        return {
            "snapshot_version": self.MEMORY_SNAPSHOT_VERSION,
            "method": "smart_mem0",
            "memory_health": self._memory_health_stats(),
            "memories": self._snapshot(self._memories),
            "relations": self._snapshot(self._relations),
            "evidence": self._snapshot(self._evidence),
            "memory_seq": int(self._memory_seq),
            "evidence_seq": int(self._evidence_seq),
            "session_seq": int(self._session_seq),
            "is_initialized": bool(self._is_initialized),
        }

    def import_memory_state(
        self, data: Dict[str, Any], context_id: Optional[int] = None
    ) -> None:
        """Replace runtime memory with one validated SmartMem0 snapshot."""
        del context_id
        if not isinstance(data, dict):
            raise ValueError("SmartMem0 snapshot must be a JSON object")
        if data.get("snapshot_version") != self.MEMORY_SNAPSHOT_VERSION:
            raise ValueError("Unsupported SmartMem0 snapshot version")
        if data.get("method") != "smart_mem0":
            raise ValueError("Snapshot belongs to a different memory method")
        for field in ("memories", "relations", "evidence"):
            if not isinstance(data.get(field), list):
                raise ValueError(f"SmartMem0 snapshot field '{field}' must be a list")

        self._load_store_data(data)
        # Raw input chunks duplicate evidence and are not part of retrieval state.
        self._memory_chunks = []
        self._memory_seq = max(int(data.get("memory_seq", 0)), len(self._memories))
        self._evidence_seq = max(int(data.get("evidence_seq", 0)), len(self._evidence))
        self._session_seq = max(
            int(data.get("session_seq", 0)),
            self._session_seq,
        )
        self._is_initialized = bool(data.get("is_initialized", True))
        self._loaded_frozen = True

    def _load_frozen_store(self, path: Path) -> None:
        self._load_store_data(json.loads(path.read_text(encoding="utf-8")))

    def _load_store_data(self, data: Dict[str, Any]) -> None:
        """Load canonical memories and rebuild all derived runtime views."""
        self._bm25 = self._embedding_matrix = None
        self._embedding_cache = {}
        self._memories = []
        for index, raw in enumerate(
            data.get("memories") or data.get("cards") or [], start=1
        ):
            normalized = self._normalise_memory(raw)
            if not normalized:
                continue
            self._memories.append(
                {
                    "id": str(raw.get("id") or f"m_{index}"),
                    "claim": normalized["claim"],
                    "kind": normalized["kind"],
                    "entities": normalized["entities"],
                    "subject": normalized["subject"],
                    "scope": normalized["scope"],
                    "state_key": normalized["state_key"],
                    "object_anchor": normalized["object_anchor"],
                    "scope_entities": normalized["scope_entities"],
                    "value": normalized["value"],
                    "stance": normalized["stance"],
                    "verbatim_value": normalized["verbatim_value"],
                    "event_time": str(
                        raw.get("event_time") or normalized["event_time"]
                    ),
                    "time_expression": normalized["time_expression"],
                    "document_time": str(
                        raw.get("document_time") or raw.get("timestamp") or ""
                    ),
                    "origin_document_time": str(
                        raw.get("origin_document_time")
                        or raw.get("document_time")
                        or raw.get("timestamp")
                        or ""
                    ),
                    "assertion_mode": normalized["assertion_mode"],
                    "origin_memory_id": normalized["origin_memory_id"],
                    "planning_tags": list(normalized.get("planning_tags") or []),
                    "decision_salience": float(
                        normalized.get("decision_salience", 0.0) or 0.0
                    ),
                    "evidence_ids": list(
                        raw.get("evidence_ids") or raw.get("source_turn_ids") or []
                    ),
                    "source_speakers": list(raw.get("source_speakers") or []),
                    "confidence": normalized["confidence"],
                    "session_idx": int(raw.get("session_idx", 0)),
                }
            )
        self._relations = []
        for raw_relation in data.get("relations", []):
            relation_type = str(raw_relation.get("type") or "").upper()
            if relation_type not in VALID_RELATIONS:
                continue
            relation = self._snapshot(raw_relation)
            relation["type"] = relation_type
            relation["provenance_evidence_ids"] = list(
                relation.get("provenance_evidence_ids") or []
            )
            self._relations.append(relation)
        self._evidence = list(data.get("evidence") or [])
        self._memory_seq, self._evidence_seq = len(self._memories), len(self._evidence)
        self._session_seq = max(
            [int(memory.get("session_idx", 0)) for memory in self._memories]
            + [int(evidence.get("session_idx", 0)) for evidence in self._evidence]
            + [0]
        )
        self._rebuild_belief_view()
        self._index_dirty = True
        self._refresh_index()

    def memorize(
        self,
        session_content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> MemoryBuildResult:
        if self.frozen_memory_path and not self._loaded_frozen:
            path = Path(self.frozen_memory_path)
            if path.exists():
                self._load_frozen_store(path)
                self._loaded_frozen = True
                return MemoryBuildResult(
                    method="smart_mem0",
                    action="load_frozen",
                    input_content=session_content,
                    stored_content=session_content,
                    chunk_count=0,
                    extra={"frozen": True, "total_memories": len(self._memories)},
                )
        turns, parsed_time = self._parse_turns(session_content)
        if not turns and session_content.strip():
            turns = [
                {
                    "speaker": "unknown",
                    "raw_text": session_content.strip(),
                    "turn_idx": 0,
                }
            ]
        document_time = str(
            (metadata or {}).get("timestamp") or parsed_time or "UNKNOWN"
        )
        next_session = self._session_seq + 1
        staged_evidence, turn_map = self._stage_evidence(
            turns, document_time, next_session
        )
        windows = self._split_write_windows(turns)
        context = MemoryWriteContext()
        self._write_context = context
        self._capture_parse_stats = {
            "malformed_windows": 0,
            "salvaged_items": 0,
            "discarded_items": 0,
        }
        responses: List[str] = []
        causal_links: List[Dict[str, Any]] = []
        write_context_trace: List[Dict[str, Any]] = []
        try:
            if self.enable_memory_write:
                for window_index, window in enumerate(windows):
                    first_turn = window[0]["turn_idx"]
                    previous_turn = turns[first_turn - 1] if first_turn > 0 else None
                    continuity_turn = (
                        previous_turn
                        if self.write_context_mode in {"window", "full"}
                        else None
                    )
                    prior, provisional_view = self._update_write_context(
                        context,
                        window,
                        continuity_turn,
                        document_time,
                    )
                    prior_payload = self._write_memory_payload(prior)
                    provisional_payload = self._write_memory_payload(provisional_view)
                    write_context_trace.append(
                        {
                            "window_index": window_index,
                            "focal_turn_ids": [turn["turn_idx"] for turn in window],
                            "focal_tokens": sum(
                                len(self._tokenizer.encode(self._format_turn(turn)))
                                for turn in window
                            ),
                            "local_context_turn_ids": (
                                [continuity_turn["turn_idx"]] if continuity_turn else []
                            ),
                            "prior_belief_ids": [
                                memory["id"] for memory in prior if memory.get("id")
                            ],
                            "prior_belief_tokens": len(
                                self._tokenizer.encode(prior_payload)
                            ),
                            "provisional_count": len(provisional_view),
                            "provisional_tokens": len(
                                self._tokenizer.encode(provisional_payload)
                            ),
                        }
                    )
                    response_text, window_memories, window_links = (
                        self._extract_write_window(
                            window,
                            continuity_turn,
                            document_time,
                            prior,
                            provisional_view,
                        )
                    )
                    responses.append(response_text)
                    write_context_trace[-1]["extracted_count"] = len(window_memories)
                    local_to_session = {
                        local_index: self._merge_provisional_memory(
                            context.provisional_memories, memory
                        )
                        for local_index, memory in enumerate(window_memories)
                    }
                    for link in window_links:
                        cause = local_to_session.get(link["cause_index"])
                        effect = local_to_session.get(link["effect_index"])
                        if cause is None or effect is None or cause == effect:
                            continue
                        causal_links.append(
                            {**link, "cause_index": cause, "effect_index": effect}
                        )

            old_counts = (
                len(self._memories),
                len(self._relations),
                len(self._evidence),
                self._memory_seq,
                self._evidence_seq,
                self._session_seq,
            )
            old_write_stats = self._snapshot(self._last_write_stats)
            try:
                self._session_seq = next_session
                self._evidence.extend(staged_evidence)
                self._evidence_seq += len(staged_evidence)
                if context.provisional_memories:
                    added = self._add_memories(
                        context.provisional_memories,
                        causal_links,
                        document_time,
                        turn_map,
                    )
                    self._refresh_index()
                else:
                    added = []
                    self._last_write_stats = {
                        "skipped_recaps": 0,
                        "committed_memories": 0,
                        "promoted_state_updates": 0,
                        "reused_state_identities": 0,
                    }
            except Exception:
                (
                    memory_count,
                    relation_count,
                    evidence_count,
                    memory_seq,
                    evidence_seq,
                    session_seq,
                ) = old_counts
                del self._memories[memory_count:]
                del self._relations[relation_count:]
                del self._evidence[evidence_count:]
                self._memory_seq, self._evidence_seq, self._session_seq = (
                    memory_seq,
                    evidence_seq,
                    session_seq,
                )
                self._last_write_stats = old_write_stats
                valid_ids = {memory["id"] for memory in self._memories}
                self._embedding_cache = {
                    memory_id: vector
                    for memory_id, vector in self._embedding_cache.items()
                    if memory_id in valid_ids
                }
                self._bm25 = self._embedding_matrix = None
                self._rebuild_belief_view()
                self._index_dirty = True
                raise
        finally:
            context.clear()
            self._write_context = None

        self._memory_chunks.append(session_content)
        self._is_initialized = True
        extraction_result = (
            responses[0]
            if len(responses) == 1
            else json.dumps(
                {"window_outputs": responses},
                ensure_ascii=False,
            )
        )
        return MemoryBuildResult(
            extraction_result=extraction_result,
            method="smart_mem0",
            action="add_to_memory",
            input_content=session_content,
            stored_content=session_content,
            memory_entries=self._snapshot(added),
            all_passages=self._snapshot(
                [e for e in self._evidence if e["session_idx"] == self._session_seq]
            ),
            chunk_count=len(added),
            extra={
                "schema": "compact_memory",
                "inserted_count": len(added),
                "total_memories": len(self._memories),
                "total_relations": len(self._relations),
                "raw_evidence_stored_cold": True,
                "write_windows": len(windows),
                "write_context_mode": self.write_context_mode,
                "write_prior_belief_limit": self.write_prior_belief_limit,
                "write_provisional_limit": self.write_provisional_limit,
                "write_context_trace": write_context_trace,
                "write_context_persisted": False,
                "skipped_recaps": self._last_write_stats.get("skipped_recaps", 0),
                "promoted_state_updates": self._last_write_stats.get(
                    "promoted_state_updates", 0
                ),
                "reused_state_identities": self._last_write_stats.get(
                    "reused_state_identities", 0
                ),
                "capture_parse": self._snapshot(self._capture_parse_stats),
                "effective_runtime_config": self._effective_runtime_config(),
                "memory_health": self._memory_health_stats(),
            },
        )

    def consolidate_offline(self) -> None:
        self._rebuild_belief_view()
        self._index_dirty = True
        self._refresh_index()
