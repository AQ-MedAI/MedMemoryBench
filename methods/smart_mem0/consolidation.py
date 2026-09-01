"""ADD-only ledger consolidation, state heads, and typed relations."""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from .contracts import STATE_LIKE_KINDS, VALID_RELATIONS
from .core import CoreMemoryMixin
from .prompts import CONSOLIDATION_PROMPT


class ConsolidationMixin:
    """Commits atomic memories and derives non-destructive belief views."""

    @staticmethod
    def _merge_provisional_memory(
        provisional: List[Dict[str, Any]],
        memory: Dict[str, Any],
    ) -> int:
        key = (memory["kind"], memory["stance"], memory["claim"].casefold())
        for index, existing in enumerate(provisional):
            existing_key = (
                existing["kind"],
                existing["stance"],
                existing["claim"].casefold(),
            )
            equivalent_state = False
            if (
                existing.get("kind") == memory.get("kind") == "STATE"
                and existing.get("stance") == memory.get("stance")
                and CoreMemoryMixin._state_identity(existing)
                and CoreMemoryMixin._state_identity(existing)
                == CoreMemoryMixin._state_identity(memory)
            ):
                existing_numbers = set(
                    re.findall(r"\d+(?:\.\d+)?", existing.get("value", ""))
                )
                new_numbers = set(re.findall(r"\d+(?:\.\d+)?", memory.get("value", "")))
                old_terms = set(CoreMemoryMixin._tokenize(existing.get("value", "")))
                new_terms = set(CoreMemoryMixin._tokenize(memory.get("value", "")))
                overlap = len(old_terms & new_terms) / max(
                    1, len(old_terms | new_terms)
                )
                equivalent_state = bool(
                    existing_numbers
                    and existing_numbers == new_numbers
                    and overlap >= 0.35
                ) or (
                    not existing_numbers
                    and not new_numbers
                    and CoreMemoryMixin._normalised_value(existing)
                    == CoreMemoryMixin._normalised_value(memory)
                )
            if existing_key != key and not equivalent_state:
                continue
            existing["source_turns"] = sorted(
                set(existing["source_turns"] + memory["source_turns"])
            )[:4]
            existing["confidence"] = max(existing["confidence"], memory["confidence"])
            existing["planning_tags"] = list(
                dict.fromkeys(
                    [
                        *existing.get("planning_tags", []),
                        *memory.get("planning_tags", []),
                    ]
                )
            )
            existing["decision_salience"] = max(
                float(existing.get("decision_salience", 0.0) or 0.0),
                float(memory.get("decision_salience", 0.0) or 0.0),
            )
            if (
                existing.get("assertion_mode") == "RECAP"
                and memory.get("assertion_mode") == "DIRECT"
            ):
                existing["assertion_mode"] = "DIRECT"
                existing["origin_memory_id"] = ""
            if (
                existing["event_time"] == "UNKNOWN"
                and memory["event_time"] != "UNKNOWN"
            ):
                existing["event_time"] = memory["event_time"]
                existing["time_expression"] = memory["time_expression"]
            return index
        provisional.append(memory)
        return len(provisional) - 1

    def _nearby_old_memories(
        self, new_memories: List[Dict[str, Any]], limit: int = 24
    ) -> List[Dict[str, Any]]:
        """Select old candidates fairly across all new memories in the session."""
        selected = {}
        by_id = {memory["id"]: memory for memory in self._memories}
        for memory in new_memories:
            origin = by_id.get(memory.get("origin_memory_id", ""))
            if origin:
                selected[origin["id"]] = self._snapshot(origin)

            # Keep recent exact-identity versions available to consolidation.
            # Ledger-order truncation previously favored the oldest versions.
            m_ident = self._state_identity(memory)
            m_lineage = self._state_lineage_identity(memory)
            if m_ident or m_lineage:
                identity_matches = sorted(
                    (
                        old_mem
                        for old_mem in self._memories
                        if (
                            (m_ident and self._state_identity(old_mem) == m_ident)
                            or (
                                m_lineage
                                and self._state_lineage_identity(old_mem) == m_lineage
                            )
                        )
                    ),
                    key=lambda old_mem: (
                        self._belief_status.get(old_mem["id"], "active")
                        in {"active", "refined", "conflicting"},
                        self._recency_date(old_mem),
                        old_mem.get("document_time", ""),
                        old_mem["id"],
                    ),
                    reverse=True,
                )
                for old_mem in identity_matches[:6]:
                    selected.setdefault(old_mem["id"], self._snapshot(old_mem))
        ranked_per_memory = [
            self._hybrid_search(memory["claim"], top_k=3) for memory in new_memories
        ]
        for rank in range(3):
            for candidates in ranked_per_memory:
                if rank < len(candidates):
                    old = candidates[rank]
                    selected.setdefault(old["id"], old)
                if len(selected) >= limit:
                    return list(selected.values())[:limit]
        return list(selected.values())[:limit]

    @staticmethod
    def _explicit_state_transition(memory: Dict[str, Any]) -> str:
        """Return a literal transition class without inferring broad intent."""
        if memory.get("kind") in {"PLAN", "PREFERENCE"}:
            return ""
        text = " ".join(
            str(memory.get(field) or "") for field in ("claim", "value")
        ).lower()
        # Hypothetical or recommended changes are plans, not observed heads.
        if re.search(
            r"\b(?:should|could|would|may|might|recommend(?:ed)?|advise[ds]?|"
            r"plan(?:s|ned)?|consider(?:s|ed)?)\b.{0,32}\b(?:start|stop|switch|"
            r"replace|increase|decrease|reduce|raise|lower|resume|restart)",
            text,
        ):
            return ""
        patterns = (
            (
                "STOP",
                r"\b(?:stopped|discontinued|ceased|quit|no longer (?:takes?|taking|uses?|using))\b",
            ),
            ("START", r"\b(?:started|began|initiated|commenced)\b"),
            ("RESUME", r"\b(?:resumed|restarted)\b"),
            (
                "REPLACE",
                r"\b(?:switched|replaced|changed\s+(?:from|to)|transitioned)\b",
            ),
            (
                "INCREASE",
                r"\b(?:increased|titrated up|went up|raised\s+(?:the\s+)?(?:dose|level|amount|value))\b",
            ),
            (
                "DECREASE",
                r"\b(?:decreased|reduced|titrated down|went down|lowered\s+(?:the\s+)?(?:dose|level|amount|value))\b",
            ),
        )
        return next(
            (name for name, pattern in patterns if re.search(pattern, text)), ""
        )

    def _explicit_state_replacement(
        self,
        newer: Dict[str, Any],
        older: Dict[str, Any],
    ) -> bool:
        """Return whether a SUPERSEDE edge is supported by observable change.

        Same identity plus different prose is insufficient: a state key can be
        broad and two observations can describe different facets of it. Only a
        literal transition, or a changed numeric measurement with a later
        source date, may create the deterministic replacement edge.
        """
        if self._state_identity(newer) != self._state_identity(older):
            return False
        if self._explicit_state_transition(newer):
            return True
        newer_text = self._memory_value(newer) or str(newer.get("claim") or "")
        older_text = self._memory_value(older) or str(older.get("claim") or "")
        newer_numbers = re.findall(r"\d+(?:\.\d+)?", newer_text)
        older_numbers = re.findall(r"\d+(?:\.\d+)?", older_text)
        if not newer_numbers or not older_numbers or newer_numbers == older_numbers:
            return False
        newer_date = self._parse_date(
            self._resolve_event_time(newer, newer.get("document_time", ""))
        ) or self._recency_date(newer)
        older_date = self._parse_date(
            self._resolve_event_time(older, older.get("document_time", ""))
        ) or self._recency_date(older)
        return bool(newer_date and older_date and newer_date >= older_date)

    def _state_predecessor(
        self,
        memory: Dict[str, Any],
        provisional_states: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find one unambiguous active state owner for a new state/update."""
        transition = self._explicit_state_transition(memory)
        if memory.get("assertion_mode") == "RECAP":
            return None
        if memory.get("kind") not in STATE_LIKE_KINDS and not transition:
            return None
        subject = self._canonical_identifier(memory.get("subject"), "patient")
        scope = self._canonical_scope(memory.get("scope"))
        anchor = self._canonical_identifier(memory.get("object_anchor"))
        key = self._canonical_state_key(memory.get("state_key") or "")
        text_terms = set(
            self._tokenize(
                " ".join(
                    [
                        str(memory.get("claim") or ""),
                        str(memory.get("value") or ""),
                        " ".join(memory.get("entities") or []),
                        " ".join(memory.get("scope_entities") or []),
                    ]
                )
            )
        )
        candidates = []
        for old in [*self._memories, *(provisional_states or [])]:
            if old.get("kind") not in STATE_LIKE_KINDS or not old.get("state_key"):
                continue
            if (
                old.get("id")
                and self._belief_status.get(old.get("id"), "active") == "superseded"
            ):
                continue
            if self._canonical_identifier(old.get("subject"), "patient") != subject:
                continue
            old_scope = self._canonical_scope(old.get("scope"))
            old_anchor = self._canonical_identifier(old.get("object_anchor"))
            if anchor and old_anchor and anchor != old_anchor:
                continue

            score = 0.0
            if scope == old_scope:
                score += 2.0
            elif "general" in {scope, old_scope}:
                score += 0.5
            if anchor and anchor == old_anchor:
                score += 4.0
            elif not anchor and old_anchor:
                anchor_terms = set(old_anchor.split("_"))
                if anchor_terms and anchor_terms.issubset(text_terms):
                    score += 3.0
                else:
                    continue
            elif not anchor and not old_anchor:
                # Without an object owner, require a strong key match below.
                score += 0.5
            old_key = self._canonical_state_key(old.get("state_key") or "")
            key_terms, old_key_terms = set(key.split("_")), set(old_key.split("_"))
            key_overlap = len(key_terms & old_key_terms)
            if key and key == old_key:
                score += 4.0
            elif key_overlap:
                score += 1.5 * key_overlap
            elif not transition:
                continue
            if transition:
                score += 1.0
            if key and key == old_key and anchor == old_anchor:
                score += 2.0
            if score >= 5.0:
                candidates.append((score, self._recency_date(old), old))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if (
            len(candidates) > 1
            and candidates[0][0] == candidates[1][0]
            and self._state_identity(candidates[0][2])
            != self._state_identity(candidates[1][2])
        ):
            return None
        return candidates[0][2]

    def _standardize_state_updates(
        self,
        new_memories: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Repair explicit state transitions before LLM consolidation."""
        stats = {"promoted_state_updates": 0, "reused_state_identities": 0}
        provisional_states: List[Dict[str, Any]] = []
        for memory in new_memories:
            predecessor = self._state_predecessor(memory, provisional_states)
            if not predecessor:
                if memory.get("kind") in STATE_LIKE_KINDS:
                    provisional_states.append(memory)
                continue
            previous_identity = self._state_identity(memory)
            if memory.get("kind") not in STATE_LIKE_KINDS:
                memory["kind"] = "STATE"
                stats["promoted_state_updates"] += 1
            for field in ("subject", "scope", "state_key", "object_anchor"):
                memory[field] = predecessor.get(field, memory.get(field, ""))
            if self._state_identity(memory) != previous_identity:
                stats["reused_state_identities"] += 1
            provisional_states.append(memory)
        return stats

    def _consolidation_relations(
        self,
        new_memories: List[Dict[str, Any]],
        old_memories: List[Dict[str, Any]],
        turn_map: Optional[Dict[int, str]] = None,
        document_time: str = "",
    ) -> List[Dict[str, Any]]:
        if not old_memories:
            return []
        fields = (
            "claim",
            "kind",
            "entities",
            "subject",
            "scope",
            "state_key",
            "object_anchor",
            "scope_entities",
            "value",
            "verbatim_value",
            "stance",
            "event_time",
            "assertion_mode",
            "origin_memory_id",
            "planning_tags",
            "decision_salience",
            "evidence_ids",
        )
        turn_map = turn_map or {}
        new_payload = []
        for index, item in enumerate(new_memories):
            payload = {k: item.get(k) for k in fields}
            payload["evidence_ids"] = [
                turn_map[turn]
                for turn in item.get("source_turns", [])
                if turn in turn_map
            ]
            # The raw extracted item's event_time is not yet resolved against
            # document_time (that happens later in _add_memories), so relative
            # expressions like "today"/"3 days ago" still read UNKNOWN here.
            # Surface document_time explicitly so the LLM (and the code-level
            # recency guard below) has a usable temporal anchor either way.
            payload["document_time"] = document_time
            new_payload.append({"id": f"new_{index}", **payload})
        old_payload = [
            {k: item.get(k) for k in ("id", *fields, "document_time")}
            for item in old_memories
        ]
        try:
            response = self._llm_client.chat(
                [
                    {
                        "role": "user",
                        "content": CONSOLIDATION_PROMPT.format(
                            new_memories=json.dumps(new_payload, ensure_ascii=False),
                            old_memories=json.dumps(old_payload, ensure_ascii=False),
                        ),
                    }
                ],
                temperature=0.0,
            )
            relations = self._parse_json(response.content).get("relations") or []
        except Exception:
            relations = []
        old_ids = {item["id"] for item in old_memories}
        old_by_id = {item["id"]: item for item in old_memories}
        valid = []
        for relation in relations:
            source, target = str(relation.get("source_id", "")), str(
                relation.get("target_id", "")
            )
            relation_type = str(relation.get("type", "")).upper()
            if (
                not re.fullmatch(r"new_\d+", source)
                or target not in old_ids
                or relation_type not in VALID_RELATIONS
            ):
                continue
            try:
                confidence = max(0.0, min(1.0, float(relation.get("confidence", 0.7))))
            except (TypeError, ValueError):
                confidence = 0.7
            if confidence >= 0.65:
                source_index = int(source.split("_", 1)[1])
                if source_index >= len(new_memories):
                    continue
                if new_memories[source_index].get(
                    "assertion_mode"
                ) == "RECAP" and relation_type in {"REFINE", "SUPERSEDE"}:
                    relation_type = "SUPPORT"

                # An older event cannot supersede a later event.  Do not emit
                # an ad-hoc relation type: unsupported relation types would
                # silently break state-head arbitration.
                #
                # new_memories[source_index] has not gone through
                # _resolve_event_time() yet at this point in the pipeline, so
                # relative time expressions ("today", "3 days ago") still show
                # up as UNKNOWN here and would silently skip this guard.
                # Resolve it the same way _add_memories will, and fall back to
                # document_time (turn recency) on both sides when no explicit
                # event date is available at all.
                resolved_source_time = self._resolve_event_time(
                    new_memories[source_index],
                    document_time,
                )
                s_time = (
                    self._parse_date(new_memories[source_index].get("event_time", ""))
                    or (
                        resolved_source_time
                        if resolved_source_time != "UNKNOWN"
                        else ""
                    )
                    or self._parse_date(document_time)
                )
                t_time = self._parse_date(
                    old_by_id[target].get("event_time", "")
                ) or self._parse_date(old_by_id[target].get("document_time", ""))
                if (
                    relation_type == "SUPERSEDE"
                    and s_time
                    and t_time
                    and s_time < t_time
                ):
                    continue
                provenance = [
                    str(value)
                    for value in relation.get("provenance_evidence_ids", [])
                    if str(value)
                ]
                if relation_type == "CAUSES":
                    source_evidence = set(
                        new_payload[source_index].get("evidence_ids") or []
                    )
                    target_evidence = set(old_by_id[target].get("evidence_ids") or [])
                    if not source_evidence or not target_evidence:
                        continue
                    allowed_provenance = source_evidence | target_evidence
                    provenance = [
                        value for value in provenance if value in allowed_provenance
                    ]
                    if not provenance:
                        continue
                valid.append(
                    {
                        "source_id": source,
                        "target_id": target,
                        "type": relation_type,
                        "confidence": confidence,
                        "provenance_evidence_ids": provenance,
                    }
                )
        return valid

    def _rebuild_belief_view(self) -> None:
        # Imported/legacy stores may contain relations produced before the
        # identity and provenance contracts were tightened. Remove those edges
        # before deriving status or heads, otherwise one invalid edge can poison
        # every subsequent query.
        by_id = {memory["id"]: memory for memory in self._memories}
        known_evidence = {str(item.get("id") or "") for item in self._evidence}
        clean_relations = []
        def relation_order(relation: Dict[str, Any]) -> tuple:
            target = by_id.get(relation.get("target_id"), {})
            return (
                self._recency_date(target),
                target.get("document_time", ""),
                float(relation.get("confidence", 0.0) or 0.0),
                str(target.get("id", "")),
            )

        supersede_sources = set()
        for relation in sorted(self._relations, key=relation_order, reverse=True):
            source = by_id.get(relation.get("source_id"))
            target = by_id.get(relation.get("target_id"))
            relation_type = str(relation.get("type") or "").upper()
            if not source or not target:
                continue
            if relation_type in {"REFINE", "SUPERSEDE", "CONFLICT"}:
                source_identity = self._state_identity(source)
                target_identity = self._state_identity(target)
                if not source_identity or source_identity != target_identity:
                    continue
            if (
                relation_type == "SUPERSEDE"
                and not self._explicit_state_replacement(source, target)
            ):
                # Legacy snapshots may contain an identity-valid but
                # semantically unsupported replacement edge. Drop it before
                # deriving status/head views so old stores cannot poison the
                # new read path.
                continue
            if relation_type == "SUPERSEDE":
                # One new memory represents one observation/update. It may
                # replace only the nearest predecessor; replacing every older
                # version creates a dense fan-out and makes trajectory
                # arbitration ambiguous. Older versions remain in the ledger.
                if relation.get("source_id") in supersede_sources:
                    continue
                supersede_sources.add(relation.get("source_id"))
            if relation_type == "CAUSES":
                provenance = set(relation.get("provenance_evidence_ids") or [])
                endpoint_evidence = set(source.get("evidence_ids") or []) | set(
                    target.get("evidence_ids") or []
                )
                focal_provenance = (
                    relation.get("provenance_kind") == "FOCAL_CAUSAL_TURN"
                )
                if (
                    not source.get("evidence_ids")
                    or not target.get("evidence_ids")
                    or not provenance
                    or any(item not in known_evidence for item in provenance)
                    or (not provenance.issubset(endpoint_evidence) and not focal_provenance)
                ):
                    continue
            clean_relations.append(relation)
        self._relations = clean_relations
        status = {memory["id"]: "active" for memory in self._memories}
        # Resolve replacement targets before applying weaker annotations. This
        # makes status derivation independent of relation ordering: a node that
        # is superseded remains historical even if it also participates in a
        # REFINE or CONFLICT edge.
        superseded_ids = {
            relation["target_id"]
            for relation in self._relations
            if relation["type"] == "SUPERSEDE"
        }
        for memory_id in superseded_ids:
            if memory_id in status:
                status[memory_id] = "superseded"
        for relation in self._relations:
            source, target, relation_type = (
                relation["source_id"],
                relation["target_id"],
                relation["type"],
            )
            if relation_type == "SUPERSEDE":
                # Status was initialized from all replacement targets above;
                # do not resurrect an intermediate version in a chain.
                continue
            elif relation_type == "REFINE":
                if status.get(source) != "superseded":
                    status[source] = "active"
                if status.get(target) != "superseded":
                    status[target] = "refined"
            elif relation_type == "CONFLICT":
                if status.get(source) == "active":
                    status[source] = "conflicting"
                if status.get(target) == "active":
                    status[target] = "conflicting"
        self._belief_status = status
        head_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for memory in self._memories:
            if not memory.get("state_key") or memory["kind"] not in STATE_LIKE_KINDS:
                continue
            if status.get(memory["id"]) not in {"active", "refined", "conflicting"}:
                continue
            identity = self._state_identity(memory)
            if identity:
                head_candidates[identity].append(memory)
        heads: Dict[str, List[str]] = {}
        for key, candidates in head_candidates.items():
            candidates.sort(
                key=lambda memory: (
                    self._recency_date(memory),
                    memory.get("document_time", ""),
                    memory["id"],
                ),
                reverse=True,
            )
            conflicting = [
                memory
                for memory in candidates
                if status.get(memory["id"]) == "conflicting"
            ]
            if conflicting:
                heads[key] = [memory["id"] for memory in conflicting]
                continue
            latest = self._recency_date(candidates[0]) or candidates[0].get(
                "document_time", ""
            )
            heads[key] = [
                memory["id"]
                for memory in candidates
                if (self._recency_date(memory) or memory.get("document_time", ""))
                == latest
            ]
        self._state_heads = heads
        active_ids = {
            memory["id"]
            for memory in self._memories
            if status.get(memory["id"]) in {"active", "conflicting"}
        }
        self._profile_pack = {
            "state_heads": self._snapshot(self._state_heads),
            "stable_facts": [
                memory["id"]
                for memory in self._memories
                if memory["id"] in active_ids and memory["kind"] == "FACT"
            ],
            "preferences": [
                memory_id
                for memory_ids in heads.values()
                for memory_id in memory_ids
                if by_id.get(memory_id, {}).get("kind") == "PREFERENCE"
            ],
            "plans": [
                memory_id
                for memory_ids in heads.values()
                for memory_id in memory_ids
                if by_id.get(memory_id, {}).get("kind") == "PLAN"
            ],
            "planning_anchors": [
                memory["id"]
                for memory in sorted(
                    (
                        memory
                        for memory in self._memories
                        if memory["id"] in active_ids
                        and memory.get("planning_tags")
                        and (
                            # TRAJECTORY, RISK, CONSTRAINT memories are always included
                            # regardless of salience — they supply the longitudinal context
                            # and decision-changing constraints that IG/DECISION queries need.
                            any(
                                tag in memory.get("planning_tags", [])
                                for tag in ("TRAJECTORY", "RISK", "CONSTRAINT")
                            )
                            or float(memory.get("decision_salience", 0.0) or 0.0)
                            >= 0.30
                        )
                    ),
                    key=lambda memory: (
                        float(memory.get("decision_salience", 0.0) or 0.0),
                        self._recency_date(memory),
                        memory["id"],
                    ),
                    reverse=True,
                )[:80]
            ],
            "trajectory_anchors": [
                memory["id"]
                for memory in sorted(
                    (
                        memory
                        for memory in self._memories
                        if memory.get("assertion_mode", "DIRECT") == "DIRECT"
                        and memory.get("evidence_ids")
                        and any(
                            tag in memory.get("planning_tags", [])
                            for tag in (
                                "EXPOSURE",
                                "RESPONSE",
                                "TRAJECTORY",
                                "RISK",
                                "CONSTRAINT",
                            )
                        )
                    ),
                    key=lambda memory: (
                        float(memory.get("decision_salience", 0.0) or 0.0),
                        self._recency_date(memory),
                        memory["id"],
                    ),
                    reverse=True,
                )[:160]
            ],
        }

    def _memory_health_stats(self) -> Dict[str, Any]:
        key_counts: Dict[str, int] = defaultdict(int)
        identity_values: Dict[str, List[str]] = defaultdict(list)
        kind_counts: Dict[str, int] = defaultdict(int)
        relation_counts: Dict[str, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        for memory in self._memories:
            kind_counts[memory.get("kind", "UNKNOWN")] += 1
            if memory.get("state_key") and memory.get("kind") in STATE_LIKE_KINDS:
                key_counts[memory["state_key"]] += 1
                identity = self._state_identity(memory)
                if identity:
                    identity_values[identity].append(self._normalised_value(memory))
            speakers = memory.get("source_speakers") or ["unknown"]
            for speaker in speakers:
                source_counts[str(speaker).lower()] += 1
        evidence_ids = {evidence["id"] for evidence in self._evidence}
        missing_evidence_pointers = sum(
            evidence_id not in evidence_ids
            for memory in self._memories
            for evidence_id in memory.get("evidence_ids", [])
        )
        memory_without_evidence = sum(
            not memory.get("evidence_ids") for memory in self._memories
        )
        state_like_missing_identity = sum(
            memory.get("kind") in STATE_LIKE_KINDS and not self._state_identity(memory)
            for memory in self._memories
        )
        invalid_relation_endpoints = 0
        relation_identity_violations = 0
        causal_provenance_violations = 0
        by_id = {memory["id"]: memory for memory in self._memories}
        for relation in self._relations:
            relation_counts[relation.get("type", "UNKNOWN")] += 1
            source = by_id.get(relation.get("source_id"))
            target = by_id.get(relation.get("target_id"))
            if not source or not target:
                invalid_relation_endpoints += 1
                continue
            if relation.get("type") in {"SUPERSEDE", "REFINE"}:
                source_identity = self._state_identity(source)
                target_identity = self._state_identity(target)
                if (
                    source_identity
                    and target_identity
                    and source_identity != target_identity
                ):
                    relation_identity_violations += 1
            if relation.get("type") == "CAUSES" and (
                not source.get("evidence_ids")
                or not target.get("evidence_ids")
                or not relation.get("provenance_evidence_ids")
            ):
                causal_provenance_violations += 1
        singleton_keys = sum(count == 1 for count in key_counts.values())
        singleton_identities = sum(
            len(values) == 1 for values in identity_values.values()
        )
        duplicate_same_values = sum(
            len([value for value in values if value])
            - len({value for value in values if value})
            for values in identity_values.values()
        )
        conflicting_identities = 0
        for memory_ids in self._state_heads.values():
            active_values = {
                self._state_value_signature(by_id[memory_id])
                for memory_id in memory_ids
                if memory_id in by_id
                and self._belief_status.get(memory_id, "active")
                in {"active", "conflicting"}
                and self._state_value_signature(by_id[memory_id])
            }
            if len(active_values) > 1:
                conflicting_identities += 1
        version_counts = [len(values) for values in identity_values.values()]
        transition_like_non_state = sum(
            bool(self._explicit_state_transition(memory))
            and memory.get("kind") not in STATE_LIKE_KINDS
            for memory in self._memories
        )
        hard_violations = (
            missing_evidence_pointers
            + memory_without_evidence
            + state_like_missing_identity
            + invalid_relation_endpoints
            + relation_identity_violations
            + causal_provenance_violations
        )
        return {
            "memory_count": len(self._memories),
            "kind_counts": dict(kind_counts),
            "unique_state_keys": len(key_counts),
            "singleton_state_keys": singleton_keys,
            "state_key_singleton_rate": round(
                singleton_keys / max(1, len(key_counts)), 4
            ),
            "identity_count": len(identity_values),
            "singleton_identity_count": singleton_identities,
            "identity_singleton_rate": round(
                singleton_identities / max(1, len(identity_values)), 4
            ),
            "versions_per_identity_mean": round(
                sum(version_counts) / max(1, len(version_counts)), 4
            ),
            "versions_per_identity_max": max(version_counts, default=0),
            "duplicate_same_value_count": duplicate_same_values,
            "conflicting_value_identity_count": conflicting_identities,
            "active_state_heads": sum(len(ids) for ids in self._state_heads.values()),
            "relation_counts": dict(relation_counts),
            "source_counts": dict(source_counts),
            "transition_like_non_state_count": transition_like_non_state,
            "missing_evidence_pointer_count": missing_evidence_pointers,
            "memory_without_evidence_count": memory_without_evidence,
            "state_like_missing_identity_count": state_like_missing_identity,
            "invalid_relation_endpoint_count": invalid_relation_endpoints,
            "relation_identity_violation_count": relation_identity_violations,
            "causal_provenance_violation_count": causal_provenance_violations,
            "hard_violation_count": hard_violations,
            "freeze_ready": hard_violations == 0,
        }

    @staticmethod
    def _inherit_state_keys(
        new_memories: List[Dict[str, Any]],
        old_memories: Dict[str, Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> None:
        """Keep one canonical key across semantically linked state versions."""
        inherited = set()
        for relation in sorted(
            relations, key=lambda item: item["confidence"], reverse=True
        ):
            if relation["type"] not in {"SUPPORT", "REFINE", "SUPERSEDE", "CONFLICT"}:
                continue
            match = re.fullmatch(r"new_(\d+)", relation["source_id"])
            old = old_memories.get(relation["target_id"])
            if not match or not old or not old.get("state_key"):
                continue
            index = int(match.group(1))
            if index >= len(new_memories) or index in inherited:
                continue
            memory = new_memories[index]
            if memory["kind"] in STATE_LIKE_KINDS:
                memory["state_key"] = old["state_key"]
                for field in ("subject", "scope", "object_anchor"):
                    if old.get(field) not in (None, ""):
                        memory[field] = old[field]
                inherited.add(index)

    def _sanitize_state_relations(
        self,
        new_memories: Sequence[Dict[str, Any]],
        old_memories: Dict[str, Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep only identity-valid state edges after canonical inheritance."""
        output: List[Dict[str, Any]] = []
        related_per_source: Dict[str, int] = defaultdict(int)
        support_per_source: Dict[str, int] = defaultdict(int)
        supersede_per_source = set()
        def relation_order(relation: Dict[str, Any]) -> tuple:
            target = old_memories.get(str(relation.get("target_id") or ""), {})
            return (
                self._recency_date(target),
                target.get("document_time", ""),
                float(relation.get("confidence", 0.0) or 0.0),
                str(target.get("id", "")),
            )

        for relation in sorted(relations, key=relation_order, reverse=True):
            source_ref = str(relation.get("source_id") or "")
            match = re.fullmatch(r"new_(\d+)", source_ref)
            target = old_memories.get(str(relation.get("target_id") or ""))
            if not match or not target:
                continue
            index = int(match.group(1))
            if index >= len(new_memories):
                continue
            source = new_memories[index]
            relation_type = str(relation.get("type") or "")
            if relation_type in {"REFINE", "SUPERSEDE", "CONFLICT"}:
                source_identity = self._state_identity(source)
                target_identity = self._state_identity(target)
                if (
                    not source_identity
                    or not target_identity
                    or source_identity != target_identity
                ):
                    continue
            if (
                relation_type == "SUPERSEDE"
                and not self._explicit_state_replacement(source, target)
            ):
                # Preserve the link as a weak topical relation rather than
                # allowing an unsupported replacement to poison state heads.
                relation_type = "RELATED"
            if relation_type == "SUPERSEDE":
                if source_ref in supersede_per_source:
                    continue
                supersede_per_source.add(source_ref)
            if relation_type == "RELATED":
                if related_per_source[source_ref] >= 1:
                    continue
                related_per_source[source_ref] += 1
            if relation_type == "SUPPORT":
                if support_per_source[source_ref] >= 2:
                    continue
                support_per_source[source_ref] += 1
            normalized_relation = self._snapshot(relation)
            normalized_relation["type"] = relation_type
            output.append(normalized_relation)
        return output

    @staticmethod
    def _reuse_nearby_state_keys(
        new_memories: List[Dict[str, Any]],
        old_memories: Sequence[Dict[str, Any]],
    ) -> None:
        """Reuse a close existing key without forcing broad semantic merges."""
        state_kinds = STATE_LIKE_KINDS
        for memory in new_memories:
            key = memory.get("state_key", "")
            if memory.get("kind") not in state_kinds or not key:
                continue
            new_terms = set(key.split("_"))
            candidates = []
            for old in old_memories:
                old_key = old.get("state_key", "")
                if old.get("kind") != memory["kind"] or not old_key:
                    continue
                if any(
                    str(memory.get(field) or "") != str(old.get(field) or "")
                    for field in ("subject", "scope", "object_anchor")
                ):
                    continue
                old_terms = set(old_key.split("_"))
                overlap = len(new_terms & old_terms)
                union = len(new_terms | old_terms)
                if overlap < 2 or not union:
                    continue
                jaccard = overlap / union
                subset = new_terms.issubset(old_terms) or old_terms.issubset(new_terms)
                if jaccard >= 0.72 or (subset and overlap >= 3):
                    candidates.append((jaccard, overlap, old_key))
            if candidates:
                memory["state_key"] = max(candidates)[-1]

    def _deterministic_state_relations(
        self,
        new_memories: Sequence[Dict[str, Any]],
        old_memories: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
        document_time: str,
    ) -> List[Dict[str, Any]]:
        """Close unambiguous same-identity state transitions missed by consolidation."""
        output = [self._snapshot(relation) for relation in relations]
        existing_pairs = {
            (relation.get("source_id"), relation.get("target_id"))
            for relation in output
        }
        # Deterministic closure is limited to observed state values. Plans and
        # preferences often share broad identities while remaining complementary.
        state_kinds = {"STATE"}
        for index, memory in enumerate(new_memories):
            source_ref = f"new_{index}"
            if (
                memory.get("kind") not in state_kinds
                or memory.get("assertion_mode", "DIRECT") != "DIRECT"
                or not self._state_identity(memory)
                or not self._normalised_value(memory)
            ):
                continue
            source_date = self._parse_date(
                self._resolve_event_time(memory, document_time)
            )
            source_date = source_date or self._parse_date(document_time)
            provisional_targets = [
                {**candidate, "id": f"new_{candidate_index}"}
                for candidate_index, candidate in enumerate(new_memories[:index])
            ]
            for old in [*old_memories, *provisional_targets]:
                pair = (source_ref, old.get("id"))
                if (
                    pair in existing_pairs
                    or old.get("kind") not in state_kinds
                    or (
                        not str(old.get("id") or "").startswith("new_")
                        and self._belief_status.get(old.get("id"), "active")
                        == "superseded"
                    )
                    or self._state_identity(memory) != self._state_identity(old)
                    or not self._normalised_value(old)
                    or not self._explicit_state_replacement(
                        {**memory, "document_time": document_time}, old
                    )
                ):
                    continue
                target_date = (
                    self._parse_date(self._resolve_event_time(old, document_time))
                    or self._recency_date(old)
                    or self._parse_date(document_time)
                )
                if source_date and target_date and source_date < target_date:
                    continue
                output.append(
                    {
                        "source_id": source_ref,
                        "target_id": old["id"],
                        "type": "SUPERSEDE",
                        "confidence": 1.0,
                        "provenance_evidence_ids": [],
                    }
                )
                existing_pairs.add(pair)
        return output

    def _add_memories(
        self,
        new_memories: List[Dict[str, Any]],
        causal_links: List[Dict[str, Any]],
        document_time: str,
        turn_map: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        nearby = self._nearby_old_memories(new_memories)
        evidence_by_id = {evidence["id"]: evidence for evidence in self._evidence}
        # Recap detection must precede state standardization. Otherwise an old
        # value restated today can inherit a current identity and look like a new
        # state transition before its provenance is recognized.
        for memory in new_memories:
            source_text = "\n".join(
                evidence_by_id[turn_map[turn_idx]]["raw_text"]
                for turn_idx in memory.get("source_turns", [])
                if turn_idx in turn_map and turn_map[turn_idx] in evidence_by_id
            )
            if not self._contains_recap_reference(source_text):
                continue
            origin = self._best_prior_origin(memory, nearby)
            if origin and not self._has_explicit_value_change(memory, origin):
                memory["assertion_mode"] = "RECAP"
                memory["origin_memory_id"] = origin["id"]
        standardization_stats = self._standardize_state_updates(new_memories)
        # Scope/key inheritance may expose better exact-identity neighbours.
        nearby = self._nearby_old_memories(new_memories)
        consolidation = self._consolidation_relations(
            new_memories, nearby, turn_map, document_time
        )
        old_by_id = {item["id"]: item for item in nearby}
        self._inherit_state_keys(new_memories, old_by_id, consolidation)
        self._reuse_nearby_state_keys(new_memories, nearby)
        consolidation = self._sanitize_state_relations(
            new_memories, old_by_id, consolidation
        )
        consolidation = self._deterministic_state_relations(
            new_memories,
            nearby,
            consolidation,
            document_time,
        )
        relations_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for relation in consolidation:
            relations_by_source[relation["source_id"]].append(relation)

        skipped_recaps: Dict[int, str] = {}
        for index, memory in enumerate(new_memories):
            if memory.get("assertion_mode") != "RECAP":
                continue
            target_id = memory.get("origin_memory_id", "")
            if target_id not in old_by_id:
                support = next(
                    (
                        relation
                        for relation in relations_by_source.get(f"new_{index}", [])
                        if relation["type"] == "SUPPORT"
                        and relation["target_id"] in old_by_id
                    ),
                    None,
                )
                target_id = support["target_id"] if support else ""
            source_text = "\n".join(
                evidence_by_id[turn_map[turn_idx]]["raw_text"]
                for turn_idx in memory.get("source_turns", [])
                if turn_idx in turn_map and turn_map[turn_idx] in evidence_by_id
            )
            # Ordinary duplicate recaps remain suppressed. A recap carrying an
            # explicit frequency, measurement, date, or transition is retained
            # as a RECAP card because it represents a distinct document event
            # and may be the only evidence for a document-time query.
            retain_answer_bearing_recap = self._recap_has_answer_bearing_detail(
                memory, source_text
            )
            if target_id in old_by_id and not retain_answer_bearing_recap:
                skipped_recaps[index] = target_id

        local_ids, added = {}, []
        for index, memory in enumerate(new_memories):
            if index in skipped_recaps:
                local_ids[f"new_{index}"] = skipped_recaps[index]
                continue
            refinement = next(
                (
                    relation
                    for relation in sorted(
                        relations_by_source.get(f"new_{index}", []),
                        key=lambda item: item["confidence"],
                        reverse=True,
                    )
                    if relation["type"] == "REFINE"
                    and relation["target_id"] in old_by_id
                ),
                None,
            )
            origin_document_time = document_time
            # SUPPORT does not make the old source the origin of a new claim.
            # REFINE inherits lineage only for the same explicitly dated event.
            if refinement:
                ancestor = old_by_id[refinement["target_id"]]
                source_event = self._parse_date(
                    self._resolve_event_time(memory, document_time)
                )
                target_event = self._parse_date(ancestor.get("event_time", ""))
                if source_event and target_event and source_event == target_event:
                    origin_document_time = (
                        ancestor.get("origin_document_time")
                        or ancestor.get("document_time")
                        or document_time
                    )
            self._memory_seq += 1
            memory_id = f"m_{self._memory_seq}"
            local_ids[f"new_{index}"] = memory_id
            card = {
                "id": memory_id,
                "claim": memory["claim"],
                "kind": memory["kind"],
                "entities": memory["entities"],
                "subject": memory.get("subject", "patient"),
                "scope": memory.get("scope", "general"),
                "state_key": memory["state_key"],
                "object_anchor": memory.get("object_anchor", ""),
                "scope_entities": memory.get("scope_entities", []),
                "value": memory.get("value", ""),
                "verbatim_value": memory.get("verbatim_value", ""),
                "stance": memory["stance"],
                "event_time": self._resolve_event_time(memory, document_time),
                "time_expression": memory["time_expression"],
                "document_time": document_time,
                "origin_document_time": origin_document_time,
                "assertion_mode": memory.get("assertion_mode", "DIRECT"),
                "origin_memory_id": memory.get("origin_memory_id", ""),
                "planning_tags": list(memory.get("planning_tags", [])),
                "decision_salience": float(memory.get("decision_salience", 0.0) or 0.0),
                "evidence_ids": [
                    turn_map[t] for t in memory["source_turns"] if t in turn_map
                ],
                "source_speakers": sorted(
                    {
                        evidence_by_id[turn_map[t]].get("speaker", "")
                        for t in memory["source_turns"]
                        if t in turn_map
                        and turn_map[t] in evidence_by_id
                        and evidence_by_id[turn_map[t]].get("speaker")
                    }
                ),
                "confidence": memory["confidence"],
                "session_idx": self._session_seq,
            }
            self._memories.append(card)
            added.append(card)
        for relation in consolidation:
            source = local_ids.get(relation["source_id"])
            target = local_ids.get(relation["target_id"], relation["target_id"])
            if source and target and source != target:
                self._relations.append(
                    {**relation, "source_id": source, "target_id": target}
                )
        for link in causal_links:
            source = local_ids.get(f"new_{link.get('cause_index')}")
            target = local_ids.get(f"new_{link.get('effect_index')}")
            if not source or not target or source == target:
                continue
            try:
                confidence = max(0.0, min(1.0, float(link.get("confidence", 0.75))))
            except (TypeError, ValueError):
                confidence = 0.75
            by_id = {memory["id"]: memory for memory in self._memories}
            source_evidence = set(by_id.get(source, {}).get("evidence_ids") or [])
            target_evidence = set(by_id.get(target, {}).get("evidence_ids") or [])
            provenance = [
                turn_map[turn]
                for turn in link.get("source_turns", [])
                if turn in turn_map
            ]
            if (
                confidence >= 0.65
                and source_evidence
                and target_evidence
                and provenance
            ):
                self._relations.append(
                    {
                        "source_id": source,
                        "target_id": target,
                        "type": "CAUSES",
                        "confidence": confidence,
                        "provenance_evidence_ids": list(dict.fromkeys(provenance)),
                        "provenance_kind": "FOCAL_CAUSAL_TURN",
                    }
                )
        self._rebuild_belief_view()
        self._index_dirty = True
        self._last_write_stats = {
            "skipped_recaps": len(skipped_recaps),
            "committed_memories": len(added),
            **standardization_stats,
        }
        return added
