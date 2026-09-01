"""Deterministic constrained retrieval operations."""

from collections import defaultdict
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import VALID_SEARCH_STRATEGIES, QueryFrame
from .core import CoreMemoryMixin


class RetrievalOperationsMixin:
    """Executes individual temporal, state, semantic, and causal operations."""

    @staticmethod
    def _date_for(memory: Dict[str, Any], axis: str = "event_time") -> str:
        if axis == "effective_event_time":
            event = CoreMemoryMixin._parse_date(memory.get("event_time", ""))
            origin = CoreMemoryMixin._parse_date(memory.get("origin_document_time", ""))
            document = CoreMemoryMixin._parse_date(memory.get("document_time", ""))
            # A day-level source date may safely refine a coarse YYYY-MM event
            # only when it falls inside that same month. This is the explicit
            # semantics of effective_event_time, not an event_time fallback.
            if event and event.count("-") == 1:
                refinement = next(
                    (
                        date
                        for date in (origin, document)
                        if len(date) == 10 and date.startswith(event)
                    ),
                    "",
                )
                return refinement or event
            return event or origin or document
        if axis == "origin_document_time":
            return CoreMemoryMixin._parse_date(memory.get("origin_document_time", ""))
        if axis == "document_time":
            return CoreMemoryMixin._parse_date(memory.get("document_time", ""))
        return CoreMemoryMixin._parse_date(memory.get("event_time", ""))

    @staticmethod
    def _date_matches(date: str, constraint: str) -> bool:
        parsed = CoreMemoryMixin._parse_date(date)
        if not parsed:
            return False
        if constraint.startswith("*-"):
            return len(parsed) >= 10 and parsed[4:] == constraint[1:]
        return parsed == constraint or parsed.startswith(constraint)

    @classmethod
    def _memory_matches_dates(
        cls, memory: Dict[str, Any], dates: Sequence[str]
    ) -> bool:
        if not dates:
            return True
        memory_dates = (
            memory.get("event_time", ""),
            memory.get("origin_document_time", ""),
            memory.get("document_time", ""),
        )
        return any(
            cls._date_matches(date, constraint)
            for date in memory_dates
            for constraint in dates
        )

    @staticmethod
    def _speaker_role_match(memory: Dict[str, Any], role: str) -> int:
        if not role:
            return 0
        speakers = {
            str(value).strip().lower() for value in memory.get("source_speakers", [])
        }
        if not speakers:
            return 0
        if role == "patient":
            return int(bool(speakers & {"patient", "user"}))
        if role == "doctor":
            return int(
                bool(speakers & {"doctor", "assistant", "clinician", "physician"})
            )
        return 0

    def _rank_for_query_frame(
        self,
        question: str,
        memories: Sequence[Dict[str, Any]],
        frame: QueryFrame,
    ) -> List[Dict[str, Any]]:
        eligible = [
            self._snapshot(memory)
            for memory in memories
            if self._memory_satisfies_frame(memory, frame)
        ]
        return sorted(
            eligible,
            key=lambda memory: (
                float(memory.get("_score", 0.0)),
                int(memory.get("_overlap", 0)),
                float(memory.get("_dense_score", 0.0)),
                memory["id"],
            ),
            reverse=True,
        )

    @staticmethod
    def _select_initial_seeds(
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep RRF quality while preserving each channel's strongest candidate."""
        selected: List[Dict[str, Any]] = []

        def add(memory: Optional[Dict[str, Any]]) -> None:
            if memory and all(item["id"] != memory["id"] for item in selected):
                selected.append(memory)

        add(candidates[0] if candidates else None)
        add(
            min(
                (
                    memory
                    for memory in candidates
                    if memory.get("_bm25_rank") is not None
                ),
                key=lambda memory: memory["_bm25_rank"],
                default=None,
            )
        )
        add(
            min(
                (
                    memory
                    for memory in candidates
                    if memory.get("_dense_rank") is not None
                ),
                key=lambda memory: memory["_dense_rank"],
                default=None,
            )
        )
        for memory in candidates:
            add(memory)
            if len(selected) >= 3:
                break
        return [CoreMemoryMixin._snapshot(memory) for memory in selected[:3]]

    def _constraint_first_search(
        self,
        question: str,
        frame: QueryFrame,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Apply only explicit hard constraints before semantic retrieval.

        Semantic intent (temporal/current/causal/comparison) is deliberately
        absent here and belongs to the planner.
        """
        top_k = min(8, max(3, int(top_k or self.INITIAL_TOP_K)))
        candidate_ids = None
        if frame.dates or frame.speaker_role or frame.hard_entities:
            candidate_ids = [
                memory["id"]
                for memory in self._memories
                if self._memory_satisfies_frame(memory, frame)
            ]
            if not candidate_ids:
                return []
        candidates = self._hybrid_search(
            question, top_k=top_k, candidate_ids=candidate_ids
        )
        return self._rank_for_query_frame(question, candidates, frame)[:top_k]

    def _semantic_operation_search(
        self,
        query: str,
        top_k: int,
        strategy: str,
        frame: QueryFrame = QueryFrame(),
        option_queries: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute one bounded, role-diversified semantic search operation.

        SHARED_OPTIONS is one logical operation. It can fan out to a few cheap
        lexical/dense probes, one per visible proposition, then merge and cap
        the candidates. This avoids a long concatenated question diluting the
        evidence for the later options while preserving one retrieval boundary.
        """
        strategy = str(strategy or "FOCAL").upper()
        if strategy not in VALID_SEARCH_STRATEGIES:
            strategy = "FOCAL"
        eligible_ids = {
            memory["id"]
            for memory in self._memories
            if self._memory_satisfies_frame(
                memory, frame, include_entities=bool(frame.hard_entities)
            )
        }
        if not eligible_ids:
            return []
        base = self._hybrid_search(
            query,
            top_k=min(max(top_k * 3, 12), len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        if strategy == "SHARED_OPTIONS" and option_queries:
            option_hits: List[Dict[str, Any]] = []
            per_option_k = max(2, min(3, int(top_k)))
            for option_query in option_queries:
                text = str(option_query or "").strip()
                if not text:
                    continue
                option_hits.extend(
                    self._hybrid_search(
                        text,
                        top_k=min(per_option_k * 2, len(eligible_ids)),
                        candidate_ids=eligible_ids,
                    )
                )
            merged = {}
            for memory in (*option_hits, *base):
                merged.setdefault(memory["id"], memory)
            base = list(merged.values())
        if strategy == "FOCAL":
            return base[:top_k]

        # TRAJECTORY and LOCATE_ANCHOR must agree on the focal episode. A
        # plain semantic search can rank an undated summary above the dated
        # state versions needed for chronology, so reuse the bounded anchor
        # primitive instead of rebuilding a wider, inconsistent pool here.
        trajectory_anchors = (
            self._locate_anchor(query, frame) if strategy == "TRAJECTORY" else []
        )

        # Query-time retrieval must be driven by the operation and its
        # authorized inputs.  The offline profile pack is useful for write
        # continuity, but using its anchors here creates an implicit recall
        # channel that can bias decision/trajectory searches toward salient
        # memories unrelated to the current question.
        routed: List[Dict[str, Any]] = []

        # The operation may expose historical versions of a state lineage that
        # its own focal candidates identify. Every exposed ID remains part of
        # this operation's output/provenance boundary.
        if strategy == "TRAJECTORY":
            # A trajectory search must follow one focal state family. Using
            # the first four semantic hits creates a union of unrelated state
            # keys and lets endpoint reservation consume the small budget.
            focal_anchor = (trajectory_anchors or base or routed or [None])[0]
            # Query-time trajectory grouping must use the durable identity,
            # including scope. The scope-tolerant lineage helper is reserved
            # for write-time reconciliation and must not mix distinct state
            # families in the final evidence set.
            anchor_lineage = (
                self._state_identity(focal_anchor) if focal_anchor else ""
            )
            focal_lineages = {anchor_lineage} if anchor_lineage else set()
        else:
            focal_lineages = {
                self._state_identity(memory)
                for memory in (*base[:4], *routed[:4])
                if self._state_identity(memory)
            }
        lineage_versions = [
            memory
            for memory in self._memories
            if memory["id"] in eligible_ids
            and self._state_identity(memory) in focal_lineages
            and memory.get("assertion_mode", "DIRECT") == "DIRECT"
        ]
        if strategy == "TRAJECTORY" and focal_lineages:
            # The operation's budget belongs to the focal trajectory, not to
            # unrelated semantic neighbours that happened to match words such
            # as "change", "rise", or "drop".
            candidate_pool = lineage_versions
        elif strategy == "TRAJECTORY":
            # Non-state events still use the anchor primitive's bounded
            # candidates; the broad base/routed union is not a safe timeline.
            candidate_pool = trajectory_anchors or base[:top_k]
        else:
            candidate_pool = (*base, *routed, *lineage_versions)
        candidate_ids = {memory["id"] for memory in candidate_pool}
        ranked = self._hybrid_search(
            query,
            top_k=min(max(top_k * 5, 24), len(candidate_ids)),
            candidate_ids=candidate_ids,
        )

        selected: List[Dict[str, Any]] = []

        def add(memory: Optional[Dict[str, Any]]) -> None:
            if (
                memory
                and memory["id"] not in {item["id"] for item in selected}
                and len(selected) < top_k
            ):
                selected.append(self._snapshot(memory))

        if strategy == "TRAJECTORY":
            # A trajectory operation is useful only if it can expose temporal
            # endpoints.  Reserve those endpoints before role diversification;
            # otherwise a small top_k is consumed by several near-duplicate
            # role candidates and the transition never reaches the executor.
            for lineage in sorted(focal_lineages):
                versions = sorted(
                    (
                        memory
                        for memory in lineage_versions
                        if self._state_identity(memory) == lineage
                    ),
                    key=lambda memory: (
                        self._recency_date(memory),
                        memory.get("document_time", ""),
                        memory["id"],
                    ),
                )
                if versions:
                    add(versions[0])
                    add(versions[-1])
        add(
            (trajectory_anchors or base or routed or [None])[0]
            if strategy == "TRAJECTORY"
            else (base[0] if base else None)
        )
        role_order = {
            "TRAJECTORY": ("EXPOSURE", "RESPONSE", "TRAJECTORY", "RISK"),
            "DECISION_BUNDLE": (
                "STATE",
                "EXPOSURE",
                "RESPONSE",
                "TRAJECTORY",
                "RISK",
                "CONSTRAINT",
                "PLAN",
                "PREFERENCE",
            ),
            "SHARED_OPTIONS": (
                "STATE",
                "ACTION_RULE",
                "RISK",
                "CONSTRAINT",
                "PLAN",
                "PREFERENCE",
                "RESOURCE",
                "TRAJECTORY",
            ),
        }[strategy]
        for tag in role_order:
            add(
                next(
                    (
                        memory
                        for memory in ranked
                        if tag in memory.get("planning_tags", [])
                        or (tag == "STATE" and self._is_state_head(memory))
                        or (tag == "ACTION_RULE" and memory.get("kind") == "PLAN")
                    ),
                    None,
                )
            )

        seen_dates = {
            memory.get("document_time") for memory in selected if memory.get("document_time")
        }
        for memory in ranked:
            date = memory.get("document_time")
            if date and date not in seen_dates:
                add(memory)
                seen_dates.add(date)
            if len(selected) >= top_k:
                break
        for memory in ranked:
            add(memory)
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    @staticmethod
    def _recency_date(memory: Dict[str, Any]) -> str:
        return CoreMemoryMixin._parse_date(
            memory.get("event_time", "")
        ) or CoreMemoryMixin._parse_date(memory.get("document_time", ""))

    def _locate_anchor(
        self,
        query: str,
        frame: QueryFrame = QueryFrame(),
    ) -> List[Dict[str, Any]]:
        candidate_ids = (
            [
                memory["id"]
                for memory in self._memories
                if self._memory_satisfies_frame(memory, frame)
            ]
            if (frame.dates or frame.speaker_role or frame.hard_entities)
            else None
        )
        candidates = self._hybrid_search(query, top_k=20, candidate_ids=candidate_ids)
        event_dated = [c for c in candidates if self._date_for(c, "event_time")]
        document_dated = [c for c in candidates if self._date_for(c, "document_time")]
        event_nodes = [c for c in event_dated if c["kind"] in {"EVENT", "STATE"}]
        focal = event_nodes or event_dated or document_dated or candidates

        # Anchor discovery is still bounded semantic recall, but it must expose
        # versions of the same state/event family so a later temporal filter can
        # compare dates instead of selecting the first high-scoring recurrence.
        # Temporal anchor discovery has one semantic focal subject. Taking
        # several top-ranked lineages mixes lexical neighbours from unrelated
        # trajectories (for example weight loss with insulin timing), and a
        # later extremum filter can then select the wrong event. Use the best
        # dated focal candidate as the anchor and expand only its lineage.
        anchor = focal[0] if focal else None
        # A query trajectory is a strict state family. Scope-tolerant lineage
        # matching is only a write-time repair mechanism, never read-time
        # evidence expansion.
        anchor_lineage = self._state_identity(anchor) if anchor else ""
        lineages = {anchor_lineage} if anchor_lineage else set()
        family = [
            memory
            for memory in self._memories
            if self._state_identity(memory) in lineages
            and memory.get("assertion_mode", "DIRECT") == "DIRECT"
        ]
        family_sorted = sorted(
            family,
            key=lambda memory: (
                self._recency_date(memory),
                memory.get("document_time", ""),
                memory["id"],
            ),
        )
        focal_candidates = [focal[0]] if focal else []
        if not lineages:
            # Event/FACT memories may not have a state lineage. Keep a small
            # semantic fallback in that case without displacing the anchor.
            focal_candidates.extend(focal[1:3])
        if lineages:
            # Once a state lineage is identified, unrelated dated focal hits
            # are not valid candidates for the same temporal question. Keep
            # reranking inside the selected family; otherwise they re-enter
            # through the ranked-fill path below.
            pool_memories = family
        else:
            pool_memories = focal
        pool = {
            memory["id"]: memory for memory in (*focal_candidates, *pool_memories)
        }
        ranked = self._hybrid_search(
            query,
            top_k=min(32, len(pool)),
            candidate_ids=set(pool),
        )
        selected: List[Dict[str, Any]] = []
        # Keep both ends of each candidate lineage. The following temporal
        # filter decides EARLIEST or LATEST; anchor discovery must not discard
        # one endpoint merely because the opposite endpoint ranked higher.
        family_endpoints: List[Dict[str, Any]] = []
        for lineage in sorted(lineages):
            versions = [
                memory
                for memory in family_sorted
                if self._state_identity(memory) == lineage
            ]
            family_endpoints.extend((*versions[:2], *versions[-2:]))
        for memory in (*focal_candidates, *family_endpoints, *ranked):
            if memory["id"] not in {item["id"] for item in selected}:
                selected.append(self._snapshot(memory))
            if len(selected) >= 8:
                break
        return selected

    def _operation_date(
        self,
        value: Any,
        axis: str,
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
    ) -> str:
        parsed = self._parse_date(str(value or ""))
        if parsed:
            return parsed
        resolved = self._resolve_refs(value, outputs, seeds)
        if not resolved:
            return ""
        return self._date_for(resolved[0], axis)

    def _temporal_filter(
        self,
        operation: Dict[str, Any],
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
        frame: QueryFrame = QueryFrame(),
    ) -> List[Dict[str, Any]]:
        relation = str(operation.get("relation", "")).upper()
        axis = str(operation.get("axis", "event_time")).lower()
        fallback_axis = str(operation.get("fallback_axis") or "").lower()
        query = str(operation.get("query") or "").strip()
        candidate_refs = operation.get("candidate_refs")
        has_focal_scope = isinstance(candidate_refs, list) and bool(candidate_refs)
        focal_candidates = (
            self._resolve_refs(candidate_refs, outputs, seeds)
            if has_focal_scope
            else []
        )
        focal_ids = {memory["id"] for memory in focal_candidates}
        if has_focal_scope and not focal_ids:
            # A declared dependency that produced no candidates is an
            # insufficiency, not permission to search outside the plan.
            return []
        anchor_date = self._operation_date(
            operation.get("anchor"), axis, outputs, seeds
        )
        end_date = self._operation_date(operation.get("end"), axis, outputs, seeds)
        if relation == "EXACT" and not anchor_date:
            anchor_date = self._parse_date(query)

        def operation_date(memory: Dict[str, Any]) -> str:
            """Use only the axis and explicit fallback declared by this operation."""
            date = self._date_for(memory, axis)
            if not date and fallback_axis and fallback_axis != axis:
                date = self._date_for(memory, fallback_axis)
            return date

        def eligible(memory: Dict[str, Any]) -> bool:
            date = operation_date(memory)
            if not date:
                return False
            if relation == "AFTER":
                return bool(anchor_date and date > anchor_date)
            if relation == "BEFORE":
                return bool(anchor_date and date < anchor_date)
            if relation == "BETWEEN":
                return bool(
                    anchor_date and end_date and anchor_date <= date <= end_date
                )
            if relation == "EXACT":
                if not anchor_date:
                    # The requested date may be the answer rather than a known
                    # filter value. In that case semantic event matching owns
                    # candidate selection and the operation still enforces axis.
                    return True
                return bool(
                    anchor_date
                    and (date == anchor_date or date.startswith(anchor_date))
                )
            return True

        ids = [
            memory["id"]
            for memory in self._memories
            if (not has_focal_scope or memory["id"] in focal_ids)
            if eligible(memory)
            and self._memory_satisfies_frame(
                memory,
                frame,
                include_dates=False,
                include_entities=False,
            )
        ]
        # Preserve explicit numeric anchors (measurements, doses, counts) when
        # the operation is exact or locates an extremum. Dense similarity alone
        # commonly promotes a nearby value; a numeric token present in the
        # query and claim is a structural constraint, not an intent classifier.
        query_numbers = set(re.findall(r"(?<!\d)\d+(?:\.\d+)?", query))
        if query_numbers:
            numeric_ids = []
            for memory in self._memories:
                if memory["id"] not in ids:
                    continue
                text = " ".join(
                    (str(memory.get("claim") or ""), self._memory_value(memory))
                )
                if query_numbers.issubset(
                    set(re.findall(r"(?<!\d)\d+(?:\.\d+)?", text))
                ):
                    numeric_ids.append(memory["id"])
            if numeric_ids:
                ids = numeric_ids
        ranked = self._hybrid_search(
            query or "event", top_k=min(40, len(ids)), candidate_ids=ids
        )
        if relation in {"EARLIEST", "LATEST"} and ranked:
            best_dense = float(ranked[0].get("_dense_score", 0.0))
            best_overlap = int(ranked[0].get("_overlap", 0))
            relevant = (
                ranked
                if focal_ids
                else [
                    item
                    for item in ranked[:20]
                    if (
                        float(item.get("_dense_score", 0.0)) >= best_dense - 0.12
                        and int(item.get("_overlap", 0)) >= max(1, best_overlap - 1)
                    )
                ]
            )
            relevant = relevant or ranked[:1]
            if not focal_ids:
                # Reconstruct chronology across versions of the best focal
                # state. A later paraphrase often ranks above the first mention;
                # semantic top-k alone therefore cannot establish an extremum.
                best = ranked[0]
                query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
                focal_identities = {
                    self._state_identity(memory): {
                        token
                        for token in self._tokenize(self._memory_value(memory))
                        if len(token) > 3
                        and token
                        not in {"patient", "current", "recent", "state"}
                    }
                    for memory in relevant
                    if self._state_identity(memory)
                }
                if focal_identities:
                    identity_versions = []
                    eligible_ids = set(ids)
                    for memory in self._memories:
                        text = " ".join(
                            (
                                str(memory.get("claim") or ""),
                                self._memory_value(memory),
                            )
                        )
                        if (
                            memory["id"] in eligible_ids
                            and self._state_identity(memory) in focal_identities
                            and focal_identities[self._state_identity(memory)].intersection(
                                self._tokenize(text)
                            )
                            and query_numbers.issubset(
                                set(re.findall(r"\d+(?:\.\d+)?", text))
                            )
                        ):
                            identity_versions.append(memory)
                    relevant = list(
                        {
                            memory["id"]: memory
                            for memory in (*identity_versions, *relevant)
                        }.values()
                    )
            # A recap is not independent event evidence, but it is a real later
            # document occurrence. Preserve it for document_time queries such as
            # "when was this mentioned most recently"; for event/origin semantics,
            # prefer the original assertion and fall back only when it is unavailable.
            pool = relevant
            if axis != "document_time":
                direct_relevant = [
                    item
                    for item in relevant
                    if item.get("assertion_mode", "DIRECT") != "RECAP"
                ]
                pool = direct_relevant if direct_relevant else relevant
            pool.sort(
                key=lambda memory: operation_date(memory) or "9999-99-99",
                reverse=relation == "LATEST",
            )
            return pool[:1]
        return ranked[:6]

    def _resolve_state(
        self,
        query: str,
        seeds: List[Dict[str, Any]],
        frame: QueryFrame = QueryFrame(),
    ) -> List[Dict[str, Any]]:
        candidate_ids = (
            [
                memory["id"]
                for memory in self._memories
                if self._memory_satisfies_frame(memory, frame)
            ]
            if (frame.dates or frame.speaker_role or frame.hard_entities)
            else None
        )
        candidates = (
            self._hybrid_search(query, top_k=8, candidate_ids=candidate_ids)
            if query
            else [
                memory
                for memory in seeds
                if self._memory_satisfies_frame(memory, frame)
            ]
        )
        selected_ids: List[str] = []
        # Resolve exactly the best matching state identity. Pulling heads for
        # every loosely related candidate makes unrelated active states look
        # like valid coverage and creates artificial conflicts.
        best_identity = next(
            (
                self._state_identity(candidate)
                for candidate in candidates
                if self._state_identity(candidate)
            ),
            "",
        )
        if best_identity:
            selected_ids.extend(self._state_heads.get(best_identity, []))
        else:
            direct = next(
                (
                    candidate
                    for candidate in candidates
                    if self._belief_status.get(candidate["id"], "active")
                    in {"active", "conflicting"}
                ),
                None,
            )
            if direct:
                selected_ids.append(direct["id"])
        by_id = {memory["id"]: memory for memory in self._memories}
        return [
            self._snapshot(by_id[memory_id])
            for memory_id in selected_ids
            if memory_id in by_id
        ][:4]

    def _follow_causes(
        self,
        operation: Dict[str, Any],
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        starts = self._resolve_refs(
            operation.get("start") or ["$seed0"], outputs, seeds
        )
        direction = str(operation.get("direction", "OUT")).upper()
        try:
            depth = max(1, min(3, int(operation.get("depth", 2))))
        except (TypeError, ValueError):
            depth = 2
        goal_terms = set(self._tokenize(str(operation.get("goal") or "")))
        by_id = {memory["id"]: memory for memory in self._memories}
        adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for relation in self._relations:
            if not self._valid_causal_relation(relation, by_id):
                continue
            source, target = relation["source_id"], relation["target_id"]
            if direction == "OUT":
                adjacency[source].append((target, relation))
            else:
                adjacency[target].append((source, relation))
        selected, used = {m["id"]: m for m in starts}, []
        frontier = [(m["id"], 0, 1.0) for m in starts]
        while frontier and len(selected) < self.HARD_MEMORY_LIMIT:
            node_id, level, path_score = frontier.pop(0)
            if level >= depth:
                continue
            neighbors = []
            for target_id, relation in adjacency.get(node_id, []):
                if target_id in selected or target_id not in by_id:
                    continue
                overlap = len(
                    goal_terms & set(self._tokenize(by_id[target_id]["claim"]))
                )
                score = (
                    path_score * float(relation["confidence"]) * (1.0 + 0.15 * overlap)
                )
                neighbors.append((score, target_id, relation))
            for score, target_id, relation in sorted(neighbors, reverse=True)[:3]:
                selected[target_id] = self._snapshot(by_id[target_id])
                used.append(relation)
                frontier.append((target_id, level + 1, score))
        return list(selected.values()), used

    def _valid_causal_relation(
        self,
        relation: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
    ) -> bool:
        if relation.get("type") != "CAUSES":
            return False
        source = by_id.get(str(relation.get("source_id") or ""))
        target = by_id.get(str(relation.get("target_id") or ""))
        provenance = set(relation.get("provenance_evidence_ids") or [])
        if (
            not source
            or not target
            or not source.get("evidence_ids")
            or not target.get("evidence_ids")
        ):
            return False
        # A causal edge may only cite evidence attached to one of its
        # endpoints, or a focal turn explicitly marked by the write extractor
        # as stating this causal link. This prevents an unrelated raw turn from
        # laundering temporal or topical association into a usable path.
        allowed = set(source["evidence_ids"]) | set(target["evidence_ids"])
        if provenance and provenance.issubset(allowed):
            return True
        evidence_ids = {str(evidence.get("id") or "") for evidence in self._evidence}
        return bool(
            relation.get("provenance_kind") == "FOCAL_CAUSAL_TURN"
            and provenance
            and provenance.issubset(evidence_ids)
        )
