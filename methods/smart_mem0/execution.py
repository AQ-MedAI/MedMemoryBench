"""Typed slot coverage, progressive execution, and pure arbitration."""

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .contracts import VALID_TEMPORAL_AXES, QueryFrame
from .prompts import SLOT_SUPPORT_GATE_PROMPT


class ExecutionMixin:
    """Stops retrieval on typed sufficiency while enforcing strict boundaries."""

    def _is_state_head(self, memory: Dict[str, Any]) -> bool:
        """Return whether a state-like memory is in its resolved head set."""
        identity = self._state_identity(memory)
        return bool(
            identity and memory.get("id") in self._state_heads.get(identity, [])
        )

    @staticmethod
    def _merge_relations(
        *relation_groups: Sequence[Dict[str, Any]],
        allowed_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        merged = {}
        for relations in relation_groups:
            for relation in relations:
                source = relation.get("source_id")
                target = relation.get("target_id")
                relation_type = relation.get("type")
                if allowed_ids is not None and (
                    source not in allowed_ids or target not in allowed_ids
                ):
                    continue
                merged[(source, target, relation_type)] = relation
        return list(merged.values())

    def _slot_seed_support(
        self,
        plan: Dict[str, Any],
        seeds: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        support: Dict[str, List[str]] = {
            slot["id"]: [] for slot in plan["required_slots"]
        }
        for coverage in plan.get("seed_coverage", []):
            resolved = self._resolve_refs(coverage["refs"], [], seeds)
            support[coverage["slot_id"]] = list(
                dict.fromkeys(memory["id"] for memory in resolved)
            )
        return support

    @staticmethod
    def _memory_matches_slot_role(
        slot: Dict[str, Any], memory: Dict[str, Any]
    ) -> bool:
        """Honor an explicit planner role; never infer one from the question."""
        role = str(slot.get("evidence_role") or "").upper()
        if not role:
            return True
        tags = {str(tag).upper() for tag in memory.get("planning_tags", [])}
        kind = str(memory.get("kind") or "").upper()
        role_tags = {
            "EXPOSURE": {"EXPOSURE", "STATE", "PLAN"},
            "RESPONSE": {"RESPONSE", "TRAJECTORY", "RISK", "STATE"},
            "TRAJECTORY": {"TRAJECTORY", "RESPONSE", "EXPOSURE", "RISK"},
            "LONGITUDINAL_CONTEXT": {
                "TRAJECTORY", "RESPONSE", "EXPOSURE", "RISK", "STATE"
            },
            "RISK": {"RISK", "CONSTRAINT", "RESPONSE"},
            "CONSTRAINT": {"CONSTRAINT", "RISK", "PLAN"},
            "ACTION_RULE": {"ACTION_RULE", "PLAN", "CONSTRAINT"},
            "ALTERNATIVE": {"ALTERNATIVE", "PREFERENCE", "RESOURCE", "PLAN"},
            # Capture models commonly tag an observed current measurement as
            # TRAJECTORY/RISK rather than STATE/EXPOSURE. Those tags are still
            # explicit write-time metadata, so accepting them here is a stable
            # compatibility rule, not query-intent inference.
            "FOCAL_STATE": {"STATE", "EXPOSURE", "RESPONSE", "TRAJECTORY", "RISK"},
            "DECISION": {"DECISION", "ACTION_RULE", "PLAN", "CONSTRAINT"},
            "TEMPORAL": {"TEMPORAL", "TRAJECTORY", "RESPONSE", "EXPOSURE", "STATE"},
            "CAUSE": {"CAUSE", "EXPOSURE", "RESPONSE", "TRAJECTORY", "RISK"},
            "COMPARAND": {"STATE", "EXPOSURE", "RESPONSE", "TRAJECTORY", "PLAN"},
        }.get(role)
        if role_tags is None:
            return True
        return bool(tags.intersection(role_tags)) or (
            role in {"ACTION_RULE", "DECISION"} and kind == "PLAN"
        )

    def _slot_covered(
        self,
        slot: Dict[str, Any],
        support_ids: Sequence[str],
        selected: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> bool:
        support_set = set(support_ids)
        memories = [memory for memory in selected if memory["id"] in support_set]
        if not memories:
            return False
        slot_type = slot["type"]

        if slot_type == "DIRECT":
            allow_history = bool(slot.get("history"))
            return any(
                bool(self._memory_value(memory))
                and memory.get("assertion_mode", "DIRECT") == "DIRECT"
                and self._memory_matches_slot_role(slot, memory)
                and (
                    allow_history
                    or memory.get(
                        "_status", self._belief_status.get(memory["id"], "active")
                    )
                    != "superseded"
                )
                for memory in memories
            )

        if slot_type == "CURRENT_STATE":
            heads = [
                memory
                for memory in memories
                if self._is_state_head(memory)
                and self._belief_status.get(memory["id"], "active")
                in {"active", "refined", "conflicting"}
                and self._memory_value(memory)
            ]
            identities = {self._state_identity(memory) for memory in heads}
            values = {self._state_value_signature(memory) for memory in heads}
            if len(heads) < 1 or len(identities) != 1 or len(values) != 1:
                return False
            identity = next(iter(identities))
            by_id = {memory["id"]: memory for memory in self._memories}
            global_values = {
                self._state_value_signature(by_id[memory_id])
                for memory_id in self._state_heads.get(identity, [])
                if memory_id in by_id
                and self._belief_status.get(memory_id, "active")
                in {"active", "refined", "conflicting"}
                and self._state_value_signature(by_id[memory_id])
            }
            return len(global_values) == 1

        if slot_type == "TEMPORAL":
            axis = slot.get("time_axis", "")
            fallback_axis = str(slot.get("fallback_axis") or "")
            if axis not in VALID_TEMPORAL_AXES:
                return False
            return any(
                bool(self._memory_value(memory))
                and bool(
                    self._date_for(memory, axis)
                    or (
                        fallback_axis in VALID_TEMPORAL_AXES
                        and self._date_for(memory, fallback_axis)
                    )
                )
                and self._memory_matches_slot_role(slot, memory)
                for memory in memories
            )

        if slot_type == "TRANSITION":
            by_id = {memory["id"]: memory for memory in memories}
            for relation in relations:
                if relation.get("type") not in {"SUPERSEDE", "REFINE"}:
                    continue
                source = by_id.get(relation.get("source_id"))
                target = by_id.get(relation.get("target_id"))
                if not source or not target:
                    continue
                same_identity = self._state_identity(source) and self._state_identity(
                    source
                ) == self._state_identity(target)
                values_differ = (
                    self._normalised_value(source)
                    and self._normalised_value(target)
                    and self._normalised_value(source) != self._normalised_value(target)
                )
                if same_identity and values_differ:
                    return True
            return False

        if slot_type == "CAUSE_PATH":
            by_id = {memory["id"]: memory for memory in memories}
            adjacency: Dict[str, List[str]] = defaultdict(list)
            for relation in relations:
                if not self._valid_causal_relation(relation, by_id):
                    continue
                adjacency[relation["source_id"]].append(relation["target_id"])
            if len(by_id) < 2 or not any(adjacency.values()):
                return False
            # All accepted support nodes must belong to one directed causal
            # component. Disconnected topical causal edges cannot jointly fill
            # one CAUSE_PATH slot; semantic start/goal matching is checked by
            # the slot-support gate.
            required = set(by_id)
            for root in required:
                reached, frontier = {root}, [root]
                while frontier:
                    current = frontier.pop(0)
                    for target in adjacency.get(current, []):
                        if target not in reached:
                            reached.add(target)
                            frontier.append(target)
                if required.issubset(reached):
                    return True
            return False

        if slot_type == "COMPARISON":
            groups = {}
            for memory in memories:
                value = self._normalised_value(memory)
                if value:
                    groups.setdefault(value, []).append(memory)
            return len(groups) >= 2 and all(groups.values())

        return False

    def _coverage_map(
        self,
        plan: Dict[str, Any],
        slot_support: Dict[str, List[str]],
        selected: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> Dict[str, bool]:
        return {
            slot["id"]: self._slot_covered(
                slot,
                slot_support.get(slot["id"], []),
                selected,
                relations,
            )
            for slot in plan["required_slots"]
        }

    @staticmethod
    def _has_unresolved_conflict(memories: Sequence[Dict[str, Any]]) -> bool:
        return any(memory.get("_status") == "conflicting" for memory in memories)

    def _operation_slot_support(
        self,
        slot: Dict[str, Any],
        result: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select structurally eligible operation output for one typed slot."""
        if not result:
            return []
        result_by_id = {memory["id"]: memory for memory in result}
        ranked = self._hybrid_search(
            str(slot.get("description") or slot["id"]),
            top_k=len(result_by_id),
            candidate_ids=result_by_id,
        )
        ranked = [
            result_by_id[memory["id"]]
            for memory in ranked
            if memory["id"] in result_by_id
        ]
        ranked.extend(
            memory
            for memory in result
            if memory["id"] not in {item["id"] for item in ranked}
        )
        slot_type = slot["type"]

        if slot_type == "DIRECT":
            allow_history = bool(slot.get("history"))
            return [
                memory
                for memory in ranked
                if (
                    memory.get("assertion_mode", "DIRECT") == "DIRECT"
                    and bool(self._memory_value(memory))
                    and self._memory_matches_slot_role(slot, memory)
                    and (
                        allow_history
                        or memory.get(
                            "_status",
                            self._belief_status.get(memory["id"], "active"),
                        )
                        != "superseded"
                    )
                )
            ][:4]

        if slot_type == "CURRENT_STATE":
            best = next(
                (memory for memory in ranked if self._is_state_head(memory)), None
            )
            if not best:
                return []
            identity = self._state_identity(best)
            return [
                memory
                for memory in ranked
                if self._state_identity(memory) == identity
                and self._is_state_head(memory)
                and self._memory_value(memory)
                and self._memory_matches_slot_role(slot, memory)
                and self._belief_status.get(memory["id"], "active")
                in {"active", "refined", "conflicting"}
            ]

        if slot_type == "TEMPORAL":
            axis = str(slot.get("time_axis") or "")
            fallback_axis = str(slot.get("fallback_axis") or "")
            return [
                memory
                for memory in ranked
                if (
                    self._memory_value(memory)
                    and self._memory_matches_slot_role(slot, memory)
                    and axis in VALID_TEMPORAL_AXES
                    and (
                        self._date_for(memory, axis)
                        or (
                            fallback_axis in VALID_TEMPORAL_AXES
                            and self._date_for(memory, fallback_axis)
                        )
                    )
                )
            ][:3]

        if slot_type == "TRANSITION":
            endpoint_ids = set()
            for relation in relations:
                if relation.get("type") not in {"SUPERSEDE", "REFINE"}:
                    continue
                source = result_by_id.get(relation.get("source_id"))
                target = result_by_id.get(relation.get("target_id"))
                if (
                    source
                    and target
                    and self._state_identity(source)
                    and self._state_identity(source) == self._state_identity(target)
                    and self._normalised_value(source) != self._normalised_value(target)
                ):
                    endpoint_ids.update((source["id"], target["id"]))
            return [memory for memory in ranked if memory["id"] in endpoint_ids and self._memory_matches_slot_role(slot, memory)]

        if slot_type == "CAUSE_PATH":
            endpoint_ids = set()
            for relation in relations:
                if self._valid_causal_relation(relation, result_by_id):
                    endpoint_ids.update((relation["source_id"], relation["target_id"]))
            return [memory for memory in ranked if memory["id"] in endpoint_ids and self._memory_matches_slot_role(slot, memory)]

        if slot_type == "COMPARISON":
            selected, values = [], set()
            for memory in ranked:
                value = self._normalised_value(memory)
                if value and value not in values and self._memory_matches_slot_role(slot, memory):
                    selected.append(memory)
                    values.add(value)
                if len(selected) >= 4:
                    break
            return selected if len(values) >= 2 else []
        return []

    def _semantic_slot_support(
        self,
        plan: Dict[str, Any],
        slot_support: Dict[str, List[str]],
        selected: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
        question: str = "",
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Validate exact slot answerability after structural coverage succeeds."""
        by_id = {memory["id"]: memory for memory in selected}
        candidate_ids = list(
            dict.fromkeys(
                memory_id
                for ids in slot_support.values()
                for memory_id in ids
                if memory_id in by_id
            )
        )
        candidate_refs = {
            memory_id: f"$candidate{index}"
            for index, memory_id in enumerate(candidate_ids)
        }
        ref_ids = {
            reference: memory_id for memory_id, reference in candidate_refs.items()
        }
        slot_by_id = {slot["id"]: slot for slot in plan["required_slots"]}
        allowed_refs: Dict[str, set] = {}
        payload = []
        for slot in plan["required_slots"]:
            refs = [
                candidate_refs[memory_id]
                for memory_id in slot_support.get(slot["id"], [])
                if memory_id in candidate_refs
            ]
            allowed_refs[slot["id"]] = set(refs)
            payload.append(
                {
                    "slot": slot,
                    "candidates": [
                        {
                            "ref": candidate_refs[memory_id],
                            "id": memory_id,
                            "claim": by_id[memory_id].get("claim", ""),
                            "kind": by_id[memory_id].get("kind", ""),
                            "value": self._memory_value(by_id[memory_id]),
                            "assertion_mode": by_id[memory_id].get(
                                "assertion_mode", "DIRECT"
                            ),
                            "origin_memory_id": by_id[memory_id].get(
                                "origin_memory_id", ""
                            ),
                            "state_identity": self._state_identity(by_id[memory_id]),
                            "event_time": by_id[memory_id].get("event_time", "UNKNOWN"),
                            "document_time": by_id[memory_id].get("document_time", ""),
                            "origin_document_time": by_id[memory_id].get(
                                "origin_document_time", ""
                            ),
                            "effective_event_time": self._date_for(
                                by_id[memory_id], "effective_event_time"
                            ),
                            "status": by_id[memory_id].get(
                                "_status",
                                self._belief_status.get(memory_id, "active"),
                            ),
                        }
                        for memory_id in slot_support.get(slot["id"], [])
                        if memory_id in by_id
                    ],
                }
            )
        payload.append(
            {
                "relations": [
                    {
                        "source_id": relation.get("source_id"),
                        "target_id": relation.get("target_id"),
                        "type": relation.get("type"),
                    }
                    for relation in relations
                    if relation.get("source_id") in candidate_refs
                    and relation.get("target_id") in candidate_refs
                ],
            }
        )
        prompt = SLOT_SUPPORT_GATE_PROMPT.format(
            slot_candidates=json.dumps(payload, ensure_ascii=False),
            question=question,
            option_coverage=json.dumps(
                plan.get("option_coverage") or [], ensure_ascii=False
            ),
        )
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                # This validator emits only slot ids and authorized refs.
                max_tokens=1024,
            )
            usage = self._response_usage(response, prompt)
            parsed = self._parse_json(response.content)
            if set(parsed) - {
                "supports",
                "query_sufficient",
                "uncovered_option_labels",
            } or not isinstance(
                parsed.get("supports"), list
            ) or not isinstance(
                parsed.get("query_sufficient"), bool
            ) or not isinstance(
                parsed.get("uncovered_option_labels"), list
            ):
                return {slot_id: [] for slot_id in slot_by_id}, {
                    **usage,
                    "called": True,
                    "valid": False,
                }
        except Exception:
            return {slot_id: [] for slot_id in slot_by_id}, {
                "called": True,
                "valid": False,
            }

        accepted: Dict[str, List[str]] = {slot_id: [] for slot_id in slot_by_id}
        seen_slots = set()
        for item in parsed["supports"]:
            slot_id = str(item.get("slot_id") or "")
            refs = item.get("refs") if isinstance(item.get("refs"), list) else []
            if slot_id not in slot_by_id or slot_id in seen_slots:
                return {candidate_id: [] for candidate_id in slot_by_id}, {
                    **usage,
                    "called": True,
                    "valid": False,
                }
            if any(str(ref) not in allowed_refs[slot_id] for ref in refs):
                return {candidate_id: [] for candidate_id in slot_by_id}, {
                    **usage,
                    "called": True,
                    "valid": False,
                }
            accepted[slot_id] = list(dict.fromkeys(ref_ids[str(ref)] for ref in refs))
            seen_slots.add(slot_id)
        option_mode = bool(plan.get("option_coverage"))
        query_sufficient = parsed.get("query_sufficient")
        uncovered = parsed.get("uncovered_option_labels")
        if option_mode and isinstance(uncovered, list):
            uncovered_labels = {str(label).upper() for label in uncovered}
            for coverage in plan.get("option_coverage", []):
                if coverage.get("label") not in uncovered_labels:
                    continue
                for slot_id in coverage.get("slot_ids", []):
                    slot = slot_by_id.get(slot_id, {})
                    if slot.get("option_label") == coverage.get("label"):
                        accepted[slot_id] = []
        return accepted, {
            **usage,
            "called": True,
            "valid": True,
            "query_sufficient": bool(query_sufficient),
            "uncovered_option_labels": uncovered if isinstance(uncovered, list) else [],
        }

    def _execute_operation(
        self,
        operation: Dict[str, Any],
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
        frame: QueryFrame = QueryFrame(),
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        op, result, relations, evidence_refs = operation["op"], [], [], []
        if op == "SEMANTIC_SEARCH":
            try:
                top_k = max(1, min(10, int(operation.get("top_k", 5))))
            except (TypeError, ValueError):
                top_k = 5
            result = self._semantic_operation_search(
                str(operation.get("query") or ""),
                top_k=top_k,
                strategy=str(operation.get("strategy") or "FOCAL"),
                frame=frame,
                option_queries=operation.get("option_queries") or [],
            )
        elif op == "LOCATE_ANCHOR":
            result = self._locate_anchor(str(operation.get("query") or ""), frame)
        elif op == "TEMPORAL_FILTER":
            result = self._temporal_filter(operation, outputs, seeds, frame)
        elif op == "RESOLVE_STATE":
            result = self._resolve_state(
                str(operation.get("query") or ""), seeds, frame
            )
        elif op == "FOLLOW_CAUSES":
            result, relations = self._follow_causes(operation, outputs, seeds)
        elif op == "VERIFY_EVIDENCE":
            result = self._resolve_refs(operation.get("memory_refs"), outputs, seeds)
            evidence_refs = [memory["id"] for memory in result]
        return result, relations, evidence_refs

    def _execute_plan(
        self,
        plan: Dict[str, Any],
        seeds: List[Dict[str, Any]],
        round_offset: int = 0,
        frame: QueryFrame = QueryFrame(),
        question: str = "",
    ) -> Dict[str, Any]:
        outputs: List[List[Dict[str, Any]]] = []
        used_relations: List[Dict[str, Any]] = []
        evidence_refs: List[str] = []
        slot_support = self._slot_seed_support(plan, seeds)
        selected: Dict[str, Dict[str, Any]] = {}
        slot_validation: List[Dict[str, Any]] = []
        confirmed_slot_support: Dict[str, List[str]] = {
            slot["id"]: [] for slot in plan["required_slots"]
        }
        validation_cache: Dict[
            Tuple[Tuple[str, Tuple[str, ...]], ...],
            Tuple[Dict[str, List[str]], Dict[str, Any]],
        ] = {}
        memory_limit = max(1, int(plan["max_memories"]))

        # Only seed_coverage can authorize initial selected memories.
        seed_by_id = {memory["id"]: memory for memory in seeds[:3]}
        for ids in slot_support.values():
            for memory_id in ids:
                if memory_id in seed_by_id:
                    if memory_id not in selected and len(selected) >= memory_limit:
                        continue
                    selected.setdefault(
                        memory_id, self._snapshot(seed_by_id[memory_id])
                    )

        beliefs: List[Dict[str, Any]] = []
        current_relations: List[Dict[str, Any]] = []
        coverage: Dict[str, bool] = {}
        sufficient = False

        def assess_sufficiency() -> None:
            nonlocal beliefs, current_relations, coverage, sufficient, selected, slot_support, confirmed_slot_support
            beliefs, belief_relations = self._reconstruct_beliefs(
                list(selected.values()),
                int(plan["max_memories"]),
            )
            current_relations = self._merge_relations(
                used_relations,
                belief_relations,
            )
            coverage = self._coverage_map(
                plan, slot_support, beliefs, current_relations
            )
            structural = (
                bool(coverage)
                and all(coverage.values())
                and not self._has_unresolved_conflict(beliefs)
            )
            semantic_pass = structural
            # Zero operations means no more retrieval, not that proposed seed
            # coverage is exempt from semantic validation.
            requires_validation = (
                getattr(self, "enable_slot_support_validation", True)
                and self._plan_requires_semantic_validation(plan)
            )
            if structural and requires_validation:
                validation_key = tuple(
                    sorted(
                        (slot_id, tuple(sorted(memory_ids)))
                        for slot_id, memory_ids in slot_support.items()
                    )
                )
                if validation_key in validation_cache:
                    cached_support, cached_usage = validation_cache[validation_key]
                    validated = self._snapshot(cached_support)
                    validation_usage = {
                        **cached_usage,
                        "called": False,
                        "cache_hit": True,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "latency": 0.0,
                    }
                else:
                    validated, validation_usage = self._semantic_slot_support(
                        plan,
                        slot_support,
                        beliefs,
                        current_relations,
                        question=question,
                    )
                    validation_cache[validation_key] = (
                        self._snapshot(validated),
                        self._snapshot(validation_usage),
                    )
                slot_validation.append(validation_usage)
                semantic_pass = bool(validation_usage.get("valid"))
                if validation_usage.get("valid"):
                    # Semantic progress is monotonic within one retrieval
                    # program. A later operation may add support, but a new
                    # validator response cannot erase an earlier accepted ID.
                    slot_support = {
                        slot_id: list(
                            dict.fromkeys(
                                (*confirmed_slot_support.get(slot_id, []), *validated.get(slot_id, []))
                            )
                        )
                        for slot_id in confirmed_slot_support
                    }
                    confirmed_slot_support = self._snapshot(slot_support)
                    accepted_ids = {
                        memory_id for ids in slot_support.values() for memory_id in ids
                    }
                    selected = {
                        memory_id: memory
                        for memory_id, memory in selected.items()
                        if memory_id in accepted_ids
                    }
                    beliefs, belief_relations = self._reconstruct_beliefs(
                        list(selected.values()),
                        int(plan["max_memories"]),
                    )
                    current_relations = self._merge_relations(
                        used_relations,
                        belief_relations,
                        allowed_ids=accepted_ids,
                    )
                    coverage = self._coverage_map(
                        plan, slot_support, beliefs, current_relations
                    )
                    semantic_pass = (
                        bool(coverage)
                        and all(coverage.values())
                        and validation_usage.get("query_sufficient") is True
                        and not self._has_unresolved_conflict(beliefs)
                    )
                else:
                    # valid=False means the validator response itself violated the
                    # schema or reference boundary, so none of it is trustworthy.
                    # A schema-valid but incomplete judgment uses valid=True with
                    # query_sufficient=False and retains its accepted slot supports
                    # through the branch above.
                    # Preserve structural candidates as best-effort context. A
                    # malformed optional validator response must not erase the
                    # evidence already authorized by the retrieval plan. It
                    # still cannot establish coverage or sufficiency.
                    beliefs, belief_relations = self._reconstruct_beliefs(
                        list(selected.values()),
                        int(plan["max_memories"]),
                    )
                    current_relations = self._merge_relations(
                        used_relations,
                        belief_relations,
                    )
                    coverage = self._coverage_map(
                        plan, slot_support, beliefs, current_relations
                    )
                    semantic_pass = False
            sufficient = (
                semantic_pass
                and bool(coverage)
                and all(coverage.values())
                and not self._has_unresolved_conflict(beliefs)
            )

        assess_sufficiency()
        trace: List[Dict[str, Any]] = []
        retrieved_ids = set(selected)

        for index, operation in enumerate(plan["operations"]):
            if sufficient:
                break
            before = dict(coverage)
            previous_ids = set(selected)
            operation_started = time.perf_counter()
            bounded_operation = self._snapshot(operation)
            if bounded_operation["op"] == "SEMANTIC_SEARCH":
                remaining = max(0, memory_limit - len(selected))
                try:
                    requested_top_k = max(
                        1, int(bounded_operation.get("top_k", remaining))
                    )
                except (TypeError, ValueError):
                    requested_top_k = max(1, remaining)
                if bounded_operation.get("strategy") == "SHARED_OPTIONS":
                    # A shared option operation must have room to expose the
                    # facts that distinguish the choices. This is still one
                    # bounded operation and never exceeds the plan's memory
                    # budget; it only prevents a model-emitted top_k=3 from
                    # starving a four-option evidence bundle.
                    option_count = len(bounded_operation.get("option_queries") or [])
                    requested_top_k = max(
                        requested_top_k,
                        min(self.HARD_MEMORY_LIMIT, max(4, option_count * 2)),
                    )
                bounded_operation["top_k"] = max(
                    1,
                    min(max(1, remaining), requested_top_k),
                )
            result, relations, requested_evidence = self._execute_operation(
                bounded_operation,
                outputs,
                seeds,
                frame,
            )

            # Rejected candidates must not permanently consume the context
            # budget. Bound this operation by currently accepted selections;
            # all returned IDs remain in telemetry and retrieval provenance.
            remaining = max(0, memory_limit - len(selected))
            bounded_result: List[Dict[str, Any]] = []
            for memory in result:
                memory_id = memory["id"]
                if memory_id in selected:
                    bounded_result.append(memory)
                elif remaining > 0:
                    bounded_result.append(memory)
                    remaining -= 1
                retrieved_ids.add(memory_id)
            result = bounded_result
            result_ids = {memory["id"] for memory in result}
            # Typed transition/comparison validation may inspect only relations
            # whose endpoints were returned by this operation. This preserves
            # pure retrieval while allowing a semantic-search fallback to expose
            # an already-stored relation between two returned versions.
            local_relations = [
                self._snapshot(relation)
                for relation in self._relations
                if relation.get("source_id") in result_ids
                and relation.get("target_id") in result_ids
            ]
            relations = self._merge_relations(relations, local_relations)
            relations = [
                relation
                for relation in relations
                if relation.get("source_id") in result_ids
                and relation.get("target_id") in result_ids
            ]
            requested_evidence = [
                memory_id for memory_id in requested_evidence if memory_id in result_ids
            ]
            operation_elapsed_ms = round(
                (time.perf_counter() - operation_started) * 1000.0, 3
            )

            # An operation may only introduce its own returned memories.
            outputs.append([self._snapshot(memory) for memory in result])
            used_relations.extend(self._snapshot(relations))
            evidence_refs.extend(requested_evidence)

            for slot_id in operation["produces"]:
                slot_support.setdefault(slot_id, [])
                slot = next(
                    candidate
                    for candidate in plan["required_slots"]
                    if candidate["id"] == slot_id
                )
                if bounded_operation["op"] == "TEMPORAL_FILTER":
                    # The fallback axis is part of the operation contract. Keep
                    # it attached to the typed slot so structural coverage uses
                    # exactly the same declared semantics as retrieval.
                    fallback_axis = str(
                        bounded_operation.get("fallback_axis") or ""
                    ).lower()
                    if fallback_axis in VALID_TEMPORAL_AXES:
                        slot["fallback_axis"] = fallback_axis
                supported = self._operation_slot_support(slot, result, relations)
                for memory in supported:
                    if memory["id"] not in selected and len(selected) >= memory_limit:
                        continue
                    if memory["id"] not in slot_support[slot_id]:
                        slot_support[slot_id].append(memory["id"])
                    selected.setdefault(memory["id"], self._snapshot(memory))

            assess_sufficiency()

            # LOCATE_ANCHOR is candidate discovery only. When the next planned
            # operation is TEMPORAL_FILTER for the same slot, do not stop on
            # the anchor's first dated candidate.
            if (
                sufficient
                and index + 1 < len(plan["operations"])
                and bounded_operation["op"] == "LOCATE_ANCHOR"
                and plan["operations"][index + 1]["op"] == "TEMPORAL_FILTER"
                and any(
                    slot_id in bounded_operation["produces"]
                    for slot_id in plan["operations"][index + 1]["produces"]
                )
            ):
                sufficient = False

            trace.append(
                {
                    "retrieval_round": round_offset + index + 1,
                    "operation_index": index,
                    "operation": operation["op"],
                    "input_refs": [
                        str(ref)
                        for field in (
                            "start",
                            "memory_refs",
                            "candidate_refs",
                            "anchor",
                            "end",
                        )
                        for ref in (
                            operation.get(field)
                            if isinstance(operation.get(field), list)
                            else (
                                [operation.get(field)] if operation.get(field) else []
                            )
                        )
                        if str(ref).startswith("$")
                    ],
                    "produces": list(operation["produces"]),
                    "output_ids": [memory["id"] for memory in result],
                    "operation_elapsed_ms": operation_elapsed_ms,
                    "new_memory_ids": [
                        memory["id"]
                        for memory in result
                        if memory["id"] not in previous_ids
                    ],
                    "slot_coverage_before": before,
                    "slot_coverage_after": dict(coverage),
                }
            )

        return {
            "selected": beliefs,
            "relations": current_relations,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
            "slot_support": slot_support,
            "slot_coverage": coverage,
            "sufficient": sufficient,
            "trace": trace,
            "operation_outputs": outputs,
            "slot_validation": slot_validation,
            "slot_validation_tokens": sum(
                int(item.get("total_tokens", 0) or 0) for item in slot_validation
            ),
            "slot_validation_latency": round(
                sum(float(item.get("latency", 0.0) or 0.0) for item in slot_validation),
                3,
            ),
        }

    @staticmethod
    def _plan_requires_semantic_validation(plan: Dict[str, Any]) -> bool:
        """Decide whether typed coverage needs an additional semantic check.

        This is deliberately contract-based, not a query-text classifier. A
        single non-inferential slot with one bounded operation is already
        checked by the executor's typed validators. Plans that combine roles,
        alternatives, history, relations, or declared inference still need
        the LLM sufficiency check because boolean structural coverage can be
        satisfied by a merely topical memory.
        """
        mode = str(plan.get("query_mode") or "DIRECT").upper()
        if mode in {"DECISION", "CAUSAL", "COMPARISON", "MULTI_OPTION"}:
            return True
        query_spec = plan.get("query_spec")
        # Hand-built/legacy plans do not carry enough semantic metadata to
        # justify skipping the check. Treat them conservatively.
        if not isinstance(query_spec, dict):
            return True
        if bool(query_spec.get("requires_inference")):
            return True
        slots = plan.get("required_slots") or []
        if len(slots) != 1 or len(plan.get("operations") or []) > 1:
            return True
        slot = slots[0]
        if slot.get("type") in {"TRANSITION", "COMPARISON", "CAUSE_PATH"}:
            return True
        if slot.get("history"):
            return True
        return str(slot.get("evidence_role") or "ANSWER").upper() in {
            "LONGITUDINAL_CONTEXT",
            "ACTION_RULE",
            "CONSTRAINT",
            "ALTERNATIVE",
            "CAUSE",
            "COMPARAND",
        }

    def _run_query_retrieval(
        self,
        question: str,
        initial_seeds: List[Dict[str, Any]],
        frame: QueryFrame,
        fast_supports: Optional[List[Dict[str, Any]]],
        gate: Dict[str, Any],
        planning_seeds: Optional[List[Dict[str, Any]]] = None,
        planning_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run one semantic route and deterministic retrieval operations."""
        query_tokens = {
            "fast_gate": int(gate.get("usage", {}).get("total_tokens", 0)),
            "planner": 0,
            "slot_validation": 0,
            "replan": 0,
            "answer": 0,
            "total": 0,
        }
        query_latency = {
            "fast_gate": round(
                float(gate.get("usage", {}).get("latency", 0.0) or 0.0), 3
            ),
            "planner": 0.0,
            "slot_validation": 0.0,
            "replan": 0.0,
            "answer": None,
            "retrieval_wall": 0.0,
            "total_wall": 0.0,
        }
        retrieval_started = time.perf_counter()
        plan: Dict[str, Any] = {
            "required_slots": [],
            "seed_coverage": [],
            "operations": [],
            "need_evidence": False,
            "budget_tier": "SMALL",
            "max_memories": 3,
            "planner_fallback": False,
            "valid": True,
        }
        replan: Optional[Dict[str, Any]] = None
        planner_called = False
        replan_called = False
        trace: List[Dict[str, Any]] = []
        slot_validation: List[Dict[str, Any]] = []
        operation_output_ids = set()
        evidence_refs: List[str] = []
        slot_support: Dict[str, List[str]] = {}
        slot_coverage: Dict[str, bool] = {}
        relations: List[Dict[str, Any]] = []
        sufficient = False
        beliefs: List[Dict[str, Any]] = []
        operation_candidates: Dict[str, Dict[str, Any]] = {}
        planned_seed_set = planning_seeds or initial_seeds
        controller = {
            "called": False,
            "route": "LEGACY",
            "support_ref": "",
            "answer": "",
            "fallback_reason": "",
            "usage": {},
        }

        if getattr(self, "enable_unified_controller", False):
            fast_supports, routed_plan, controller = self._semantic_controller(
                question,
                planned_seed_set,
                frame,
                context_map=planning_context,
            )
            usage = controller.get("usage", {})
            controller_stage = (
                "fast_gate" if controller.get("route") == "DIRECT" else "planner"
            )
            query_tokens[controller_stage] = int(
                usage.get("total_tokens", 0) or 0
            )
            query_latency[controller_stage] = round(
                float(usage.get("latency", 0.0) or 0.0), 3
            )
            if fast_supports is None:
                plan = routed_plan
                planner_called = True

        if fast_supports is not None:
            beliefs, relations = self._reconstruct_beliefs(
                fast_supports, min(3, len(fast_supports))
            )
            sufficient = bool(beliefs) and not self._has_unresolved_conflict(beliefs)
        else:
            if not getattr(self, "enable_unified_controller", False):
                planner_called = True
                plan, usage = self._plan_operations(
                    question,
                    planned_seed_set,
                    context_map=planning_context,
                )
                query_tokens["planner"] = int(usage.get("total_tokens", 0))
                query_latency["planner"] = round(
                    float(usage.get("latency", 0.0) or 0.0), 3
                )
            execution = self._execute_plan(
                plan, planned_seed_set, frame=frame, question=question
            )
            beliefs = execution["selected"]
            relations = execution["relations"]
            evidence_refs.extend(execution["evidence_refs"])
            slot_support = self._snapshot(execution["slot_support"])
            slot_coverage = dict(execution["slot_coverage"])
            sufficient = bool(execution["sufficient"])
            trace.extend(execution["trace"])
            slot_validation.extend(execution["slot_validation"])
            query_tokens["slot_validation"] += int(execution["slot_validation_tokens"])
            query_latency["slot_validation"] += float(
                execution.get("slot_validation_latency", 0.0) or 0.0
            )
            operation_output_ids.update(
                memory["id"]
                for output in execution["operation_outputs"]
                for memory in output
            )
            operation_candidates.update(
                {
                    memory["id"]: self._snapshot(memory)
                    for output in execution["operation_outputs"]
                    for memory in output
                }
            )

            missing_slots = [
                slot
                for slot in plan.get("required_slots", [])
                if not slot_coverage.get(slot["id"], False)
            ]
            unified = bool(getattr(self, "enable_unified_controller", False))
            broad_replan = bool(getattr(self, "enable_replan", False)) and not unified
            deterministic_recovery = bool(
                getattr(self, "enable_zero_result_recovery", False)
            )

            replan = None
            if not sufficient and deterministic_recovery and missing_slots:
                # Typed deterministic recovery is available on both controller
                # variants. It never changes slot semantics or performs hidden
                # retrieval; it only emits explicit operations for missing slots.
                replan = self._make_deterministic_recovery_plan(
                    missing_slots, question, plan
                )
                if replan:
                    replan_called = True
            elif not sufficient and broad_replan and missing_slots:
                # Legacy LLM-based replan (only when not in unified mode).
                replan_called = True
                replan_seeds = []
                for memory in (*beliefs, *planned_seed_set):
                    if all(item["id"] != memory["id"] for item in replan_seeds):
                        replan_seeds.append(self._snapshot(memory))
                    if len(replan_seeds) >= 8:
                        break
                replan, usage = self._plan_operations(
                    question,
                    replan_seeds,
                    missing_slots=missing_slots,
                    prior_trace=[
                        {
                            "operation": item.get("operation"),
                            "produces": item.get("produces", []),
                            "output_ids": item.get("output_ids", []),
                            "slot_coverage_before": item.get("slot_coverage_before", {}),
                            "slot_coverage_after": item.get("slot_coverage_after", {}),
                        }
                        for item in trace
                    ],
                    context_map=planning_context,
                    allow_repair=False,
                )
                query_tokens["replan"] = int(usage.get("total_tokens", 0))
                query_latency["replan"] = round(
                    float(usage.get("latency", 0.0) or 0.0), 3
                )

            if replan_called and replan:
                replan_seeds = []
                for memory in (*beliefs, *planned_seed_set):
                    if all(item["id"] != memory["id"] for item in replan_seeds):
                        replan_seeds.append(self._snapshot(memory))
                    if len(replan_seeds) >= 8:
                        break
                second = self._execute_plan(
                    replan,
                    replan_seeds,
                    round_offset=len(trace),
                    frame=frame,
                    question=question,
                )
                trace.extend(second["trace"])
                slot_validation.extend(second["slot_validation"])
                query_tokens["slot_validation"] += int(second["slot_validation_tokens"])
                query_latency["slot_validation"] += float(
                    second.get("slot_validation_latency", 0.0) or 0.0
                )
                operation_output_ids.update(
                    memory["id"]
                    for output in second["operation_outputs"]
                    for memory in output
                )
                operation_candidates.update(
                    {
                        memory["id"]: self._snapshot(memory)
                        for output in second["operation_outputs"]
                        for memory in output
                    }
                )
                evidence_refs.extend(second["evidence_refs"])

                by_id = {
                    memory["id"]: memory for memory in (*beliefs, *second["selected"])
                }
                beliefs, merged_relations = self._reconstruct_beliefs(
                    list(by_id.values()),
                    max(int(plan["max_memories"]), int(replan["max_memories"])),
                )
                relations = self._merge_relations(
                    relations,
                    second["relations"],
                    merged_relations,
                )
                for slot_id, memory_ids in second["slot_support"].items():
                    slot_support.setdefault(slot_id, [])
                    slot_support[slot_id].extend(
                        memory_id
                        for memory_id in memory_ids
                        if memory_id not in slot_support[slot_id]
                    )
                initial_coverage = dict(slot_coverage)
                second_coverage = dict(second["slot_coverage"])
                slot_coverage = {
                    slot["id"]: bool(initial_coverage.get(slot["id"]))
                    or bool(second_coverage.get(slot["id"]))
                    for slot in plan.get("required_slots", [])
                }
                sufficient = (
                    bool(second.get("sufficient"))
                    and bool(slot_coverage)
                    and all(slot_coverage.values())
                    and not self._has_unresolved_conflict(beliefs)
                )

        query_latency["slot_validation"] = round(query_latency["slot_validation"], 3)
        query_latency["retrieval_wall"] = round(
            time.perf_counter() - retrieval_started, 3
        )
        return {
            "query_tokens": query_tokens,
            "query_latency": query_latency,
            "controller": controller,
            "fast_supports": fast_supports,
            "precomputed_answer": str(controller.get("answer") or ""),
            "plan": plan,
            "replan": replan,
            "planner_called": planner_called,
            "replan_called": replan_called,
            "trace": trace,
            "slot_validation": slot_validation,
            "operation_output_ids": operation_output_ids,
            "operation_candidates": list(operation_candidates.values()),
            "planning_seeds": [self._snapshot(memory) for memory in planned_seed_set],
            "evidence_refs": evidence_refs,
            "slot_support": slot_support,
            "slot_coverage": slot_coverage,
            "relations": relations,
            "sufficient": sufficient,
            "beliefs": beliefs,
        }

    def _reconstruct_beliefs(
        self,
        memories: List[Dict[str, Any]],
        limit: int,
        prefer_active: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pure arbitration: inspect only relations whose endpoints are already selected."""
        selected = {m["id"]: self._snapshot(m) for m in memories}
        input_ids = set(selected)
        relevant = []
        for relation in self._relations:
            if (
                relation.get("source_id") not in input_ids
                or relation.get("target_id") not in input_ids
            ):
                continue
            relation_type = str(relation.get("type") or "").upper()
            if relation_type == "CAUSES" and not self._valid_causal_relation(
                relation, selected
            ):
                continue
            if relation_type in {"REFINE", "SUPERSEDE", "CONFLICT"}:
                source = selected[relation["source_id"]]
                target = selected[relation["target_id"]]
                if (
                    not self._state_identity(source)
                    or self._state_identity(source) != self._state_identity(target)
                ):
                    continue
            relevant.append(self._snapshot(relation))
        output = list(selected.values())
        for memory in output:
            memory["_status"] = self._belief_status.get(memory["id"], "active")

        # Arbitration may reorder/select within the authorized set, but never add IDs.
        output.sort(
            key=lambda m: (
                int(m.get("_best_effort_priority", 0)),
                (
                    m.get("_status") in {"active", "conflicting"}
                    if prefer_active
                    else True
                ),
                float(m.get("_score", 0.0)),
                self._recency_date(m),
                m["id"],
            ),
            reverse=True,
        )
        output = output[: max(0, int(limit))]
        output_ids = {memory["id"] for memory in output}
        if not output_ids.issubset(input_ids):
            raise AssertionError("arbitration expanded beyond its input IDs")
        return output, relevant
