"""Answerability gating and validated LLM retrieval planning."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    CLINICAL_SCOPES,
    RETRIEVAL_BUDGETS,
    SLOT_REQUIRED_FIELDS,
    VALID_EVIDENCE_ROLES,
    VALID_OPERATIONS,
    VALID_QUERY_MODES,
    VALID_REASONING_TYPES,
    VALID_SEARCH_STRATEGIES,
    VALID_SLOT_TYPES,
    VALID_TEMPORAL_AXES,
    VALID_TEMPORAL_RELATIONS,
    QueryFrame,
)
from .prompts import (
    ANSWERABILITY_GATE_PROMPT,
    COMPACT_CONTROLLER_PROMPT,
    MEDICAL_PLANNER_GUIDANCE,
    TYPED_PLANNER_PROMPT,
)


class PlanningMixin:
    """Defines requirements and a bounded retrieval program without executing it."""

    def _response_usage(self, response: Any, prompt: str) -> Dict[str, Any]:
        input_tokens = int(getattr(response, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response, "output_tokens", 0) or 0)
        if not input_tokens:
            input_tokens = len(self._tokenizer.encode(prompt))
        if not output_tokens:
            output_tokens = len(
                self._tokenizer.encode(str(getattr(response, "content", "")))
            )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency": float(getattr(response, "latency", 0.0) or 0.0),
        }

    def _compact_seed(self, memory: Dict[str, Any], index: int) -> Dict[str, Any]:
        return {
            "ref": f"$seed{index}",
            "id": memory["id"],
            "claim": memory["claim"],
            "kind": memory["kind"],
            "subject": memory.get("subject", ""),
            "scope": memory.get("scope", ""),
            "state_key": memory.get("state_key", ""),
            "object_anchor": memory.get("object_anchor", ""),
            "assertion_mode": memory.get("assertion_mode", "DIRECT"),
            "state_identity": self._state_identity(memory),
            "value": self._memory_value(memory),
            "verbatim_value": memory.get("verbatim_value", ""),
            "event_time": memory.get("event_time", "UNKNOWN"),
            "time_expression": memory.get("time_expression", ""),
            "document_time": memory.get("document_time", ""),
            "origin_document_time": memory.get("origin_document_time", ""),
            "effective_event_time": self._date_for(memory, "effective_event_time"),
            "source_speakers": list(memory.get("source_speakers", [])),
            "planning_tags": list(memory.get("planning_tags", [])),
            "decision_salience": float(memory.get("decision_salience", 0.0) or 0.0),
            "status": memory.get(
                "_status", self._belief_status.get(memory["id"], "active")
            ),
        }

    def _compact_planner_seed(
        self, memory: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """Keep only fields that can change a retrieval program."""
        return {
            "ref": f"$seed{index}",
            "claim": str(memory.get("claim") or "")[:300],
            "kind": memory.get("kind", "FACT"),
            "subject": memory.get("subject", ""),
            "scope": memory.get("scope", ""),
            "state_key": memory.get("state_key", ""),
            "object_anchor": memory.get("object_anchor", ""),
            "value": self._memory_value(memory)[:180],
            "event_time": memory.get("event_time", "UNKNOWN"),
            "document_time": memory.get("document_time", ""),
            "origin_document_time": memory.get("origin_document_time", ""),
            "planning_tags": list(memory.get("planning_tags", [])),
            "status": memory.get(
                "_status", self._belief_status.get(memory.get("id"), "active")
            ),
        }

    @staticmethod
    def _parse_query_spec(
        parsed: Dict[str, Any],
        slots: List[Dict[str, Any]],
        query_mode: str,
    ) -> Dict[str, Any]:
        """Canonicalize semantic metadata without turning it into retrieval logic."""
        raw = parsed.get("query_spec")
        raw = raw if isinstance(raw, dict) else {}
        entities = raw.get("target_entities") or []
        if isinstance(entities, str):
            entities = [entities]
        answer_type = str(raw.get("answer_type") or slots[0]["type"]).upper()
        if answer_type not in VALID_SLOT_TYPES:
            answer_type = slots[0]["type"]
        default_reasoning = {
            "DECISION": "DECISION",
            "CAUSAL": "CAUSAL",
            "COMPARISON": "COMPARISON",
            "MULTI_OPTION": "SYNTHESIS",
        }.get(query_mode, "NONE")
        reasoning = str(raw.get("reasoning") or default_reasoning).upper()
        if reasoning not in VALID_REASONING_TYPES:
            reasoning = default_reasoning
        temporal = raw.get("temporal")
        temporal = temporal if isinstance(temporal, dict) else {}
        temporal_slot = next(
            (slot for slot in slots if slot["type"] == "TEMPORAL"), None
        )
        axis = str(
            temporal.get("axis") or (temporal_slot or {}).get("time_axis") or ""
        ).lower()
        relation = str(
            temporal.get("relation")
            or (temporal_slot or {}).get("temporal_relation")
            or ""
        ).upper()
        if axis not in VALID_TEMPORAL_AXES:
            axis = ""
        if relation not in VALID_TEMPORAL_RELATIONS:
            relation = ""
        return {
            "target_entities": [
                str(value).strip()[:80] for value in entities if str(value).strip()
            ][:6],
            "target_property": str(raw.get("target_property") or "").strip()[:160],
            "answer_type": answer_type,
            "reasoning": reasoning,
            "requires_inference": reasoning != "NONE",
            "temporal": {
                "axis": axis,
                "relation": relation,
                "anchor": str(temporal.get("anchor") or "").strip()[:80],
            },
        }

    @staticmethod
    def _derive_budget_tier(
        query_mode: str,
        slots: List[Dict[str, Any]],
        operations: List[Dict[str, Any]],
        query_spec: Dict[str, Any],
    ) -> str:
        """Let executable structure, not LLM self-scoring, own the budget."""
        slot_types = {slot["type"] for slot in slots}
        if (
            query_mode in {"MULTI_OPTION", "CAUSAL"}
            or "CAUSE_PATH" in slot_types
            or len(slots) > 2
            or len(operations) > 2
        ):
            return "LARGE"
        if (
            query_mode == "DECISION"
            or query_spec.get("reasoning")
            in {
                "SYNTHESIS",
                "DECISION",
                "COMPARISON",
            }
            or slot_types.intersection({"TRANSITION", "COMPARISON"})
            or len(slots) > 1
            or len(operations) > 1
        ):
            return "MEDIUM"
        return "SMALL"

    @staticmethod
    def _query_mode_from_spec(query_spec: Dict[str, Any]) -> str:
        reasoning = query_spec.get("reasoning")
        answer_type = query_spec.get("answer_type")
        if reasoning == "DECISION":
            return "DECISION"
        if reasoning == "CAUSAL":
            return "CAUSAL"
        if reasoning == "COMPARISON" or answer_type == "COMPARISON":
            return "COMPARISON"
        if answer_type == "TEMPORAL":
            return "TEMPORAL"
        if answer_type in {"CURRENT_STATE", "TRANSITION"}:
            return "STATE"
        return "DIRECT"

    @staticmethod
    def _enrich_operation_query(
        operation: Dict[str, Any], query_spec: Dict[str, Any]
    ) -> None:
        """Carry explicit semantic targets into retrieval without classifying text."""
        if operation.get("op") not in {
            "SEMANTIC_SEARCH",
            "LOCATE_ANCHOR",
            "RESOLVE_STATE",
            "TEMPORAL_FILTER",
        }:
            return
        query = str(operation.get("query") or "").strip()
        lowered = query.casefold()
        missing_entities = [
            entity
            for entity in query_spec.get("target_entities", [])
            if entity.casefold() not in {"patient", "user", "person"}
            and entity.casefold() not in lowered
        ]
        if missing_entities:
            operation["query"] = (
                f"{query} | target: {', '.join(missing_entities)}".strip(" |")
            )

    def _planning_seed_set(
        self,
        question: str,
        recalled: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return exactly the top-three authorized RRF seeds for planning.

        The context map may suggest retrieval roles, but it cannot promote a
        profile memory into a seed. Any memory outside this set must be
        introduced by a validated operation and therefore has explicit
        retrieval provenance.
        """
        del question
        selected: List[Dict[str, Any]] = []
        for memory in recalled:
            if memory and all(item["id"] != memory["id"] for item in selected):
                selected.append(self._snapshot(memory))
            if len(selected) >= 3:
                break
        return selected

    def _planning_context_map(
        self, question: str, limit: int = 12
    ) -> List[Dict[str, Any]]:
        """Return compact routing hints; these are never valid memory refs."""
        by_id = {memory["id"]: memory for memory in self._memories}
        anchor_ids = list(
            dict.fromkeys(
                memory_id
                for group in (
                    self._profile_pack.get("planning_anchors", []),
                    self._profile_pack.get("trajectory_anchors", []),
                )
                for memory_id in group
                if memory_id in by_id
            )
        )
        if not anchor_ids:
            return []
        related = self._hybrid_search(
            question,
            top_k=min(limit, len(anchor_ids)),
            candidate_ids=set(anchor_ids),
        )
        selected: List[Dict[str, Any]] = []

        def add(memory: Dict[str, Any]) -> None:
            if memory and all(item["id"] != memory["id"] for item in selected):
                selected.append(memory)

        for memory in related[:6]:
            add(memory)
        tag_order = (
            "IDENTITY",
            "RISK",
            "CONSTRAINT",
            "EXPOSURE",
            "RESPONSE",
            "TRAJECTORY",
            "PREFERENCE",
            "RESOURCE",
            "PLAN",
            "STATE",
        )
        anchors = [by_id[memory_id] for memory_id in anchor_ids]
        for tag in tag_order:
            candidate = next(
                (
                    memory
                    for memory in anchors
                    if tag in memory.get("planning_tags", [])
                ),
                None,
            )
            if candidate:
                add(candidate)
            if len(selected) >= limit:
                break
        return [
            {
                "hint": index,
                "planning_tags": list(memory.get("planning_tags", [])),
                "decision_salience": float(memory.get("decision_salience", 0.0) or 0.0),
                "subject": memory.get("subject", ""),
                "scope": memory.get("scope", ""),
                "state_key": memory.get("state_key", ""),
                "value": self._memory_value(memory)[:180],
                "claim": str(memory.get("claim") or "")[:240],
                "document_time": memory.get("document_time", ""),
            }
            for index, memory in enumerate(selected[:limit])
        ]

    def _memory_satisfies_frame(
        self,
        memory: Dict[str, Any],
        frame: QueryFrame,
        include_dates: bool = True,
        include_entities: bool = True,
    ) -> bool:
        if (
            include_dates
            and frame.dates
            and not self._memory_matches_dates(memory, frame.dates)
        ):
            return False
        if frame.speaker_role and not self._speaker_role_match(
            memory, frame.speaker_role
        ):
            return False
        if include_entities and frame.hard_entities:
            values = {
                str(value).strip().lower()
                for value in (
                    *memory.get("entities", []),
                    *memory.get("scope_entities", []),
                    memory.get("subject", ""),
                    memory.get("object_anchor", ""),
                )
                if str(value).strip()
            }
            # Multiple explicit objects often denote comparison groups. A
            # candidate belongs to the frame when it supports at least one
            # group; requiring every object on one atomic memory is impossible.
            if not set(frame.hard_entities).intersection(values):
                return False
        return True

    def _has_competing_active_value(self, memory: Dict[str, Any]) -> bool:
        identity = self._state_identity(memory)
        if not identity:
            return False
        by_id = {item["id"]: item for item in self._memories}
        values = {
            self._state_value_signature(by_id[memory_id])
            for memory_id in self._state_heads.get(identity, [])
            if memory_id in by_id
            and self._belief_status.get(memory_id, "active")
            in {"active", "conflicting"}
            and self._state_value_signature(by_id[memory_id])
        }
        return len(values) > 1

    def _validate_fast_support(
        self,
        reference: Any,
        seeds: List[Dict[str, Any]],
        frame: QueryFrame,
    ) -> Optional[List[Dict[str, Any]]]:
        match = re.fullmatch(r"\$seed(\d+)", str(reference or ""))
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None
        memory = seeds[int(match.group(1))]
        status = memory.get("_status", self._belief_status.get(memory["id"], "active"))
        if memory.get("assertion_mode", "DIRECT") != "DIRECT":
            return None
        if not self._memory_value(memory) or status in {
            "superseded",
            "conflicting",
        }:
            return None
        if not self._memory_satisfies_frame(memory, frame):
            return None
        if self._has_competing_active_value(memory):
            return None
        support = self._snapshot(memory)
        if self._has_unresolved_conflict([support]):
            return None
        return [support]

    def _answerability_gate(
        self,
        question: str,
        seeds: List[Dict[str, Any]],
        frame: QueryFrame,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        telemetry = {
            "called": False,
            "pass": False,
            "support_ref": "",
            "rejection_reason": "",
            "usage": {},
        }
        if not seeds:
            telemetry["rejection_reason"] = "no_seeds"
            return None, telemetry
        compact = [
            self._compact_seed(memory, index) for index, memory in enumerate(seeds[:3])
        ]
        prompt = ANSWERABILITY_GATE_PROMPT.format(
            seeds=json.dumps(compact, ensure_ascii=False),
            question=question,
        )
        telemetry["called"] = True
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                # This gate returns only {pass, support_ref}; do not make a
                # control request inherit the answer model's large budget.
                max_tokens=256,
            )
            telemetry["usage"] = self._response_usage(response, prompt)
            parsed = self._parse_json(response.content)
        except Exception:
            telemetry["rejection_reason"] = "gate_error"
            return None, telemetry
        if set(parsed) - {"pass", "support_ref"}:
            telemetry["rejection_reason"] = "invalid_schema"
            return None, telemetry
        if parsed.get("pass") is not True:
            telemetry["rejection_reason"] = "model_insufficient"
            return None, telemetry
        support_ref = str(parsed.get("support_ref") or "")
        supports = self._validate_fast_support(support_ref, seeds, frame)
        telemetry.update(
            {
                "pass": supports is not None,
                "support_ref": support_ref,
                "rejection_reason": (
                    "" if supports is not None else "structural_rejection"
                ),
            }
        )
        return supports, telemetry

    @staticmethod
    def _fallback_plan(question: str, reason: str) -> Dict[str, Any]:
        return {
            "query_spec": {
                "target_entities": [],
                "target_property": question[:160],
                "answer_type": "DIRECT",
                "reasoning": "NONE",
                "requires_inference": False,
                "temporal": {"axis": "", "relation": "", "anchor": ""},
            },
            "query_mode": "DIRECT",
            "required_slots": [
                {
                    "id": "fallback_direct",
                    "description": "best direct evidence",
                    "type": "DIRECT",
                    "required_fields": ["value"],
                    "time_axis": "",
                    "history": False,
                    "evidence_role": "ANSWER",
                    "option_label": "",
                },
            ],
            "seed_coverage": [],
            "operations": [
                {
                    "op": "SEMANTIC_SEARCH",
                    "query": question,
                    "top_k": 3,
                    "produces": ["fallback_direct"],
                },
            ],
            "need_evidence": True,
            "budget_tier": "SMALL",
            "max_memories": RETRIEVAL_BUDGETS["SMALL"]["max_memories"],
            "planner_fallback": True,
            "fallback_reason": reason,
            "valid": True,
            "option_coverage": [],
        }

    def _finalize_fallback_plan(
        self, plan: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Normalize visible-option fallback into one shared evidence plan."""
        options = self._question_options(question)
        if not options:
            return plan
        stem = self._question_stem(question)
        plan.update(
            {
                "query_spec": {
                    "target_entities": [],
                    "target_property": stem[:160],
                    "answer_type": "DIRECT",
                    "reasoning": "SYNTHESIS",
                    "requires_inference": True,
                    "temporal": {"axis": "", "relation": "", "anchor": ""},
                },
                "query_mode": "MULTI_OPTION",
                "required_slots": [
                    {
                        "id": "mq_context",
                        "description": (
                            "participant-specific current and longitudinal context "
                            "needed to evaluate all visible options"
                        ),
                        "type": "DIRECT",
                        "required_fields": ["value"],
                        "time_axis": "",
                        "temporal_relation": "",
                        "history": False,
                        "evidence_role": "FOCAL_STATE",
                        "option_label": "",
                    },
                    {
                        "id": "mq_option_evidence",
                        "description": (
                            "participant-specific guidance, constraints, alternatives, "
                            "and safety facts needed to evaluate the options"
                        ),
                        "type": "DIRECT",
                        "required_fields": ["value"],
                        "time_axis": "",
                        "temporal_relation": "",
                        "history": False,
                        "evidence_role": "ACTION_RULE",
                        "option_label": "",
                    },
                ],
                "seed_coverage": [],
                "operations": [
                    {
                        "op": "SEMANTIC_SEARCH",
                        "query": (
                            f"{stem} participant-specific current state, longitudinal "
                            "evidence, guidance, constraints, and acceptable alternatives "
                            + " ".join(options.values())
                        ),
                        "top_k": 8,
                        "strategy": "SHARED_OPTIONS",
                        "evidence_roles": ["FOCAL_STATE", "ACTION_RULE"],
                        "produces": ["mq_context", "mq_option_evidence"],
                    },
                ],
                "option_coverage": [],
                "budget_tier": "LARGE",
                "max_memories": RETRIEVAL_BUDGETS["LARGE"]["max_memories"],
            }
        )
        return plan

    def _preserve_fallback_semantics(
        self,
        plan: Dict[str, Any],
        parsed: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        """Carry a valid controller query spec through deterministic fallback."""
        plan = self._finalize_fallback_plan(plan, question)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("query_spec"), dict):
            return plan
        declared = str(parsed.get("query_mode") or plan.get("query_mode") or "DIRECT")
        query_spec = self._parse_query_spec(
            parsed, plan.get("required_slots") or [], declared
        )
        plan["query_spec"] = query_spec
        if query_spec.get("requires_inference"):
            # Preserve the planner's semantic signal even when its slot JSON
            # was invalid, but make deterministic fallback retrieve a bounded
            # reasoning bundle rather than a three-item direct lookup.
            plan["budget_tier"] = "MEDIUM"
            plan["max_memories"] = RETRIEVAL_BUDGETS["MEDIUM"]["max_memories"]
            for operation in plan.get("operations") or []:
                if operation.get("op") == "SEMANTIC_SEARCH":
                    operation["strategy"] = "DECISION_BUNDLE"
                    operation["top_k"] = max(
                        int(operation.get("top_k", 3) or 3),
                        RETRIEVAL_BUDGETS["MEDIUM"]["max_memories"],
                    )
        if not self._question_options(question):
            query_mode = self._query_mode_from_spec(query_spec)
            plan["query_mode"] = query_mode
            # Preserve a temporal contract even when the controller's slots
            # were malformed. A generic DIRECT search cannot answer an
            # earliest/latest/date-localization question reliably.
            if query_mode == "TEMPORAL":
                temporal = query_spec.get("temporal") or {}
                axis = str(temporal.get("axis") or "").lower()
                relation = str(temporal.get("relation") or "").upper()
                if axis not in VALID_TEMPORAL_AXES:
                    axis = "effective_event_time"
                if relation not in VALID_TEMPORAL_RELATIONS:
                    relation = "EXACT"
                description = (
                    str(query_spec.get("target_property") or "").strip()
                    or question[:180]
                )
                slot_id = "fallback_temporal"
                slot = {
                    "id": slot_id,
                    "description": description,
                    "type": "TEMPORAL",
                    "required_fields": ["value", axis],
                    "time_axis": axis,
                    "temporal_relation": relation,
                    "history": False,
                    "evidence_role": "TEMPORAL",
                    "option_label": "",
                }
                anchor = str(temporal.get("anchor") or "").strip()
                if relation in {"EARLIEST", "LATEST"}:
                    operations = [
                        {
                            "op": "LOCATE_ANCHOR",
                            "query": description,
                            "produces": [slot_id],
                        },
                        {
                            "op": "TEMPORAL_FILTER",
                            "relation": relation,
                            "axis": axis,
                            "fallback_axis": (
                                "document_time"
                                if axis in {"event_time", "effective_event_time"}
                                else ""
                            ),
                            "candidate_refs": ["$0"],
                            "query": description,
                            "produces": [slot_id],
                        },
                    ]
                else:
                    operation = {
                        "op": "TEMPORAL_FILTER",
                        "relation": relation,
                        "axis": axis,
                        "fallback_axis": "",
                        "query": description,
                        "produces": [slot_id],
                    }
                    if anchor:
                        operation["anchor"] = anchor
                    operations = [operation]
                plan.update(
                    {
                        "required_slots": [slot],
                        "seed_coverage": [],
                        "operations": operations,
                        "need_evidence": False,
                        "budget_tier": "MEDIUM",
                        "max_memories": RETRIEVAL_BUDGETS["MEDIUM"]["max_memories"],
                        "planner_fallback": True,
                        "fallback_reason": "temporal_contract_repair",
                        "valid": True,
                        "option_coverage": [],
                    }
                )
        return plan

    @staticmethod
    def _fallback_missing_plan(
        question: str,
        reason: str,
        missing_slots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build bounded best-effort operations without weakening typed slots."""
        slots = [dict(slot) for slot in missing_slots]
        operations: List[Dict[str, Any]] = []
        produced = set()
        for slot in slots:
            if len(operations) >= RETRIEVAL_BUDGETS["LARGE"]["max_operations"]:
                break
            slot_id = slot["id"]
            description = str(slot.get("description") or slot_id or question)
            slot_type = str(slot.get("type") or "DIRECT")
            if slot_type == "CURRENT_STATE":
                operation = {
                    "op": "RESOLVE_STATE",
                    "query": description,
                    "produces": [slot_id],
                }
            elif slot_type == "TEMPORAL":
                relation = str(slot.get("temporal_relation") or "EXACT").upper()
                if relation in {"EARLIEST", "LATEST"} and len(operations) <= 2:
                    anchor_index = len(operations)
                    operations.append(
                        {
                            "op": "LOCATE_ANCHOR",
                            "query": description,
                            "produces": [slot_id],
                        }
                    )
                    operation = {
                        "op": "TEMPORAL_FILTER",
                        "relation": relation,
                        "axis": str(slot.get("time_axis") or "event_time"),
                        "fallback_axis": "",
                        "candidate_refs": [f"${anchor_index}"],
                        "query": description,
                        "produces": [slot_id],
                    }
                else:
                    operation = {
                        "op": "TEMPORAL_FILTER",
                        "relation": relation,
                        "axis": str(slot.get("time_axis") or "event_time"),
                        "fallback_axis": "",
                        "query": description,
                        "produces": [slot_id],
                    }
            else:
                role = str(slot.get("evidence_role") or "ANSWER")
                strategy = (
                    "TRAJECTORY"
                    if role in {"LONGITUDINAL_CONTEXT", "CAUSE"}
                    else (
                        "DECISION_BUNDLE"
                        if role
                        in {"FOCAL_STATE", "ACTION_RULE", "CONSTRAINT", "ALTERNATIVE"}
                        else "FOCAL"
                    )
                )
                operation = {
                    "op": "SEMANTIC_SEARCH",
                    "query": description,
                    "top_k": 3,
                    "strategy": strategy,
                    "evidence_roles": [role],
                    "produces": [slot_id],
                }
            operations.append(operation)
            produced.add(slot_id)

            if (
                slot_type == "CAUSE_PATH"
                and len(operations) < RETRIEVAL_BUDGETS["LARGE"]["max_operations"]
            ):
                operations.append(
                    {
                        "op": "FOLLOW_CAUSES",
                        "start": [f"${len(operations) - 1}"],
                        "direction": "OUT",
                        "depth": 2,
                        "goal": description,
                        "produces": [slot_id],
                    }
                )

        # If the planner emitted more roles than the operation budget permits,
        # the final search remains explicit about every uncovered role. The
        # semantic support gate still decides whether any result covers them.
        unproduced = [slot for slot in slots if slot["id"] not in produced]
        if unproduced and operations:
            last_producer = next(
                (
                    operation
                    for operation in reversed(operations)
                    if operation["op"] in {"SEMANTIC_SEARCH", "RESOLVE_STATE"}
                ),
                None,
            )
            if last_producer is not None:
                last_producer["produces"] = list(
                    dict.fromkeys(
                        [
                            *last_producer["produces"],
                            *(slot["id"] for slot in unproduced),
                        ]
                    )
                )
                last_producer["query"] = "; ".join(
                    [last_producer["query"]]
                    + [
                        str(slot.get("description") or slot["id"])
                        for slot in unproduced
                    ]
                )

        slot_types = {slot.get("type") for slot in slots}
        if len(operations) > 2 or len(slots) > 2 or "CAUSE_PATH" in slot_types:
            budget_tier = "LARGE"
        elif (
            len(operations) > 1
            or len(slots) > 1
            or slot_types
            & {
                "TRANSITION",
                "COMPARISON",
            }
        ):
            budget_tier = "MEDIUM"
        else:
            budget_tier = "SMALL"
        fallback_mode = (
            "MULTI_OPTION"
            if any(slot.get("option_label") for slot in slots)
            else (
                "DECISION"
                if {
                    "FOCAL_STATE",
                    "LONGITUDINAL_CONTEXT",
                    "ACTION_RULE",
                }.issubset({slot.get("evidence_role") for slot in slots})
                else (
                    "TEMPORAL"
                    if any(slot.get("type") == "TEMPORAL" for slot in slots)
                    else "DIRECT"
                )
            )
        )
        fallback_spec = PlanningMixin._parse_query_spec({}, slots, fallback_mode)
        fallback_spec["target_property"] = question[:160]
        return {
            "query_spec": fallback_spec,
            "query_mode": fallback_mode,
            "required_slots": slots,
            "seed_coverage": [],
            "operations": operations,
            "need_evidence": True,
            "budget_tier": budget_tier,
            "max_memories": RETRIEVAL_BUDGETS[budget_tier]["max_memories"],
            "planner_fallback": True,
            "fallback_reason": reason,
            "valid": True,
            "option_coverage": [],
        }

    @staticmethod
    def _preserves_missing_slot_contract(
        plan: Dict[str, Any], missing_slots: List[Dict[str, Any]]
    ) -> bool:
        expected = {slot["id"]: slot for slot in missing_slots}
        actual = {slot["id"]: slot for slot in plan.get("required_slots", [])}
        if set(actual) != set(expected):
            return False
        for slot_id, prior in expected.items():
            candidate = actual[slot_id]
            if candidate.get("type") != prior.get("type"):
                return False
            if set(candidate.get("required_fields", [])) != set(
                prior.get("required_fields", [])
            ):
                return False
            if str(candidate.get("time_axis") or "") != str(
                prior.get("time_axis") or ""
            ):
                return False
            if bool(candidate.get("history", False)) != bool(
                prior.get("history", False)
            ):
                return False
            if str(candidate.get("temporal_relation") or "") != str(
                prior.get("temporal_relation") or ""
            ):
                return False
            if str(candidate.get("evidence_role") or "ANSWER") != str(
                prior.get("evidence_role") or "ANSWER"
            ):
                return False
            if str(candidate.get("option_label") or "") != str(
                prior.get("option_label") or ""
            ):
                return False
        return True

    @staticmethod
    def _valid_ref(value: Any, seed_count: int, operation_index: int) -> bool:
        text = str(value or "")
        seed_match = re.fullmatch(r"\$seed(\d+)", text)
        output_match = re.fullmatch(r"\$(\d+)", text)
        if seed_match:
            return int(seed_match.group(1)) < seed_count
        if output_match:
            return int(output_match.group(1)) < operation_index
        return False

    @staticmethod
    def _parse_required_slots(
        parsed: Dict[str, Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        if not isinstance(parsed.get("required_slots"), list):
            return None, "missing_required_slots"
        slots, slot_ids = [], set()
        for raw in parsed["required_slots"]:
            if not isinstance(raw, dict):
                return None, "invalid_slot"
            slot_id = str(raw.get("id") or "").strip()
            slot_type = str(raw.get("type") or "").upper()
            raw_fields = raw.get("required_fields", [])
            if isinstance(raw_fields, str):
                raw_fields = [raw_fields]
            fields = {str(value).strip() for value in raw_fields if str(value).strip()}
            if not slot_id or slot_id in slot_ids or slot_type not in VALID_SLOT_TYPES:
                return None, "invalid_slot"
            if slot_type == "TEMPORAL":
                axis = str(raw.get("time_axis") or "").lower()
                temporal_relation = str(raw.get("temporal_relation") or "").upper()
                if axis not in VALID_TEMPORAL_AXES:
                    return None, "invalid_temporal_slot"
                if temporal_relation not in VALID_TEMPORAL_RELATIONS:
                    return None, "invalid_temporal_relation"
                allowed_fields = {"value", axis}
                if fields - allowed_fields:
                    return None, "invalid_slot_fields"
                # The temporal axis determines the second required field.
                # This is deterministic schema repair, not intent inference.
                fields = allowed_fields
            else:
                expected_fields = SLOT_REQUIRED_FIELDS[slot_type]
                if fields - expected_fields:
                    return None, "invalid_slot_fields"
                # Compact controller responses often omit fields implied by
                # the declared type. Canonicalize them before validation.
                fields = set(expected_fields)
            slot_ids.add(slot_id)
            slots.append(
                {
                    "id": slot_id,
                    "description": str(raw.get("description") or slot_id),
                    "type": slot_type,
                    "required_fields": sorted(fields),
                    "time_axis": str(raw.get("time_axis") or "").lower(),
                    "temporal_relation": (
                        temporal_relation if slot_type == "TEMPORAL" else ""
                    ),
                    "history": bool(raw.get("history", False))
                    and slot_type == "DIRECT",
                    "evidence_role": str(raw.get("evidence_role") or "ANSWER").upper(),
                    "option_label": str(raw.get("option_label") or "").upper(),
                }
            )
            if slots[-1]["evidence_role"] not in VALID_EVIDENCE_ROLES:
                return None, "invalid_evidence_role"
        if not slots:
            return None, "empty_required_slots"
        return slots, ""

    def _validate_plan(
        self,
        parsed: Dict[str, Any],
        question: str,
        seed_count: int,
        missing_slots: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        slots, reason = self._parse_required_slots(parsed)
        if slots is None:
            return None, reason
        slot_ids = {slot["id"] for slot in slots}

        declared_mode = str(parsed.get("query_mode") or "DIRECT").upper()
        if declared_mode not in VALID_QUERY_MODES:
            declared_mode = "DIRECT"
        query_spec = self._parse_query_spec(parsed, slots, declared_mode)
        query_mode = (
            self._query_mode_from_spec(query_spec)
            if isinstance(parsed.get("query_spec"), dict)
            else declared_mode
        )

        # A temporal query cannot be executable as an untyped DIRECT
        # zero-operation plan. This catches a controller response whose
        # semantic metadata says TEMPORAL but whose slot stayed generic.
        if query_mode == "TEMPORAL" and not any(
            slot["type"] == "TEMPORAL" for slot in slots
        ):
            return None, "temporal_query_requires_temporal_slot"

        question_options = self._question_options(question)
        shared_option_evidence = bool(question_options)
        option_coverage = []
        if question_options:
            query_mode = "MULTI_OPTION"
            parsed["query_mode"] = query_mode
            for slot in slots:
                slot["option_label"] = ""
                if slot.get("evidence_role") == "OPTION":
                    slot["evidence_role"] = "ANSWER"
            # Options are propositions to evaluate, not facts that must each
            # exist in memory. The plan retrieves one shared evidence bundle.
            option_coverage = []
        elif any(slot.get("option_label") for slot in slots):
            return None, "option_label_without_visible_option"

        # Keep the typed contract honest when the model confuses an observed
        # episode/trajectory with a resolved state head. CURRENT_STATE is only
        # for a unique canonical head. Longitudinal roles, visible-option
        # bundles, and inferred answer conclusions are evidence claims, so a
        # DIRECT slot with history is the executable representation. This is a
        # structural contract repair, not a query-text classifier and it does
        # not create new slots or retrieval operations.
        for slot in slots:
            role = str(slot.get("evidence_role") or "ANSWER").upper()
            should_be_evidence_claim = (
                role in {"LONGITUDINAL_CONTEXT", "TRAJECTORY", "RESPONSE"}
                or bool(question_options)
                or (
                    bool(query_spec.get("requires_inference"))
                    and role == "ANSWER"
                )
            )
            if slot.get("type") == "CURRENT_STATE" and should_be_evidence_claim:
                slot["type"] = "DIRECT"
                slot["required_fields"] = ["value"]
                slot["history"] = True

        if question_options and query_spec.get("reasoning") == "NONE":
            query_spec["reasoning"] = "SYNTHESIS"
            query_spec["requires_inference"] = True

        # An inferred/synthesis answer cannot be certified by one generic
        # direct slot: that lets a topical seed short-circuit the explicit
        # retrieval operation and hides the context needed for reasoning.
        # Reject this inconsistent contract so the bounded fallback path can
        # retrieve a small evidence bundle instead of silently taking the
        # direct route.
        if (
            query_spec.get("requires_inference")
            and query_spec.get("reasoning") == "SYNTHESIS"
            and not question_options
            and len(slots) == 1
            and slots[0]["type"] not in {"TRANSITION", "COMPARISON", "CAUSE_PATH"}
        ):
            return None, "inference_requires_decomposed_slots"

        # Semantic decomposition belongs to the planner. The executor validates
        # typed fields and operation dependencies, not a second hand-written
        # query-mode classifier.

        seed_coverage = []
        for raw in parsed.get("seed_coverage") or []:
            slot_id = str(raw.get("slot_id") or "")
            refs = raw.get("refs") if isinstance(raw.get("refs"), list) else []
            if (
                slot_id not in slot_ids
                or not refs
                or not all(self._valid_ref(ref, seed_count, 0) for ref in refs)
            ):
                return None, "invalid_seed_coverage"
            seed_coverage.append(
                {"slot_id": slot_id, "refs": [str(ref) for ref in refs]}
            )

        raw_operations = parsed.get("operations")
        if not isinstance(raw_operations, list) or len(raw_operations) > 4:
            return None, "invalid_operations"
        requested_budget_tier = str(parsed.get("budget_tier") or "").upper()
        budget_tier = self._derive_budget_tier(
            query_mode, slots, raw_operations, query_spec
        )
        budget = RETRIEVAL_BUDGETS[budget_tier]
        operations = []
        for index, raw in enumerate(raw_operations):
            op = str(raw.get("op") or "").upper()
            produces = (
                raw.get("produces") if isinstance(raw.get("produces"), list) else []
            )
            if (
                op not in VALID_OPERATIONS
                or not produces
                or any(slot not in slot_ids for slot in produces)
            ):
                return None, "invalid_operation_contract"
            operation = {**raw, "op": op, "produces": list(dict.fromkeys(produces))}
            refs = []
            for field in ("start", "memory_refs", "candidate_refs"):
                value = operation.get(field)
                refs.extend(
                    value if isinstance(value, list) else ([value] if value else [])
                )
            for field in ("anchor", "end"):
                value = operation.get(field)
                if value and not self._parse_date(str(value)):
                    refs.append(value)
            if not all(self._valid_ref(ref, seed_count, index) for ref in refs):
                return None, "invalid_reference"
            if op == "TEMPORAL_FILTER":
                axis = str(operation.get("axis") or "").lower()
                fallback_axis = str(operation.get("fallback_axis") or "").lower()
                relation = str(operation.get("relation") or "").upper()
                if axis not in VALID_TEMPORAL_AXES:
                    return None, "unsupported_axis"
                if fallback_axis and fallback_axis not in VALID_TEMPORAL_AXES:
                    return None, "unsupported_fallback_axis"
                if relation not in {
                    "BEFORE",
                    "AFTER",
                    "BETWEEN",
                    "EXACT",
                    "EARLIEST",
                    "LATEST",
                }:
                    return None, "invalid_temporal_relation"
                if relation in {"BEFORE", "AFTER"} and not operation.get("anchor"):
                    return None, "missing_temporal_anchor"
                if relation == "BETWEEN" and (
                    not operation.get("anchor") or not operation.get("end")
                ):
                    return None, "missing_temporal_bounds"
                if relation in {"EARLIEST", "LATEST"}:
                    candidate_refs = operation.get("candidate_refs") or []
                    if candidate_refs and all(
                        str(ref).startswith("$seed") for ref in candidate_refs
                    ):
                        return None, "temporal_extremum_requires_search_scope"
                operation["axis"], operation["fallback_axis"] = axis, fallback_axis
                operation["relation"] = relation
            if op == "FOLLOW_CAUSES":
                if not operation.get("start") or str(
                    operation.get("direction") or ""
                ).upper() not in {"IN", "OUT"}:
                    return None, "invalid_causal_dependency"
                try:
                    depth = int(operation.get("depth"))
                except (TypeError, ValueError):
                    return None, "invalid_causal_depth"
                if depth < 1 or depth > 3:
                    return None, "invalid_causal_depth"
                operation["direction"], operation["depth"] = (
                    str(operation["direction"]).upper(),
                    depth,
                )
            if (
                op in {"SEMANTIC_SEARCH", "LOCATE_ANCHOR", "RESOLVE_STATE"}
                and not str(operation.get("query") or "").strip()
            ):
                return None, "missing_operation_query"
            if op == "VERIFY_EVIDENCE" and not operation.get("memory_refs"):
                return None, "missing_evidence_refs"
            self._enrich_operation_query(operation, query_spec)
            # The planner's slot descriptions are the semantic contract for a
            # search. Carry them into the lexical/embedding query so an
            # operation that produces a longitudinal or decision role cannot
            # silently search only the shorter free-form query written by the
            # model. This is not an intent classifier: it uses only fields the
            # planner has already declared for that operation.
            if op in {"SEMANTIC_SEARCH", "LOCATE_ANCHOR", "RESOLVE_STATE", "TEMPORAL_FILTER"}:
                produced_descriptions = " ".join(
                    slot["description"]
                    for slot in slots
                    if slot["id"] in operation["produces"]
                ).strip()
                if produced_descriptions:
                    operation["query"] = (
                        f"{str(operation.get('query') or '').strip()} | "
                        f"required evidence: {produced_descriptions}"
                    ).strip(" |")[:800]
            if op == "SEMANTIC_SEARCH":
                produced_roles = {
                    slot["evidence_role"]
                    for slot in slots
                    if slot["id"] in operation["produces"]
                }
                requested_strategy = str(
                    operation.get("strategy") or ""
                ).upper()
                if question_options:
                    # Visible options are one shared evidence problem. Keep
                    # the planner's slots and budget, but force the operation
                    # strategy to include the propositions themselves even if
                    # the model emitted a generic decision-bundle strategy.
                    requested_strategy = "SHARED_OPTIONS"
                elif requested_strategy not in VALID_SEARCH_STRATEGIES:
                    if question_options:
                        requested_strategy = "SHARED_OPTIONS"
                    elif "LONGITUDINAL_CONTEXT" in produced_roles or "CAUSE" in produced_roles:
                        requested_strategy = "TRAJECTORY"
                    elif produced_roles.intersection(
                        {"FOCAL_STATE", "ACTION_RULE", "CONSTRAINT", "ALTERNATIVE"}
                    ):
                        requested_strategy = "DECISION_BUNDLE"
                    else:
                        requested_strategy = "FOCAL"
                operation["strategy"] = requested_strategy
                operation["evidence_roles"] = sorted(produced_roles)
                if requested_strategy == "SHARED_OPTIONS":
                    # Keep the propositions as structured inputs to this one
                    # shared retrieval operation. The executor may issue
                    # several cheap local searches and merge their outputs,
                    # but the planner still exposes one bounded operation and
                    # one provenance boundary rather than one search per option.
                    operation["option_queries"] = list(question_options.values())
                    shared_terms = " ".join(
                        part
                        for part in (
                            self._question_stem(question),
                            str(query_spec.get("target_property") or ""),
                            " ".join(query_spec.get("target_entities") or []),
                            # Shared-option retrieval must see the propositions
                            # themselves. Labels are excluded, but removing the
                            # option text entirely loses entities that occur only
                            # in a choice and makes the shared bundle incomplete.
                            " ".join(question_options.values()),
                            " ".join(
                                slot["description"]
                                for slot in slots
                                if slot["id"] in operation["produces"]
                            ),
                        )
                        if part
                    )
                    operation["query"] = shared_terms[:800]
            operations.append(operation)

        # A planner may validly declare operations=[] when its seed coverage is
        # complete. No fallback search is synthesized for a valid zero-op plan.

        seed_covered_slots = {coverage["slot_id"] for coverage in seed_coverage}
        non_verify_producers = {
            slot_id
            for operation in operations
            if operation["op"] != "VERIFY_EVIDENCE"
            for slot_id in operation["produces"]
        }
        verify_only_slots = (
            {
                slot_id
                for operation in operations
                if operation["op"] == "VERIFY_EVIDENCE"
                for slot_id in operation["produces"]
            }
            - seed_covered_slots
            - non_verify_producers
        )
        if verify_only_slots:
            return None, "verify_evidence_is_not_an_answer_slot"
        uncovered_slots = slot_ids - seed_covered_slots - non_verify_producers
        if uncovered_slots:
            return None, "uncovered_required_slot"

        seed_covered_slots = {item["slot_id"] for item in seed_coverage}
        for slot in slots:
            if slot["type"] != "TEMPORAL":
                continue
            if (
                slot["id"] in seed_covered_slots
                and slot.get("temporal_relation") == "EXACT"
            ):
                # A zero-operation plan is valid when the planner identifies a
                # direct event-date seed. Typed execution still verifies the
                # requested axis before it can count as covered. Extremum and
                # range relations need an operation to establish comparison.
                continue
            relation = slot.get("temporal_relation")
            temporal_producers = [
                operation
                for operation in operations
                if operation["op"] == "TEMPORAL_FILTER"
                and slot["id"] in operation["produces"]
            ]
            if not temporal_producers:
                return None, "temporal_slot_requires_temporal_filter"
            if not any(
                operation.get("axis") == slot.get("time_axis")
                and operation.get("relation") == relation
                for operation in temporal_producers
            ):
                return None, "temporal_contract_mismatch"
        for slot in slots:
            if slot["type"] != "CAUSE_PATH" or slot["id"] in seed_covered_slots:
                continue
            if not any(
                operation["op"] == "FOLLOW_CAUSES"
                and slot["id"] in operation["produces"]
                for operation in operations
            ):
                return None, "cause_path_requires_follow_causes"
        return {
            "query_spec": query_spec,
            "query_mode": query_mode,
            "required_slots": slots,
            "seed_coverage": seed_coverage,
            "operations": operations,
            "need_evidence": bool(parsed.get("need_evidence", False)),
            "budget_tier": budget_tier,
            "requested_budget_tier": requested_budget_tier,
            "max_memories": budget["max_memories"],
            "planner_fallback": False,
            "fallback_reason": "",
            "valid": True,
            "option_coverage": option_coverage,
        }, ""

    def _make_deterministic_recovery_plan(
        self,
        missing_slots: List[Dict[str, Any]],
        question: str,
        existing_plan: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build a typed recovery plan without another LLM call.

        Recovery preserves the missing slot's semantics. A generic semantic
        search is appropriate for direct/option/comparison evidence, but it
        cannot replace temporal filtering, state resolution, or causal walks.

        Returns a valid plan dict, or None when nothing useful can be generated.
        """
        if not missing_slots:
            return None

        # Extract keywords from the question stem (strip option choices).
        stem = self._question_stem(question)
        # Keep the most discriminative tokens from the question as a suffix.
        stem_tokens = " ".join(self._tokenize(stem)[:12])
        visible_options = self._question_options(question)
        option_text = " ".join(
            f"{label}: {text}" for label, text in visible_options.items()
        )

        operations = []
        budget = existing_plan.get("budget_tier", "LARGE")
        if budget == "SMALL" and any(
            str(slot.get("type") or "").upper() == "TEMPORAL"
            and str(slot.get("temporal_relation") or "").upper()
            in {"EARLIEST", "LATEST"}
            for slot in missing_slots
        ):
            budget = "MEDIUM"
        max_operations = RETRIEVAL_BUDGETS.get(budget, {}).get("max_operations", 4)
        for slot in missing_slots:
            slot_id = slot.get("id", f"recovery_{len(operations)}")
            description = str(slot.get("description") or slot_id)
            query = f"{description} {stem_tokens}".strip()
            slot_type = str(slot.get("type") or "DIRECT").upper()
            if slot_type == "TEMPORAL":
                relation = str(slot.get("temporal_relation") or "EXACT").upper()
                if relation in {"EARLIEST", "LATEST"}:
                    # Anchor discovery and chronology are separate phases. The
                    # executor knows that the first operation cannot satisfy a
                    # temporal extremum by itself.
                    operations.append(
                        {
                            "op": "LOCATE_ANCHOR",
                            "query": query,
                            "produces": [slot_id],
                        }
                    )
                    if len(operations) < max_operations:
                        operations.append(
                            {
                                "op": "TEMPORAL_FILTER",
                                "relation": relation,
                                "axis": str(
                                    slot.get("time_axis") or "event_time"
                                ).lower(),
                                "fallback_axis": "",
                                "candidate_refs": [f"${len(operations) - 1}"],
                                "query": query,
                                "produces": [slot_id],
                            }
                        )
                else:
                    operations.append(
                        {
                            "op": "TEMPORAL_FILTER",
                            "relation": relation,
                            "axis": str(slot.get("time_axis") or "event_time").lower(),
                            "fallback_axis": "",
                            "query": query,
                            "produces": [slot_id],
                        }
                    )
            elif slot_type == "CURRENT_STATE":
                operations.append(
                    {"op": "RESOLVE_STATE", "query": query, "produces": [slot_id]}
                )
            elif slot_type == "CAUSE_PATH":
                anchor_index = len(operations)
                operations.append(
                    {
                        "op": "SEMANTIC_SEARCH",
                        "query": query,
                        "top_k": 3,
                        "produces": [slot_id],
                    }
                )
                if len(operations) < max_operations:
                    operations.append(
                        {
                            "op": "FOLLOW_CAUSES",
                            "start": [f"${anchor_index}"],
                            "direction": "OUT",
                            "depth": 3,
                            "goal": description,
                            "produces": [slot_id],
                        }
                    )
            else:
                # Recovery must preserve the planner's shared-option contract.
                # A normal FOCAL search can return the first remembered rule and
                # starve the answer model of the participant facts needed to
                # evaluate the remaining choices. This is still one bounded
                # operation, not one search per option.
                strategy = "SHARED_OPTIONS" if visible_options else "FOCAL"
                recovery_query = " ".join(
                    part for part in (query, option_text) if part
                )
                operation = {
                    "op": "SEMANTIC_SEARCH",
                    "query": recovery_query,
                    "top_k": 8 if visible_options else 5,
                    "strategy": strategy,
                    "produces": [slot_id],
                }
                if visible_options:
                    operation["option_queries"] = list(visible_options.values())
                operations.append(
                    operation
                )
            if len(operations) >= max_operations:
                break

        if not operations:
            return None

        max_memories = RETRIEVAL_BUDGETS.get(budget, {}).get("max_memories", 8)
        return {
            "query_mode": existing_plan.get("query_mode", "DIRECT"),
            "required_slots": missing_slots,
            "seed_coverage": [],
            "operations": operations,
            "option_coverage": existing_plan.get("option_coverage", []),
            "need_evidence": False,
            "budget_tier": budget,
            "max_memories": max_memories,
            "planner_fallback": True,
            "fallback_reason": "zero_result_recovery",
            "valid": True,
        }

    def _semantic_controller(
        self,
        question: str,
        seeds: List[Dict[str, Any]],
        frame: QueryFrame,
        context_map: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any], Dict[str, Any]]:
        """Choose a direct support or one complete plan in a single LLM call."""
        if not self.enable_planner or not seeds:
            reason = "planner_disabled" if not self.enable_planner else "no_seeds"
            fallback = self._finalize_fallback_plan(
                self._fallback_plan(question, reason), question
            )
            return (
                None,
                fallback,
                {
                    "called": False,
                    "route": "PLAN",
                    "support_ref": "",
                    "fallback_reason": reason,
                    "usage": {},
                },
            )

        compact = [
            self._compact_planner_seed(memory, index)
            for index, memory in enumerate(seeds[:3])
        ]
        domain_guidance = (
            MEDICAL_PLANNER_GUIDANCE
            if any(
                str(memory.get("scope") or "").lower() in CLINICAL_SCOPES
                for memory in compact
            )
            else ""
        )
        prompt = COMPACT_CONTROLLER_PROMPT.format(
            domain_guidance=domain_guidance,
            seeds=json.dumps(compact, ensure_ascii=False),
            context_map=json.dumps(context_map or [], ensure_ascii=False),
            question=question,
        )
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                # Both controller routes return a small JSON contract. Keeping
                # this cap tight prevents an easy query from reserving the
                # answer model's full completion budget.
                max_tokens=768,
            )
            usage = self._response_usage(response, prompt)
            parsed = self._parse_json(response.content)
        except Exception:
            fallback = self._finalize_fallback_plan(
                self._fallback_plan(question, "controller_error"), question
            )
            return (
                None,
                fallback,
                {
                    "called": True,
                    "route": "PLAN",
                    "support_ref": "",
                    "fallback_reason": "controller_error",
                    "usage": {},
                },
            )

        route = str(parsed.get("route") or "PLAN").upper()
        if route == "PLAN":
            plan, reason = self._validate_plan(parsed, question, len(compact))
            if plan is not None:
                return (
                    None,
                    plan,
                    {
                        "called": True,
                        "route": "PLAN",
                        "support_ref": "",
                        "fallback_reason": "",
                        "usage": usage,
                    },
                )
        elif route in {"ANSWER", "DIRECT"}:
            support_ref = str(parsed.get("support_ref") or "")
            supports = self._validate_fast_support(support_ref, seeds, frame)
            if supports is not None and set(parsed).issubset(
                {"route", "support_ref"}
            ):
                return (
                    supports,
                    {},
                    {
                        "called": True,
                        "route": "DIRECT",
                        "support_ref": support_ref,
                        "fallback_reason": "",
                        "usage": usage,
                    },
                )
            reason = "invalid_direct_route"
        else:
            reason = "invalid_controller_route"

        # Preserve valid typed requirements, but never spend a second LLM call
        # repairing controller JSON.
        slots, _ = (
            self._parse_required_slots(parsed)
            if isinstance(parsed, dict)
            else (None, "")
        )
        fallback = (
            self._fallback_missing_plan(question, reason, slots)
            if slots
            else self._fallback_plan(question, reason)
        )
        fallback = self._preserve_fallback_semantics(fallback, parsed, question)
        return (
            None,
            fallback,
            {
                "called": True,
                "route": "PLAN",
                "support_ref": "",
                "fallback_reason": reason,
                "usage": usage,
            },
        )

    def _plan_operations(
        self,
        question: str,
        seeds: List[Dict[str, Any]],
        missing_slots: Optional[List[Dict[str, Any]]] = None,
        prior_trace: Optional[List[Dict[str, Any]]] = None,
        forbid_single_direct_zero_op: bool = False,
        context_map: Optional[List[Dict[str, Any]]] = None,
        allow_repair: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.enable_planner:
            return (
                self._finalize_fallback_plan(
                    self._fallback_plan(question, "planner_disabled"), question
                ),
                {},
            )
        compact = [
            self._compact_planner_seed(memory, index)
            for index, memory in enumerate(seeds[:3])
        ]
        domain_guidance = (
            MEDICAL_PLANNER_GUIDANCE
            if any(
                str(memory.get("scope") or "").lower() in CLINICAL_SCOPES
                for memory in compact
            )
            else ""
        )

        def request_plan(
            validation_error: str = "",
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            # The baseline planner sees only the authorized seeds unless a
            # caller explicitly supplies routing metadata.  Do not silently
            # turn the offline profile pack into a second query-time recall
            # channel.
            planner_context_map = context_map if context_map is not None else []
            prompt = TYPED_PLANNER_PROMPT.format(
                domain_guidance=domain_guidance,
                seeds=json.dumps(compact, ensure_ascii=False),
                context_map=json.dumps(planner_context_map, ensure_ascii=False),
                question=question,
                missing_slots=json.dumps(missing_slots or [], ensure_ascii=False),
                prior_trace=json.dumps(prior_trace or [], ensure_ascii=False),
                validation_error=validation_error,
            )
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                # A retrieval plan is bounded by four operations and a small
                # typed slot list. The answer call retains the full budget.
                max_tokens=768,
            )
            return self._parse_json(response.content), self._response_usage(
                response, prompt
            )

        def merge_usage(
            aggregate: Dict[str, Any], current: Dict[str, Any]
        ) -> Dict[str, Any]:
            merged = dict(aggregate)
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                merged[field] = int(merged.get(field, 0)) + int(current.get(field, 0))
            merged["latency"] = float(merged.get("latency", 0.0)) + float(
                current.get("latency", 0.0)
            )
            return merged

        usage: Dict[str, Any] = {}

        def normalize_partial_metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
            """Keep semantic plan metadata unchanged; validators own contracts."""
            return candidate

        # Kept as a compatibility argument for older ablation callers. A valid
        # zero-operation plan is never invalidated because an earlier gate was
        # conservative; planner seed coverage is an independent semantic claim.
        del forbid_single_direct_zero_op

        try:
            parsed, first_usage = request_plan()
            parsed = normalize_partial_metadata(parsed)
            usage = merge_usage(usage, first_usage)
        except Exception:
            fallback = (
                self._fallback_missing_plan(question, "replan_error", missing_slots)
                if missing_slots
                else self._fallback_plan(question, "planner_error")
            )
            return self._finalize_fallback_plan(fallback, question), {}
        plan, reason = self._validate_plan(
            parsed, question, len(compact), missing_slots=missing_slots
        )
        if (
            plan
            and missing_slots
            and not self._preserves_missing_slot_contract(plan, missing_slots)
        ):
            plan, reason = None, "replan_contract_mismatch"
        if plan:
            usage["repair_called"] = False
            return plan, usage

        if allow_repair is None:
            allow_repair = bool(getattr(self, "enable_planner_repair", False))
        if not allow_repair:
            if missing_slots:
                fallback = self._fallback_missing_plan(question, reason, missing_slots)
            else:
                salvaged_slots, _ = self._parse_required_slots(parsed)
                fallback = (
                    self._fallback_missing_plan(question, reason, salvaged_slots)
                    if salvaged_slots
                    else self._fallback_plan(question, reason)
                )
            usage["repair_called"] = False
            return self._preserve_fallback_semantics(
                fallback, parsed, question
            ), usage

        # Invalid syntax/dependencies are repairable planning errors. Give the
        # semantic authority one bounded correction before deterministic fallback.
        # Valid zero-operation plans never enter this branch.
        first_reason = reason
        repaired: Optional[Dict[str, Any]] = None
        try:
            repaired, repair_usage = request_plan(reason)
            repaired = normalize_partial_metadata(repaired)
            usage = merge_usage(usage, repair_usage)
            usage["repair_called"] = True
            plan, reason = self._validate_plan(
                repaired, question, len(compact), missing_slots=missing_slots
            )
            if (
                plan
                and missing_slots
                and not self._preserves_missing_slot_contract(plan, missing_slots)
            ):
                plan, reason = None, "replan_contract_mismatch"
            if plan:
                return plan, usage
        except Exception:
            reason = first_reason
            usage["repair_called"] = True

        if missing_slots:
            fallback = self._fallback_missing_plan(question, reason, missing_slots)
        else:
            salvaged_slots = None
            for candidate in (repaired, parsed):
                if not isinstance(candidate, dict):
                    continue
                candidate_slots, _ = self._parse_required_slots(candidate)
                if candidate_slots:
                    salvaged_slots = candidate_slots
                    break
            fallback = (
                self._fallback_missing_plan(question, reason, salvaged_slots)
                if salvaged_slots
                else self._fallback_plan(question, reason)
            )
        semantic_source = repaired if isinstance(repaired, dict) else parsed
        return self._preserve_fallback_semantics(
            fallback, semantic_source, question
        ), usage

    def _resolve_refs(
        self,
        refs: Any,
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resolve only authorized references; arbitrary memory IDs are forbidden."""
        values = refs if isinstance(refs, list) else [refs]
        resolved: List[Dict[str, Any]] = []
        for value in values:
            text = str(value or "")
            seed_match = re.fullmatch(r"\$seed(\d+)", text)
            output_match = re.fullmatch(r"\$(\d+)", text)
            if seed_match:
                index = int(seed_match.group(1))
                if index < len(seeds):
                    resolved.append(self._snapshot(seeds[index]))
            elif output_match:
                index = int(output_match.group(1))
                if index < len(outputs):
                    resolved.extend(self._snapshot(memory) for memory in outputs[index])
            # Bare memory IDs intentionally resolve to nothing.
        return list({memory["id"]: memory for memory in resolved}.values())
