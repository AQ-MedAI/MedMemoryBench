"""Context packing, query telemetry, and public query lifecycle."""

import time
from typing import Any, Dict, Iterable, List, Optional

from methods.base import AgentResponse
from utils.llm_client import format_messages

from .contracts import CLINICAL_SCOPES

QUERY_TOKEN_STAGES = ("fast_gate", "planner", "slot_validation", "replan", "answer")


class QueryMixin:
    """Turns retrieval results into a minimal answer request and telemetry."""

    @staticmethod
    def _total_query_tokens(tokens: Dict[str, Any]) -> int:
        return sum(int(tokens.get(stage, 0) or 0) for stage in QUERY_TOKEN_STAGES)

    @staticmethod
    def _multiple_choice_answer_instruction(option_labels: Iterable[str]) -> str:
        labels = ", ".join(sorted(str(label) for label in option_labels))
        return (
            f" This is a multiple-choice question with options {labels}. "
            "First identify the predicate requested by the question stem, such as supported, recommended, "
            "unsafe, contradicted, or contraindicated. Then evaluate EVERY option independently against "
            "that same predicate and the participant-specific facts in memory. A contraindication or allergy "
            "disqualifies a proposed action only when the stem asks for a safe or recommended action; it supports "
            "an option when the stem asks which action is contraindicated or unsafe. Do not invert the stem's "
            "polarity. After evaluation, output ONLY the uppercase labels of all options satisfying the stem, "
            "separated by commas, with absolutely no prose, explanation, or other text. "
            "If no option satisfies it, output NONE."
        )

    def _format_answer_memory(self, memory: Dict[str, Any]) -> str:
        """Render evidence without exposing internal ledger identifiers."""
        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        context_marker = (
            " supplementary_context=true"
            if memory.get("_supplementary_context")
            else ""
        )
        return (
            f"- kind={memory.get('kind', 'FACT')} status={status}{context_marker} "
            f"session={memory.get('session_idx', 'UNKNOWN')} "
            f"stance={memory.get('stance', 'AFFIRM')} "
            f"event_time={memory.get('event_time') or 'UNKNOWN'} "
            f"document_time={memory.get('document_time') or 'UNKNOWN'} "
            f"origin_document_time={memory.get('origin_document_time') or 'UNKNOWN'} "
            f"effective_event_time={self._date_for(memory, 'effective_event_time') or 'UNKNOWN'} "
            f"value={self._memory_value(memory) or 'NONE'}: "
            f"{memory.get('claim', '')}"
            + (
                f" [verbatim_value={memory['verbatim_value']}]"
                if memory.get("verbatim_value")
                else ""
            )
        )

    def _format_episode_context(self, memories: List[Dict[str, Any]]) -> str:
        """Group authorized atomic memories into a compact session trajectory view."""
        if not memories:
            return "- No matching memory."

        episodes: Dict[str, List[Dict[str, Any]]] = {}
        for memory in memories:
            session = str(memory.get("session_idx", "UNKNOWN"))
            episodes.setdefault(session, []).append(memory)

        blocks: List[str] = []
        for session, episode_memories in episodes.items():
            dates = list(
                dict.fromkeys(
                    str(memory.get("document_time"))
                    for memory in episode_memories
                    if memory.get("document_time")
                )
            )
            date_label = ", ".join(dates) if dates else "UNKNOWN"
            lines = [
                self._format_answer_memory(memory) for memory in episode_memories
            ]
            blocks.append(
                f"[EPISODE session={session} document_date={date_label}]\n"
                + "\n".join(lines)
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _context_time_axis(slots: List[Dict[str, Any]]) -> Optional[str]:
        """Return one requested temporal axis, or None for a mixed-axis plan."""
        axes = list(
            dict.fromkeys(
                str(slot.get("time_axis") or "")
                for slot in slots
                if slot.get("type") == "TEMPORAL" and slot.get("time_axis")
            )
        )
        if len(axes) > 1:
            return None
        return axes[0] if axes else "event_time"

    @staticmethod
    def _role_aware_support_ids(
        slots: List[Dict[str, Any]],
        slot_support: Dict[str, List[str]],
        candidate_order: List[str],
        limit: int,
    ) -> List[str]:
        """Reserve context for every covered role before adding redundancy."""
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []
        unique_slots = []
        seen_slots = set()
        for slot in slots:
            slot_id = str(slot.get("id") or "")
            if slot_id and slot_id not in seen_slots:
                unique_slots.append(slot)
                seen_slots.add(slot_id)

        allowed = set(candidate_order)
        selected: List[str] = []

        def add(memory_id: str) -> None:
            if (
                memory_id in allowed
                and memory_id not in selected
                and len(selected) < bounded_limit
            ):
                selected.append(memory_id)

        # First preserve one answer-bearing memory for every independent role.
        for slot in unique_slots:
            for memory_id in slot_support.get(slot["id"], []):
                if memory_id in allowed:
                    add(memory_id)
                    break

        # Rich reasoning roles need a second independent observation when the
        # executor exposed one. A single state head can satisfy a boolean
        # contract yet be insufficient for an inference or option decision;
        # retain the bounded second support without admitting candidates that
        # were not already authorized by the plan.
        rich_roles = {
            "FOCAL_STATE",
            "LONGITUDINAL_CONTEXT",
            "ACTION_RULE",
            "CONSTRAINT",
            "ALTERNATIVE",
            "CAUSE",
            "COMPARAND",
        }
        for slot in unique_slots:
            if slot.get("evidence_role") not in rich_roles:
                continue
            count = 0
            for memory_id in slot_support.get(slot["id"], []):
                if memory_id in allowed:
                    add(memory_id)
                    count += 1
                if count >= 2 or len(selected) >= bounded_limit:
                    break

        # Relational slots need at least two endpoints when available.
        relational = {"TRANSITION", "CAUSE_PATH", "COMPARISON"}
        for slot in unique_slots:
            if slot.get("type") not in relational:
                continue
            count = 0
            for memory_id in slot_support.get(slot["id"], []):
                if memory_id in allowed:
                    add(memory_id)
                    count += 1
                if count >= 2 or len(selected) >= bounded_limit:
                    break

        supported = {
            memory_id
            for memory_ids in slot_support.values()
            for memory_id in memory_ids
        }
        for memory_id in candidate_order:
            if memory_id in supported:
                add(memory_id)
        return selected

    def _dereference_evidence(
        self, memory_ids: Iterable[str], limit: int = 3
    ) -> List[Dict[str, Any]]:
        requested, pointers = set(memory_ids), []
        for memory in self._memories:
            if memory["id"] in requested:
                pointers.extend(memory["evidence_ids"])
        by_id, output, seen = {e["id"]: e for e in self._evidence}, [], set()
        for evidence_id in pointers:
            if evidence_id in by_id and evidence_id not in seen:
                output.append(self._snapshot(by_id[evidence_id]))
                seen.add(evidence_id)
                if len(output) >= limit:
                    break
        return output

    def _format_memory(self, memory: Dict[str, Any]) -> str:
        status = memory.get("_status", self._belief_status.get(memory["id"], "active"))
        return (
            f"- id={memory['id']} kind={memory['kind']} status={status} stance={memory['stance']} "
            f"event_time={memory['event_time']} document_time={memory['document_time']} "
            f"origin_document_time={memory.get('origin_document_time') or 'UNKNOWN'} "
            f"effective_event_time={self._date_for(memory, 'effective_event_time') or 'UNKNOWN'} "
            f"assertion_mode={memory.get('assertion_mode', 'DIRECT')} "
            f"source={','.join(memory.get('source_speakers', [])) or 'UNKNOWN'} "
            f"state_identity={self._state_identity(memory) or 'NONE'} "
            f"value={self._memory_value(memory) or 'NONE'}: {memory['claim']}"
            + (
                f" [verbatim_value={memory['verbatim_value']}]"
                if memory.get("verbatim_value")
                else ""
            )
        )

    def prepare_batch_query(
        self, question: str, system_message: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        started = time.time()
        retrieval_question = self._unwrap_question(question)
        question_options = self._question_options(retrieval_question)
        frame = self._query_frame(retrieval_question)

        # Phase-1 read contract: RRF Top-8 and Top-3 seeds.
        seeds = self._constraint_first_search(retrieval_question, frame, top_k=8)[:8]
        initial_seeds = self._select_initial_seeds(seeds)
        planning_seeds = self._planning_seed_set(retrieval_question, seeds)
        # The routing map is needed by the unified controller, but a validated
        # legacy fast-path query never sends it to an LLM and does not need the
        # extra local hybrid search. Planned legacy queries build it lazily in
        # _plan_operations, so direct queries pay only the initial recall.
        planning_context = (
            self._planning_context_map(retrieval_question)
            if getattr(self, "enable_unified_controller", False)
            else None
        )

        if getattr(self, "enable_unified_controller", False):
            fast_supports, gate = None, {
                "called": False,
                "skip_reason": "unified_controller",
                "usage": {},
            }
        else:
            gate_skip_reason = self._gate_skip_surface(retrieval_question)
            if gate_skip_reason:
                # Surface routing only says that one-seed direct composition is
                # not applicable. Planner still owns semantic decomposition.
                fast_supports, gate = None, {
                    "called": False,
                    "skip_reason": gate_skip_reason,
                    "usage": {},
                }
            else:
                fast_supports, gate = self._answerability_gate(
                    retrieval_question, initial_seeds, frame
                )

        run = self._run_query_retrieval(
            retrieval_question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        # The unified controller owns the direct/planned route decision. Keep
        # its validated single support on the fast path all the way to context
        # packing instead of treating it as an ordinary planned result.
        fast_supports = run.get("fast_supports")
        query_tokens = run["query_tokens"]
        plan = run["plan"]
        replan = run["replan"]
        planner_called = run["planner_called"]
        replan_called = run["replan_called"]
        trace = run["trace"]
        slot_validation_events = run["slot_validation"]
        operation_output_ids = run["operation_output_ids"]
        operation_candidates = run.get("operation_candidates", [])
        authorized_seeds = run.get("planning_seeds", initial_seeds)
        evidence_refs = run["evidence_refs"]
        slot_support = run["slot_support"]
        slot_coverage = run["slot_coverage"]
        all_relations = run["relations"]
        sufficient = run["sufficient"]
        beliefs = run["beliefs"]
        query_latency = dict(run.get("query_latency", {}))
        controller = run.get("controller", {})
        precomputed_answer = str(run.get("precomputed_answer") or "").strip()

        context_slots = [
            slot
            for candidate_plan in (plan, replan or {})
            for slot in candidate_plan.get("required_slots", [])
        ]

        # Final support set contains only gate-authorized seeds or typed slot supports.
        if fast_supports is not None:
            support_ids = {memory["id"] for memory in fast_supports}
        else:
            support_ids = {
                memory_id
                for memory_ids in slot_support.values()
                for memory_id in memory_ids
            }
        context_memory_limit = (
            min(3, len(fast_supports))
            if fast_supports is not None
            else max(
                int(plan.get("max_memories", 3)),
                int((replan or {}).get("max_memories", 0)),
            )
        )
        role_floor = sum(
            2 if slot.get("type") in {"TRANSITION", "CAUSE_PATH", "COMPARISON"} else 1
            for slot in {
                slot.get("id"): slot
                for slot in context_slots
                if slot_support.get(str(slot.get("id") or ""))
            }.values()
        )
        context_memory_limit = min(
            self.HARD_MEMORY_LIMIT,
            max(context_memory_limit, role_floor),
        )
        # IDs added below are authorized operation outputs, but were not
        # selected as the validator's answer-bearing supports.
        unverified_context_ids = set()

        # Structural/semantic support proves slot sufficiency; it is not a
        # license to discard every other candidate returned by the same
        # authorized operation. Inference needs a small amount of independent
        # longitudinal or decision evidence even after each typed slot has one
        # accepted support. Add only candidates from operations already in the
        # trace, match them against the planner-declared role, and keep them
        # explicitly marked as supplementary context.
        supplementary_ids = set()
        route_spec = (plan or {}).get("query_spec") or {}
        hard_reasoning = bool(route_spec.get("requires_inference")) or str(
            (plan or {}).get("query_mode") or ""
        ).upper() in {"DECISION", "CAUSAL", "COMPARISON", "MULTI_OPTION"}
        if fast_supports is None and hard_reasoning and operation_candidates:
            output_ids_by_operation = {
                int(item.get("operation_index")): set(item.get("output_ids") or [])
                for item in trace
                if str(item.get("operation_index", "")).lstrip("-").isdigit()
            }
            rich_roles = {
                "FOCAL_STATE",
                "LONGITUDINAL_CONTEXT",
                "ACTION_RULE",
                "CONSTRAINT",
                "ALTERNATIVE",
                "CAUSE",
                "COMPARAND",
            }
            if str((plan or {}).get("query_mode") or "").upper() == "MULTI_OPTION":
                rich_roles.add("ANSWER")
            candidate_by_id_for_supplement = {
                memory["id"]: memory for memory in operation_candidates
            }
            for slot in context_slots:
                role = str(slot.get("evidence_role") or "ANSWER").upper()
                if role not in rich_roles:
                    continue
                operation_ids = {
                    operation_index
                    for operation_index, operation in enumerate(
                        (plan or {}).get("operations") or []
                    )
                    if slot.get("id") in (operation.get("produces") or [])
                }
                operation_ids.update(
                    operation_index
                    for operation_index, operation in enumerate(
                        (replan or {}).get("operations") or []
                    )
                    if slot.get("id") in (operation.get("produces") or [])
                )
                added_for_role = 0
                for operation_index in sorted(operation_ids):
                    for memory_id in output_ids_by_operation.get(operation_index, set()):
                        memory = candidate_by_id_for_supplement.get(memory_id)
                        if not memory or memory_id in support_ids:
                            continue
                        if not self._memory_matches_slot_role(slot, memory):
                            continue
                        status = memory.get(
                            "_status", self._belief_status.get(memory_id, "active")
                        )
                        if status == "superseded" and not slot.get("history"):
                            continue
                        supplementary_ids.add(memory_id)
                        added_for_role += 1
                        if added_for_role >= (3 if role == "LONGITUDINAL_CONTEXT" else 2):
                            break
                    if added_for_role >= (3 if role == "LONGITUDINAL_CONTEXT" else 2):
                        break
                    
        if supplementary_ids:
            remaining = max(0, context_memory_limit - len(support_ids))
            for memory_id in list(supplementary_ids):
                if remaining <= 0:
                    supplementary_ids.discard(memory_id)
                    continue
                support_ids.add(memory_id)
                remaining -= 1
            unverified_context_ids.update(supplementary_ids)

        # If planner failed or exhausted without coverage, preserve only authorized seed/output IDs.
        allow_historical_context = any(
            bool(slot.get("history")) for slot in context_slots
        )

        def eligible_best_effort(memory: Dict[str, Any]) -> bool:
            status = memory.get(
                "_status", self._belief_status.get(memory["id"], "active")
            )
            return allow_historical_context or status != "superseded"

        if not support_ids:
            authorized = {
                memory["id"] for memory in authorized_seeds
            } | operation_output_ids
            support_ids = {
                memory["id"]
                for memory in beliefs
                if memory["id"] in authorized and eligible_best_effort(memory)
            }
            if not support_ids:
                # Preserve the best outputs from the explicit retrieval program
                # before falling back to broad recall seeds. Previously these
                # candidates disappeared when semantic validation was incomplete.
                ordered_best_effort = []
                for memory in (*operation_candidates, *authorized_seeds):
                    if (
                        eligible_best_effort(memory)
                        and memory["id"] not in ordered_best_effort
                    ):
                        ordered_best_effort.append(memory["id"])
                support_ids = set(ordered_best_effort[:context_memory_limit])
            unverified_context_ids = set(support_ids)
        elif not sufficient and len(support_ids) < context_memory_limit:
            # Keep validated partial progress, then add bounded best-effort
            # candidates for roles that remained uncovered after the one replan.
            for memory in (*operation_candidates, *authorized_seeds):
                if len(support_ids) >= context_memory_limit:
                    break
                if (
                    memory["id"] not in support_ids
                    and memory["id"] in operation_output_ids
                    and eligible_best_effort(memory)
                ):
                    support_ids.add(memory["id"])
                    unverified_context_ids.add(memory["id"])

        candidate_by_id = {
            memory["id"]: memory
            for memory in (
                *beliefs,
                *operation_candidates,
                *authorized_seeds,
                *initial_seeds,
                *seeds,
            )
            if memory["id"] in support_ids
        }
        candidate_order = list(
            dict.fromkeys(
                memory["id"]
                for memory in (
                    *beliefs,
                    *operation_candidates,
                    *authorized_seeds,
                    *initial_seeds,
                    *seeds,
                )
                if memory["id"] in candidate_by_id
            )
        )
        if fast_supports is not None:
            ordered_ids = candidate_order[:context_memory_limit]
        else:
            ordered_ids = self._role_aware_support_ids(
                context_slots,
                slot_support,
                candidate_order,
                context_memory_limit,
            )
            # Supplementary candidates are authorized by operation output but
            # intentionally absent from slot_support. Keep them only after the
            # role-bearing supports have been reserved.
            for memory_id in candidate_order:
                if len(ordered_ids) >= context_memory_limit:
                    break
                if memory_id in supplementary_ids and memory_id not in ordered_ids:
                    ordered_ids.append(memory_id)
            if not ordered_ids and unverified_context_ids:
                # No typed support survived arbitration. Preserve the bounded
                # best-effort candidates selected by the explicit retrieval
                # program, matching the pre-existing incomplete-plan contract.
                ordered_ids = [
                    memory_id
                    for memory_id in candidate_order
                    if memory_id in support_ids
                ][:context_memory_limit]
        ordered_supports = [
            self._snapshot(candidate_by_id[memory_id]) for memory_id in ordered_ids
        ]
        for memory in ordered_supports:
            if memory["id"] in supplementary_ids:
                memory["_supplementary_context"] = True
        for memory in ordered_supports:
            if memory["id"] not in unverified_context_ids:
                continue
            memory["_best_effort_priority"] = (
                2 if memory["id"] in operation_output_ids else 1
            )
        arbitration_input_ids = set(ordered_ids)
        beliefs, fallback_relations = self._reconstruct_beliefs(
            ordered_supports,
            context_memory_limit,
        )
        all_relations = self._merge_relations(all_relations, fallback_relations)
        final_ids = {memory["id"] for memory in beliefs}

        # Pure arbitration relation view: only relations with both endpoints in final IDs.
        all_relations = [
            relation
            for relation in all_relations
            if relation.get("source_id") in final_ids
            and relation.get("target_id") in final_ids
        ]
        unresolved = not beliefs or self._has_unresolved_conflict(beliefs)

        needed_relation_types = set()
        slot_types = {slot.get("type") for slot in context_slots}
        if "TRANSITION" in slot_types:
            needed_relation_types.update({"SUPERSEDE", "REFINE"})
        if "CAUSE_PATH" in slot_types:
            needed_relation_types.add("CAUSES")
        active_query_modes = {
            str(candidate_plan.get("query_mode") or "").upper()
            for candidate_plan in (plan, replan or {})
        }
        if "CAUSAL" in active_query_modes:
            needed_relation_types.add("CAUSES")
        if "COMPARISON" in slot_types:
            needed_relation_types.update({"SUPERSEDE", "REFINE", "SUPPORT", "CONFLICT"})
        if unresolved:
            needed_relation_types.add("CONFLICT")
        all_relations = [
            r for r in all_relations if r.get("type") in needed_relation_types
        ]

        need_evidence = (
            bool(plan.get("need_evidence"))
            or bool((replan or {}).get("need_evidence"))
            or bool(evidence_refs)
        )
        linked_evidence_refs = [
            memory_id for memory_id in evidence_refs if memory_id in final_ids
        ]
        if need_evidence and not linked_evidence_refs:
            linked_evidence_refs = [
                memory["id"] for memory in beliefs[: self.MAX_EVIDENCE]
            ]
        evidence = (
            self._dereference_evidence(linked_evidence_refs, self.MAX_EVIDENCE)
            if need_evidence
            else []
        )

        # Boundary: all final memory IDs must be either seeds or explicit operation outputs.
        seed_ids = {
            memory["id"]
            for memory in (
                fast_supports if fast_supports is not None else authorized_seeds
            )
        }
        allowed_ids = seed_ids | operation_output_ids
        boundary_violation = not final_ids.issubset(allowed_ids)

        linked_evidence_ids = {
            evidence_id
            for memory in beliefs
            for evidence_id in memory.get("evidence_ids", [])
        }
        evidence_boundary_violation = any(
            item.get("id") not in linked_evidence_ids for item in evidence
        )

        # Check arbitration purity explicitly using the pre-arbitration selected IDs.
        # Pure-arbitration violation is equivalent to a final-ID escaping the
        # already authorized seed/operation set because arbitration never retrieves.
        arbitration_expansion_violation = not final_ids.issubset(
            arbitration_input_ids
        )

        slot_descriptions = {
            slot["id"]: slot["description"]
            for candidate_plan in (plan, replan or {})
            for slot in candidate_plan.get("required_slots", [])
        }
        provenance = []
        for memory in beliefs:
            slot_ids = [
                slot_id
                for slot_id, memory_ids in slot_support.items()
                if memory["id"] in memory_ids
            ]
            introducing_trace = next(
                (item for item in trace if memory["id"] in item["output_ids"]), None
            )
            provenance.append(
                {
                    "memory_id": memory["id"],
                    "source": "seed" if memory["id"] in seed_ids else "operation",
                    "operation_index": (
                        introducing_trace.get("operation_index")
                        if introducing_trace
                        else None
                    ),
                    "slot_ids": slot_ids,
                    "reason": (
                        "validated_seed_answer"
                        if fast_supports is not None
                        else (
                            "best_effort_unverified_candidate"
                            if memory["id"] in unverified_context_ids
                            else "; ".join(
                                slot_descriptions.get(slot_id, slot_id)
                                for slot_id in slot_ids
                            )
                        )
                    ),
                }
            )

        # Chronology follows the planner's requested axis.  A mixed-axis plan
        # keeps support order rather than silently substituting event_time.
        context_time_axis = self._context_time_axis(context_slots)
        if context_time_axis:
            beliefs.sort(
                key=lambda memory: self._date_for(memory, context_time_axis)
                or "9999-99-99"
            )
        belief_heading = (
            "=== UNVERIFIED RETRIEVAL CANDIDATES ==="
            if beliefs and final_ids.issubset(unverified_context_ids)
            else "=== RETRIEVED BELIEFS ==="
        )
        blocks = [
            belief_heading
            + "\n"
            + self._format_episode_context(beliefs)
        ]
        if planner_called:
            # Build a slot-coverage summary the Answer LLM can use to self-verify.
            # Internal memory IDs remain telemetry-only.
            belief_by_id = {m["id"]: m for m in beliefs}
            slot_coverage_lines = []
            unique_slots = list(
                {
                    slot.get("id"): slot
                    for slot in context_slots
                    if slot.get("id")
                }.values()
            )
            for slot in unique_slots:
                sid = slot["id"]
                role = slot.get("evidence_role", "ANSWER")
                desc = slot.get("description", sid)
                covered = slot_coverage.get(sid, False)
                support = slot_support.get(sid, [])
                if covered and support:
                    # Show the first supporting memory's claim snippet.
                    first = belief_by_id.get(support[0])
                    claim_snippet = (
                        str(first.get("claim", ""))[:100] if first else support[0]
                    )
                    slot_axis = str(slot.get("time_axis") or "")
                    slot_contract = (
                        f" type={slot.get('type', 'DIRECT')}"
                        + (f" time_axis={slot_axis}" if slot_axis else "")
                    )
                    slot_coverage_lines.append(
                        f"- [{role}]{slot_contract} {desc}: "
                        f"[FOUND] {claim_snippet}"
                    )
            if slot_coverage_lines:
                blocks.append(
                    "=== SLOT COVERAGE REQUIRED ===\n"
                    + "\n".join(slot_coverage_lines)
                )

        if all_relations:
            relation_memory = {memory["id"]: memory for memory in beliefs}
            blocks.append(
                "=== DIRECTED RELATIONS ===\n"
                + "\n".join(
                    "- "
                    + str(
                        relation_memory.get(r["source_id"], {}).get("claim", "")
                    )[:120]
                    + f" -[{r['type']}]-> "
                    + str(
                        relation_memory.get(r["target_id"], {}).get("claim", "")
                    )[:120]
                    for r in all_relations[:12]
                )
            )
        if evidence:
            blocks.append(
                "=== POINTER-VERIFIED EVIDENCE ===\n"
                + "\n".join(
                    f"- [{e.get('document_time', 'UNKNOWN')}] [{e.get('speaker', '?')}] {e.get('raw_text', '')}"
                    for e in evidence
                )
            )

        route_plan = replan or plan or {}
        # A direct controller route has no planner budget. Keep it genuinely
        # cheap; otherwise a one-seed answer silently pays the MEDIUM budget.
        tier = (
            "FAST"
            if fast_supports is not None
            else str(route_plan.get("budget_tier") or "MEDIUM").upper()
        )
        tier_budget = {
            "FAST": self.EASY_CONTEXT_TOKENS,
            "SMALL": self.SMALL_CONTEXT_TOKENS,
            "MEDIUM": self.MEDIUM_CONTEXT_TOKENS,
            "LARGE": self.HARD_CONTEXT_TOKENS,
        }.get(tier, self.HARD_CONTEXT_TOKENS if planner_called else self.EASY_CONTEXT_TOKENS)
        # Keep the context proportional to the plan.  The previous code gave
        # every planned query the LARGE budget, so even a small direct lookup
        # paid the latency/token cost of a full evidence bundle.
        budget = min(int(self.max_context_tokens), tier_budget)
        context = self._truncate_to_tokens("\n\n".join(blocks), budget)

        core_instruction = (
            "Ground every patient-specific claim in the supplied memory context. "
            "For inference or reasoning questions, use general domain knowledge to connect grounded "
            "endpoints and make the intermediate mechanism explicit; clearly distinguish that inferred "
            "bridge from remembered patient history. Never invent patient history, measurements, or treatment. "
            "Preserve exact names, values, units, qualifiers, and dates. For each TEMPORAL slot, answer from its "
            "declared time_axis, not from another date displayed on the same memory. Respect relation direction; "
            "never substitute one temporal axis for another. Answer only the comparison "
            "or temporal scope requested, without adding a different baseline. "
            "If conflicts remain, state uncertainty. Pointer evidence supports only its linked memory. "
            "For a multi-step clinical question, preserve exact measurements and chronology and present a "
            "complete cause-to-effect chain rather than merely restating retrieved facts. Follow the requested "
            "output format and answer directly."
        )
        if question_options:
            core_instruction += self._multiple_choice_answer_instruction(
                question_options.keys()
            )
            # Make the polarity check visible at the answer boundary. Retrieval
            # supplies shared participant evidence; it must not turn an unsafe
            # option into the answer merely because the memory mentions its
            # contraindication.
            core_instruction += (
                " Before selecting labels, restate the stem's predicate internally: "
                "if it asks which option is safe, recommended, or okay, exclude "
                "options contradicted by an allergy or contraindication; if it asks "
                "which is unsafe or contraindicated, select those contradicted "
                "options. Never output the contraindicated labels for a safe/okay "
                "stem just because those labels appear in the memory."
            )
        # Add clinical guidance only when the selected context is medical.
        is_medical = any(
            str(memory.get("scope") or "").lower() in CLINICAL_SCOPES
            for memory in beliefs
        )
        if is_medical:
            medical_extension = (
                " For a clinical decision, synthesize all covered roles instead of repeating one clinician interpretation: "
                "weigh exposure, objective trajectory, warning signs, contraindications, and feasible alternatives shown "
                "in context. Persistent objective worsening must not be reduced to routine monitoring merely because a "
                "reassuring assessment is also present. When asked whether to use a medicine, directly address medication "
                "eligibility and safety; lifestyle measures alone are not a complete answer when the context also supports "
                "a permitted backup or a contraindicated class. You may name a generally known option only when its safety "
                "properties satisfy the participant-specific constraints supplied in memory. For causal clinical reasoning, "
                "connect the grounded exposure and outcome through a concrete physiological chain: trigger, regulatory or "
                "hormonal response, measured trajectory, and resulting symptom. Include exact patient values and timing from "
                "memory; supply standard mechanisms such as sympathetic or HPA-axis activation only when clinically relevant, "
                "and label them as inference rather than remembered history."
            )
            instruction = core_instruction + medical_extension
        else:
            instruction = core_instruction
        full_system = f"{instruction}\n\n{context}"
        if system_message:
            full_system = f"{system_message}\n\n{full_system}"

        request_time = kwargs.get("batch_request_time") or time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        full_question = self._truncate_to_tokens(
            f"{question}\n\nCurrent Time: {request_time}", self.max_question_tokens
        )
        query_tokens["total"] = self._total_query_tokens(query_tokens)
        retrieval_ids = seed_ids | operation_output_ids
        elapsed_ms = round((time.time() - started) * 1000.0, 3)
        query_latency["prepare_wall"] = round(elapsed_ms / 1000.0, 3)
        rer = (len(retrieval_ids) - len(seed_ids)) / max(1, len(seed_ids))

        return {
            "messages": format_messages(full_question, full_system),
            "precomputed_answer": precomputed_answer,
            "retrieved_count": len(beliefs),
            "retrieved_memories": [dict(memory, type="memory") for memory in beliefs],
            "extra": {
                "method": "smart_mem0",
                "effective_runtime_config": self._effective_runtime_config(),
                "planner_called": planner_called,
                "plan": plan,
                "replan_called": replan_called,
                "replan": replan,
                "fast_gate": gate,
                "semantic_controller": controller,
                "fast_gate_skipped": bool(gate.get("skip_reason")),
                "fast_gate_skip_reason": gate.get("skip_reason", ""),
                "initial_seeds": len(initial_seeds),
                "planning_seeds": len(authorized_seeds),
                "final_memories": len(beliefs),
                "budget_tier": (
                    "FAST"
                    if fast_supports is not None
                    else (replan or plan).get("budget_tier", "LARGE")
                ),
                "context_memory_limit": context_memory_limit,
                "relations_used": sorted({r["type"] for r in all_relations}),
                "evidence_count": len(evidence),
                "raw_evidence_injected": bool(evidence),
                "memory_tokens": len(self._tokenizer.encode(context)),
                "context_temporal_axis": context_time_axis or "mixed",
                "retrieval_question": retrieval_question,
                "query_dates": list(frame.dates),
                "query_speaker_role": frame.speaker_role,
                "query_entities": list(frame.entities),
                "query_hard_entities": list(frame.hard_entities),
                "constraint_first_applied": bool(
                    frame.dates or frame.speaker_role or frame.hard_entities
                ),
                "slot_coverage": slot_coverage,
                "slot_validation": slot_validation_events,
                "sufficient": sufficient,
                "retrieval_trace": trace,
                "retrieval_provenance": provenance,
                "unverified_context_ids": sorted(unverified_context_ids & final_ids),
                "seed_ids": sorted(seed_ids),
                "operation_output_ids": sorted(operation_output_ids),
                "final_memory_ids": sorted(final_ids),
                "boundary_violation": boundary_violation,
                "evidence_boundary_violation": evidence_boundary_violation,
                "arbitration_expansion_violation": arbitration_expansion_violation,
                "elapsed_ms_until_sufficient": elapsed_ms if sufficient else None,
                "tokens_until_sufficient": (
                    query_tokens["total"] if sufficient else None
                ),
                "unique_memories_until_sufficient": (
                    len(retrieval_ids) if sufficient else None
                ),
                "retrieval_rounds_until_sufficient": len(trace) if sufficient else None,
                "retrieval_expansion_ratio": round(rer, 4),
                "zero_operation_plan": bool(
                    planner_called and plan.get("valid") and not plan.get("operations")
                ),
                "query_tokens": query_tokens,
                "query_latency": query_latency,
                "retrieval_elapsed_ms": elapsed_ms,
            },
        }

    def query(
        self, question: str, system_message: Optional[str] = None, **kwargs
    ) -> AgentResponse:
        prepared = self.prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        if prepared.get("precomputed_answer"):
            result = self.finalize_batch_query(
                prepared, str(prepared["precomputed_answer"])
            )
            result.query_time = prepared["extra"].get(
                "retrieval_elapsed_ms", 0
            ) / 1000.0
            latency = prepared["extra"].setdefault("query_latency", {})
            latency["total_wall"] = round(result.query_time, 3)
            return result
        started = time.time()
        # Query evaluation must be reproducible; write-time creativity is
        # configured separately and is already frozen in the memory snapshot.
        response = self._llm_client.chat(
            prepared["messages"], temperature=0.0, max_tokens=1024
        )
        usage = self._response_usage(
            response,
            "\n".join(message.get("content", "") for message in prepared["messages"]),
        )
        answer_latency = round(float(usage.get("latency", 0.0) or 0.0), 3)
        prepared["extra"]["query_tokens"]["answer"] = usage["total_tokens"]
        prepared["extra"]["query_tokens"]["total"] = self._total_query_tokens(
            prepared["extra"]["query_tokens"]
        )
        latency = prepared["extra"].setdefault("query_latency", {})
        latency["answer"] = answer_latency
        result = self.finalize_batch_query(prepared, response.content)
        result.query_time = (
            time.time()
            - started
            + prepared["extra"].get("retrieval_elapsed_ms", 0) / 1000.0
        )
        latency["total_wall"] = round(result.query_time, 3)
        return result

    def finalize_batch_query(
        self, prepared: Dict[str, Any], content: str
    ) -> AgentResponse:
        if prepared.get("precomputed_answer"):
            content = str(prepared["precomputed_answer"])
        return AgentResponse(
            output=content,
            query_time=0.0,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
            extra=prepared["extra"],
        )

    @staticmethod
    def record_batch_query_usage(
        response: AgentResponse,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        tokens = response.extra.setdefault("query_tokens", {})
        answer_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        tokens["answer"] = answer_tokens
        tokens["total"] = QueryMixin._total_query_tokens(tokens)

    def reset(self) -> None:
        super().reset()
        self._memories, self._evidence, self._relations = [], [], []
        self._belief_status, self._state_heads, self._profile_pack = {}, {}, {}
        self._memory_seq = self._evidence_seq = self._session_seq = 0
        self._loaded_frozen = False
        self._bm25 = self._embedding_matrix = None
        self._embedding_cache, self._index_dirty = {}, True
        self._last_write_stats = {
            "skipped_recaps": 0,
            "committed_memories": 0,
            "promoted_state_updates": 0,
            "reused_state_identities": 0,
        }
        if self._write_context is not None:
            self._write_context.clear()
        self._write_context = None
