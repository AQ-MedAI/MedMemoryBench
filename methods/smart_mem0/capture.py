"""Turn-window capture and ephemeral memory-write context."""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.metrics.pairwise import cosine_similarity

from .contracts import MemoryWriteContext
from .core import CoreMemoryMixin
from .prompts import MEMORY_WRITE_PROMPT


class CaptureMixin:
    """Extracts focal-turn memories without persisting interpretation context."""

    def _select_provisional_memories(
        self,
        query: str,
        provisional: List[Dict[str, Any]],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        if not provisional or limit <= 0:
            return []
        query_terms = set(self._tokenize(query))
        memory_texts = [self._memory_text(memory) for memory in provisional]
        try:
            vectors = self._embedder.encode(
                [query, *memory_texts], show_progress_bar=False
            )
            dense_scores = cosine_similarity(vectors[:1], vectors[1:])[0]
        except Exception:
            dense_scores = [0.0] * len(provisional)
        ranked = []
        for index, memory in enumerate(provisional):
            memory_terms = set(self._tokenize(memory_texts[index]))
            overlap = len(query_terms & memory_terms)
            dense_score = float(dense_scores[index])
            combined = dense_score + 0.03 * min(overlap, 4)
            ranked.append((combined, overlap, index, memory))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [self._snapshot(item[3]) for item in ranked[:limit]]

    def _select_write_beliefs(
        self,
        query: str,
        limit: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve current and relevant historical beliefs for write interpretation."""
        if not self._memories or limit <= 0:
            return [], []
        candidates = self._hybrid_search(
            query,
            top_k=min(max(24, limit * 4), len(self._memories)),
        )
        if not candidates:
            return [], []

        # Hybrid recall can surface either the old or current version. Add the
        # closest same-identity counterparts and scope-tolerant lineage matches
        # so extraction can reuse an established scope instead of fragmenting a
        # state merely because one session says "lab result" and another says
        # "glycemic control".
        candidate_by_id = {memory["id"]: memory for memory in candidates}
        for anchor in candidates[: max(5, limit)]:
            identity = self._state_identity(anchor)
            lineage = self._state_lineage_identity(anchor)
            if not identity and not lineage:
                continue
            versions = sorted(
                (
                    memory
                    for memory in self._memories
                    if (
                        (identity and self._state_identity(memory) == identity)
                        or (
                            lineage
                            and self._state_lineage_identity(memory) == lineage
                        )
                    )
                ),
                key=lambda memory: (
                    self._belief_status.get(memory["id"], "active")
                    in {"active", "refined", "conflicting"},
                    self._recency_date(memory),
                    memory.get("document_time", ""),
                    memory["id"],
                ),
                reverse=True,
            )
            for version in versions[:2]:
                if version["id"] in candidate_by_id:
                    continue
                enriched = self._snapshot(version)
                enriched.update(
                    {
                        "_score": float(anchor.get("_score", 0.0)) * 0.97,
                        "_bm25_score": 0.0,
                        "_dense_score": float(anchor.get("_dense_score", 0.0)),
                        "_overlap": 0,
                        "_status": self._belief_status.get(
                            version["id"], "active"
                        ),
                    }
                )
                candidate_by_id[version["id"]] = enriched
        candidates = list(candidate_by_id.values())

        profile_ids = set(self._profile_pack.get("stable_facts", []))
        profile_ids.update(self._profile_pack.get("preferences", []))
        profile_ids.update(self._profile_pack.get("plans", []))
        state_head_ids = {
            memory_id
            for values in self._profile_pack.get("state_heads", {}).values()
            for memory_id in values
        }
        profile_ids.update(state_head_ids)
        top_ids = {memory["id"] for memory in candidates[:5]}
        related_ids = {
            endpoint
            for relation in self._relations
            if relation["source_id"] in top_ids or relation["target_id"] in top_ids
            for endpoint in (relation["source_id"], relation["target_id"])
        }
        query_terms = set(self._tokenize(query))
        scored = []
        for rank, memory in enumerate(candidates):
            entity_terms = set(self._tokenize(" ".join(memory.get("entities", []))))
            entity_match = len(query_terms & entity_terms)
            active_status = self._belief_status.get(memory["id"], "active") in {
                "active",
                "refined",
                "conflicting",
            }
            bonus = (
                0.008 * min(entity_match, 3)
                + (0.009 if memory["id"] in state_head_ids else 0.0)
                + (0.004 if memory["id"] in related_ids else 0.0)
                + (0.006 if active_status else 0.0)
            )
            scored.append((float(memory.get("_score", 0.0)) + bonus, -rank, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        ranked = [item[2] for item in scored]

        core = [
            self._snapshot(memory)
            for memory in ranked
            if memory["id"] in profile_ids
            and memory["kind"] in {"FACT", "STATE"}
            and self._belief_status.get(memory["id"], "active")
            in {"active", "refined", "conflicting"}
        ][: min(2, limit)]
        core_ids = {memory["id"] for memory in core}
        relevant = [
            self._snapshot(memory) for memory in ranked if memory["id"] not in core_ids
        ][: max(0, limit - len(core))]
        return core, relevant

    def _update_write_context(
        self,
        context: MemoryWriteContext,
        focal_turns: List[Dict[str, Any]],
        previous_turn: Optional[Dict[str, Any]],
        document_time: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        continuity_enabled = self.write_context_mode in {"window", "full"}
        prior_enabled = self.write_context_mode == "full"
        query_parts = [document_time]
        if continuity_enabled and previous_turn:
            query_parts.append(self._format_turn(previous_turn, "CONTEXT TURN"))
        query_parts.extend(self._format_turn(turn) for turn in focal_turns)
        query = "\n".join(query_parts)
        selected_provisional = (
            self._select_provisional_memories(
                query,
                context.provisional_memories,
                limit=self.write_provisional_limit,
            )
            if continuity_enabled
            else []
        )
        core, relevant = (
            self._select_write_beliefs(query, self.write_prior_belief_limit)
            if prior_enabled
            else ([], [])
        )
        context.local_turn_context = (
            [self._snapshot(previous_turn)]
            if continuity_enabled and previous_turn
            else []
        )
        context.core_beliefs = core
        context.relevant_prior_beliefs = relevant
        return core + relevant, selected_provisional

    @staticmethod
    def _write_memory_payload(memories: Sequence[Dict[str, Any]]) -> str:
        if not memories:
            return "[]"
        fields = (
            "id",
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
            "document_time",
            "origin_document_time",
            "assertion_mode",
        )
        return json.dumps(
            [
                {
                    key: memory.get(key)
                    for key in fields
                    if memory.get(key) not in (None, "")
                }
                for memory in memories
            ],
            ensure_ascii=False,
        )

    @staticmethod
    def _contains_recap_reference(text: str) -> bool:
        patterns = (
            r"\bi (?:also )?remember (?:that )?you\b",
            r"\bi (?:still )?remember\b",
            r"\b(?:i|you|we) recall(?:ed|ing)?\b",
            r"\byou (?:previously |already )?(?:mentioned|said|told me|reported|shared)\b",
            r"\bas you (?:mentioned|said|told me|reported)\b",
            r"\b(?:earlier|previously|last time),? you\b",
            r"\bwe (?:already |previously )?(?:discussed|talked about|noted)\b",
            r"\byou(?:'ve| have) already\b",
            r"\bback (?:then|when)\b",
            r"\bin the past\b",
            r"\byou used to\b",
            r"\byour (?:previous|earlier|past|prior)\b",
            r"\b(?:had|have|has|used|did|was|were)\b[^.!?]{0,100}\bbefore\b",
            r"\b(?:last|previous|earlier|prior)\s+(?:visit|follow-up|followup|appointment|episode|time|experience|period)\b",
            r"\b(?:previous|earlier|prior)\s+(?:use|treatment|regimen|experience|incident)\b",
            r"\b(?:at|from|since)\s+(?:the\s+)?(?:beginning|start|end)\s+of\b",
            r"\b(?:had|has|have)\s+(?:already\s+)?(?:dropped|risen|increased|decreased|changed)\s+from\b",
            r"\b(?:the|that)\s+(?:earlier|previous|prior|initial)\s+(?:value|result|reading|episode|measurement)\b",
        )
        lowered = str(text or "").lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _recap_has_answer_bearing_detail(
        memory: Dict[str, Any], source_text: str = ""
    ) -> bool:
        """Keep recaps that add a searchable observation to the document ledger.

        Recap suppression is useful for preventing duplicate current-state
        heads, but it must not erase the only card that points to a later record
        containing an exact frequency, measurement, date, or transition. This
        predicate is intentionally structural: it looks for explicit temporal
        or quantitative detail in the extracted card/source, never for query
        intent or a benchmark label.
        """
        text = " ".join(
            str(value or "")
            for value in (
                memory.get("claim"),
                memory.get("value"),
                memory.get("verbatim_value"),
                memory.get("time_expression"),
                source_text,
            )
        ).lower()
        if not text.strip():
            return False
        # A recap with an already-resolved event date is normally a state
        # duplicate or a genuine new update handled by the regular DIRECT
        # path. The retention rule is for document occurrences whose recalled
        # detail has no independent event date and would otherwise disappear
        # from document-time retrieval.
        event_time = str(memory.get("event_time") or "").upper()
        if event_time not in {"", "UNKNOWN"}:
            return False
        has_number = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\b",
                text,
            )
        )
        has_temporal_marker = bool(
            re.search(
                r"\b(?:on|by|from|to|between|during|over|within|after|before|"
                r"recently|today|yesterday|month|week|day|time|times|recorded|"
                r"documented|mentioned|first|last|earliest|latest)\b",
                text,
            )
        )
        has_explicit_change = bool(
            re.search(
                r"\b(?:from\s+.+\s+to|increased|decreased|dropped|rose|"
                r"fell|returned|relapsed|changed|became|reduced|improved)\b",
                text,
            )
        )
        return bool((has_number and has_temporal_marker) or has_explicit_change)

    def _recover_unrepresented_quantified_evidence(
        self,
        focal_turns: Sequence[Dict[str, Any]],
        accepted: List[Dict[str, Any]],
        prior_beliefs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create a tiny provenance card for a quantified fact the LLM omitted.

        This is a loss-prevention guard for write-time extraction, not a second
        query retriever. It fires only for a bounded number of source sentences
        containing an explicit number plus temporal/frequency language that is
        absent from all extracted cards for that turn. The resulting card is a
        low-confidence FACT/RECAP pointer, so it cannot create a state head or
        overwrite a stronger LLM capture, but it keeps exact document evidence
        reachable through the normal BM25/dense index.
        """
        recovered: List[Dict[str, Any]] = []
        for turn in focal_turns:
            if len(recovered) >= 4:
                break
            turn_id = int(turn.get("turn_idx", -1))
            source = str(turn.get("raw_text") or "")
            if not source:
                continue
            source_sentences = [
                re.sub(r"\s+", " ", sentence).strip(" -*•\t")
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", source)
            ]
            accepted_for_turn = [
                memory
                for memory in accepted
                if turn_id in set(memory.get("source_turns") or [])
            ]
            covered_numbers = set(
                number
                for memory in accepted_for_turn
                for number in re.findall(
                    r"(?<![A-Za-z])\d+(?:\.\d+)?", 
                    " ".join(
                        (
                            str(memory.get("claim") or ""),
                            str(memory.get("value") or ""),
                            str(memory.get("verbatim_value") or ""),
                        )
                    ),
                )
            )
            for sentence in source_sentences:
                if len(recovered) >= 4 or len(sentence) < 18:
                    break
                numbers = set(
                    re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", sentence)
                )
                if not numbers or numbers.issubset(covered_numbers):
                    continue
                lowered = sentence.lower()
                if accepted_for_turn and numbers.intersection(covered_numbers) and re.search(
                    r"\b(?:now|changed|from|to|increased|decreased|dropped|rose|fell|"
                    r"returned|reduced|improved)\b",
                    lowered,
                ):
                    # An extracted state update already owns the changed value;
                    # do not manufacture a second generic card for its old
                    # comparison baseline.
                    continue
                temporal = bool(
                    re.search(
                        r"\b(?:every|once|twice|times?|per|over|within|during|"
                        r"before|after|from|to|between|past|recent|month|week|"
                        r"day|morning|night|recorded|documented|mentioned)\b",
                        lowered,
                    )
                )
                participant_or_event = bool(
                    re.search(
                        r"\b(?:patient|you|your|my|i|standing|waking|glucose|"
                        r"weight|vision|dizziness|fatigue|symptom|sleep)\b",
                        lowered,
                    )
                )
                if not temporal or not participant_or_event:
                    continue
                claim = sentence[:360]
                value = sentence[:160]
                candidate = {
                    "claim": claim,
                    "kind": "FACT",
                    "entities": ["patient"],
                    "subject": "patient",
                    "scope": "general",
                    "state_key": "",
                    "object_anchor": "",
                    "scope_entities": [],
                    "value": value,
                    "verbatim_value": value,
                    "stance": "AFFIRM",
                    "event_time": "UNKNOWN",
                    "time_expression": sentence[:120],
                    "assertion_mode": "DIRECT",
                    "origin_memory_id": "",
                    "planning_tags": ["TRAJECTORY"],
                    "decision_salience": 0.45,
                    "source_turns": [turn_id],
                    "confidence": 0.55,
                }
                if self._contains_recap_reference(source):
                    origin = self._best_prior_origin(candidate, prior_beliefs)
                    if origin:
                        candidate["assertion_mode"] = "RECAP"
                        candidate["origin_memory_id"] = origin["id"]
                recovered.append(candidate)
                covered_numbers.update(numbers)
        return recovered

    def _best_prior_origin(
        self,
        memory: Dict[str, Any],
        prior_beliefs: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        claim_terms = set(
            self._tokenize(f"{memory['claim']} {' '.join(memory.get('entities', []))}")
        )
        candidates = []
        for prior in prior_beliefs:
            prior_id = str(prior.get("id") or "")
            if not prior_id:
                continue
            prior_terms = set(
                self._tokenize(
                    f"{prior.get('claim', '')} {' '.join(prior.get('entities', []))}"
                )
            )
            overlap = len(claim_terms & prior_terms)
            same_identity = bool(
                self._state_identity(memory)
                and self._state_identity(memory) == self._state_identity(prior)
            )
            same_lineage = bool(
                self._state_lineage_identity(memory)
                and self._state_lineage_identity(memory)
                == self._state_lineage_identity(prior)
            )
            same_numbers = bool(
                set(re.findall(r"\d+(?:\.\d+)?", self._memory_value(memory)))
                and set(re.findall(r"\d+(?:\.\d+)?", self._memory_value(memory)))
                == set(re.findall(r"\d+(?:\.\d+)?", self._memory_value(prior)))
            )
            if overlap >= 2 or same_identity or same_lineage:
                candidates.append(
                    (
                        same_identity,
                        same_lineage,
                        same_numbers,
                        overlap,
                        float(prior.get("confidence", 0.0)),
                        prior,
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:-1], reverse=True)
        return candidates[0][-1]

    @staticmethod
    def _has_explicit_value_change(
        memory: Dict[str, Any], prior: Dict[str, Any]
    ) -> bool:
        if memory.get("stance") != prior.get("stance"):
            return True
        number_pattern = r"(?<![a-zA-Z])\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?"
        new_numbers = set(re.findall(number_pattern, memory.get("claim", "")))
        new_numbers.update(
            re.findall(number_pattern, CoreMemoryMixin._memory_value(memory))
        )
        old_numbers = set(re.findall(number_pattern, prior.get("claim", "")))
        old_numbers.update(
            re.findall(number_pattern, CoreMemoryMixin._memory_value(prior))
        )
        if new_numbers and old_numbers:
            # Extra prose around the same measurement is a refinement/recap, not
            # a changed state. A new numeric set is an explicit update.
            return new_numbers != old_numbers
        new_value = CoreMemoryMixin._normalised_value(memory)
        old_value = CoreMemoryMixin._normalised_value(prior)
        if new_value and old_value:
            return new_value != old_value
        generic_entities = {
            "patient",
            "user",
            "person",
            "speaker",
            "doctor",
            "assistant",
        }
        new_entities = {
            str(entity).strip().lower()
            for entity in memory.get("entities", [])
            if str(entity).strip().lower() not in generic_entities
        }
        old_entities = {
            str(entity).strip().lower()
            for entity in prior.get("entities", [])
            if str(entity).strip().lower() not in generic_entities
        }
        return bool(
            new_entities and old_entities and new_entities.isdisjoint(old_entities)
        )

    def _extract_write_window(
        self,
        focal_turns: List[Dict[str, Any]],
        previous_turn: Optional[Dict[str, Any]],
        document_time: str,
        prior_beliefs: List[Dict[str, Any]],
        provisional_memories: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        focal_ids = {turn["turn_idx"] for turn in focal_turns}
        local_context = (
            self._truncate_to_tokens(
                self._format_turn(previous_turn, "CONTEXT TURN"),
                self.WRITE_LOCAL_CONTEXT_TOKENS,
            )
            if previous_turn
            else "NONE"
        )
        response = self._llm_client.chat(
            [
                {
                    "role": "user",
                    "content": MEMORY_WRITE_PROMPT.format(
                        document_time=document_time,
                        prior_beliefs=self._write_memory_payload(prior_beliefs),
                        provisional_memories=self._write_memory_payload(
                            provisional_memories
                        ),
                        local_context=local_context,
                        focal_turns="\n".join(
                            self._format_turn(turn) for turn in focal_turns
                        ),
                    ),
                }
            ],
            temperature=0.0,
        )
        parsed, parse_stats = self._parse_capture_json(response.content)
        for key in ("malformed_windows", "salvaged_items", "discarded_items"):
            source_key = "malformed_window" if key == "malformed_windows" else key
            self._capture_parse_stats[key] = int(
                self._capture_parse_stats.get(key, 0)
            ) + int(parse_stats.get(source_key, 0))
        accepted, raw_to_accepted = [], {}
        focal_by_id = {turn["turn_idx"]: turn for turn in focal_turns}
        prior_by_id = {
            str(memory.get("id")): memory
            for memory in prior_beliefs
            if memory.get("id")
        }
        for raw_index, raw in enumerate(
            (parsed.get("memories") or [])[: self.MAX_NEW_MEMORIES]
        ):
            if not isinstance(raw, dict):
                continue
            normalized = self._normalise_memory(raw)
            if not normalized:
                continue
            source_turns = set(normalized["source_turns"])
            if not source_turns or not source_turns.issubset(focal_ids):
                continue
            source_text = "\n".join(
                focal_by_id[index]["raw_text"] for index in sorted(source_turns)
            )
            origin = prior_by_id.get(normalized.get("origin_memory_id", ""))
            if origin is None and (
                normalized["assertion_mode"] == "RECAP"
                or self._contains_recap_reference(source_text)
            ):
                origin = self._best_prior_origin(normalized, prior_beliefs)
            recap_signal = normalized[
                "assertion_mode"
            ] == "RECAP" or self._contains_recap_reference(source_text)
            if (
                origin is not None
                and recap_signal
                and not self._has_explicit_value_change(normalized, origin)
            ):
                normalized["assertion_mode"] = "RECAP"
                normalized["origin_memory_id"] = origin["id"]
            elif normalized["assertion_mode"] == "RECAP" or origin is not None:
                normalized["assertion_mode"] = "DIRECT"
                normalized["origin_memory_id"] = ""
            raw_to_accepted[raw_index] = len(accepted)
            accepted.append(normalized)

        # Keep a compact pointer when a source turn contains an exact
        # quantitative/temporal observation that the model's extraction
        # accidentally skipped. This is especially important for later
        # document-time queries over clinician recaps.
        accepted.extend(
            self._recover_unrepresented_quantified_evidence(
                focal_turns,
                accepted,
                prior_beliefs,
            )
        )

        links = []
        for link in parsed.get("causal_links") or []:
            if not isinstance(link, dict):
                continue
            try:
                raw_cause, raw_effect = int(link.get("cause_index")), int(
                    link.get("effect_index")
                )
            except (TypeError, ValueError):
                continue
            cause = raw_to_accepted.get(raw_cause)
            effect = raw_to_accepted.get(raw_effect)
            if cause is None or effect is None or cause == effect:
                continue
            relation_turns = link.get("source_turns") or []
            relation_turns = [
                int(value)
                for value in relation_turns
                if (isinstance(value, int) or str(value).isdigit())
                and int(value) in focal_ids
            ][:4]
            if not relation_turns:
                continue
            links.append(
                {
                    "cause_index": cause,
                    "effect_index": effect,
                    "source_turns": relation_turns,
                    "confidence": link.get("confidence", 0.75),
                }
            )
        return response.content, accepted, links
