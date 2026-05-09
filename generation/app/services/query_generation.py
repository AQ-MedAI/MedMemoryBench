"""Query generation service.

Supports five query types:
- EEM: Fill-in-the-blank from a single key point
- TLA: Time-location question from a single key point
- SUA: Status-update question from all key points in a category
- MQ/IG: Two-phase generation (trap reasoning then question synthesis)
"""
import json
import logging
from typing import Optional

from ..schemas.query import (
    QueryType,
    Query,
    Answer,
    SourceKeyPoint,
)
from ..prompts.query_generation import (
    EEM_SINGLE_KP_PROMPT,
    TLA_SINGLE_KP_PROMPT,
    SUA_CATEGORY_PROMPT,
    TRAP_REASONING_PROMPT,
    MQ_FROM_TRAP_PROMPT,
    IG_FROM_TRAP_PROMPT,
)
from .llm import get_llm_service, LLMService
from ..config import get_settings
from ..schemas.dialogue import filter_kps_for_query_generation

settings = get_settings()
logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 3000


class QueryGenerationService:
    """Query generation service."""

    def __init__(
        self,
        llm: Optional[LLMService] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize the service."""
        self.llm = llm or get_llm_service()
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_eem_from_single_kp(
        self,
        session_id: int,
        kp: dict,
        query_idx: int,
    ) -> Optional[Query]:
        """Generate an EEM (fill-in-the-blank) query from a single key point."""
        kp_text = self._format_single_kp(kp)

        prompt = EEM_SINGLE_KP_PROMPT.format(
            current_session_id=session_id,
            kp_text=kp_text,
        )

        caller = "query_generation.generate_eem_single"

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=caller,
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] EEM: LLM did not return a valid query")
                return None

            source_kp = SourceKeyPoint(
                category=kp.get("category", ""),
                name=kp.get("name", ""),
                content=kp.get("content", ""),
                trap_score=float(kp.get("trap_score", 0.5)),
                time=kp.get("time"),
                session_id=int(kp.get("session_id", session_id)),
            )

            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            query = Query(
                query_id=f"session_{session_id}_eem_{query_idx}",
                session_id=session_id,
                query_type=QueryType.EEM,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=[source_kp],
                metadata=query_data.get("metadata", {}),
            )

            logger.info(
                f"[QueryGen Response] caller={caller} "
                f"session_id={session_id} kp_name={kp.get('name')} success=True"
            )
            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller} "
                f"session_id={session_id} error={type(e).__name__}: {str(e)}"
            )
            return None

    async def generate_tla_from_single_kp(
        self,
        session_id: int,
        kp: dict,
        query_idx: int,
    ) -> Optional[Query]:
        """Generate a TLA (time-location) query from a single key point."""
        kp_text = self._format_single_kp(kp)

        prompt = TLA_SINGLE_KP_PROMPT.format(
            current_session_id=session_id,
            kp_text=kp_text,
        )

        caller = "query_generation.generate_tla_single"

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=caller,
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] TLA: LLM did not return a valid query")
                return None

            source_kp = SourceKeyPoint(
                category=kp.get("category", ""),
                name=kp.get("name", ""),
                content=kp.get("content", ""),
                trap_score=float(kp.get("trap_score", 0.5)),
                time=kp.get("time"),
                session_id=int(kp.get("session_id", session_id)),
            )

            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            query = Query(
                query_id=f"session_{session_id}_tla_{query_idx}",
                session_id=session_id,
                query_type=QueryType.TLA,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=[source_kp],
                metadata=query_data.get("metadata", {}),
            )

            logger.info(
                f"[QueryGen Response] caller={caller} "
                f"session_id={session_id} kp_name={kp.get('name')} success=True"
            )
            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller} "
                f"session_id={session_id} error={type(e).__name__}: {str(e)}"
            )
            return None

    async def generate_sua_from_category_kps(
        self,
        session_id: int,
        category: str,
        category_kps: list[dict],
        query_idx: int,
    ) -> Optional[Query]:
        """Generate an SUA (status-update) query from all key points in a category."""
        # Sort key points by time
        sorted_kps = sorted(category_kps, key=lambda x: x.get("time", "") or "")
        kps_text = self._format_category_kps(sorted_kps)

        prompt = SUA_CATEGORY_PROMPT.format(
            current_session_id=session_id,
            category=category,
            kps_text=kps_text,
            kps_count=len(category_kps),
        )

        caller = "query_generation.generate_sua_category"

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=caller,
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] SUA: LLM did not return a valid query")
                return None

            source_kps = []
            for kp in category_kps:
                source_kps.append(SourceKeyPoint(
                    category=kp.get("category", ""),
                    name=kp.get("name", ""),
                    content=kp.get("content", ""),
                    trap_score=float(kp.get("trap_score", 0.5)),
                    time=kp.get("time"),
                    session_id=int(kp.get("session_id", session_id)),
                ))

            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            query = Query(
                query_id=f"session_{session_id}_sua_{query_idx}",
                session_id=session_id,
                query_type=QueryType.SUA,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=source_kps,
                metadata=query_data.get("metadata", {}),
            )

            logger.info(
                f"[QueryGen Response] caller={caller} "
                f"session_id={session_id} category={category} kps_count={len(category_kps)} success=True"
            )
            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller} "
                f"session_id={session_id} category={category} error={type(e).__name__}: {str(e)}"
            )
            return None

    def _format_single_kp(self, kp: dict) -> str:
        """Format a single key point for prompt injection."""
        return (
            f"Category: {kp.get('category', 'Unknown')}\n"
            f"Name: {kp.get('name', 'Unknown')}\n"
            f"Content: {kp.get('content', '')}\n"
            f"Time: {kp.get('time', 'Unknown time')}\n"
            f"Difficulty score: {kp.get('trap_score', 0.5):.2f}\n"
            f"Source session: {kp.get('session_id', 0)}"
        )

    def _format_category_kps(self, kps: list[dict]) -> str:
        """Format all key points in a category for prompt injection."""
        lines = []
        for i, kp in enumerate(kps, 1):
            lines.append(
                f"[{i}] {kp.get('name', 'Unknown')} | "
                f"Time: {kp.get('time', 'Unknown')} | "
                f"Content: {kp.get('content', '')}"
            )
        return "\n".join(lines)

    # ========== Two-phase generation methods ==========

    async def generate_trap_query_two_phase(
        self,
        session_id: int,
        target_kp: dict,
        all_key_points: list[dict],
        query_type: str,
        query_idx: int,
        use_layered_memory: bool = True,
        trap_score_threshold: float = 0.5,
        source_event_content: str = "",
        existing_queries: Optional[list[dict]] = None,
    ) -> Optional[Query]:
        """Two-phase generation of trap-type queries (MQ/IG).

        Phase 1: Trap reasoning based on target kp and background info.
        Phase 2: Generate question from trap reasoning result.
        """
        caller = f"query_generation.trap_two_phase_{query_type}"

        # Filter background key points using layered memory
        if use_layered_memory:
            filtered_kps = filter_kps_for_query_generation(all_key_points, trap_score_threshold)
        else:
            filtered_kps = all_key_points

        logger.info(
            f"[QueryGen Request] caller={caller} "
            f"session_id={session_id} target_kp={target_kp.get('name')} "
            f"all_kps_count={len(all_key_points)} filtered_kps_count={len(filtered_kps)} "
            f"has_event_content={bool(source_event_content)} "
            f"existing_queries_count={len(existing_queries or [])}"
        )

        # Phase 1: Trap reasoning
        trap_reasoning = await self._phase1_trap_reasoning(
            target_kp=target_kp,
            all_key_points=filtered_kps,
            caller=caller,
            source_event_content=source_event_content,
        )

        if not trap_reasoning:
            logger.warning(
                f"[QueryGen] {query_type.upper()}: Trap reasoning phase failed"
            )
            return None

        # Phase 2: Generate question
        query = await self._phase2_generate_query(
            session_id=session_id,
            target_kp=target_kp,
            all_key_points=filtered_kps,
            trap_reasoning=trap_reasoning,
            query_type=query_type,
            query_idx=query_idx,
            caller=caller,
            source_event_content=source_event_content,
            existing_queries=existing_queries,
        )

        return query

    async def _phase1_trap_reasoning(
        self,
        target_kp: dict,
        all_key_points: list[dict],
        caller: str,
        source_event_content: str = "",
    ) -> Optional[dict]:
        """Phase 1: Trap reasoning based on target kp, source event, and background key points."""
        # Format background key points
        background_kps = self._format_background_kps(all_key_points, target_kp)

        # Format source event content
        if source_event_content:
            formatted_event = f"**Event content**: {source_event_content}"
        else:
            formatted_event = "(No source event info)"

        prompt = TRAP_REASONING_PROMPT.format(
            target_category=target_kp.get("category", "Unknown"),
            target_name=target_kp.get("name", "Unknown"),
            target_content=target_kp.get("content", ""),
            target_time=target_kp.get("time", "Unknown time"),
            target_session_id=target_kp.get("session_id", 0),
            source_event_content=formatted_event,
            background_kps=background_kps,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase1",
            )

            if not result.get("trap_points"):
                logger.warning(
                    f"[QueryGen] Phase1: Failed to generate valid trap points"
                )
                return None

            logger.info(
                f"[QueryGen Response] caller={caller}_phase1 "
                f"trap_points_count={len(result.get('trap_points', []))}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase1 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    async def _phase2_generate_query(
        self,
        session_id: int,
        target_kp: dict,
        all_key_points: list[dict],
        trap_reasoning: dict,
        query_type: str,
        query_idx: int,
        caller: str,
        source_event_content: str = "",
        existing_queries: Optional[list[dict]] = None,
    ) -> Optional[Query]:
        """Phase 2: Generate question from trap reasoning result."""
        prompt_map = {
            "mq": MQ_FROM_TRAP_PROMPT,
            "ig": IG_FROM_TRAP_PROMPT,
        }
        query_type_map = {
            "mq": QueryType.MQ,
            "ig": QueryType.IG,
        }

        prompt_template = prompt_map.get(query_type)
        if not prompt_template:
            logger.error(f"[QueryGen] Unknown query_type: {query_type}")
            return None

        trap_reasoning_text = json.dumps(trap_reasoning, ensure_ascii=False, indent=2)

        background_summary = self._format_background_summary(all_key_points)

        existing_queries_hint = self._format_existing_queries_hint(existing_queries)

        prompt = prompt_template.format(
            target_category=target_kp.get("category", "Unknown"),
            target_name=target_kp.get("name", "Unknown"),
            target_content=target_kp.get("content", ""),
            source_event_content=source_event_content if source_event_content else "(No source event info)",
            trap_reasoning=trap_reasoning_text,
            background_summary=background_summary,
            existing_queries_hint=existing_queries_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase2",
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(
                    f"[QueryGen] Phase2: LLM did not return a valid query"
                )
                return None

            # Parse source_key_points
            source_kps = []

            # Add target kp first
            source_kps.append(SourceKeyPoint(
                category=target_kp.get("category", ""),
                name=target_kp.get("name", ""),
                content=target_kp.get("content", ""),
                trap_score=float(target_kp.get("trap_score", 0.5)),
                time=target_kp.get("time"),
                session_id=int(target_kp.get("session_id", session_id)),
            ))

            # Add other related kps from LLM response
            for skp in query_data.get("source_key_points", []):
                try:
                    # Skip duplicate of target kp
                    if (skp.get("name") == target_kp.get("name") and
                        skp.get("content") == target_kp.get("content")):
                        continue

                    source_kps.append(SourceKeyPoint(
                        category=skp.get("category", ""),
                        name=skp.get("name", ""),
                        content=skp.get("content", ""),
                        trap_score=float(skp.get("trap_score", 0.5)),
                        time=skp.get("time"),
                        session_id=int(skp.get("session_id", session_id)),
                    ))
                except Exception as e:
                    logger.warning(
                        f"[QueryGen] source_key_point ParseFailed: {e}"
                    )

            # Parse answers
            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            metadata = query_data.get("metadata", {})

            # Add trap design info to metadata
            trap_design = query_data.get("trap_design", {})
            if trap_design:
                metadata["trap_design"] = trap_design

            # For IG type, add common wrong answer
            common_wrong = query_data.get("common_wrong_answer", {})
            if common_wrong:
                metadata["common_wrong_answer"] = common_wrong

            query = Query(
                query_id=f"session_{session_id}_{query_type}_{query_idx}",
                session_id=session_id,
                query_type=query_type_map[query_type],
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=source_kps,
                metadata=metadata,
            )

            logger.info(
                f"[QueryGen Response] caller={caller}_phase2 "
                f"session_id={session_id} query_type={query_type} success=True"
            )

            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase2 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    def _format_background_kps(self, all_key_points: list[dict], target_kp: dict) -> str:
        """Format background key points for trap reasoning, grouped by category."""
        if not all_key_points:
            return "(No background info available)"

        grouped: dict[str, list[dict]] = {}
        for kp in all_key_points:
            category = kp.get("category", "Unknown")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(kp)

        lines = []
        for category, kps in grouped.items():
            lines.append(f"### {category}")
            for kp in kps:
                is_target = (
                    kp.get("name") == target_kp.get("name") and
                    kp.get("content") == target_kp.get("content")
                )
                marker = " ★ [Target knowledge point]" if is_target else ""

                lines.append(
                    f"- [{kp.get('name', 'Unknown')}] {kp.get('content', '')}"
                    f" (Time: {kp.get('time', 'Unknown')}, session: {kp.get('session_id', 0)})"
                    f"{marker}"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_background_summary(self, all_key_points: list[dict]) -> str:
        """Format a concise background info summary for question generation."""
        if not all_key_points:
            return "(No background info available)"

        category_counts: dict[str, int] = {}
        key_info: list[str] = []

        for kp in all_key_points:
            category = kp.get("category", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

            content = kp.get("content", "")
            name = kp.get("name", "")

            # Extract critical info (allergies, contraindications, preferences)
            important_keywords = ["allergy", "contraindication", "dislike", "cannot", "prohibited", "preference", "like"]
            for keyword in important_keywords:
                if keyword in content or keyword in name:
                    key_info.append(f"- {name}: {content}")
                    break

        lines = ["**Patient info overview:**"]
        lines.append(f"Total {len(all_key_points)} knowledge point records")
        lines.append(f"Covering categories: {', '.join(category_counts.keys())}")

        if key_info:
            lines.append("\n**Important info notes (pay special attention):**")
            for info in list(set(key_info))[:10]:  # Max 10 items
                lines.append(info)

        return "\n".join(lines)


    def _format_existing_queries_hint(self, existing_queries: Optional[list[dict]]) -> str:
        """Format existing same-type queries as a deduplication hint for the LLM."""
        if not existing_queries:
            return ""

        lines = [
            "",
            "## ⚠️ Previously generated questions of the same type (avoid repetition!)",
            "",
            "Below are previously generated questions of this type. **Please avoid repeating the same test points, question content, or answer options**, and find new angles:",
            "",
        ]

        for i, q in enumerate(existing_queries, 1):
            question = q.get("question", "")
            answers = q.get("answers", [])

            lines.append(f"### Existing question {i}")
            lines.append(f"**Question**：{question}")

            if answers:
                lines.append("**Answer**：")
                for ans in answers:
                    content = ans.get("content", "")
                    is_correct = ans.get("is_correct", False)
                    marker = "✓" if is_correct else "✗"
                    lines.append(f"  - [{marker}] {content}")

            lines.append("")

        lines.extend([
            "**Requirements**:",
            "1. Do not ask questions similar to the above",
            "2. Do not use the same test angle or trap type",
            "3. Find new medical points to design questions",
        ])

        return "\n".join(lines)

    # ========== MCD (Multi-hop Clinical Deduction) three-phase generation ==========

    async def generate_mcd_three_phase(
        self,
        session_id: int,
        all_key_points: list[dict],
        events_data: list[dict],
        query_idx: int,
        existing_mcd_queries: Optional[list[dict]] = None,
        sessions_data: Optional[list[dict]] = None,
        dialogues_data: Optional[list[dict]] = None,
        max_retry: int = 2,
    ) -> Optional[Query]:
        """Multi-phase generation of MCD (Multi-hop Clinical Deduction) query.

        Phase 1: Causal chain mining from events and key points across sessions.
        Phase 2: Chain validation for medical correctness.
        Phase 1.5: Improve chain based on validation feedback (up to max_retry times).
        Phase 2.5: Content enrichment using real dialogue/event data.
        Phase 3: Question synthesis from the validated chain.
        """
        caller = "query_generation.mcd_three_phase"

        # Filter data to only include session_id <= current_session_id
        filtered_key_points = [
            kp for kp in all_key_points
            if kp.get("session_id", 0) <= session_id
        ]
        filtered_sessions = [
            s for s in (sessions_data or [])
            if s.get("session_id", 0) <= session_id
        ]
        # Filter events by valid session event_ids
        valid_event_ids = set()
        for s in filtered_sessions:
            eid = s.get("event_id")
            if eid:
                valid_event_ids.add(eid)
        filtered_events = [
            e for e in events_data
            if e.get("event_id") in valid_event_ids
        ]
        filtered_dialogues = [
            d for d in (dialogues_data or [])
            if d.get("session_id", 0) <= session_id
        ]

        logger.info(
            f"[QueryGen Request] caller={caller} "
            f"session_id={session_id} kps_count={len(filtered_key_points)} "
            f"events_count={len(filtered_events)} existing_mcd_count={len(existing_mcd_queries or [])}"
        )

        # Phase 1: Causal chain mining
        candidate_chains = await self._mcd_phase1_mine_causal_chains(
            session_id=session_id,
            all_key_points=filtered_key_points,
            events_data=filtered_events,
            existing_mcd_queries=existing_mcd_queries,
            sessions_data=filtered_sessions,
            caller=caller,
        )

        if not candidate_chains:
            logger.warning(f"[QueryGen] MCD Phase1: Failed to extract valid causal chain")
            return None

        # Phase 2: Chain validation with retry
        validated_chain = None
        current_chain = candidate_chains[0]  # Start with best candidate
        retry_count = 0

        while retry_count <= max_retry:
            validation_result = await self._mcd_phase2_validate_chain(
                session_id=session_id,
                candidate_chain=current_chain,
                all_key_points=filtered_key_points,
                existing_mcd_queries=existing_mcd_queries,
                caller=caller,
            )

            if validation_result.get("is_valid"):
                validated_chain = validation_result.get("refined_chain")
                break

            # Validation failed, attempt improvement
            rejection_reason = validation_result.get("rejection_reason", "Unknown reason")
            improvement_suggestions = validation_result.get("improvement_suggestions", [])

            logger.warning(
                f"[QueryGen] MCD Phase2: Validation failed (attempt {retry_count + 1}) - {rejection_reason}"
            )

            if retry_count >= max_retry:
                logger.warning(
                    f"[QueryGen] MCD: Reached max retries ({max_retry}), giving up generation"
                )
                break

            # Try improving the chain
            improved_chain = await self._mcd_phase1_5_improve_chain(
                session_id=session_id,
                rejected_chain=current_chain,
                rejection_reason=rejection_reason,
                improvement_suggestions=improvement_suggestions,
                all_key_points=filtered_key_points,
                events_data=filtered_events,
                existing_mcd_queries=existing_mcd_queries,
                sessions_data=filtered_sessions,
                caller=caller,
            )

            if not improved_chain:
                logger.warning(
                    f"[QueryGen] MCD Phase1.5: Failed to improve reasoning chain, trying next candidate"
                )
                # Try next candidate chain if available
                chain_idx = retry_count + 1
                if chain_idx < len(candidate_chains):
                    current_chain = candidate_chains[chain_idx]
                else:
                    break
            else:
                current_chain = improved_chain

            retry_count += 1

        if not validated_chain:
            logger.warning(f"[QueryGen] MCD: All retries failed, unable to generate valid reasoning chain")
            return None

        # Phase 2.5: Content enrichment
        enriched_chain = await self._mcd_phase2_5_enrich_chain(
            session_id=session_id,
            validated_chain=validated_chain,
            all_key_points=filtered_key_points,
            events_data=filtered_events,
            sessions_data=filtered_sessions,
            dialogues_data=filtered_dialogues,
            caller=caller,
        )

        # Phase 3: Question synthesis
        query = await self._mcd_phase3_synthesize_question(
            session_id=session_id,
            validated_chain=enriched_chain,
            all_key_points=filtered_key_points,
            query_idx=query_idx,
            caller=caller,
            existing_mcd_queries=existing_mcd_queries,
        )

        return query

    async def _mcd_phase1_mine_causal_chains(
        self,
        session_id: int,
        all_key_points: list[dict],
        events_data: list[dict],
        existing_mcd_queries: Optional[list[dict]],
        sessions_data: Optional[list[dict]],
        caller: str,
    ) -> Optional[list[dict]]:
        """MCD Phase 1: Mine causal chains from events and key points."""
        from ..prompts.mcd_generation import MCD_PHASE1_CAUSAL_CHAIN_MINING_PROMPT

        events_timeline = self._format_events_timeline(events_data, session_id)

        kps_by_session = self._format_kps_by_session(
            all_key_points, sessions_data, current_session_id=session_id
        )

        existing_chains_hint = self._format_existing_mcd_chains(existing_mcd_queries)

        prompt = MCD_PHASE1_CAUSAL_CHAIN_MINING_PROMPT.format(
            current_session_id=session_id,
            events_timeline=events_timeline,
            knowledge_points_by_session=kps_by_session,
            existing_chains_hint=existing_chains_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase1",
            )

            candidate_chains = result.get("candidate_chains", [])
            if not candidate_chains:
                logger.warning(f"[QueryGen] MCD Phase1: LLM did not return candidate chains")
                return None

            # Filter out chains containing future information
            valid_chains = []
            for chain in candidate_chains:
                is_valid = True
                for node in chain.get("nodes", []):
                    node_session_id = node.get("session_id", 0)
                    # session_id=0 means medical knowledge node, allowed
                    if node_session_id > session_id and node_session_id != 0:
                        logger.warning(
                            f"[QueryGen] MCD Phase1: Candidate chain contains future info "
                            f"(node session_id={node_session_id} > current={session_id})"
                        )
                        is_valid = False
                        break
                if is_valid:
                    valid_chains.append(chain)

            if not valid_chains:
                logger.warning(f"[QueryGen] MCD Phase1: All candidate chains contain future info")
                return None

            # Sort by quality_score
            valid_chains.sort(
                key=lambda x: x.get("quality_score", 0),
                reverse=True
            )

            logger.info(
                f"[QueryGen Response] caller={caller}_phase1 "
                f"candidates_count={len(valid_chains)} "
                f"best_score={valid_chains[0].get('quality_score', 0):.2f}"
            )

            return valid_chains

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase1 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    async def _mcd_phase2_validate_chain(
        self,
        session_id: int,
        candidate_chain: dict,
        all_key_points: list[dict],
        existing_mcd_queries: Optional[list[dict]],
        caller: str,
    ) -> dict:
        """MCD Phase 2: Validate the reasoning chain for medical correctness.

        Returns a dict with:
        - is_valid: whether validation passed
        - refined_chain: refined chain (if valid)
        - rejection_reason: reason for rejection (if invalid)
        - improvement_suggestions: suggestions for improvement (if invalid)
        """
        from ..prompts.mcd_generation import MCD_PHASE2_CHAIN_VALIDATION_PROMPT

        candidate_chain_json = json.dumps(candidate_chain, ensure_ascii=False, indent=2)

        all_kps_text = self._format_all_kps_for_validation(all_key_points)

        existing_queries_hint = self._format_existing_mcd_queries_hint(existing_mcd_queries)

        prompt = MCD_PHASE2_CHAIN_VALIDATION_PROMPT.format(
            current_session_id=session_id,
            candidate_chain_json=candidate_chain_json,
            all_knowledge_points=all_kps_text,
            existing_queries_hint=existing_queries_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase2",
            )

            validation_result = result.get("validation_result", {})
            is_valid = validation_result.get("is_valid", False)

            if not is_valid:
                rejection_reason = result.get("rejection_reason", "Unknown reason")
                improvement_suggestions = result.get("improvement_suggestions", [])
                logger.info(
                    f"[QueryGen Response] caller={caller}_phase2 "
                    f"is_valid=False reason={rejection_reason}"
                )
                return {
                    "is_valid": False,
                    "rejection_reason": rejection_reason,
                    "improvement_suggestions": improvement_suggestions,
                    "validation_details": validation_result.get("validation_details", {}),
                }

            refined_chain = result.get("refined_chain", {})
            if not refined_chain.get("nodes"):
                logger.warning(f"[QueryGen] MCD Phase2: Did not return refined chain")
                return {
                    "is_valid": False,
                    "rejection_reason": "Validation passed but refined chain not returned",
                    "improvement_suggestions": [],
                }

            logger.info(
                f"[QueryGen Response] caller={caller}_phase2 "
                f"is_valid=True overall_score={validation_result.get('overall_score', 0):.2f}"
            )

            return {
                "is_valid": True,
                "refined_chain": refined_chain,
                "validation_details": validation_result.get("validation_details", {}),
            }

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase2 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return {
                "is_valid": False,
                "rejection_reason": f"Validation error: {str(e)}",
                "improvement_suggestions": [],
            }


    async def _mcd_phase1_5_improve_chain(
        self,
        session_id: int,
        rejected_chain: dict,
        rejection_reason: str,
        improvement_suggestions: list[str],
        all_key_points: list[dict],
        events_data: list[dict],
        existing_mcd_queries: Optional[list[dict]],
        sessions_data: Optional[list[dict]],
        caller: str,
    ) -> Optional[dict]:
        """MCD Phase 1.5: Improve chain based on validation feedback."""
        from ..prompts.mcd_generation import MCD_PHASE1_5_CHAIN_IMPROVEMENT_PROMPT

        rejected_chain_json = json.dumps(rejected_chain, ensure_ascii=False, indent=2)

        suggestions_text = "\n".join(
            f"- {s}" for s in improvement_suggestions
        ) if improvement_suggestions else "(No specific suggestions)"

        events_timeline = self._format_events_timeline(events_data, session_id)

        kps_by_session = self._format_kps_by_session(all_key_points, sessions_data)

        existing_chains_hint = self._format_existing_mcd_chains(existing_mcd_queries)

        prompt = MCD_PHASE1_5_CHAIN_IMPROVEMENT_PROMPT.format(
            current_session_id=session_id,
            rejected_chain_json=rejected_chain_json,
            rejection_reason=rejection_reason,
            improvement_suggestions=suggestions_text,
            events_timeline=events_timeline,
            knowledge_points_by_session=kps_by_session,
            existing_chains_hint=existing_chains_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase1_5",
            )

            improved_chain = result.get("improved_chain", {})
            if not improved_chain.get("nodes"):
                logger.warning(f"[QueryGen] MCD Phase1.5: LLM did not return a valid improved chain")
                return None

            # Verify improved chain has no future information
            for node in improved_chain.get("nodes", []):
                node_session_id = node.get("session_id", 0)
                if node_session_id > session_id and node_session_id != 0:
                    logger.warning(
                        f"[QueryGen] MCD Phase1.5: Improved chain still contains future info "
                        f"(node session_id={node_session_id} > current={session_id})"
                    )
                    return None

            improvement_made = improved_chain.get("improvement_made", "")
            logger.info(
                f"[QueryGen Response] caller={caller}_phase1_5 "
                f"improvement_made={improvement_made[:50]}..."
            )

            return improved_chain

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase1_5 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    async def _mcd_phase2_5_enrich_chain(
        self,
        session_id: int,
        validated_chain: dict,
        all_key_points: list[dict],
        events_data: list[dict],
        sessions_data: Optional[list[dict]],
        dialogues_data: Optional[list[dict]],
        caller: str,
    ) -> Optional[dict]:
        """MCD Phase 2.5: Enrich chain content using real dialogue/event data.

        Returns the enriched chain, or the original chain on failure.
        """
        from ..prompts.mcd_generation import MCD_PHASE2_5_CONTENT_ENRICHMENT_PROMPT

        # Collect session IDs involved in the chain
        involved_sessions = set()
        for node in validated_chain.get("nodes", []):
            sid = node.get("session_id", 0)
            if sid > 0:
                involved_sessions.add(sid)

        if not involved_sessions:
            logger.warning(f"[QueryGen] MCD Phase2.5: Referenced session not found")
            return validated_chain

        validated_chain_json = json.dumps(validated_chain, ensure_ascii=False, indent=2)

        dialogues_content = self._format_dialogues_for_sessions(
            dialogues_data, involved_sessions, sessions_data
        )

        events_content = self._format_events_for_sessions(
            events_data, involved_sessions, sessions_data
        )

        kps_content = self._format_kps_for_sessions(
            all_key_points, involved_sessions
        )

        prompt = MCD_PHASE2_5_CONTENT_ENRICHMENT_PROMPT.format(
            current_session_id=session_id,
            validated_chain_json=validated_chain_json,
            dialogues_content=dialogues_content,
            events_content=events_content,
            kps_content=kps_content,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase2_5",
            )

            enriched_chain = result.get("enriched_chain", {})
            if not enriched_chain.get("nodes"):
                logger.warning(f"[QueryGen] MCD Phase2.5: LLM did not return a valid refined chain, using original chain")
                return validated_chain

            enrichment_summary = enriched_chain.get("enrichment_summary", "")
            logger.info(
                f"[QueryGen Response] caller={caller}_phase2_5 "
                f"enrichment_summary={enrichment_summary[:80]}..."
            )

            return enriched_chain

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase2_5 "
                f"error={type(e).__name__}: {str(e)}"
            )
            # Fall back to original chain on error
            return validated_chain

    def _format_dialogues_for_sessions(
        self,
        dialogues_data: Optional[list[dict]],
        involved_sessions: set[int],
        sessions_data: Optional[list[dict]],
    ) -> str:
        """Format dialogue content for involved sessions."""
        if not dialogues_data:
            return "(No dialogue data)"

        # Build session_id -> event_id mapping
        session_to_event = {}
        if sessions_data:
            for s in sessions_data:
                sid = s.get("session_id", 0)
                eid = s.get("event_id")
                if sid and eid:
                    session_to_event[sid] = eid

        lines = []
        for sid in sorted(involved_sessions):
            dialogue = None
            for d in dialogues_data:
                if d.get("session_id") == sid:
                    dialogue = d
                    break

            if not dialogue:
                lines.append(f"\n### Session {sid}")
                lines.append("(No dialogue data found)")
                continue

            lines.append(f"\n### Session {sid}")
            event_id = session_to_event.get(sid)
            if event_id:
                lines.append(f"Associated event ID: {event_id}")

            turns = dialogue.get("turns", [])
            if turns:
                lines.append("Dialogue content:")
                for turn in turns[-6:]:  # Last 6 turns max
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    role_label = "Patient" if role == "user" else "Doctor"
                    # Truncate long content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    lines.append(f"- {role_label}: {content}")
            else:
                lines.append("(No dialogue turns)")

        return "\n".join(lines)

    def _format_events_for_sessions(
        self,
        events_data: list[dict],
        involved_sessions: set[int],
        sessions_data: Optional[list[dict]],
    ) -> str:
        """Format event details for involved sessions."""
        if not events_data:
            return "(No event data)"

        # Build session_id -> event_id mapping
        session_to_event = {}
        if sessions_data:
            for s in sessions_data:
                sid = s.get("session_id", 0)
                eid = s.get("event_id")
                if sid and eid:
                    session_to_event[sid] = eid

        relevant_event_ids = {
            session_to_event.get(sid)
            for sid in involved_sessions
            if session_to_event.get(sid)
        }

        lines = []
        for event in events_data:
            eid = event.get("event_id")
            if eid not in relevant_event_ids:
                continue

            lines.append(f"\n### Event {eid}")
            lines.append(f"Date: {event.get('event_date', 'Unknown')}")
            lines.append(f"Type: {event.get('type', 'Unknown')}")
            lines.append(f"Content: {event.get('event', '')}")
            triggered_by = event.get("triggered_by", [])
            if triggered_by:
                lines.append(f"Trigger: triggered by event {triggered_by}")

        return "\n".join(lines) if lines else "(No related events)"

    def _format_kps_for_sessions(
        self,
        all_key_points: list[dict],
        involved_sessions: set[int],
    ) -> str:
        """Format key points for involved sessions."""
        if not all_key_points:
            return "(No knowledge points data)"

        grouped: dict[int, list[dict]] = {}
        for kp in all_key_points:
            sid = kp.get("session_id", 0)
            if sid in involved_sessions:
                if sid not in grouped:
                    grouped[sid] = []
                grouped[sid].append(kp)

        lines = []
        for sid in sorted(grouped.keys()):
            kps = grouped[sid]
            lines.append(f"\n### Session {sid} Knowledge points")
            for kp in kps:
                category = kp.get("category", "Unknown")
                name = kp.get("name", "Unknown")
                content = kp.get("content", "")
                trap_score = kp.get("trap_score", 0.5)
                lines.append(f"- [{category}] {name}: {content} (trap_score={trap_score:.2f})")

        return "\n".join(lines) if lines else "(No related knowledge points)"

    async def _mcd_phase3_synthesize_question(
        self,
        session_id: int,
        validated_chain: dict,
        all_key_points: list[dict],
        query_idx: int,
        caller: str,
        existing_mcd_queries: Optional[list[dict]] = None,
    ) -> Optional[Query]:
        """MCD Phase 3: Synthesize question from validated reasoning chain."""
        from ..prompts.mcd_generation import MCD_PHASE3_QUESTION_SYNTHESIS_PROMPT

        validated_chain_json = json.dumps(validated_chain, ensure_ascii=False, indent=2)

        background_summary = self._format_background_summary(all_key_points)

        # Use v2 dedup format with specificity info
        existing_questions_list = self._format_existing_questions_for_dedup_v2(
            existing_mcd_queries
        )

        prompt = MCD_PHASE3_QUESTION_SYNTHESIS_PROMPT.format(
            current_session_id=session_id,
            query_idx=query_idx,
            validated_chain_json=validated_chain_json,
            background_summary=background_summary,
            existing_questions_list=existing_questions_list,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase3",
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] MCD Phase3: LLM did not return a valid query")
                return None

            diversity_check = result.get("diversity_check", {})

            # Parse answers with defensive type checks
            answers = []
            answers_raw = query_data.get("answers", [])
            if isinstance(answers_raw, list):
                for ans_data in answers_raw:
                    if isinstance(ans_data, dict):
                        answers.append(Answer(
                            content=ans_data.get("content", ""),
                            is_correct=ans_data.get("is_correct", True),
                            explanation=ans_data.get("explanation"),
                        ))
                    elif isinstance(ans_data, str):
                        # Handle string-form answers from LLM
                        answers.append(Answer(
                            content=ans_data,
                            is_correct=True,
                            explanation=None,
                        ))

            # Parse source_key_points with defensive type checks
            source_kps = []
            source_kps_raw = query_data.get("source_key_points", [])
            if isinstance(source_kps_raw, list):
                for skp in source_kps_raw:
                    if isinstance(skp, dict):
                        source_kps.append(SourceKeyPoint(
                            category=skp.get("category", ""),
                            name=skp.get("name", ""),
                            content=skp.get("content", ""),
                            trap_score=float(skp.get("trap_score", 0.5)),
                            time=skp.get("time"),
                            session_id=int(skp.get("session_id", session_id)),
                        ))

            # Build metadata with defensive type checks
            metadata_raw = query_data.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            reasoning_chain_raw = query_data.get("reasoning_chain", [])
            reasoning_chain = reasoning_chain_raw if isinstance(reasoning_chain_raw, list) else []
            metadata["reasoning_chain"] = reasoning_chain

            required_memory_nodes_raw = query_data.get("required_memory_nodes", [])
            required_memory_nodes = required_memory_nodes_raw if isinstance(required_memory_nodes_raw, list) else []
            metadata["required_memory_nodes"] = required_memory_nodes

            question_style = query_data.get("question_style", "")
            if question_style:
                metadata["question_style"] = question_style

            if diversity_check:
                metadata["diversity_check"] = diversity_check

            query = Query(
                query_id=f"session_{session_id}_mcd_{query_idx}",
                session_id=session_id,
                query_type=QueryType.MCD,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=source_kps,
                metadata=metadata,
            )

            logger.info(
                f"[QueryGen Response] caller={caller}_phase3 "
                f"session_id={session_id} hop_count={metadata.get('hop_count', 0)} "
                f"question_style={question_style} "
                f"reasoning_pattern={metadata.get('reasoning_pattern', 'unknown')} success=True"
            )

            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase3 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    def _format_existing_questions_for_dedup(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing questions for deduplication check."""
        if not existing_mcd_queries:
            return "(No generated questions yet)"

        lines = []
        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            metadata = q.get("metadata", {})
            pattern = metadata.get("question_pattern", "Unknown")
            lines.append(f"{i}. [{pattern} pattern] {question}")

        return "\n".join(lines)

    def _format_existing_questions_for_dedup_v2(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing questions for deduplication check (v2 with specificity info)."""
        if not existing_mcd_queries:
            return "(No generated questions yet)"

        lines = [
            "Below are generated MCD questions. Please ensure the new question's **specificity entry type** differs from these:",
            "",
        ]

        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            metadata = q.get("metadata", {})

            specificity_type = metadata.get("specificity_type", "Unknown")
            question_pattern = metadata.get("question_pattern", "Unknown")

            question_design = metadata.get("question_design", {})
            professional_info = question_design.get("professional_info_used", [])
            hidden_reasoning = question_design.get("hidden_reasoning", "")

            lines.append(f"### Existing question {i}")
            lines.append(f"**Question**: {question}")
            lines.append(f"**Specificity type**: {specificity_type}")
            lines.append(f"**Question pattern**: {question_pattern}")

            if professional_info:
                lines.append(f"**Related info**: {', '.join(professional_info[:5])}")

            if hidden_reasoning:
                lines.append(f"**Core reasoning**: {hidden_reasoning[:150]}...")

            lines.append("")

        lines.extend([
            "---",
            "**⚠️ Deduplication requirements**:",
            "1. The new question's **specificity entry type** must differ from the above questions",
            "2. Do not just change numerical values; use a **completely different angle**",
            "3. If medication-related questions already exist above, consider other types such as life events/lab results/symptom combinations",
        ])

        return "\n".join(lines)

    def _format_events_timeline(
        self,
        events_data: list[dict],
        current_session_id: int,
    ) -> str:
        """Format events timeline for MCD generation.

        Note: events_data should already be filtered to session_id <= current_session_id.
        """
        if not events_data:
            return f"(No event data up to session {current_session_id})"

        # Sort by event date
        sorted_events = sorted(events_data, key=lambda x: x.get("event_date", ""))

        lines = [f"⚠️ Note: The following events are from session {current_session_id} and earlier. Do not use any information beyond this scope.", ""]
        for event in sorted_events:
            event_id = event.get("event_id", 0)
            event_date = event.get("event_date", "Unknown date")
            event_type = event.get("type", "Unknown type")
            event_content = event.get("event", "")
            triggered_by = event.get("triggered_by", [])

            lines.append(f"[Event {event_id}] {event_date} ({event_type})")
            lines.append(f"Content: {event_content}")
            if triggered_by:
                lines.append(f"Trigger: triggered by event {triggered_by}")
            lines.append("---")

        return "\n".join(lines)

    def _format_kps_by_session(
        self,
        all_key_points: list[dict],
        sessions_data: Optional[list[dict]],
        current_session_id: Optional[int] = None,
    ) -> str:
        """Format key points grouped by session.

        Note: all_key_points should already be filtered to session_id <= current_session_id.
        """
        if not all_key_points:
            return "(No knowledge points)"

        # Build session_id -> date mapping
        session_dates = {}
        if sessions_data:
            for session in sessions_data:
                sid = session.get("session_id")
                event_info = session.get("event_info", {})
                session_dates[sid] = event_info.get("date", "Unknown date")

        grouped: dict[int, list[dict]] = {}
        for kp in all_key_points:
            sid = kp.get("session_id", 0)
            if sid not in grouped:
                grouped[sid] = []
            grouped[sid].append(kp)

        lines = []
        if current_session_id:
            lines.append(f"⚠️ Note: The following knowledge points are from session {current_session_id} and earlier. Do not assume or use any information beyond this scope.")
            lines.append("")

        max_session_in_data = max(grouped.keys()) if grouped else 0
        for sid in sorted(grouped.keys()):
            kps = grouped[sid]
            date = session_dates.get(sid, kps[0].get("time", "Unknown date") if kps else "Unknown date")
            lines.append(f"\n### Session {sid} ({date})")

            for kp in kps:
                lines.append(
                    f"- [{kp.get('category', 'Unknown')}] {kp.get('name', 'Unknown')}: "
                    f"{kp.get('content', '')}"
                )

        if current_session_id:
            lines.append(f"\n⚠️ Max available session ID: {max_session_in_data}")

        return "\n".join(lines)

    def _format_existing_mcd_chains(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing MCD reasoning chains as a deduplication hint."""
        if not existing_mcd_queries:
            return "(No generated reasoning chains yet)"

        lines = [
            "## ⚠️ Previously generated reasoning chains (avoid repetition!)",
            "",
            "Below are generated MCD questions and their reasoning chains. **Please find new, non-repetitive reasoning paths**:",
            "",
        ]

        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            metadata = q.get("metadata", {})
            reasoning_chain = metadata.get("reasoning_chain", [])
            pattern = metadata.get("reasoning_pattern", "Unknown")

            lines.append(f"### Existing question {i}")
            lines.append(f"**Question**: {question}")
            lines.append(f"**Reasoning pattern**: {pattern}")
            lines.append(f"**Reasoning chain**:")

            for node in reasoning_chain:
                node_id = node.get("node_id", 0)
                content = node.get("content", "")
                role = node.get("role", "")
                lines.append(f"  - Node {node_id} ({role}): {content}")

            lines.append("")

        return "\n".join(lines)

    def _format_all_kps_for_validation(
        self,
        all_key_points: list[dict],
    ) -> str:
        """Format all key points for the validation phase, grouped by category."""
        if not all_key_points:
            return "(No knowledge points)"

        grouped: dict[str, list[dict]] = {}
        for kp in all_key_points:
            cat = kp.get("category", "Unknown")
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(kp)

        lines = []
        for cat, kps in grouped.items():
            lines.append(f"\n### {cat}")
            for kp in kps:
                lines.append(
                    f"- [{kp.get('name', 'Unknown')}] {kp.get('content', '')} "
                    f"(Session {kp.get('session_id', 0)}, {kp.get('time', 'Unknown time')})"
                )

        return "\n".join(lines)

    def _format_existing_mcd_queries_hint(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing MCD queries hint for validation phase."""
        if not existing_mcd_queries:
            return "(No generated questions yet)"

        lines = []
        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            answers = q.get("answers", [])
            correct_answer = next(
                (a.get("content", "") for a in answers if a.get("is_correct")),
                ""
            )

            lines.append(f"{i}. Question: {question}")
            lines.append(f"   Answer: {correct_answer[:200]}...")
            lines.append("")

        return "\n".join(lines)
