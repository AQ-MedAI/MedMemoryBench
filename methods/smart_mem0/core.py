"""Core normalization, evidence staging, and hybrid indexing."""

import copy
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.llm_client import create_llm_client

from .contracts import (
    GENERIC_OBJECT_ANCHORS,
    GENERIC_STATE_KEYS,
    MONTH_NUMBERS,
    SCOPE_ALIASES,
    STATE_KEY_NOISE,
    STATE_LIKE_KINDS,
    STOPWORDS,
    VALID_ASSERTION_MODES,
    VALID_KINDS,
    VALID_PLANNING_TAGS,
    MemoryWriteContext,
    QueryFrame,
)

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:
    raise ImportError(f"[SmartMem0] Missing dependency: {exc}")


class CoreMemoryMixin:
    """Owns runtime state and deterministic memory/index primitives."""

    METHOD_TYPE = "agentic_memory"
    RRF_K = 60
    INITIAL_TOP_K = 8
    EASY_MEMORY_LIMIT = 5
    HARD_MEMORY_LIMIT = 8
    MAX_EVIDENCE = 6
    EASY_CONTEXT_TOKENS = 1200
    SMALL_CONTEXT_TOKENS = 1800
    MEDIUM_CONTEXT_TOKENS = 3600
    HARD_CONTEXT_TOKENS = 6000
    MAX_NEW_MEMORIES = 36
    WRITE_PRIOR_BELIEF_LIMIT = 8
    WRITE_PROVISIONAL_LIMIT = 4
    WRITE_LOCAL_CONTEXT_TOKENS = 500
    WRITE_WINDOW_MAX_TURNS = 64
    # Bound extraction windows so a single completion does not compress away
    # low-salience atomic facts from a long session. Splits happen at turns.
    # Keep extraction local enough to preserve atomic turn facts. Long sessions
    # are split only at turn boundaries and receive bounded continuity context.
    WRITE_WINDOW_MAX_TOKENS = 3000
    VALID_WRITE_CONTEXT_MODES = {"none", "window", "full"}

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_provider: str = "local",
        embedding_model_path: Optional[str] = None,
        retrieve_num: int = 8,
        **kwargs,
    ):
        super().__init__(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        self._llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        # Phase-1 read contract: exactly Top-8 RRF candidates and Top-3 seeds.
        self.retrieve_num = min(8, max(3, int(retrieve_num or self.INITIAL_TOP_K)))
        self.max_context_tokens = int(kwargs.get("max_context_tokens") or 32768)
        self.max_question_tokens = int(kwargs.get("max_question_tokens") or 1200)
        self.enable_memory_write = bool(
            kwargs.get(
                "enable_memory_write", kwargs.get("enable_profile_extraction", True)
            )
        )
        self.enable_planner = bool(kwargs.get("enable_planner", True))
        self.enable_unified_controller = bool(
            kwargs.get("enable_unified_controller", True)
        )
        self.enable_replan = bool(kwargs.get("enable_replan", False))
        self.enable_zero_result_recovery = bool(
            kwargs.get("enable_zero_result_recovery", True)
        )
        self.enable_planner_repair = bool(kwargs.get("enable_planner_repair", False))
        self.enable_slot_support_validation = bool(
            kwargs.get("enable_slot_support_validation", False)
        )
        self.frozen_memory_path = kwargs.get("frozen_memory_path")
        self.write_window_max_turns = max(
            1, int(kwargs.get("write_window_max_turns") or self.WRITE_WINDOW_MAX_TURNS)
        )
        self.write_window_max_tokens = max(
            256,
            int(kwargs.get("write_window_max_tokens") or self.WRITE_WINDOW_MAX_TOKENS),
        )
        self.write_context_mode = (
            str(kwargs.get("write_context_mode") or "full").strip().lower()
        )
        if self.write_context_mode not in self.VALID_WRITE_CONTEXT_MODES:
            raise ValueError("write_context_mode must be one of: none, window, full")
        self.write_prior_belief_limit = max(
            0,
            int(
                kwargs.get("write_prior_belief_limit")
                if kwargs.get("write_prior_belief_limit") is not None
                else self.WRITE_PRIOR_BELIEF_LIMIT
            ),
        )
        self.write_provisional_limit = max(
            0,
            int(
                kwargs.get("write_provisional_limit")
                if kwargs.get("write_provisional_limit") is not None
                else self.WRITE_PROVISIONAL_LIMIT
            ),
        )

        local_model = embedding_model_path or embedding_model
        if local_model.startswith("sentence-transformers/"):
            local_model = local_model.split("/", 1)[1]
        print("[SmartMem0] Loading local embedding model...")
        try:
            self._embedder = SentenceTransformer(local_model, local_files_only=True)
        except Exception:
            self._embedder = SentenceTransformer(local_model)

        self._memories: List[Dict[str, Any]] = []
        self._evidence: List[Dict[str, Any]] = []
        self._relations: List[Dict[str, Any]] = []
        self._belief_status: Dict[str, str] = {}
        self._state_heads: Dict[str, List[str]] = {}
        self._profile_pack: Dict[str, Any] = {}
        self._memory_seq = self._evidence_seq = self._session_seq = 0
        self._loaded_frozen = False
        self._bm25 = None
        self._embedding_matrix = None
        self._embedding_cache: Dict[str, Any] = {}
        self._index_dirty = True
        self._write_context: Optional[MemoryWriteContext] = None
        self._last_write_stats: Dict[str, int] = {
            "skipped_recaps": 0,
            "committed_memories": 0,
            "promoted_state_updates": 0,
            "reused_state_identities": 0,
        }
        self._capture_parse_stats: Dict[str, int] = {
            "malformed_windows": 0,
            "salvaged_items": 0,
            "discarded_items": 0,
        }
        print("[SmartMem0] Initialized adaptive evidence memory")

    @staticmethod
    def _snapshot(value: Any) -> Any:
        return copy.deepcopy(value)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        return (
            text
            if len(tokens) <= max_tokens
            else self._tokenizer.decode(tokens[:max_tokens])
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            text = re.sub(r"^json\s*", "", text.strip(), flags=re.I)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            try:
                value = json.loads(match.group(0)) if match else {}
            except json.JSONDecodeError:
                value = {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _json_array_objects(raw: str, key: str) -> Tuple[List[Any], int]:
        """Salvage independent objects from one JSON array without reindexing.

        LLM output can contain one malformed object among many valid memories.
        Returning ``None`` for that object preserves the raw array index so
        causal-link endpoints still refer to the intended surviving memories.
        """
        text = str(raw or "")
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
        if not match:
            return [], 0
        start = match.end()
        items: List[Any] = []
        malformed = 0
        depth = 0
        object_start: Optional[int] = None
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                if depth == 0:
                    object_start = index
                depth += 1
                continue
            if char == "}" and depth:
                depth -= 1
                if depth == 0 and object_start is not None:
                    candidate = text[object_start : index + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        value = None
                        malformed += 1
                    items.append(value if isinstance(value, dict) else None)
                    object_start = None
                continue
            if char == "]" and depth == 0:
                break
        return items, malformed

    @classmethod
    def _parse_capture_json(cls, raw: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Parse a capture response and salvage valid memory/link objects."""
        parsed = cls._parse_json(raw)
        if parsed:
            return parsed, {
                "malformed_window": 0,
                "salvaged_items": 0,
                "discarded_items": 0,
            }
        memories, bad_memories = cls._json_array_objects(raw, "memories")
        links, bad_links = cls._json_array_objects(raw, "causal_links")
        valid_items = sum(item is not None for item in (*memories, *links))
        return {
            "memories": memories,
            "causal_links": links,
        }, {
            "malformed_window": 1,
            "salvaged_items": valid_items,
            "discarded_items": bad_memories + bad_links,
        }

    def _effective_runtime_config(self) -> Dict[str, Any]:
        return {
            "enable_memory_write": self.enable_memory_write,
            "enable_planner": self.enable_planner,
            "enable_unified_controller": self.enable_unified_controller,
            "enable_slot_support_validation": self.enable_slot_support_validation,
            "enable_replan": self.enable_replan,
            "enable_zero_result_recovery": self.enable_zero_result_recovery,
            "enable_planner_repair": self.enable_planner_repair,
            "retrieve_num": self.retrieve_num,
            "max_context_tokens": self.max_context_tokens,
            "write_window_max_turns": self.write_window_max_turns,
            "write_window_max_tokens": self.write_window_max_tokens,
            "write_context_mode": self.write_context_mode,
            "write_prior_belief_limit": self.write_prior_belief_limit,
            "write_provisional_limit": self.write_provisional_limit,
        }

    @staticmethod
    def _parse_date(text: str) -> str:
        value = str(text or "")
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", value)
        if match:
            return match.group(1)
        match = re.search(r"\b(\d{4}-\d{2})\b", value)
        if match:
            return match.group(1)
        months = "January|February|March|April|May|June|July|August|September|October|November|December"
        for pattern, fmt in (
            (rf"\b(\d{{1,2}})\s+({months}),?\s+(\d{{4}})\b", "%d %B %Y"),
            (
                rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
                "%B %d %Y",
            ),
        ):
            match = re.search(pattern, value, re.I)
            if match:
                try:
                    return datetime.strptime(" ".join(match.groups()), fmt).strftime(
                        "%Y-%m-%d"
                    )
                except ValueError:
                    pass
        return ""

    @staticmethod
    def _parse_turns(text: str) -> Tuple[List[Dict[str, Any]], str]:
        first_date = re.search(r"\[(\d{4}-\d{2}-\d{2})(?:[^\]]*)\]", text)
        date_line = re.search(r"(?im)^\s*DATE:\s*([^\n]+)", text)
        document_time = (
            first_date.group(1)
            if first_date
            else CoreMemoryMixin._parse_date(date_line.group(1)) if date_line else ""
        )
        turns: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        uses_said_format = bool(re.search(r"(?im)^\s*CONVERSATION:\s*$", text))

        def flush() -> None:
            nonlocal current
            if not current:
                return
            content = "\n".join(current.pop("lines")).strip()
            if current.pop("said_format"):
                content = re.sub(r'"\s+and shared\s+', " and shared ", content)
                content = re.sub(r'"\s*$', "", content).strip()
            if content:
                turns.append(
                    {
                        "speaker": current["speaker"].lower(),
                        "raw_text": content,
                        "turn_idx": len(turns),
                    }
                )
            current = None

        role_names = {"patient", "doctor", "user", "assistant", "human", "ai"}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if re.match(r"(?i)^<assistant>\s+i have memorized", line):
                flush()
                break
            said = (
                re.match(r'^(.+?)\s+said,\s+"(.*)$', line) if uses_said_format else None
            )
            colon = re.match(r"^([^:\n]{1,80}):\s*(.*)$", line)
            if said:
                flush()
                current = {
                    "speaker": said.group(1).strip(),
                    "lines": [said.group(2)],
                    "said_format": True,
                }
                continue
            if colon:
                speaker = colon.group(1).strip()
                if speaker.upper() in {
                    "DATE",
                    "CURRENT DATE",
                    "CONVERSATION",
                    "ANSWER",
                }:
                    continue
                if speaker.lower() in role_names:
                    if (
                        current
                        and current["speaker"].strip().lower() == speaker.lower()
                    ):
                        current["lines"].append(raw_line.rstrip())
                        continue
                    flush()
                    current = {
                        "speaker": speaker,
                        "lines": [colon.group(2)],
                        "said_format": False,
                    }
                    continue
            if current is not None:
                current["lines"].append(raw_line.rstrip())
        flush()
        return turns, document_time

    @staticmethod
    def _format_turn(turn: Dict[str, Any], label: str = "TURN") -> str:
        return f"[{label} {turn['turn_idx']}] [{turn['speaker']}] {turn['raw_text']}"

    def _split_write_windows(
        self, turns: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Greedily batch complete turns without cutting a long turn in half."""
        windows: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0
        for turn in turns:
            turn_tokens = len(self._tokenizer.encode(self._format_turn(turn)))
            exceeds_limit = current and (
                len(current) >= self.write_window_max_turns
                or current_tokens + turn_tokens > self.write_window_max_tokens
            )
            if exceeds_limit:
                windows.append(current)
                current, current_tokens = [], 0
            current.append(turn)
            current_tokens += turn_tokens
        if current:
            windows.append(current)
        return windows

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        raw = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.%+-]*", str(text or "").lower())
        output = []
        for token in raw:
            if token in STOPWORDS or len(token) <= 1:
                continue
            terms = [token]
            if "-" in token or "_" in token:
                terms.extend(
                    part
                    for part in re.split(r"[-_]", token)
                    if len(part) > 1 and part not in STOPWORDS
                )
            for term in terms:
                output.append(term)
                if not term.isalpha() or len(term) <= 4:
                    continue
                if term.endswith("ing") and len(term) > 6:
                    output.append(term[:-3])
                elif term.endswith("ied") and len(term) > 5:
                    output.append(term[:-3] + "y")
                elif term.endswith("ed") and len(term) > 5:
                    output.append(term[:-2])
                elif term.endswith("ies") and len(term) > 5:
                    output.append(term[:-3] + "y")
                elif term.endswith("s") and term not in {"diabetes", "status"}:
                    output.append(term[:-1])
        return output

    @staticmethod
    def _canonical_state_key(value: str) -> str:
        """Remove volatile wording while preserving the domain attribute."""
        parts = re.findall(r"[a-z0-9]+", str(value or "").lower())
        aliases = {
            "waking": "morning",
            "wake": "morning",
            "wakeup": "morning",
        }
        parts = [aliases.get(part, part) for part in parts]
        parts = [
            part
            for part in parts
            if part not in STATE_KEY_NOISE
            and part not in STOPWORDS
            and not part.isdigit()
        ]
        if "glucose" in parts:
            parts = [part for part in parts if part != "blood"]
        tail_noise = {
            "level",
            "levels",
            "value",
            "values",
            "reading",
            "readings",
            "result",
        }
        while len(parts) > 1 and parts[-1] in tail_noise:
            parts.pop()
        normalized = []
        for part in parts:
            if (
                part.endswith("s")
                and len(part) > 4
                and not part.endswith(("ss", "us", "is", "ness"))
                and part not in {"diabetes", "status"}
            ):
                part = part[:-1]
            if not normalized or normalized[-1] != part:
                normalized.append(part)
        return "_".join(normalized[:6])

    @staticmethod
    def _canonical_identifier(value: Any, default: str = "") -> str:
        parts = re.findall(r"[a-z0-9]+", str(value or "").lower())
        return "_".join(parts[:8]) or default

    @staticmethod
    def _canonical_scope(value: Any) -> str:
        """Keep scope semantic; entity-specific ownership belongs in anchor."""
        raw = str(value or "").strip().lower()
        # Accept legacy ``scope:entity`` data without allowing the entity to
        # silently become part of state identity.
        raw = re.split(r"[:|/]", raw, maxsplit=1)[0]
        scope = CoreMemoryMixin._canonical_identifier(raw, "general")
        return SCOPE_ALIASES.get(scope, scope)

    @classmethod
    def _state_identity(cls, memory: Dict[str, Any]) -> str:
        state_key = cls._canonical_state_key(memory.get("state_key") or "")
        if not state_key:
            return ""
        subject = cls._canonical_identifier(memory.get("subject"), "patient")
        scope = cls._canonical_scope(memory.get("scope"))
        anchor = cls._canonical_identifier(memory.get("object_anchor"))
        return "|".join((subject, scope, state_key, anchor))

    @classmethod
    def _state_lineage_identity(cls, memory: Dict[str, Any]) -> str:
        """Return a scope-tolerant key used only to reconcile write-time drift.

        Durable state identity still includes scope.  This weaker key is never a
        state head and never merges values on its own; it only lets a new card
        inherit the already committed scope when subject, attribute and object
        owner are identical.
        """
        state_key = cls._canonical_state_key(memory.get("state_key") or "")
        if not state_key:
            return ""
        subject = cls._canonical_identifier(memory.get("subject"), "patient")
        anchor = cls._canonical_identifier(memory.get("object_anchor"))
        return "|".join((subject, state_key, anchor))

    @staticmethod
    def _memory_value(memory: Dict[str, Any]) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(memory.get("value") or memory.get("verbatim_value") or "").strip(),
        )

    @classmethod
    def _normalised_value(cls, memory: Dict[str, Any]) -> str:
        return cls._canonical_identifier(cls._memory_value(memory))

    @classmethod
    def _state_value_signature(cls, memory: Dict[str, Any]) -> str:
        """Collapse same-measurement paraphrases without hiding numeric changes."""
        value = cls._memory_value(memory).lower()
        numbers = re.findall(r"\d+(?:\.\d+)?", value)
        if numbers:
            return "numbers:" + "|".join(numbers)
        return cls._normalised_value(memory)

    @staticmethod
    def _extract_date_constraints(question: str) -> Tuple[str, ...]:
        text = str(question or "")
        found: List[Tuple[int, str]] = []
        occupied: List[Tuple[int, int]] = []
        for match in re.finditer(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b", text):
            value = match.group(0)
            found.append((match.start(), value))
            occupied.append(match.span())

        month_names = "|".join(MONTH_NUMBERS)
        pattern = re.compile(
            rf"\b({month_names})\s+(?:(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?|(\d{{4}}))\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            month = MONTH_NUMBERS[match.group(1).lower()]
            day, day_year, month_year = match.group(2), match.group(3), match.group(4)
            if day:
                value = (
                    f"{int(day_year):04d}-{month:02d}-{int(day):02d}"
                    if day_year
                    else f"*-{month:02d}-{int(day):02d}"
                )
            else:
                value = f"{int(month_year):04d}-{month:02d}"
            found.append((match.start(), value))

        unique = []
        for _, value in sorted(found):
            if value not in unique:
                unique.append(value)
        day_level = [
            value for value in unique if value.startswith("*-") or value.count("-") == 2
        ]
        return tuple(day_level or unique)

    @staticmethod
    def _query_speaker_role(question: str) -> str:
        """Return a role only for explicit source attribution, never a vocative."""
        text = str(question or "").strip().lower()
        text = re.sub(
            r"^\s*(?:doctor|physician|clinician|nurse|assistant)\s*[,;:]\s*",
            "",
            text,
        )
        action = (
            r"say|said|says|report|reported|reports|mention|mentioned|mentions|"
            r"state|stated|states|describe|described|describes|share|shared|shares|"
            r"confirm|confirmed|confirms|ask|asked|asks|advise|advised|advises|"
            r"recommend|recommended|recommends|diagnose|diagnosed|diagnoses|"
            r"prescribe|prescribed|prescribes|explain|explained|explains|"
            r"judge|judged|judges|assess|assessed|assesses|conclude|concluded|concludes"
        )

        def attributed(role_pattern: str) -> bool:
            return bool(
                re.search(
                    rf"\b(?:according to|reported by|stated by)\s+(?:the\s+)?(?:{role_pattern})\b",
                    text,
                )
                or re.search(
                    rf"\b(?:{role_pattern})\b(?:\W+\w+){{0,5}}\W+(?:{action})\b", text
                )
            )

        patient_hit = attributed(r"patient|user")
        doctor_hit = attributed(r"doctor|physician|clinician|nurse")
        if patient_hit != doctor_hit:
            return "patient" if patient_hit else "doctor"
        return ""

    @staticmethod
    def _question_stem(question: str) -> str:
        """Exclude answer choices before deriving retrieval constraints."""
        text = str(question or "")
        text = re.split(
            r"(?im)\n\s*(?:answer\s+choices?|options?|choices?)\s*:\s*",
            text,
            maxsplit=1,
        )[0]
        return re.split(
            r"(?im)\n\s*(?:\(?[a-h]\)|[a-h][.)])\s+",
            text,
            maxsplit=1,
        )[0]

    @staticmethod
    def _question_options(question: str) -> Dict[str, str]:
        """Parse visibly enumerated answer options without inferring query intent."""
        matches = list(
            re.finditer(
                r"(?im)(?:^|\n)\s*(?:\(([a-h])\)|([a-h])[.)])\s+",
                str(question or ""),
            )
        )
        options: Dict[str, str] = {}
        for index, match in enumerate(matches):
            label = str(match.group(1) or match.group(2) or "").upper()
            end = matches[index + 1].start() if index + 1 < len(matches) else None
            value = str(question or "")[match.end() : end].strip()
            if label and value:
                options[label] = value
        return options

    @staticmethod
    def _gate_skip_surface(question: str) -> str:
        """Return only locked surface patterns that bypass the direct gate."""
        text = str(question or "")
        if re.search(r"(?im)(?:^|\n)\s*(?:\(?[a-h]\)|[a-h][.)])\s+\S", text):
            return "option_enumeration"
        stem = CoreMemoryMixin._question_stem(text)
        if re.search(r"\b(?:earliest|latest)\b", stem.lower()):
            return "temporal_extremum_enum"
        return ""

    def _query_frame(self, question: str) -> QueryFrame:
        stem = self._question_stem(question)
        query_terms = set(self._tokenize(stem))
        explicit_entities = []
        role_entities = {
            "patient",
            "user",
            "person",
            "doctor",
            "physician",
            "clinician",
            "nurse",
            "assistant",
            "human",
            "ai",
        }
        for memory in self._memories:
            anchor = str(memory.get("object_anchor") or "").strip().lower()
            values = [
                *memory.get("entities", []),
                *memory.get("scope_entities", []),
                anchor,
            ]
            for value in values:
                canonical = str(value or "").strip().lower()
                terms = {
                    term
                    for term in re.findall(r"[a-z0-9]+", canonical)
                    if len(term) > 1 and term not in STOPWORDS
                }
                if (
                    canonical
                    and canonical not in role_entities
                    and terms
                    and terms.issubset(query_terms)
                    and canonical not in explicit_entities
                ):
                    explicit_entities.append(canonical)
        return QueryFrame(
            dates=self._extract_date_constraints(stem),
            speaker_role=self._query_speaker_role(stem),
            entities=tuple(explicit_entities[:4]),
            # Extractor-generated anchors are useful retrieval telemetry, but
            # they are not reliable hard filters. The same concept may be an
            # anchor on one memory and plain claim text on another; filtering
            # by it caused catastrophic false negatives for common concepts
            # such as blood sugar. Dates and explicit speakers remain hard.
            hard_entities=(),
        )

    @staticmethod
    def _unwrap_question(question: str) -> str:
        text = str(question or "").strip()
        match = re.search(
            r"(?is)\bquestion\s*:\s*(.+?)(?=\n\s*(?:\[answer requirements?\]|"
            r"answer format(?:\s*\(critical\))?|answer\s*:|$))",
            text,
        )
        if match:
            return match.group(1).strip()
        body = re.split(
            r"(?is)\n\s*(?:\[answer requirements?\]|answer format(?:\s*\(critical\))?|"
            r"format requirements|critical instructions|answer\s*:)",
            text,
            maxsplit=1,
        )[0]
        marker = re.search(r"(?is)answer the following question\s*:\s*", body)
        return body[marker.end() :].strip() if marker else text

    @staticmethod
    def _normalise_memory(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        claim = re.sub(
            r"\s+", " ", str(item.get("claim") or item.get("content") or "").strip()
        )
        if not claim:
            return None
        kind = str(item.get("kind") or item.get("fact_type") or "FACT").upper()
        kind = kind if kind in VALID_KINDS else "FACT"
        stance = str(item.get("stance") or item.get("polarity") or "AFFIRM").upper()
        stance = {"POSITIVE": "AFFIRM", "NEGATIVE": "DENY"}.get(stance, stance)
        stance = stance if stance in {"AFFIRM", "DENY", "UNCERTAIN"} else "AFFIRM"
        assertion_mode = str(item.get("assertion_mode") or "DIRECT").upper()
        assertion_mode = (
            assertion_mode if assertion_mode in VALID_ASSERTION_MODES else "DIRECT"
        )
        origin_memory_id = str(item.get("origin_memory_id") or "").strip()
        if assertion_mode != "RECAP":
            origin_memory_id = ""
        elif not origin_memory_id:
            # A recap without a linked origin is not auditable. Keep the claim
            # usable, but do not let it receive recap-specific temporal rules.
            assertion_mode = "DIRECT"
        entities = item.get("entities") or []
        if isinstance(entities, str):
            entities = [entities]
        scope_entities = item.get("scope_entities") or []
        if isinstance(scope_entities, str):
            scope_entities = [scope_entities]
        planning_tags = item.get("planning_tags") or []
        if isinstance(planning_tags, str):
            planning_tags = [planning_tags]
        planning_tags = list(
            dict.fromkeys(
                str(tag).upper()
                for tag in planning_tags
                if str(tag).upper() in VALID_PLANNING_TAGS
            )
        )
        try:
            decision_salience = max(
                0.0, min(1.0, float(item.get("decision_salience", 0.0) or 0.0))
            )
        except (TypeError, ValueError):
            decision_salience = 0.0
        subject = CoreMemoryMixin._canonical_identifier(
            item.get("subject") or "patient",
            "patient",
        )
        raw_scope = str(item.get("scope") or "").strip()
        scope_parts = re.split(r"[:|/]", raw_scope, maxsplit=1)
        scope = CoreMemoryMixin._canonical_scope(scope_parts[0])
        object_anchor = CoreMemoryMixin._canonical_identifier(item.get("object_anchor"))
        # Legacy extractors sometimes encoded ownership as
        # ``scope:entity``. Canonicalize that form into the schema's explicit
        # object_anchor so old snapshots do not create a second state identity.
        if not object_anchor and len(scope_parts) == 2:
            object_anchor = CoreMemoryMixin._canonical_identifier(scope_parts[1])
        if object_anchor in GENERIC_OBJECT_ANCHORS:
            object_anchor = ""
        value = re.sub(
            r"\s+",
            " ",
            str(item.get("value") or item.get("verbatim_value") or "").strip(),
        )[:160]
        turns = item.get("source_turns") or []
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))
        except (TypeError, ValueError):
            confidence = 0.8
        # A state key is an identity component, not a topical label. Keep it
        # only on memories that can participate in versioned state heads.
        state_key = (
            CoreMemoryMixin._canonical_state_key(item.get("state_key") or "")
            if kind in STATE_LIKE_KINDS
            else ""
        )
        if kind in STATE_LIKE_KINDS and not state_key:
            # A state-like card without an identity cannot safely participate in
            # head resolution. Preserve its claim as a searchable FACT instead
            # of creating a malformed state that can poison the whole profile.
            kind = "FACT"
        # A topical label is not a stable state identity. Keep the claim in
        # the searchable ledger, but prevent unrelated observations from being
        # forced into one state head. A specific object owner makes even a
        # broad key usable (for example symptom|metformin is intentional).
        if (
            kind in STATE_LIKE_KINDS
            and state_key in GENERIC_STATE_KEYS
            and not object_anchor
        ):
            kind = "FACT"
            state_key = ""
        return {
            "claim": claim,
            "kind": kind,
            "entities": [
                str(value).strip() for value in entities if str(value).strip()
            ][:8],
            "subject": subject,
            "scope": scope,
            "object_anchor": object_anchor,
            "scope_entities": [
                str(value).strip() for value in scope_entities if str(value).strip()
            ][:8],
            "state_key": state_key,
            "value": value,
            "verbatim_value": re.sub(
                r"\s+", " ", str(item.get("verbatim_value") or "").strip()
            )[:160],
            "stance": stance,
            "event_time": str(item.get("event_time") or "UNKNOWN"),
            "time_expression": str(item.get("time_expression") or "").strip(),
            "assertion_mode": assertion_mode,
            "origin_memory_id": origin_memory_id,
            "planning_tags": planning_tags,
            "decision_salience": decision_salience,
            "source_turns": [
                int(v) for v in turns if isinstance(v, int) or str(v).isdigit()
            ][:4],
            "confidence": confidence,
        }

    def _resolve_event_time(self, memory: Dict[str, Any], document_time: str) -> str:
        # Source-grounded dates outrank a conflicting date copied into the
        # extractor's event_time field.  Parsing all fields as one string made
        # the first value (event_time) win even when the atomic claim itself
        # explicitly named a different focal date.
        source_date = self._parse_date(
            f"{memory.get('time_expression', '')} "
            f"{memory.get('claim', '')} {memory.get('verbatim_value', '')}"
        )
        if source_date:
            return source_date
        declared_date = self._parse_date(memory.get("event_time", ""))
        if declared_date:
            return declared_date
        document_date = self._parse_date(document_time)
        expression = memory.get("time_expression", "").lower()
        if not document_date:
            return "UNKNOWN"
        try:
            anchor = datetime.strptime(document_date[:10], "%Y-%m-%d")
        except ValueError:
            return "UNKNOWN"
        ago = re.search(r"\b(\d+)\s+days?\s+ago\b", expression)
        if ago:
            return (anchor - timedelta(days=int(ago.group(1)))).strftime("%Y-%m-%d")
        weeks_ago = re.search(r"\b(\d+)\s+weeks?\s+ago\b", expression)
        if weeks_ago:
            return (anchor - timedelta(weeks=int(weeks_ago.group(1)))).strftime(
                "%Y-%m-%d"
            )
        months_ago = re.search(r"\b(\d+)\s+months?\s+ago\b", expression)
        if months_ago:
            # Month-level statements should remain coarse instead of inventing a day.
            month_index = anchor.year * 12 + anchor.month - 1 - int(months_ago.group(1))
            return f"{month_index // 12:04d}-{month_index % 12 + 1:02d}"
        if re.search(r"\byesterday\b|\blast night\b", expression):
            return (anchor - timedelta(days=1)).strftime("%Y-%m-%d")
        if re.search(
            r"\btoday\b|\bcurrently\b|\bright now\b|\bthis morning\b", expression
        ):
            return anchor.strftime("%Y-%m-%d")
        if re.search(r"\btomorrow\b", expression):
            return (anchor + timedelta(days=1)).strftime("%Y-%m-%d")
        return "UNKNOWN"

    @staticmethod
    def _stage_evidence(
        turns: List[Dict[str, Any]],
        document_time: str,
        session_idx: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
        staged, mapping = [], {}
        for turn in turns:
            evidence_id = f"ev_{session_idx}_{turn['turn_idx']}"
            staged.append(
                {
                    "id": evidence_id,
                    "session_idx": session_idx,
                    "turn_idx": turn["turn_idx"],
                    "speaker": turn["speaker"],
                    "raw_text": turn["raw_text"],
                    "document_time": document_time,
                }
            )
            mapping[turn["turn_idx"]] = evidence_id
        return staged, mapping

    def _memory_text(self, memory: Dict[str, Any]) -> str:
        # Index answer-bearing content and discriminative anchors. High-frequency
        # bookkeeping fields such as patient/AFFIRM/STATE/source speaker create
        # artificial BM25 and dense overlap and are enforced separately by the
        # query frame and typed-slot validators.
        role_entities = {
            "assistant",
            "clinician",
            "doctor",
            "human",
            "patient",
            "person",
            "physician",
            "user",
        }
        entities = [
            str(entity)
            for entity in memory.get("entities", [])
            if str(entity).strip().lower() not in role_entities
        ]
        return " ".join(
            filter(
                None,
                [
                    memory.get("claim", ""),
                    self._memory_value(memory),
                    memory.get("scope", ""),
                    memory.get("object_anchor", ""),
                    " ".join(memory.get("scope_entities", [])),
                    " ".join(entities),
                    memory.get("state_key", ""),
                    memory.get("event_time", ""),
                    memory.get("document_time", ""),
                    memory.get("origin_document_time", ""),
                ],
            )
        )

    def _refresh_index(self) -> None:
        if not self._index_dirty:
            return
        texts = [self._memory_text(memory) for memory in self._memories]
        self._bm25 = (
            BM25Okapi([self._tokenize(text) for text in texts]) if texts else None
        )
        if texts:
            missing = [
                (m["id"], text)
                for m, text in zip(self._memories, texts)
                if m["id"] not in self._embedding_cache
            ]
            if missing:
                vectors = self._embedder.encode(
                    [text for _, text in missing], show_progress_bar=False
                )
                for (memory_id, _), vector in zip(missing, vectors):
                    self._embedding_cache[memory_id] = vector
            self._embedding_matrix = np.asarray(
                [self._embedding_cache[m["id"]] for m in self._memories]
            )
        else:
            self._embedding_matrix = None
        self._index_dirty = False

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._memories:
            return []
        self._refresh_index()
        allowed = set(candidate_ids) if candidate_ids is not None else None
        indices = [
            i
            for i, m in enumerate(self._memories)
            if allowed is None or m["id"] in allowed
        ]
        if not indices:
            return []
        bm25_scores = (
            self._bm25.get_scores(self._tokenize(query))
            if self._bm25
            else [0.0] * len(self._memories)
        )
        query_vector = self._embedder.encode([query], show_progress_bar=False)
        dense_scores = cosine_similarity(query_vector, self._embedding_matrix)[0]
        channel_limit = min(len(indices), max(top_k * 3, top_k))
        bm25_rank = sorted(
            (i for i in indices if float(bm25_scores[i]) > 0.0),
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )[:channel_limit]
        dense_rank = sorted(
            indices, key=lambda i: float(dense_scores[i]), reverse=True
        )[:channel_limit]
        rank_maps = [
            ({i: rank + 1 for rank, i in enumerate(bm25_rank)}, 1.0),
            ({i: rank + 1 for rank, i in enumerate(dense_rank)}, 1.0),
        ]
        bm25_positions, dense_positions = rank_maps[0][0], rank_maps[1][0]
        query_terms = set(self._tokenize(query))
        scored = []
        for index in indices:
            score = sum(
                w / (self.RRF_K + ranks[index])
                for ranks, w in rank_maps
                if index in ranks
            )
            if score <= 0:
                continue
            memory = self._snapshot(self._memories[index])
            memory.update(
                {
                    "_score": score,
                    "_bm25_score": float(bm25_scores[index]),
                    "_dense_score": float(dense_scores[index]),
                    "_bm25_rank": bm25_positions.get(index),
                    "_dense_rank": dense_positions.get(index),
                    "_overlap": len(
                        query_terms & set(self._tokenize(self._memory_text(memory)))
                    ),
                    "_status": self._belief_status.get(memory["id"], "active"),
                }
            )
            scored.append(memory)
        scored.sort(
            key=lambda m: (m["_score"], m["_overlap"], m["_dense_score"], m["id"]),
            reverse=True,
        )
        return scored[:top_k]
