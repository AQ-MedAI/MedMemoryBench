"""Shared schemas and deterministic constants for SmartMem0."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "she",
        "the",
        "their",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "with",
        "about",
        "does",
        "did",
        "do",
        "that",
        "this",
        "these",
        "those",
        "my",
        "me",
        "i",
        "you",
        "your",
        "please",
        "can",
        "would",
        "could",
        "tell",
        "said",
        "say",
        "mentioned",
    }
)

VALID_KINDS = {"FACT", "EVENT", "STATE", "PREFERENCE", "PLAN"}

# Bump this whenever capture, normalization, identity, or consolidation changes
# can alter the durable ledger. It is included in the snapshot fingerprint.
MEMORY_WRITE_SCHEMA_VERSION = 6

# These are topical labels, not versionable attributes. If a capture model emits
# one without an object owner, treating every later symptom/emotion as a new
# version of the same state makes unrelated claims supersede one another.
GENERIC_STATE_KEYS = frozenset(
    {
        "state",
        "status",
        "symptom",
        "symptoms",
        "condition",
        "clinical",
        "clinical_profile",
        "emotional",
        "emotion",
        "general",
        "current",
        "recent",
        "update",
        "issue",
        "problem",
    }
)

# Values sometimes emitted in object_anchor are really the subject or a broad
# body-level scope. They do not identify an owned object and must not rescue a
# generic state key into versioned state tracking.
GENERIC_OBJECT_ANCHORS = frozenset(
    {
        "patient",
        "user",
        "person",
        "doctor",
        "physician",
        "clinician",
        "body",
        "general",
        "health",
    }
)

# Only these kinds participate in versioned state-head resolution. FACT and
# EVENT nodes may mention a state-like topic without being state versions.
STATE_LIKE_KINDS = frozenset({"STATE", "PREFERENCE", "PLAN"})

VALID_ASSERTION_MODES = {"DIRECT", "RECAP", "INFERRED"}

VALID_RELATIONS = {
    "SUPPORT",
    "REFINE",
    "SUPERSEDE",
    "CONFLICT",
    "RELATED",
    "CAUSES",
}

VALID_OPERATIONS = {
    "SEMANTIC_SEARCH",
    "LOCATE_ANCHOR",
    "TEMPORAL_FILTER",
    "RESOLVE_STATE",
    "FOLLOW_CAUSES",
    "VERIFY_EVIDENCE",
}

# SEMANTIC_SEARCH remains the only general recall primitive.  A strategy tells
# the deterministic executor how to diversify that operation's bounded output;
# it never authorizes retrieval outside the operation boundary.
VALID_SEARCH_STRATEGIES = {
    "FOCAL",
    "TRAJECTORY",
    "DECISION_BUNDLE",
    "SHARED_OPTIONS",
}

VALID_SLOT_TYPES = {
    "DIRECT",
    "CURRENT_STATE",
    "TEMPORAL",
    "TRANSITION",
    "CAUSE_PATH",
    "COMPARISON",
}

VALID_QUERY_MODES = {
    "DIRECT",
    "STATE",
    "TEMPORAL",
    "DECISION",
    "MULTI_OPTION",
    "CAUSAL",
    "COMPARISON",
}

VALID_REASONING_TYPES = {
    "NONE",
    "SYNTHESIS",
    "DECISION",
    "COMPARISON",
    "CAUSAL",
}

# Scope aliases are deliberately small and domain-agnostic. They normalize
# grammatical variants only; entity ownership remains in object_anchor.
SCOPE_ALIASES = {
    "drugs": "medication",
    "drug": "medication",
    "medications": "medication",
    "medicines": "medication",
    "medicine": "medication",
    "symptoms": "symptom",
    "tests": "test",
    "labs": "lab",
    "laboratory": "lab",
    "vitals": "vital",
    "vital_signs": "vital",
    "treatments": "treatment",
    "preferences": "preference",
    "plans": "plan",
}

VALID_EVIDENCE_ROLES = {
    "ANSWER",
    "FOCAL_STATE",
    "LONGITUDINAL_CONTEXT",
    "ACTION_RULE",
    "CONSTRAINT",
    "ALTERNATIVE",
    "TEMPORAL",
    "CAUSE",
    "COMPARAND",
    "OPTION",
}

VALID_PLANNING_TAGS = {
    "IDENTITY",
    "STATE",
    "EXPOSURE",
    "RESPONSE",
    "TRAJECTORY",
    "RISK",
    "CONSTRAINT",
    "PREFERENCE",
    "RESOURCE",
    "PLAN",
}

VALID_TEMPORAL_AXES = {
    "event_time",
    "document_time",
    "origin_document_time",
    "effective_event_time",
}

VALID_TEMPORAL_RELATIONS = {
    "EXACT",
    "EARLIEST",
    "LATEST",
    "BEFORE",
    "AFTER",
    "BETWEEN",
}

CLINICAL_SCOPES = frozenset(
    {
        "allergy",
        "diagnosis",
        "lab",
        "medication",
        "procedure",
        "surgery",
        "symptom",
        "test",
        "treatment",
        "vital",
    }
)

# Planner chooses a coarse workload class; the executor owns the exact limits.
# This avoids free-form numeric budgets drifting toward the prompt example.
RETRIEVAL_BUDGETS = {
    "SMALL": {"max_memories": 3, "max_operations": 1},
    "MEDIUM": {"max_memories": 5, "max_operations": 2},
    "LARGE": {"max_memories": 8, "max_operations": 4},
}

SLOT_REQUIRED_FIELDS = {
    "DIRECT": {"value"},
    "CURRENT_STATE": {"state_identity", "value"},
    "TRANSITION": {"old_value", "new_value"},
    "CAUSE_PATH": {"path"},
    "COMPARISON": {"left", "right"},
}

MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

STATE_KEY_NOISE = frozenset(
    {
        "patient",
        "doctor",
        "clinician",
        "physician",
        "assistant",
        "user",
        "person",
        "current",
        "currently",
        "recent",
        "recently",
        "latest",
        "present",
        "ongoing",
        "status",
        "state",
        "update",
        "updated",
        "information",
        "details",
        "reported",
        "assessment",
        "experience",
        "pattern",
        "trajectory",
    }
)


@dataclass(frozen=True)
class QueryFrame:
    """Deterministic query constraints applied before semantic ranking."""

    dates: Tuple[str, ...] = ()
    speaker_role: str = ""
    entities: Tuple[str, ...] = ()
    hard_entities: Tuple[str, ...] = ()


@dataclass
class MemoryWriteContext:
    """Ephemeral interpretation context used only while ingesting one session."""

    local_turn_context: List[Dict[str, Any]] = field(default_factory=list)
    core_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    relevant_prior_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    provisional_memories: List[Dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.local_turn_context.clear()
        self.core_beliefs.clear()
        self.relevant_prior_beliefs.clear()
        self.provisional_memories.clear()
