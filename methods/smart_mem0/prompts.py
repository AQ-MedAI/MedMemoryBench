"""LLM contracts used by SmartMem0 write and read paths."""

MEMORY_WRITE_PROMPT = """You write durable memory for a long-lived conversational agent.
Extract at most 36 compact, faithful memories from FOCAL TURNS. Preserve names,
numbers, units, negation, decisions, preferences, state changes, and explicit
times inside each claim. Merge details describing the same event or state.
Never invent information. Point each memory to its smallest supporting focal turn set.

DOCUMENT TIME: {document_time}

COMMITTED PRIOR BELIEFS (interpretation context only; never use as new evidence):
{prior_beliefs}

PROVISIONAL MEMORIES FROM EARLIER WINDOWS
(continuity and deduplication context only; never use as new evidence):
{provisional_memories}

LOCAL CONTEXT TURN (pronoun/topic context only; never cite as source):
{local_context}

FOCAL TURNS (the only allowed source evidence):
{focal_turns}

Return JSON only:
{{
  "memories": [{{
    "claim": "self-contained atomic claim",
    "kind": "FACT|EVENT|STATE|PREFERENCE|PLAN",
    "entities": ["canonical entity"],
    "subject": "canonical owner of the memory, usually a person",
    "scope": "stable semantic scope such as medication, symptom, test, or general",
    "state_key": "short subject-free stable attribute name, or empty",
    "object_anchor": "canonical object owning an object-specific state, or empty",
    "scope_entities": ["retrieval metadata only"],
    "value": "compact normalized atomic value, or empty",
    "verbatim_value": "exact short name/value/unit phrase from source, or empty",
    "stance": "AFFIRM|DENY|UNCERTAIN",
    "event_time": "YYYY-MM-DD, YYYY-MM, or UNKNOWN",
    "time_expression": "source time phrase, or empty",
    "assertion_mode": "DIRECT|RECAP|INFERRED",
    "origin_memory_id": "matching committed prior belief id for RECAP, or empty",
    "planning_tags": [],
    "decision_salience": 0.0,
    "source_turns": [0],
    "confidence": 0.0
  }}],
  "causal_links": [
    {{"cause_index": 0, "effect_index": 1, "source_turns": [0], "confidence": 0.0}}
  ]
}}

Rules:
- source_turns may contain only TURN ids shown under FOCAL TURNS.
- Never extract a memory supported only by prior beliefs, provisional memories,
  or the local context turn.
- Do not repeat an earlier memory unless a focal turn restates, refines, changes,
  contradicts, or causally connects it.
- DIRECT means a focal speaker directly states, confirms, or updates the claim.
- RECAP means a focal turn only recalls or paraphrases an already committed belief
  (for example, "you mentioned", "I remember", or "previously"). For RECAP,
  set origin_memory_id to the matching committed belief id. A sentence that recalls
  an old value and then states a new value must extract the new value as DIRECT.
- If a recap explicitly preserves an observed date, frequency, numeric range,
  measurement, or before/after change, emit one compact RECAP memory for that
  answer-bearing detail even when an equivalent origin already exists. It remains
  linked to the origin and is not a new current state; it is retained so a later
  document-time query can locate the actual record that stated the detail.
- INFERRED is allowed only for an unavoidable implication directly supported by
  focal turns; prefer DIRECT and use lower confidence for INFERRED.
- For each focal speaker, preserve unique named entities, exact results, explicit
  changes, durable concerns, preferences, decisions, and commitments.
- Preserve participant-specific conditional safety instructions as one atomic PLAN
  with both sides in value: "when/if <trigger>, <action>". Do not split away or omit
  the trigger, including named symptoms, medicines, risks, tests, and stop/seek-help
  conditions. A recalled clinician instruction is still durable evidence when the
  focal speaker explicitly reports it; mark it RECAP when an origin exists.
- Each memory must describe one focal event or one versioned state. When a turn
  mentions an episode on one date and recovery, follow-up, or recurrence on another
  date, emit separate memories with separate event_time values. Never assign the
  recovery date to the original episode or combine two dated observations into one
  value. document_time is supplied by the system and is not a substitute for an
  unstated event_time.
- Preserve the scope of restrictions exactly. When a focal turn states both a
  named-item fact and a broader class-level restriction (for example, an allergy to
  one medicine and avoidance of its whole drug class), retain both answer-bearing
  scopes as separate atomic memories or in one value that explicitly names both.
  Never silently narrow a class-wide instruction to the single example medicine.
- Preserve meaning-changing response and trajectory qualifiers such as temporarily,
  then, returns, again, no longer, still, only when, before/after, increased/decreased,
  and approximate ranges. Never collapse "temporarily relieved, then returned" into
  either "not relieved" or "resolved". When a compact value cannot retain the full
  pattern, keep that pattern in value and copy its shortest exact phrase into
  verbatim_value.
- Use a short subject-free state_key of 2-5 stable attribute tokens. Never put
  patient/doctor, current/recent/latest, dates, values, status, or update in it.
  Emit state_key only for a versionable STATE, PREFERENCE, or PLAN. For FACT and
  EVENT, state_key must be empty; a topical fact is not a state head.
  Do not use generic topical keys such as symptom, emotional, condition,
  clinical_profile, or state unless an object_anchor makes the owned attribute
  unambiguous. Independent symptoms are separate FACT/EVENT cards, not versions
  that supersede one another.
  Reuse the exact state_key from prior beliefs when the attribute is the same.
- subject, scope, state_key, and object_anchor identify a state. value, time,
  and scope_entities never identify a state.
- object_anchor is empty for a scope-wide state such as current medication. Use
  one canonical object only for object-specific states such as a metformin dose.
- scope must remain semantic and must never contain an entity or colon path.
- value is the normalized answer-bearing value. verbatim_value preserves the
  exact source phrase when exact spelling, number, unit, or name matters.
- STATE can replace an older value for the same state_key.
- A PLAN receives a state_key only when the participant's active plan itself can
  be updated or superseded. A one-off recommendation remains an ordinary claim.
- A stated start, stop, resume, replacement, or dose/value change is a DIRECT
  state update. Reuse the prior subject, scope, state_key, and object_anchor so
  the update remains the same state identity with a new value.
- EVENT happened; FACT is a stable assertion.
- Use UNKNOWN when event time is not stated. Document time is stored separately.
- verbatim_value must be copied from a focal turn, not normalized or expanded.
- planning_tags are routing metadata, not new facts. Use only this exact vocabulary
  and choose every role explicitly supported by the focal turn: IDENTITY (patient
  identity/diagnosis), STATE (current durable state), EXPOSURE (treatment or other
  intervention actually used), RESPONSE (observed response to an exposure),
  TRAJECTORY (change/persistence/recurrence across time), RISK (warning sign or
  vulnerability), CONSTRAINT (contraindication/resource/safety limitation),
  PREFERENCE (stable preference), RESOURCE (feasibility/resource), and PLAN
  (accepted participant-specific action). TRAJECTORY requires change/persistence;
  RESPONSE requires an observed response to an exposure; RISK/CONSTRAINT requires
  a durable warning, contraindication, vulnerability, or decision-changing
  limitation. Do not use a tag merely because it is relevant to the topic.
- decision_salience is 0.0-1.0. Use high values only for durable information likely
  to change future decisions: identity/diagnosis, treatment exposure and response,
  objective progression, severe prior events, contraindications, stable preferences,
  resource constraints, and accepted plans. Generic advice and transient small talk
  must remain low.
- A causal link must be explicit and directed cause -> effect. source_turns must
  identify the focal turn(s) that explicitly state the causal relation. Temporal
  order or topical association is never enough.
- Do not store generic textbook explanations or hypothetical advice. Do preserve
  participant-specific clinical reasoning: warning signs, interpreted trends,
  suspected mechanisms, and reasons for a diagnosis, test, treatment decision,
  or accepted plan. When a focal turn explicitly says participant fact A causes
  participant outcome B, extract both endpoint memories and one causal_link.
- Do not create aliases, likely questions, importance scores, or domain categories.
"""

CONSOLIDATION_PROMPT = """Compare NEW MEMORIES with nearby OLD MEMORIES.
Return only semantically justified relations. Direction matters.

NEW MEMORIES (including assertion_mode and origin_memory_id):
{new_memories}

OLD MEMORIES:
{old_memories}

Return JSON only:
{{"relations": [{{
  "source_id": "new_0",
    "target_id": "m_3",
    "type": "SUPPORT|REFINE|SUPERSEDE|CONFLICT|RELATED|CAUSES",
    "provenance_evidence_ids": ["ev_1_0"],
    "confidence": 0.0
}}]}}

Semantics:
- SUPPORT: same compatible belief.
- REFINE: source is a more precise replacement for target.
- SUPERSEDE: source is a newer state replacing target.
- CONFLICT: source and target cannot both hold for the same time/state.
- RELATED: source has a clear, useful semantic connection to target, but none of
  SUPPORT, REFINE, SUPERSEDE, CONFLICT, or CAUSES applies. RELATED is navigation
  metadata only: it never changes state validity, identity, time, or causality.
- CAUSES: source explicitly causes target. Never reverse it.
- CAUSES requires non-empty evidence on both endpoints and non-empty
  provenance_evidence_ids chosen from endpoint evidence. Temporal order is not causality.
- A RECAP of the same value is SUPPORT, never SUPERSEDE or REFINE.
- State-key wording is metadata, not proof that two memories describe different
  states. When subject, semantic scope, object anchor, and durable attribute meaning
  match but the state keys are synonymous, relate them as SUPPORT/REFINE/SUPERSEDE
  according to their values and times so the older canonical identity can be reused.
- SUPERSEDE requires an explicit changed value/state, not merely a later mention.
- Each new memory's document_time is when it was stated. When event_time is UNKNOWN
  on either side, you may use document_time as a weak recency signal, but prefer
  no relation over guessing direction from a weak or absent temporal signal.
- RELATED requires a concrete shared episode, object, decision, or participant-specific
  context that could aid later retrieval. Mere lexical overlap or broad topic similarity
  is not enough. Prefer no relation over a weak or speculative RELATED edge.
"""

ANSWERABILITY_GATE_PROMPT = """Decide whether exactly one supplied seed directly and
atomically contains the complete answer to the question. PASS only for an explicit
answer already stated in that seed, without outside knowledge, unresolved conflict,
causal inference, combining memories, or additional retrieval.

A merely relevant seed is not sufficient. For advice, diagnosis, explanation, why,
comparison, transition, or multi-part questions, FAIL unless that single seed itself
explicitly contains the complete requested conclusion and all requested rationale.
Match the requested answer slot, not merely the episode anchor. If the question asks
for a reaction during an event, a seed describing the event but omitting the reaction
must FAIL. If it asks when the participant performed an action, a later instruction
or proposed timing is not the observed timing and must FAIL.
For a question asking whether the user should, can, or needs to take an action, a
seed containing only symptoms, an explanation, or an old plan must FAIL unless it
also states the requested decision under the currently described conditions. For a
request to reduce, stop, continue, or change an action, an earlier plan must FAIL when
the question introduces a changed symptom or new reason that the plan did not address.
For a before/after comparison, a seed containing only one side must FAIL. For earliest,
latest, first, last, onset, or first-documented questions, FAIL: one seed cannot
establish an extremum over the participant's timeline.
For an entity request, the seed must explicitly state an entity of the requested
type. A symptom, laboratory abnormality, mechanism, or broad assessment is not a
disease diagnosis; an exposure is not an allergy; and a plan is not a completed test
result merely because the concepts are medically related.
For a simple category/entity lookup, accept a seed when it explicitly names the
requested entity and the category is only a paraphrase of the question (for example,
"diabetes" directly answers "which chronic metabolic disease"). Do not require the
seed to repeat the question's exact wording.
Offering a non-drug coping method does not by itself answer whether medication should
or should not be used. For a conditional decision, the seed must state the decision,
its trigger or qualification, and any requested alternative. FAIL when another supplied
seed exposes a safety constraint, contraindication, explicit preference, or alternative
that materially qualifies the proposed one-seed answer.
Choose only the smallest direct answer, never an extra supporting seed. Do not classify
the query, define slots, create operations, or assign scores.

SEEDS:
{seeds}

QUESTION: {question}

Return JSON only:
{{"pass": true, "support_ref": "$seed0"}}
"""

COMPACT_CONTROLLER_PROMPT = """You are the only semantic controller for a memory query.
Return exactly one JSON object. Choose DIRECT only when exactly one supplied seed
atomically contains the complete answer. Otherwise emit one minimal retrieval PLAN.
Never answer the question and never request a second semantic validator.

DIRECT shape:
{{"route":"DIRECT","support_ref":"$seed0"}}
DIRECT is forbidden for temporal localization/extrema, comparison, advice/decision,
visible options, causal explanation, or any question that needs two facts. A current
state question may use DIRECT when the one seed has a canonical state_identity, an
active/non-conflicting status, and the requested value. A merely topical or
reassuring seed is not a complete answer. Do not PASS a seed that only says a
participant generally feels weak, improved, or unwell when the question asks for
a dated or multi-attribute symptom pattern; the seed must itself contain the
requested qualifiers and answer-bearing detail.

PLAN retrieves only participant-specific evidence. The final answer model performs
option evaluation and domain reasoning.

Query modes: DIRECT, STATE, TEMPORAL, DECISION, MULTI_OPTION, CAUSAL, COMPARISON.
Slot types and fields:
- DIRECT: ["value"]
- CURRENT_STATE: ["state_identity","value"]
- TEMPORAL: ["value", exact time_axis], relation EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN
- TRANSITION: ["old_value","new_value"]
- COMPARISON: ["left","right"]
- CAUSE_PATH: ["path"], only for an explicit stored participant-specific causal edge

Evidence roles: ANSWER, FOCAL_STATE, LONGITUDINAL_CONTEXT, ACTION_RULE, CONSTRAINT,
ALTERNATIVE, TEMPORAL, CAUSE, COMPARAND.

Planning rules:
- DECISION: retrieve only roles that can change the decision. Usually current evidence,
  trajectory/warning signs, and active guidance or constraints; do not force all roles
  when the supplied seeds already cover one.
- MULTI_OPTION: Retrieve a shared evidence set that covers the distinct medical concepts mentioned across all options. If an option mentions a specific symptom or fact not covered by the seeds, use SEMANTIC_SEARCH to find it. Do not use 'option_label' or 'evidence_role'='OPTION'. Use option_coverage=[].
- CAUSAL: retrieve grounded cause/effect endpoints and relevant trajectory. General
  medical mechanisms are inferred by the answer model. Use CAUSE_PATH/FOLLOW_CAUSES
  only when the ledger is expected to contain an explicit causal relation.
- TEMPORAL: Every temporal query (EARLIEST, LATEST, EXACT date limit, BEFORE, AFTER) MUST use TEMPORAL_FILTER on the specific time_axis (event_time or document_time). For EARLIEST/LATEST, use LOCATE_ANCHOR then TEMPORAL_FILTER over that pool. For EXACT, use SEMANTIC_SEARCH then TEMPORAL_FILTER.
- Seed refs must be $seed0, $seed1, $seed2. Operation refs MUST use $0, $1 to refer to the outputs of the 0th, 1st previous operations. Do NOT use slot IDs (like "s1") as refs.
- The "produces" array MUST contain only valid slot IDs defined in required_slots. Do not invent intermediate slot names. Multiple operations can produce the same slot.
- operations=[] is valid when seed_coverage covers every slot.
- SEMANTIC_SEARCH strategy is FOCAL, TRAJECTORY, DECISION_BUNDLE, or SHARED_OPTIONS.
  Use TRAJECTORY for change over time, DECISION_BUNDLE for advice, and SHARED_OPTIONS
  for visible choices. Strategy diversifies one bounded operation; it is not a new search.

Code derives memory budget from the validated slots and operations. Do not emit a
budget field; make the plan complete within four operations.

Operations:
- {{"op":"SEMANTIC_SEARCH","query":"...","top_k":5,"strategy":"FOCAL|TRAJECTORY|DECISION_BUNDLE|SHARED_OPTIONS","produces":["s1"]}}
- {{"op":"LOCATE_ANCHOR","query":"...","produces":["s1"]}}
- {{"op":"TEMPORAL_FILTER","relation":"...","axis":"event_time|document_time|origin_document_time|effective_event_time","fallback_axis":"","anchor":"$0 or ISO","end":"$1 or ISO","candidate_refs":["$0"],"query":"...","produces":["s1"]}}
- {{"op":"RESOLVE_STATE","query":"...","produces":["s1"]}}
- {{"op":"FOLLOW_CAUSES","start":["$seed0"],"direction":"OUT|IN","depth":2,"goal":"...","produces":["s1"]}}

{domain_guidance}

SEEDS:
{seeds}

CONTEXT HINTS (routing only, never evidence or refs):
{context_map}

QUESTION: {question}

PLAN shape:
{{
  "route":"PLAN",
  "query_mode":"DIRECT",
  "required_slots":[{{"id":"s1","description":"answer-bearing evidence","type":"DIRECT","required_fields":["value"],"time_axis":"","temporal_relation":"","evidence_role":"ANSWER","option_label":""}}],
  "seed_coverage":[{{"slot_id":"s1","refs":["$seed0"]}}],
  "operations":[],
  "option_coverage":[],
  "need_evidence":false
}}
"""

SLOT_SUPPORT_GATE_PROMPT = """Validate whether proposed memories directly support
each exact retrieval slot. This is a sufficiency check, not a search or answer task.
Use only the supplied candidates. A topical, chronologically nearby, or merely
plausible memory is not support. For a temporal slot, select the candidate describing
the exact focal event, not a later recurrence or broader recap just because it is more
recent. The effective_event_time field is a valid explicit temporal value when present.
For an observation, result, response, or trajectory slot, a plan or instruction to
measure something is not evidence that the measurement or trajectory occurred. For a
response or trajectory under treatment, accept direct patient observations, measured
results, or explicit changes over time; reject a clinician's reassurance, conclusion,
or causal interpretation when the underlying observations are not in that candidate.
A trajectory requires an explicit trend or at least two compatible observations that
establish change/persistence across time; one isolated reading is only a point result.
For an intervention-use slot, a lifestyle or monitoring plan does not establish that
the intervention is permitted, contraindicated, or needed unless it states that
decision directly.
For an entity-bearing DIRECT slot, the candidate must explicitly contain the requested
answer category: symptoms or abnormal measurements cannot fill a diagnosis slot, and
a named-item restriction cannot establish a broader class-wide restriction. For a
conditional rule, the memory must preserve both the trigger and the action. For a
comparison, keep distinct support for both named sides. Return no reference for an
unsupported slot.
For a CAUSE_PATH slot, the selected memories and supplied CAUSES relations must form a
directed path that answers the exact start-to-goal mechanism in the slot description.
An unrelated causal edge or a RELATED/SUPPORT edge is not causal-path support.

Respect evidence_role exactly. FOCAL_STATE is the present observation being acted on;
LONGITUDINAL_CONTEXT must contain prior response, progression, recurrence, or a
decision-changing history rather than another paraphrase of the focal state;
ACTION_RULE must directly preserve participant-specific guidance, a contraindication,
a warning threshold, or a trigger-action policy for the requested action. One broad
clinician interpretation must not fill all three roles unless it explicitly contains
all three distinct pieces. For visible choices, validate the shared factual roles that
the answer model will use to evaluate every option. Options are propositions, never
memory slots, and option_coverage must remain empty. Set query_sufficient=false when
the shared evidence still lacks a participant-specific state, constraint, rule, or
alternative needed to distinguish the choices.

ORIGINAL QUESTION:
{question}

OPTION COVERAGE:
{option_coverage}

SLOTS AND ALLOWED CANDIDATES:
{slot_candidates}

Return JSON only:
{{
  "supports": [{{"slot_id":"s1", "refs":["$candidate0"]}}],
  "query_sufficient": true,
  "uncovered_option_labels": []
}}
"""

MEDICAL_PLANNER_GUIDANCE = """CLINICAL DECISION GUIDANCE:
- For treatment escalation or dose change, retrieve the smallest shared evidence bundle
  that preserves exposure/adherence, observed trajectory, and decision-changing warning
  signs or constraints. Do not create one slot for every possible answer.
- A plan to measure is not an observation. A clinician conclusion is not the observed
  trajectory; retrieve measurements, symptoms, or explicit change over time. One
  isolated measurement is not a trajectory.
- For medication use, separately retrieve whether/when it is permitted or needed,
  participant-specific contraindications, and a stated acceptable alternative or
  preference when available. Lifestyle advice alone does not establish medication
  eligibility or safety.
- For multi-hop reasoning, retrieve patient-specific endpoints, measurements, treatments,
  and chronology. The final answer may supply a standard clinical mechanism connecting
  those grounded endpoints; do not require an unstated mechanism to exist as CAUSES.
"""

PLANNER_PROMPT = """Create a minimal retrieval program; do not answer the question.
Define every piece of information required to answer as a typed slot. Use at most
4 operations. $0 references operation 0; $seed0 references initial seed 0.
An empty operation list is valid when seed_coverage already covers every slot.

First choose query_mode: DIRECT, STATE, TEMPORAL, DECISION, MULTI_OPTION,
CAUSAL, or COMPARISON. This is a semantic planning declaration, not a benchmark
label. Questions asking whether an action should/can/needs to be taken, whether a
situation is serious, or what the participant should do use DECISION. Visibly
enumerated options always use MULTI_OPTION. Questions asking when, on what date,
at what time, first/last occurrence, onset, or documented occurrence use TEMPORAL.
Questions asking whether event/state X caused, explains, contributed to, or is
related through a participant-specific chain to Y use CAUSAL. The one-seed direct
answerability gate has already failed before this planner is called, so do not
replace a temporal, causal, comparison, decision, or multi-option requirement with
one generic DIRECT slot.

A slot is an independent answer requirement, not a paraphrase of the question.
Before returning a plan, mentally test whether the covered slots would let an
answer model justify the full answer without filling gaps from outside knowledge.

{domain_guidance}

CORE DECOMPOSITION RULES:
- Advice, decision, and inference questions retrieve only factual roles that can
  change the answer. Use FOCAL_STATE, LONGITUDINAL_CONTEXT, ACTION_RULE, CONSTRAINT,
  or ALTERNATIVE as needed, without forcing a fixed three-role template. Retrieve
  facts and let the answer model infer the conclusion.
- Use CURRENT_STATE only for a state that must be resolved to a current head. A
  merely relevant active-looking memory is not current-state evidence; use
  RESOLVE_STATE when seeds do not already establish the resolved head.
- For multiple-choice questions, retrieve one shared set of participant facts that
  lets the answer model evaluate all visible options: current state, applicable rule,
  constraints, contraindications, preferences, and alternatives. Never search each
  distractor proposition and never create one slot per option. Use option_coverage=[].
- When options differ by a conditional action, retrieve the participant-specific
  trigger-action rule and the state that makes it applicable. Preserve named entities
  and the requested evidence role in the search query.
- For an explicit left-versus-right or before-versus-after question, require support
  for both named sides using COMPARISON or two clearly named DIRECT slots.
- For first/onset/earliest/latest questions, use a TEMPORAL slot with the intended
  axis and a TEMPORAL_FILTER EARLIEST or LATEST operation. LOCATE_ANCHOR alone does
  not establish an extremum.
- A semantic-search query must name the answer-bearing memory role when the question
  asks for one, such as person, place, item, preference, diagnosis, measurement, or
  plan. Preserve that role alongside the subject instead of using a broad paraphrase.
- Choose origin_document_time only when the target is the act of recording or
  reporting itself. If a record describes when an event happened, localize the event,
  not the later record. Choose effective_event_time for onset/first-appearance when
  event_time may be coarse or relative and an origin source date can refine it.
- For an undated question asking when claim X was documented or recorded, use
  document_time: EXACT for a stated record date, EARLIEST for the first record,
  and LATEST for the most recent record. Use origin_document_time only when the
  provenance of the original assertion itself is the target. Do not choose
  event_time merely because the record describes an event. For onset or
  first-appearance questions where the event is described relatively (such as
  "recently" or "over the past few days"), choose effective_event_time and
  explicitly declare document_time as fallback_axis when event_time is absent.
- Use CAUSE_PATH only when the answer requires an explicit participant-specific
  causal relation stored in the ledger. Otherwise use DIRECT endpoint/trajectory
  slots and let the answer model supply standard domain mechanisms. If FOLLOW_CAUSES depends on
  an earlier operation, reference that output ($0, $1), not an unrelated seed.
  For a causal query, retrieve a specific cause/effect anchor first when needed,
  then use FOLLOW_CAUSES. CAUSAL mode does not itself require a stored CAUSE_PATH.
- On replanning, preserve the meaning of each missing slot and retrieve only its
  missing evidence. Do not rename or weaken an uncovered requirement. Inspect the
  prior trace and do not repeat a search that returned no new support or no coverage
  gain. Emit only the supplied missing shared-evidence slots.
- operations=[] is valid only when seed_coverage structurally and semantically
  covers every required slot.
- For synthesis or decision questions, do not collapse a required longitudinal,
  warning, exposure, or action-rule role into a generic current-state slot. A
  CURRENT_STATE slot means a resolved active head; historical progression belongs
  in a DIRECT slot with the appropriate longitudinal role and history=true when
  older versions are required. If one supplied candidate cannot establish the
  declared role, add an explicit operation for that role.
- Every slot not exactly covered by seed_coverage must be listed in produces of at
  least one operation. Do not postpone a known missing slot to replanning.
- VERIFY_EVIDENCE never establishes an answer slot by itself. It may only target a
  slot already covered by seed_coverage or produced by another operation.
- A TEMPORAL EXACT slot may use zero operations only when typed seed_coverage names
  a seed that directly contains the requested axis. EARLIEST, LATEST, BEFORE, AFTER,
  and BETWEEN still require TEMPORAL_FILTER because one seed cannot prove an
  extremum or comparison range.

BUDGET TIERS:
- SMALL: one direct/current/temporal requirement, at most 1 operation and 3 memories.
- MEDIUM: a transition, comparison, or two coupled requirements, at most 2 operations
  and 5 memories.
- LARGE: genuinely multi-role advice, inference, multi-option, causal, or multi-hop
  evidence, at most 4 operations and 8 memories. Simpler instances may use MEDIUM.
- A LARGE plan must decompose at least two independent evidence roles unless its
  single requirement is TRANSITION, COMPARISON, or CAUSE_PATH. Never represent a
  LARGE workload as one generic DIRECT slot.
- Choose the smallest tier whose limits can execute the complete plan. Do not output
  a numeric max_memories value. Budget is an execution limit, not a query classifier:
  a single typed CAUSE_PATH can legitimately require LARGE resources.

SLOT TYPES AND REQUIRED FIELDS:
- DIRECT: ["value"]
- CURRENT_STATE: ["state_identity", "value"]
- TEMPORAL: ["value", one exact temporal axis], with temporal_relation set to
  EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN.
- TRANSITION: ["old_value", "new_value"]
- CAUSE_PATH: ["path"]
- COMPARISON: ["left", "right"]
- A DIRECT slot may set "history": true only when a former or superseded value is
  itself required. Omit it otherwise. This flag never changes the temporal axis.
- evidence_role is one of ANSWER, FOCAL_STATE, LONGITUDINAL_CONTEXT, ACTION_RULE,
  CONSTRAINT, ALTERNATIVE, TEMPORAL, CAUSE, COMPARAND.
- option_label is empty. Visible options are evaluated against shared evidence.

OPERATIONS:
- {{"op":"SEMANTIC_SEARCH","query":"...","top_k":5,"produces":["s1"]}}
- {{"op":"LOCATE_ANCHOR","query":"event X","produces":["s1"]}}
- {{"op":"TEMPORAL_FILTER","relation":"BEFORE|AFTER|BETWEEN|EXACT|EARLIEST|LATEST",
   "axis":"event_time|document_time|origin_document_time|effective_event_time",
   "fallback_axis":"optional explicit fallback axis","anchor":"$0 or ISO date",
   "end":"$1 or ISO date","candidate_refs":["$0"],
   "query":"target event","produces":["s1"]}}
- {{"op":"RESOLVE_STATE","query":"state being asked","produces":["s1"]}}
- {{"op":"FOLLOW_CAUSES","start":["$seed0"],"direction":"OUT|IN",
   "depth":2,"goal":"reasoning target","produces":["s1"]}}
- {{"op":"VERIFY_EVIDENCE","memory_refs":["$0"],"produces":["s1"]}}

INITIAL SEEDS:
{seeds}

PLANNING CONTEXT MAP (routing hints only):
{context_map}

Context-map entries summarize durable memory that may matter to planning. They are
not evidence, have no valid $ reference, and cannot appear in seed_coverage. When a
hint identifies a missing diagnosis, trajectory, risk, constraint, preference, or
plan, create an explicit operation that retrieves the underlying memory.

QUESTION: {question}

MISSING SLOTS FROM A PRIOR ROUND (empty on the first plan):
{missing_slots}

PRIOR RETRIEVAL TRACE (empty on the first plan):
{prior_trace}

PLAN VALIDATION ERROR (empty on the first attempt):
{validation_error}

Return JSON only:
{{
  "query_mode": "DIRECT",
  "required_slots": [{{
    "id":"s1", "description":"required evidence", "type":"DIRECT",
    "required_fields":["value"], "time_axis":"", "temporal_relation":"",
    "evidence_role":"ANSWER", "option_label":""
  }}],
  "seed_coverage": [{{"slot_id":"s1", "refs":["$seed0"]}}],
  "operations": [],
  "option_coverage": [],
  "need_evidence": false,
  "budget_tier": "SMALL"
}}

Use option_coverage=[] when the question has no visibly enumerated options.
"""

# Runtime planner contract. The longer prompt above is retained only as a
# readable specification while experiments migrate; query execution uses this
# compact contract to reduce latency and schema drift.
TYPED_PLANNER_PROMPT = """Return one JSON retrieval program, never an answer.

First describe the information need:
- target_entities: explicit people/objects/concepts being asked about
- target_property: the answer-bearing property or event
- answer_type: DIRECT|CURRENT_STATE|TEMPORAL|TRANSITION|CAUSE_PATH|COMPARISON
- reasoning: NONE|SYNTHESIS|DECISION|COMPARISON|CAUSAL
- temporal: axis, relation, anchor; use empty strings when not temporal

Then define 1-3 independent typed required_slots and at most 4 operations.
Slot fields are fixed: DIRECT=[value], CURRENT_STATE=[state_identity,value],
TEMPORAL=[value, chosen axis], TRANSITION=[old_value,new_value],
CAUSE_PATH=[path], COMPARISON=[left,right].
Evidence roles: ANSWER, FOCAL_STATE, LONGITUDINAL_CONTEXT, ACTION_RULE,
CONSTRAINT, ALTERNATIVE, TEMPORAL, CAUSE, COMPARAND.

Rules:
- A decision needs focal state plus only the trajectory/rule/constraint that can
  change the decision. Retrieve facts; the answer model performs inference.
- Visible options use MULTI_OPTION and one shared participant evidence bundle,
  never one slot/search per option. option_coverage is always []. A zero-operation
  plan is allowed only when the supplied seeds already contain the shared rule or
  facts needed to evaluate every option; one topical memory about a single option
  is not sufficient. When that shared evidence is absent, emit one
  SHARED_OPTIONS search with the option propositions included in its query.
- For a decision or inference question, do not certify a single generic DIRECT or
  CURRENT_STATE slot merely because one seed is topical. If the conclusion depends
  on observed change, symptoms, exposure, warning signs, or an action rule, declare
  the distinct evidence roles needed to connect those endpoints. A question asking
  whether an event explains an outcome requires both participant-specific endpoints
  and their chronology; use CAUSE_PATH only when an explicit CAUSES edge is stored.
- Temporal slots use the exact requested axis. EARLIEST/LATEST/ranges require
  TEMPORAL_FILTER over a searched pool. Never silently change axis.
- CAUSE_PATH/FOLLOW_CAUSES is only for an explicit stored causal edge. Otherwise
  retrieve distinct participant-specific endpoints and trajectory.
- seed refs are $seed0..$seed2. Operation refs are backward-only $0..$3.
  Slot IDs are never refs. produces contains only declared slot IDs.
- SEMANTIC_SEARCH strategy is FOCAL, TRAJECTORY, DECISION_BUNDLE, or SHARED_OPTIONS.
  Strategy must follow the declared evidence role; it diversifies one bounded output
  and never opens a hidden retrieval path.
- operations=[] is valid when seed_coverage covers every required slot.
- VERIFY_EVIDENCE cannot be the sole producer of an answer slot.
- Context hints route a search but are not evidence and have no refs.
- Do not emit query_mode or budget_tier. Code derives both from query_spec and
  the validated executable structure.
{domain_guidance}

SEEDS: {seeds}
CONTEXT_HINTS: {context_map}
QUESTION: {question}
MISSING_SLOTS: {missing_slots}
PRIOR_TRACE: {prior_trace}
VALIDATION_ERROR: {validation_error}

Return JSON only:
{{
  "query_spec":{{
    "target_entities":[], "target_property":"",
    "answer_type":"DIRECT", "reasoning":"NONE",
    "temporal":{{"axis":"","relation":"","anchor":""}}
  }},
  "required_slots":[{{
    "id":"s1", "description":"answer-bearing evidence", "type":"DIRECT",
    "required_fields":["value"], "time_axis":"", "temporal_relation":"",
    "evidence_role":"ANSWER", "option_label":""
  }}],
  "seed_coverage":[{{"slot_id":"s1","refs":["$seed0"]}}],
  "operations":[{{"op":"SEMANTIC_SEARCH","query":"...","top_k":3,"strategy":"FOCAL","produces":["s1"]}}],
  "option_coverage":[], "need_evidence":false
}}
"""
