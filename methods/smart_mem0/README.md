# SmartMem0 Module Map

`methods.smart_mem0_agent` remains the compatibility import used by the benchmark.

## Evaluation memory snapshots

SmartMem0 evaluations persist the complete memory state after every evaluation
unit under `outputs/memory_snapshots/`. A normal rerun automatically restores the
exact snapshot matching dataset, method, model, config, context, unit content,
and parent-unit lineage before evaluating that unit's queries.

For MedMemoryBench, each later unit snapshot is cumulative for its persona even
though the evaluator only injects the new sessions at that boundary. LoCoMo uses
one isolated snapshot per sample. Snapshots are never shared across personas or
samples.

Capture/consolidation prompt hashes and `MEMORY_WRITE_SCHEMA_VERSION` are part of
the cache key, so a write-contract change automatically creates a new snapshot
generation. Use `--rebuild-memory` only to force a fresh stochastic rebuild under
the same contract. Query-only retrieval experiments omit the flag and reuse the
frozen unit memories for a fair ablation. `memory_snapshot_revision` remains a
manual escape hatch for an intentional cache generation change.
The implementation lives in this package and is split by runtime responsibility.

| Module | Owns |
| --- | --- |
| `agent.py` | Public `SmartMem0Agent` facade only |
| `contracts.py` | Dataclasses, enums, and validation constants |
| `prompts.py` | LLM input/output contracts |
| `core.py` | Runtime state, normalization, evidence staging, BM25/dense index |
| `capture.py` | Turn windows, MWC, and LLM memory extraction |
| `consolidation.py` | ADD-only commit, state heads, and typed relations |
| `write.py` | Session transaction, rollback, frozen-store lifecycle |
| `planning.py` | One-seed answerability gate, bounded planning context map, and validated retrieval plans |
| `retrieval.py` | Deterministic temporal, state, semantic, and causal operations |
| `execution.py` | Deterministic typed coverage, stopping, and pure arbitration |
| `query.py` | Final support boundary, context packing, telemetry, and answer request |

## Runtime Paths

```text
WRITE: capture -> consolidation -> transactional commit -> index refresh
READ EASY: recall -> one-seed answerability gate -> answer
READ HARD: recall -> compact controller -> typed query spec -> local operations -> context -> answer
```

The planner emits target entities/property, answer type, reasoning requirement,
typed slots, and operations. Code derives query mode and memory/operation budget
from that validated structure; the LLM does not self-score query difficulty.
Planner zero-operation plans remain valid. Slot coverage is structural and
deterministic; the baseline does not call a separate semantic validator or recovery
LLM. `CURRENT_STATE` accepts only resolved state-head memories. Temporal operations
preserve a small focal candidate set. For inference, patient-specific facts remain
memory-grounded while the answer model may explicitly infer standard domain mechanisms
between those grounded endpoints.

The planning context map is built from compact write-time tags and salience metadata.
It gives the planner a bounded view of durable identity, trajectory, risk, constraint,
preference, resource, and plan anchors. Map entries are routing hints only: they have
no valid `$` reference and cannot enter final answer context without an explicit
retrieval operation.

Decision plans retrieve only roles that can change the answer. Enumerated-option
plans retrieve one shared evidence bundle of current facts, active guidance,
constraints, contraindications, preferences, and alternatives; distractor text is
never treated as a memory fact that must be retrieved. Causal plans retrieve grounded
endpoints and use stored `CAUSES` only when an explicit participant-specific edge
exists.

The default read path separates answerability from planning. The mini gate can only
authorize one atomic seed; it cannot define slots, operations, or an answer. A failed
or structurally skipped gate invokes the plan-only controller. Deterministic recovery
may issue explicit operations for uncovered typed slots without another LLM call.
The default config uses one compact controller call that returns either a direct seed
reference or a typed plan. The separate semantic slot validator, LLM replan, and
planner repair remain ablation switches; the legacy gate-plus-planner route is also
available for controlled comparison.

The split follows the useful boundaries visible in the bundled implementations:
Mem0 separates memory/index backends, MemoRAG separates prompts and retrieval, and
LightMem separates construction from search. SmartMem0 keeps one stable benchmark
facade while applying those boundaries to its evidence and typed-slot architecture.

## Dependency Rule

Pipeline modules may call methods from modules to their right in `agent.py`'s mixin
order. They must not import the public facade. This keeps the compatibility layer
free of algorithmic behavior and avoids circular imports.
