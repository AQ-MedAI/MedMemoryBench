# Event-State Query Retrieval Options

All options below live under `retrieval_config`. Defaults preserve the former
retrieval semantics except `query_manager_reuse_mode: auto`, which shares one
read-only imported store for a context instead of importing it for every
query. Set it to `legacy` for the historical import behavior.

```yaml
# Exact legacy retrieval behavior
query_manager_reuse_mode: legacy
fusion_mode: rrf
selector_mode: state_mmr
turn_lexical_mode: overlap
turn_sparse_unit: turn
episode_relevance_mode: summary
episode_excerpt_mode: global
local_reranker_mode: off
temporal_query_mode: legacy
evidence_budget_mode: fixed
```

```yaml
# Low-RAM, planner-off experiment
planner_mode: off
query_manager_reuse_mode: shared
query_memory_guard_enabled: true
query_memory_reserve_mb: 1024
query_memory_max_fraction: 0.5
query_memory_min_available_mb: 512
fusion_mode: weighted_score
score_calibration_mode: minmax
selector_mode: source_coherent_mmr
source_coherence_max_sources: 3
source_coherence_score_ratio: 0.7
source_coherence_score_mass: 0.85
turn_lexical_retrieval_enabled: true
turn_lexical_mode: bm25
turn_sparse_unit: sentence_chunk
bm25_k1: 1.2
bm25_b: 0.75
episode_relevance_mode: summary_plus_best_turn
episode_excerpt_mode: joint
temporal_query_mode: dual_axis_semantic
```

```yaml
# Full optional stack; cross-encoder is lazy and RAM-gated
local_reranker_mode: cross_encoder
local_reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
local_reranker_top_k: 20
local_reranker_batch_size: 8
local_reranker_weight: 0.35
local_reranker_max_length: 512
local_reranker_min_available_mb: 2048
local_reranker_auto_disable_on_low_memory: true
evidence_budget_mode: adaptive
adaptive_evidence_min: 8
adaptive_evidence_max: 14
```

`query_manager_reuse_mode` accepts `legacy`, `shared`, `worker_local`, and
`auto` (default). `auto` selects `shared`; `worker_local` uses a RAM guard
with `query_worker_memory_estimate_mb` or a conservative estimate. The guard
uses psutil when installed, then Linux `MemAvailable`, then one worker.

`fusion_mode` is `rrf` (default) or `weighted_score`. Score fusion calibrates
each query-local representation list with `none`, `minmax` (default), or
`sigmoid_zscore`, then applies the configured representation weights.

`selector_mode` is `topk`, `mmr`, `state_mmr` (default), or
`source_coherent_mmr`. The latter aggregates a session as its best relevance
plus `source_coherence_additional_weight` times up to
`source_coherence_additional_candidates` more strong candidates. It prefers
the adaptive source set determined by `source_coherence_max_sources`,
`source_coherence_score_ratio`, and `source_coherence_score_mass`; it is not a
hard filter.

Sparse retrieval remains disabled unless `turn_lexical_retrieval_enabled` is
true. Its `overlap` mode is the legacy token-set baseline. `bm25` uses standard
BM25 with `bm25_k1` (positive, default 1.2) and `bm25_b` ([0,1], default .75).
`turn_sparse_unit: sentence_chunk` has bounded punctuation chunks controlled
by `turn_chunk_min_chars` and `turn_chunk_max_per_turn`; chunk hits resolve to
the parent immutable turn and never create dense embeddings.

Episode scoring is `summary` by default or `summary_plus_best_turn`, a
normalized blend of `episode_summary_weight` and `episode_best_turn_weight`.
Excerpt allocation is `global` by default or `joint`; joint globally ranks
selected support turns before the global allocator fills remaining slots.

`temporal_query_mode: dual_axis_semantic` only reranks semantic candidates. It
computes event-time and record-time compatibility separately and applies
`semantic * (1 + temporal_retrieval_weight * compatibility)`, where matching
both axes adds a bounded confidence increment. Missing event time is neutral.

`local_reranker_mode: cross_encoder` needs an explicit model. It reranks only
the bounded top-k pool using `(1-alpha)*base + alpha*calibrated_reranker`, is
process-shared per model, and does not run or load when off. Adaptive evidence
uses top-score gap and strong-candidate count between the configured bounds.

## Query-Local Source Coherence

`_source_ids()` remains the canonical full-provenance view: every session in a
claim's `EvidenceRef`s remains visible and renderable. Query-time source
coherence must not treat each corroborating session as equally relevant,
however. Under the default `query_relevant` mode, Event-State scores persisted
immutable source-turn vectors against the current query, takes the best turn
per reference, takes the best reference per source session, and keeps the
configured strongest session count. It never sums arbitrary turn counts.

```yaml
source_coherence_claim_source_mode: query_relevant # or all_provenance
source_coherence_claim_max_sources: 2
source_coherent_use_state_relation_bonus: true
source_coherent_use_representation_balance_bonus: true
source_coherent_use_source_diversity_bonus: false
```

`all_provenance` is retained for backwards-comparison ablations. If cited turn
vectors are unavailable, the selector deterministically uses an available
reference (preferring `origin`) and ultimately full provenance; the claim is
never discarded. Diagnostics separately report full-provenance and query-local
source counts/IDs, preferred-source scores, selected preferred/non-preferred
items, query-local claims, and full-provenance fallbacks. Source preference is
soft: MMR relevance and redundancy still apply, so strong non-preferred items
can be selected. State relation and representation bonuses are explicit in
source-coherent mode; source diversity is off there by default.

## Global Joint Excerpts

`episode_excerpt_mode: global` is unchanged. In `joint` mode, Event-State now
collects one scored preferred support turn from each selected episode,
deduplicates against claim provenance, direct immutable turns, and duplicate
episode turns, then ranks all eligible support turns globally by query-turn
similarity. It reserves the strongest ones up to
`max_episode_source_excerpts_total`, and uses the established global allocator
only to fill remaining slots. This prevents selected-episode iteration order
from consuming the excerpt budget. Diagnostics include support candidates,
reserved/global-fill/dedup counts, and bounded selected support-turn IDs.

`summary_plus_best_turn` uses a runtime-only `episode_id -> tuple[turn_key,
...]` index. It is rebuilt from snapshots, contains neither raw text nor
embeddings, and avoids scanning every turn for every episode.

## Numbered Ablations

The full configs in `configs/method_config/test/` share model, embedding,
build, answer, evidence-budget, candidate-depth, planner-off, shared-manager,
and RAM-guard settings. Cross-encoder and adaptive evidence are off/fixed.

| config | BM25 | episode support | selector | excerpts | fusion | temporal |
| --- | --- | --- | --- | --- | --- | --- |
| gemini1 | no | summary | state MMR | global | RRF | legacy |
| gemini2 | chunks | summary | state MMR | global | RRF | legacy |
| gemini3 | chunks | summary + best turn | state MMR | global | RRF | legacy |
| gemini4 | chunks | summary + best turn | state MMR, no source diversity | global | RRF | legacy |
| gemini5 | chunks | summary + best turn | state MMR, lambda .85 | global | RRF | legacy |
| gemini6 | chunks | summary + best turn | query-local source coherent | global | RRF | legacy |
| gemini7 | chunks | summary + best turn | state MMR | joint/global ranked | RRF | legacy |
| gemini8 | chunks | summary + best turn | state MMR | global | weighted/minmax | legacy |
| gemini9 | chunks | summary + best turn | state MMR | global | RRF | dual axis |
| gemini10 | chunks | summary + best turn | query-local source coherent | joint/global ranked | RRF | legacy |

Use the normal query stage against a frozen snapshot, for example:

```bash
python3 main.py -m test/event_state_gemini3 -d locomo_1 --stage query --memory-run <memory-run-directory>
```

The repository has no separate no-answer retrieval-only CLI, so correctness
tests validate implementation but not benchmark effectiveness.
