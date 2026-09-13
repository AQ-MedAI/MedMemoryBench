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
Excerpt allocation is `global` by default or `joint`; joint reserves existing
excerpt slots for selected support turns before the global allocator runs.

`temporal_query_mode: dual_axis_semantic` only reranks semantic candidates. It
computes event-time and record-time compatibility separately and applies
`semantic * (1 + temporal_retrieval_weight * compatibility)`, where matching
both axes adds a bounded confidence increment. Missing event time is neutral.

`local_reranker_mode: cross_encoder` needs an explicit model. It reranks only
the bounded top-k pool using `(1-alpha)*base + alpha*calibrated_reranker`, is
process-shared per model, and does not run or load when off. Adaptive evidence
uses top-score gap and strong-candidate count between the configured bounds.
