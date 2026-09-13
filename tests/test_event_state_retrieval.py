from copy import deepcopy
from types import SimpleNamespace

from methods.event_state import retrieval as retrieval_module
from methods.event_state.retrieval import EventStateRetriever, calibrate_scores
from methods.event_state.context import episode_turn_embedding_text, select_claim_evidence, select_global_episode_evidence
from methods.event_state.schemas import Claim, Episode, EvidenceRef, TurnEvidence
from methods.event_state.store import EventStateStore
from methods.event_state.temporal import parse_temporal_query
from methods.event_state_agent import EventStateAgent


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class DeterministicEmbedder:
    def embed_query(self, text):
        values = [float((sum(map(ord, text)) + index * 17) % 31 + 1) for index in range(8)]
        magnitude = sum(value * value for value in values) ** 0.5
        return [value / magnitude for value in values]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class CountingEmbedder(DeterministicEmbedder):
    def __init__(self):
        self.document_batches = []

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return super().embed_documents(texts)


def test_score_calibration_is_deterministic_for_equal_negative_and_single_values():
    assert calibrate_scores([], "minmax") == []
    assert calibrate_scores([-.2], "minmax") == [1.0]
    assert calibrate_scores([-.2, -.2], "sigmoid_zscore") == [.5, .5]
    assert calibrate_scores([.55, .9], "minmax") == [0.0, 1.0]


def test_yaml_false_and_legacy_null_modes_resolve_to_legacy_defaults():
    agent = EventStateAgent(
        embedding_client=Embedder(), local_reranker_mode=False,
        temporal_query_mode=None,
    )
    assert agent._retrieval_config["local_reranker_mode"] == "off"
    assert agent._retrieval_config["temporal_query_mode"] == "legacy"


def test_weighted_score_fusion_retains_dense_magnitude_and_rrf_remains_available():
    store = EventStateStore("fusion")
    retriever = EventStateRetriever(store, Embedder(), fusion_mode="weighted_score", score_calibration_mode="minmax")
    rows = retriever._fuse([("A", .9)], [("B", .55)], (), (), (), ())
    assert {row["id"]: row["dense_raw_score"] for row in rows} == {"A": .9, "B": .55}
    assert {row["id"]: row["dense_calibrated_score"] for row in rows} == {"A": 1.0, "B": 0.0}
    legacy = EventStateRetriever(store, Embedder(), fusion_mode="rrf")._fuse([("A", .9)], [("B", .55)], (), (), (), ())
    assert all("dense_raw_score" not in row for row in legacy)


def _original_select(retriever, candidates, count):
    """The pre-optimization selector, retained only as an equivalence oracle."""
    mode, selected = retriever.config.get("selector_mode", "state_mmr"), []
    remaining = list(candidates)
    if remaining and all(item.get("_planner_merged_final_score") for item in remaining):
        relevance = [float(item["final_score"]) for item in remaining]
    else:
        relevance = retrieval_module.normalize_scores(
            [float(item.get("final_score", item.get("score", 0.0))) for item in remaining]
        )
    relevance_by_id = {item["id"]: value for item, value in zip(remaining, relevance)}
    while remaining and len(selected) < count:
        semantic_remaining = [item for item in remaining if item["type"] != "turn"]
        choices = semantic_remaining if semantic_remaining and not selected else remaining
        if mode == "topk":
            choice, choice_score = max(
                ((item, item.get("final_score", item.get("score", 0.0))) for item in choices),
                key=lambda pair: (pair[1], pair[0]["id"]),
            )
        else:
            weight = float(retriever.config.get("mmr_lambda", 0.7))

            def score(item):
                vector = retriever._vector(item["id"], item["type"])
                redundancy = max(
                    (
                        retrieval_module.cosine(
                            vector, retriever._vector(other["id"], other["type"])
                        )
                        for other in selected
                    ),
                    default=0.0,
                )
                value = weight * relevance_by_id[item["id"]] - (1 - weight) * redundancy
                if mode == "state_mmr" and selected:
                    if any(
                        edge["source_id"] == item["id"] and edge["target_id"] == other["id"]
                        or edge["target_id"] == item["id"] and edge["source_id"] == other["id"]
                        for edge in retriever.store.edges
                        for other in selected
                    ):
                        value += float(retriever.config.get("state_relation_bonus", 0.05))
                    if item["type"] != selected[-1]["type"]:
                        value += float(retriever.config.get("representation_balance_bonus", 0.02))
                    if retriever._source_ids(item) - set().union(
                        *(retriever._source_ids(other) for other in selected)
                    ):
                        value += float(retriever.config.get("source_diversity_bonus", 0.02))
                return value

            choice = max(choices, key=lambda item: (score(item), item["id"]))
            choice_score = score(choice)
        choice["selection_score"] = choice_score
        selected.append(choice)
        remaining.remove(choice)
    for rank, item in enumerate(selected, 1):
        item["selected_rank"] = rank
    return selected


def _selector_store(item_count=24):
    store = EventStateStore("ctx")
    embedder = DeterministicEmbedder()
    for index in range(item_count):
        episode_id = f"E{index:02d}"
        turn = TurnEvidence(f"T{index:02d}", "User", "user", f"turn evidence {index}")
        episode = Episode(
            episode_id,
            "ctx",
            f"session-{index}",
            index,
            None,
            f"2025-01-{index % 28 + 1:02d}",
            ["User"],
            "primary_user",
            "",
            f"episode {index}",
            [turn],
        )
        vector = embedder.embed_query(episode_id)
        store.add_episode(episode, vector, [embedder.embed_query(episode_turn_embedding_text(turn))])
        claim = Claim(
            f"C{index:02d}",
            "User",
            "primary_user",
            "preference",
            f"value {index}",
            evidence=[EvidenceRef(episode_id, episode.source_session_id, [])],
        )
        store.add_claim(claim, embedder.embed_query(claim.claim_id))
    for index in range(item_count - 1):
        store.add_edge(f"C{index:02d}", f"C{index + 1:02d}", "REFINES")
    return store


def test_state_mmr_cache_preserves_selection_prompt_and_final_response(monkeypatch):
    store = _selector_store()
    retriever = EventStateRetriever(
        store,
        DeterministicEmbedder(),
        selector_mode="state_mmr",
        evidence_count=12,
        candidate_count=50,
    )
    candidates = [
        {"id": f"C{index:02d}", "type": "state_claim", "score": 1.0 - index / 100}
        for index in range(24)
    ]

    original_cosine = retrieval_module.cosine
    calls = {"count": 0}

    def counted_cosine(left, right):
        calls["count"] += 1
        return original_cosine(left, right)

    monkeypatch.setattr(retrieval_module, "cosine", counted_cosine)
    expected = _original_select(retriever, deepcopy(candidates), 12)
    original_calls = calls["count"]
    calls["count"] = 0
    actual = retriever._select_impl(deepcopy(candidates), 12)

    assert actual == expected
    assert calls["count"] * 3 < original_calls

    agent = EventStateAgent(
        llm_client=SimpleNamespace(chat=lambda *args, **kwargs: SimpleNamespace(content="fixed answer")),
        memory_llm_client=SimpleNamespace(chat=lambda *args, **kwargs: SimpleNamespace(content="fixed answer")),
        embedding_client=DeterministicEmbedder(),
        selector_mode="state_mmr",
        evidence_count=12,
        candidate_count=50,
        max_context_tokens=120000,
    )
    agent.set_context_id("ctx")
    agent._stores["ctx"] = store

    with monkeypatch.context() as legacy_patch:
        legacy_patch.setattr(EventStateRetriever, "_select_impl", _original_select)
        expected_prepared = agent.prepare_batch_query(
            "What preference did the user mention?", system_message="Answer exactly."
        )
    actual_prepared = agent.prepare_batch_query(
        "What preference did the user mention?", system_message="Answer exactly."
    )

    assert actual_prepared == expected_prepared
    assert agent.finalize_batch_query(actual_prepared, "fixed answer").to_dict() == (
        agent.finalize_batch_query(expected_prepared, "fixed answer").to_dict()
    )


def test_persisted_turn_vectors_preserve_evidence_selection_without_reembedding():
    store = _selector_store(2)
    embedder = CountingEmbedder()
    claim = store.claims["C00"]
    query_vector = embedder.embed_query("question")

    expected = select_claim_evidence(
        claim, store.episodes, query_vector, embedder, ref_limit=1,
    )
    expected_global = select_global_episode_evidence(
        [(0, store.episodes["E00"])], query_vector, embedder, limit=1,
    )
    assert embedder.document_batches

    embedder.document_batches.clear()
    cached_vectors = EventStateAgent._persisted_turn_vector_cache(store)
    actual = select_claim_evidence(
        claim, store.episodes, query_vector, embedder, ref_limit=1,
        turn_vector_cache=cached_vectors,
    )
    actual_global = select_global_episode_evidence(
        [(0, store.episodes["E00"])], query_vector, embedder, limit=1,
        turn_vector_cache=cached_vectors,
    )

    assert actual == expected
    assert actual_global == expected_global
    assert embedder.document_batches == []


def _episode(identifier, recorded_at, vector):
    return Episode(identifier, "ctx", identifier, 0, None, recorded_at, ["User"], "primary_user", "", identifier, []), vector


def test_temporal_parser_is_conservative_and_supports_bounded_iso_forms():
    assert parse_temporal_query("What did we discuss in the record dated 2025-03-15?").kind == "exact_record_time"
    assert parse_temporal_query("Where was I living as of 2025/03/15?").kind == "as_of"
    interval = parse_temporal_query("What happened between 2025-01-05 and 2025-01-15?")
    assert interval.start_date.isoformat() == "2025-01-05"
    assert parse_temporal_query("What happened around the 5th?") is None


def test_temporal_episode_channel_adds_exact_record_date_candidate():
    store = EventStateStore("ctx")
    for episode, vector in (_episode("A", "2025-01-01", [0.0, 1.0]), _episode("B", "2025-03-15", [0.0, 1.0]), _episode("C", "2025-06-01", [1.0, 0.0])):
        store.add_episode(episode, vector)
    retriever = EventStateRetriever(store, Embedder(), retrieve_claims=False, episode_top_k=1, candidate_count=1, evidence_count=1)
    selected, extra = retriever.retrieve("What did we discuss in the record dated 2025-03-15?")
    assert extra["temporal_constraint_detected"] is True
    assert extra["temporal_episode_candidate_count"] == 1
    assert selected[0]["id"] == "B"
    assert selected[0]["temporal_match_type"] == "exact_record_time"


def test_as_of_retrieval_exposes_historical_state_but_hides_current_version():
    store = EventStateStore("ctx")
    store.add_episode(Episode("E1", "ctx", "s1", 0, None, "2025-01-01", ["User"], "primary_user", "", "", []), [1.0, 0.0])
    store.add_episode(Episode("E2", "ctx", "s2", 1, None, "2025-03-15", ["User"], "primary_user", "", "", []), [1.0, 0.0])
    old = Claim("A", "User", "primary_user", "lives_in", "Boston", state_slot="residence_location", recorded_at="2025-01-01", valid_from="2025-01-01", valid_to="2025-03-01", status="superseded", evidence=[EvidenceRef("E1", "s1", [])])
    current = Claim("B", "User", "primary_user", "lives_in", "Tokyo", state_slot="residence_location", recorded_at="2025-03-15", valid_from="2025-03-01", status="active", evidence=[EvidenceRef("E2", "s2", [])])
    store.add_claim(old, [1.0, 0.0])
    store.add_claim(current, [1.0, 0.0])
    retriever = EventStateRetriever(store, Embedder(), retrieve_episodes=False, claim_top_k=10, candidate_count=10, evidence_count=2)
    selected, extra = retriever.retrieve("Where was the user living as of 2025-02-15?")
    assert [item["id"] for item in selected] == ["A"]
    assert extra["temporal_historical_state_candidate_count"] == 1
    assert extra["temporal_future_state_filtered_count"] == 1
    assert old.status == "superseded" and current.status == "active"


def test_non_temporal_retrieval_does_not_activate_temporal_channel():
    store = EventStateStore("ctx")
    for episode, vector in (_episode("A", "2025-01-01", [1.0, 0.0]), _episode("B", "2025-03-15", [0.0, 1.0])):
        store.add_episode(episode, vector)
    config = dict(retrieve_claims=False, episode_top_k=2, candidate_count=2, evidence_count=2)
    selected_a, extra_a = EventStateRetriever(store, Embedder(), **config, temporal_retrieval_enabled=False).retrieve("Where does the user live?")
    selected_b, extra_b = EventStateRetriever(store, Embedder(), **config, temporal_retrieval_enabled=True).retrieve("Where does the user live?")
    assert [item["id"] for item in selected_a] == [item["id"] for item in selected_b]
    assert [(item["fusion_score"], item["final_score"]) for item in selected_a] == [(item["fusion_score"], item["final_score"]) for item in selected_b]
    assert extra_b["temporal_constraint_detected"] is False


def test_planner_merge_recomputes_final_score_and_clears_stale_selection_metadata():
    store = EventStateStore("ctx")
    retriever = EventStateRetriever(
        store,
        Embedder(),
        planner_merge_mode="coverage_interleave",
        candidate_count=3,
        evidence_count=1,
        selector_mode="topk",
    )
    merged = retriever.merge_rank_channels([
        [{"id": "A", "type": "episode", "final_score": 0.99, "selection_score": 0.99, "selected_rank": 7}],
        [{"id": "C", "type": "episode", "final_score": 0.01, "selection_score": -3.0, "selected_rank": 8}],
        [{"id": "B", "type": "episode", "final_score": 0.5}],
    ])
    by_id = {item["id"]: item for item in merged}
    assert by_id["A"]["final_score"] == 0.99
    assert by_id["C"]["final_score"] == 0.01
    assert "selection_score" not in by_id["A"] and "selected_rank" not in by_id["A"]
    selected, _ = retriever.select_candidates(merged)
    assert selected[0]["id"] == "A"


def test_ppr_is_bounded_and_conserves_personalized_mass():
    store = EventStateStore("ctx")
    store.episodes = {
        "E1": Episode("E1", "ctx", 1, 0, None, None, ["User"], "primary_user", "", "one"),
        "E2": Episode("E2", "ctx", 2, 1, None, None, ["User"], "primary_user", "", "two"),
        "E3": Episode("E3", "ctx", 3, 2, None, None, ["User"], "primary_user", "", "three"),
    }
    store.episode_embeddings = {key: [1.0, 0.0] for key in store.episodes}
    store.add_edge("E1", "E2", "EPISODE_SUPPORTS_CLAIM")
    store.add_edge("E2", "E3", "EPISODE_SUPPORTS_CLAIM")
    retriever = EventStateRetriever(store, Embedder(), ppr_expand_hops=1, ppr_max_iterations=100, ppr_tolerance=1e-12)
    candidates = [
        {"id": "E1", "type": "episode", "score": 0.5},
        {"id": "E2", "type": "episode", "score": 0.25},
    ]
    result = retriever._ppr_impl(candidates)
    assert {item["id"] for item in result} == {"E1", "E2", "E3"}
    assert abs(sum(item["ppr_score"] for item in result) - 1.0) < 1e-9


def test_ppr_expansion_does_not_escape_hop_bound():
    store = EventStateStore("ctx")
    store.claims = {key: SimpleNamespace() for key in ("C1", "C2", "C3")}
    store.claim_embeddings = {key: [1.0, 0.0] for key in store.claims}
    store.add_edge("C1", "C2", "REFINES")
    store.add_edge("C2", "C3", "REFINES")
    retriever = EventStateRetriever(store, Embedder(), ppr_expand_hops=1)
    result = retriever._ppr_impl([{"id": "C1", "type": "state_claim", "score": 1.0}])
    assert {item["id"] for item in result} == {"C1", "C2"}


def test_ppr_follows_claim_episode_claim_and_excludes_disconnected_component():
    store = EventStateStore("ctx")
    store.claims = {key: SimpleNamespace() for key in ("C1", "C2", "C99")}
    store.episodes = {key: SimpleNamespace(source_session_id=key) for key in ("E1", "E99")}
    store.claim_embeddings = {key: [1.0, 0.0] for key in store.claims}
    store.episode_embeddings = {key: [1.0, 0.0] for key in store.episodes}
    store.add_edge("C1", "E1", "CLAIM_SUPPORTED_BY_EPISODE")
    store.add_edge("E1", "C2", "EPISODE_SUPPORTS_CLAIM")
    store.add_edge("C99", "E99", "CLAIM_SUPPORTED_BY_EPISODE")
    retriever = EventStateRetriever(store, Embedder(), ppr_expand_hops=2)
    result = retriever._ppr_impl([{"id": "C1", "type": "state_claim", "score": 1.0}])
    assert {item["id"] for item in result} == {"C1", "E1", "C2"}


def test_retrieval_hides_superseded_state_versions_but_keeps_history_and_reports_statuses():
    store = EventStateStore("ctx")
    store.claims = {
        "OLD": Claim("OLD", "Alice", "alice", "lives_in", "Boston", persistence="state", status="superseded", evidence=[EvidenceRef("E1", "s1", ["1"])]),
        "CURRENT": Claim("CURRENT", "Alice", "alice", "lives_in", "Tokyo", persistence="state", status="active", evidence=[EvidenceRef("E2", "s2", ["1"])]),
        "HISTORY": Claim("HISTORY", "Alice", "alice", "worked_at", "Acme", persistence="history", status="standalone", evidence=[EvidenceRef("E3", "s3", ["1"])]),
    }
    store.claim_embeddings = {key: [1.0, 0.0] for key in store.claims}
    store.add_edge("CURRENT", "OLD", "SUPERSEDES")
    retriever = EventStateRetriever(store, Embedder(), claim_top_k=10, episode_top_k=0, candidate_count=10, evidence_count=10, retrieve_episodes=False)
    selected, extra = retriever.retrieve("where does Alice live")
    ids = {item["id"] for item in selected}
    assert "CURRENT" in ids and "HISTORY" in ids and "OLD" not in ids
    assert extra["hidden_prior_state_candidate_count"] == 1
    assert extra["claim_candidate_status_counts"]["active"] == 1
