"""Regression coverage for immutable Event-State turn retrieval."""

from datetime import date
from types import SimpleNamespace

from benchmarks.locomo.dataset import LoCoMoQuery
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from methods.event_state.schemas import (
    Claim,
    EPISODE_RETRIEVAL_MAX_CHARS,
    EvidenceRef,
    Episode,
    TurnEvidence,
    select_episode_retrieval_turns,
)
from methods.event_state.store import EventStateStore
from methods.event_state.temporal import parse_temporal_query
from methods.event_state.retrieval import EventStateRetriever
from methods.event_state_agent import EventStateAgent
from src.config import ConfigLoader, MethodConfig


class KeywordEmbedder:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "needle" in text.casefold() else [0.0, 1.0]


class EmptyExtractionLLM:
    def chat(self, messages, **kwargs):
        if "Extract conversational memory" in messages[0]["content"]:
            return SimpleNamespace(content='{"episode_summary":"summary", "claims":[]}')
        return SimpleNamespace(content="answer")


class ClaimedNeedleLLM:
    def chat(self, messages, **kwargs):
        if "Extract conversational memory" in messages[0]["content"]:
            return SimpleNamespace(content=(
                '{"episode_summary":"summary", "claims":[{"subject":"A",'
                '"predicate":"detail","value":"needle","source_turn_ids":["t"]}]}'
            ))
        return SimpleNamespace(content='{"operation":"NEW","confidence":1}')


def test_bm25_sentence_chunks_return_immutable_parent_turn_and_cache_once():
    store = EventStateStore("bm25")
    long_turn = TurnEvidence("long", "A", "user", "common " * 80 + "rareneedle detail.")
    short_turn = TurnEvidence("short", "A", "user", "common common common")
    episode = Episode("E", "bm25", "session", 0, None, None, ["A"], "primary_user", "", "summary", [long_turn, short_turn])
    store.add_episode(episode, [0.0, 1.0], [[0.0, 1.0], [0.0, 1.0]])
    retriever = EventStateRetriever(store, KeywordEmbedder(), turn_lexical_mode="bm25", turn_sparse_unit="sentence_chunk", turn_top_k=2)
    first = retriever._lexical_turn_rank("rareneedle")
    cache = next(iter(store._turn_bm25_caches.values()))
    second = retriever._lexical_turn_rank("rareneedle")
    assert first == second == [(store.turn_key("E", 0), first[0][1])]
    assert next(iter(store._turn_bm25_caches.values())) is cache


def test_source_coherent_selector_keeps_dominant_session_without_hard_filter():
    agent, store, keys = _selection_fixture(("a", "b", "c"))
    store.turn_metadata[keys["b"]]["source_session_id"] = "session-a"
    retriever = EventStateRetriever(store, KeywordEmbedder(), selector_mode="source_coherent_mmr", evidence_count=2,
                                    source_coherence_score_ratio=.7, source_coherence_bonus=.2)
    candidates = [
        {"id": keys["a"], "type": "turn", "final_score": .99},
        {"id": keys["b"], "type": "turn", "final_score": .95},
        {"id": keys["c"], "type": "turn", "final_score": .2},
    ]
    selected = retriever._select(candidates, 2)
    assert [item["id"] for item in selected] == [keys["a"], keys["b"]]


def _selection_fixture(turn_ids=("x", "y", "z", "w")):
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), evidence_count=3,
        max_source_excerpts_per_claim=2,
    )
    store = EventStateStore("ctx")
    keys = {}
    for index, turn_id in enumerate(turn_ids):
        episode = Episode(
            f"E{turn_id}", "ctx", f"session-{turn_id}", index, None, None,
            ["A"], "primary_user", "", f"episode-{turn_id}",
            [TurnEvidence(turn_id, "A", "user", f"needle {turn_id}")],
        )
        store.add_episode(episode, [1.0, 0.0], [[1.0, 0.0]])
        keys[turn_id] = store.turn_key(episode.episode_id, 0)
    return agent, store, keys


def _claim_for_turns(turn_ids):
    evidence = [
        EvidenceRef(f"E{turn_id}", f"session-{turn_id}", [turn_id])
        for turn_id in turn_ids
    ]
    return Claim("C", "A", "speaker:a", "detail", "needle", evidence=evidence)


def _budget_fixture(
    *,
    structured_count=4,
    turn_count=4,
    evidence_count=3,
    turn_evidence_count=0,
    retrieve_turns=True,
    turn_lexical_retrieval_enabled=False,
):
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), evidence_count=evidence_count,
        turn_evidence_count=turn_evidence_count, candidate_count=max(20, structured_count * 2),
        turn_top_k=turn_count, retrieve_turns=retrieve_turns,
        turn_lexical_retrieval_enabled=turn_lexical_retrieval_enabled,
        inject_source_evidence=False,
    )
    store = EventStateStore("budget")
    for index in range(max(structured_count, turn_count)):
        episode = Episode(
            f"E{index}", "budget", f"session-{index}", index, None, None,
            ["A"], "primary_user", "", f"episode {index}",
            [TurnEvidence(f"t{index}", "A", "user", f"needle turn {index}")]
            if index < turn_count else [],
        )
        store.add_episode(
            episode,
            [1.0, 0.0],
            [[1.0, 0.0]] if index < turn_count else [],
        )
        if index < structured_count:
            store.add_claim(
                Claim(f"C{index}", "A", "speaker:a", "detail", f"needle {index}"),
                [1.0, 0.0],
            )
    agent.set_context_id("budget")
    agent._stores["budget"] = store
    return agent, store


def test_zero_turn_budget_reproduces_the_shared_selection_pipeline():
    agent, store = _budget_fixture(evidence_count=3, turn_evidence_count=0)
    retriever = EventStateRetriever(store, KeywordEmbedder(), **agent._retrieval_config)

    legacy_ranked, legacy_extra = retriever.rank_candidates("needle", [1.0, 0.0])
    pooled_ranked, turn_ranked, pooled_extra = retriever.rank_candidate_pools(
        "needle", [1.0, 0.0],
    )
    assert pooled_ranked == legacy_ranked
    assert turn_ranked == []
    assert pooled_extra == legacy_extra
    legacy_order, legacy_extra = retriever.select_candidates(
        legacy_ranked, legacy_extra, count=len(legacy_ranked),
    )
    legacy_selected, legacy_effective = agent._effective_evidence_selection(
        store, legacy_order, [[1.0, 0.0]],
    )
    prepared = agent.prepare_batch_query("needle", raw_question="needle")

    assert [record["id"] for record in prepared["retrieved_memories"]] == [
        item["id"] for item in legacy_selected if item["type"] != "turn"
    ] + [item["id"] for item in legacy_selected if item["type"] == "turn"]
    assert prepared["extra"]["effective_selected_turn_count"] == legacy_effective[
        "effective_selected_turn_count"
    ]
    assert "separate_turn_evidence_budget" not in prepared["extra"]


def test_separate_turn_budget_keeps_twelve_structured_items_and_four_turns():
    agent, _store = _budget_fixture(
        structured_count=12,
        turn_count=4,
        evidence_count=12,
        turn_evidence_count=4,
    )
    prepared = agent.prepare_batch_query("needle", raw_question="needle")

    assert prepared["extra"]["evidence_count"] == 12
    assert prepared["extra"]["turn_evidence_count"] == 4
    assert prepared["extra"]["separate_turn_evidence_budget"] is True
    assert prepared["extra"]["selected_non_turn_count"] == 12
    assert prepared["extra"]["selected_direct_turn_count"] == 4
    assert prepared["extra"]["selected_total_count"] == 16
    assert len(prepared["retrieved_memories"]) == 16


def test_separate_direct_turns_do_not_reduce_structured_selection():
    without_turns, _ = _budget_fixture(
        structured_count=6, turn_count=4, evidence_count=3, retrieve_turns=False,
    )
    with_turns, _ = _budget_fixture(
        structured_count=6, turn_count=4, evidence_count=3, turn_evidence_count=2,
    )
    structured_only = without_turns.prepare_batch_query("needle", raw_question="needle")
    separate = with_turns.prepare_batch_query("needle", raw_question="needle")

    assert [record["id"] for record in separate["retrieved_memories"] if record["type"] != "immutable_turn"] == [
        record["id"] for record in structured_only["retrieved_memories"]
    ]
    assert separate["extra"]["selected_non_turn_count"] == 3
    assert separate["extra"]["selected_direct_turn_count"] == 2


def test_separate_turn_budget_returns_only_available_turns_and_honors_retrieve_flag():
    available, _ = _budget_fixture(
        structured_count=3, turn_count=2, evidence_count=3, turn_evidence_count=4,
    )
    disabled, _ = _budget_fixture(
        structured_count=3, turn_count=4, evidence_count=3, turn_evidence_count=4,
        retrieve_turns=False,
    )

    available_prepared = available.prepare_batch_query("needle", raw_question="needle")
    disabled_prepared = disabled.prepare_batch_query("needle", raw_question="needle")

    assert available_prepared["extra"]["selected_direct_turn_count"] == 2
    assert available_prepared["extra"]["selected_total_count"] == 5
    assert disabled_prepared["extra"]["selected_direct_turn_count"] == 0
    assert disabled_prepared["extra"]["selected_total_count"] == 3


def test_separate_turn_budget_fuses_dense_and_lexical_turns_before_its_cap():
    agent, store = _budget_fixture(
        structured_count=0,
        turn_count=4,
        evidence_count=1,
        turn_evidence_count=2,
        turn_lexical_retrieval_enabled=True,
    )
    retriever = EventStateRetriever(store, KeywordEmbedder(), **agent._retrieval_config)
    _structured, fused_turns, _diagnostics = retriever.rank_candidate_pools(
        "needle", [1.0, 0.0],
    )
    expected_turns, _expected_diagnostics = retriever.rank_candidates(
        "needle", [1.0, 0.0],
        retrieve_claims_override=False,
        retrieve_episodes_override=False,
    )
    prepared = agent.prepare_batch_query("needle", raw_question="needle")

    assert fused_turns == expected_turns
    assert prepared["extra"]["turn_candidate_count"] == 4
    assert prepared["extra"]["lexical_turn_candidates"] == 4
    assert prepared["extra"]["selected_direct_turn_count"] == 2
    direct_ids = [
        record["id"] for record in prepared["retrieved_memories"]
        if record["type"] == "immutable_turn"
    ]
    assert len(direct_ids) == len(set(direct_ids)) == 2


def test_separate_budget_regression_expands_total_without_expanding_structured_budget():
    shared, _ = _budget_fixture(
        structured_count=4, turn_count=4, evidence_count=3, turn_evidence_count=0,
    )
    separate, _ = _budget_fixture(
        structured_count=4, turn_count=4, evidence_count=3, turn_evidence_count=2,
    )
    shared_prepared = shared.prepare_batch_query("needle", raw_question="needle")
    separate_prepared = separate.prepare_batch_query("needle", raw_question="needle")

    assert len(shared_prepared["retrieved_memories"]) <= 3
    assert separate_prepared["extra"]["selected_non_turn_count"] <= 3
    assert separate_prepared["extra"]["selected_direct_turn_count"] <= 2
    assert len(separate_prepared["retrieved_memories"]) <= 5
    assert len(separate_prepared["retrieved_memories"]) == 5


def test_turn_evidence_count_validation_and_omitted_config_default():
    try:
        EventStateAgent(turn_evidence_count=-1, embedding_client=KeywordEmbedder())
    except ValueError as exc:
        assert "turn_evidence_count" in str(exc)
    else:
        raise AssertionError("negative turn_evidence_count must fail validation")

    config = MethodConfig.from_dict({
        "method_name": "event_state",
        "method_type": "agentic_memory",
        "model": {"provider": "openai", "name": "test"},
        "retrieval_config": {"evidence_count": 3, "candidate_count": 3},
    })
    assert config.retrieval_config["turn_evidence_count"] == 0
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), **config.agent_params,
    )
    assert agent._retrieval_config["turn_evidence_count"] == 0


def test_episode_retrieval_sampling_spans_source_order_and_keeps_captions():
    turns = [
        TurnEvidence(
            turn_id=f"t{index}", speaker="A", role="user",
            text=f"turn-{index}", image_caption="final-caption" if index == 19 else None,
        )
        for index in range(20)
    ]
    sampled = select_episode_retrieval_turns(turns)
    assert [turn.turn_id for turn in select_episode_retrieval_turns(turns[:8])] == [f"t{index}" for index in range(8)]
    assert sampled[0].turn_id == "t0"
    assert sampled[-1].turn_id == "t19"
    assert any(turn.turn_id in {"t9", "t10"} for turn in sampled)
    assert [turn.turn_id for turn in sampled] == sorted(
        (turn.turn_id for turn in sampled), key=lambda value: int(value[1:])
    )
    assert [turn.turn_id for turn in sampled] == [turn.turn_id for turn in select_episode_retrieval_turns(turns)]

    episode = Episode("E", "ctx", "s", 0, None, None, [], None, "", "summary", turns)
    assert "final-caption" in episode.retrieval_text()
    oversized = Episode("E2", "ctx", "s2", 1, None, None, [], None, "", "summary", [
        TurnEvidence(str(index), "A", "user", "x" * 1000) for index in range(20)
    ])
    assert len(oversized.retrieval_text()) <= EPISODE_RETRIEVAL_MAX_CHARS


def test_temporal_parser_accepts_comma_day_month_dates_conservatively():
    assert parse_temporal_query("What happened on 1 February, 2023?").target_date == date(2023, 2, 1)
    assert parse_temporal_query("What happened on 1 February 2023?").target_date == date(2023, 2, 1)
    assert parse_temporal_query("What happened on 13 Oct, 2023?").target_date == date(2023, 10, 13)
    assert parse_temporal_query("What happened on October 13, 2023?").target_date == date(2023, 10, 13)
    assert parse_temporal_query("What happened in February 2023?") is None
    assert parse_temporal_query("What happened last February?") is None


def test_global_turn_index_recovers_late_unclaimed_turn_and_snapshot_restore():
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), retrieve_claims=False,
        retrieve_episodes=False, retrieve_turns=True, candidate_count=4,
        evidence_count=1, max_episode_source_excerpts_total=2,
    )
    agent.set_context_id("ctx")
    agent.memorize("", memory_items=[
        {
            "source_session_id": "session-1", "source_turn_id": str(index),
            "speaker": "A", "content": "needle detail" if index == 19 else "ordinary discussion",
        }
        for index in range(20)
    ])
    prepared = agent.prepare_batch_query("Where was the needle detail?", raw_question="needle detail")
    selected = prepared["retrieved_memories"]
    assert selected[0]["type"] == "immutable_turn"
    assert selected[0]["source_turn_id"] == "19"
    assert selected[0]["source_session_id"] == "session-1"
    assert not agent._store().claims
    assert selected[0]["included_in_context"] is True
    assert prepared["extra"]["configured_global_raw_excerpt_budget"] == 2
    assert prepared["extra"]["selected_raw_turn_count"] == 1

    legacy_state = agent.export_memory_state()
    artifacts = agent.export_memory_binary_artifacts()
    legacy_state["schema_version"] = 4
    legacy_state.pop("embedding_artifacts")
    for name, descriptor in artifacts.items():
        legacy_state[name] = {
            identifier: descriptor["values"][row].tolist()
            for row, identifier in enumerate(descriptor["ids"])
        }
    legacy_state.pop("turn_embeddings")
    legacy_state.pop("turn_metadata")
    restored = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), retrieve_claims=False,
        retrieve_episodes=False, retrieve_turns=True, candidate_count=4,
        evidence_count=1, max_episode_source_excerpts_total=8,
    )
    restored.import_memory_state(legacy_state, context_id="ctx")
    restored_prepared = restored.prepare_batch_query("needle detail", raw_question="needle detail")
    assert restored_prepared["retrieved_memories"][0]["source_turn_id"] == "19"
    assert restored_prepared["extra"]["configured_global_raw_excerpt_budget"] == 8


def test_configured_raw_excerpt_budget_reaches_effective_agent():
    config = ConfigLoader().load_method_config("event_state_gemini")
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), **config.agent_params,
    )
    assert config.retrieval_config["max_episode_source_excerpts_total"] == 4
    assert agent.max_episode_source_excerpts_total == 4
    assert agent._retrieval_config["max_episode_source_excerpts_total"] == 4


def test_claim_provenance_and_direct_turn_retrieval_render_one_copy():
    agent = EventStateAgent(
        llm_client=ClaimedNeedleLLM(), memory_llm_client=ClaimedNeedleLLM(),
        embedding_client=KeywordEmbedder(), candidate_count=4, evidence_count=2,
        retrieve_turns=True,
    )
    agent.memorize("", memory_items=[{
        "source_session_id": "session-1", "source_turn_id": "t",
        "speaker": "A", "content": "needle detail",
    }])
    prepared = agent.prepare_batch_query("needle detail", raw_question="needle detail")
    records = prepared["retrieved_memories"]
    assert any(record["type"] == "state_claim" for record in records)
    assert not any(record["type"] == "immutable_turn" for record in records)
    claim = next(record for record in records if record["type"] == "state_claim")
    assert claim["included_provenance_evidence"][0]["evidence"]["source_turn_ids"] == ["t"]


def test_effective_selection_backfills_direct_turn_deduplicated_by_claim_provenance():
    agent, store, keys = _selection_fixture()
    store.add_claim(_claim_for_turns(["x"]), [1.0, 0.0])
    selection_order = [
        {"id": "C", "type": "state_claim"},
        {"id": keys["x"], "type": "turn"},
        {"id": keys["y"], "type": "turn"},
    ]
    agent._retrieval_config["evidence_count"] = 2

    selected, telemetry = agent._effective_evidence_selection(
        store, selection_order, [[1.0, 0.0]],
    )

    assert [item["id"] for item in selected] == ["C", keys["y"]]
    assert telemetry["effective_retrieved_record_count"] == 2
    assert telemetry["deduplicated_selected_turn_count"] == 1
    assert telemetry["backfilled_after_turn_dedup_count"] == 1
    assert telemetry["selected_turn_ids_before_dedup"] == [keys["x"], keys["y"]]
    assert telemetry["selected_turn_ids_after_dedup"] == [keys["y"]]


def test_effective_selection_backfills_multiple_claim_provenance_duplicates_deterministically():
    agent, store, keys = _selection_fixture()
    store.add_claim(_claim_for_turns(["x", "y"]), [1.0, 0.0])
    selection_order = [
        {"id": "C", "type": "state_claim"},
        {"id": keys["x"], "type": "turn"},
        {"id": keys["y"], "type": "turn"},
        {"id": keys["z"], "type": "turn"},
        {"id": keys["w"], "type": "turn"},
    ]

    first, telemetry = agent._effective_evidence_selection(store, selection_order, [[1.0, 0.0]])
    second, _ = agent._effective_evidence_selection(store, selection_order, [[1.0, 0.0]])

    assert [item["id"] for item in first] == ["C", keys["z"], keys["w"]]
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert telemetry["effective_retrieved_record_count"] == 3
    assert telemetry["deduplicated_turn_count"] == 2
    assert telemetry["backfilled_candidate_count"] == 2


def test_evidence_count_is_filled_after_provenance_deduplication_when_candidates_exist():
    turn_ids = tuple(f"t{index}" for index in range(10))
    agent, store, keys = _selection_fixture(turn_ids)
    store.add_claim(_claim_for_turns(["t0"]), [1.0, 0.0])
    selection_order = [{"id": "C", "type": "state_claim"}] + [
        {"id": keys[turn_id], "type": "turn"} for turn_id in turn_ids
    ]
    agent._retrieval_config["evidence_count"] = 8

    selected, telemetry = agent._effective_evidence_selection(store, selection_order, [[1.0, 0.0]])

    assert len(selected) == 8
    assert telemetry["effective_retrieved_record_count"] == 8
    assert keys["t0"] not in [item["id"] for item in selected]
    assert telemetry["backfilled_candidate_count"] == 1


def test_episode_object_is_not_collapsed_when_its_turn_duplicates_claim_provenance():
    agent, store, keys = _selection_fixture()
    store.add_claim(_claim_for_turns(["x"]), [1.0, 0.0])
    selection_order = [
        {"id": "C", "type": "state_claim"},
        {"id": "Ex", "type": "episode"},
        {"id": keys["x"], "type": "turn"},
        {"id": keys["y"], "type": "turn"},
    ]

    selected, telemetry = agent._effective_evidence_selection(store, selection_order, [[1.0, 0.0]])

    assert [item["id"] for item in selected] == ["C", "Ex", keys["y"]]
    assert telemetry["deduplicated_selected_turn_count"] == 1


def test_all_selector_modes_and_turn_source_identity_handle_turn_candidates():
    _agent, store, keys = _selection_fixture()
    store.add_claim(_claim_for_turns(["x"]), [1.0, 0.0])
    candidates = [
        {"id": "C", "type": "state_claim", "final_score": 1.0, "score": 1.0},
        {"id": "Ex", "type": "episode", "final_score": 0.8, "score": 0.8},
        {"id": keys["y"], "type": "turn", "final_score": 0.6, "score": 0.6},
    ]
    for selector_mode in ("topk", "mmr", "state_mmr"):
        retriever = EventStateRetriever(
            store, KeywordEmbedder(), selector_mode=selector_mode, evidence_count=3,
        )
        selected, _ = retriever.select_candidates(candidates)
        assert {item["id"] for item in selected} == {"C", "Ex", keys["y"]}
        assert retriever._source_ids(next(item for item in selected if item["type"] == "turn")) == {"session-y"}


def test_turn_rrf_channels_apply_their_independent_weights_once():
    _agent, store, keys = _selection_fixture()
    retriever = EventStateRetriever(
        store, KeywordEmbedder(), rrf_k=60.0,
        turn_retrieval_weight=2.0, turn_lexical_retrieval_weight=3.0,
    )
    fused = retriever._rrf([], [], [(keys["x"], 0.9)], [(keys["x"], 0.8)])

    assert len(fused) == 1
    assert abs(fused[0]["score"] - 5.0 / 61.0) < 1e-12


def test_locomo_stage_telemetry_remains_evaluator_only():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    query = LoCoMoQuery(
        query_id="q", question="", query_type="single_hop", expected_answers=[""],
        evidence=["D2:4"],
    )
    quality = evaluator._locomo_retrieval_quality(
        query,
        [
            {
                "type": "episode", "source_session_id": 2,
                "episode_archive_turn_ids": ["4"], "included_in_context": False,
                "episode_evidence_turn_ids": [],
            },
            {
                "type": "immutable_turn", "source_session_id": 2,
                "episode_evidence_turn_ids": ["4"], "included_in_context": True,
            },
        ],
        {
            "retrieval_stage_candidates": {
                "pre_candidate_truncation_fused": [{"source_session_ids": ["1"]}],
                "post_candidate_count": [{"source_session_ids": ["2"]}],
                "final_memory_object_selection": [{"source_session_ids": ["2"]}],
            }
        },
    )
    assert quality["pre_candidate_truncation_fused_session"]["hit"] is False
    assert quality["post_candidate_count_session"]["hit"] is True
    assert quality["selected_memory_session"]["hit"] is True
    assert quality["selected_episode_archive_exact_turn"]["hit"] is True
    assert quality["answer_visible_exact_turn"]["hit"] is True
    assert quality["answer_visible_gold_turn_via_direct_turn_count"] == 1
    assert quality["answer_visible_gold_turn_via_episode_excerpt_count"] == 0


def test_compiler_stage_telemetry_uses_semantic_stage_names():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    query = LoCoMoQuery(
        query_id="q", question="", query_type="single_hop", expected_answers=[""],
        evidence=["D2:4"],
    )
    quality = evaluator._locomo_retrieval_quality(
        query, [],
        {"retrieval_stage_candidates": {
            "channel_semantic_candidates": [{"source_session_ids": ["1"]}],
            "merged_semantic_union": [{"source_session_ids": ["2"]}],
            "temporally_reranked_union": [{"source_session_ids": ["2"]}],
            "final_memory_object_selection": [{"source_session_ids": ["2"]}],
        }},
    )
    assert quality["channel_semantic_candidates_session"]["hit"] is False
    assert quality["merged_semantic_union_session"]["hit"] is True
    assert quality["temporally_reranked_union_session"]["hit"] is True
    assert quality["final_memory_object_selection_session"]["hit"] is True
    assert "post_candidate_count_session" not in quality
