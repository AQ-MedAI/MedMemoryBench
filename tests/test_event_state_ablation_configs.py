"""Guard the controlled Event-State retrieval ablation matrix."""

from pathlib import Path

from benchmarks.medmemorybench.checkpoint import compute_build_config_hash
from src.config import ConfigLoader


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "method_config" / "test"


def test_event_state_gemini_ablation_configs_are_full_and_controlled():
    loader = ConfigLoader()
    dataset = loader.load_dataset_config("locomo_1")
    configs = [
        loader.load_method_config(str(CONFIG_DIR / f"event_state_gemini{number}.yaml"))
        for number in range(1, 11)
    ]
    baseline = configs[0]
    fixed_retrieval = {
        "planner_mode": "off",
        "query_manager_reuse_mode": "shared",
        "query_memory_guard_enabled": True,
        "local_reranker_mode": "off",
        "evidence_budget_mode": "fixed",
    }
    for config in configs:
        assert config.method_name.lower() == "event_state"
        assert config.model == baseline.model
        assert config.memorize_model == baseline.memorize_model
        assert config.embedding == baseline.embedding
        assert config.build_config == baseline.build_config
        assert config.retrieval_config["evidence_count"] == baseline.retrieval_config["evidence_count"]
        assert config.retrieval_config["turn_evidence_count"] == baseline.retrieval_config["turn_evidence_count"]
        for key in ("claim_top_k", "episode_top_k", "turn_top_k", "candidate_count", "max_context_tokens"):
            assert config.retrieval_config[key] == baseline.retrieval_config[key]
        for key, value in fixed_retrieval.items():
            assert config.retrieval_config[key] == value

    retrieval = [config.retrieval_config for config in configs]
    assert retrieval[0]["turn_lexical_mode"] == "overlap"
    assert retrieval[1]["turn_lexical_mode"] == "bm25"
    assert retrieval[2]["episode_relevance_mode"] == "summary_plus_best_turn"
    assert retrieval[3]["source_diversity_bonus"] == 0.0
    assert retrieval[4]["mmr_lambda"] == 0.85
    assert retrieval[5]["selector_mode"] == "source_coherent_mmr"
    assert retrieval[5]["source_coherence_claim_source_mode"] == "query_relevant"
    assert retrieval[6]["episode_excerpt_mode"] == "joint"
    assert retrieval[7]["fusion_mode"] == "weighted_score"
    assert retrieval[7]["score_calibration_mode"] == "minmax"
    assert retrieval[8]["temporal_query_mode"] == "dual_axis_semantic"
    assert retrieval[9]["selector_mode"] == "source_coherent_mmr"
    assert retrieval[9]["episode_excerpt_mode"] == "joint"

    hashes = {compute_build_config_hash(config, dataset) for config in configs}
    assert len(hashes) == 1
