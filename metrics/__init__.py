"""Evaluation metrics module."""

from typing import Dict, List, Any, Type, Optional

from .base import BaseMetric, MetricResult
from .string_match import StringContainMetric, ExactMatchMetric, OptionMatchMetric
from .llm_judge import LLMJudgeMetric, LLMJudgeMCDMetric
from .locomo_metrics import LoCoMoF1Metric, LoCoMoAdversarialMetric, LoCoMoTemporalMetric
from .longmemeval_metrics import LongMemEvalJudgeMetric
from .ama_bench_metrics import AMABenchJudgeMetric
from .longmemeval_v2_metrics import LongMemEvalV2Metric


METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "string_contain": StringContainMetric,
    "exact_match": ExactMatchMetric,
    "option_match": OptionMatchMetric,
    "llm_judge": LLMJudgeMetric,
    "llm_judge_mcd": LLMJudgeMCDMetric,
    "locomo_f1": LoCoMoF1Metric,
    "locomo_adversarial": LoCoMoAdversarialMetric,
    "locomo_temporal": LoCoMoTemporalMetric,
    "longmemeval_judge": LongMemEvalJudgeMetric,
    "ama_bench_judge": AMABenchJudgeMetric,
    "longmemeval_v2": LongMemEvalV2Metric,
}


class MetricsCalculator:
    """Metrics calculator - auto selects metric based on query_type."""

    DEFAULT_METRIC_MAPPING = {
        "entity_exact_match": "string_contain",
        "temporal_localization": "llm_judge",
        "state_update": "llm_judge",
        "multiple_choice": "option_match",
        "inference_generation": "llm_judge",
        "multi_hop_clinical_deduction": "llm_judge_mcd",
        "single_hop": "locomo_f1",
        "multi_hop": "locomo_f1",
        "temporal": "locomo_f1",
        "open_domain": "locomo_f1",
        "adversarial": "locomo_adversarial",
        # LongMemEval types
        "single-session-user": "longmemeval_judge",
        "single-session-assistant": "longmemeval_judge",
        "single-session-preference": "longmemeval_judge",
        "multi-session": "longmemeval_judge",
        "temporal-reasoning": "longmemeval_judge",
        "knowledge-update": "longmemeval_judge",
        # AMA-Bench QA types (all use binary yes/no LLM-as-Judge)
        "recall": "ama_bench_judge",
        "causal_inference": "ama_bench_judge",
        "state_updating": "ama_bench_judge",
        "state_abstraction": "ama_bench_judge",
        # LongMemEval-V2 types
        "static-environment": "longmemeval_v2",
        "static-environment-abs": "longmemeval_v2",
        "dynamic-environment": "longmemeval_v2",
        "dynamic-environment-abs": "longmemeval_v2",
        "procedure": "longmemeval_v2",
        "procedure-abs": "longmemeval_v2",
        "errors-gotchas": "longmemeval_v2",
    }

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None, dataset: str = "medmemorybench",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self.metric_mapping = self.DEFAULT_METRIC_MAPPING.copy()
        if custom_mapping:
            self.metric_mapping.update(custom_mapping)
        self._metric_instances: Dict[str, BaseMetric] = {}
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._language = language

    def _get_metric(self, metric_name: str) -> BaseMetric:
        if metric_name not in self._metric_instances:
            metric_class = METRIC_REGISTRY.get(metric_name)
            if metric_class is None:
                raise ValueError(f"Unknown metric: {metric_name}, available: {list(METRIC_REGISTRY.keys())}")
            if metric_name in ("llm_judge", "llm_judge_mcd", "longmemeval_judge", "ama_bench_judge"):
                self._metric_instances[metric_name] = metric_class(
                    dataset=self._dataset,
                    judge_model=self._judge_model,
                    judge_api_key=self._judge_api_key,
                    judge_base_url=self._judge_base_url,
                    language=self._language,
                )
            else:
                self._metric_instances[metric_name] = metric_class()
        return self._metric_instances[metric_name]

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        metric_name: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        # Determine metric
        if metric_name is None:
            metric_name = self.metric_mapping.get(query_type, "string_contain")
        metric = self._get_metric(metric_name)
        return metric.compute(
            query_id=query_id,
            query_type=query_type,
            model_output=model_output,
            expected_answers=expected_answers,
            question=question,
            **kwargs
        )


class MetricsAggregator:
    """Aggregates evaluation results."""

    def __init__(self):
        self.results: List[MetricResult] = []

    def add_result(self, result: MetricResult) -> None:
        self.results.append(result)

    def add_results(self, results: List[MetricResult]) -> None:
        self.results.extend(results)

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"total": 0, "message": "No results"}

        total = len(self.results)
        correct_count = sum(1 for r in self.results if r.is_correct)
        total_score = sum(r.score for r in self.results)

        by_type: Dict[str, List[MetricResult]] = {}
        for result in self.results:
            if result.query_type not in by_type:
                by_type[result.query_type] = []
            by_type[result.query_type].append(result)

        type_stats = {}
        for query_type, type_results in by_type.items():
            type_total = len(type_results)
            type_correct = sum(1 for r in type_results if r.is_correct)
            type_score = sum(r.score for r in type_results)

            stats = {
                "total": type_total,
                "correct": type_correct,
                "accuracy": type_correct / type_total if type_total > 0 else 0.0,
                "avg_score": type_score / type_total if type_total > 0 else 0.0,
            }

            # MCD type extra stats
            if query_type == "multi_hop_clinical_deduction":
                ncr_scores = [r.details.get("ncr_score", 0.0) for r in type_results]
                crc_scores = [r.details.get("crc_score", 0.0) for r in type_results]
                cc_scores = [r.details.get("cc_score", 0.0) for r in type_results]

                stats["avg_ncr"] = sum(ncr_scores) / len(ncr_scores) if ncr_scores else 0.0
                stats["avg_crc"] = sum(crc_scores) / len(crc_scores) if crc_scores else 0.0
                stats["avg_cc"] = sum(cc_scores) / len(cc_scores) if cc_scores else 0.0

                total_nodes_validated = 0
                total_nodes_mentioned = 0
                total_causal_correct = 0
                for r in type_results:
                    node_validations = r.details.get("node_validations", [])
                    for nv in node_validations:
                        total_nodes_validated += 1
                        if nv.get("mentioned", False):
                            total_nodes_mentioned += 1
                        if nv.get("causal_link_correct", False):
                            total_causal_correct += 1

                if total_nodes_validated > 0:
                    stats["node_mention_rate"] = total_nodes_mentioned / total_nodes_validated
                    stats["node_causal_rate"] = total_causal_correct / total_nodes_validated
                    stats["total_nodes_validated"] = total_nodes_validated

            type_stats[query_type] = stats

        total_memory_time = sum(r.memory_construction_time for r in self.results)
        total_query_time = sum(r.query_time for r in self.results)

        efficiency_stats = {
            "total_memory_construction_time": total_memory_time,
            "total_query_time": total_query_time,
            "avg_memory_construction_time": total_memory_time / total if total > 0 else 0.0,
            "avg_query_time": total_query_time / total if total > 0 else 0.0,
        }

        return {
            "total": total,
            "correct": correct_count,
            "overall_accuracy": correct_count / total if total > 0 else 0.0,
            "overall_avg_score": total_score / total if total > 0 else 0.0,
            "by_type": type_stats,
            "efficiency": efficiency_stats,
        }

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def clear(self) -> None:
        self.results.clear()


__all__ = [
    "BaseMetric",
    "MetricResult",
    "StringContainMetric",
    "ExactMatchMetric",
    "OptionMatchMetric",
    "LLMJudgeMetric",
    "LLMJudgeMCDMetric",
    "LoCoMoF1Metric",
    "LoCoMoAdversarialMetric",
    "LoCoMoTemporalMetric",
    "LongMemEvalJudgeMetric",
    "AMABenchJudgeMetric",
    "LongMemEvalV2Metric",
    "MetricsCalculator",
    "MetricsAggregator",
    "METRIC_REGISTRY",
]
