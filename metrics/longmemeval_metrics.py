"""LongMemEval LLM-as-Judge metric - follows original LongMemEval evaluation logic.

Uses the same yes/no prompt format and scoring as the original LongMemEval paper
(evaluate_qa.py) to ensure comparable results.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from .base import BaseMetric, MetricResult
from utils.templates import get_prompt_manager

logger = logging.getLogger(__name__)


class LongMemEvalJudgeMetric(BaseMetric):
    """LLM-as-Judge metric for LongMemEval using original evaluation prompts.

    Supports all 6 question types + abstention detection.
    Uses binary yes/no scoring consistent with the original LongMemEval paper.
    """

    NAME = "longmemeval_judge"

    def __init__(self, dataset: str = "longmemeval",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "en"):
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._language = language
        self._prompt_manager = get_prompt_manager(dataset, language=language)
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        if not self._initialized:
            from utils.llm_client import create_llm_client
            from src.config import get_api_config

            api_config = get_api_config()
            model = self._judge_model or api_config.get_judge_model()
            api_key = self._judge_api_key or api_config.get_judge_api_key()
            base_url = self._judge_base_url or api_config.get_judge_base_url()

            self._client = create_llm_client(
                provider="openai",
                model=model,
                temperature=0.0,
                max_tokens=10,
                api_key=api_key,
                base_url=base_url,
            )
            self._initialized = True

    def _call_judge(self, prompt: str) -> Optional[Dict[str, Any]]:
        self._ensure_client()
        try:
            response = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            result_text = response.content.strip().lower()

            is_correct = bool(re.search(r'\byes\b', result_text))
            return {"is_correct": is_correct, "reason": response.content.strip()}

        except Exception as e:
            logger.error(f"[LongMemEvalJudge] API call failed: {e}")
            return None

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        is_abstention: bool = False,
        **kwargs
    ) -> MetricResult:
        expected_answer = expected_answers[0] if expected_answers else ""

        if not model_output or not model_output.strip():
            return MetricResult(
                query_id=query_id,
                query_type=query_type,
                score=0.0,
                is_correct=False,
                model_output=model_output,
                expected_answer=expected_answer,
                question=question,
                details={"reason": "Empty model output", "metric": self.NAME},
            )

        if is_abstention:
            judge_key = "abstention"
        else:
            judge_key = query_type

        prompt = self._prompt_manager.format_judge(
            query_type=judge_key,
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
        )

        result = self._call_judge(prompt)

        if result:
            is_correct = result.get("is_correct", False)
            reason = result.get("reason", "")
        else:
            is_correct = False
            reason = "Judge call failed"

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            model_output=model_output,
            expected_answer=expected_answer,
            question=question,
            details={
                "reason": reason,
                "is_abstention": is_abstention,
                "metric": self.NAME,
            },
        )
