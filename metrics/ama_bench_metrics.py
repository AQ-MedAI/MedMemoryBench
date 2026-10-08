"""AMA-Bench LLM-as-Judge metric.

Uses binary yes/no scoring matching the official AMA-Bench evaluation:
the judge is asked "Is the predicted answer correct?" and responds with
only "yes" or "no". The last occurrence of yes/no is taken as the verdict.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from .base import BaseMetric, MetricResult
from utils.templates import get_prompt_manager

logger = logging.getLogger(__name__)


class AMABenchJudgeMetric(BaseMetric):
    """LLM-as-Judge metric for AMA-Bench with binary yes/no scoring."""

    NAME = "ama_bench_judge"

    def __init__(
        self,
        dataset: str = "ama_bench",
        judge_model: str = None,
        judge_api_key: str = None,
        judge_base_url: str = None,
        language: str = "en",
    ):
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
                max_tokens=2048,
                api_key=api_key,
                base_url=base_url,
            )
            self._initialized = True

    def _call_judge(self, prompt: str) -> Optional[Dict[str, Any]]:
        self._ensure_client()
        try:
            response = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            result_text = response.content.strip()

            # Remove thinking tags if present
            cleaned = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.strip()
            result_lower = cleaned.lower()

            # Find last occurrences of "yes" and "no" as complete words
            yes_matches = list(re.finditer(r'\byes\b', result_lower))
            no_matches = list(re.finditer(r'\bno\b', result_lower))

            last_yes_pos = yes_matches[-1].start() if yes_matches else -1
            last_no_pos = no_matches[-1].start() if no_matches else -1

            if last_yes_pos > last_no_pos:
                return {"is_correct": True, "reason": cleaned}
            elif last_no_pos > last_yes_pos:
                return {"is_correct": False, "reason": cleaned}
            else:
                logger.warning(f"[AMABenchJudge] Could not parse yes/no from: {cleaned[:200]}")
                return {"is_correct": False, "reason": f"Parse failed: {cleaned[:200]}"}

        except Exception as e:
            logger.error(f"[AMABenchJudge] API call failed: {e}")
            return None

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        **kwargs,
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

        prompt = self._prompt_manager.format_judge(
            query_type=query_type,
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
                "metric": self.NAME,
            },
        )
