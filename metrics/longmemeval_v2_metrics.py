"""LongMemEval-V2 evaluation metrics.

Faithfully reproduces the six eval functions from the official LME-V2 harness
(``evaluation/qa_eval_metrics.py``):

  1. norm_phrase_set_match — normalized phrase set containment (order-free)
  2. norm_phrase_set_match_ordered — same but order-sensitive
  3. mc_choice_match — single-choice letter match
  4. mc_choice_set_match — multi-choice letter set match
  5. llm_abstention_checker — LLM judge for flawed-premise / abstention questions
  6. llm_gotchas_checker — LLM judge for environment-gotcha insight questions

The metric class ``LongMemEvalV2Metric`` auto-dispatches based on the
``eval_function`` field carried by each query.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Sequence

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS: Sequence[str] = (",", ";")


# ---------- deterministic eval helpers (from official code) ----------

def _extract_boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    i = idx + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    parsed = "".join(out).strip()
    return parsed if parsed else text.strip()


def _is_unknown(parsed: str) -> bool:
    return parsed.strip().lower() == "unknown"


def _normalize_phrase(
    text: Optional[str],
    *,
    lower: bool = True,
    normalize_hyphen: bool = True,
    strip_punct: bool = True,
) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if normalize_hyphen:
        text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[,;]", " ", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_phrases(
    text: Optional[str],
    *,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    **normalize_kwargs: bool,
) -> List[str]:
    if text is None:
        return []
    sep_list = list(separators)
    if not sep_list:
        n = _normalize_phrase(text, **normalize_kwargs)
        return [n] if n else []
    pattern = "|".join(re.escape(s) for s in sep_list)
    parts = re.split(pattern, text)
    return [p for p in (_normalize_phrase(pt, **normalize_kwargs) for pt in parts) if p]


def norm_phrase_set_match(
    prediction: Optional[str],
    answer: Optional[str],
    *,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = _normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = _split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    for phrase in set(answer_phrases):
        pat = r"\b%s\b" % re.escape(phrase)
        if re.search(pat, normalized_pred) is None:
            return False
    return True


def norm_phrase_set_match_ordered(
    prediction: Optional[str],
    answer: Optional[str],
    *,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = _normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = _split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    start = 0
    for phrase in answer_phrases:
        pat = r"\b%s\b" % re.escape(phrase)
        m = re.search(pat, normalized_pred[start:])
        if m is None:
            return False
        start += m.end()
    return True


def mc_choice_match(
    prediction: Optional[str],
    answer: Optional[str],
    *,
    strip_chars: str = ".",
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    if prediction is None or answer is None:
        return False
    prediction = str(prediction)
    answer = str(answer)
    boxed = re.search(r"\\boxed\{([^}]*)\}", prediction.lower())
    candidate = boxed.group(1) if boxed else prediction
    cleaned = re.sub(r"\b(choice|option)\b", "", candidate, flags=re.IGNORECASE)
    for ch in strip_chars:
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip().upper()
    expected = answer.strip().upper()
    if require_non_empty and (not cleaned or not expected):
        return False
    return cleaned == expected


_MULTI_SELECT_FILLER = {
    "AND", "ANSWER", "ANSWERS", "CHOICE", "CHOICES",
    "FINAL", "LETTER", "LETTERS", "OPTION", "OPTIONS",
}


def _extract_multi_select_letters(text: Optional[str]) -> list[str]:
    if text is None:
        return []
    chunks = re.findall(r"[A-Z]+", str(text).upper())
    letters: list[str] = []
    for chunk in chunks:
        if chunk in _MULTI_SELECT_FILLER:
            continue
        letters.extend(list(chunk))
    return letters


def mc_choice_set_match(
    prediction: Optional[str],
    answer: Optional[str],
    *,
    require_non_empty: bool = True,
    **_: Any,
) -> bool:
    pred_letters = _extract_multi_select_letters(prediction)
    answer_letters = _extract_multi_select_letters(answer)
    if require_non_empty and (not pred_letters or not answer_letters):
        return False
    return set(pred_letters) == set(answer_letters)


# ---------- eval_function spec parser (from official code) ----------

def _parse_eval_value(key: str, value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if key in {"separators", "separator"}:
        if not value:
            return []
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return json.loads(stripped)
        return [ch for ch in value if not ch.isspace()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


_DETERMINISTIC_FUNCS = {
    "norm_phrase_set_match": norm_phrase_set_match,
    "norm_phrase_set_match_ordered": norm_phrase_set_match_ordered,
    "mc_choice_match": mc_choice_match,
    "mc_choice_set_match": mc_choice_set_match,
}


def parse_eval_spec(spec: str) -> tuple:
    parts = [p.strip() for p in spec.split("|")]
    name = parts[0]
    kwargs: Dict[str, Any] = {}
    for part in parts[1:]:
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        kwargs[k.strip()] = _parse_eval_value(k.strip(), v.strip())
    return name, kwargs


# ---------- LLM judge helpers (from official code) ----------

_ABSTENTION_SYSTEM = (
    "You are a strict grader for flawed-premise (abstention) questions. "
    "Judge whether a model answer correctly identifies that the question premise is wrong, "
    "consistent with the reference answer. "
    "If the model follows the flawed premise and gives a concrete answer under that premise, "
    "it must be graded 0. "
    "If the model's final answer is just UNKNOWN / cannot determine without identifying the flaw, grade 0. "
    "If the model is contradictory (both rejects premise and also gives a concrete premise-following answer), grade 0. "
    "Paraphrases are allowed when they preserve the same core flaw described by the reference answer."
)

_GOTCHAS_SYSTEM = (
    "You are a strict grader for gotchas-style insight questions. "
    "The reference answer describes the key insight(s). "
    "Grade 1 if the model response includes at least one correct insight point from the reference answer "
    "(paraphrase allowed), and does not contradict any reference point. "
    "If the model's direction is wrong, or it contains contradictions against any reference point, grade 0. "
    "If the model gives multiple points, partial coverage is enough for 1 as long as no contradictions appear."
)


def _build_abstention_judge_messages(
    question: str, reference: str, full_response: str, final_answer: str,
) -> List[Dict[str, str]]:
    user = (
        "Evaluate whether the model correctly handled a flawed-premise question.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer (the expected flaw-aware conclusion):\n{reference}\n\n"
        f"Model full response:\n{full_response}\n\n"
        f"Model extracted final answer:\n{final_answer}\n\n"
        "Scoring rubric:\n"
        "- Label 1 only if the model clearly identifies the flawed premise and reaches a conclusion "
        "equivalent to the reference answer.\n"
        "- Label 1 also if the model explicitly says it lacks access to the user's specific "
        "live environment/instance/configuration and therefore cannot verify, provided it does not "
        "give a concrete premise-following answer.\n"
        "- Label 0 if the model follows the flawed premise and gives a concrete answer under that premise.\n"
        "- Label 0 for generic UNKNOWN/insufficient-info replies that do not identify a flaw and do not "
        "make the explicit environment-access limitation clear.\n"
        "- Label 0 if contradictory.\n\n"
        "Output JSON only:\n"
        '{"label": 0 or 1, "reason": "short rationale"}'
    )
    return [
        {"role": "system", "content": _ABSTENTION_SYSTEM},
        {"role": "user", "content": user},
    ]


def _build_gotchas_judge_messages(
    question: str, reference: str, full_response: str, final_answer: str,
) -> List[Dict[str, str]]:
    user = (
        "Evaluate whether the model answer captures the gotcha insight.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer (insight points):\n{reference}\n\n"
        f"Model full response:\n{full_response}\n\n"
        f"Model extracted final answer:\n{final_answer}\n\n"
        "Scoring rubric:\n"
        "- Label 1 if the model includes at least one correct insight point from the reference answer "
        "(paraphrase acceptable), and does not contradict any reference point.\n"
        "- Label 1 even if only part of a multi-point reference answer is covered, as long as there is "
        "no contradiction.\n"
        "- Label 0 if direction is wrong (suggests opposite action/cause), even if some wording overlaps.\n"
        "- Label 0 if any point in the model response contradicts any reference point.\n"
        "- Label 0 if the response is irrelevant or generic without insight.\n\n"
        "Output JSON only:\n"
        '{"label": 0 or 1, "reason": "short rationale"}'
    )
    return [
        {"role": "system", "content": _GOTCHAS_SYSTEM},
        {"role": "user", "content": user},
    ]


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return s


def _parse_judge_label(text: str) -> tuple[int, str]:
    cleaned = _strip_code_fence(text.strip())
    if not cleaned:
        raise ValueError("Empty judge response")

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict):
                label = payload.get("label")
                if label in {0, 1, "0", "1"}:
                    return int(label), str(payload.get("reason", ""))
        except json.JSONDecodeError:
            pass

    label_match = re.search(r'"label"\s*:\s*([01])', cleaned, flags=re.IGNORECASE)
    if not label_match:
        label_match = re.search(r"'label'\s*:\s*([01])", cleaned, flags=re.IGNORECASE)
    if not label_match:
        label_match = re.search(r"\blabel\b\s*[:=]\s*([01])", cleaned, flags=re.IGNORECASE)
    if label_match:
        return int(label_match.group(1)), cleaned

    raise ValueError(f"Cannot parse judge label from: {cleaned!r}")


# ---------- Main metric class ----------

class LongMemEvalV2Metric(BaseMetric):
    """Metric for LongMemEval-V2 that dispatches based on the official eval_function spec.

    Deterministic eval functions are computed locally.  LLM-based eval
    functions (``llm_abstention_checker``, ``llm_gotchas_checker``) call the
    configured judge model with the exact same prompts as the official harness.
    """

    NAME = "longmemeval_v2"

    def __init__(
        self,
        dataset: str = "longmemeval_v2",
        judge_model: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        language: str = "en",
    ):
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        if self._initialized:
            return
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
            max_tokens=512,
            api_key=api_key,
            base_url=base_url,
        )
        self._initialized = True

    def _call_judge(self, messages: List[Dict[str, str]]) -> tuple[int, str]:
        self._ensure_client()
        try:
            response = self._client.chat(messages=messages, max_tokens=512)
            text = response.content.strip()
            return _parse_judge_label(text)
        except Exception as e:
            logger.error(f"[LME-V2 Judge] failed: {e}")
            return 0, f"judge_error: {e}"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        **kwargs,
    ) -> MetricResult:
        eval_function_spec: str = kwargs.get("eval_function", "")
        if not eval_function_spec:
            eval_function_spec = "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true"

        func_name, func_kwargs = parse_eval_spec(eval_function_spec)
        answer_gold = expected_answers[0] if expected_answers else ""

        parsed_answer = _extract_boxed_answer(model_output) if model_output else ""
        unknown = _is_unknown(parsed_answer)

        details: Dict[str, Any] = {
            "metric": self.NAME,
            "eval_function": eval_function_spec,
            "parsed_boxed_answer": parsed_answer,
            "is_unknown": unknown,
            "query_type": query_type,
            "category": kwargs.get("category", ""),
            "is_abstention": kwargs.get("is_abstention", False),
        }

        if func_name in _DETERMINISTIC_FUNCS:
            score_bool = _DETERMINISTIC_FUNCS[func_name](parsed_answer, answer_gold, **func_kwargs)
            if unknown:
                score_bool = False
            details["eval_method"] = "deterministic"

        elif func_name == "llm_abstention_checker":
            label, reason = self._call_judge(
                _build_abstention_judge_messages(
                    question=question,
                    reference=answer_gold,
                    full_response=model_output,
                    final_answer=parsed_answer,
                )
            )
            score_bool = label == 1
            details["eval_method"] = "llm_abstention_checker"
            details["judge_reason"] = reason

        elif func_name == "llm_gotchas_checker":
            label, reason = self._call_judge(
                _build_gotchas_judge_messages(
                    question=question,
                    reference=answer_gold,
                    full_response=model_output,
                    final_answer=parsed_answer,
                )
            )
            score_bool = label == 1
            details["eval_method"] = "llm_gotchas_checker"
            details["judge_reason"] = reason

        else:
            logger.warning(f"Unknown eval function '{func_name}', falling back to norm_phrase_set_match")
            score_bool = norm_phrase_set_match(
                parsed_answer, answer_gold,
                lower=True, normalize_hyphen=True, strip_punct=True,
            )
            if unknown:
                score_bool = False
            details["eval_method"] = "fallback_norm_phrase_set_match"

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=1.0 if score_bool else 0.0,
            is_correct=score_bool,
            model_output=model_output,
            expected_answer=answer_gold,
            question=question,
            details=details,
        )
