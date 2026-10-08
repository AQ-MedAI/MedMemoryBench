"""Prompt template management module."""

from typing import Dict, Optional
import re

from .prompts_memorize import MEMORIZE_TEMPLATES
from .prompts_qa import QA_TEMPLATES
from .prompts_judge import JUDGE_TEMPLATES


SYSTEM_MESSAGES: Dict[str, str] = {
    "medmemorybench": "你是患者的个性化医疗助手，能够准确记忆患者的完整病史。请基于记忆中的患者信息进行推理回复，语气亲切专业，回答直接，避免冗长解释和套话。",
    "medmemorybench_en": "You are the patient's personalized medical assistant, capable of accurately memorizing the patient's complete medical history. Please reason and respond based on patient information in memory, maintaining a warm yet professional tone, answering directly, and avoiding lengthy explanations and boilerplate.",
    "locomo": "You are a helpful assistant that can read the context and memorize it for future retrieval.",
    "longmemeval": "You are a helpful assistant with long-term memory. You can remember past conversations and use that information to answer questions accurately.",
    "ama_bench": "You are an expert agent trajectory analyst. You can memorize and recall detailed information from agent task execution trajectories, including actions taken, observations received, state changes, and causal relationships between events.",
    "longmemeval_v2": "You are an experienced colleague who has observed many web-agent task trajectories. You can recall details from these trajectories — including the agent's goals, actions, page states, and outcomes — to answer questions accurately. If you do not know the answer, output exactly \\boxed{UNKNOWN}. Do not guess.",
}

METHOD_TYPE_MAPPING: Dict[str, str] = {
    "long_context": "long_context",
    "embedding_rag": "rag",
    "bm25_rag": "rag",
    "graph_rag": "rag",
    "raptor": "rag",
    "self_rag": "rag",
    "memo_rag": "rag",
    "mem0": "agentic",
    "mirix": "agentic",
    "zep": "agentic",
    "letta": "agentic",
    "cognee": "agentic",
    "q2q": "agentic",
}

MEMORY_SOURCE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "medmemorybench": {
        "long_context": "之前记忆的对话内容",
        "rag": "检索到的相关对话记录",
        "agentic": "记忆库中的相关信息",
    },
    "medmemorybench_en": {
        "long_context": "the previously memorized dialogue content",
        "rag": "the retrieved relevant dialogue records",
        "agentic": "the relevant information from the memory store",
    },
    "locomo": {
        "long_context": "the memorized conversation records",
        "rag": "the retrieved relevant records",
        "agentic": "the archival memory",
    },
    "longmemeval": {
        "long_context": "the memorized conversation history",
        "rag": "the retrieved relevant conversations",
        "agentic": "your memory of past conversations",
    },
    "ama_bench": {
        "long_context": "the memorized records",
        "rag": "the retrieved relevant trajectory segments",
        "agentic": "the relevant information from memory",
    },
    "longmemeval_v2": {
        "long_context": "the memorized web-agent trajectory records",
        "rag": "the retrieved relevant trajectory records",
        "agentic": "your memory of past web-agent trajectories",
    },
}


class PromptManager:
    def __init__(self, dataset: str, method: str = None, language: str = "zh"):
        self.dataset = dataset.lower()
        self.method = method.lower() if method else None
        self.language = language.lower()
        self.method_type = METHOD_TYPE_MAPPING.get(self.method, "rag") if self.method else None

        # Determine the effective dataset key for template lookup
        # When dataset is "medmemorybench" and language is "en", use "medmemorybench_en" prefix
        if self.dataset == "medmemorybench" and self.language == "en":
            self._template_prefix = "medmemorybench_en"
        else:
            self._template_prefix = self.dataset

    def get_system_message(self) -> str:
        return SYSTEM_MESSAGES.get(self._template_prefix, "")

    def format_memorize(self, context: str, timestamp: Optional[str] = None) -> str:
        key = f"{self._template_prefix}_{self.method_type}_memorize"
        template = MEMORIZE_TEMPLATES.get(key)

        if not template:
            # Fallback to base dataset key (without language suffix)
            key = f"{self.dataset}_{self.method_type}_memorize"
            template = MEMORIZE_TEMPLATES.get(key)

        if not template:
            return context

        params = {"context": context}
        if timestamp and "{timestamp}" in template:
            params["timestamp"] = timestamp

        return template.format(**params)

    def format_query(self, question: str, query_type: str, question_date: str = "", **kwargs) -> str:
        key = f"{self._template_prefix}_{query_type}_qa"
        template = QA_TEMPLATES.get(key)

        if not template:
            # Fallback to base dataset key
            key = f"{self.dataset}_{query_type}_qa"
            template = QA_TEMPLATES.get(key)

        if not template:
            key = f"{self._template_prefix}_default_qa"
            template = QA_TEMPLATES.get(key)

        if not template:
            key = f"{self.dataset}_default_qa"
            template = QA_TEMPLATES.get(key, "Question: {question}\n\nAnswer:")

        memory_source = MEMORY_SOURCE_DESCRIPTIONS.get(self._template_prefix, {}).get(
            self.method_type, MEMORY_SOURCE_DESCRIPTIONS.get(self.dataset, {}).get(
                self.method_type, "the relevant memories"
            )
        )

        params = {"question": question, "memory_source": memory_source}
        if question_date and "{question_date}" in template:
            params["question_date"] = question_date
        elif "{question_date}" in template:
            params["question_date"] = ""

        return template.format(**params)

    def format_judge(
        self,
        query_type: str,
        question: str,
        model_output: str,
        expected_answer: str,
        explanation: str = "",
        **kwargs
    ) -> str:
        key = f"{self._template_prefix}_{query_type}_judge"
        template = JUDGE_TEMPLATES.get(key)

        if not template:
            # Fallback to base dataset key
            key = f"{self.dataset}_{query_type}_judge"
            template = JUDGE_TEMPLATES.get(key)

        if not template:
            raise ValueError(f"No judge template found: {self._template_prefix}_{query_type}_judge or {self.dataset}_{query_type}_judge")

        params = {
            "question": question,
            "model_output": model_output,
            "expected_answer": expected_answer,
            "explanation": explanation,
        }
        params.update(kwargs)

        placeholders = re.findall(r'\{(\w+)\}', template)
        for ph in placeholders:
            if ph not in params:
                params[ph] = ""

        return template.format(**params)

    def has_judge_template(self, query_type: str) -> bool:
        key = f"{self._template_prefix}_{query_type}_judge"
        if key in JUDGE_TEMPLATES:
            return True
        # Fallback check
        key = f"{self.dataset}_{query_type}_judge"
        return key in JUDGE_TEMPLATES


def get_prompt_manager(dataset: str, method: str = None, language: str = "zh") -> PromptManager:
    return PromptManager(dataset, method, language)


# Backward compatibility
class TemplateManager(PromptManager):
    def get_memorize_template(self, with_timestamp: bool = False) -> str:
        key = f"{self.dataset}_{self.method_type}_memorize"
        return MEMORIZE_TEMPLATES.get(key, "{context}")

    def get_query_template(self, method_name: str, query_type: Optional[str] = None) -> str:
        if query_type:
            key = f"{self.dataset}_{query_type}_qa"
            template = QA_TEMPLATES.get(key)
            if template:
                return template
        key = f"{self.dataset}_default_qa"
        return QA_TEMPLATES.get(key, "Question: {question}\n\nAnswer:")


def get_template_manager(dataset_name: str) -> TemplateManager:
    return TemplateManager(dataset_name)
