"""Base metric classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class MetricResult:
    """Metric evaluation result."""
    query_id: str
    query_type: str
    score: float
    is_correct: bool
    model_output: str
    expected_answer: str
    question: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    memory_construction_time: float = 0.0
    query_time: float = 0.0
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "score": self.score,
            "is_correct": self.is_correct,
            "model_output": self.model_output,
            "expected_answer": self.expected_answer,
            "question": self.question,
            "details": self.details,
            "memory_construction_time": self.memory_construction_time,
            "query_time": self.query_time,
            "retrieved_memories": self.retrieved_memories,
            "retrieved_count": self.retrieved_count,
            "extra": self.extra,
        }


class BaseMetric(ABC):
    """Base class for evaluation metrics."""

    NAME = "base"

    @abstractmethod
    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        **kwargs
    ) -> MetricResult:
        pass

    @classmethod
    def get_name(cls) -> str:
        return cls.NAME
