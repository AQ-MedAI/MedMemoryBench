"""Dataset base classes - unified interface for dataset processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Iterator, Union
from pathlib import Path


@dataclass
class Session:
    """Session data base class."""
    session_id: Union[int, str]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_memory_text(self) -> str:
        """Convert to memory text."""
        return self.content


@dataclass
class Query:
    """Query data base class."""
    query_id: str
    question: str
    query_type: str
    expected_answers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_correct_answers(self) -> List[str]:
        """Get correct answers."""
        return self.expected_answers


@dataclass
class EvaluationUnit:
    """Evaluation unit - contains sessions to inject and queries to evaluate."""
    unit_id: int
    sessions_to_inject: List[Session]
    queries_to_evaluate: List[Query]
    context_id: Optional[Union[int, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """Dataset base class - defines unified interface for loading, processing, and evaluation."""

    NAME = "base"

    def __init__(
        self,
        data_dir: Path,
        config: Dict[str, Any],
    ):
        self.data_dir = data_dir
        self.config = config
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load dataset."""
        pass

    @abstractmethod
    def get_evaluation_units(self) -> Iterator[EvaluationUnit]:
        """Get evaluation unit iterator."""
        pass

    @abstractmethod
    def get_total_sessions(self) -> int:
        """Get total session count."""
        pass

    @abstractmethod
    def get_total_queries(self) -> int:
        """Get total query count."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if loaded."""
        return self._is_loaded

    def get_info(self) -> Dict[str, Any]:
        """Get dataset info."""
        return {
            "name": self.NAME,
            "data_dir": str(self.data_dir),
            "is_loaded": self._is_loaded,
            "total_sessions": self.get_total_sessions() if self._is_loaded else 0,
            "total_queries": self.get_total_queries() if self._is_loaded else 0,
        }
