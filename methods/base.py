"""Base agent class and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import tiktoken


@dataclass
class MemoryBuildResult:
    """Result of memorize() method."""
    success: bool = True
    method: str = ""
    action: str = ""
    input_content: str = ""
    stored_content: str = ""
    memory_entries: List[Dict[str, Any]] = field(default_factory=list)
    chunk_count: int = 0
    time_cost: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    # New fields for detailed intermediate results
    extraction_result: str = ""  # LLM extraction output
    all_passages: List[Dict[str, Any]] = field(default_factory=list)  # All stored passages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "method": self.method,
            "action": self.action,
            "input_content": self.input_content,
            "stored_content": self.stored_content,
            "memory_entries": self.memory_entries,
            "chunk_count": self.chunk_count,
            "time_cost": self.time_cost,
            "extraction_result": self.extraction_result,
            "all_passages": self.all_passages,
            **self.extra
        }


@dataclass
class AgentResponse:
    """Result of query() method."""
    output: str
    query_time: float = 0.0
    retrieved_count: int = 0
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "query_time": self.query_time,
            "retrieved_count": self.retrieved_count,
            "retrieved_memories": self.retrieved_memories,
            **self.extra
        }


class _SafeTokenizer:
    """Wrapper around tiktoken that allows special tokens as normal text."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str, **kwargs) -> list:
        kwargs.setdefault("disallowed_special", ())
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


class BaseAgent(ABC):
    """Base class for all memory agents."""

    METHOD_TYPE = "base"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

        try:
            base_tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            base_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self._tokenizer = _SafeTokenizer(base_tokenizer)

        self._memory_chunks: List[str] = []
        self._is_initialized = False
        self._context_id: Optional[int] = None

    @abstractmethod
    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store text into memory."""
        pass

    @abstractmethod
    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent."""
        pass

    def on_query_feedback(self, score: float, is_correct: bool, **kwargs) -> None:
        """Receive evaluation feedback after a query. Default no-op."""
        pass

    def reset(self) -> None:
        """Reset agent state."""
        self._memory_chunks = []
        self._is_initialized = False
        self._context_id = None

    def set_context_id(self, context_id: int) -> None:
        """Set context ID for distinguishing personas."""
        self._context_id = context_id

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    @property
    def memory_size(self) -> int:
        """Get current memory chunk count."""
        return len(self._memory_chunks)

    @property
    def total_memory_tokens(self) -> int:
        """Get total memory tokens."""
        return sum(self.count_tokens(chunk) for chunk in self._memory_chunks)

    @property
    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._is_initialized

    def get_memory_chunks(self) -> List[str]:
        """Get copy of memory chunks."""
        return self._memory_chunks.copy()

    @classmethod
    def get_method_type(cls) -> str:
        """Get method type."""
        return cls.METHOD_TYPE

    def get_info(self) -> Dict[str, Any]:
        """Get agent info."""
        return {
            "method_type": self.METHOD_TYPE,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "memory_size": self.memory_size,
            "total_memory_tokens": self.total_memory_tokens,
            "is_initialized": self._is_initialized,
        }

    def truncate_docs_to_context(
        self,
        docs: List[str],
        question: str,
        system_message: Optional[str] = None,
        max_context_tokens: int = 120000,
        doc_format: str = "[Memory {index}]\n{content}",
        overhead_tokens: int = 100,
        min_remaining_tokens: int = 100,
    ) -> List[str]:
        """Truncate retrieved docs to fit within context limit.

        Args:
            docs: List of documents to truncate
            question: The question being asked
            system_message: Optional system message
            max_context_tokens: Maximum tokens allowed for context
            doc_format: Format string for each doc (must contain {index} and {content})
            overhead_tokens: Reserved tokens for formatting overhead
            min_remaining_tokens: Minimum tokens needed to include partial doc

        Returns:
            List of truncated documents that fit within context limit
        """
        if not docs:
            return docs

        question_tokens = self.count_tokens(question)
        system_tokens = self.count_tokens(system_message) if system_message else 0
        available_tokens = max_context_tokens - question_tokens - system_tokens - overhead_tokens

        if available_tokens <= 0:
            return []

        truncated_docs = []
        current_tokens = 0

        for doc in docs:
            doc_tokens = self.count_tokens(doc)
            format_overhead = self.count_tokens(
                doc_format.format(index=len(truncated_docs) + 1, content="")
            )

            if current_tokens + doc_tokens + format_overhead <= available_tokens:
                truncated_docs.append(doc)
                current_tokens += doc_tokens + format_overhead
            else:
                remaining_tokens = available_tokens - current_tokens - format_overhead
                if remaining_tokens > min_remaining_tokens:
                    tokens = self._tokenizer.encode(doc)
                    truncated_text = self._tokenizer.decode(tokens[:remaining_tokens])
                    truncated_docs.append(truncated_text + "...")
                break

        return truncated_docs
