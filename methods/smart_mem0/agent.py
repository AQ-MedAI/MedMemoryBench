"""Public SmartMem0 facade assembled from cohesive pipeline components."""

from methods.base import BaseAgent

from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .planning import PlanningMixin
from .query import QueryMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin


class SmartMem0Agent(
    QueryMixin,
    ExecutionMixin,
    RetrievalOperationsMixin,
    PlanningMixin,
    WriteLifecycleMixin,
    ConsolidationMixin,
    CaptureMixin,
    CoreMemoryMixin,
    BaseAgent,
):
    """Compact evidence-grounded long-term memory for conversational agents."""
