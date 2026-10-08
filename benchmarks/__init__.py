"""Benchmark dataset module."""

from pathlib import Path
from typing import Dict, Type, Optional

from .base import BaseDataset, Session, Query, EvaluationUnit
from .medmemorybench import MedMemoryBenchDataset
from .locomo import LoCoMoDataset
from .longmemeval_v2 import LongMemEvalV2Dataset
from .ama_bench import AMABenchDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "medmemorybench": MedMemoryBenchDataset,
    "locomo": LoCoMoDataset,
    "longmemeval_v2": LongMemEvalV2Dataset,
    "ama_bench": AMABenchDataset,
}


def create_dataset(
    dataset_name: str,
    data_dir: Path,
    config: Dict,
) -> BaseDataset:
    """Create dataset instance."""
    dataset_class = DATASET_REGISTRY.get(dataset_name.lower())
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}, available: {list(DATASET_REGISTRY.keys())}")

    return dataset_class(data_dir, config)


__all__ = [
    "BaseDataset",
    "Session",
    "Query",
    "EvaluationUnit",
    "MedMemoryBenchDataset",
    "LoCoMoDataset",
    "LongMemEvalV2Dataset",
    "AMABenchDataset",
    "create_dataset",
    "DATASET_REGISTRY",
]
