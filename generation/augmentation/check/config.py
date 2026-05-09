"""Configuration for the check module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

_GENERATION_DIR = str(Path(__file__).parent.parent.parent.resolve())


@dataclass
class CheckerConfig:
    """Query difficulty checker configuration."""

    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 2000
    verbose: bool = True
    save_intermediate: bool = True


@dataclass
class EnhancerConfig:
    """Difficulty enhancement configuration."""

    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 3000
    auto_regenerate: bool = False


@dataclass
class CheckRunnerConfig:
    """Check runner configuration."""

    dataset_dir: str = field(default_factory=lambda: str(Path(_GENERATION_DIR) / "dataset"))
    output_dir: str = field(default_factory=lambda: str(Path(_GENERATION_DIR) / "augmentation"))
    persona_ids: List[int] = field(default_factory=list)
    checker: CheckerConfig = field(default_factory=CheckerConfig)
    enhancer: EnhancerConfig = field(default_factory=EnhancerConfig)
    enable_regenerate: bool = False
    dry_run: bool = False
    verbose: bool = True
    generate_report: bool = True
    report_format: str = "markdown"
