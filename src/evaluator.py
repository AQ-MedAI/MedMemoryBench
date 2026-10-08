from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from src.config import MethodConfig, DatasetConfig, ConfigLoader, PROJECT_ROOT
from src.result import EvaluationReport, ResultCollector
from utils.logger import get_eval_logger


# {dataset_name: evaluate_function}
DATASET_EVALUATOR_REGISTRY: Dict[str, Callable] = {}


def register_evaluator(dataset_name: str):
    """
    Example:
        @register_evaluator("medmemorybench")
        def evaluate_medmemorybench(method_config, dataset_config, **kwargs):
            ...
    """
    def decorator(func: Callable):
        DATASET_EVALUATOR_REGISTRY[dataset_name.lower()] = func
        return func
    return decorator


class Evaluator:

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        verbose: bool = True,
        resume: bool = False,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.dry_run = dry_run
        self.verbose = verbose
        self.resume = resume

        self.output_dir = output_dir or (PROJECT_ROOT / "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_eval_logger(
            method_config.method_name,
            dataset_config.dataset_name,
        )

        self.result_collector = ResultCollector()

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        self.logger.info(message)

    def run(self) -> EvaluationReport:
        _ensure_evaluators_registered()

        dataset_name = self.dataset_config.dataset_name.lower()

        evaluate_func = DATASET_EVALUATOR_REGISTRY.get(dataset_name)
        if evaluate_func is None:
            raise ValueError(
                f"No evaluator found for dataset '{dataset_name}', "
                f"available: {list(DATASET_EVALUATOR_REGISTRY.keys())}"
            )

        start_time = datetime.now()
        self._log("=" * 60)
        self._log("Starting evaluation")
        self._log(f"  Method: {self.method_config.method_name}")
        self._log(f"  Model: {self.method_config.model.name}")
        self._log(f"  Dataset: {self.dataset_config.dataset_name}")
        self._log(f"  Dry Run: {self.dry_run}")
        self._log(f"  Resume: {self.resume}")
        self._log("=" * 60)

        report = evaluate_func(
            method_config=self.method_config,
            dataset_config=self.dataset_config,
            output_dir=self.output_dir,
            dry_run=self.dry_run,
            verbose=self.verbose,
            logger=self.logger,
            resume=self.resume,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._log("=" * 60)
        self._log(f"Evaluation completed in {duration:.2f}s")
        self._log(f"  Total Queries: {report.summary.get('total', 0)}")
        self._log(f"  Accuracy: {report.summary.get('overall_accuracy', 0):.2%}")
        self._log("=" * 60)

        return report


def _ensure_evaluators_registered():
    """Lazy import to avoid circular import."""
    if "medmemorybench" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.medmemorybench.evaluator import evaluate_medmemorybench  # noqa: F401
    if "locomo" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.locomo.evaluator import evaluate_locomo  # noqa: F401
    if "longmemeval" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.longmemeval.evaluator import evaluate_longmemeval  # noqa: F401
    if "longmemeval_v2" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.longmemeval_v2.evaluator import evaluate_longmemeval_v2  # noqa: F401
    if "ama_bench" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.ama_bench.evaluator import evaluate_ama_bench  # noqa: F401


def create_evaluator(
    method_config_name: str,
    dataset_name: str,
    config_loader: Optional[ConfigLoader] = None,
    **kwargs
) -> Evaluator:
    if config_loader is None:
        config_loader = ConfigLoader()

    method_config = config_loader.load_method_config(method_config_name)
    dataset_config = config_loader.load_dataset_config(dataset_name)

    return Evaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        **kwargs
    )


def list_available_evaluators() -> list:
    return list(DATASET_EVALUATOR_REGISTRY.keys())
