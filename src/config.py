"""Configuration loading module - method config, dataset config, and environment variables."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import yaml

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class APIConfig:
    """API configuration."""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    bigmodel_api_key: str = ""
    bigmodel_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    azure_api_key: str = ""
    azure_endpoint: str = ""
    azure_api_version: str = "2024-02-01"
    azure_deployment: str = ""

    anthropic_api_key: str = ""

    default_llm_model: str = "gpt-4o-mini"
    default_embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"

    judge_model: str = ""
    judge_api_key: str = ""
    judge_base_url: str = ""

    @property
    def use_azure(self) -> bool:
        return bool(self.azure_api_key and self.azure_endpoint)

    @property
    def is_configured(self) -> bool:
        if self.use_azure:
            return True
        return bool(self.openai_api_key)

    def get_judge_model(self) -> str:
        return self.judge_model or self.default_llm_model

    def get_judge_api_key(self) -> str:
        return self.judge_api_key or self.openai_api_key

    def get_judge_base_url(self) -> str:
        return self.judge_base_url or self.openai_base_url


def load_env_config(env_path: Optional[Path] = None) -> APIConfig:
    """Load environment configuration."""
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"

    if HAS_DOTENV and env_path.exists():
        load_dotenv(env_path)

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    bigmodel_api_key = os.getenv("BIGMODEL_API_KEY", "")
    bigmodel_base_url = os.getenv("BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

    # Only use BigModel config when BIGMODEL_API_KEY is explicitly set
    if bigmodel_api_key:
        openai_api_key = bigmodel_api_key
        # Only override base_url if BIGMODEL_BASE_URL is explicitly set in env
        env_bigmodel_base_url = os.getenv("BIGMODEL_BASE_URL")
        if env_bigmodel_base_url:
            openai_base_url = env_bigmodel_base_url

    return APIConfig(
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        bigmodel_api_key=bigmodel_api_key,
        bigmodel_base_url=bigmodel_base_url,
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        default_llm_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        default_embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        judge_model=os.getenv("JUDGE_MODEL", ""),
        judge_api_key=os.getenv("JUDGE_API_KEY", ""),
        judge_base_url=os.getenv("JUDGE_BASE_URL", ""),
    )


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: str = "openai"
    name: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 2000
    max_completion_tokens: Optional[int] = None  # For new models (gpt-5.x, o1-*, o3-*)
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    model_path: Optional[str] = None  # Local model path
    dim: Optional[int] = None  # Embedding dimension
    api_key: Optional[str] = None  # API key for embedding service
    base_url: Optional[str] = None  # Base URL for embedding service


@dataclass
class MethodConfig:
    """Method configuration."""
    method_name: str
    method_type: str  # baseline / rag / agentic_memory
    description: str = ""

    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: Optional[EmbeddingConfig] = None
    memorize_model: Optional[ModelConfig] = None  # Optional separate model for memorize phase
    agent_params: Dict[str, Any] = field(default_factory=dict)

    # Raw config (preserve all fields)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodConfig":
        """Create config from dict."""
        model_data = data.get("model", {})
        model_config = ModelConfig(
            provider=model_data.get("provider", "openai"),
            name=model_data.get("name", "gpt-4o-mini"),
            temperature=model_data.get("temperature", 1.0),
            max_tokens=model_data.get("max_tokens", 2000),
            max_completion_tokens=model_data.get("max_completion_tokens"),
            api_key=model_data.get("api_key"),
            base_url=model_data.get("base_url"),
        )

        embedding_config = None
        if "embedding" in data:
            emb_data = data["embedding"]
            raw_model_path = emb_data.get("model_path")
            # Resolve relative model paths against PROJECT_ROOT
            if raw_model_path and not os.path.isabs(raw_model_path):
                raw_model_path = str(PROJECT_ROOT / raw_model_path)
            embedding_config = EmbeddingConfig(
                provider=emb_data.get("provider", "openai"),
                model=emb_data.get("model", "text-embedding-3-small"),
                model_path=raw_model_path,
                dim=emb_data.get("dim"),
                api_key=emb_data.get("api_key"),
                base_url=emb_data.get("base_url"),
            )

        memorize_model_config = None
        if "memorize_model" in data:
            mem_data = data["memorize_model"]
            memorize_model_config = ModelConfig(
                provider=mem_data.get("provider", "openai"),
                name=mem_data.get("name", model_config.name),
                temperature=mem_data.get("temperature", 0.7),
                max_tokens=mem_data.get("max_tokens", 1000),
                max_completion_tokens=mem_data.get("max_completion_tokens"),
                api_key=mem_data.get("api_key"),
                base_url=mem_data.get("base_url"),
            )

        # Resolve relative paths in agent_params against PROJECT_ROOT
        raw_agent_params = data.get("agent_params", {})
        _path_keys = {
            "amem_embedding_model", "embedding_model_path", "model_path",
            "working_dir",
        }
        resolved_agent_params = {}
        for k, v in raw_agent_params.items():
            if k in _path_keys and isinstance(v, str) and v and not os.path.isabs(v):
                resolved_agent_params[k] = str(PROJECT_ROOT / v)
            else:
                resolved_agent_params[k] = v

        return cls(
            method_name=data.get("method_name", "unknown"),
            method_type=data.get("method_type", "baseline"),
            description=data.get("description", ""),
            model=model_config,
            embedding=embedding_config,
            memorize_model=memorize_model_config,
            agent_params=resolved_agent_params,
            raw_config=data,
        )


@dataclass
class QueryTypeConfig:
    """Query type configuration."""
    name: str
    abbr: str = ""
    metric: str = "exact_match"
    description: str = ""


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_name: str
    description: str = ""
    language: str = "zh"

    # Data paths
    data_root_dir: str = ""
    data_files: Dict[str, Any] = field(default_factory=dict)

    # Evaluation config
    evaluation_mode: str = "independent"
    persona_ids: Optional[List[int]] = None
    max_personas: Optional[int] = None
    max_sessions_per_persona: Optional[int] = None
    evaluation_interval: int = 10
    inject_noise: bool = True

    # Query types
    query_types: List[QueryTypeConfig] = field(default_factory=list)

    # Output config
    save_intermediate: bool = True
    save_retrieved_context: bool = True

    # Raw config
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create config from dict."""
        data_cfg = data.get("data", {})
        eval_cfg = data.get("evaluation", {})
        output_cfg = data.get("output", {})

        query_types = []
        for qt in data.get("query_types", []):
            query_types.append(QueryTypeConfig(
                name=qt.get("name", ""),
                abbr=qt.get("abbr", ""),
                metric=qt.get("metric", "exact_match"),
                description=qt.get("description", ""),
            ))

        return cls(
            dataset_name=data.get("dataset_name", "unknown"),
            description=data.get("description", ""),
            language=data.get("language", "zh"),
            data_root_dir=data_cfg.get("root_dir", ""),
            data_files=data_cfg,
            evaluation_mode=eval_cfg.get("mode", "independent"),
            persona_ids=eval_cfg.get("persona_ids"),
            max_personas=eval_cfg.get("max_personas"),
            max_sessions_per_persona=eval_cfg.get("max_sessions_per_persona"),
            evaluation_interval=eval_cfg.get("evaluation_interval", 10),
            inject_noise=eval_cfg.get("inject_noise", True),
            query_types=query_types,
            save_intermediate=output_cfg.get("save_intermediate", True),
            save_retrieved_context=output_cfg.get("save_retrieved_context", True),
            raw_config=data,
        )


class ConfigLoader:
    """Configuration loader."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self.configs_dir = self.project_root / "configs"
        self.method_config_dir = self.configs_dir / "method_config"
        self.dataset_config_dir = self.configs_dir / "dataset_config"

        # Cache
        self._api_config: Optional[APIConfig] = None
        self._method_configs: Dict[str, MethodConfig] = {}
        self._dataset_configs: Dict[str, DatasetConfig] = {}

    @property
    def api_config(self) -> APIConfig:
        """Get API config (lazy load)."""
        if self._api_config is None:
            self._api_config = load_env_config(self.project_root / ".env")
        return self._api_config

    def load_method_config(self, config_name: str) -> MethodConfig:
        """Load method config."""
        if config_name in self._method_configs:
            return self._method_configs[config_name]

        config_path = self.method_config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Method config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = MethodConfig.from_dict(data)
        self._method_configs[config_name] = config
        return config

    def load_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Load dataset config."""
        if dataset_name in self._dataset_configs:
            return self._dataset_configs[dataset_name]

        config_path = self.dataset_config_dir / f"{dataset_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = DatasetConfig.from_dict(data)
        self._dataset_configs[dataset_name] = config
        return config

    def list_method_configs(self) -> List[str]:
        """List all available method configs."""
        if not self.method_config_dir.exists():
            return []
        return [f.stem for f in self.method_config_dir.glob("*.yaml")]

    def list_dataset_configs(self) -> List[str]:
        """List all available dataset configs."""
        if not self.dataset_config_dir.exists():
            return []
        return [f.stem for f in self.dataset_config_dir.glob("*.yaml")]

_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get global config loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_api_config() -> APIConfig:
    """Get API config."""
    return get_config_loader().api_config


if __name__ == "__main__":
    loader = ConfigLoader()

    print("=== API Config ===")
    api_cfg = loader.api_config
    print(f"  OpenAI Key: {'configured' if api_cfg.openai_api_key else 'not set'}")
    print(f"  Azure: {'configured' if api_cfg.use_azure else 'not set'}")

    print("\n=== Available Method Configs ===")
    for name in loader.list_method_configs():
        print(f"  - {name}")

    print("\n=== Available Dataset Configs ===")
    for name in loader.list_dataset_configs():
        print(f"  - {name}")
