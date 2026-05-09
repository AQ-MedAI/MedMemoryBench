"""Tokenizer utility module - supports local HuggingFace and tiktoken encodings."""

import os
from pathlib import Path
from typing import Optional, Union, Protocol, List
import logging

logger = logging.getLogger(__name__)


class TokenizerProtocol(Protocol):
    """Tokenizer protocol."""
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


class LocalHFTokenizer:
    """Local HuggingFace Tokenizer wrapper."""

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model_path = model_path

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)


class TiktokenWrapper:
    """Tiktoken wrapper."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def encode(self, text: str) -> List[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._encoding.decode(tokens)


class CharEstimationTokenizer:
    """Character estimation Tokenizer (fallback)."""

    def __init__(self, chars_per_token: float = 3.0):
        self.chars_per_token = chars_per_token

    def encode(self, text: str) -> List[int]:
        # Return pseudo list of estimated token count
        estimated_count = int(len(text) / self.chars_per_token)
        return list(range(estimated_count))

    def decode(self, tokens: List[int]) -> str:
        # Cannot truly decode, return empty string
        return ""


# Default local models directory (configurable via MODELS_DIR env var)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(_PROJECT_ROOT / "models")))

# Tokenizer model mapping
# key: model family, value: (local dir name, tiktoken encoding, HuggingFace model name)
TOKENIZER_MODEL_MAP = {
    # GPT-4/GPT-4o series use cl100k_base
    "gpt-4": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),
    "gpt-4o": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),
    "gpt-4-turbo": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),

    # GPT-5 series
    "gpt-5": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),

    # Chinese optimized models
    "qwen": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),
    "chinese": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),

    # Default uses Qwen (Chinese task optimized)
    "default": ("Qwen2.5-0.5B-Instruct", "cl100k_base", "Qwen/Qwen2.5-0.5B-Instruct"),
}


def get_tokenizer(
    model_name: str = "default",
    local_models_dir: Optional[Path] = None,
    prefer_local: bool = True,
) -> TokenizerProtocol:
    """Get tokenizer instance. Priority: local model > tiktoken > char estimation."""
    models_dir = local_models_dir or DEFAULT_MODELS_DIR

    # Find config by model name
    model_key = "default"
    model_name_lower = model_name.lower()
    for key in TOKENIZER_MODEL_MAP.keys():
        if key in model_name_lower:
            model_key = key
            break

    local_dir_name, tiktoken_encoding, hf_model_name = TOKENIZER_MODEL_MAP[model_key]

    # Try local model
    if prefer_local:
        local_path = models_dir / local_dir_name
        if local_path.exists():
            try:
                tokenizer = LocalHFTokenizer(str(local_path))
                logger.info(f"Using local Tokenizer: {local_path}")
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load local Tokenizer: {e}")

    # Try tiktoken
    try:
        tokenizer = TiktokenWrapper(tiktoken_encoding)
        logger.info(f"Using tiktoken encoding: {tiktoken_encoding}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tiktoken: {e}")

    # Fallback: char estimation
    logger.warning("Using char estimation Tokenizer (lower accuracy)")
    return CharEstimationTokenizer()


def count_tokens(text: str, tokenizer: Optional[TokenizerProtocol] = None) -> int:
    """Count tokens in text."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    return len(tokenizer.encode(text))


# ============================================================================
# Recommended local tokenizer models
# ============================================================================
RECOMMENDED_TOKENIZERS = """
Recommended tokenizer models (place in <project_root>/models/ or set MODELS_DIR env var):

1. GPT-2 (general purpose, good for English and code)
   - Dir name: gpt2
   - HuggingFace: openai-community/gpt2
   - Download:
     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('openai-community/gpt2').save_pretrained('models/gpt2')"

2. Qwen2.5-0.5B (Chinese optimized, recommended for Chinese tasks)
   - Dir name: Qwen2.5-0.5B
   - HuggingFace: Qwen/Qwen2.5-0.5B
   - Download:
     python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True).save_pretrained('models/Qwen2.5-0.5B')"

Note:
- GPT-2 tokenizer is small (~1MB), good for general use
- Qwen2.5-0.5B tokenizer has better Chinese support, recommended for medical dialogue
- If both downloaded, system auto-selects based on task
- Set MODELS_DIR environment variable to customize the models directory
"""


def print_recommended_tokenizers():
    """Print recommended tokenizer models."""
    print(RECOMMENDED_TOKENIZERS)


__all__ = [
    "TokenizerProtocol",
    "LocalHFTokenizer",
    "TiktokenWrapper",
    "CharEstimationTokenizer",
    "get_tokenizer",
    "count_tokens",
    "print_recommended_tokenizers",
    "RECOMMENDED_TOKENIZERS",
]
