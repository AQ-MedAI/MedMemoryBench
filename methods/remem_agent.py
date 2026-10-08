"""ReMem agent adapter for MedMemoryBench."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import (
    BaseLLMClient,
    LLMResponse,
    create_llm_client,
    format_messages,
    get_usage_tracker,
)

logger = logging.getLogger(__name__)

class TrackedLLMWrapper:
    """Wraps ReMem's LLM calls to route through the evaluation framework's llm_client.

    Implements the same interface as CacheOpenAI (infer, batch_infer) while ensuring
    accurate token tracking via our llm_client.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        llm_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        seed: int = 0,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.temperature = temperature
        self.seed = seed
        self.kwargs = kwargs

        self.llm_config = _LLMConfigProxy(
            llm_name=llm_name,
            temperature=temperature,
            seed=seed,
        )

    def infer(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, Dict, bool]:
        """Mimics CacheOpenAI.infer() interface.

        Returns:
            Tuple[response_content, metadata, cache_hit]
        """
        temperature = kwargs.get("temperature", self.temperature)

        extra_kwargs = {}
        if "response_format" in kwargs:
            extra_kwargs["response_format"] = kwargs["response_format"]
        if "max_completion_tokens" in kwargs:
            extra_kwargs["max_tokens"] = kwargs["max_completion_tokens"]

        try:
            response = self.llm_client.chat(
                messages=messages,
                temperature=temperature,
                **extra_kwargs,
            )

            content = response.content

            # Strip markdown code fences from JSON responses
            if kwargs.get("response_format", {}).get("type") == "json_object":
                if content.startswith("```json\n") and content.endswith("```"):
                    content = content[8:-3].strip()

            metadata = {
                "prompt": messages,
                "response": content,
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "finish_reason": "stop",
            }

            return content, metadata, False  # cache_hit=False

        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return "", {"error": str(e)}, False

    def batch_infer(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_workers: int = 10,
        **kwargs,
    ) -> List[Tuple[str, Dict, bool]]:
        """Mimics CacheOpenAI.batch_infer() using a thread pool."""
        results = [None] * len(messages_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.infer, msgs, **kwargs): idx
                for idx, msgs in enumerate(messages_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch infer error at index {idx}: {e}")
                    results[idx] = ("", {"error": str(e)}, False)

        return results


class _LLMConfigProxy:
    """Proxy mimicking ReMem's LLMConfig structure."""

    def __init__(self, llm_name: str, temperature: float, seed: int):
        self.generate_params = {
            "model": llm_name,
            "temperature": temperature,
            "seed": seed,
            "n": 1,
        }


class TrackedEmbeddingWrapper:
    """Wraps ReMem's embedding calls with support for local models and OpenAI-compatible APIs.

    Implements the same interface as ReMem's BaseEmbeddingModel.
    """

    def __init__(
        self,
        provider: str = "local",
        model: str = "BAAI/bge-small-zh-v1.5",
        model_path: Optional[str] = None,
        dim: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 16,
        max_seq_len: int = 512,
        normalize: bool = True,
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.model_path = model_path or model
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.normalize = normalize

        # Attributes required by ReMem internals
        self.embedding_model_name = model
        self.embedding_size = dim  # Fallback dimension for EmbeddingStore

        self._model = None
        self._openai_client = None

        if provider == "local":
            self._init_local_model()
        elif provider in ["openai", "api"]:
            self._init_api_client(api_key, base_url)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _init_local_model(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model_path}")
            self._model = SentenceTransformer(
                self.model_path,
                trust_remote_code=True,
            )

            # Detect actual dimension
            if self.dim is None:
                self.dim = self._model.get_sentence_embedding_dimension()

            self.embedding_size = self.dim

            logger.info(f"Embedding model loaded, dim={self.dim}")

        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _init_api_client(self, api_key: Optional[str], base_url: Optional[str]):
        """Initialize OpenAI-compatible API client for embeddings."""
        from openai import OpenAI

        self._openai_client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=base_url or os.environ.get("EMBEDDING_BASE_URL"),
            timeout=60,
        )
        logger.info(f"Embedding API client initialized, model={self.model_name}")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts into embedding vectors.

        Returns:
            np.ndarray of shape (len(texts), dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        texts = [t if t.strip() else "empty" for t in texts]

        # ReMem uses instruction prefix to distinguish queries from documents
        instruction = kwargs.get("instruction", "")
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]

        if self.provider == "local":
            embeddings = self._encode_local(texts)
        else:
            embeddings = self._encode_api(texts)

        # Normalize
        if self.normalize and kwargs.get("norm", True):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        """Encode using local model."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,  # We handle normalization ourselves
        )
        return np.array(embeddings)

    def _encode_api(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI-compatible API."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self._openai_client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding API error: {e}")
                # Fall back to zero vectors
                dim = self.dim or 1536
                all_embeddings.extend([np.zeros(dim) for _ in batch])

        return np.array(all_embeddings)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Batch encode (alias for encode, for interface compatibility)."""
        return self.encode(texts, **kwargs)

class RememAgent(BaseAgent):
    """ReMem adapter for MedMemoryBench.

    Integrates ReMem (Reasoning with Episodic Memory) into the evaluation framework,
    routing all LLM calls through llm_client for token tracking.

    Cumulative build mode:
    - Graph construction is triggered in memorize() when is_last_session=True
    - query() only handles retrieval and QA
    """

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        # Base model config
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # ReMem-specific parameters
        extract_method: str = "episodic_gist",  # openie / episodic / episodic_gist / temporal
        # Graph config
        is_directed_graph: bool = False,
        synonymy_edge_sim_threshold: float = 0.8,
        synonymy_edge_topk: int = 2047,
        # Retrieval config
        retrieval_top_k: int = 200,
        qa_top_k: int = 5,
        linking_top_k: int = 5,
        damping: float = 0.5,
        passage_node_weight: float = 0.05,
        # Agent config
        use_agent: bool = False,
        agent_fixed_tools: bool = False,
        agent_max_steps: int = 5,
        # Cache config
        use_cache: bool = True,
        force_index_from_scratch: bool = False,
        save_openie: bool = True,
        # API concurrency (lower values reduce API timeouts)
        extraction_max_workers: int = 5,
        # Cumulative build config
        session_batch_size: int = 10,
        # Embedding config
        embedding_provider: str = "local",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_model_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_batch_size: int = 16,
        embedding_max_seq_len: int = 512,
        # Chunking config
        chunk_size_tokens: int = 8000,
        chunk_overlap_tokens: int = 200,
        # Text preprocessing
        text_preprocessor_class_name: str = "TextPreprocessor",
        window_sizes: Optional[List[int]] = None,
        # Token limits
        max_input_tokens: int = 8000,
        max_context_tokens: int = 120000,
        # Working directory
        working_dir: Optional[str] = None,
        # Dataset awareness for prompt template routing
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.provider = provider
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        self.extract_method = extract_method
        self.is_directed_graph = is_directed_graph
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        self.synonymy_edge_topk = synonymy_edge_topk
        self.retrieval_top_k = retrieval_top_k
        self.qa_top_k = qa_top_k
        self.linking_top_k = linking_top_k
        self.damping = damping
        self.passage_node_weight = passage_node_weight
        self.use_agent = use_agent
        self.agent_fixed_tools = agent_fixed_tools
        self.agent_max_steps = agent_max_steps
        self.use_cache = use_cache
        self.force_index_from_scratch = force_index_from_scratch
        self.save_openie = save_openie
        self.extraction_max_workers = extraction_max_workers
        self.session_batch_size = session_batch_size

        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_seq_len = embedding_max_seq_len

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.text_preprocessor_class_name = text_preprocessor_class_name
        self.window_sizes = window_sizes or [1, 3]

        self.max_input_tokens = max_input_tokens
        self.max_context_tokens = max_context_tokens

        self.working_dir = working_dir
        self.dataset_name = dataset_name

        self._llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # ReMem instance pool (keyed by context_id)
        self._remem_instances: Dict[int, Any] = {}

        # Accumulation buffer: {context_id: [session_text1, session_text2, ...]}
        self._pending_sessions: Dict[int, List[str]] = {}
        self._session_counts: Dict[int, int] = {}
        # Tracks whether indexing has been performed (used to detect new evaluation units)
        self._indexed_flags: Dict[int, bool] = {}

        self._setup_remem_path()
        self._remem_modules_loaded = False

    def _setup_remem_path(self):
        """Add ReMem src directory to Python path."""
        remem_src = Path(__file__).resolve().parent / "REMem" / "src"
        if not remem_src.exists():
            raise ImportError(f"ReMem source folder not found at {remem_src}")

        remem_src_str = str(remem_src)
        if remem_src_str not in sys.path:
            sys.path.insert(0, remem_src_str)

        logger.info(f"ReMem path added: {remem_src_str}")

    def _load_remem_modules(self):
        """Lazy-load ReMem modules."""
        if self._remem_modules_loaded:
            return

        from remem.remem import ReMem
        from remem.utils.config_utils import BaseConfig

        self._ReMem = ReMem
        self._BaseConfig = BaseConfig
        self._remem_modules_loaded = True

        logger.info("ReMem modules loaded successfully")

    def _get_remem_dataset_label(self) -> str:
        """Map framework dataset_name to ReMem's internal dataset label.

        This controls prompt template selection in episodic_gist_extraction:
        - "locomo_*" -> selects English templates (*_locomo)
        - "medmemorybench" -> selects Chinese templates (*_medmemorybench)
        """
        if not self.dataset_name:
            return "medmemorybench"
        name = self.dataset_name.lower()
        if name.startswith("locomo"):
            return "locomo_episodic"
        if name.startswith("longmemeval"):
            return "longmemeval"
        return "medmemorybench"

    def _get_rerank_dspy_path(self) -> Optional[str]:
        """Resolve the DSPy filter prompt file path for reranking."""
        remem_src = Path(__file__).resolve().parent / "REMem" / "src"
        dspy_path = remem_src / "remem" / "prompts" / "dspy_prompts" / "filter_llama3.3-70B-Instruct.json"
        if dspy_path.exists():
            return str(dspy_path)
        return None

    def _get_qa_prompt_template(self) -> str:
        """Route qa_prompt_template based on dataset_name."""
        if not self.dataset_name:
            return "rag_qa_unified"
        name = self.dataset_name.lower()
        if "longmemeval" in name:
            return "rag_qa_longmemeval"
        return "rag_qa_unified"

    def _build_remem_config(self, context_id: int) -> Any:
        """Build a ReMem BaseConfig object for the given context."""
        self._load_remem_modules()

        if self.working_dir:
            save_dir = os.path.join(self.working_dir, f"context_{context_id}")
        else:
            save_dir = os.path.join(
                tempfile.gettempdir(),
                "remem_medmemorybench",
                f"context_{context_id}",
            )
        os.makedirs(save_dir, exist_ok=True)

        dataset_label = self._get_remem_dataset_label()
        rerank_path = self._get_rerank_dspy_path()

        config = self._BaseConfig(
            dataset=dataset_label,
            save_dir=save_dir,

            # LLM config
            llm_name=self.model,
            llm_base_url=self._base_url,
            llm_infer_mode="online",
            temperature=self.temperature,

            # Extraction LLM (same model)
            extract_llm_label=self.model,

            # Extraction method
            extract_method=self.extract_method,

            # Graph config
            is_directed_graph=self.is_directed_graph,
            synonymy_edge_sim_threshold=self.synonymy_edge_sim_threshold,
            synonymy_edge_topk=self.synonymy_edge_topk,
            graph_type="facts_and_sim_passage_node_unidirectional",

            # Retrieval config
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.qa_top_k,
            linking_top_k=self.linking_top_k,
            damping=self.damping,
            passage_node_weight=self.passage_node_weight,

            # Agent config
            agent_fixed_tools=self.agent_fixed_tools,
            agent_max_steps=self.agent_max_steps,

            # Cache config
            force_index_from_scratch=self.force_index_from_scratch,
            force_openie_from_scratch=self.force_index_from_scratch,
            save_openie=self.save_openie,

            # API concurrency
            extraction_max_workers=self.extraction_max_workers,

            # Embedding config
            embedding_model_name=self.embedding_model,
            embedding_batch_size=self.embedding_batch_size,
            embedding_max_seq_len=self.embedding_max_seq_len,
            embedding_return_as_normalized=True,

            # Text preprocessing: use "by_session" to preserve date/role metadata
            # in chunks, matching official implementation (locomo.py:140)
            text_preprocessor_class_name=self.text_preprocessor_class_name,
            preprocess_chunk_func="by_session",
            preprocess_chunk_max_token_size=self.chunk_size_tokens,
            preprocess_chunk_overlap_token_size=self.chunk_overlap_tokens,

            # Rerank filter for fact triple filtering
            rerank_dspy_file_path=rerank_path,

            # QA configuration (aligned with official: locomo.py:144-145)
            qa_passage_prefix="",
            qa_prompt_template=self._get_qa_prompt_template(),

            # Evaluation (disabled - we use our own)
            do_eval_qa=False,
            do_eval_retrieval=False,
        )

        # BaseConfig.__post_init__ overrides save_dir based on dataset name.
        # Restore per-context save_dir for proper isolation.
        config.save_dir = save_dir

        logger.info(f"[ReMem] Config: dataset={dataset_label}, chunk_func=by_session, "
                    f"rerank={rerank_path is not None}, qa_template={self._get_qa_prompt_template()}")

        return config

    def _create_tracked_remem(self, context_id: int) -> Any:
        """Create a ReMem instance with token-tracked LLM and embedding components."""
        self._load_remem_modules()

        config = self._build_remem_config(context_id)

        logger.info(f"Creating ReMem instance for context_id={context_id}")
        logger.info(f"  extract_method: {self.extract_method}")
        logger.info(f"  save_dir: {config.save_dir}")

        tracked_llm = TrackedLLMWrapper(
            llm_client=self._llm_client,
            llm_name=self.model,
            temperature=self.temperature,
        )

        remem = self._ReMem(
            global_config=config,
            llm=tracked_llm,
            extract_llm=tracked_llm,
            qa_llm=tracked_llm,
        )

        tracked_embedding = TrackedEmbeddingWrapper(
            provider=self.embedding_provider,
            model=self.embedding_model,
            model_path=self.embedding_model_path,
            dim=self.embedding_dim,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
            batch_size=self.embedding_batch_size,
            max_seq_len=self.embedding_max_seq_len,
        )
        remem.set_embedding_model(tracked_embedding)

        logger.info(f"ReMem instance created successfully")

        return remem

    def _get_context_id(self) -> int:
        """Get current context ID."""
        return self._context_id if self._context_id is not None else 0

    def _get_remem_instance(self, context_id: int) -> Any:
        """Get or create the ReMem instance for the given context_id."""
        if context_id not in self._remem_instances:
            self._remem_instances[context_id] = self._create_tracked_remem(context_id)
        return self._remem_instances[context_id]

    def _parse_session_to_messages(self, text: str) -> List[Dict[str, str]]:
        """Parse formatted session text into structured messages for ReMem.

        Handles three formats from the evaluation framework:

        LoCoMo format (from LoCoMoSession.to_memory_text):
            DATE: 1:56 pm on 8 May, 2023
            CONVERSATION:
            Caroline said, "Hey Mel! Good to see you!"
            Melanie said, "Hi Caroline!"

        LongMemEval format (from LongMemEvalSession.to_memory_text):
            [2023-05-08]
            user: Hello there
            assistant: Hi! How can I help?

        MedMemoryBench format (from MedSession.to_memory_text):
            [2024年1月6日]

            Patient: 医生您好...
            Doctor: 您好...

        Returns List[Dict] with keys: role, content, date
        (matching official get_sessions() output for chunk_by_session).
        """
        import re

        messages = []
        lines = text.strip().split('\n')
        current_date = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            # Skip "CONVERSATION:" header
            if line.upper() == "CONVERSATION:":
                continue

            # LoCoMo: "DATE: 1:56 pm on 8 May, 2023"
            date_match = re.match(
                r'^(?:DATE|Date|日期|Time|时间)\s*[:：]\s*(.+)$', line
            )
            if date_match:
                current_date = date_match.group(1).strip()
                continue

            # LongMemEval/MedMemoryBench: "[2024年1月6日]" or "[2023-05-08]"
            bracket_date = re.match(r'^\[([^\]]+)\]$', line)
            if bracket_date:
                current_date = bracket_date.group(1).strip()
                continue

            # LoCoMo with image: 'Speaker said, "content" and shared ...'
            said_shared_match = re.match(r'^(\w+)\s+said,\s*"(.+?)"\s+and\s+shared\s+(.+)$', line)
            if said_shared_match:
                role = said_shared_match.group(1).strip()
                content = said_shared_match.group(2).strip()
                shared = said_shared_match.group(3).strip()
                messages.append({
                    "role": role,
                    "content": f"{content} [shared: {shared}]",
                    "date": current_date or "",
                })
                continue

            # LoCoMo: 'Speaker said, "content"' — handle multi-line quotes
            said_match = re.match(r'^(\w+)\s+said,\s*"(.+)', line)
            if said_match:
                role = said_match.group(1).strip()
                content_part = said_match.group(2)
                # Check if quote closes on this line
                if content_part.endswith('"'):
                    content = content_part[:-1]
                else:
                    # Multi-line: accumulate until closing quote
                    content_lines = [content_part]
                    while i < len(lines):
                        next_line = lines[i]
                        i += 1
                        if next_line.rstrip().endswith('"'):
                            content_lines.append(next_line.rstrip()[:-1])
                            break
                        content_lines.append(next_line.rstrip())
                    content = "\n".join(content_lines)
                messages.append({
                    "role": role,
                    "content": content,
                    "date": current_date or "",
                })
                continue

            # LongMemEval/MedMemoryBench: "role: content" format
            speaker_match = re.match(r'^([^:：]{1,30})\s*[:：]\s*(.+)$', line)
            if speaker_match:
                role = speaker_match.group(1).strip()
                content = speaker_match.group(2).strip()
                messages.append({
                    "role": role,
                    "content": content,
                    "date": current_date or "",
                })
                continue

            # Continuation of previous message or standalone line
            if messages:
                messages[-1]["content"] += "\n" + line
            else:
                messages.append({
                    "role": "unknown",
                    "content": line,
                    "date": current_date or "",
                })

        return messages

    def _format_input_documents(self, text: str) -> List[List[Dict[str, str]]]:
        """Format input text as structured session data for ReMem.

        A single memorize() call may contain multiple sessions (joined by
        double newlines in the evaluator's _split_sessions_into_chunks).
        We split them back into individual sessions for chunk_by_session().

        Returns List[List[Dict]] matching the official get_sessions() format:
        each session is a List[Dict] with keys: role, content, date.
        """
        import re

        text = text.strip()
        if not text:
            return []

        # Split on session boundaries:
        # LoCoMo: "DATE: ..." marks each session start
        # MedMemoryBench: "[date]" or "[Health consultation...]" marks start
        session_pattern = re.compile(
            r'(?=^DATE\s*:)|(?=^\[[^\]]+\]\s*$)',
            re.MULTILINE
        )
        parts = session_pattern.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]

        sessions = []
        for part in parts:
            messages = self._parse_session_to_messages(part)
            if messages:
                sessions.append(messages)

        # If splitting didn't work (no recognized boundaries), treat as single session
        if not sessions:
            messages = self._parse_session_to_messages(text)
            if messages:
                sessions.append(messages)

        return sessions

    def _extract_raw_question(self, text: str) -> str:
        """Extract raw question from format_query wrapped text.

        ReMem's GraphAgent needs bare questions for reasoning.
        The evaluator passes format_query output like:
          "Based on ... Question: {q}\n\nAnswer:"
        """
        import re
        match = re.search(r'Question:\s*(.+?)(?:\n\s*\n\s*Answer:|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def memorize(self, text: str, is_last_session: bool = False, **kwargs) -> MemoryBuildResult:
        """Memory construction phase.

        The framework signals via is_last_session whether this is the last session
        in the current evaluation unit:
        - Not last: accumulate only, no graph build
        - Last: accumulate then trigger graph construction

        New evaluation unit detection uses _indexed_flags to clear old state
        when a context that was already indexed receives new sessions.
        """
        context_id = self._get_context_id()

        get_usage_tracker().set_phase("memorize")

        start_time = time.time()

        if context_id not in self._pending_sessions:
            self._pending_sessions[context_id] = []
            self._session_counts[context_id] = 0

        # Detect new evaluation unit: if already indexed, reset state
        if self._indexed_flags.get(context_id, False):
            logger.info(f"[ReMem] Context {context_id} was already indexed - new evaluation unit detected")
            logger.info(f"[ReMem] Clearing old state for fresh indexing...")

            if context_id in self._pending_sessions:
                old_count = len(self._pending_sessions[context_id])
                self._pending_sessions[context_id] = []
                logger.info(f"[ReMem] Cleared {old_count} old pending sessions")

            if context_id in self._remem_instances:
                del self._remem_instances[context_id]
                logger.info(f"[ReMem] Cleared old ReMem instance")

            self._session_counts[context_id] = 0
            self._indexed_flags[context_id] = False

        self._pending_sessions[context_id].append(text)
        self._session_counts[context_id] += 1
        session_count = self._session_counts[context_id]
        pending_count = len(self._pending_sessions[context_id])

        self._memory_chunks.append(text)

        logger.info(f"[ReMem] Session {session_count} accumulated for context_id={context_id} "
                    f"(text length: {len(text)} chars, pending: {pending_count}, is_last: {is_last_session})")

        # Trigger graph construction on last session
        if is_last_session:
            logger.info(f"[ReMem] Last session received, triggering graph construction for {pending_count} sessions...")

            flush_result = self._flush_pending_sessions(context_id)

            if flush_result and flush_result.get("success"):
                time_cost = time.time() - start_time
                graph_info = flush_result.get("graph_info", {})
                openie_count = flush_result.get("openie_count", 0)
                openie_results = flush_result.get("openie_results", [])

                memory_entries = []
                for doc_result in openie_results:
                    entry = {
                        "content": doc_result.get("passage", doc_result.get("verbatim", ""))[:500],
                    }
                    if "gists" in doc_result:
                        entry["gists"] = doc_result.get("gists", [])[:5]
                    if "facts" in doc_result:
                        entry["facts"] = doc_result.get("facts", [])[:5]
                    if "extracted_entities" in doc_result:
                        entry["entities"] = doc_result.get("extracted_entities", [])[:5]
                    if "extracted_triples" in doc_result:
                        entry["triples"] = doc_result.get("extracted_triples", [])[:5]
                    memory_entries.append(entry)

                return MemoryBuildResult(
                    success=True,
                    method="remem",
                    action="index",
                    input_content=text,
                    stored_content=text,
                    memory_entries=memory_entries,
                    all_passages=[{"content": text[:500]}],
                    chunk_count=len(self._memory_chunks),
                    time_cost=time_cost,
                    extraction_result=json.dumps({
                        "status": "indexed",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "num_documents": flush_result.get("num_documents", 0),
                        "openie_count": openie_count,
                        "graph_info": graph_info,
                        "sample_gists": memory_entries[0].get("gists", [])[:3] if memory_entries else [],
                        "sample_facts": memory_entries[0].get("facts", [])[:3] if memory_entries else [],
                    }, ensure_ascii=False, default=str)[:5000],
                    extra={
                        "context_id": context_id,
                        "extract_method": self.extract_method,
                        "action": "index",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "graph_build_time": flush_result.get("time_cost", 0),
                        "graph_info": graph_info,
                    },
                )
            else:
                error_msg = flush_result.get("error", "Unknown error") if flush_result else "Flush returned None"
                time_cost = time.time() - start_time
                return MemoryBuildResult(
                    success=False,
                    method="remem",
                    action="index_failed",
                    input_content=text,
                    stored_content="",
                    memory_entries=[],
                    all_passages=[],
                    chunk_count=len(self._memory_chunks),
                    time_cost=time_cost,
                    extraction_result=f"Error: {error_msg}",
                    extra={
                        "context_id": context_id,
                        "action": "index_failed",
                        "error": error_msg,
                    },
                )
        else:
            # Not the last session; return accumulation result only
            time_cost = time.time() - start_time

            return MemoryBuildResult(
                success=True,
                method="remem",
                action="accumulate",
                input_content=text,
                stored_content=text,
                memory_entries=[],
                all_passages=[{"content": text[:500]}],
                chunk_count=len(self._memory_chunks),
                time_cost=time_cost,
                extraction_result=json.dumps({
                    "status": "accumulated",
                    "session_number": session_count,
                    "pending_count": pending_count,
                    "message": f"Session {session_count} accumulated. Graph will be built on last session.",
                }, ensure_ascii=False),
                extra={
                    "context_id": context_id,
                    "extract_method": self.extract_method,
                    "action": "accumulate",
                    "session_number": session_count,
                    "pending_sessions": pending_count,
                    "total_accumulated_chars": sum(len(s) for s in self._pending_sessions[context_id]),
                },
            )

    def _flush_pending_sessions(self, context_id: int) -> Optional[Dict[str, Any]]:
        """Batch-process accumulated sessions and build the knowledge graph.

        This is where ReMem's index() is actually called.
        """
        if context_id not in self._pending_sessions or not self._pending_sessions[context_id]:
            logger.info(f"[ReMem] No pending sessions for context_id={context_id}")
            return None

        pending_texts = self._pending_sessions[context_id]
        session_count = len(pending_texts)

        logger.info(f"[ReMem] Flushing {session_count} accumulated sessions for context_id={context_id}")

        start_time = time.time()

        try:
            remem = self._get_remem_instance(context_id)

            # Each session is converted to List[Dict] (structured messages)
            # matching official format: List[List[Dict]] passed to rag.index()
            docs = []
            for session_text in pending_texts:
                session_messages = self._format_input_documents(session_text)
                docs.extend(session_messages)

            logger.info(f"[ReMem] Combined {session_count} sessions into {len(docs)} structured documents")

            # Force fresh index build
            original_force_index = remem.global_config.force_index_from_scratch
            original_force_openie = remem.global_config.force_openie_from_scratch
            remem.global_config.force_index_from_scratch = True
            remem.global_config.force_openie_from_scratch = True

            logger.info(f"[ReMem] Calling index() with extract_method={self.extract_method}")
            remem.index(docs)

            remem.global_config.force_index_from_scratch = original_force_index
            remem.global_config.force_openie_from_scratch = original_force_openie

            # Reset ready_to_retrieve so next query refreshes retrieval objects
            remem.ready_to_retrieve = False
            logger.info(f"[ReMem] Reset ready_to_retrieve=False to refresh retrieval cache on next query")

            self._indexed_flags[context_id] = True

            graph_info = remem.get_graph_info()
            openie_results = self._load_openie_results(remem)

            time_cost = time.time() - start_time

            logger.info(f"[ReMem] Graph construction complete in {time_cost:.2f}s")
            logger.info(f"[ReMem] Graph statistics: {graph_info}")

            if openie_results:
                logger.info(f"[ReMem] OpenIE extraction samples ({len(openie_results)} total):")
                for i, doc in enumerate(openie_results[:2]):
                    logger.info(f"  Sample {i+1}:")
                    logger.info(f"    Passage: {doc.get('passage', doc.get('verbatim', ''))[:100]}...")
                    if "gists" in doc:
                        logger.info(f"    Gists: {doc.get('gists', [])[:3]}")
                    if "facts" in doc:
                        logger.info(f"    Facts: {doc.get('facts', [])[:3]}")

            self._pending_sessions[context_id] = []
            self._is_initialized = True

            return {
                "success": True,
                "session_count": session_count,
                "num_documents": len(docs),
                "graph_info": graph_info,
                "openie_count": len(openie_results) if openie_results else 0,
                "openie_results": openie_results,
                "time_cost": time_cost,
            }

        except (ImportError, ModuleNotFoundError) as e:
            # Critical: missing dependencies should not be silently swallowed.
            # Re-raise so the user gets immediate, visible feedback.
            logger.error(f"[ReMem] FATAL: Missing dependency for ReMem: {e}")
            raise

        except Exception as e:
            logger.error(f"[ReMem] Graph construction error: {e}")
            import traceback
            traceback.print_exc()

            self._pending_sessions[context_id] = []

            return {
                "success": False,
                "session_count": session_count,
                "error": str(e),
                "time_cost": time.time() - start_time,
            }

    def _load_openie_results(self, remem) -> List[Dict]:
        """Load OpenIE extraction results from disk."""
        openie_path = remem.openie_results_path
        if os.path.exists(openie_path):
            try:
                with open(openie_path, 'r', encoding='utf-8') as f:
                    openie_data = json.load(f)
                return openie_data.get("docs", [])
            except Exception as e:
                logger.warning(f"Failed to load OpenIE results: {e}")
        return []

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query phase.

        The graph should already be built during memorize() (on the last session).
        This method only performs retrieval and QA.
        """
        context_id = self._get_context_id()

        get_usage_tracker().set_phase("query")

        # Extract raw question from format_query wrapped text
        raw_question = self._extract_raw_question(question)
        logger.info(f"[ReMem] Starting query for context_id={context_id}")
        logger.info(f"[ReMem] Raw question: {raw_question[:100]}...")

        start_time = time.time()

        try:
            if not self._indexed_flags.get(context_id, False):
                logger.warning("[ReMem] No data indexed. Please call memorize() first.")
                return AgentResponse(
                    output="Error: No memory data available. Please memorize content first before querying.",
                    query_time=time.time() - start_time,
                    retrieved_count=0,
                    extra={
                        "method": "remem",
                        "error": "no_data_indexed",
                        "message": "No content has been indexed. Call memorize() before query().",
                    },
                )

            remem = self._get_remem_instance(context_id)

            if not remem.ready_to_retrieve:
                logger.info("[ReMem] Preparing retrieval objects...")
                remem.prepare_retrieval_objects()

            # Build question_metadata for official rag_for_qa alignment
            question_metadata = [kwargs.get("question_metadata", {"type": "default"})]

            if self.use_agent:
                answer, retrieved_docs, extra_info = self._query_with_agent(
                    remem, raw_question, system_message
                )
            else:
                answer, retrieved_docs, extra_info = self._query_with_rag(
                    remem, raw_question, system_message,
                    question_metadata=question_metadata,
                )

            query_time = time.time() - start_time

            logger.info(f"[ReMem] Query complete in {query_time:.2f}s")
            logger.info(f"[ReMem] Retrieved {len(retrieved_docs)} documents")
            logger.info(f"[ReMem] Answer preview: {answer[:100]}...")

            retrieved_memories = [
                {
                    "memory": doc.get("content", "")[:2000],
                    "type": "remem_retrieval",
                    "score": doc.get("score"),
                    "rank": doc.get("rank", i + 1),
                    "source": doc.get("source", "solution_docs"),
                }
                for i, doc in enumerate(retrieved_docs)
            ]
            logger.info(f"[ReMem] Converted to {len(retrieved_memories)} retrieved_memories")

            extra_data = {
                "method": "remem",
                "extract_method": self.extract_method,
                "use_agent": self.use_agent,
                **extra_info,
            }

            return AgentResponse(
                output=answer,
                query_time=query_time,
                retrieved_count=len(retrieved_docs),
                retrieved_memories=retrieved_memories,
                extra=extra_data,
            )

        except Exception as e:
            logger.error(f"[ReMem] Query error: {e}")
            import traceback
            traceback.print_exc()

            return AgentResponse(
                output=f"Error: {str(e)}",
                query_time=time.time() - start_time,
                retrieved_count=0,
                extra={"error": str(e)},
            )

    def _query_with_rag(
        self,
        remem: Any,
        question: str,
        system_message: Optional[str],
        question_metadata: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict], Dict]:
        """Standard RAG query flow using ReMem's rag_for_qa.

        Aligned with official implementation (locomo.py:248-249):
        rag.rag_for_qa(questions, gold_docs, gold_answers, selected_metrics, question_metadata=...)

        Returns:
            Tuple[answer, retrieved_docs, extra_info]
        """
        if question_metadata is None:
            question_metadata = [{"type": "default"}]

        logger.info(f"[ReMem] Calling rag_for_qa with question: {question[:50]}...")
        solutions, responses, metadata, qa_results, retrieval_results = remem.rag_for_qa(
            queries=[question],
            gold_docs=None,
            gold_answers=None,
            metrics=(),
            question_metadata=question_metadata,
            to_save=False,
        )

        logger.info(f"[ReMem] rag_for_qa returned: solutions={len(solutions) if solutions else 0}")

        if solutions and len(solutions) > 0:
            solution = solutions[0]
            answer = solution.answer if hasattr(solution, 'answer') else str(solution)

            logger.info(f"[ReMem] Solution type: {type(solution)}")
            logger.info(f"[ReMem] Solution has 'docs': {hasattr(solution, 'docs')}")

            retrieved_docs = []
            if hasattr(solution, 'docs') and solution.docs:
                has_doc_scores = hasattr(solution, 'doc_scores') and solution.doc_scores is not None and len(solution.doc_scores) > 0
                for i, doc in enumerate(solution.docs[:self.qa_top_k]):
                    score = None
                    if has_doc_scores and i < len(solution.doc_scores):
                        score = float(solution.doc_scores[i]) if solution.doc_scores[i] is not None else None
                    retrieved_docs.append({
                        "rank": i + 1,
                        "content": doc[:500] if isinstance(doc, str) else str(doc)[:500],
                        "score": score,
                    })
                logger.info(f"[ReMem] Extracted {len(retrieved_docs)} retrieved_docs from solution.docs")
            else:
                # Fallback: try responses or embedding store when solution.docs is empty
                logger.warning(f"[ReMem] solution.docs is empty, trying fallback retrieval...")
                retrieved_docs = self._fallback_retrieval(remem, question, responses)

            graph_seeds = []
            if hasattr(solution, 'graph_seeds') and solution.graph_seeds is not None:
                try:
                    graph_seeds = [str(seed) for seed in solution.graph_seeds[:5]]
                except (TypeError, ValueError):
                    graph_seeds = []

            extra_info = {
                "graph_seeds": graph_seeds,
                "qa_rationale": solution.qa_rationale if hasattr(solution, 'qa_rationale') else None,
            }

            return answer, retrieved_docs, extra_info
        else:
            return "No answer generated", [], {}

    def _query_with_agent(
        self,
        remem: Any,
        question: str,
        system_message: Optional[str],
    ) -> Tuple[str, List[Dict], Dict]:
        """Multi-step reasoning query using GraphAgent.

        Note: This is rarely invoked directly because ReMem's rag_for_qa already
        uses an internal agent strategy for episodic_gist/temporal extract methods.
        This serves as a manual fallback for explicit agent mode.

        Returns:
            Tuple[answer, retrieved_docs, extra_info]
        """
        from remem.agent.graph_agent import GraphAgent

        node_chunks_dict = self._prepare_node_chunks_dict(remem)

        agent = GraphAgent(
            llm_model=remem.qa_llm,
            node_chunks_dict=node_chunks_dict,
            remem_instance=remem,
        )

        chunk_ids, chunk_scores, agent_logs = agent.retrieve_with_agent(
            query=question,
            beam_size=self.qa_top_k,
        )

        retrieved_docs = []
        if chunk_ids and remem._chunk_embedding_store:
            hash_ids = [remem.passage_node_keys[idx] for idx in chunk_ids[:self.qa_top_k] if idx < len(remem.passage_node_keys)]
            chunk_rows = remem.chunk_embedding_store.get_rows(hash_ids)
            for i, hash_id in enumerate(hash_ids):
                if hash_id in chunk_rows:
                    retrieved_docs.append({
                        "rank": i + 1,
                        "content": chunk_rows[hash_id].get("content", "")[:500],
                        "score": chunk_scores[i] if i < len(chunk_scores) else None,
                    })

        answer = agent_logs.get("agent_answer", "") if isinstance(agent_logs, dict) else ""

        # If agent didn't produce an answer, fall back to standard QA generation
        if not answer and retrieved_docs:
            from remem.utils.misc_utils import QuerySolution

            docs_content = [d["content"] for d in retrieved_docs]
            solution = QuerySolution(
                question=question,
                docs=docs_content,
                doc_scores=[d.get("score", 0) for d in retrieved_docs],
            )

            solutions, responses, metadata = remem.answer_each_question([solution])
            if solutions and len(solutions) > 0:
                answer = solutions[0].answer if hasattr(solutions[0], 'answer') else str(solutions[0])

        extra_info = {
            "agent_logs": agent_logs if isinstance(agent_logs, dict) else {},
            "chunk_ids": chunk_ids[:10] if chunk_ids else [],
        }

        return answer, retrieved_docs, extra_info

    def _fallback_retrieval(
        self,
        remem: Any,
        question: str,
        responses: Optional[List] = None,
    ) -> List[Dict]:
        """Fallback retrieval when solution.docs is empty.

        Attempts recovery from responses, semantic search on gists store,
        or raw gist enumeration as a last resort.
        """
        retrieved_docs = []

        if responses and len(responses) > 0:
            response = responses[0]
            if isinstance(response, dict):
                if "agent_chunks" in response and response["agent_chunks"]:
                    logger.info(f"[ReMem Fallback] Found {len(response['agent_chunks'])} agent_chunks in response")
                    for i, chunk in enumerate(response["agent_chunks"][:self.qa_top_k]):
                        retrieved_docs.append({
                            "rank": i + 1,
                            "content": str(chunk)[:500],
                            "score": None,
                            "source": "response_agent_chunks",
                        })
                    return retrieved_docs

        # Try semantic search on embedding store
        try:
            if hasattr(remem, 'episodic_embedding_stores') and remem.episodic_embedding_stores:
                gists_store = remem.episodic_embedding_stores.get("gists")
                if gists_store and hasattr(gists_store, 'search'):
                    logger.info("[ReMem Fallback] Attempting semantic search on gists store...")
                    if hasattr(remem, 'embedding_model') and remem.embedding_model:
                        query_embedding = remem.embedding_model.encode([question])[0]
                        results = gists_store.search(query_embedding, top_k=self.qa_top_k)
                        if results:
                            logger.info(f"[ReMem Fallback] Found {len(results)} results from gists semantic search")
                            for i, result in enumerate(results):
                                content = result.get("content", "") if isinstance(result, dict) else str(result)
                                score = result.get("score", None) if isinstance(result, dict) else None
                                retrieved_docs.append({
                                    "rank": i + 1,
                                    "content": content[:500],
                                    "score": score,
                                    "source": "gists_semantic_search",
                                })
                            return retrieved_docs

                # Last resort: enumerate all gists
                if gists_store:
                    all_ids = gists_store.get_all_ids() if hasattr(gists_store, 'get_all_ids') else []
                    if all_ids:
                        logger.info(f"[ReMem Fallback] Getting first {self.qa_top_k} gists from store (total: {len(all_ids)})")
                        rows = gists_store.get_rows(all_ids[:self.qa_top_k])
                        for i, (hash_id, row) in enumerate(rows.items()):
                            content = row.get("content", "") if isinstance(row, dict) else str(row)
                            retrieved_docs.append({
                                "rank": i + 1,
                                "content": content[:500],
                                "score": None,
                                "source": "gists_store_fallback",
                            })
                        return retrieved_docs

        except Exception as e:
            logger.warning(f"[ReMem Fallback] Error during fallback retrieval: {e}")

        logger.warning("[ReMem Fallback] No documents retrieved from any source")
        return retrieved_docs

    def _prepare_node_chunks_dict(self, remem: Any) -> Dict[str, str]:
        """Prepare node content dictionary for GraphAgent."""
        node_chunks_dict = {}

        if remem._chunk_embedding_store:
            chunk_rows = remem.chunk_embedding_store.get_text_for_all_rows()
            for hash_id, row in chunk_rows.items():
                node_chunks_dict[hash_id] = row.get("content", "")

        if remem._phrase_embedding_store:
            phrase_rows = remem.phrase_embedding_store.get_text_for_all_rows()
            for hash_id, row in phrase_rows.items():
                node_chunks_dict[hash_id] = row.get("content", "")

        return node_chunks_dict

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._remem_instances = {}
        self._pending_sessions = {}
        self._session_counts = {}
        self._indexed_flags = {}
        logger.info("[ReMem] Agent reset, all instances and pending sessions cleared")

    def set_context_id(self, context_id: int) -> None:
        """Set the context ID."""
        super().set_context_id(context_id)
        logger.info(f"[ReMem] Context ID set to {context_id}")

    def get_info(self) -> Dict[str, Any]:
        """Get agent configuration and status info."""
        info = super().get_info()
        context_id = self._get_context_id()
        pending_count = len(self._pending_sessions.get(context_id, []))
        is_indexed = self._indexed_flags.get(context_id, False)

        info.update({
            "extract_method": self.extract_method,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "retrieval_top_k": self.retrieval_top_k,
            "qa_top_k": self.qa_top_k,
            "use_agent": self.use_agent,
            "session_batch_size": self.session_batch_size,
            "num_instances": len(self._remem_instances),
            "pending_sessions": {k: len(v) for k, v in self._pending_sessions.items()},
            "is_indexed": is_indexed,
        })
        return info
