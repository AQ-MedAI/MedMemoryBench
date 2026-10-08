"""Letta (MemGPT) agent adapter for MedMemoryBench."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import get_usage_tracker, LLMResponse

logger = logging.getLogger(__name__)


class LettaAgent(BaseAgent):
    """Adapter that uses Letta's official Agent API for memory operations.

    This implementation uses Letta's native agent-driven memory system:
    - memorize(): sends message chunks to agent which decides how to store memories
    - query(): sends question to agent which retrieves and responds using memories

    Agent lifecycle:
    - One agent per context_id (persona)
    - Agent is reused within the same persona for memory accumulation
    - Agent is deleted and recreated when switching to a new persona
    """

    METHOD_TYPE = "agentic_memory"
    BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_num: int = 5,
        context_window: int = 128000,
        embedding_model: str = "embedding-3",
        embedding_model_path: Optional[str] = None,
        embedding_dim: int = 2048,
        embedding_provider: str = "openai",
        memory_persona: str = "I am an assistant helping with medical QA while preserving long-term memory.",
        memory_human: str = "The user is a patient in a longitudinal medical dialogue setting.",
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.provider = provider
        self.api_key = api_key or os.environ.get("BIGMODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )
        self.retrieve_num = retrieve_num
        self.context_window = context_window
        self.embedding_model = embedding_model
        # Resolve embedding_model_path to absolute path
        if embedding_model_path and not os.path.isabs(embedding_model_path):
            project_root = Path(__file__).resolve().parent.parent
            # Try project root first, then parent of project root
            for base in [project_root, project_root.parent]:
                resolved = base / embedding_model_path
                if resolved.exists():
                    self.embedding_model_path = str(resolved)
                    break
            else:
                self.embedding_model_path = embedding_model_path
        else:
            self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider
        self.embedding_chunk_size = int(kwargs.get("embedding_chunk_size", 300))

        # Token limits for chunking and truncation
        self.max_input_tokens = int(kwargs.get("max_input_tokens", 8000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))
        self.max_context_tokens = int(kwargs.get("max_context_tokens", context_window))
        # Chunk size for splitting long memorize inputs
        self.memorize_chunk_tokens = int(kwargs.get("memorize_chunk_tokens", 6000))
        self.memorize_chunk_overlap_tokens = int(kwargs.get("memorize_chunk_overlap_tokens", 200))

        self.memory_persona = memory_persona
        self.memory_human = memory_human
        # Use per-process persistence_root to avoid concurrent SQLite writes
        default_root = f".tmp/letta_runtime_{os.getpid()}"
        self.persistence_root = Path(kwargs.get("persistence_root", default_root)).resolve()

        # Agent management: one agent per context_id
        self._agent_ids: Dict[int, str] = {}
        self._client = None

        self._apply_openai_compatible_env()
        self._load_letta_package()
        self._init_client()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _chunk_text_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks by token count with overlap.

        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        if not text or chunk_size <= 0:
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start = end - overlap if end < len(tokens) else end

        return chunks

    def _apply_openai_compatible_env(self) -> None:
        """Set OpenAI-compatible env vars so vendored Letta can pick up credentials."""
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
        self.persistence_root.mkdir(parents=True, exist_ok=True)
        os.environ["LETTA_DIR"] = str(self.persistence_root)

    def _load_letta_package(self) -> None:
        """Load vendored Letta package as top-level module name `letta`.

        This method forces the use of the vendored Letta package even if
        a pip-installed version exists, to avoid version conflicts.
        """
        letta_root = Path(__file__).resolve().parent / "letta"
        init_file = letta_root / "__init__.py"
        if not init_file.exists():
            raise ImportError("Letta source not found at methods/letta")

        # Remove any existing letta modules from sys.modules to avoid conflicts
        modules_to_remove = [key for key in sys.modules.keys() if key == "letta" or key.startswith("letta.")]
        for mod_name in modules_to_remove:
            del sys.modules[mod_name]

        # Load vendored letta package
        spec = importlib.util.spec_from_file_location(
            "letta",
            str(init_file),
            submodule_search_locations=[str(letta_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Failed to create import spec for vendored Letta")

        module = importlib.util.module_from_spec(spec)
        sys.modules["letta"] = module
        spec.loader.exec_module(module)

    def _init_client(self) -> None:
        """Initialize Letta client with proper settings."""
        letta_settings = importlib.import_module("letta.settings")
        if self.api_key:
            letta_settings.model_settings.openai_api_key = self.api_key
        if self.base_url:
            letta_settings.model_settings.openai_api_base = self.base_url

        letta_client = importlib.import_module("letta.client.client")
        self._LLMConfig = importlib.import_module("letta.schemas.llm_config").LLMConfig
        self._EmbeddingConfig = importlib.import_module("letta.schemas.embedding_config").EmbeddingConfig
        self._ChatMemory = importlib.import_module("letta.schemas.memory").ChatMemory
        self._client = letta_client.create_client(base_url=None, token=None)

    def _get_context_id(self) -> int:
        """Get current context ID, defaulting to 0."""
        return self._context_id if self._context_id is not None else 0

    def _record_usage_to_tracker(self, parsed_response: Dict[str, Any], phase: str, latency: float = 0.0) -> None:
        """Record Letta's token usage to the global LLMUsageTracker.

        Args:
            parsed_response: Parsed response dictionary containing 'usage' field
            phase: Either 'memorize' or 'query'
            latency: Time cost for the API call
        """
        usage = parsed_response.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            if prompt_tokens > 0 or completion_tokens > 0:
                llm_response = LLMResponse(
                    content="",
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    latency=latency,
                    model=self.model,
                )
                get_usage_tracker().set_phase(phase)
                get_usage_tracker().record(llm_response)
                logger.debug(f"Recorded Letta usage [{phase}]: prompt={prompt_tokens}, completion={completion_tokens}")

    def _build_llm_config(self):
        """Build LLMConfig for Letta agent creation."""
        endpoint = self.base_url or self.BIGMODEL_BASE_URL
        return self._LLMConfig(
            model=self.model,
            model_endpoint_type=self.provider,
            model_endpoint=endpoint,
            model_wrapper=None,
            context_window=self.context_window,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_embedding_config(self):
        """Build EmbeddingConfig for Letta agent creation."""
        if self.embedding_provider == "local" and self.embedding_model_path:
            return self._EmbeddingConfig(
                embedding_endpoint_type="local",
                embedding_endpoint=self.embedding_model_path,
                embedding_model=self.embedding_model,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=self.embedding_chunk_size,
            )
        else:
            endpoint = self.base_url or self.BIGMODEL_BASE_URL
            return self._EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint=endpoint,
                embedding_model=self.embedding_model,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=self.embedding_chunk_size,
            )

    def _create_agent(self, context_id: int) -> str:
        """Create a new Letta agent for the given context_id.

        Args:
            context_id: The context/persona ID

        Returns:
            The agent ID string
        """
        name = f"medmemorybench_letta_ctx_{context_id}_{int(time.time())}"
        memory = self._ChatMemory(persona=self.memory_persona, human=self.memory_human)

        state = self._client.create_agent(
            name=name,
            llm_config=self._build_llm_config(),
            embedding_config=self._build_embedding_config(),
            memory=memory,
        )
        agent_id = str(state.id)
        self._agent_ids[context_id] = agent_id
        logger.info(f"Created Letta agent for context {context_id}: {agent_id}")
        return agent_id

    def _delete_agent(self, context_id: int) -> None:
        """Delete the Letta agent for the given context_id.

        Args:
            context_id: The context/persona ID
        """
        agent_id = self._agent_ids.get(context_id)
        if agent_id:
            try:
                self._client.delete_agent(agent_id)
                logger.info(f"Deleted Letta agent for context {context_id}: {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to delete agent {agent_id}: {e}")
            del self._agent_ids[context_id]

    def _export_in_context_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Export the full in_context_messages (conversation buffer) for logging.

        Args:
            agent_id: The Letta agent ID

        Returns:
            List of serialized message dicts with role, content, created_at, etc.
        """
        try:
            from letta.schemas.message import TextContent
            messages = self._client.get_in_context_messages(agent_id)
            exported = []
            for msg in messages:
                entry = {
                    "role": msg.role,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                }
                if msg.content:
                    if len(msg.content) == 1:
                        if isinstance(msg.content[0], TextContent):
                            entry["content"] = msg.content[0].text
                        else:
                            entry["content"] = str(msg.content[0])
                    else:
                        parts = []
                        for part in msg.content:
                            if isinstance(part, TextContent):
                                parts.append(part.text)
                            else:
                                parts.append(str(part))
                        entry["content"] = parts
                else:
                    entry["content"] = None

                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                if msg.name:
                    entry["name"] = msg.name
                if msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id

                exported.append(entry)
            return exported
        except Exception as e:
            logger.warning(f"Failed to export in_context_messages: {e}")
            return [{"error": str(e)}]

    def _ensure_agent(self, context_id: int) -> str:
        """Ensure Letta agent exists for the given context_id.

        Returns existing agent if available, otherwise creates a new one.

        Args:
            context_id: The context/persona ID

        Returns:
            The agent ID string
        """
        existing = self._agent_ids.get(context_id)
        if existing:
            return existing
        return self._create_agent(context_id)

    def _parse_letta_response(self, response) -> Dict[str, Any]:
        """Parse LettaResponse to extract useful information.

        Args:
            response: LettaResponse object from Letta API

        Returns:
            Dictionary with parsed response data including:
            - assistant_message: The agent's response text
            - function_calls: List of function calls made
            - internal_monologue: Agent's internal thoughts
            - usage: Token usage statistics
        """
        import json

        result = {
            "assistant_message": "",
            "function_calls": [],
            "internal_monologue": [],
            "usage": {},
        }

        # Extract usage statistics
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result["usage"] = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        # Parse messages from response
        if not hasattr(response, "messages") or not response.messages:
            return result

        for msg in response.messages:
            msg_type = getattr(msg, "message_type", None)
            if msg_type is not None:
                msg_type_str = str(msg_type.value) if hasattr(msg_type, "value") else str(msg_type)
            else:
                msg_type_str = type(msg).__name__

            # Handle AssistantMessage
            if msg_type_str == "assistant_message" or "AssistantMessage" in str(type(msg)):
                content = getattr(msg, "content", None)
                if content:
                    if isinstance(content, str):
                        result["assistant_message"] = content
                    elif isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if hasattr(part, "text"):
                                text_parts.append(part.text)
                            elif isinstance(part, str):
                                text_parts.append(part)
                            elif hasattr(part, "content"):
                                text_parts.append(str(part.content))
                        result["assistant_message"] = " ".join(text_parts)
                    elif hasattr(content, "text"):
                        result["assistant_message"] = content.text
                    elif hasattr(content, "message"):
                        result["assistant_message"] = content.message

            # Handle ToolCallMessage / function_call
            elif msg_type_str == "tool_call_message" or "ToolCall" in str(type(msg)):
                tool_call = getattr(msg, "tool_call", None)
                if tool_call:
                    func_name = getattr(tool_call, "name", "")
                    func_args = getattr(tool_call, "arguments", "{}")

                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            pass

                    if func_name:
                        result["function_calls"].append({
                            "name": func_name,
                            "arguments": func_args if isinstance(func_args, dict) else str(func_args),
                        })

                        # Letta uses send_message tool to communicate with user
                        if func_name == "send_message" and isinstance(func_args, dict):
                            msg_content = func_args.get("message", "")
                            if msg_content and not result["assistant_message"]:
                                result["assistant_message"] = msg_content
                else:
                    # Fallback for legacy format
                    func_name = getattr(msg, "function_name", None) or getattr(msg, "name", "")
                    func_args = getattr(msg, "function_args", None) or getattr(msg, "arguments", {})

                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            pass

                    if func_name:
                        result["function_calls"].append({
                            "name": func_name,
                            "arguments": func_args if isinstance(func_args, dict) else str(func_args),
                        })

                        if func_name == "send_message" and isinstance(func_args, dict):
                            msg_content = func_args.get("message", "")
                            if msg_content and not result["assistant_message"]:
                                result["assistant_message"] = msg_content

            # Handle ReasoningMessage / internal_monologue
            elif msg_type_str == "reasoning_message" or "Reasoning" in str(type(msg)):
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "content", "")
                if reasoning:
                    result["internal_monologue"].append(reasoning)
            elif msg_type_str == "internal_monologue" or "InternalMonologue" in str(type(msg)):
                monologue = getattr(msg, "internal_monologue", None) or getattr(msg, "content", "")
                if monologue:
                    result["internal_monologue"].append(monologue)

            # Handle ToolReturnMessage / function_return
            elif msg_type_str == "tool_return_message" or "ToolReturn" in str(type(msg)):
                func_return = getattr(msg, "tool_return", None) or getattr(msg, "function_return", None)
                if func_return and result["function_calls"]:
                    result["function_calls"][-1]["return"] = str(func_return)[:1000]
            elif msg_type_str == "function_return" or "FunctionReturn" in str(type(msg)):
                func_return = getattr(msg, "function_return", None)
                if func_return and result["function_calls"]:
                    result["function_calls"][-1]["return"] = str(func_return)[:1000]

        return result

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store memory using Letta's official agent API.

        Sends message directly to the Letta agent (following official design).
        The agent will internally decide what to store using archival_memory_insert.

        For long inputs, text is split into chunks and sent sequentially.
        """
        context_id = self._get_context_id()
        agent_id = self._ensure_agent(context_id)

        # Split input into chunks if needed
        chunks = self._chunk_text_by_tokens(
            text,
            self.memorize_chunk_tokens,
            self.memorize_chunk_overlap_tokens
        )

        if not chunks:
            chunks = [text] if text else []

        total_time_cost = 0.0
        all_memory_entries = []
        all_passages = []
        all_function_calls = []
        all_internal_monologue = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_assistant_message = ""

        for chunk_idx, chunk in enumerate(chunks):
            # Send chunk directly to agent (official Letta design - agent decides what to store)
            start_time = time.time()
            try:
                response = self._client.user_message(agent_id=agent_id, message=chunk)
                chunk_time = time.time() - start_time
                total_time_cost += chunk_time
            except Exception as e:
                logger.error(f"Letta user_message failed for chunk {chunk_idx}: {e}")
                return MemoryBuildResult(
                    success=False,
                    method="letta",
                    action="user_message",
                    input_content=text,
                    stored_content="",
                    extraction_result=f"[Error at chunk {chunk_idx}: {e}]",
                    all_passages=all_passages,
                    memory_entries=all_memory_entries,
                    chunk_count=chunk_idx,
                    time_cost=total_time_cost,
                    extra={"agent_id": agent_id, "error": str(e), "failed_chunk": chunk_idx},
                )

            # Parse response
            parsed = self._parse_letta_response(response)

            # Record token usage
            self._record_usage_to_tracker(parsed, "memorize", chunk_time)

            # Accumulate usage
            chunk_usage = parsed.get("usage", {})
            total_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += chunk_usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += chunk_usage.get("total_tokens", 0)

            # Extract memory operations
            for func_call in parsed["function_calls"]:
                all_function_calls.append(func_call)
                if func_call["name"] in ["archival_memory_insert", "core_memory_append", "core_memory_replace"]:
                    args = func_call.get("arguments", {})
                    if isinstance(args, dict):
                        content = args.get("content", args.get("new_content", ""))
                    else:
                        content = str(args)

                    all_memory_entries.append({
                        "event": "ADD",
                        "function": func_call["name"],
                        "memory": content[:500] if content else "",
                        "chunk_index": chunk_idx,
                    })
                    all_passages.append({
                        "text": content,
                        "function": func_call["name"],
                        "chunk_index": chunk_idx,
                    })

            all_internal_monologue.extend(parsed.get("internal_monologue", []))
            if parsed.get("assistant_message"):
                last_assistant_message = parsed["assistant_message"]

        # Build extraction result summary
        extraction_parts = []
        if all_internal_monologue:
            extraction_parts.append("Agent thinking:\n" + "\n".join(all_internal_monologue[:5]))
        if last_assistant_message:
            extraction_parts.append(f"Agent response: {last_assistant_message[:500]}")
        extraction_result = "\n\n".join(extraction_parts) if extraction_parts else "[No extraction details]"

        self._memory_chunks.append(text)
        self._is_initialized = True

        # Export in_context_messages (conversation buffer) snapshot
        in_context_messages_snapshot = self._export_in_context_messages(agent_id)

        return MemoryBuildResult(
            success=True,
            method="letta",
            action="user_message",
            input_content=text,
            stored_content=last_assistant_message,
            extraction_result=extraction_result,
            all_passages=all_passages,
            memory_entries=all_memory_entries,
            chunk_count=len(chunks),
            time_cost=total_time_cost,
            extra={
                "agent_id": agent_id,
                "function_calls": all_function_calls,
                "internal_monologue": all_internal_monologue,
                "usage": total_usage,
                "chunks_processed": len(chunks),
                "in_context_messages": in_context_messages_snapshot,
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query using Letta's official agent API.

        Sends the question to the Letta agent which will use its internal
        archival_memory_search tool to retrieve relevant memories and respond.

        Question is truncated if exceeding max_question_tokens.
        """
        context_id = self._get_context_id()
        agent_id = self._ensure_agent(context_id)

        # Truncate question if needed
        bounded_question = self._truncate_to_tokens(question, self.max_question_tokens)

        # Build query message
        if system_message:
            query_message = f"{system_message}\n\nQuestion: {bounded_question}"
        else:
            query_message = bounded_question

        start_time = time.time()
        try:
            # Switch to restricted context_window for query phase (token budget truncation)
            # Only update once per agent (build phase is always completed before query phase)
            if self.max_context_tokens and self.max_context_tokens < self.context_window:
                if not getattr(self, '_query_context_applied', {}).get(agent_id):
                    query_llm_config = self._build_llm_config().model_copy(
                        update={"context_window": self.max_context_tokens}
                    )
                    self._client.update_agent(agent_id, llm_config=query_llm_config)
                    if not hasattr(self, '_query_context_applied'):
                        self._query_context_applied = {}
                    self._query_context_applied[agent_id] = True

            response = self._client.user_message(agent_id=agent_id, message=query_message)
            query_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Letta user_message failed: {e}")
            return AgentResponse(
                output=f"[Error: {e}]",
                query_time=0.0,
                retrieved_count=0,
                extra={
                    "method": "letta",
                    "agent_id": agent_id,
                    "error": str(e),
                },
            )

        parsed = self._parse_letta_response(response)
        self._record_usage_to_tracker(parsed, "query", query_time)

        # Extract retrieved memories from search function calls
        retrieved_memories = []
        search_tool_names = {"archival_memory_search", "conversation_search", "conversation_search_date"}
        for func_call in parsed["function_calls"]:
            if func_call["name"] in search_tool_names:
                args = func_call.get("arguments", {})
                search_query = args.get("query", "") if isinstance(args, dict) else str(args)
                search_results = func_call.get("return", "")

                # Truncate retrieved memory results
                truncated_results = self._truncate_to_tokens(
                    search_results, self.max_context_tokens // 4
                ) if search_results else ""

                retrieved_memories.append({
                    "memory": truncated_results if truncated_results else f"Search: {search_query}",
                    "type": func_call["name"],
                    "query": search_query,
                })

        # Export in_context_messages (conversation buffer) snapshot
        in_context_messages_snapshot = self._export_in_context_messages(agent_id)

        return AgentResponse(
            output=parsed["assistant_message"],
            query_time=query_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "letta",
                "agent_id": agent_id,
                "function_calls": parsed["function_calls"],
                "internal_monologue": parsed["internal_monologue"],
                "usage": parsed["usage"],
                "in_context_messages": in_context_messages_snapshot,
            },
        )

    def reset(self) -> None:
        """Reset agent state and delete all created agents."""
        # Delete all agents for this instance
        for context_id in list(self._agent_ids.keys()):
            self._delete_agent(context_id)

        super().reset()
        self._agent_ids = {}

    def set_context_id(self, context_id: int) -> None:
        """Set context ID and handle agent lifecycle.

        When switching to a different context (persona), the old agent is deleted
        to ensure clean separation between personas.
        """
        old_context_id = self._context_id

        # If switching to a different context, delete the old agent
        if old_context_id is not None and old_context_id != context_id:
            if old_context_id in self._agent_ids:
                self._delete_agent(old_context_id)
                logger.info(f"Switched context from {old_context_id} to {context_id}, deleted old agent")

        super().set_context_id(context_id)
