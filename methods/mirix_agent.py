"""MIRIX Agent - Multi-agent memory system with six-component memory architecture."""

import asyncio
import gc
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add MIRIX to path
MIRIX_PATH = os.path.join(os.path.dirname(__file__), "MIRIX")
if MIRIX_PATH not in sys.path:
    sys.path.insert(0, MIRIX_PATH)

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    get_usage_tracker,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class MIRIXAgent(BaseAgent):
    """MIRIX Agent for intelligent multi-component memory management.

    MIRIX provides a six-component memory system:
    - Core Memory: Essential persona and human blocks
    - Episodic Memory: Event/conversation memories with timestamps
    - Semantic Memory: Concept/fact knowledge
    - Procedural Memory: Step/procedure memories
    - Resource Memory: Resource/file memories
    - Knowledge Vault: Sensitive information storage
    """

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # Embedding config
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        embedding_model_path: Optional[str] = None,
        embedding_dim: int = 1536,
        embedding_device: Optional[str] = None,  # For local embedding: "cuda", "cpu", "mps"
        # Memory retrieval config
        retrieve_num: int = 5,  # Number of memories to retrieve per memory type
        # Chunking config
        memorize_chunk_tokens: int = 2500,
        memorize_chunk_overlap_tokens: int = 200,
        # Context limits
        max_input_tokens: int = 8000,
        max_question_tokens: int = 4096,
        max_context_tokens: int = 120000,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # Store config
        self._provider = provider
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Embedding config
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_device = embedding_device

        # Memory config
        self.retrieve_num = retrieve_num

        # Chunking config
        self.memorize_chunk_tokens = memorize_chunk_tokens
        self.memorize_chunk_overlap_tokens = memorize_chunk_overlap_tokens

        # Limits
        self.max_input_tokens = max_input_tokens
        self.max_question_tokens = max_question_tokens
        self.max_context_tokens = max_context_tokens

        # LLM client for fallback Q&A (uses utils/llm_client for token tracking)
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # MIRIX components (lazy initialization)
        self._client = None
        self._meta_agent = None
        self._is_client_owner = True  # Track if we own the client for cleanup

        # Token tracking for MIRIX internal calls (memory phase)
        self._mirix_input_tokens = 0
        self._mirix_output_tokens = 0
        self._mirix_step_count = 0

        # Initialize MIRIX
        self._init_mirix()

    def _get_event_loop(self):
        """Get or create event loop for async operations."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        loop = self._get_event_loop()
        return loop.run_until_complete(coro)

    def _init_mirix(self) -> None:
        """Initialize MIRIX LocalClient and MetaAgent."""
        logger.info("[MIRIXAgent] Initializing MIRIX...")

        try:
            # Import MIRIX components
            from mirix.local_client.local_client import LocalClient
            from mirix.schemas.llm_config import LLMConfig
            from mirix.schemas.embedding_config import EmbeddingConfig
            from mirix.schemas.agent import CreateMetaAgent
            from mirix.server.server import ensure_tables_created

            # Ensure database tables are created first
            logger.info("[MIRIXAgent] Ensuring database tables are created...")
            self._run_async(ensure_tables_created())

            # Build LLM config
            llm_config = LLMConfig(
                model=self.model,
                model_endpoint_type=self._get_mirix_endpoint_type(self._provider),
                model_endpoint=self._base_url,
                context_window=self.max_context_tokens,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Build embedding config
            embedding_config = self._build_embedding_config()

            # Create LocalClient
            async def create_client():
                client = await LocalClient.create(
                    debug=False,
                    default_llm_config=llm_config,
                    default_embedding_config=embedding_config,
                )
                return client

            self._client = self._run_async(create_client())
            self._is_client_owner = True
            logger.info("[MIRIXAgent] LocalClient created successfully")

            # Create MetaAgent (always create new one, don't reuse)
            self._create_meta_agent(llm_config, embedding_config)

        except Exception as e:
            logger.error(f"[MIRIXAgent] Failed to initialize MIRIX: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_meta_agent(self, llm_config, embedding_config) -> None:
        """Create a new MetaAgent."""
        from mirix.schemas.agent import CreateMetaAgent

        async def setup_meta_agent():
            logger.info("[MIRIXAgent] Creating new MetaAgent...")
            create_request = CreateMetaAgent(
                llm_config=llm_config,
                embedding_config=embedding_config,
            )
            return await self._client.create_meta_agent(request=create_request)

        self._meta_agent = self._run_async(setup_meta_agent())
        if self._meta_agent:
            logger.info(f"[MIRIXAgent] MetaAgent ready: {self._meta_agent.id}")
        else:
            logger.warning("[MIRIXAgent] MetaAgent creation returned None")

    def _get_mirix_endpoint_type(self, provider: str) -> str:
        """Convert provider name to MIRIX endpoint type."""
        mapping = {
            "openai": "openai",
            "azure": "azure_openai",
            "anthropic": "anthropic",
            "google": "google_ai",
            "gemini": "google_ai",
        }
        return mapping.get(provider.lower(), "openai")

    def _build_embedding_config(self):
        """Build MIRIX embedding config based on provider."""
        from mirix.schemas.embedding_config import EmbeddingConfig

        provider = self.embedding_provider.lower()

        if provider == "local":
            # Local sentence-transformers model (no API required)
            model_path = self.embedding_model_path or self.embedding_model
            logger.info(f"[MIRIXAgent] Using local embedding: {model_path}")
            return EmbeddingConfig(
                embedding_model=model_path,
                embedding_endpoint_type="local",
                embedding_endpoint=self.embedding_device,  # Device: "cuda", "cpu", "mps", or None for auto
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=512,
            )
        elif provider in ("huggingface", "hugging-face"):
            # HuggingFace TEI server (requires running TEI service)
            model_path = self.embedding_model_path or self.embedding_model
            tei_url = self.embedding_device  # Reuse device field for TEI URL
            if tei_url and tei_url.startswith("http"):
                logger.info(f"[MIRIXAgent] Using HuggingFace TEI: {model_path} at {tei_url}")
                return EmbeddingConfig(
                    embedding_model=model_path,
                    embedding_endpoint_type="hugging-face",
                    embedding_endpoint=tei_url,
                    embedding_dim=self.embedding_dim,
                    embedding_chunk_size=512,
                )
            else:
                # Fall back to local if no TEI URL provided
                logger.info(f"[MIRIXAgent] No TEI URL provided, using local embedding: {model_path}")
                return EmbeddingConfig(
                    embedding_model=model_path,
                    embedding_endpoint_type="local",
                    embedding_endpoint=None,
                    embedding_dim=self.embedding_dim,
                    embedding_chunk_size=512,
                )
        elif provider == "openai":
            # OpenAI embedding API
            base_url = self._base_url or "https://api.openai.com/v1"
            logger.info(f"[MIRIXAgent] Using OpenAI embedding: {self.embedding_model} at {base_url}")
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type="openai",
                embedding_endpoint=base_url,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=8191,
            )
        elif provider == "ollama":
            # Ollama embedding
            ollama_url = self.embedding_model_path or "http://localhost:11434"
            logger.info(f"[MIRIXAgent] Using Ollama embedding: {self.embedding_model} at {ollama_url}")
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type="ollama",
                embedding_endpoint=ollama_url,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=512,
            )
        else:
            # Default to OpenAI-compatible
            base_url = self._base_url or "https://api.openai.com/v1"
            logger.info(f"[MIRIXAgent] Using OpenAI-compatible embedding: {self.embedding_model}")
            return EmbeddingConfig(
                embedding_model=self.embedding_model,
                embedding_endpoint_type="openai",
                embedding_endpoint=base_url,
                embedding_dim=self.embedding_dim,
                embedding_chunk_size=300,
            )

    def _split_text_into_chunks(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Split text into chunks with optional overlap."""
        if not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._tokenizer.decode(chunk_tokens))

            if end >= len(tokens):
                break

            # Move start with overlap
            start = end - overlap_tokens if overlap_tokens > 0 else end

        return chunks

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _record_mirix_usage(self, usage, latency: float = 0.0) -> None:
        """Record MIRIX usage statistics to global tracker."""
        if usage is None:
            return

        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        step_count = getattr(usage, 'step_count', 0)

        self._mirix_input_tokens += input_tokens
        self._mirix_output_tokens += output_tokens
        self._mirix_step_count += step_count

        # Record to global tracker
        if input_tokens > 0 or output_tokens > 0:
            response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                model=self.model,
            )
            get_usage_tracker().record(response)
            logger.debug(f"[MIRIXAgent] Token usage: in={input_tokens}, out={output_tokens}, steps={step_count}")

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store text into MIRIX memory system.

        This uses MIRIX's native memory extraction pipeline through send_message().
        MIRIX internally routes content to appropriate memory components (Episodic,
        Semantic, Procedural, etc.) based on content analysis.

        Token tracking is done via MirixUsageStatistics from MirixResponse.
        """
        start_time = time.time()

        # Reset MIRIX token counters for this operation
        self._mirix_input_tokens = 0
        self._mirix_output_tokens = 0
        self._mirix_step_count = 0

        # Truncate input if needed
        bounded_text = self._truncate_to_tokens(text, self.max_input_tokens)

        # Split text into chunks
        chunks = self._split_text_into_chunks(
            bounded_text,
            max_tokens=self.memorize_chunk_tokens,
            overlap_tokens=self.memorize_chunk_overlap_tokens
        )

        memory_entries = []
        all_passages = []
        chunk_responses = []

        async def process_chunks():
            nonlocal memory_entries, all_passages, chunk_responses

            for i, chunk in enumerate(chunks):
                chunk_start = time.time()
                try:
                    # Send message to MIRIX for memory extraction
                    # MIRIX's MetaAgent will automatically route content to appropriate
                    # memory components (Episodic, Semantic, Procedural, etc.)
                    # NOTE: We don't pass user_id here - MIRIX uses the LocalClient's
                    # default user which was created during initialization.
                    response = await self._client.send_message(
                        agent_id=self._meta_agent.id,
                        role="user",
                        message=chunk,
                    )

                    chunk_latency = time.time() - chunk_start

                    # Extract usage statistics from MirixResponse
                    if response and hasattr(response, 'usage') and response.usage:
                        self._record_mirix_usage(response.usage, chunk_latency)

                    # Record passage info for logging
                    passage_info = {
                        "chunk_index": i,
                        "chunk_tokens": self.count_tokens(chunk),
                        "content_preview": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                        "latency": round(chunk_latency, 3),
                    }

                    # Extract response messages for logging
                    # MIRIX MirixResponse has messages as List[Union[ToolCallMessage, ToolReturnMessage, AssistantMessage, ...]]
                    # Each message has 'message_type' field to identify the type
                    # - ToolCallMessage: has tool_call.name, tool_call.arguments (JSON string)
                    # - ToolReturnMessage: has tool_return (string)
                    # - AssistantMessage: has content (string or list)
                    response_messages = []
                    if response and hasattr(response, 'messages'):
                        logger.debug(f"[MIRIXAgent] Response has {len(response.messages)} messages")
                        for msg in response.messages:
                            msg_type = getattr(msg, 'message_type', None)
                            if hasattr(msg_type, 'value'):
                                msg_type = msg_type.value

                            msg_content = ""
                            func_name = ""
                            func_args = {}

                            if msg_type == 'tool_call_message':
                                # ToolCallMessage has tool_call with name and arguments
                                tool_call = getattr(msg, 'tool_call', None)
                                if tool_call:
                                    func_name = getattr(tool_call, 'name', 'unknown')
                                    args_str = getattr(tool_call, 'arguments', '{}')
                                    try:
                                        if isinstance(args_str, str):
                                            func_args = json.loads(args_str) if args_str else {}
                                        elif isinstance(args_str, dict):
                                            func_args = args_str
                                    except Exception as e:
                                        logger.debug(f"[MIRIXAgent] Failed to parse tool call args: {e}")
                                        func_args = {"raw": str(args_str)[:200]}

                                    args_preview = str(func_args)[:200]
                                    msg_content = f"{func_name}({args_preview}{'...' if len(str(func_args)) > 200 else ''})"

                                    response_messages.append({
                                        "type": "tool_call",
                                        "content": msg_content,
                                        "function_name": func_name,
                                        "arguments": func_args,
                                    })

                                    # Extract memory entries from tool calls
                                    # MIRIX memory-related function names
                                    memory_functions = [
                                        'trigger_memory_update', 'trigger_memory_update_with_instruction',
                                        'episodic_memory_insert', 'episodic_memory_merge', 'episodic_memory_replace',
                                        'semantic_memory_insert', 'semantic_memory_update',
                                        'procedural_memory_insert', 'procedural_memory_update',
                                        'resource_memory_insert', 'knowledge_vault_insert',
                                        'finish_memory_update', 'check_episodic_memory', 'check_semantic_memory',
                                    ]
                                    if func_name in memory_functions or '_memory' in func_name.lower():
                                        memory_entries.append({
                                            "type": "mirix_memory_operation",
                                            "function": func_name,
                                            "arguments": func_args,
                                            "chunk_index": i,
                                        })

                            elif msg_type == 'tool_return_message':
                                # ToolReturnMessage has tool_return
                                tool_return = getattr(msg, 'tool_return', '') or ''
                                msg_content = str(tool_return)[:500]
                                response_messages.append({
                                    "type": "tool_return",
                                    "content": msg_content,
                                })

                            elif msg_type == 'assistant_message':
                                # AssistantMessage has content (string or list)
                                content = getattr(msg, 'content', '')
                                if isinstance(content, str):
                                    msg_content = content[:500]
                                elif isinstance(content, list):
                                    # Extract text from TextContent items
                                    texts = []
                                    for c in content:
                                        if hasattr(c, 'text'):
                                            texts.append(str(c.text))
                                    msg_content = ' '.join(texts)[:500]
                                response_messages.append({
                                    "type": "assistant_message",
                                    "content": msg_content,
                                })

                            elif msg_type == 'reasoning_message':
                                # ReasoningMessage has reasoning field
                                reasoning = getattr(msg, 'reasoning', '') or ''
                                msg_content = str(reasoning)[:500]
                                response_messages.append({
                                    "type": "reasoning_message",
                                    "content": msg_content,
                                })

                            elif msg_type == 'internal_monologue':
                                # Legacy LegacyInternalMonologue
                                monologue = getattr(msg, 'internal_monologue', '') or ''
                                msg_content = str(monologue)[:500]
                                response_messages.append({
                                    "type": "internal_monologue",
                                    "content": msg_content,
                                })

                            else:
                                # Other message types
                                response_messages.append({
                                    "type": str(msg_type) if msg_type else "unknown",
                                    "content": str(msg)[:200],
                                })

                    passage_info["response_messages"] = response_messages[:10]  # Limit for logging
                    all_passages.append(passage_info)

                    # Store chunk response summary
                    chunk_responses.append({
                        "chunk_index": i,
                        "message_count": len(response_messages),
                        "usage": {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response and response.usage else 0,
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response and response.usage else 0,
                        } if response and response.usage else {},
                    })

                except Exception as e:
                    logger.warning(f"[MIRIXAgent] Failed to process chunk {i}: {e}")
                    all_passages.append({
                        "chunk_index": i,
                        "error": str(e),
                        "chunk_tokens": self.count_tokens(chunk),
                    })
                    memory_entries.append({
                        "type": "error",
                        "error": str(e),
                        "chunk_index": i,
                    })

        # Run async memory processing
        self._run_async(process_chunks())

        # Update internal state
        self._memory_chunks.append(text)
        self._is_initialized = True

        time_cost = time.time() - start_time

        # Build extraction result summary
        extraction_summary = f"Processed {len(chunks)} chunks through MIRIX MetaAgent\n"
        extraction_summary += f"Total tokens: input={self._mirix_input_tokens}, output={self._mirix_output_tokens}\n"
        extraction_summary += f"Step count: {self._mirix_step_count}\n\n"

        for i, passage in enumerate(all_passages[:5]):
            extraction_summary += f"Chunk {i}: {passage.get('chunk_tokens', 0)} tokens"
            if 'error' in passage:
                extraction_summary += f" [ERROR: {passage['error']}]"
            else:
                extraction_summary += f" [{len(passage.get('response_messages', []))} responses]"
            extraction_summary += "\n"

        return MemoryBuildResult(
            success=True,
            method="mirix",
            action="add_to_memory",
            input_content=text,
            stored_content=bounded_text,
            memory_entries=memory_entries,
            chunk_count=len(chunks),
            time_cost=time_cost,
            extraction_result=extraction_summary,
            all_passages=all_passages,
            extra={
                "mirix_input_tokens": self._mirix_input_tokens,
                "mirix_output_tokens": self._mirix_output_tokens,
                "mirix_step_count": self._mirix_step_count,
                "total_input_tokens": self.count_tokens(text),
                "bounded_input_tokens": self.count_tokens(bounded_text),
                "chunk_responses": chunk_responses,
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
            }
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent with memory-augmented response.

        Uses MIRIX's official extract_memory_for_system_prompt() API which
        internally: (1) extracts topics from question via LLM,
        (2) retrieves from all 6 memory types (episodic, semantic, procedural,
        resource, knowledge_vault, core), (3) builds a formatted system prompt.
        Then uses external LLM to generate the final answer.

        If extract_memory_for_system_prompt fails, falls back to retrieve_memory API.
        """
        start_time = time.time()

        full_question = self._truncate_to_tokens(question, self.max_question_tokens)
        full_question = f"{full_question}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        memory_prompt = ""
        retrieved_memories = []

        async def do_extract():
            nonlocal memory_prompt, retrieved_memories
            try:
                memory_prompt = await self._client.extract_memory_for_system_prompt(
                    agent_id=self._meta_agent.id,
                    message=full_question,
                )
                if memory_prompt:
                    retrieved_memories.append({
                        "memory": memory_prompt[:2000],
                        "type": "mirix_system_prompt",
                        "source": "extract_memory_for_system_prompt",
                    })
            except Exception as e:
                logger.warning(f"[MIRIXAgent] extract_memory_for_system_prompt failed: {e}, trying retrieve_memory fallback...")
                # Fallback: use retrieve_memory API which correctly handles user
                try:
                    result = await self._client.retrieve_memory(
                        agent_id=self._meta_agent.id,
                        query=full_question,
                        memory_type="all",
                        search_method="bm25",
                        limit=self.retrieve_num,
                    )
                    if result and result.get("count", 0) > 0:
                        memory_items = result.get("results", [])
                        memory_texts = []
                        for item in memory_items:
                            mem_type = item.get("memory_type", "unknown")
                            summary = item.get("summary", "")
                            details = item.get("details", "")
                            content = summary or details
                            if content:
                                memory_texts.append(f"[{mem_type}] {content}")
                                retrieved_memories.append({
                                    "memory": content[:500],
                                    "type": mem_type,
                                    "source": "retrieve_memory_fallback",
                                })
                        if memory_texts:
                            memory_prompt = "<memory>\n" + "\n".join(memory_texts) + "\n</memory>"
                except Exception as fallback_err:
                    logger.error(f"[MIRIXAgent] retrieve_memory fallback also failed: {fallback_err}")
                    memory_prompt = ""

        self._run_async(do_extract())

        if memory_prompt:
            full_system = f"{system_message}\n\n{memory_prompt}" if system_message else memory_prompt
        else:
            full_system = system_message

        # Enforce max_context_tokens: truncate system message if total exceeds limit
        question_tokens = len(self._tokenizer.encode(full_question))
        reserved_output = self.max_tokens
        max_system_tokens = self.max_context_tokens - question_tokens - reserved_output
        if max_system_tokens > 0 and full_system:
            full_system = self._truncate_to_tokens(full_system, max_system_tokens)

        messages = format_messages(full_question, full_system)
        response = self._llm_client.chat(messages)

        query_time = time.time() - start_time

        return AgentResponse(
            output=response.content,
            query_time=query_time,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={
                "method": "mirix_extract",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
                "memory_prompt_tokens": self.count_tokens(memory_prompt) if memory_prompt else 0,
                "tokens_used": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
            }
        )

    def reset(self) -> None:
        """Reset agent state for new persona evaluation.

        Keeps the existing LocalClient (preserving client_id/organization_id consistency)
        and only clears memory data + recreates the MetaAgent and its sub-agents.

        This ensures:
        - Complete data isolation between different personas
        - No agent_id/client_id mismatch after reset (the root cause of NoResultFound errors)
        """
        logger.info("[MIRIXAgent] Resetting agent state...")
        super().reset()

        # Reset token counters
        self._mirix_input_tokens = 0
        self._mirix_output_tokens = 0
        self._mirix_step_count = 0

        if not self._client:
            logger.warning("[MIRIXAgent] No client exists, performing full initialization")
            self._init_mirix()
            return

        try:
            async def cleanup_and_recreate():
                client_id = self._client.client_id

                # Step 1: Delete all memory data (episodic, semantic, procedural, etc.)
                try:
                    await self._client.server.client_manager.delete_memories_by_client_id(client_id)
                    logger.info(f"[MIRIXAgent] Deleted all memories for client {client_id}")
                except Exception as e:
                    logger.warning(f"[MIRIXAgent] Bulk delete memories failed: {e}, trying fallback...")
                    await self._fallback_cleanup_memories()

                # Step 2: Delete all agents (MetaAgent + sub-agents + orphans)
                try:
                    agents = await self._client.list_agents()
                    for agent in agents:
                        try:
                            await self._client.delete_agent(agent.id)
                        except Exception as e:
                            logger.debug(f"[MIRIXAgent] Failed to delete agent {agent.id}: {e}")
                    if agents:
                        logger.info(f"[MIRIXAgent] Deleted {len(agents)} agents")
                except Exception as e:
                    logger.warning(f"[MIRIXAgent] Failed to list/delete agents: {e}")
                    # If listing fails, at least try to delete the known MetaAgent
                    if self._meta_agent:
                        try:
                            await self._client.delete_agent(self._meta_agent.id)
                        except Exception:
                            pass

            self._run_async(cleanup_and_recreate())

        except Exception as e:
            logger.warning(f"[MIRIXAgent] Cleanup failed: {e}")
            import traceback
            traceback.print_exc()

        # Step 3: Recreate MetaAgent on the SAME client (no LocalClient recreation)
        self._meta_agent = None

        from mirix.schemas.llm_config import LLMConfig
        llm_config = LLMConfig(
            model=self.model,
            model_endpoint_type=self._get_mirix_endpoint_type(self._provider),
            model_endpoint=self._base_url,
            context_window=self.max_context_tokens,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        embedding_config = self._build_embedding_config()

        try:
            self._create_meta_agent(llm_config, embedding_config)
            logger.info("[MIRIXAgent] Reset complete - fresh MetaAgent created on existing client")
        except Exception as e:
            logger.error(f"[MIRIXAgent] Failed to recreate MetaAgent: {e}")
            raise

    async def _fallback_cleanup_memories(self) -> None:
        """Fallback: delete memories individually if bulk delete fails."""
        try:
            from mirix.services.episodic_memory_manager import EpisodicMemoryManager
            from mirix.services.semantic_memory_manager import SemanticMemoryManager
            from mirix.services.procedural_memory_manager import ProceduralMemoryManager
            from mirix.services.resource_memory_manager import ResourceMemoryManager
            from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
            from mirix.services.message_manager import MessageManager

            client = self._client.client
            managers = [
                ("episodic", EpisodicMemoryManager()),
                ("semantic", SemanticMemoryManager()),
                ("procedural", ProceduralMemoryManager()),
                ("resource", ResourceMemoryManager()),
                ("knowledge_vault", KnowledgeVaultManager()),
                ("messages", MessageManager()),
            ]
            for name, manager in managers:
                try:
                    count = await manager.delete_by_client_id(actor=client)
                    logger.debug(f"[MIRIXAgent] Fallback deleted {count} {name} records")
                except Exception as e:
                    logger.warning(f"[MIRIXAgent] Failed to delete {name}: {e}")
        except Exception as e:
            logger.error(f"[MIRIXAgent] Fallback cleanup failed: {e}")


    def cleanup(self) -> None:
        """Full cleanup of MIRIX resources (call when completely done with agent).

        This should be called when the agent is no longer needed, to release
        all database connections and resources.
        """
        logger.info("[MIRIXAgent] Performing full cleanup...")

        # Delete MetaAgent and all agents
        if self._meta_agent and self._client:
            try:
                async def full_cleanup():
                    # Delete all agents
                    agents = await self._client.list_agents()
                    for agent in agents:
                        try:
                            await self._client.delete_agent(agent.id)
                        except Exception:
                            pass

                self._run_async(full_cleanup())
            except Exception as e:
                logger.warning(f"[MIRIXAgent] Cleanup error: {e}")

        self._meta_agent = None
        self._client = None
        self._is_client_owner = False

        # Force garbage collection
        gc.collect()

        logger.info("[MIRIXAgent] Full cleanup complete")

    def set_context_id(self, context_id: int) -> None:
        """Set context ID for distinguishing personas.

        Note: MIRIX uses the LocalClient's default user for all operations,
        so we don't need to pass user_id to MIRIX APIs. The context_id is
        used for logging and internal tracking purposes.
        """
        old_context = self._context_id
        super().set_context_id(context_id)

        # If context changed, log it for debugging
        if old_context != context_id and old_context is not None:
            logger.info(f"[MIRIXAgent] Context changed from {old_context} to {context_id}")

    def get_info(self) -> Dict[str, Any]:
        """Get agent info."""
        info = super().get_info()
        info.update({
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model_path": self.embedding_model_path,
            "embedding_dim": self.embedding_dim,
            "embedding_device": self.embedding_device,
            "retrieve_num": self.retrieve_num,
            "memorize_chunk_tokens": self.memorize_chunk_tokens,
            "memorize_chunk_overlap_tokens": self.memorize_chunk_overlap_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_question_tokens": self.max_question_tokens,
            "max_context_tokens": self.max_context_tokens,
            "meta_agent_id": self._meta_agent.id if self._meta_agent else None,
            "mirix_input_tokens": self._mirix_input_tokens,
            "mirix_output_tokens": self._mirix_output_tokens,
            "mirix_step_count": self._mirix_step_count,
        })
        return info

    @property
    def memory_count(self) -> int:
        """Get memory count."""
        return len(self._memory_chunks)

    def __del__(self):
        """Destructor - attempt cleanup on garbage collection."""
        try:
            if hasattr(self, '_is_client_owner') and self._is_client_owner:
                self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction
