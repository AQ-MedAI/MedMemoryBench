import json
import logging
import os
import time
import warnings
from typing import Dict, List, Optional

from openai import OpenAI

from methods.mem0.configs.llms.base import BaseLlmConfig
from methods.mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

        self._http_client = None  # Keep reference for proper cleanup
        self._init_client()

    def _create_http_client(self):
        """Create a new httpx client with optimized settings for proxy reliability."""
        import httpx
        import socket

        # Timeout for LLM API calls (large models like Qwen3-235B need longer read time)
        timeout = httpx.Timeout(timeout=300.0, connect=30.0, read=300.0, write=30.0)

        # Minimal connection pooling - proxy connections are unreliable for keepalive
        limits = httpx.Limits(
            max_keepalive_connections=1,  # Minimal keepalive
            max_connections=5,
            keepalive_expiry=10.0,  # Very short expiry - proxy connections die fast
        )

        # Create transport with socket-level options for better dead connection detection
        transport = httpx.HTTPTransport(
            retries=0,  # We handle retries ourselves
            limits=limits,
            socket_options=[
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),  # Enable TCP keepalive
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),  # Keepalive interval: 10s
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),  # Max keepalive probes
            ]
        )

        return httpx.Client(timeout=timeout, transport=transport)

    def _init_client(self):
        """Initialize or reinitialize the OpenAI client."""
        # Close existing client if any
        self.close()

        self._http_client = self._create_http_client()

        if os.environ.get("OPENROUTER_API_KEY"):
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
                http_client=self._http_client,
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )
            if os.environ.get("OPENAI_API_BASE"):
                warnings.warn(
                    "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                    "Please use 'OPENAI_BASE_URL' instead.",
                    DeprecationWarning,
                )

            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=self._http_client)

    def close(self):
        """Close the HTTP client and release resources."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if hasattr(self, '_http_client') and self._http_client:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # Retry logic for rate limiting and connection errors
        max_retries = 5
        last_exception = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                latency = time.time() - start_time

                if self._usage_callback and response.usage:
                    self._usage_callback(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        latency,
                    )

                return self._parse_response(response, tools)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check for retryable errors
                is_rate_limit = "rate limit" in error_msg or "429" in error_msg
                is_connection_error = any(keyword in error_msg for keyword in [
                    "connection", "ssl", "eof", "timeout", "reset",
                    "connect", "network", "socket", "refused", "closed",
                    "broken", "pipe", "aborted"
                ])

                if is_rate_limit or is_connection_error:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 5s, 10s, 20s, 40s
                        wait_time = 5 * (2 ** attempt)
                        error_type = "Rate limit" if is_rate_limit else "Connection error"
                        logger.warning(f"[Mem0 LLM] {error_type} hit, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                        print(f"⚠️  [Mem0 LLM] {error_type}: {e}")
                        print(f"   Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")

                        # On connection errors, recreate the HTTP client to get fresh connections
                        if is_connection_error and attempt >= 2:
                            try:
                                logger.warning("[Mem0 LLM] Recreating HTTP client after repeated connection errors...")
                                print("   Recreating HTTP client...")
                                self._recreate_client()
                            except Exception as recreate_error:
                                logger.warning(f"[Mem0 LLM] Failed to recreate client: {recreate_error}")

                        time.sleep(wait_time)
                    else:
                        error_type = "Rate limit" if is_rate_limit else "Connection errors"
                        logger.error(f"[Mem0 LLM] {error_type} after {max_retries} retries")
                        raise
                else:
                    # Non-retryable error
                    raise

        raise last_exception

    def _recreate_client(self):
        """Recreate HTTP client with fresh connections."""
        logger.warning("[Mem0 LLM] Recreating HTTP client...")
        self._init_client()
