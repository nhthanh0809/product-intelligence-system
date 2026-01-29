"""OpenAI LLM provider implementation.

Supports OpenAI API for:
- Text generation (chat completions)
- Embeddings
"""

import time
from typing import Any, AsyncIterator

import httpx
import structlog

from .base import (
    BaseLLMProvider,
    EmbeddingResult,
    GenerationConfig,
    GenerationResult,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMProviderStatus,
    LLMRateLimitError,
    Message,
)

logger = structlog.get_logger()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider for GPT models.

    Supports:
    - Chat completions with GPT-3.5, GPT-4, etc.
    - Embeddings with text-embedding-ada-002, text-embedding-3-small, etc.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        provider_id: int,
        name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        settings: dict[str, Any] | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        organization: str | None = None,
    ):
        super().__init__(
            provider_id=provider_id,
            name=name,
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key=api_key,
            settings=settings,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.organization = organization or settings.get("organization") if settings else None
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize HTTP client for OpenAI."""
        if self._client is not None and not self._client.is_closed:
            return

        if not self.api_key:
            raise LLMAuthenticationError(
                "OpenAI API key is required",
                self.name,
            )

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.timeout,
                    write=30.0,
                    pool=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )

            # Verify connection with a simple models list call
            response = await self._client.get("/models")

            if response.status_code == 401:
                raise LLMAuthenticationError(
                    "Invalid OpenAI API key",
                    self.name,
                )

            if response.status_code != 200:
                raise LLMConnectionError(
                    f"Failed to connect to OpenAI: HTTP {response.status_code}",
                    self.name,
                )

            self._status = LLMProviderStatus.CONNECTED
            logger.info(
                "openai_provider_connected",
                provider=self.name,
                base_url=self.base_url,
            )

        except LLMAuthenticationError:
            self._status = LLMProviderStatus.ERROR
            raise
        except httpx.ConnectError as e:
            self._status = LLMProviderStatus.ERROR
            self._last_error = str(e)
            raise LLMConnectionError(
                f"Cannot connect to OpenAI at {self.base_url}: {e}",
                self.name,
            )
        except Exception as e:
            self._status = LLMProviderStatus.ERROR
            self._last_error = str(e)
            raise LLMConnectionError(
                f"OpenAI connection error: {e}",
                self.name,
            )

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._status = LLMProviderStatus.DISCONNECTED
            logger.info("openai_provider_disconnected", provider=self.name)

    async def health_check(self) -> dict[str, Any]:
        """Check OpenAI API health."""
        start_time = time.time()

        try:
            if self._client is None:
                await self.connect()

            response = await self._client.get("/models")
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]

                self._status = LLMProviderStatus.CONNECTED
                self._last_error = None

                return {
                    "status": "healthy",
                    "provider": self.name,
                    "base_url": self.base_url,
                    "latency_ms": latency_ms,
                    "model_count": len(models),
                }
            elif response.status_code == 401:
                self._status = LLMProviderStatus.ERROR
                return {
                    "status": "unhealthy",
                    "provider": self.name,
                    "error": "Invalid API key",
                    "latency_ms": latency_ms,
                }
            else:
                self._status = LLMProviderStatus.ERROR
                return {
                    "status": "unhealthy",
                    "provider": self.name,
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": latency_ms,
                }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._status = LLMProviderStatus.ERROR
            self._last_error = str(e)

            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e),
                "latency_ms": latency_ms,
            }

    def _handle_error_response(self, response: httpx.Response, model: str) -> None:
        """Handle error responses from OpenAI API."""
        status_code = response.status_code

        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        if status_code == 401:
            raise LLMAuthenticationError(
                f"OpenAI authentication failed: {error_message}",
                self.name,
            )
        elif status_code == 404:
            raise LLMModelNotFoundError(
                f"Model {model} not found: {error_message}",
                self.name,
                model,
            )
        elif status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {error_message}",
                self.name,
                retry_after=float(retry_after) if retry_after else None,
            )
        else:
            raise LLMProviderError(
                f"OpenAI API error ({status_code}): {error_message}",
                self.name,
                retryable=status_code >= 500,
            )

    async def generate(
        self,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text using OpenAI chat completions API."""
        config = config or GenerationConfig()
        messages = self._build_messages(prompt)

        async def _do_generate():
            start_time = time.time()

            if self._client is None:
                await self.connect()

            body = {
                "model": model,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if config.max_tokens:
                body["max_tokens"] = config.max_tokens
            if config.stop_sequences:
                body["stop"] = config.stop_sequences
            if config.presence_penalty:
                body["presence_penalty"] = config.presence_penalty
            if config.frequency_penalty:
                body["frequency_penalty"] = config.frequency_penalty
            if config.seed is not None:
                body["seed"] = config.seed
            if config.response_format == "json":
                body["response_format"] = {"type": "json_object"}

            try:
                response = await self._client.post("/chat/completions", json=body)
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code != 200:
                    self._handle_error_response(response, model)

                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})

                return GenerationResult(
                    content=choice["message"]["content"],
                    model=model,
                    provider=self.name,
                    finish_reason=choice.get("finish_reason"),
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    latency_ms=latency_ms,
                    metadata={
                        "id": data.get("id"),
                        "created": data.get("created"),
                        "system_fingerprint": data.get("system_fingerprint"),
                    },
                )

            except httpx.TimeoutException as e:
                raise LLMProviderError(
                    f"OpenAI request timed out: {e}",
                    self.name,
                    retryable=True,
                )
            except httpx.ConnectError as e:
                raise LLMConnectionError(
                    f"Cannot connect to OpenAI: {e}",
                    self.name,
                )

        return await self._retry_with_backoff(_do_generate)

    async def generate_stream(
        self,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming using OpenAI chat completions API."""
        config = config or GenerationConfig()
        messages = self._build_messages(prompt)

        if self._client is None:
            await self.connect()

        body = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        if config.max_tokens:
            body["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            body["stop"] = config.stop_sequences
        if config.presence_penalty:
            body["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            body["frequency_penalty"] = config.frequency_penalty
        if config.seed is not None:
            body["seed"] = config.seed

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    self._handle_error_response(
                        httpx.Response(
                            status_code=response.status_code,
                            content=error_text,
                        ),
                        model,
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            break

                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})

                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            continue

        except httpx.TimeoutException as e:
            raise LLMProviderError(
                f"OpenAI streaming timed out: {e}",
                self.name,
                retryable=True,
            )
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to OpenAI: {e}",
                self.name,
            )

    async def embed(
        self,
        texts: list[str],
        model: str,
    ) -> EmbeddingResult:
        """Generate embeddings using OpenAI embeddings API."""

        async def _do_embed():
            start_time = time.time()

            if self._client is None:
                await self.connect()

            body = {
                "model": model,
                "input": texts,
            }

            try:
                response = await self._client.post("/embeddings", json=body)
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code != 200:
                    self._handle_error_response(response, model)

                data = response.json()
                usage = data.get("usage", {})

                # Sort embeddings by index to maintain order
                embedding_data = sorted(data["data"], key=lambda x: x["index"])
                embeddings = [e["embedding"] for e in embedding_data]
                dimensions = len(embeddings[0]) if embeddings else 0

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    provider=self.name,
                    dimensions=dimensions,
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    latency_ms=latency_ms,
                )

            except httpx.TimeoutException as e:
                raise LLMProviderError(
                    f"OpenAI embedding timed out: {e}",
                    self.name,
                    retryable=True,
                )
            except httpx.ConnectError as e:
                raise LLMConnectionError(
                    f"Cannot connect to OpenAI: {e}",
                    self.name,
                )

        return await self._retry_with_backoff(_do_embed)

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from OpenAI.

        Returns:
            List of model info dicts
        """
        if self._client is None:
            await self.connect()

        response = await self._client.get("/models")

        if response.status_code != 200:
            self._handle_error_response(response, "")

        data = response.json()
        return data.get("data", [])
