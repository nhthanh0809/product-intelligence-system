"""Ollama LLM provider implementation.

Supports local and remote Ollama instances for:
- Text generation (chat and completion)
- Embeddings
- Reranking (with BGE reranker models)
"""

import json
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
    RerankResult,
)

logger = structlog.get_logger()


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local/remote Ollama instances.

    Supports:
    - Chat completions with llama, mistral, etc.
    - Embeddings with nomic-embed, mxbai-embed, etc.
    - Reranking with BGE reranker models
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        provider_id: int,
        name: str,
        base_url: str | None = None,
        api_key: str | None = None,  # Not used for Ollama
        settings: dict[str, Any] | None = None,
        timeout: float = 120.0,  # Ollama can be slow for first load
        max_retries: int = 3,
        retry_delay: float = 1.0,
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
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize HTTP client for Ollama."""
        if self._client is not None and not self._client.is_closed:
            return

        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
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

            # Verify connection
            response = await self._client.get("/api/tags")
            if response.status_code != 200:
                raise LLMConnectionError(
                    f"Failed to connect to Ollama: HTTP {response.status_code}",
                    self.name,
                )

            self._status = LLMProviderStatus.CONNECTED
            logger.info(
                "ollama_provider_connected",
                provider=self.name,
                base_url=self.base_url,
            )

        except httpx.ConnectError as e:
            self._status = LLMProviderStatus.ERROR
            self._last_error = str(e)
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}: {e}",
                self.name,
            )
        except Exception as e:
            self._status = LLMProviderStatus.ERROR
            self._last_error = str(e)
            raise LLMConnectionError(
                f"Ollama connection error: {e}",
                self.name,
            )

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._status = LLMProviderStatus.DISCONNECTED
            logger.info("ollama_provider_disconnected", provider=self.name)

    async def health_check(self) -> dict[str, Any]:
        """Check Ollama service health."""
        start_time = time.time()

        try:
            if self._client is None:
                await self.connect()

            response = await self._client.get("/api/tags")
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]

                self._status = LLMProviderStatus.CONNECTED
                self._last_error = None

                return {
                    "status": "healthy",
                    "provider": self.name,
                    "base_url": self.base_url,
                    "latency_ms": latency_ms,
                    "models": models,
                    "model_count": len(models),
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

    async def generate(
        self,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text using Ollama chat API."""
        config = config or GenerationConfig()
        messages = self._build_messages(prompt)

        async def _do_generate():
            start_time = time.time()

            if self._client is None:
                await self.connect()

            # Build request body
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                },
            }

            if config.max_tokens:
                body["options"]["num_predict"] = config.max_tokens
            if config.top_k:
                body["options"]["top_k"] = config.top_k
            if config.seed is not None:
                body["options"]["seed"] = config.seed
            if config.stop_sequences:
                body["options"]["stop"] = config.stop_sequences
            if config.response_format == "json":
                body["format"] = "json"

            try:
                response = await self._client.post("/api/chat", json=body)
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Model {model} not found in Ollama",
                        self.name,
                        model,
                    )

                if response.status_code != 200:
                    error_text = response.text
                    raise LLMProviderError(
                        f"Ollama generation failed: {error_text}",
                        self.name,
                        retryable=response.status_code >= 500,
                    )

                data = response.json()

                return GenerationResult(
                    content=data["message"]["content"],
                    model=model,
                    provider=self.name,
                    finish_reason=data.get("done_reason"),
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (
                            data.get("prompt_eval_count", 0) +
                            data.get("eval_count", 0)
                        ),
                    },
                    latency_ms=latency_ms,
                    metadata={
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                        "eval_duration": data.get("eval_duration"),
                    },
                )

            except httpx.TimeoutException as e:
                raise LLMProviderError(
                    f"Ollama request timed out: {e}",
                    self.name,
                    retryable=True,
                )
            except httpx.ConnectError as e:
                raise LLMConnectionError(
                    f"Cannot connect to Ollama: {e}",
                    self.name,
                )

        return await self._retry_with_backoff(_do_generate)

    async def generate_stream(
        self,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming using Ollama chat API."""
        config = config or GenerationConfig()
        messages = self._build_messages(prompt)

        if self._client is None:
            await self.connect()

        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
        }

        if config.max_tokens:
            body["options"]["num_predict"] = config.max_tokens
        if config.top_k:
            body["options"]["top_k"] = config.top_k
        if config.seed is not None:
            body["options"]["seed"] = config.seed
        if config.stop_sequences:
            body["options"]["stop"] = config.stop_sequences
        if config.response_format == "json":
            body["format"] = "json"

        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Model {model} not found in Ollama",
                        self.name,
                        model,
                    )

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMProviderError(
                        f"Ollama streaming failed: {error_text.decode()}",
                        self.name,
                        retryable=response.status_code >= 500,
                    )

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            raise LLMProviderError(
                f"Ollama streaming timed out: {e}",
                self.name,
                retryable=True,
            )
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama: {e}",
                self.name,
            )

    async def embed(
        self,
        texts: list[str],
        model: str,
    ) -> EmbeddingResult:
        """Generate embeddings using Ollama embedding API."""

        async def _do_embed():
            start_time = time.time()

            if self._client is None:
                await self.connect()

            embeddings = []
            dimensions = 0

            # Process in batches (Ollama embed API supports single text)
            for text in texts:
                body = {
                    "model": model,
                    "input": text,
                }

                try:
                    response = await self._client.post("/api/embed", json=body)

                    if response.status_code == 404:
                        raise LLMModelNotFoundError(
                            f"Embedding model {model} not found in Ollama",
                            self.name,
                            model,
                        )

                    if response.status_code != 200:
                        error_text = response.text
                        raise LLMProviderError(
                            f"Ollama embedding failed: {error_text}",
                            self.name,
                            retryable=response.status_code >= 500,
                        )

                    data = response.json()
                    embedding = data.get("embeddings", [[]])[0]
                    embeddings.append(embedding)

                    if not dimensions and embedding:
                        dimensions = len(embedding)

                except httpx.TimeoutException as e:
                    raise LLMProviderError(
                        f"Ollama embedding timed out: {e}",
                        self.name,
                        retryable=True,
                    )
                except httpx.ConnectError as e:
                    raise LLMConnectionError(
                        f"Cannot connect to Ollama: {e}",
                        self.name,
                    )

            latency_ms = (time.time() - start_time) * 1000

            return EmbeddingResult(
                embeddings=embeddings,
                model=model,
                provider=self.name,
                dimensions=dimensions,
                usage={"total_tokens": sum(len(t.split()) for t in texts)},
                latency_ms=latency_ms,
            )

        return await self._retry_with_backoff(_do_embed)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_k: int | None = None,
    ) -> RerankResult:
        """Rerank documents using Ollama with a reranker model.

        Note: This uses the generation API with a specific prompt format
        for reranker models like bge-reranker.
        """

        async def _do_rerank():
            start_time = time.time()

            if self._client is None:
                await self.connect()

            scores = []

            # Score each document against the query
            for doc in documents:
                # Format for BGE reranker style
                prompt = f"""Given the following query and document, rate the relevance of the document to the query on a scale of 0 to 1.

Query: {query}

Document: {doc}

Relevance score (0.0 to 1.0):"""

                body = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 10,
                    },
                }

                try:
                    response = await self._client.post("/api/generate", json=body)

                    if response.status_code == 404:
                        raise LLMModelNotFoundError(
                            f"Reranker model {model} not found in Ollama",
                            self.name,
                            model,
                        )

                    if response.status_code != 200:
                        error_text = response.text
                        raise LLMProviderError(
                            f"Ollama reranking failed: {error_text}",
                            self.name,
                            retryable=response.status_code >= 500,
                        )

                    data = response.json()
                    response_text = data.get("response", "0.5").strip()

                    # Parse the score
                    try:
                        score = float(response_text.split()[0])
                        score = max(0.0, min(1.0, score))
                    except (ValueError, IndexError):
                        score = 0.5

                    scores.append(score)

                except httpx.TimeoutException as e:
                    raise LLMProviderError(
                        f"Ollama reranking timed out: {e}",
                        self.name,
                        retryable=True,
                    )
                except httpx.ConnectError as e:
                    raise LLMConnectionError(
                        f"Cannot connect to Ollama: {e}",
                        self.name,
                    )

            latency_ms = (time.time() - start_time) * 1000

            # Sort by score descending
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )

            if top_k:
                sorted_indices = sorted_indices[:top_k]
                scores = [scores[i] for i in sorted_indices]
            else:
                scores = [scores[i] for i in sorted_indices]

            return RerankResult(
                scores=scores,
                indices=sorted_indices,
                model=model,
                provider=self.name,
                latency_ms=latency_ms,
            )

        return await self._retry_with_backoff(_do_rerank)

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models in Ollama.

        Returns:
            List of model info dicts with name, size, etc.
        """
        if self._client is None:
            await self.connect()

        response = await self._client.get("/api/tags")

        if response.status_code != 200:
            raise LLMProviderError(
                f"Failed to list models: HTTP {response.status_code}",
                self.name,
                retryable=True,
            )

        data = response.json()
        return data.get("models", [])

    async def pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """Pull a model from Ollama registry.

        Args:
            model: Model name to pull

        Yields:
            Progress updates with status and completion percentage
        """
        if self._client is None:
            await self.connect()

        body = {"name": model, "stream": True}

        async with self._client.stream("POST", "/api/pull", json=body) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise LLMProviderError(
                    f"Failed to pull model {model}: {error_text.decode()}",
                    self.name,
                    retryable=True,
                )

            async for line in response.aiter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
