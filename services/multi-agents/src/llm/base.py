"""Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement,
providing a consistent API for text generation, embeddings, and reranking.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

import structlog

logger = structlog.get_logger()


class LLMProviderStatus(str, Enum):
    """Provider connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    top_k: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    response_format: str | None = None  # "json" for structured output


@dataclass
class Message:
    """Chat message for conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: str | None = None


@dataclass
class GenerationResult:
    """Result from text generation."""
    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embeddings: list[list[float]]
    model: str
    provider: str
    dimensions: int
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RerankResult:
    """Result from reranking."""
    scores: list[float]
    indices: list[int]  # Sorted by relevance
    model: str
    provider: str
    latency_ms: float = 0.0


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider: str, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class LLMConnectionError(LLMProviderError):
    """Error connecting to LLM provider."""
    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, retryable=True)


class LLMRateLimitError(LLMProviderError):
    """Rate limit exceeded."""
    def __init__(self, message: str, provider: str, retry_after: float | None = None):
        super().__init__(message, provider, retryable=True)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMProviderError):
    """Authentication failed."""
    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, retryable=False)


class LLMModelNotFoundError(LLMProviderError):
    """Model not found."""
    def __init__(self, message: str, provider: str, model: str):
        super().__init__(message, provider, retryable=False)
        self.model = model


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to provide
    a consistent API for generation, embeddings, and reranking.
    """

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
    ):
        """Initialize the provider.

        Args:
            provider_id: Database ID of the provider
            name: Provider name
            base_url: Base URL for API calls
            api_key: API key for authentication
            settings: Additional provider settings
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.provider_id = provider_id
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.settings = settings or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._status = LLMProviderStatus.DISCONNECTED
        self._last_error: str | None = None
        self._last_health_check: datetime | None = None

    @property
    def status(self) -> LLMProviderStatus:
        """Get current provider status."""
        return self._status

    @property
    def provider_type(self) -> str:
        """Get provider type name."""
        return self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the provider.

        Should set up HTTP clients, verify credentials, etc.

        Raises:
            LLMConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the provider.

        Should clean up resources, close HTTP clients, etc.
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check provider health.

        Returns:
            dict with status, latency, and any provider-specific info
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text completion.

        Args:
            prompt: Text prompt or list of chat messages
            model: Model name to use
            config: Generation configuration

        Returns:
            GenerationResult with generated text

        Raises:
            LLMProviderError: On generation failure
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate text completion with streaming.

        Args:
            prompt: Text prompt or list of chat messages
            model: Model name to use
            config: Generation configuration

        Yields:
            Text chunks as they are generated

        Raises:
            LLMProviderError: On generation failure
        """
        pass

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
    ) -> EmbeddingResult:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            EmbeddingResult with embedding vectors

        Raises:
            LLMProviderError: On embedding failure
        """
        pass

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_k: int | None = None,
    ) -> RerankResult:
        """Rerank documents by relevance to query.

        Default implementation raises NotImplementedError.
        Override in providers that support reranking.

        Args:
            query: Query to rank against
            documents: Documents to rerank
            model: Reranker model to use
            top_k: Return only top k results

        Returns:
            RerankResult with scores and sorted indices

        Raises:
            NotImplementedError: If provider doesn't support reranking
        """
        raise NotImplementedError(
            f"Provider {self.name} does not support reranking"
        )

    async def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with exponential backoff retry.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            LLMProviderError: After all retries exhausted
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except LLMProviderError as e:
                last_error = e

                if not e.retryable:
                    raise

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)

                    # Handle rate limit retry-after
                    if isinstance(e, LLMRateLimitError) and e.retry_after:
                        delay = max(delay, e.retry_after)

                    logger.warning(
                        "llm_provider_retry",
                        provider=self.name,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        raise last_error

    def _build_messages(self, prompt: str | list[Message]) -> list[dict[str, str]]:
        """Convert prompt to message format.

        Args:
            prompt: String prompt or list of Messages

        Returns:
            List of message dicts with role and content
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        return [
            {"role": msg.role, "content": msg.content}
            for msg in prompt
        ]
