"""LLM provider module for multi-agent system.

This module provides a unified interface for interacting with
various LLM providers (Ollama, OpenAI, etc.) with support for:
- Text generation (chat and completion)
- Streaming responses
- Embeddings
- Reranking

Example:
    from src.llm import OllamaProvider, GenerationConfig

    provider = OllamaProvider(
        provider_id=1,
        name="local-ollama",
        base_url="http://localhost:11434",
    )

    await provider.connect()
    result = await provider.generate(
        prompt="What is machine learning?",
        model="llama3.1:8b",
    )
    print(result.content)
"""

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
from .manager import (
    LLMProviderManager,
    get_llm_manager,
    shutdown_llm_manager,
)
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

__all__ = [
    # Base classes and types
    "BaseLLMProvider",
    "GenerationConfig",
    "GenerationResult",
    "EmbeddingResult",
    "RerankResult",
    "Message",
    "LLMProviderStatus",
    # Exceptions
    "LLMProviderError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    # Provider implementations
    "OllamaProvider",
    "OpenAIProvider",
    # Manager
    "LLMProviderManager",
    "get_llm_manager",
    "shutdown_llm_manager",
]
