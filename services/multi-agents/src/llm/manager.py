"""LLM Provider Manager for multi-agent system.

This module provides centralized management of LLM providers including:
- Provider initialization from database configuration
- Agent-to-provider routing
- Fallback logic when primary provider fails
- Health monitoring and metrics
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.config.models import (
    AgentModelConfig,
    HealthStatus,
    LLMModel,
    LLMProvider as LLMProviderModel,
    ModelType,
    ProviderType,
)
from src.config.repository import ConfigRepository

from .base import (
    BaseLLMProvider,
    EmbeddingResult,
    GenerationConfig,
    GenerationResult,
    LLMProviderError,
    LLMProviderStatus,
    Message,
    RerankResult,
)
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logger = structlog.get_logger()


@dataclass
class ProviderMetrics:
    """Usage metrics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    last_request_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


@dataclass
class AgentProviderConfig:
    """Configuration for an agent's LLM provider."""
    agent_name: str
    primary_provider: BaseLLMProvider | None = None
    primary_model: str | None = None
    fallback_provider: BaseLLMProvider | None = None
    fallback_model: str | None = None
    embedding_provider: BaseLLMProvider | None = None
    embedding_model: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)


class LLMProviderManager:
    """Centralized manager for LLM providers.

    Handles provider initialization, routing, and fallback logic.
    """

    # Map of provider types to implementation classes
    PROVIDER_CLASSES: dict[ProviderType, type[BaseLLMProvider]] = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.OPENAI: OpenAIProvider,
    }

    def __init__(self, config_repository: ConfigRepository | None = None):
        """Initialize the manager.

        Args:
            config_repository: Repository for loading configuration.
                              If None, must call initialize() with repository.
        """
        self._config_repository = config_repository
        self._providers: dict[int, BaseLLMProvider] = {}
        self._provider_models: dict[int, list[LLMModel]] = {}
        self._agent_configs: dict[str, AgentProviderConfig] = {}
        self._metrics: dict[int, ProviderMetrics] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    async def initialize(
        self,
        config_repository: ConfigRepository | None = None,
    ) -> None:
        """Initialize providers from database configuration.

        Args:
            config_repository: Optional repository override
        """
        async with self._lock:
            if config_repository:
                self._config_repository = config_repository

            if not self._config_repository:
                raise ValueError("Config repository is required")

            try:
                # Load providers from database
                provider_models = await self._config_repository.get_llm_providers(
                    enabled_only=True
                )

                for provider_model in provider_models:
                    await self._initialize_provider(provider_model)

                # Load agent configurations
                agent_configs = await self._config_repository.get_agent_model_configs()

                for agent_config in agent_configs:
                    await self._configure_agent(agent_config)

                self._initialized = True
                logger.info(
                    "llm_provider_manager_initialized",
                    provider_count=len(self._providers),
                    agent_count=len(self._agent_configs),
                )

            except Exception as e:
                logger.error("llm_provider_manager_init_failed", error=str(e))
                raise

    async def _initialize_provider(self, provider_model: LLMProviderModel) -> None:
        """Initialize a single provider from database model.

        Args:
            provider_model: Database model for the provider
        """
        provider_class = self.PROVIDER_CLASSES.get(provider_model.provider_type)

        if not provider_class:
            logger.warning(
                "unsupported_provider_type",
                provider_type=provider_model.provider_type,
                provider_name=provider_model.name,
            )
            return

        try:
            # Get API key from settings (it's stored encrypted)
            api_key = provider_model.settings.get("api_key")

            provider = provider_class(
                provider_id=provider_model.id,
                name=provider_model.name,
                base_url=provider_model.base_url,
                api_key=api_key,
                settings=provider_model.settings,
                timeout=provider_model.settings.get("timeout", 60.0),
                max_retries=provider_model.settings.get("max_retries", 3),
            )

            await provider.connect()
            self._providers[provider_model.id] = provider
            self._metrics[provider_model.id] = ProviderMetrics()

            # Load models for this provider
            models = await self._config_repository.get_llm_models(
                provider_id=provider_model.id,
                enabled_only=True,
            )
            self._provider_models[provider_model.id] = models

            logger.info(
                "llm_provider_initialized",
                provider_id=provider_model.id,
                provider_name=provider_model.name,
                provider_type=provider_model.provider_type,
                model_count=len(models),
            )

        except Exception as e:
            logger.error(
                "llm_provider_init_failed",
                provider_id=provider_model.id,
                provider_name=provider_model.name,
                error=str(e),
            )

    async def _configure_agent(self, agent_config: AgentModelConfig) -> None:
        """Configure provider routing for an agent.

        Args:
            agent_config: Agent model configuration from database
        """
        agent_name = agent_config.agent_name

        if agent_name not in self._agent_configs:
            self._agent_configs[agent_name] = AgentProviderConfig(
                agent_name=agent_name,
                settings=agent_config.settings,
            )

        config = self._agent_configs[agent_name]

        # Get the model and its provider
        if agent_config.model_id:
            model = await self._config_repository.get_llm_model(agent_config.model_id)

            if model and model.provider_id in self._providers:
                provider = self._providers[model.provider_id]

                if agent_config.purpose == "primary":
                    config.primary_provider = provider
                    config.primary_model = model.model_name
                elif agent_config.purpose == "fallback":
                    config.fallback_provider = provider
                    config.fallback_model = model.model_name
                elif agent_config.purpose == "embedding":
                    config.embedding_provider = provider
                    config.embedding_model = model.model_name

                logger.debug(
                    "agent_provider_configured",
                    agent_name=agent_name,
                    purpose=agent_config.purpose,
                    model=model.model_name,
                    provider=provider.name,
                )

    async def shutdown(self) -> None:
        """Shutdown all providers and clean up resources."""
        async with self._lock:
            for provider_id, provider in self._providers.items():
                try:
                    await provider.disconnect()
                    logger.info(
                        "llm_provider_disconnected",
                        provider_id=provider_id,
                        provider_name=provider.name,
                    )
                except Exception as e:
                    logger.error(
                        "llm_provider_disconnect_failed",
                        provider_id=provider_id,
                        error=str(e),
                    )

            self._providers.clear()
            self._agent_configs.clear()
            self._metrics.clear()
            self._initialized = False

    def get_provider(self, provider_id: int) -> BaseLLMProvider | None:
        """Get a provider by ID.

        Args:
            provider_id: Database ID of the provider

        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(provider_id)

    def get_provider_for_agent(
        self,
        agent_name: str,
        purpose: str = "primary",
    ) -> tuple[BaseLLMProvider | None, str | None]:
        """Get the provider and model configured for an agent.

        Args:
            agent_name: Name of the agent
            purpose: "primary", "fallback", or "embedding"

        Returns:
            Tuple of (provider, model_name) or (None, None)
        """
        config = self._agent_configs.get(agent_name)

        if not config:
            # Return default provider if no specific config
            return self._get_default_provider(purpose)

        if purpose == "primary":
            return config.primary_provider, config.primary_model
        elif purpose == "fallback":
            return config.fallback_provider, config.fallback_model
        elif purpose == "embedding":
            return config.embedding_provider, config.embedding_model

        return None, None

    def _get_default_provider(
        self,
        purpose: str,
    ) -> tuple[BaseLLMProvider | None, str | None]:
        """Get the default provider for a purpose.

        Args:
            purpose: "primary", "fallback", or "embedding"

        Returns:
            Tuple of (provider, model_name) or (None, None)
        """
        # Find first available provider
        for provider_id, provider in self._providers.items():
            if provider.status == LLMProviderStatus.CONNECTED:
                models = self._provider_models.get(provider_id, [])

                # Find appropriate model type
                model_type = ModelType.CHAT
                if purpose == "embedding":
                    model_type = ModelType.EMBEDDING

                for model in models:
                    if model.model_type == model_type:
                        return provider, model.model_name

        return None, None

    async def generate_for_agent(
        self,
        agent_name: str,
        prompt: str | list[Message],
        config: GenerationConfig | None = None,
        use_fallback: bool = True,
    ) -> GenerationResult:
        """Generate text for an agent with automatic fallback.

        Args:
            agent_name: Name of the agent
            prompt: Text prompt or messages
            config: Generation configuration
            use_fallback: Whether to use fallback provider on failure

        Returns:
            GenerationResult

        Raises:
            LLMProviderError: If all providers fail
        """
        # Try primary provider
        provider, model = self.get_provider_for_agent(agent_name, "primary")

        if provider and model:
            try:
                result = await self._generate_with_metrics(
                    provider, prompt, model, config
                )
                return result
            except LLMProviderError as e:
                logger.warning(
                    "primary_provider_failed",
                    agent_name=agent_name,
                    provider=provider.name,
                    error=str(e),
                )

                if not use_fallback:
                    raise

        # Try fallback provider
        if use_fallback:
            fallback_provider, fallback_model = self.get_provider_for_agent(
                agent_name, "fallback"
            )

            if fallback_provider and fallback_model:
                try:
                    result = await self._generate_with_metrics(
                        fallback_provider, prompt, fallback_model, config
                    )
                    logger.info(
                        "fallback_provider_used",
                        agent_name=agent_name,
                        provider=fallback_provider.name,
                    )
                    return result
                except LLMProviderError as e:
                    logger.error(
                        "fallback_provider_failed",
                        agent_name=agent_name,
                        provider=fallback_provider.name,
                        error=str(e),
                    )
                    raise

        raise LLMProviderError(
            f"No available provider for agent {agent_name}",
            "manager",
            retryable=False,
        )

    async def _generate_with_metrics(
        self,
        provider: BaseLLMProvider,
        prompt: str | list[Message],
        model: str,
        config: GenerationConfig | None,
    ) -> GenerationResult:
        """Generate text and track metrics.

        Args:
            provider: Provider to use
            prompt: Text prompt or messages
            model: Model name
            config: Generation configuration

        Returns:
            GenerationResult
        """
        metrics = self._metrics.get(provider.provider_id)

        if metrics:
            metrics.total_requests += 1
            metrics.last_request_at = datetime.now()

        try:
            result = await provider.generate(prompt, model, config)

            if metrics:
                metrics.successful_requests += 1
                metrics.total_tokens += result.total_tokens
                metrics.total_latency_ms += result.latency_ms

            return result

        except LLMProviderError as e:
            if metrics:
                metrics.failed_requests += 1
                metrics.last_error = str(e)
                metrics.last_error_at = datetime.now()
            raise

    async def embed_for_agent(
        self,
        agent_name: str,
        texts: list[str],
    ) -> EmbeddingResult:
        """Generate embeddings for an agent.

        Args:
            agent_name: Name of the agent
            texts: Texts to embed

        Returns:
            EmbeddingResult

        Raises:
            LLMProviderError: If embedding fails
        """
        provider, model = self.get_provider_for_agent(agent_name, "embedding")

        if not provider or not model:
            # Try default embedding provider
            provider, model = self._get_default_provider("embedding")

        if not provider or not model:
            raise LLMProviderError(
                f"No embedding provider available for agent {agent_name}",
                "manager",
                retryable=False,
            )

        return await provider.embed(texts, model)

    async def health_check_all(self) -> dict[str, Any]:
        """Check health of all providers.

        Returns:
            Dict with health status for each provider
        """
        results = {}

        for provider_id, provider in self._providers.items():
            health = await provider.health_check()
            results[provider.name] = {
                "provider_id": provider_id,
                **health,
            }

            # Update health status in database
            if self._config_repository:
                status = (
                    HealthStatus.HEALTHY
                    if health.get("status") == "healthy"
                    else HealthStatus.UNHEALTHY
                )

                await self._config_repository.update_provider_health(
                    provider_id, status
                )

        return results

    def get_metrics(self, provider_id: int | None = None) -> dict[str, Any]:
        """Get usage metrics for providers.

        Args:
            provider_id: Optional specific provider ID

        Returns:
            Dict with metrics
        """
        if provider_id:
            metrics = self._metrics.get(provider_id)
            if metrics:
                return {
                    "provider_id": provider_id,
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.success_rate,
                    "total_tokens": metrics.total_tokens,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "last_request_at": (
                        metrics.last_request_at.isoformat()
                        if metrics.last_request_at
                        else None
                    ),
                    "last_error": metrics.last_error,
                }
            return {}

        # Return all metrics
        all_metrics = {}
        for pid, metrics in self._metrics.items():
            provider = self._providers.get(pid)
            all_metrics[provider.name if provider else str(pid)] = {
                "provider_id": pid,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.success_rate,
                "total_tokens": metrics.total_tokens,
                "avg_latency_ms": metrics.avg_latency_ms,
            }

        return all_metrics

    async def reload_config(self) -> None:
        """Reload configuration from database.

        Useful for picking up configuration changes without restart.
        """
        logger.info("llm_provider_manager_reloading")
        await self.shutdown()
        await self.initialize()


# Singleton instance
_manager: LLMProviderManager | None = None


async def get_llm_manager(
    config_repository: ConfigRepository | None = None,
) -> LLMProviderManager:
    """Get or create the LLM provider manager singleton.

    Args:
        config_repository: Config repository for initialization

    Returns:
        LLMProviderManager instance
    """
    global _manager

    if _manager is None:
        _manager = LLMProviderManager(config_repository)

    if not _manager.is_initialized and config_repository:
        await _manager.initialize(config_repository)

    return _manager


async def shutdown_llm_manager() -> None:
    """Shutdown the LLM manager singleton."""
    global _manager

    if _manager:
        await _manager.shutdown()
        _manager = None
