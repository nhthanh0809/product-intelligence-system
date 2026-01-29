"""Configuration manager for high-level configuration access.

This module provides:
- Simplified access to configuration values
- Type-safe configuration getters
- Configuration change notifications
- Lazy loading with caching
"""

import json
from typing import Any, TypeVar, Callable
from functools import wraps

import asyncpg
import structlog
from redis.asyncio import Redis

from src.config.repository import ConfigRepository, get_config_repository
from src.config.models import (
    ConfigSetting,
    LLMProvider,
    LLMModel,
    LLMModelWithProvider,
    AgentModelConfigWithModel,
    SearchStrategy,
    RerankerConfigWithModel,
    ConversationSession,
    ConversationContext,
    ConversationSessionUpdate,
    ConversationMessageCreate,
)

logger = structlog.get_logger()

T = TypeVar("T")


class ConfigManager:
    """High-level configuration manager.

    Provides type-safe access to configuration values with caching
    and change notifications.
    """

    def __init__(self, repository: ConfigRepository):
        """Initialize config manager.

        Args:
            repository: Configuration repository instance
        """
        self.repository = repository
        self._listeners: list[Callable[[str, Any], None]] = []

    # =========================================================================
    # Setting Getters
    # =========================================================================

    async def get_string(self, key: str, default: str = "") -> str:
        """Get a string configuration value."""
        value = await self.repository.get_setting_value(key, default)
        if isinstance(value, str):
            return value
        return str(value) if value is not None else default

    async def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        value = await self.repository.get_setting_value(key, default)
        if isinstance(value, int):
            return value
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    async def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        value = await self.repository.get_setting_value(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    async def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        value = await self.repository.get_setting_value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value) if value is not None else default

    async def get_json(self, key: str, default: dict | None = None) -> dict:
        """Get a JSON configuration value."""
        value = await self.repository.get_setting_value(key, default)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default or {}
        return default or {}

    async def get_list(self, key: str, default: list | None = None) -> list:
        """Get a list configuration value."""
        value = await self.repository.get_setting_value(key, default)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else default or []
            except json.JSONDecodeError:
                return default or []
        return default or []

    async def set_value(self, key: str, value: Any) -> ConfigSetting | None:
        """Set a configuration value."""
        from src.config.models import ConfigSettingUpdate
        setting = await self.repository.update_setting(
            key, ConfigSettingUpdate(value=value)
        )
        if setting:
            self._notify_change(key, value)
        return setting

    # =========================================================================
    # LLM Configuration
    # =========================================================================

    async def get_default_llm_provider(self) -> LLMProvider | None:
        """Get the default LLM provider."""
        return await self.repository.get_default_llm_provider()

    async def get_llm_provider(self, name: str) -> LLMProvider | None:
        """Get an LLM provider by name."""
        providers = await self.repository.get_llm_providers()
        for provider in providers:
            if provider.name == name:
                return provider
        return None

    async def get_llm_models_for_type(
        self, model_type: str, enabled_only: bool = True
    ) -> list[LLMModelWithProvider]:
        """Get available models for a specific type."""
        return await self.repository.get_llm_models(
            model_type=model_type, enabled_only=enabled_only
        )

    async def get_agent_model(
        self, agent_name: str, purpose: str = "primary"
    ) -> AgentModelConfigWithModel | None:
        """Get the model configuration for an agent."""
        return await self.repository.get_agent_model_config(agent_name, purpose)

    async def get_agent_settings(self, agent_name: str) -> dict[str, Any]:
        """Get all settings for an agent.

        Combines:
        - Agent-specific config settings (agent.{name}.*)
        - Agent model config settings
        """
        settings = {}

        # Get agent-specific settings
        prefix = f"agent.{agent_name}."
        all_settings = await self.repository.get_settings()
        for setting in all_settings:
            if setting.key.startswith(prefix):
                short_key = setting.key[len(prefix):]
                settings[short_key] = setting.value

        # Get model config settings
        model_config = await self.get_agent_model(agent_name)
        if model_config and model_config.settings:
            settings.update(model_config.settings)

        return settings

    # =========================================================================
    # Search Strategy Configuration
    # =========================================================================

    async def get_search_strategies(
        self, strategy_type: str | None = None
    ) -> list[SearchStrategy]:
        """Get available search strategies."""
        return await self.repository.get_search_strategies(
            strategy_type=strategy_type, enabled_only=True
        )

    async def get_strategy_for_query_type(self, query_type: str) -> SearchStrategy | None:
        """Get the primary strategy for a query type."""
        strategies = await self.repository.get_strategies_for_query_type(query_type)
        return strategies[0] if strategies else None

    async def get_strategies_for_query_type(self, query_type: str) -> list[SearchStrategy]:
        """Get all strategies for a query type (in priority order)."""
        return await self.repository.get_strategies_for_query_type(query_type)

    async def get_search_weights(self) -> tuple[float, float]:
        """Get hybrid search weights (keyword, semantic)."""
        keyword = await self.get_float("search.hybrid.keyword_weight", 0.65)
        semantic = await self.get_float("search.hybrid.semantic_weight", 0.35)
        return keyword, semantic

    # =========================================================================
    # Reranker Configuration
    # =========================================================================

    async def is_reranker_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return await self.get_bool("reranker.enabled", False)

    async def get_reranker_config(self) -> RerankerConfigWithModel | None:
        """Get the active reranker configuration."""
        if not await self.is_reranker_enabled():
            return None
        return await self.repository.get_default_reranker()

    async def get_reranker_top_k(self) -> int:
        """Get the reranker top_k setting."""
        return await self.get_int("reranker.top_k", 10)

    # =========================================================================
    # Conversation Configuration
    # =========================================================================

    async def get_session_ttl_hours(self) -> int:
        """Get conversation session TTL in hours."""
        return await self.get_int("conversation.session_ttl_hours", 24)

    async def get_max_context_products(self) -> int:
        """Get maximum products to keep in context."""
        return await self.get_int("conversation.max_context_products", 10)

    async def is_pronoun_resolution_enabled(self) -> bool:
        """Check if pronoun resolution is enabled."""
        return await self.get_bool("conversation.enable_pronoun_resolution", True)

    # =========================================================================
    # Conversation Management
    # =========================================================================

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> ConversationSession:
        """Get an existing session or create a new one."""
        from uuid import UUID

        if session_id:
            try:
                uuid = UUID(session_id)
                session = await self.repository.get_conversation_session(uuid)
                if session and session.is_active:
                    return session
            except ValueError:
                pass

        # Create new session
        return await self.repository.create_conversation_session()

    async def update_session_context(
        self,
        session_id: str,
        products: list[dict] | None = None,
        intent: str | None = None,
        agent: str | None = None,
    ) -> ConversationSession | None:
        """Update session context with new data."""
        from uuid import UUID

        try:
            uuid = UUID(session_id)
        except ValueError:
            return None

        session = await self.repository.get_conversation_session(uuid)
        if not session:
            return None

        # Build updated context
        context = session.context.model_copy() if session.context else ConversationContext()

        if products:
            max_products = await self.get_max_context_products()
            # Add new products to the front, keep most recent
            existing_asins = {p.get("asin") for p in products}
            filtered_existing = [
                p for p in context.products if p.get("asin") not in existing_asins
            ]
            context.products = (products + filtered_existing)[:max_products]
            context.product_asins = [p.get("asin") for p in context.products if p.get("asin")]

            # Update brands and categories
            new_brands = {p.get("brand") for p in products if p.get("brand")}
            context.brands = list(set(context.brands) | new_brands)[:10]

            new_categories = {p.get("category") for p in products if p.get("category")}
            context.categories = list(set(context.categories) | new_categories)[:10]

        if intent:
            context.last_intent = intent
        if agent:
            context.last_agent = agent

        return await self.repository.update_conversation_session(
            uuid, ConversationSessionUpdate(context=context)
        )

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        entities: dict | None = None,
        products: list[dict] | None = None,
        resolved_references: list[dict] | None = None,
    ):
        """Add a message to the conversation."""
        from uuid import UUID

        try:
            uuid = UUID(session_id)
        except ValueError:
            return None

        return await self.repository.add_conversation_message(
            ConversationMessageCreate(
                session_id=uuid,
                role=role,
                content=content,
                intent=intent,
                entities=entities or {},
                products=products or [],
                resolved_references=resolved_references or [],
            )
        )

    async def get_session_products(self, session_id: str) -> list[dict]:
        """Get products from session context."""
        from uuid import UUID

        try:
            uuid = UUID(session_id)
        except ValueError:
            return []

        session = await self.repository.get_conversation_session(uuid)
        if not session or not session.context:
            return []

        return session.context.products

    # =========================================================================
    # Performance Configuration
    # =========================================================================

    async def get_cache_ttl(self, cache_type: str = "search") -> int:
        """Get cache TTL for a specific cache type."""
        return await self.get_int(f"cache.{cache_type}_ttl_seconds", 300)

    async def get_request_timeout(self) -> int:
        """Get the request timeout in seconds."""
        return await self.get_int("system.request_timeout", 120)

    # =========================================================================
    # Change Notifications
    # =========================================================================

    def add_change_listener(self, listener: Callable[[str, Any], None]) -> None:
        """Add a listener for configuration changes.

        Args:
            listener: Callback function that receives (key, new_value)
        """
        self._listeners.append(listener)

    def remove_change_listener(self, listener: Callable[[str, Any], None]) -> None:
        """Remove a change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_change(self, key: str, value: Any) -> None:
        """Notify all listeners of a configuration change."""
        for listener in self._listeners:
            try:
                listener(key, value)
            except Exception as e:
                logger.error("config_change_listener_error", key=key, error=str(e))

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def get_all_settings_by_category(self) -> dict[str, list[ConfigSetting]]:
        """Get all settings grouped by category."""
        categories = await self.repository.get_categories()
        result = {}

        for category in categories:
            settings = await self.repository.get_settings(category.id)
            result[category.name] = settings

        # Also get uncategorized settings
        all_settings = await self.repository.get_settings()
        uncategorized = [s for s in all_settings if s.category_id is None]
        if uncategorized:
            result["uncategorized"] = uncategorized

        return result

    async def export_config(self) -> dict[str, Any]:
        """Export all configuration as a dictionary."""
        settings = await self.repository.get_settings()
        return {s.key: s.value for s in settings}

    async def import_config(self, config: dict[str, Any]) -> int:
        """Import configuration from a dictionary.

        Args:
            config: Dictionary of key-value pairs

        Returns:
            Number of settings updated
        """
        count = 0
        for key, value in config.items():
            setting = await self.set_value(key, value)
            if setting:
                count += 1
        return count


# =============================================================================
# Singleton
# =============================================================================

_config_manager: ConfigManager | None = None


async def get_config_manager(
    db_pool: asyncpg.Pool | None = None,
    redis: Redis | None = None,
) -> ConfigManager:
    """Get or create config manager singleton.

    Args:
        db_pool: PostgreSQL connection pool
        redis: Redis client for caching

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        repository = await get_config_repository(db_pool, redis)
        _config_manager = ConfigManager(repository)
    return _config_manager


def reset_config_manager() -> None:
    """Reset the config manager singleton (for testing)."""
    global _config_manager
    _config_manager = None
