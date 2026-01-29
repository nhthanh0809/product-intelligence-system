"""Configuration repository for database operations.

This module provides:
- CRUD operations for all configuration tables
- Redis caching with automatic invalidation
- Async database access using asyncpg
"""

import json
from datetime import datetime, timedelta
from typing import Any, TypeVar
from uuid import UUID

import asyncpg
import structlog
from redis.asyncio import Redis

from src.config.models import (
    ConfigCategory,
    ConfigCategoryCreate,
    ConfigCategoryUpdate,
    ConfigSetting,
    ConfigSettingCreate,
    ConfigSettingUpdate,
    LLMProvider,
    LLMProviderCreate,
    LLMProviderUpdate,
    LLMModel,
    LLMModelCreate,
    LLMModelUpdate,
    LLMModelWithProvider,
    AgentModelConfig,
    AgentModelConfigCreate,
    AgentModelConfigUpdate,
    AgentModelConfigWithModel,
    SearchStrategy,
    SearchStrategyCreate,
    SearchStrategyUpdate,
    QueryStrategyMapping,
    QueryStrategyMappingCreate,
    QueryStrategyMappingUpdate,
    QueryStrategyMappingWithStrategy,
    RerankerConfig,
    RerankerConfigCreate,
    RerankerConfigUpdate,
    RerankerConfigWithModel,
    ConfigAuditLog,
    ConversationSession,
    ConversationSessionCreate,
    ConversationSessionUpdate,
    ConversationMessage,
    ConversationMessageCreate,
    ConversationContext,
    HealthStatus,
)

logger = structlog.get_logger()

T = TypeVar("T")


def _parse_json_fields(row_dict: dict, fields: list[str]) -> dict:
    """Parse JSON string fields into Python dicts.

    Args:
        row_dict: Dictionary from database row
        fields: List of field names to parse

    Returns:
        Dictionary with JSON fields parsed
    """
    result = row_dict.copy()
    for field in fields:
        if field in result and isinstance(result[field], str):
            try:
                result[field] = json.loads(result[field])
            except json.JSONDecodeError:
                result[field] = {}
    return result


class ConfigRepository:
    """Repository for configuration database operations.

    Provides async CRUD operations with Redis caching for all
    configuration tables. Cache invalidation is automatic on writes.
    """

    CACHE_PREFIX = "config:"
    CACHE_TTL = 60  # seconds

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        redis: Redis | None = None,
    ):
        """Initialize repository.

        Args:
            db_pool: PostgreSQL connection pool
            redis: Optional Redis client for caching
        """
        self.db_pool = db_pool
        self.redis = redis

    # =========================================================================
    # Cache Helpers
    # =========================================================================

    async def _cache_get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.redis:
            return None
        try:
            data = await self.redis.get(f"{self.CACHE_PREFIX}{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("cache_get_error", key=key, error=str(e))
        return None

    async def _cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        if not self.redis:
            return
        try:
            await self.redis.setex(
                f"{self.CACHE_PREFIX}{key}",
                ttl or self.CACHE_TTL,
                json.dumps(value, default=str),
            )
        except Exception as e:
            logger.warning("cache_set_error", key=key, error=str(e))

    async def _cache_delete(self, pattern: str) -> None:
        """Delete cache keys matching pattern."""
        if not self.redis:
            return
        try:
            keys = []
            async for key in self.redis.scan_iter(f"{self.CACHE_PREFIX}{pattern}*"):
                keys.append(key)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.warning("cache_delete_error", pattern=pattern, error=str(e))

    # =========================================================================
    # Config Categories
    # =========================================================================

    async def get_categories(self) -> list[ConfigCategory]:
        """Get all configuration categories."""
        cache_key = "categories:all"
        cached = await self._cache_get(cache_key)
        if cached:
            return [ConfigCategory(**c) for c in cached]

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM config_categories
                WHERE is_visible = TRUE
                ORDER BY display_order, name
                """
            )

        categories = [ConfigCategory(**dict(row)) for row in rows]
        await self._cache_set(cache_key, [c.model_dump() for c in categories])
        return categories

    async def get_category(self, category_id: int) -> ConfigCategory | None:
        """Get a category by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM config_categories WHERE id = $1",
                category_id,
            )
        return ConfigCategory(**dict(row)) if row else None

    async def create_category(self, data: ConfigCategoryCreate) -> ConfigCategory:
        """Create a new category."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO config_categories (name, description, icon, display_order, is_visible)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                data.name,
                data.description,
                data.icon,
                data.display_order,
                data.is_visible,
            )
        await self._cache_delete("categories:")
        return ConfigCategory(**dict(row))

    async def update_category(
        self, category_id: int, data: ConfigCategoryUpdate
    ) -> ConfigCategory | None:
        """Update a category."""
        updates = {k: v for k, v in data.model_dump().items() if v is not None}
        if not updates:
            return await self.get_category(category_id)

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [category_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE config_categories SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("categories:")
        return ConfigCategory(**dict(row)) if row else None

    async def delete_category(self, category_id: int) -> bool:
        """Delete a category."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM config_categories WHERE id = $1",
                category_id,
            )
        await self._cache_delete("categories:")
        return result == "DELETE 1"

    # =========================================================================
    # Config Settings
    # =========================================================================

    async def get_settings(
        self, category_id: int | None = None
    ) -> list[ConfigSetting]:
        """Get all settings, optionally filtered by category."""
        cache_key = f"settings:cat:{category_id or 'all'}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [ConfigSetting(**s) for s in cached]

        async with self.db_pool.acquire() as conn:
            if category_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM config_settings
                    WHERE category_id = $1
                    ORDER BY display_order, key
                    """,
                    category_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM config_settings ORDER BY display_order, key"
                )

        settings = [ConfigSetting(**_parse_json_fields(dict(row), ["value", "default_value", "validation_rules"])) for row in rows]
        await self._cache_set(cache_key, [s.model_dump() for s in settings])
        return settings

    async def get_setting(self, key: str) -> ConfigSetting | None:
        """Get a setting by key."""
        cache_key = f"settings:key:{key}"
        cached = await self._cache_get(cache_key)
        if cached:
            return ConfigSetting(**cached)

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM config_settings WHERE key = $1",
                key,
            )

        if row:
            setting = ConfigSetting(**_parse_json_fields(dict(row), ["value", "default_value", "validation_rules"]))
            await self._cache_set(cache_key, setting.model_dump())
            return setting
        return None

    async def get_setting_value(self, key: str, default: Any = None) -> Any:
        """Get just the value of a setting."""
        setting = await self.get_setting(key)
        if setting is None:
            return default
        return setting.value

    async def create_setting(self, data: ConfigSettingCreate) -> ConfigSetting:
        """Create a new setting."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO config_settings
                (category_id, key, value, value_type, default_value, label,
                 description, is_sensitive, is_readonly, validation_rules, display_order)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING *
                """,
                data.category_id,
                data.key,
                json.dumps(data.value),
                data.value_type.value,
                json.dumps(data.default_value) if data.default_value else None,
                data.label,
                data.description,
                data.is_sensitive,
                data.is_readonly,
                json.dumps(data.validation_rules.model_dump()) if data.validation_rules else None,
                data.display_order,
            )
        await self._cache_delete("settings:")
        return ConfigSetting(**_parse_json_fields(dict(row), ["value", "default_value", "validation_rules"]))

    async def update_setting(self, key: str, data: ConfigSettingUpdate) -> ConfigSetting | None:
        """Update a setting."""
        updates = {}
        if data.value is not None:
            updates["value"] = json.dumps(data.value)
        if data.label is not None:
            updates["label"] = data.label
        if data.description is not None:
            updates["description"] = data.description
        if data.is_sensitive is not None:
            updates["is_sensitive"] = data.is_sensitive
        if data.validation_rules is not None:
            updates["validation_rules"] = json.dumps(data.validation_rules.model_dump())
        if data.display_order is not None:
            updates["display_order"] = data.display_order

        if not updates:
            return await self.get_setting(key)

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [key] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE config_settings SET {set_clause} WHERE key = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("settings:")
        return ConfigSetting(**_parse_json_fields(dict(row), ["value", "default_value", "validation_rules"])) if row else None

    async def delete_setting(self, key: str) -> bool:
        """Delete a setting."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM config_settings WHERE key = $1",
                key,
            )
        await self._cache_delete("settings:")
        return result == "DELETE 1"

    # =========================================================================
    # LLM Providers
    # =========================================================================

    async def get_llm_providers(self, enabled_only: bool = False) -> list[LLMProvider]:
        """Get all LLM providers."""
        cache_key = f"llm_providers:enabled:{enabled_only}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [LLMProvider(**p) for p in cached]

        async with self.db_pool.acquire() as conn:
            if enabled_only:
                rows = await conn.fetch(
                    "SELECT * FROM llm_providers WHERE is_enabled = TRUE ORDER BY name"
                )
            else:
                rows = await conn.fetch("SELECT * FROM llm_providers ORDER BY name")

        providers = [
            LLMProvider(**_parse_json_fields(dict(row), ["settings"]))
            for row in rows
        ]
        await self._cache_set(cache_key, [p.model_dump() for p in providers])
        return providers

    async def get_llm_provider(self, provider_id: int) -> LLMProvider | None:
        """Get an LLM provider by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM llm_providers WHERE id = $1",
                provider_id,
            )
        if row:
            return LLMProvider(**_parse_json_fields(dict(row), ["settings"]))
        return None

    async def get_default_llm_provider(self) -> LLMProvider | None:
        """Get the default LLM provider."""
        cache_key = "llm_providers:default"
        cached = await self._cache_get(cache_key)
        if cached:
            return LLMProvider(**cached)

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM llm_providers WHERE is_default = TRUE AND is_enabled = TRUE"
            )

        if row:
            provider = LLMProvider(**_parse_json_fields(dict(row), ["settings"]))
            await self._cache_set(cache_key, provider.model_dump())
            return provider
        return None

    async def create_llm_provider(self, data: LLMProviderCreate) -> LLMProvider:
        """Create a new LLM provider."""
        try:
            # If setting as default, unset other defaults first
            async with self.db_pool.acquire() as conn:
                if data.is_default:
                    await conn.execute(
                        "UPDATE llm_providers SET is_default = FALSE WHERE is_default = TRUE"
                    )

                row = await conn.fetchrow(
                    """
                    INSERT INTO llm_providers
                    (name, display_name, provider_type, base_url, api_key_encrypted,
                     is_enabled, is_default, settings, health_check_url)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING *
                    """,
                    data.name,
                    data.display_name,
                    data.provider_type.value,
                    data.base_url,
                    data.api_key,  # TODO: Encrypt this
                    data.is_enabled,
                    data.is_default,
                    json.dumps(data.settings),
                    data.health_check_url,
                )
            await self._cache_delete("llm_providers:")
            return LLMProvider(**_parse_json_fields(dict(row), ["settings"]))
        except asyncpg.UniqueViolationError:
            raise ValueError(
                f"Provider with name '{data.name}' already exists. "
                "Please use a unique name."
            )

    async def update_llm_provider(
        self, provider_id: int, data: LLMProviderUpdate
    ) -> LLMProvider | None:
        """Update an LLM provider."""
        async with self.db_pool.acquire() as conn:
            # If setting as default, unset other defaults first
            if data.is_default:
                await conn.execute(
                    "UPDATE llm_providers SET is_default = FALSE WHERE is_default = TRUE AND id != $1",
                    provider_id,
                )

            updates = {}
            if data.display_name is not None:
                updates["display_name"] = data.display_name
            if data.base_url is not None:
                updates["base_url"] = data.base_url
            if data.api_key is not None:
                updates["api_key_encrypted"] = data.api_key  # TODO: Encrypt
            if data.is_enabled is not None:
                updates["is_enabled"] = data.is_enabled
            if data.is_default is not None:
                updates["is_default"] = data.is_default
            if data.settings is not None:
                updates["settings"] = json.dumps(data.settings)
            if data.health_check_url is not None:
                updates["health_check_url"] = data.health_check_url

            if not updates:
                return await self.get_llm_provider(provider_id)

            set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
            values = [provider_id] + list(updates.values())

            row = await conn.fetchrow(
                f"UPDATE llm_providers SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )

        await self._cache_delete("llm_providers:")
        if row:
            return LLMProvider(**_parse_json_fields(dict(row), ["settings"]))
        return None

    async def delete_llm_provider(self, provider_id: int) -> bool:
        """Delete an LLM provider."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM llm_providers WHERE id = $1",
                provider_id,
            )
        await self._cache_delete("llm_providers:")
        return result == "DELETE 1"

    async def check_provider_health(self, provider_id: int):
        """Check health of an LLM provider and update status."""
        from src.config.models import HealthStatus
        import httpx

        provider = await self.get_llm_provider(provider_id)
        if not provider:
            return None

        health_url = provider.health_check_url or provider.base_url
        if not health_url:
            return HealthStatus.UNKNOWN

        status = HealthStatus.UNKNOWN
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try to reach the health endpoint
                if health_url.endswith("/"):
                    health_url = health_url.rstrip("/")
                resp = await client.get(f"{health_url}/api/tags")
                status = HealthStatus.HEALTHY if resp.status_code == 200 else HealthStatus.UNHEALTHY
        except Exception as e:
            logger.warning("health_check_failed", provider_id=provider_id, error=str(e))
            status = HealthStatus.UNHEALTHY

        # Update the provider health status
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE llm_providers
                SET health_status = $1, last_health_check = NOW()
                WHERE id = $2
                """,
                status.value,
                provider_id,
            )
        await self._cache_delete("llm_providers:")
        return status

    async def update_provider_health(
        self, provider_id: int, status: HealthStatus
    ) -> None:
        """Update provider health status without doing a health check.

        Args:
            provider_id: ID of the provider
            status: New health status
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE llm_providers
                SET health_status = $1, last_health_check = NOW()
                WHERE id = $2
                """,
                status.value,
                provider_id,
            )
        await self._cache_delete("llm_providers:")

    # =========================================================================
    # LLM Models
    # =========================================================================

    async def get_llm_models(
        self,
        provider_id: int | None = None,
        model_type: str | None = None,
        enabled_only: bool = False,
    ) -> list[LLMModelWithProvider]:
        """Get LLM models with optional filters."""
        cache_key = f"llm_models:p:{provider_id}:t:{model_type}:e:{enabled_only}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [LLMModelWithProvider(**m) for m in cached]

        query = """
            SELECT m.*, p.name as provider_name, p.display_name as provider_display_name,
                   p.provider_type, p.base_url, p.is_enabled as provider_enabled
            FROM llm_models m
            JOIN llm_providers p ON m.provider_id = p.id
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if provider_id:
            query += f" AND m.provider_id = ${param_idx}"
            params.append(provider_id)
            param_idx += 1

        if model_type:
            query += f" AND m.model_type = ${param_idx}"
            params.append(model_type)
            param_idx += 1

        if enabled_only:
            query += " AND m.is_enabled = TRUE AND p.is_enabled = TRUE"

        query += " ORDER BY m.model_name"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        models = []
        for row in rows:
            row_dict = dict(row)
            provider = LLMProvider(
                id=row_dict["provider_id"],
                name=row_dict["provider_name"],
                display_name=row_dict["provider_display_name"],
                provider_type=row_dict["provider_type"],
                base_url=row_dict["base_url"],
                is_enabled=row_dict["provider_enabled"],
                last_health_check=None,
                health_status=HealthStatus.UNKNOWN,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            # Parse JSON fields for model (keep provider_id but exclude other provider_ fields)
            model_dict = {k: v for k, v in row_dict.items() if k == "provider_id" or not k.startswith("provider_")}
            model_dict = _parse_json_fields(model_dict, ["capabilities", "settings", "performance_metrics"])
            model = LLMModelWithProvider(
                **model_dict,
                provider=provider,
            )
            models.append(model)

        await self._cache_set(cache_key, [m.model_dump() for m in models])
        return models

    async def get_llm_model(self, model_id: int) -> LLMModel | None:
        """Get an LLM model by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM llm_models WHERE id = $1",
                model_id,
            )
        return LLMModel(**_parse_json_fields(dict(row), ["capabilities", "settings", "performance_metrics"])) if row else None

    async def create_llm_model(self, data: LLMModelCreate) -> LLMModel:
        """Create a new LLM model."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO llm_models
                    (provider_id, model_name, display_name, model_type, description,
                     capabilities, is_enabled, is_default_for_type, settings, performance_metrics)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING *
                    """,
                    data.provider_id,
                    data.model_name,
                    data.display_name,
                    data.model_type.value,
                    data.description,
                    json.dumps(data.capabilities.model_dump()),
                    data.is_enabled,
                    data.is_default_for_type,
                    json.dumps(data.settings),
                    json.dumps(data.performance_metrics),
                )
            await self._cache_delete("llm_models:")
            return LLMModel(**_parse_json_fields(dict(row), ["capabilities", "settings", "performance_metrics"]))
        except asyncpg.UniqueViolationError:
            raise ValueError(
                f"Model '{data.model_name}' already exists for this provider. "
                "Each provider can only have one model with the same name."
            )

    async def update_llm_model(
        self, model_id: int, data: LLMModelUpdate
    ) -> LLMModel | None:
        """Update an LLM model."""
        updates = {}
        if data.model_name is not None:
            updates["model_name"] = data.model_name
        if data.display_name is not None:
            updates["display_name"] = data.display_name
        if data.description is not None:
            updates["description"] = data.description
        if data.capabilities is not None:
            updates["capabilities"] = json.dumps(data.capabilities.model_dump())
        if data.is_enabled is not None:
            updates["is_enabled"] = data.is_enabled
        if data.is_default_for_type is not None:
            updates["is_default_for_type"] = data.is_default_for_type
        if data.settings is not None:
            updates["settings"] = json.dumps(data.settings)
        if data.performance_metrics is not None:
            updates["performance_metrics"] = json.dumps(data.performance_metrics)

        if not updates:
            return await self.get_llm_model(model_id)

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [model_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE llm_models SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("llm_models:")
        return LLMModel(**_parse_json_fields(dict(row), ["capabilities", "settings", "performance_metrics"])) if row else None

    async def delete_llm_model(self, model_id: int) -> bool:
        """Delete an LLM model."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM llm_models WHERE id = $1",
                model_id,
            )
        await self._cache_delete("llm_models:")
        return result == "DELETE 1"

    # =========================================================================
    # Agent Model Configs
    # =========================================================================

    async def get_agent_model_configs(
        self, agent_name: str | None = None
    ) -> list[AgentModelConfigWithModel]:
        """Get agent model configurations."""
        cache_key = f"agent_configs:agent:{agent_name or 'all'}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [AgentModelConfigWithModel(**c) for c in cached]

        query = """
            SELECT a.*, m.model_name, m.display_name as model_display_name,
                   m.model_type, m.provider_id,
                   p.name as provider_name, p.provider_type, p.base_url
            FROM agent_model_configs a
            LEFT JOIN llm_models m ON a.model_id = m.id
            LEFT JOIN llm_providers p ON m.provider_id = p.id
        """
        params = []

        if agent_name:
            query += " WHERE a.agent_name = $1"
            params.append(agent_name)

        query += " ORDER BY a.agent_name, a.priority"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        configs = []
        for row in rows:
            row_dict = dict(row)
            model = None
            if row_dict.get("model_id"):
                provider = LLMProvider(
                    id=row_dict["provider_id"],
                    name=row_dict["provider_name"],
                    provider_type=row_dict["provider_type"],
                    base_url=row_dict["base_url"],
                    last_health_check=None,
                    health_status=HealthStatus.UNKNOWN,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                model = LLMModelWithProvider(
                    id=row_dict["model_id"],
                    provider_id=row_dict["provider_id"],
                    model_name=row_dict["model_name"],
                    display_name=row_dict["model_display_name"],
                    model_type=row_dict["model_type"],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    provider=provider,
                )

            # Parse settings JSON if needed
            settings = row_dict["settings"]
            if isinstance(settings, str):
                import json as json_module
                try:
                    settings = json_module.loads(settings)
                except json_module.JSONDecodeError:
                    settings = {}

            config = AgentModelConfigWithModel(
                id=row_dict["id"],
                agent_name=row_dict["agent_name"],
                model_id=row_dict["model_id"],
                purpose=row_dict["purpose"],
                is_enabled=row_dict["is_enabled"],
                priority=row_dict["priority"],
                settings=settings,
                created_at=row_dict["created_at"],
                updated_at=row_dict["updated_at"],
                model=model,
            )
            configs.append(config)

        await self._cache_set(cache_key, [c.model_dump() for c in configs])
        return configs

    async def get_agent_model_config(
        self, agent_name: str, purpose: str = "primary"
    ) -> AgentModelConfigWithModel | None:
        """Get a specific agent model configuration."""
        configs = await self.get_agent_model_configs(agent_name)
        for config in configs:
            if config.purpose == purpose:
                return config
        return None

    async def create_agent_model_config(
        self, data: AgentModelConfigCreate
    ) -> AgentModelConfig:
        """Create a new agent model configuration."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO agent_model_configs
                (agent_name, model_id, purpose, is_enabled, priority, settings)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING *
                """,
                data.agent_name,
                data.model_id,
                data.purpose,
                data.is_enabled,
                data.priority,
                json.dumps(data.settings),
            )
        await self._cache_delete("agent_configs:")
        return AgentModelConfig(**_parse_json_fields(dict(row), ["settings"]))

    async def update_agent_model_config(
        self,
        agent_name: str,
        purpose: str,
        data: AgentModelConfigUpdate,
    ) -> AgentModelConfig | None:
        """Update an agent model configuration."""
        updates = {}
        if data.model_id is not None:
            updates["model_id"] = data.model_id
        if data.is_enabled is not None:
            updates["is_enabled"] = data.is_enabled
        if data.priority is not None:
            updates["priority"] = data.priority
        if data.settings is not None:
            updates["settings"] = json.dumps(data.settings)

        if not updates:
            config = await self.get_agent_model_config(agent_name, purpose)
            return config

        set_clause = ", ".join(f"{k} = ${i+3}" for i, k in enumerate(updates.keys()))
        values = [agent_name, purpose] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE agent_model_configs SET {set_clause}
                WHERE agent_name = $1 AND purpose = $2
                RETURNING *
                """,
                *values,
            )
        await self._cache_delete("agent_configs:")
        return AgentModelConfig(**_parse_json_fields(dict(row), ["settings"])) if row else None

    async def update_agent_model_config_by_id(
        self, config_id: int, data: AgentModelConfigUpdate
    ) -> AgentModelConfig | None:
        """Update an agent model configuration by ID."""
        updates = {}
        if data.model_id is not None:
            updates["model_id"] = data.model_id
        if data.is_enabled is not None:
            updates["is_enabled"] = data.is_enabled
        if data.priority is not None:
            updates["priority"] = data.priority
        if data.settings is not None:
            updates["settings"] = json.dumps(data.settings)

        if not updates:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM agent_model_configs WHERE id = $1", config_id
                )
            return AgentModelConfig(**_parse_json_fields(dict(row), ["settings"])) if row else None

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [config_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE agent_model_configs SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("agent_configs:")
        return AgentModelConfig(**_parse_json_fields(dict(row), ["settings"])) if row else None

    async def delete_agent_model_config(self, config_id: int) -> bool:
        """Delete an agent model configuration."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_model_configs WHERE id = $1",
                config_id,
            )
        await self._cache_delete("agent_configs:")
        return result == "DELETE 1"

    # =========================================================================
    # Search Strategies
    # =========================================================================

    async def get_search_strategies(
        self,
        strategy_type: str | None = None,
        enabled_only: bool = False,
    ) -> list[SearchStrategy]:
        """Get search strategies."""
        cache_key = f"search_strategies:type:{strategy_type}:enabled:{enabled_only}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [SearchStrategy(**s) for s in cached]

        query = "SELECT * FROM search_strategies WHERE 1=1"
        params = []
        param_idx = 1

        if strategy_type:
            query += f" AND strategy_type = ${param_idx}"
            params.append(strategy_type)
            param_idx += 1

        if enabled_only:
            query += " AND is_enabled = TRUE"

        query += " ORDER BY strategy_type, name"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        strategies = [SearchStrategy(**_parse_json_fields(dict(row), ["settings", "performance_metrics"])) for row in rows]
        await self._cache_set(cache_key, [s.model_dump() for s in strategies])
        return strategies

    async def get_search_strategy(self, strategy_id: int) -> SearchStrategy | None:
        """Get a search strategy by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM search_strategies WHERE id = $1",
                strategy_id,
            )
        return SearchStrategy(**_parse_json_fields(dict(row), ["settings", "performance_metrics"])) if row else None

    async def get_search_strategy_by_name(self, name: str) -> SearchStrategy | None:
        """Get a search strategy by name."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM search_strategies WHERE name = $1",
                name,
            )
        return SearchStrategy(**_parse_json_fields(dict(row), ["settings", "performance_metrics"])) if row else None

    async def create_search_strategy(
        self, data: SearchStrategyCreate
    ) -> SearchStrategy:
        """Create a new search strategy."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO search_strategies
                (name, display_name, strategy_type, description, implementation_class,
                 settings, is_enabled, is_default, performance_metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING *
                """,
                data.name,
                data.display_name,
                data.strategy_type.value,
                data.description,
                data.implementation_class,
                json.dumps(data.settings),
                data.is_enabled,
                data.is_default,
                json.dumps(data.performance_metrics),
            )
        await self._cache_delete("search_strategies:")
        return SearchStrategy(**_parse_json_fields(dict(row), ["settings", "performance_metrics"]))

    async def update_search_strategy(
        self, strategy_id: int, data: SearchStrategyUpdate
    ) -> SearchStrategy | None:
        """Update a search strategy."""
        updates = {}
        if data.display_name is not None:
            updates["display_name"] = data.display_name
        if data.description is not None:
            updates["description"] = data.description
        if data.settings is not None:
            updates["settings"] = json.dumps(data.settings)
        if data.is_enabled is not None:
            updates["is_enabled"] = data.is_enabled
        if data.is_default is not None:
            updates["is_default"] = data.is_default
        if data.performance_metrics is not None:
            updates["performance_metrics"] = json.dumps(data.performance_metrics)

        if not updates:
            return await self.get_search_strategy(strategy_id)

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [strategy_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE search_strategies SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("search_strategies:")
        return SearchStrategy(**_parse_json_fields(dict(row), ["settings", "performance_metrics"])) if row else None

    async def delete_search_strategy(self, strategy_id: int) -> bool:
        """Delete a search strategy."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM search_strategies WHERE id = $1",
                strategy_id,
            )
        await self._cache_delete("search_strategies:")
        return result == "DELETE 1"

    # =========================================================================
    # Query Strategy Mappings
    # =========================================================================

    async def get_query_strategy_mappings(
        self, query_type: str | None = None
    ) -> list[QueryStrategyMappingWithStrategy]:
        """Get query-strategy mappings."""
        cache_key = f"query_mappings:type:{query_type or 'all'}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [QueryStrategyMappingWithStrategy(**m) for m in cached]

        query = """
            SELECT m.*, s.name as strategy_name, s.display_name as strategy_display_name,
                   s.strategy_type, s.implementation_class as strategy_implementation_class,
                   s.settings as strategy_settings, s.is_enabled as strategy_enabled,
                   s.performance_metrics as strategy_performance_metrics,
                   s.created_at as strategy_created_at, s.updated_at as strategy_updated_at
            FROM query_strategy_mapping m
            JOIN search_strategies s ON m.strategy_id = s.id
        """
        params = []

        if query_type:
            query += " WHERE m.query_type = $1"
            params.append(query_type)

        query += " ORDER BY m.query_type, m.priority"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        mappings = []
        for row in rows:
            row_dict = dict(row)
            # Parse JSON fields if they're strings (from cache or DB)
            settings = row_dict["strategy_settings"]
            if isinstance(settings, str):
                try:
                    settings = json.loads(settings)
                except json.JSONDecodeError:
                    settings = {}
            performance_metrics = row_dict.get("strategy_performance_metrics", {})
            if isinstance(performance_metrics, str):
                try:
                    performance_metrics = json.loads(performance_metrics)
                except json.JSONDecodeError:
                    performance_metrics = {}
            # Parse conditions field if it's a string
            conditions = row_dict.get("conditions", {})
            if isinstance(conditions, str):
                try:
                    conditions = json.loads(conditions)
                except json.JSONDecodeError:
                    conditions = {}
            strategy = SearchStrategy(
                id=row_dict["strategy_id"],
                name=row_dict["strategy_name"],
                display_name=row_dict["strategy_display_name"],
                strategy_type=row_dict["strategy_type"],
                implementation_class=row_dict["strategy_implementation_class"],
                settings=settings,
                is_enabled=row_dict["strategy_enabled"],
                performance_metrics=performance_metrics,
                created_at=row_dict["strategy_created_at"],
                updated_at=row_dict["strategy_updated_at"],
            )
            mapping = QueryStrategyMappingWithStrategy(
                id=row_dict["id"],
                query_type=row_dict["query_type"],
                strategy_id=row_dict["strategy_id"],
                priority=row_dict["priority"],
                is_enabled=row_dict["is_enabled"],
                conditions=conditions,
                created_at=row_dict["created_at"],
                strategy=strategy,
            )
            mappings.append(mapping)

        await self._cache_set(cache_key, [m.model_dump() for m in mappings])
        return mappings

    async def get_strategies_for_query_type(
        self, query_type: str
    ) -> list[SearchStrategy]:
        """Get ordered strategies for a query type."""
        mappings = await self.get_query_strategy_mappings(query_type)
        if not mappings:
            # Fall back to DEFAULT
            mappings = await self.get_query_strategy_mappings("DEFAULT")

        strategies = []
        for mapping in mappings:
            if mapping.is_enabled and mapping.strategy and mapping.strategy.is_enabled:
                full_strategy = await self.get_search_strategy(mapping.strategy_id)
                if full_strategy:
                    strategies.append(full_strategy)

        return strategies

    async def create_query_strategy_mapping(
        self, data: QueryStrategyMappingCreate
    ) -> QueryStrategyMapping:
        """Create a new query-strategy mapping."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO query_strategy_mapping
                (query_type, strategy_id, priority, is_enabled, conditions)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                data.query_type,
                data.strategy_id,
                data.priority,
                data.is_enabled,
                json.dumps(data.conditions),
            )
        await self._cache_delete("query_mappings:")
        return QueryStrategyMapping(**dict(row))

    async def update_query_strategy_mapping(
        self, mapping_id: int, data: QueryStrategyMappingUpdate
    ) -> QueryStrategyMapping | None:
        """Update a query-strategy mapping."""
        updates = {}
        if data.priority is not None:
            updates["priority"] = data.priority
        if data.is_enabled is not None:
            updates["is_enabled"] = data.is_enabled
        if data.conditions is not None:
            updates["conditions"] = json.dumps(data.conditions)

        if not updates:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM query_strategy_mapping WHERE id = $1", mapping_id
                )
            return QueryStrategyMapping(**dict(row)) if row else None

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [mapping_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE query_strategy_mapping SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("query_mappings:")
        return QueryStrategyMapping(**dict(row)) if row else None

    async def delete_query_strategy_mapping(self, mapping_id: int) -> bool:
        """Delete a query-strategy mapping."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM query_strategy_mapping WHERE id = $1",
                mapping_id,
            )
        await self._cache_delete("query_mappings:")
        return result == "DELETE 1"

    # =========================================================================
    # Reranker Configs
    # =========================================================================

    async def get_reranker_configs(
        self, enabled_only: bool = False
    ) -> list[RerankerConfigWithModel]:
        """Get reranker configurations."""
        cache_key = f"reranker_configs:enabled:{enabled_only}"
        cached = await self._cache_get(cache_key)
        if cached:
            return [RerankerConfigWithModel(**c) for c in cached]

        query = """
            SELECT r.*, m.model_name, m.display_name as model_display_name,
                   m.model_type, m.provider_id
            FROM reranker_configs r
            LEFT JOIN llm_models m ON r.model_id = m.id
        """
        if enabled_only:
            query += " WHERE r.is_enabled = TRUE"
        query += " ORDER BY r.name"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query)

        configs = []
        for row in rows:
            row_dict = dict(row)
            model = None
            if row_dict.get("model_id"):
                model = LLMModelWithProvider(
                    id=row_dict["model_id"],
                    provider_id=row_dict["provider_id"],
                    model_name=row_dict["model_name"],
                    display_name=row_dict["model_display_name"],
                    model_type=row_dict["model_type"],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

            # Parse JSON fields if they're strings (from cache)
            settings = row_dict.get("settings", {})
            if isinstance(settings, str):
                try:
                    settings = json.loads(settings)
                except json.JSONDecodeError:
                    settings = {}
            performance_metrics = row_dict.get("performance_metrics", {})
            if isinstance(performance_metrics, str):
                try:
                    performance_metrics = json.loads(performance_metrics)
                except json.JSONDecodeError:
                    performance_metrics = {}

            config = RerankerConfigWithModel(
                id=row_dict["id"],
                name=row_dict["name"],
                display_name=row_dict["display_name"],
                model_id=row_dict["model_id"],
                is_enabled=row_dict["is_enabled"],
                is_default=row_dict["is_default"],
                settings=settings,
                performance_metrics=performance_metrics,
                created_at=row_dict["created_at"],
                updated_at=row_dict["updated_at"],
                model=model,
            )
            configs.append(config)

        await self._cache_set(cache_key, [c.model_dump() for c in configs])
        return configs

    async def get_reranker_config(self, reranker_id: int) -> RerankerConfigWithModel | None:
        """Get a reranker configuration by ID."""
        configs = await self.get_reranker_configs(enabled_only=False)
        for config in configs:
            if config.id == reranker_id:
                return config
        return None

    async def get_default_reranker(self) -> RerankerConfigWithModel | None:
        """Get the default reranker configuration."""
        configs = await self.get_reranker_configs(enabled_only=True)
        for config in configs:
            if config.is_default:
                return config
        return configs[0] if configs else None

    async def create_reranker_config(
        self, data: RerankerConfigCreate
    ) -> RerankerConfig:
        """Create a new reranker configuration."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO reranker_configs
                (name, display_name, model_id, is_enabled, is_default, settings, performance_metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                data.name,
                data.display_name,
                data.model_id,
                data.is_enabled,
                data.is_default,
                json.dumps(data.settings),
                json.dumps(data.performance_metrics),
            )
        await self._cache_delete("reranker_configs:")
        return RerankerConfig(**_parse_json_fields(dict(row), ["settings", "performance_metrics"]))

    async def update_reranker_config(
        self, config_id: int, data: RerankerConfigUpdate
    ) -> RerankerConfig | None:
        """Update a reranker configuration."""
        updates = {}
        if data.display_name is not None:
            updates["display_name"] = data.display_name
        if data.model_id is not None:
            updates["model_id"] = data.model_id
        if data.is_enabled is not None:
            updates["is_enabled"] = data.is_enabled
        if data.is_default is not None:
            updates["is_default"] = data.is_default
        if data.settings is not None:
            updates["settings"] = json.dumps(data.settings)
        if data.performance_metrics is not None:
            updates["performance_metrics"] = json.dumps(data.performance_metrics)

        if not updates:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM reranker_configs WHERE id = $1", config_id
                )
            return RerankerConfig(**_parse_json_fields(dict(row), ["settings", "performance_metrics"])) if row else None

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [config_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE reranker_configs SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        await self._cache_delete("reranker_configs:")
        return RerankerConfig(**_parse_json_fields(dict(row), ["settings", "performance_metrics"])) if row else None

    async def delete_reranker_config(self, reranker_id: int) -> bool:
        """Delete a reranker configuration."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM reranker_configs WHERE id = $1",
                reranker_id,
            )
        await self._cache_delete("reranker_configs:")
        return result == "DELETE 1"

    # =========================================================================
    # Conversation Sessions
    # =========================================================================

    async def get_conversation_session(
        self, session_id: UUID
    ) -> ConversationSession | None:
        """Get a conversation session by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversation_sessions WHERE id = $1",
                session_id,
            )
        return ConversationSession(**_parse_json_fields(dict(row), ["context", "metadata"])) if row else None

    async def create_conversation_session(
        self, data: ConversationSessionCreate | None = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        data = data or ConversationSessionCreate()
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_sessions (context, metadata, is_active)
                VALUES ($1, $2, $3)
                RETURNING *
                """,
                json.dumps(data.context.model_dump()),
                json.dumps(data.metadata),
                data.is_active,
            )
        return ConversationSession(**_parse_json_fields(dict(row), ["context", "metadata"]))

    async def update_conversation_session(
        self, session_id: UUID, data: ConversationSessionUpdate
    ) -> ConversationSession | None:
        """Update a conversation session."""
        updates = {"last_activity": datetime.now()}
        if data.context is not None:
            updates["context"] = json.dumps(data.context.model_dump())
        if data.metadata is not None:
            updates["metadata"] = json.dumps(data.metadata)
        if data.is_active is not None:
            updates["is_active"] = data.is_active

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
        values = [session_id] + list(updates.values())

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE conversation_sessions SET {set_clause} WHERE id = $1 RETURNING *",
                *values,
            )
        return ConversationSession(**_parse_json_fields(dict(row), ["context", "metadata"])) if row else None

    async def add_conversation_message(
        self, data: ConversationMessageCreate
    ) -> ConversationMessage:
        """Add a message to a conversation."""
        async with self.db_pool.acquire() as conn:
            # Insert message
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_messages
                (session_id, role, content, intent, entities, products, resolved_references, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
                """,
                data.session_id,
                data.role,
                data.content,
                data.intent,
                json.dumps(data.entities),
                json.dumps(data.products),
                json.dumps(data.resolved_references),
                json.dumps(data.metadata),
            )

            # Update session message count
            await conn.execute(
                """
                UPDATE conversation_sessions
                SET message_count = message_count + 1, last_activity = NOW()
                WHERE id = $1
                """,
                data.session_id,
            )

        return ConversationMessage(**_parse_json_fields(dict(row), ["entities", "products", "resolved_references", "metadata"]))

    async def get_conversation_messages(
        self, session_id: UUID, limit: int = 50
    ) -> list[ConversationMessage]:
        """Get messages for a conversation session."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_messages
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )
        return [ConversationMessage(**_parse_json_fields(dict(row), ["entities", "products", "resolved_references", "metadata"])) for row in reversed(rows)]

    # =========================================================================
    # Audit Logs
    # =========================================================================

    async def get_audit_logs(
        self,
        table_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConfigAuditLog]:
        """Get configuration audit logs."""
        query = "SELECT * FROM config_audit_log"
        params = []
        param_idx = 1

        if table_name:
            query += f" WHERE table_name = ${param_idx}"
            params.append(table_name)
            param_idx += 1

        query += f" ORDER BY changed_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            ConfigAuditLog(**_parse_json_fields(dict(row), ["old_value", "new_value"]))
            for row in rows
        ]


# =============================================================================
# Singleton
# =============================================================================

_config_repository: ConfigRepository | None = None


async def get_config_repository(
    db_pool: asyncpg.Pool | None = None,
    redis: Redis | None = None,
) -> ConfigRepository:
    """Get or create config repository singleton.

    Args:
        db_pool: PostgreSQL connection pool
        redis: Redis client for caching

    Returns:
        ConfigRepository instance
    """
    global _config_repository
    if _config_repository is None:
        if db_pool is None:
            raise ValueError("db_pool required for initial repository creation")
        _config_repository = ConfigRepository(db_pool, redis)
    return _config_repository
