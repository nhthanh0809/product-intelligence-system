"""Configuration API router for managing system settings."""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Depends

from src.config.models import (
    # Categories
    ConfigCategory,
    ConfigCategoryCreate,
    ConfigCategoryUpdate,
    ConfigCategoryList,
    # Settings
    ConfigSetting,
    ConfigSettingCreate,
    ConfigSettingUpdate,
    ConfigSettingList,
    # LLM Providers
    LLMProvider,
    LLMProviderCreate,
    LLMProviderUpdate,
    LLMProviderList,
    # LLM Models
    LLMModel,
    LLMModelCreate,
    LLMModelUpdate,
    LLMModelList,
    LLMModelWithProvider,
    # Agent Model Configs
    AgentModelConfig,
    AgentModelConfigCreate,
    AgentModelConfigUpdate,
    AgentModelConfigList,
    AgentModelConfigWithModel,
    # Search Strategies
    SearchStrategy,
    SearchStrategyCreate,
    SearchStrategyUpdate,
    SearchStrategyList,
    # Query Strategy Mapping
    QueryStrategyMapping,
    QueryStrategyMappingCreate,
    QueryStrategyMappingUpdate,
    QueryStrategyMappingList,
    # Reranker Configs
    RerankerConfig,
    RerankerConfigCreate,
    RerankerConfigUpdate,
    RerankerConfigList,
    RerankerConfigWithModel,
    # Audit
    ConfigAuditLog,
)
from src.config.manager import ConfigManager, get_config_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/api/config", tags=["Configuration"])


# =============================================================================
# Dependencies
# =============================================================================

async def get_manager() -> ConfigManager:
    """Get configuration manager dependency."""
    try:
        return await get_config_manager()
    except ValueError as e:
        logger.error("config_manager_not_initialized", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Configuration service not available. Database may not be initialized."
        )


# =============================================================================
# Category Endpoints
# =============================================================================

@router.get("/categories", response_model=ConfigCategoryList)
async def list_categories(
    manager: ConfigManager = Depends(get_manager),
):
    """List all configuration categories."""
    categories = await manager.repository.get_categories()
    return ConfigCategoryList(categories=categories, total=len(categories))


@router.post("/categories", response_model=ConfigCategory, status_code=201)
async def create_category(
    data: ConfigCategoryCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new configuration category."""
    category = await manager.repository.create_category(data)
    if not category:
        raise HTTPException(status_code=400, detail="Failed to create category")
    return category


@router.get("/categories/{category_id}", response_model=ConfigCategory)
async def get_category(
    category_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Get a configuration category by ID."""
    category = await manager.repository.get_category(category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category


@router.patch("/categories/{category_id}", response_model=ConfigCategory)
async def update_category(
    category_id: int,
    data: ConfigCategoryUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update a configuration category."""
    category = await manager.repository.update_category(category_id, data)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category


@router.delete("/categories/{category_id}", status_code=204)
async def delete_category(
    category_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete a configuration category."""
    deleted = await manager.repository.delete_category(category_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Category not found")


# =============================================================================
# Settings Endpoints
# =============================================================================

@router.get("/settings", response_model=ConfigSettingList)
async def list_settings(
    category_id: int | None = Query(None, description="Filter by category"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all configuration settings."""
    settings = await manager.repository.get_settings(category_id)
    return ConfigSettingList(settings=settings, total=len(settings))


@router.post("/settings", response_model=ConfigSetting, status_code=201)
async def create_setting(
    data: ConfigSettingCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new configuration setting."""
    setting = await manager.repository.create_setting(data)
    if not setting:
        raise HTTPException(status_code=400, detail="Failed to create setting")
    return setting


@router.get("/settings/value/{key:path}")
async def get_setting_value(
    key: str,
    default: str | None = Query(None, description="Default value if not found"),
    manager: ConfigManager = Depends(get_manager),
):
    """Get just the value of a configuration setting."""
    value = await manager.repository.get_setting_value(key, default)
    return {"key": key, "value": value}


@router.get("/settings/{key:path}", response_model=ConfigSetting)
async def get_setting(
    key: str,
    manager: ConfigManager = Depends(get_manager),
):
    """Get a configuration setting by key."""
    setting = await manager.repository.get_setting(key)
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting


@router.patch("/settings/{key:path}", response_model=ConfigSetting)
async def update_setting(
    key: str,
    data: ConfigSettingUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update a configuration setting."""
    setting = await manager.repository.update_setting(key, data)
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting


@router.delete("/settings/{key:path}", status_code=204)
async def delete_setting(
    key: str,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete a configuration setting."""
    deleted = await manager.repository.delete_setting(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Setting not found")


# =============================================================================
# LLM Provider Endpoints
# =============================================================================

@router.get("/llm/providers", response_model=LLMProviderList)
async def list_llm_providers(
    enabled_only: bool = Query(False, description="Only return enabled providers"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all LLM providers."""
    providers = await manager.repository.get_llm_providers(enabled_only=enabled_only)
    return LLMProviderList(providers=providers, total=len(providers))


@router.post("/llm/providers", response_model=LLMProvider, status_code=201)
async def create_llm_provider(
    data: LLMProviderCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new LLM provider."""
    try:
        provider = await manager.repository.create_llm_provider(data)
        if not provider:
            raise HTTPException(status_code=400, detail="Failed to create provider")
        return provider
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/llm/providers/default", response_model=LLMProvider)
async def get_default_llm_provider(
    manager: ConfigManager = Depends(get_manager),
):
    """Get the default LLM provider."""
    provider = await manager.get_default_llm_provider()
    if not provider:
        raise HTTPException(status_code=404, detail="No default provider configured")
    return provider


@router.get("/llm/providers/{provider_id}", response_model=LLMProvider)
async def get_llm_provider(
    provider_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Get an LLM provider by ID."""
    provider = await manager.repository.get_llm_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.patch("/llm/providers/{provider_id}", response_model=LLMProvider)
async def update_llm_provider(
    provider_id: int,
    data: LLMProviderUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update an LLM provider."""
    provider = await manager.repository.update_llm_provider(provider_id, data)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.delete("/llm/providers/{provider_id}", status_code=204)
async def delete_llm_provider(
    provider_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete an LLM provider."""
    deleted = await manager.repository.delete_llm_provider(provider_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Provider not found")


@router.post("/llm/providers/{provider_id}/health-check")
async def check_provider_health(
    provider_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Check health of an LLM provider."""
    provider = await manager.repository.get_llm_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Perform health check
    health = await manager.repository.check_provider_health(provider_id)
    return {
        "provider_id": provider_id,
        "provider_name": provider.name,
        "health_status": health.value if health else "unknown",
    }


@router.post("/llm/providers/{provider_id}/test-connection")
async def test_provider_connection(
    provider_id: int,
    model_id: int | None = Query(None, description="Specific model to test"),
    test_prompt: str = Query("Say hello in one sentence.", description="Test prompt"),
    manager: ConfigManager = Depends(get_manager),
):
    """Test LLM provider connection by generating a response.

    This endpoint actually calls the LLM to verify the connection works
    and returns the generated response along with timing information.
    """
    from src.llm.manager import get_llm_manager
    from src.llm.base import GenerationConfig, LLMProviderError
    import time

    # Get provider info
    provider_model = await manager.repository.get_llm_provider(provider_id)
    if not provider_model:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Get the LLM manager
    try:
        llm_manager = await get_llm_manager(manager.repository)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM Manager not available: {str(e)}"
        )

    # Get the provider instance
    provider = llm_manager.get_provider(provider_id)
    if not provider:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{provider_model.name}' is not initialized"
        )

    # Determine which model to use
    model_name = None
    if model_id:
        model = await manager.repository.get_llm_model(model_id)
        if model and model.provider_id == provider_id:
            model_name = model.model_name

    if not model_name:
        # Get first chat model for this provider
        models = await manager.repository.get_llm_models(
            provider_id=provider_id,
            model_type="chat",
            enabled_only=True,
        )
        if models:
            model_name = models[0].model_name
        else:
            raise HTTPException(
                status_code=400,
                detail="No chat model configured for this provider"
            )

    # Test the connection
    start_time = time.time()
    try:
        result = await provider.generate(
            prompt=test_prompt,
            model=model_name,
            config=GenerationConfig(
                max_tokens=100,
                temperature=0.7,
            ),
        )
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "success": True,
            "provider_id": provider_id,
            "provider_name": provider_model.name,
            "model": model_name,
            "prompt": test_prompt,
            "response": result.content,
            "tokens": {
                "prompt": result.usage.get("prompt_tokens", 0),
                "completion": result.usage.get("completion_tokens", 0),
                "total": result.total_tokens,
            },
            "latency_ms": round(elapsed_ms, 2),
            "finish_reason": result.finish_reason,
        }

    except LLMProviderError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "provider_id": provider_id,
            "provider_name": provider_model.name,
            "model": model_name,
            "prompt": test_prompt,
            "error": str(e),
            "latency_ms": round(elapsed_ms, 2),
            "retryable": e.retryable,
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "test_connection_failed",
            provider_id=provider_id,
            error=str(e),
        )
        return {
            "success": False,
            "provider_id": provider_id,
            "provider_name": provider_model.name,
            "model": model_name,
            "prompt": test_prompt,
            "error": str(e),
            "latency_ms": round(elapsed_ms, 2),
        }


# =============================================================================
# LLM Model Endpoints
# =============================================================================

@router.get("/llm/models", response_model=LLMModelList)
async def list_llm_models(
    provider_id: int | None = Query(None, description="Filter by provider"),
    model_type: str | None = Query(None, description="Filter by model type"),
    enabled_only: bool = Query(False, description="Only return enabled models"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all LLM models."""
    models = await manager.repository.get_llm_models(
        provider_id=provider_id,
        model_type=model_type,
        enabled_only=enabled_only,
    )
    return LLMModelList(models=models, total=len(models))


@router.post("/llm/models", response_model=LLMModel, status_code=201)
async def create_llm_model(
    data: LLMModelCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new LLM model."""
    try:
        model = await manager.repository.create_llm_model(data)
        if not model:
            raise HTTPException(status_code=400, detail="Failed to create model")
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/llm/models/{model_id}", response_model=LLMModelWithProvider)
async def get_llm_model(
    model_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Get an LLM model by ID."""
    model = await manager.repository.get_llm_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.patch("/llm/models/{model_id}", response_model=LLMModel)
async def update_llm_model(
    model_id: int,
    data: LLMModelUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update an LLM model."""
    model = await manager.repository.update_llm_model(model_id, data)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/llm/models/{model_id}", status_code=204)
async def delete_llm_model(
    model_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete an LLM model."""
    deleted = await manager.repository.delete_llm_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")


# =============================================================================
# Agent Model Config Endpoints
# =============================================================================

@router.get("/agents", response_model=AgentModelConfigList)
async def list_agent_configs(
    agent_name: str | None = Query(None, description="Filter by agent name"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all agent model configurations."""
    configs = await manager.repository.get_agent_model_configs(agent_name=agent_name)
    return AgentModelConfigList(configs=configs, total=len(configs))


@router.post("/agents", response_model=AgentModelConfig, status_code=201)
async def create_agent_config(
    data: AgentModelConfigCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new agent model configuration."""
    config = await manager.repository.create_agent_model_config(data)
    if not config:
        raise HTTPException(status_code=400, detail="Failed to create agent config")
    return config


@router.get("/agents/{agent_name}", response_model=AgentModelConfigWithModel)
async def get_agent_config(
    agent_name: str,
    purpose: str = Query("primary", description="Purpose of the model config"),
    manager: ConfigManager = Depends(get_manager),
):
    """Get agent model configuration."""
    config = await manager.get_agent_model(agent_name, purpose)
    if not config:
        raise HTTPException(status_code=404, detail="Agent config not found")
    return config


@router.get("/agents/{agent_name}/settings")
async def get_agent_settings(
    agent_name: str,
    manager: ConfigManager = Depends(get_manager),
):
    """Get all settings for an agent."""
    settings = await manager.get_agent_settings(agent_name)
    return {"agent_name": agent_name, "settings": settings}


@router.patch("/agents/{config_id}", response_model=AgentModelConfig)
async def update_agent_config(
    config_id: int,
    data: AgentModelConfigUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update an agent model configuration."""
    config = await manager.repository.update_agent_model_config_by_id(config_id, data)
    if not config:
        raise HTTPException(status_code=404, detail="Agent config not found")
    return config


@router.delete("/agents/{config_id}", status_code=204)
async def delete_agent_config(
    config_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete an agent model configuration."""
    deleted = await manager.repository.delete_agent_model_config(config_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent config not found")


# =============================================================================
# Search Strategy Endpoints
# =============================================================================

@router.get("/search/strategies", response_model=SearchStrategyList)
async def list_search_strategies(
    strategy_type: str | None = Query(None, description="Filter by strategy type"),
    enabled_only: bool = Query(True, description="Only return enabled strategies"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all search strategies."""
    strategies = await manager.repository.get_search_strategies(
        strategy_type=strategy_type,
        enabled_only=enabled_only,
    )
    return SearchStrategyList(strategies=strategies, total=len(strategies))


@router.post("/search/strategies", response_model=SearchStrategy, status_code=201)
async def create_search_strategy(
    data: SearchStrategyCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new search strategy."""
    strategy = await manager.repository.create_search_strategy(data)
    if not strategy:
        raise HTTPException(status_code=400, detail="Failed to create strategy")
    return strategy


@router.get("/search/strategies/{strategy_id}", response_model=SearchStrategy)
async def get_search_strategy(
    strategy_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Get a search strategy by ID."""
    strategy = await manager.repository.get_search_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


@router.patch("/search/strategies/{strategy_id}", response_model=SearchStrategy)
async def update_search_strategy(
    strategy_id: int,
    data: SearchStrategyUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update a search strategy."""
    strategy = await manager.repository.update_search_strategy(strategy_id, data)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


@router.delete("/search/strategies/{strategy_id}", status_code=204)
async def delete_search_strategy(
    strategy_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete a search strategy."""
    deleted = await manager.repository.delete_search_strategy(strategy_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Strategy not found")


# =============================================================================
# Query Strategy Mapping Endpoints
# =============================================================================

@router.get("/search/mappings", response_model=QueryStrategyMappingList)
async def list_query_strategy_mappings(
    query_type: str | None = Query(None, description="Filter by query type"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all query-strategy mappings."""
    mappings = await manager.repository.get_query_strategy_mappings(query_type=query_type)
    return QueryStrategyMappingList(mappings=mappings, total=len(mappings))


@router.post("/search/mappings", response_model=QueryStrategyMapping, status_code=201)
async def create_query_strategy_mapping(
    data: QueryStrategyMappingCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new query-strategy mapping."""
    mapping = await manager.repository.create_query_strategy_mapping(data)
    if not mapping:
        raise HTTPException(status_code=400, detail="Failed to create mapping")
    return mapping


@router.get("/search/mappings/for-query-type/{query_type}")
async def get_strategies_for_query_type(
    query_type: str,
    manager: ConfigManager = Depends(get_manager),
):
    """Get all strategies for a query type (in priority order)."""
    strategies = await manager.get_strategies_for_query_type(query_type)
    return {
        "query_type": query_type,
        "strategies": [
            {"id": s.id, "name": s.name, "strategy_type": s.strategy_type.value}
            for s in strategies
        ],
    }


@router.patch("/search/mappings/{mapping_id}", response_model=QueryStrategyMapping)
async def update_query_strategy_mapping(
    mapping_id: int,
    data: QueryStrategyMappingUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update a query-strategy mapping."""
    mapping = await manager.repository.update_query_strategy_mapping(mapping_id, data)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    return mapping


@router.delete("/search/mappings/{mapping_id}", status_code=204)
async def delete_query_strategy_mapping(
    mapping_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete a query-strategy mapping."""
    deleted = await manager.repository.delete_query_strategy_mapping(mapping_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Mapping not found")


# =============================================================================
# Reranker Config Endpoints
# =============================================================================

@router.get("/rerankers", response_model=RerankerConfigList)
async def list_reranker_configs(
    enabled_only: bool = Query(True, description="Only return enabled rerankers"),
    manager: ConfigManager = Depends(get_manager),
):
    """List all reranker configurations."""
    configs = await manager.repository.get_reranker_configs(enabled_only=enabled_only)
    return RerankerConfigList(configs=configs, total=len(configs))


@router.post("/rerankers", response_model=RerankerConfig, status_code=201)
async def create_reranker_config(
    data: RerankerConfigCreate,
    manager: ConfigManager = Depends(get_manager),
):
    """Create a new reranker configuration."""
    config = await manager.repository.create_reranker_config(data)
    if not config:
        raise HTTPException(status_code=400, detail="Failed to create reranker config")
    return config


@router.get("/rerankers/default", response_model=RerankerConfigWithModel)
async def get_default_reranker(
    manager: ConfigManager = Depends(get_manager),
):
    """Get the default reranker configuration."""
    config = await manager.get_reranker_config()
    if not config:
        raise HTTPException(status_code=404, detail="No default reranker configured or reranker is disabled")
    return config


@router.get("/rerankers/{reranker_id}", response_model=RerankerConfigWithModel)
async def get_reranker_config(
    reranker_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Get a reranker configuration by ID."""
    config = await manager.repository.get_reranker_config(reranker_id)
    if not config:
        raise HTTPException(status_code=404, detail="Reranker config not found")
    return config


@router.patch("/rerankers/{reranker_id}", response_model=RerankerConfig)
async def update_reranker_config(
    reranker_id: int,
    data: RerankerConfigUpdate,
    manager: ConfigManager = Depends(get_manager),
):
    """Update a reranker configuration."""
    config = await manager.repository.update_reranker_config(reranker_id, data)
    if not config:
        raise HTTPException(status_code=404, detail="Reranker config not found")
    return config


@router.delete("/rerankers/{reranker_id}", status_code=204)
async def delete_reranker_config(
    reranker_id: int,
    manager: ConfigManager = Depends(get_manager),
):
    """Delete a reranker configuration."""
    deleted = await manager.repository.delete_reranker_config(reranker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Reranker config not found")


# =============================================================================
# Bulk Operations
# =============================================================================

@router.get("/export")
async def export_config(
    manager: ConfigManager = Depends(get_manager),
):
    """Export all configuration as JSON."""
    config = await manager.export_config()
    return config


@router.post("/import")
async def import_config(
    config: dict[str, Any],
    manager: ConfigManager = Depends(get_manager),
):
    """Import configuration from JSON."""
    count = await manager.import_config(config)
    return {"imported": count}


@router.get("/by-category")
async def get_settings_by_category(
    manager: ConfigManager = Depends(get_manager),
):
    """Get all settings grouped by category."""
    result = await manager.get_all_settings_by_category()
    return {
        category: [s.model_dump() for s in settings]
        for category, settings in result.items()
    }


# =============================================================================
# Audit Log Endpoints
# =============================================================================

@router.get("/audit")
async def get_audit_logs(
    table_name: str | None = Query(None, description="Filter by table name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    manager: ConfigManager = Depends(get_manager),
):
    """Get configuration audit logs."""
    logs = await manager.repository.get_audit_logs(
        table_name=table_name,
        limit=limit,
        offset=offset,
    )
    return {"logs": logs, "total": len(logs)}


# =============================================================================
# LLM Manager Reload
# =============================================================================

@router.post("/llm/reload")
async def reload_llm_providers(
    manager: ConfigManager = Depends(get_manager),
):
    """Reload LLM providers from database configuration.

    Use this after updating provider or model configurations to apply changes
    without restarting the service.
    """
    from src.llm.manager import get_llm_manager

    try:
        llm_manager = await get_llm_manager(manager.repository)
        await llm_manager.reload_config()

        # Get updated provider info
        providers = list(llm_manager._providers.keys())

        return {
            "success": True,
            "message": "LLM providers reloaded successfully",
            "initialized_providers": providers,
        }

    except Exception as e:
        logger.error("llm_reload_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload LLM providers: {str(e)}"
        )
