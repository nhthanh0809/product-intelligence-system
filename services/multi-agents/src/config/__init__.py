"""Configuration management module.

This module provides:
- Environment-based settings via get_settings()
- Database-managed configuration with ConfigManager/ConfigRepository
- PostgreSQL storage for configurations
- Redis caching for performance
- Pydantic models for validation
"""

# Environment-based settings (backward compatible)
from src.config.settings import Settings, get_settings

# Database-managed configuration models
from src.config.models import (
    ConfigCategory,
    ConfigSetting,
    LLMProvider,
    LLMModel,
    AgentModelConfig,
    SearchStrategy,
    QueryStrategyMapping,
    RerankerConfig,
    ConfigAuditLog,
    ConversationSession,
    ConversationMessage,
)
from src.config.repository import ConfigRepository, get_config_repository
from src.config.manager import ConfigManager, get_config_manager

__all__ = [
    # Environment settings
    "Settings",
    "get_settings",
    # Configuration models
    "ConfigCategory",
    "ConfigSetting",
    "LLMProvider",
    "LLMModel",
    "AgentModelConfig",
    "SearchStrategy",
    "QueryStrategyMapping",
    "RerankerConfig",
    "ConfigAuditLog",
    "ConversationSession",
    "ConversationMessage",
    # Repository
    "ConfigRepository",
    "get_config_repository",
    # Manager
    "ConfigManager",
    "get_config_manager",
]
