"""Pydantic models for configuration management.

These models are used for:
- API request/response validation
- Database record mapping
- Configuration serialization
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ValueType(str, Enum):
    """Supported configuration value types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    JSON = "json"
    LIST = "list"
    SECRET = "secret"


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelType(str, Enum):
    """Supported LLM model types."""
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class StrategyType(str, Enum):
    """Supported search strategy types."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SECTION = "section"


class HealthStatus(str, Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# Configuration Category
# =============================================================================

class ConfigCategoryBase(BaseModel):
    """Base model for config category."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    icon: str | None = None
    display_order: int = 0
    is_visible: bool = True


class ConfigCategoryCreate(ConfigCategoryBase):
    """Model for creating a config category."""
    pass


class ConfigCategoryUpdate(BaseModel):
    """Model for updating a config category."""
    description: str | None = None
    icon: str | None = None
    display_order: int | None = None
    is_visible: bool | None = None


class ConfigCategory(ConfigCategoryBase):
    """Full config category model with database fields."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Configuration Setting
# =============================================================================

class ValidationRules(BaseModel):
    """Validation rules for a config setting."""
    min: float | None = None
    max: float | None = None
    pattern: str | None = None
    options: list[str] | None = None
    required: bool = False


class ConfigSettingBase(BaseModel):
    """Base model for config setting."""
    key: str = Field(..., min_length=1, max_length=200)
    value: Any
    value_type: ValueType
    default_value: Any | None = None
    label: str | None = None
    description: str | None = None
    is_sensitive: bool = False
    is_readonly: bool = False
    validation_rules: ValidationRules | None = None
    display_order: int = 0


class ConfigSettingCreate(ConfigSettingBase):
    """Model for creating a config setting."""
    category_id: int | None = None


class ConfigSettingUpdate(BaseModel):
    """Model for updating a config setting."""
    value: Any | None = None
    label: str | None = None
    description: str | None = None
    is_sensitive: bool | None = None
    validation_rules: ValidationRules | None = None
    display_order: int | None = None


class ConfigSetting(ConfigSettingBase):
    """Full config setting model with database fields."""
    id: int
    category_id: int | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# LLM Provider
# =============================================================================

class LLMProviderBase(BaseModel):
    """Base model for LLM provider."""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str | None = None
    provider_type: ProviderType
    base_url: str | None = None
    is_enabled: bool = True
    is_default: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)
    health_check_url: str | None = None


class LLMProviderCreate(LLMProviderBase):
    """Model for creating an LLM provider."""
    api_key: str | None = None  # Will be encrypted before storage


class LLMProviderUpdate(BaseModel):
    """Model for updating an LLM provider."""
    display_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None
    settings: dict[str, Any] | None = None
    health_check_url: str | None = None


class LLMProvider(LLMProviderBase):
    """Full LLM provider model with database fields."""
    id: int
    last_health_check: datetime | None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# LLM Model
# =============================================================================

class ModelCapabilities(BaseModel):
    """LLM model capabilities."""
    max_context: int | None = None
    max_tokens: int | None = None
    supports_json: bool = False
    supports_streaming: bool = False
    supports_function_calling: bool = False
    dimensions: int | None = None  # For embedding models


class LLMModelBase(BaseModel):
    """Base model for LLM model."""
    model_name: str = Field(..., min_length=1, max_length=200)
    display_name: str | None = None
    model_type: ModelType
    description: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    is_enabled: bool = True
    is_default_for_type: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, Any] = Field(default_factory=dict)


class LLMModelCreate(LLMModelBase):
    """Model for creating an LLM model."""
    provider_id: int


class LLMModelUpdate(BaseModel):
    """Model for updating an LLM model."""
    model_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    capabilities: ModelCapabilities | None = None
    is_enabled: bool | None = None
    is_default_for_type: bool | None = None
    settings: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None


class LLMModel(LLMModelBase):
    """Full LLM model with database fields."""
    id: int
    provider_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMModelWithProvider(LLMModel):
    """LLM model with provider information."""
    provider: LLMProvider | None = None


# =============================================================================
# Agent Model Config
# =============================================================================

class AgentModelConfigBase(BaseModel):
    """Base model for agent model configuration."""
    agent_name: str = Field(..., min_length=1, max_length=100)
    purpose: str = "primary"  # primary, fallback, embedding, reranker
    is_enabled: bool = True
    priority: int = 0
    settings: dict[str, Any] = Field(default_factory=dict)


class AgentModelConfigCreate(AgentModelConfigBase):
    """Model for creating an agent model config."""
    model_id: int | None = None


class AgentModelConfigUpdate(BaseModel):
    """Model for updating an agent model config."""
    model_id: int | None = None
    is_enabled: bool | None = None
    priority: int | None = None
    settings: dict[str, Any] | None = None


class AgentModelConfig(AgentModelConfigBase):
    """Full agent model config with database fields."""
    id: int
    model_id: int | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentModelConfigWithModel(AgentModelConfig):
    """Agent model config with model information."""
    model: LLMModelWithProvider | None = None


# =============================================================================
# Search Strategy
# =============================================================================

class SearchStrategyBase(BaseModel):
    """Base model for search strategy."""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str | None = None
    strategy_type: StrategyType
    description: str | None = None
    implementation_class: str = Field(..., min_length=1, max_length=200)
    settings: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True
    is_default: bool = False
    performance_metrics: dict[str, Any] = Field(default_factory=dict)


class SearchStrategyCreate(SearchStrategyBase):
    """Model for creating a search strategy."""
    pass


class SearchStrategyUpdate(BaseModel):
    """Model for updating a search strategy."""
    display_name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None
    performance_metrics: dict[str, Any] | None = None


class SearchStrategy(SearchStrategyBase):
    """Full search strategy with database fields."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Query Strategy Mapping
# =============================================================================

class QueryStrategyMappingBase(BaseModel):
    """Base model for query-strategy mapping."""
    query_type: str = Field(..., min_length=1, max_length=100)
    priority: int = 0
    is_enabled: bool = True
    conditions: dict[str, Any] = Field(default_factory=dict)


class QueryStrategyMappingCreate(QueryStrategyMappingBase):
    """Model for creating a query-strategy mapping."""
    strategy_id: int


class QueryStrategyMappingUpdate(BaseModel):
    """Model for updating a query-strategy mapping."""
    priority: int | None = None
    is_enabled: bool | None = None
    conditions: dict[str, Any] | None = None


class QueryStrategyMapping(QueryStrategyMappingBase):
    """Full query-strategy mapping with database fields."""
    id: int
    strategy_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class QueryStrategyMappingWithStrategy(QueryStrategyMapping):
    """Query-strategy mapping with strategy information."""
    strategy: SearchStrategy | None = None


# =============================================================================
# Reranker Config
# =============================================================================

class RerankerConfigBase(BaseModel):
    """Base model for reranker configuration."""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str | None = None
    is_enabled: bool = True
    is_default: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, Any] = Field(default_factory=dict)


class RerankerConfigCreate(RerankerConfigBase):
    """Model for creating a reranker config."""
    model_id: int | None = None


class RerankerConfigUpdate(BaseModel):
    """Model for updating a reranker config."""
    display_name: str | None = None
    model_id: int | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None
    settings: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None


class RerankerConfig(RerankerConfigBase):
    """Full reranker config with database fields."""
    id: int
    model_id: int | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RerankerConfigWithModel(RerankerConfig):
    """Reranker config with model information."""
    model: LLMModelWithProvider | None = None


# =============================================================================
# Config Audit Log
# =============================================================================

class ConfigAuditLog(BaseModel):
    """Audit log entry for configuration changes."""
    id: int
    table_name: str
    record_id: int
    action: str  # INSERT, UPDATE, DELETE
    old_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    changed_fields: list[str] | None
    changed_by: str | None
    ip_address: str | None
    user_agent: str | None
    changed_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Conversation Session
# =============================================================================

class ConversationContext(BaseModel):
    """Conversation context for multi-turn interactions."""
    products: list[dict[str, Any]] = Field(default_factory=list)
    product_asins: list[str] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    last_query_type: str | None = None
    last_agent: str | None = None
    last_intent: str | None = None


class ConversationSessionBase(BaseModel):
    """Base model for conversation session."""
    context: ConversationContext = Field(default_factory=ConversationContext)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ConversationSessionCreate(ConversationSessionBase):
    """Model for creating a conversation session."""
    pass


class ConversationSessionUpdate(BaseModel):
    """Model for updating a conversation session."""
    context: ConversationContext | None = None
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None


class ConversationSession(ConversationSessionBase):
    """Full conversation session with database fields."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    last_activity: datetime
    expires_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


# =============================================================================
# Conversation Message
# =============================================================================

class ConversationMessageBase(BaseModel):
    """Base model for conversation message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    intent: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    products: list[dict[str, Any]] = Field(default_factory=list)
    resolved_references: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessageCreate(ConversationMessageBase):
    """Model for creating a conversation message."""
    session_id: UUID


class ConversationMessage(ConversationMessageBase):
    """Full conversation message with database fields."""
    id: int
    session_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Response Models for API
# =============================================================================

class ConfigCategoryList(BaseModel):
    """Response model for listing config categories."""
    categories: list[ConfigCategory]
    total: int


class ConfigSettingList(BaseModel):
    """Response model for listing config settings."""
    settings: list[ConfigSetting]
    total: int


class LLMProviderList(BaseModel):
    """Response model for listing LLM providers."""
    providers: list[LLMProvider]
    total: int


class LLMModelList(BaseModel):
    """Response model for listing LLM models."""
    models: list[LLMModelWithProvider]
    total: int


class AgentModelConfigList(BaseModel):
    """Response model for listing agent model configs."""
    configs: list[AgentModelConfigWithModel]
    total: int


class SearchStrategyList(BaseModel):
    """Response model for listing search strategies."""
    strategies: list[SearchStrategy]
    total: int


class QueryStrategyMappingList(BaseModel):
    """Response model for listing query-strategy mappings."""
    mappings: list[QueryStrategyMappingWithStrategy]
    total: int


class RerankerConfigList(BaseModel):
    """Response model for listing reranker configs."""
    configs: list[RerankerConfigWithModel]
    total: int
