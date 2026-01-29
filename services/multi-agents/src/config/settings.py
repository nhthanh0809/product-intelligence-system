"""Application settings for Multi-Agent Service.

This module provides environment-based configuration using pydantic-settings.
For database-managed configuration, use the ConfigManager and ConfigRepository.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    service_name: str = "multi-agent"
    service_port: int = 8001
    log_level: str = "INFO"

    # Service URLs
    vector_store_url: str = Field(default="http://vector-store:8002")
    ollama_service_url: str = Field(default="http://ollama-service:8010")

    # PostgreSQL
    postgres_host: str = Field(default="postgres")
    postgres_port: int = 5432
    postgres_user: str = Field(default="pis_user")
    postgres_password: str = Field(default="pis_password")
    postgres_db: str = Field(default="product_intelligence")

    # Redis
    redis_host: str = Field(default="redis")
    redis_port: int = 6379

    # Qdrant (for direct search tools)
    qdrant_host: str = Field(default="qdrant")
    qdrant_port: int = 6333
    qdrant_collection: str = Field(default="products")

    # Elasticsearch (for direct search tools)
    elasticsearch_host: str = Field(default="elasticsearch")
    elasticsearch_port: int = 9200
    elasticsearch_index: str = Field(default="products")

    # Embedding model
    embedding_model: str = Field(default="bge-large:latest")

    # LLM settings
    default_llm_model: str = "llama3.1:8b"
    max_agent_steps: int = 10

    # Search defaults
    search_default_limit: int = 10
    search_fetch_multiplier: int = 5

    # Pipeline mode (original or enrich)
    pipeline_mode: str = Field(default="enrich")

    # Rate limiting
    rate_limit_requests_per_second: float = 10.0
    rate_limit_burst_size: int = 20

    # Cache settings
    cache_search_ttl: int = 300      # 5 minutes
    cache_agent_ttl: int = 600       # 10 minutes
    cache_stats_ttl: int = 900       # 15 minutes

    # Middleware
    enable_rate_limiting: bool = True
    enable_correlation_id: bool = True

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}"

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
