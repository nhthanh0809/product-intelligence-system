"""Configuration for Data Pipeline Service."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Service settings
    service_name: str = "data-pipeline"
    service_port: int = 8005
    log_level: str = "INFO"

    # Pipeline settings
    default_batch_size: int = 100
    max_concurrent_tasks: int = 4
    chunk_size: int = 1000

    # CSV source
    csv_path: str = Field(
        default="/data/amazon_products.csv",
        alias="CSV_PATH",
    )

    # PostgreSQL
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="product_intelligence", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_pool_min: int = 2
    postgres_pool_max: int = 10

    @property
    def postgres_dsn(self) -> str:
        """Get PostgreSQL connection DSN."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Qdrant
    qdrant_host: str = Field(default="qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="products", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")

    # Elasticsearch
    elasticsearch_host: str = Field(default="elasticsearch", alias="ELASTICSEARCH_HOST")
    elasticsearch_port: int = Field(default=9200, alias="ELASTICSEARCH_PORT")
    elasticsearch_index: str = Field(default="products", alias="ELASTICSEARCH_INDEX")

    # Redis (for job queue)
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Ollama (embedding and LLM service)
    ollama_service_url: str = Field(
        default="http://ollama-service:8010",
        alias="OLLAMA_SERVICE_URL",
    )
    embedding_model: str = Field(
        default="bge-large",
        alias="EMBEDDING_MODEL",
        description="Model for generating embeddings (1024 dimensions)",
    )
    embedding_batch_size: int = Field(
        default=50,
        alias="EMBEDDING_BATCH_SIZE",
    )
    llm_model: str = Field(
        default="llama3.2:3b",
        alias="LLM_MODEL",
        description="Model for LLM extraction in enrich mode",
    )
    llm_temperature: float = Field(
        default=0.3,
        alias="LLM_TEMPERATURE",
        description="Temperature for LLM generation (lower = more consistent)",
    )
    llm_max_tokens: int = Field(
        default=1000,
        alias="LLM_MAX_TOKENS",
        description="Maximum tokens for LLM response",
    )

    # Timeouts
    connect_timeout: float = 10.0
    read_timeout: float = 120.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Data directory paths (auto-detected or from env)
    data_dir_path: str = Field(
        default="",
        alias="DATA_DIR",
        description="Base data directory (auto-detected if empty)",
    )

    @property
    def data_dir(self) -> Path:
        """Get base data directory."""
        if self.data_dir_path:
            return Path(self.data_dir_path)
        # Auto-detect: look for common locations
        possible_dirs = [
            Path(__file__).parent.parent.parent.parent.parent / "data",  # Product_intelligence_system/data
            Path.cwd() / "data",
            Path("/data"),
        ]
        for d in possible_dirs:
            if d.exists():
                return d
        return possible_dirs[0]  # Default to first option

    @property
    def source_csv_path(self) -> Path:
        """Get source CSV file path."""
        return self.data_dir / "archive" / "amz_ca_total_products_data_processed.csv"

    @property
    def raw_data_dir(self) -> Path:
        """Get raw data directory."""
        return self.data_dir / "raw"

    @property
    def scraped_data_dir(self) -> Path:
        """Get scraped data directory."""
        return self.data_dir / "scraped"

    @property
    def cleaned_data_dir(self) -> Path:
        """Get cleaned data directory."""
        return self.data_dir / "cleaned"

    @property
    def embedded_data_dir(self) -> Path:
        """Get embedded data directory."""
        return self.data_dir / "embedded"

    @property
    def metrics_dir(self) -> Path:
        """Get metrics directory."""
        return self.data_dir / "metrics"

    class Config:
        env_prefix = ""
        case_sensitive = False
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in .env file


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
