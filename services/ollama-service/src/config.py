"""Configuration for Ollama Service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Service settings
    service_name: str = "ollama-service"
    service_port: int = 8010
    log_level: str = "INFO"

    # Ollama connection
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")

    # Default models
    default_embedding_model: str = "bge-large"
    default_llm_model: str = "llama3.1:8b"

    # Embedding settings
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 100
    embedding_max_length: int = 8192

    # Generation settings
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_num_predict: int = 2048

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Timeouts
    connect_timeout: float = 10.0
    read_timeout: float = 120.0

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
