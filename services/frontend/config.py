"""Configuration for Frontend Service."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Service URLs
    multi_agent_url: str = os.getenv(
        "MULTI_AGENT_URL",
        "http://multi-agents:8001"
    )
    data_pipeline_url: str = os.getenv(
        "DATA_PIPELINE_URL",
        "http://data-pipeline:8005"
    )

    # App settings
    app_title: str = "Product Intelligence System"
    page_icon: str = "🛒"

    # Timeouts (increased for compound queries with multiple LLM calls)
    request_timeout: float = 300.0  # 5 minutes for complex queries

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
