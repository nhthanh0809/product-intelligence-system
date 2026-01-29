"""Pipeline configuration with YAML support and schema validation."""

from pathlib import Path
from typing import Any, Literal

import structlog
import yaml
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()


class StageConfig(BaseModel):
    """Configuration for a pipeline stage."""

    enabled: bool = True
    batch_size: int = Field(default=100, ge=1, le=2000)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    retry_count: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)


class ExtractConfig(StageConfig):
    """Extract stage configuration."""

    file_type: Literal["csv", "parquet", "auto"] = "auto"
    encoding: str = "utf-8"
    chunk_size: int = Field(default=10000, ge=100, le=100000)


class CleanConfig(StageConfig):
    """Clean stage configuration."""

    build_chunks: bool = True
    min_title_length: int = Field(default=5, ge=1, le=100)
    max_title_length: int = Field(default=500, ge=50, le=2000)


class EmbedConfig(StageConfig):
    """Embed stage configuration."""

    model: str = "bge-large"
    batch_size: int = Field(default=50, ge=1, le=500)
    dimensions: int | None = None  # Auto-detected if None


class LlmConfig(StageConfig):
    """LLM extraction stage configuration."""

    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)
    batch_size: int = Field(default=10, ge=1, le=100)


class LoadConfig(StageConfig):
    """Load stage configuration."""

    batch_size: int = Field(default=500, ge=1, le=2000)
    parallel: bool = True


class PostgresLoadConfig(LoadConfig):
    """PostgreSQL load configuration."""

    use_copy: bool = True  # Use COPY protocol for bulk inserts


class QdrantLoadConfig(LoadConfig):
    """Qdrant load configuration."""

    indexing_strategy: Literal["parent_only", "enrich_existing", "add_child_node", "full_replace"] = "parent_only"
    wait_for_index: bool = True


class ElasticsearchLoadConfig(LoadConfig):
    """Elasticsearch load configuration."""

    refresh_interval: str = "-1"  # Disable refresh during bulk load
    refresh_after_load: bool = True


class StagesConfig(BaseModel):
    """Configuration for all pipeline stages."""

    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    llm_extract: LlmConfig = Field(default_factory=LlmConfig)
    load_postgres: PostgresLoadConfig = Field(default_factory=PostgresLoadConfig)
    load_qdrant: QdrantLoadConfig = Field(default_factory=QdrantLoadConfig)
    load_elasticsearch: ElasticsearchLoadConfig = Field(default_factory=ElasticsearchLoadConfig)


class CheckpointConfig(BaseModel):
    """Checkpoint configuration for resumable pipelines."""

    enabled: bool = True
    checkpoint_interval: int = Field(default=1000, ge=100, le=10000, description="Save checkpoint every N products")
    redis_prefix: str = "pipeline:checkpoint:"
    ttl_hours: int = Field(default=24, ge=1, le=168, description="Checkpoint TTL in hours")


class JobQueueConfig(BaseModel):
    """Job queue configuration."""

    enabled: bool = True
    redis_stream: str = "pipeline:jobs"
    consumer_group: str = "pipeline-workers"
    max_concurrent_jobs: int = Field(default=2, ge=1, le=10)
    job_timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    retry_failed_jobs: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    # Pipeline settings
    name: str = "default"
    description: str = ""
    mode: Literal["original", "enrich"] = "original"

    # Input settings
    csv_path: str | None = None
    parquet_path: str | None = None
    product_count: int | None = None
    offset: int = 0

    # Stage configurations
    stages: StagesConfig = Field(default_factory=StagesConfig)

    # Checkpoint and queue settings
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    job_queue: JobQueueConfig = Field(default_factory=JobQueueConfig)

    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dry_run: bool = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate pipeline mode."""
        if v not in ("original", "enrich"):
            raise ValueError(f"Invalid mode: {v}. Must be 'original' or 'enrich'")
        return v


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PipelineConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path) as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    try:
        config = PipelineConfig(**config_dict)
        logger.info(
            "config_loaded",
            path=str(config_path),
            mode=config.mode,
            name=config.name,
        )
        return config
    except Exception as e:
        logger.error("config_validation_failed", path=str(config_path), error=str(e))
        raise ValueError(f"Invalid configuration: {e}")


def save_config(config: PipelineConfig, config_path: str | Path) -> None:
    """Save pipeline configuration to YAML file.

    Args:
        config: PipelineConfig to save
        config_path: Path to save configuration
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump(exclude_none=True)

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("config_saved", path=str(config_path))


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


# Example configuration template
EXAMPLE_CONFIG = """
# Pipeline Configuration
name: product-pipeline
description: Load products from CSV to all stores
mode: original  # or 'enrich' for LLM enrichment

# Input settings
csv_path: /app/data/products.csv
# parquet_path: /app/data/products.parquet  # Alternative to csv_path
product_count: 10000  # null for all products
offset: 0

# Stage configurations
stages:
  extract:
    enabled: true
    batch_size: 100
    file_type: auto  # csv, parquet, or auto
    chunk_size: 10000

  clean:
    enabled: true
    batch_size: 100
    build_chunks: true  # Only for enrich mode

  embed:
    enabled: true
    model: bge-large  # or mxbai-embed-large, bge-large
    batch_size: 50

  llm_extract:  # Only for enrich mode
    enabled: true
    model: llama3.2:3b
    temperature: 0.3
    max_tokens: 1000
    batch_size: 10

  load_postgres:
    enabled: true
    batch_size: 500
    use_copy: true

  load_qdrant:
    enabled: true
    batch_size: 500
    indexing_strategy: parent_only  # or enrich_existing, add_child_node, full_replace
    wait_for_index: true

  load_elasticsearch:
    enabled: true
    batch_size: 500
    refresh_after_load: true

# Checkpoint settings (for resumable pipelines)
checkpoint:
  enabled: true
  checkpoint_interval: 1000
  ttl_hours: 24

# Job queue settings (for background processing)
job_queue:
  enabled: true
  max_concurrent_jobs: 2
  job_timeout_seconds: 3600
  retry_failed_jobs: true
  max_retries: 3

# Global settings
log_level: INFO
dry_run: false
"""


def create_example_config(output_path: str | Path) -> None:
    """Create an example configuration file.

    Args:
        output_path: Path to save example configuration
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(EXAMPLE_CONFIG)

    logger.info("example_config_created", path=str(output_path))
