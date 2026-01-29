"""API request/response schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.enums import PipelineMode, IndexingStrategy, JobStatus


class ModelConfig(BaseModel):
    """Model configuration for embedding and LLM."""

    embedding_model: str | None = Field(
        default=None,
        description="Embedding model (e.g., 'bge-large', 'bge-large', 'mxbai-embed-large'). Uses default if not specified.",
    )
    llm_model: str | None = Field(
        default=None,
        description="LLM model for enrich mode (e.g., 'llama3.2:3b', 'qwen2.5:7b', 'mistral:7b'). Uses default if not specified.",
    )
    llm_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0-1.0). Lower = more consistent output.",
    )
    llm_max_tokens: int | None = Field(
        default=None,
        ge=100,
        le=4000,
        description="Maximum tokens for LLM response.",
    )


class PipelineRunRequest(BaseModel):
    """Request to run a pipeline."""

    mode: PipelineMode = Field(
        default=PipelineMode.ORIGINAL,
        description="Pipeline mode: 'original' or 'enrich'",
    )
    csv_path: str | None = Field(
        default=None,
        description="Path to CSV file (uses default if not specified)",
    )
    product_count: int | None = Field(
        default=None,
        ge=1,
        description="Number of products to process (all if not specified)",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for processing",
    )
    indexing_strategy: IndexingStrategy | None = Field(
        default=None,
        description="Qdrant indexing strategy: 'parent_only', 'enrich_existing', 'add_child_node', 'full_replace' (auto-selected if not specified)",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Starting row in CSV (0-indexed)",
    )
    background: bool = Field(
        default=True,
        description="Run in background (returns immediately)",
    )
    model_config_options: ModelConfig | None = Field(
        default=None,
        description="Optional model configuration for embedding and LLM",
        alias="model_config",
    )

    class Config:
        populate_by_name = True


class PipelineRunResponse(BaseModel):
    """Response from pipeline run request."""

    job_id: str
    status: str
    message: str
    mode: str
    indexing_strategy: str
    product_count: int | None
    created_at: str


class StageStatus(BaseModel):
    """Status of a single pipeline stage."""

    status: str
    progress: float
    processed: int
    total: int
    failed: int
    error: str | None = None


class JobStatusResponse(BaseModel):
    """Response with job status details."""

    job_id: str
    status: str
    mode: str
    indexing_strategy: str
    progress: float
    current_stage: str | None
    duration_seconds: float | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    stages: dict[str, StageStatus]
    metrics: dict[str, Any]
    error_message: str | None
    error_count: int


class ServiceHealth(BaseModel):
    """Health status of a service/dependency."""

    status: str
    details: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    timestamp: str
    dependencies: dict[str, ServiceHealth]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    job_id: str | None = None
