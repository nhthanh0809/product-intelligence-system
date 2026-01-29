"""Job and progress tracking models for the pipeline."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.enums import (
    PipelineMode,
    IndexingStrategy,
    JobStatus,
    StageStatus,
)


class StageProgress(BaseModel):
    """Progress tracking for a single pipeline stage."""

    name: str = Field(..., description="Stage name (e.g., 'extract', 'clean')")
    status: StageStatus = Field(default=StageStatus.PENDING)

    # Progress counters
    total: int = Field(default=0, description="Total items to process")
    processed: int = Field(default=0, description="Items processed so far")
    failed: int = Field(default=0, description="Items that failed processing")

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Error info
    error_message: str | None = None
    last_error: str | None = None

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.total == 0:
            return 0.0
        return (self.processed / self.total) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed == 0:
            return 0.0
        return ((self.processed - self.failed) / self.processed) * 100

    @property
    def duration_seconds(self) -> float | None:
        """Calculate stage duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def items_per_second(self) -> float | None:
        """Calculate processing rate."""
        duration = self.duration_seconds
        if not duration or duration == 0:
            return None
        return self.processed / duration

    def start(self, total: int = 0) -> None:
        """Mark stage as started."""
        self.status = StageStatus.RUNNING
        self.started_at = datetime.now()
        self.total = total

    def update(self, processed: int, failed: int = 0) -> None:
        """Update progress counters."""
        self.processed = processed
        self.failed = failed

    def increment(self, count: int = 1, failed: int = 0) -> None:
        """Increment progress counters."""
        self.processed += count
        self.failed += failed

    def complete(self) -> None:
        """Mark stage as completed."""
        self.status = StageStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark stage as failed."""
        self.status = StageStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error
        self.last_error = error

    def skip(self, reason: str = "Not needed for this mode") -> None:
        """Mark stage as skipped."""
        self.status = StageStatus.SKIPPED
        self.error_message = reason


class JobMetrics(BaseModel):
    """Aggregated metrics for a pipeline job."""

    total_products: int = 0
    products_extracted: int = 0
    products_cleaned: int = 0
    products_embedded: int = 0
    products_loaded_postgres: int = 0
    products_loaded_qdrant: int = 0
    products_loaded_elasticsearch: int = 0

    # Chunk metrics (for enrich mode)
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_loaded: int = 0

    # Error counts
    errors_extract: int = 0
    errors_clean: int = 0
    errors_embed: int = 0
    errors_load: int = 0

    # Performance
    avg_embed_time_ms: float = 0.0
    avg_load_time_ms: float = 0.0


class ModelConfig(BaseModel):
    """Model configuration for embedding and LLM."""

    embedding_model: str | None = Field(
        default=None,
        description="Embedding model (uses default if not specified)",
    )
    llm_model: str | None = Field(
        default=None,
        description="LLM model for enrich mode (uses default if not specified)",
    )
    llm_temperature: float | None = Field(
        default=None,
        description="LLM temperature (0.0-1.0)",
    )
    llm_max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for LLM response",
    )


class PipelineJob(BaseModel):
    """Pipeline job tracking."""

    # Identity
    job_id: UUID = Field(default_factory=uuid4)
    status: JobStatus = Field(default=JobStatus.PENDING)

    # Configuration
    mode: PipelineMode = Field(default=PipelineMode.ORIGINAL)
    indexing_strategy: IndexingStrategy = Field(default=IndexingStrategy.PARENT_ONLY)
    product_count: int | None = Field(
        default=None,
        description="Number of products to process (None = all)",
    )
    batch_size: int = Field(default=100)

    # Model configuration
    model_config_options: ModelConfig | None = Field(
        default=None,
        description="Optional model configuration for embedding and LLM",
    )

    # Source
    csv_path: str | None = None
    offset: int = Field(default=0, description="Starting row in CSV")

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Stage progress
    stages: dict[str, StageProgress] = Field(default_factory=dict)

    # Metrics
    metrics: JobMetrics = Field(default_factory=JobMetrics)

    # Error tracking
    error_message: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def current_stage(self) -> str | None:
        """Get the currently running stage."""
        for name, stage in self.stages.items():
            if stage.status == StageStatus.RUNNING:
                return name
        return None

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress across all stages."""
        if not self.stages:
            return 0.0

        total_weight = 0
        weighted_progress = 0.0

        # Stage weights (some stages are heavier than others)
        weights = {
            "extract": 1.0,
            "clean": 1.0,
            "embed": 2.0,  # Embedding is slower
            "load_postgres": 1.0,
            "load_qdrant": 1.5,
            "load_elasticsearch": 1.0,
            # Enrich-mode stages
            "download": 2.0,
            "html_to_md": 1.0,
            "llm_extract": 3.0,  # LLM extraction is slowest
        }

        for name, stage in self.stages.items():
            weight = weights.get(name, 1.0)
            total_weight += weight

            if stage.status == StageStatus.COMPLETED:
                weighted_progress += weight * 100
            elif stage.status == StageStatus.RUNNING:
                weighted_progress += weight * stage.progress_pct
            elif stage.status == StageStatus.SKIPPED:
                weighted_progress += weight * 100  # Count skipped as done

        if total_weight == 0:
            return 0.0
        return weighted_progress / total_weight

    def initialize_stages(self) -> None:
        """Initialize stages based on pipeline mode."""
        # Common stages for both modes
        common_stages = ["extract", "clean", "embed", "load_postgres", "load_qdrant", "load_elasticsearch"]

        for name in common_stages:
            self.stages[name] = StageProgress(name=name)

        # Enrich mode has additional stages
        if self.mode == PipelineMode.ENRICH:
            enrich_stages = ["download", "html_to_md", "llm_extract"]
            for name in enrich_stages:
                self.stages[name] = StageProgress(name=name)

    def start(self) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error

    def cancel(self) -> None:
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now()

    def pause(self) -> None:
        """Mark job as paused."""
        self.status = JobStatus.PAUSED

    def add_error(self, stage: str, error: str, item_id: str | None = None) -> None:
        """Add an error to the error list."""
        self.errors.append({
            "stage": stage,
            "error": error,
            "item_id": item_id,
            "timestamp": datetime.now().isoformat(),
        })

    def get_stage(self, name: str) -> StageProgress:
        """Get or create a stage progress tracker."""
        if name not in self.stages:
            self.stages[name] = StageProgress(name=name)
        return self.stages[name]

    def to_status_dict(self) -> dict[str, Any]:
        """Convert to status response dictionary."""
        return {
            "job_id": str(self.job_id),
            "status": self.status.value,
            "mode": self.mode.value,
            "indexing_strategy": self.indexing_strategy.value,
            "progress": self.overall_progress,
            "current_stage": self.current_stage,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "stages": {
                name: {
                    "status": stage.status.value,
                    "progress": stage.progress_pct,
                    "processed": stage.processed,
                    "total": stage.total,
                    "failed": stage.failed,
                    "error": stage.error_message,
                }
                for name, stage in self.stages.items()
            },
            "metrics": self.metrics.model_dump(),
            "error_message": self.error_message,
            "error_count": len(self.errors),
        }
