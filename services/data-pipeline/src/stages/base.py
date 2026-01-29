"""Base class for pipeline stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Generic, TypeVar

import structlog

from src.models.job import PipelineJob, StageProgress

logger = structlog.get_logger()

# Type variables for input/output
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass
class StageContext:
    """Context passed between stages."""

    job: PipelineJob
    batch_size: int = 100

    # Shared data between stages
    data: dict[str, Any] = field(default_factory=dict)

    # Callbacks
    on_progress: Any = None  # Callable[[str, int, int], None]

    def update_progress(self, stage: str, processed: int, total: int) -> None:
        """Update progress for a stage."""
        if self.on_progress:
            self.on_progress(stage, processed, total)


class BaseStage(ABC, Generic[TIn, TOut]):
    """Base class for all pipeline stages.

    Each stage processes items in batches, yielding results
    that can be passed to the next stage.
    """

    # Stage metadata
    name: str = "base"
    description: str = "Base stage"

    def __init__(self, context: StageContext):
        self.context = context
        self.progress = context.job.get_stage(self.name)
        self._started = False
        self._completed = False

    @property
    def job(self) -> PipelineJob:
        """Get current job."""
        return self.context.job

    async def run(self, items: list[TIn]) -> list[TOut]:
        """Run the stage on a list of items.

        This is the main entry point for processing a batch.

        Args:
            items: Input items to process

        Returns:
            Processed output items
        """
        if not items:
            return []

        self._start(len(items))

        try:
            results = []
            processed = 0
            failed = 0

            async for batch_results in self.process_batches(items):
                results.extend(batch_results)
                processed += len(batch_results)
                self.progress.update(processed, failed)
                self.context.update_progress(self.name, processed, len(items))

            self._complete()
            return results

        except Exception as e:
            self._fail(str(e))
            raise

    async def process_batches(
        self,
        items: list[TIn],
    ) -> AsyncIterator[list[TOut]]:
        """Process items in batches.

        Override this method for custom batch processing logic.

        Args:
            items: All input items

        Yields:
            Batches of processed items
        """
        batch_size = self.context.batch_size

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            processed_batch = await self.process_batch(batch)
            yield processed_batch

    @abstractmethod
    async def process_batch(self, batch: list[TIn]) -> list[TOut]:
        """Process a single batch of items.

        This method must be implemented by each stage.

        Args:
            batch: Batch of input items

        Returns:
            Batch of processed output items
        """
        pass

    async def process_item(self, item: TIn) -> TOut | None:
        """Process a single item.

        Override this for simple item-by-item processing.
        The default implementation raises NotImplementedError.

        Args:
            item: Single input item

        Returns:
            Processed item or None if skipped
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process_batch or process_item"
        )

    def _start(self, total: int) -> None:
        """Mark stage as started."""
        if self._started:
            return

        self._started = True
        self.progress.start(total)

        logger.info(
            "stage_started",
            stage=self.name,
            total=total,
            job_id=str(self.job.job_id),
        )

    def _complete(self) -> None:
        """Mark stage as completed."""
        if self._completed:
            return

        self._completed = True
        self.progress.complete()

        duration = self.progress.duration_seconds or 0
        rate = self.progress.items_per_second or 0

        logger.info(
            "stage_completed",
            stage=self.name,
            processed=self.progress.processed,
            failed=self.progress.failed,
            duration_seconds=round(duration, 2),
            items_per_second=round(rate, 2),
            job_id=str(self.job.job_id),
        )

    def _fail(self, error: str) -> None:
        """Mark stage as failed."""
        self.progress.fail(error)
        self.job.add_error(self.name, error)

        logger.error(
            "stage_failed",
            stage=self.name,
            error=error,
            processed=self.progress.processed,
            job_id=str(self.job.job_id),
        )

    def _log_progress(self, processed: int, total: int) -> None:
        """Log progress periodically."""
        if processed % 1000 == 0 or processed == total:
            pct = (processed / total * 100) if total > 0 else 0
            logger.info(
                "stage_progress",
                stage=self.name,
                processed=processed,
                total=total,
                progress=f"{pct:.1f}%",
                job_id=str(self.job.job_id),
            )
