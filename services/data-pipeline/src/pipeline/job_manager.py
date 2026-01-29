"""Job manager - handles job lifecycle and background execution."""

import asyncio
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

import structlog

from src.config import get_settings
from src.models.job import PipelineJob, ModelConfig
from src.models.enums import PipelineMode, IndexingStrategy, JobStatus
from src.clients.redis_client import RedisClient, get_redis_client
from src.pipeline.runner import PipelineRunner

logger = structlog.get_logger()


class JobManager:
    """Manages pipeline job lifecycle.

    Handles:
    - Job creation and validation
    - Background execution
    - Progress tracking via Redis
    - Job status queries
    """

    def __init__(self, redis_client: RedisClient | None = None):
        self._redis = redis_client
        self._running_jobs: dict[str, asyncio.Task] = {}
        self._jobs: dict[str, PipelineJob] = {}  # In-memory cache

    async def _ensure_redis(self) -> RedisClient:
        """Ensure Redis client is initialized."""
        if self._redis is None:
            self._redis = await get_redis_client()
        return self._redis

    async def create_job(
        self,
        mode: PipelineMode = PipelineMode.ORIGINAL,
        csv_path: str | None = None,
        product_count: int | None = None,
        batch_size: int = 100,
        indexing_strategy: IndexingStrategy | None = None,
        offset: int = 0,
        model_config: ModelConfig | None = None,
    ) -> PipelineJob:
        """Create a new pipeline job.

        Args:
            mode: Pipeline mode
            csv_path: Path to CSV file
            product_count: Number of products to process
            batch_size: Batch size for processing
            indexing_strategy: Qdrant indexing strategy
            offset: Starting row in CSV
            model_config: Optional model configuration for embedding/LLM

        Returns:
            Created PipelineJob
        """
        settings = get_settings()

        # Determine indexing strategy
        if indexing_strategy is None:
            if mode == PipelineMode.ORIGINAL:
                indexing_strategy = IndexingStrategy.PARENT_ONLY
            else:
                indexing_strategy = IndexingStrategy.ADD_CHILD_NODE

        # Create job
        job = PipelineJob(
            mode=mode,
            csv_path=csv_path or settings.csv_path,
            product_count=product_count,
            batch_size=batch_size,
            indexing_strategy=indexing_strategy,
            offset=offset,
            model_config_options=model_config,
        )

        # Initialize stages
        job.initialize_stages()

        # Save to Redis and cache
        redis = await self._ensure_redis()
        await redis.save_job(job)
        self._jobs[str(job.job_id)] = job

        logger.info(
            "job_created",
            job_id=str(job.job_id),
            mode=mode.value,
            product_count=product_count,
        )

        return job

    async def start_job(
        self,
        job_id: str | UUID,
        background: bool = True,
    ) -> PipelineJob:
        """Start a job execution.

        Args:
            job_id: Job ID to start
            background: Run in background if True

        Returns:
            Updated job
        """
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job cannot be started: status={job.status.value}")

        if background:
            # Start in background
            task = asyncio.create_task(self._run_job(job))
            self._running_jobs[str(job.job_id)] = task

            logger.info(
                "job_started_background",
                job_id=str(job.job_id),
            )
        else:
            # Run synchronously
            await self._run_job(job)

        return job

    async def _run_job(self, job: PipelineJob) -> None:
        """Execute a job."""
        redis = await self._ensure_redis()

        def on_progress(stage: str, processed: int, total: int):
            # Update Redis progress (fire-and-forget)
            asyncio.create_task(
                redis.update_progress(job.job_id, stage, processed, total)
            )

        try:
            runner = PipelineRunner(job, on_progress=on_progress)
            await runner.run()
        finally:
            # Save final state
            await redis.save_job(job)
            self._jobs[str(job.job_id)] = job

            # Clean up running task reference
            job_id_str = str(job.job_id)
            if job_id_str in self._running_jobs:
                del self._running_jobs[job_id_str]

    async def get_job(self, job_id: str | UUID) -> PipelineJob | None:
        """Get a job by ID.

        First checks in-memory cache, then Redis.
        """
        job_id_str = str(job_id)

        # Check cache
        if job_id_str in self._jobs:
            return self._jobs[job_id_str]

        # Check Redis
        redis = await self._ensure_redis()
        job = await redis.get_job(job_id)

        if job:
            self._jobs[job_id_str] = job

        return job

    async def get_job_status(self, job_id: str | UUID) -> dict[str, Any] | None:
        """Get job status summary."""
        job = await self.get_job(job_id)
        if not job:
            return None

        # Get real-time progress from Redis
        redis = await self._ensure_redis()
        progress = await redis.get_progress(job_id)

        status = job.to_status_dict()

        # Merge real-time progress if available
        if progress:
            status["real_time_progress"] = progress

        return status

    async def cancel_job(self, job_id: str | UUID) -> bool:
        """Cancel a running job."""
        job_id_str = str(job_id)

        # Cancel task if running
        if job_id_str in self._running_jobs:
            task = self._running_jobs[job_id_str]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            del self._running_jobs[job_id_str]

        # Update job status
        job = await self.get_job(job_id)
        if job:
            job.cancel()
            redis = await self._ensure_redis()
            await redis.save_job(job)
            self._jobs[job_id_str] = job
            return True

        return False

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List recent jobs."""
        redis = await self._ensure_redis()
        job_ids = await redis.list_jobs(limit=limit)

        jobs = []
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if job:
                if status is None or job.status == status:
                    jobs.append(job.to_status_dict())

        return jobs

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs."""
        redis = await self._ensure_redis()
        job_ids = await redis.list_jobs(limit=1000)

        cleaned = 0
        cutoff = datetime.now()

        for job_id in job_ids:
            job = await self.get_job(job_id)
            if not job:
                continue

            # Check if job is old and completed/failed
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    age_hours = (cutoff - job.completed_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        await redis.delete_job(job_id)
                        if job_id in self._jobs:
                            del self._jobs[job_id]
                        cleaned += 1

        logger.info("jobs_cleaned_up", count=cleaned)
        return cleaned


# Singleton instance
_job_manager: JobManager | None = None


async def get_job_manager() -> JobManager:
    """Get or create JobManager singleton."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
