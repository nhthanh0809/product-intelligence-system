"""Redis Streams job queue for background pipeline processing."""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

import structlog
import redis.asyncio as redis

from src.config import get_settings
from src.models.enums import JobStatus, PipelineMode, IndexingStrategy
from src.models.job import PipelineJob

logger = structlog.get_logger()


class JobQueue:
    """Redis Streams-based job queue for pipeline processing.

    Features:
    - Reliable message delivery with consumer groups
    - Automatic retry for failed jobs
    - Job timeout handling
    - Concurrent job processing
    """

    def __init__(
        self,
        redis_url: str | None = None,
        stream_name: str = "pipeline:jobs",
        consumer_group: str = "pipeline-workers",
        consumer_name: str | None = None,
    ):
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name or f"worker-{id(self)}"
        self._redis: redis.Redis | None = None
        self._running = False
        self._current_tasks: dict[str, asyncio.Task] = {}

    async def connect(self) -> None:
        """Connect to Redis and create consumer group."""
        if self._redis is None:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)

        # Create consumer group if it doesn't exist
        try:
            await self._redis.xgroup_create(
                self._stream_name,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                "consumer_group_created",
                stream=self._stream_name,
                group=self._consumer_group,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            # Group already exists, that's fine

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False

        # Cancel running tasks
        for task in self._current_tasks.values():
            task.cancel()

        if self._redis:
            await self._redis.close()
            self._redis = None

    async def enqueue(
        self,
        job: PipelineJob,
        priority: int = 0,
    ) -> str:
        """Add a job to the queue.

        Args:
            job: PipelineJob to enqueue
            priority: Job priority (higher = more important)

        Returns:
            Stream message ID
        """
        if self._redis is None:
            await self.connect()

        job_data = {
            "job_id": str(job.job_id),
            "mode": job.mode.value,
            "csv_path": job.csv_path or "",
            "product_count": str(job.product_count or ""),
            "batch_size": str(job.batch_size),
            "indexing_strategy": job.indexing_strategy.value,
            "offset": str(job.offset),
            "priority": str(priority),
            "created_at": job.created_at.isoformat(),
            "status": JobStatus.PENDING.value,
        }

        # Add model config if present
        if job.model_config_options:
            job_data["model_config"] = json.dumps(job.model_config_options.model_dump())

        message_id = await self._redis.xadd(self._stream_name, job_data)

        logger.info(
            "job_enqueued",
            job_id=str(job.job_id),
            message_id=message_id,
            priority=priority,
        )

        return message_id

    async def dequeue(
        self,
        timeout_ms: int = 5000,
    ) -> tuple[str, dict[str, Any]] | None:
        """Get next job from queue.

        Args:
            timeout_ms: Timeout in milliseconds to wait for job

        Returns:
            Tuple of (message_id, job_data) or None if no job available
        """
        if self._redis is None:
            await self.connect()

        # Read from stream using consumer group
        messages = await self._redis.xreadgroup(
            groupname=self._consumer_group,
            consumername=self._consumer_name,
            streams={self._stream_name: ">"},
            count=1,
            block=timeout_ms,
        )

        if not messages:
            return None

        # Extract message
        stream_messages = messages[0][1]
        if not stream_messages:
            return None

        message_id, job_data = stream_messages[0]
        return message_id, job_data

    async def ack(self, message_id: str) -> None:
        """Acknowledge successful job processing.

        Args:
            message_id: Stream message ID to acknowledge
        """
        if self._redis is None:
            await self.connect()

        await self._redis.xack(self._stream_name, self._consumer_group, message_id)
        logger.debug("job_acknowledged", message_id=message_id)

    async def reject(
        self,
        message_id: str,
        job_id: str,
        error: str,
        retry: bool = True,
    ) -> None:
        """Reject a failed job.

        Args:
            message_id: Stream message ID
            job_id: Job ID
            error: Error message
            retry: Whether to retry the job
        """
        if self._redis is None:
            await self.connect()

        # Acknowledge to remove from pending
        await self._redis.xack(self._stream_name, self._consumer_group, message_id)

        if retry:
            # Re-add to stream for retry
            retry_count = await self._get_retry_count(job_id)
            if retry_count < 3:
                await self._redis.xadd(
                    self._stream_name,
                    {
                        "job_id": job_id,
                        "retry_count": str(retry_count + 1),
                        "last_error": error,
                        "status": JobStatus.PENDING.value,
                    },
                )
                logger.info("job_requeued", job_id=job_id, retry_count=retry_count + 1)
            else:
                logger.error("job_max_retries_exceeded", job_id=job_id, error=error)
        else:
            logger.error("job_rejected", job_id=job_id, error=error)

    async def _get_retry_count(self, job_id: str) -> int:
        """Get retry count for a job."""
        key = f"pipeline:retry:{job_id}"
        count = await self._redis.get(key)
        if count:
            await self._redis.incr(key)
            return int(count)
        else:
            await self._redis.setex(key, 3600, "1")
            return 0

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Queue statistics including length, pending, etc.
        """
        if self._redis is None:
            await self.connect()

        # Get stream info
        stream_info = await self._redis.xinfo_stream(self._stream_name)

        # Get consumer group info
        try:
            groups_info = await self._redis.xinfo_groups(self._stream_name)
            group_info = next(
                (g for g in groups_info if g["name"] == self._consumer_group),
                None,
            )
        except redis.ResponseError:
            group_info = None

        return {
            "stream": self._stream_name,
            "length": stream_info.get("length", 0),
            "first_entry": stream_info.get("first-entry"),
            "last_entry": stream_info.get("last-entry"),
            "consumer_group": {
                "name": self._consumer_group,
                "pending": group_info.get("pending", 0) if group_info else 0,
                "consumers": group_info.get("consumers", 0) if group_info else 0,
            } if group_info else None,
        }

    async def get_pending_jobs(self, count: int = 100) -> list[dict[str, Any]]:
        """Get list of pending jobs.

        Args:
            count: Maximum number of pending jobs to return

        Returns:
            List of pending job info
        """
        if self._redis is None:
            await self.connect()

        try:
            pending = await self._redis.xpending_range(
                self._stream_name,
                self._consumer_group,
                min="-",
                max="+",
                count=count,
            )

            return [
                {
                    "message_id": p["message_id"],
                    "consumer": p["consumer"],
                    "time_since_delivered": p["time_since_delivered"],
                    "times_delivered": p["times_delivered"],
                }
                for p in pending
            ]
        except redis.ResponseError:
            return []

    async def claim_stale_jobs(
        self,
        min_idle_time_ms: int = 60000,
        count: int = 10,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Claim stale jobs from other consumers.

        Args:
            min_idle_time_ms: Minimum idle time to consider a job stale
            count: Maximum number of jobs to claim

        Returns:
            List of claimed (message_id, job_data) tuples
        """
        if self._redis is None:
            await self.connect()

        try:
            # Get pending entries
            pending = await self._redis.xpending_range(
                self._stream_name,
                self._consumer_group,
                min="-",
                max="+",
                count=count,
            )

            claimed = []
            for entry in pending:
                if entry["time_since_delivered"] >= min_idle_time_ms:
                    # Claim the message
                    messages = await self._redis.xclaim(
                        self._stream_name,
                        self._consumer_group,
                        self._consumer_name,
                        min_idle_time=min_idle_time_ms,
                        message_ids=[entry["message_id"]],
                    )
                    if messages:
                        for msg_id, msg_data in messages:
                            claimed.append((msg_id, msg_data))
                            logger.info(
                                "stale_job_claimed",
                                message_id=msg_id,
                                idle_time_ms=entry["time_since_delivered"],
                            )

            return claimed
        except redis.ResponseError as e:
            logger.error("claim_stale_jobs_failed", error=str(e))
            return []


class JobWorker:
    """Worker that processes jobs from the queue."""

    def __init__(
        self,
        queue: JobQueue,
        job_handler: Callable[[dict[str, Any]], Any],
        max_concurrent: int = 2,
    ):
        self._queue = queue
        self._job_handler = job_handler
        self._max_concurrent = max_concurrent
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        await self._queue.connect()

        logger.info(
            "job_worker_started",
            max_concurrent=self._max_concurrent,
        )

        while self._running:
            try:
                # Wait for semaphore slot
                async with self._semaphore:
                    # Get next job
                    result = await self._queue.dequeue(timeout_ms=5000)
                    if result:
                        message_id, job_data = result
                        # Process job in background
                        task = asyncio.create_task(
                            self._process_job(message_id, job_data)
                        )
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("worker_error", error=str(e))
                await asyncio.sleep(1)

        # Wait for remaining tasks
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        logger.info("job_worker_stopping")

    async def _process_job(
        self,
        message_id: str,
        job_data: dict[str, Any],
    ) -> None:
        """Process a single job."""
        job_id = job_data.get("job_id", "unknown")

        try:
            logger.info("processing_job", job_id=job_id, message_id=message_id)

            # Execute job handler
            await self._job_handler(job_data)

            # Acknowledge success
            await self._queue.ack(message_id)
            logger.info("job_completed", job_id=job_id)

        except Exception as e:
            logger.error("job_failed", job_id=job_id, error=str(e))
            await self._queue.reject(
                message_id=message_id,
                job_id=job_id,
                error=str(e),
                retry=True,
            )


# Singleton instances
_job_queue: JobQueue | None = None


async def get_job_queue() -> JobQueue:
    """Get or create JobQueue singleton."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
        await _job_queue.connect()
    return _job_queue
