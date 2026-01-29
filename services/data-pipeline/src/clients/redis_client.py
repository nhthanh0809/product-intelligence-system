"""Redis client for job queue and progress tracking."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from src.config import get_settings
from src.models.job import PipelineJob

logger = structlog.get_logger()

# Import redis-py
try:
    import redis.asyncio as redis
except ImportError:
    redis = None
    logger.warning("redis-py not installed, Redis functionality disabled")


class RedisClient:
    """Redis client for job management and caching."""

    # Key prefixes
    JOB_PREFIX = "pipeline:job:"
    PROGRESS_PREFIX = "pipeline:progress:"
    QUEUE_KEY = "pipeline:queue"

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if redis is None:
            logger.warning("Redis client unavailable - redis-py not installed")
            return

        if self._client is not None:
            return

        self._client = redis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("redis_connected", url=self.settings.redis_url)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("redis_disconnected")

    @property
    def client(self):
        """Get Redis client."""
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    def _job_key(self, job_id: str | UUID) -> str:
        """Get Redis key for a job."""
        return f"{self.JOB_PREFIX}{str(job_id)}"

    def _progress_key(self, job_id: str | UUID) -> str:
        """Get Redis key for job progress."""
        return f"{self.PROGRESS_PREFIX}{str(job_id)}"

    async def save_job(self, job: PipelineJob, ttl: int = 86400) -> None:
        """Save job to Redis.

        Args:
            job: Pipeline job to save
            ttl: Time-to-live in seconds (default 24 hours)
        """
        if self._client is None:
            await self.connect()

        if self._client is None:
            logger.warning("redis_unavailable", operation="save_job")
            return

        key = self._job_key(job.job_id)
        data = job.model_dump_json()

        await self._client.set(key, data, ex=ttl)
        logger.debug("job_saved", job_id=str(job.job_id))

    async def get_job(self, job_id: str | UUID) -> PipelineJob | None:
        """Get job from Redis."""
        if self._client is None:
            await self.connect()

        if self._client is None:
            return None

        key = self._job_key(job_id)
        data = await self._client.get(key)

        if not data:
            return None

        return PipelineJob.model_validate_json(data)

    async def delete_job(self, job_id: str | UUID) -> None:
        """Delete job from Redis."""
        if self._client is None:
            await self.connect()

        if self._client is None:
            return

        key = self._job_key(job_id)
        await self._client.delete(key)

    async def update_progress(
        self,
        job_id: str | UUID,
        stage: str,
        processed: int,
        total: int,
        failed: int = 0,
    ) -> None:
        """Update job progress in Redis.

        Args:
            job_id: Job ID
            stage: Current stage name
            processed: Items processed
            total: Total items
            failed: Failed items
        """
        if self._client is None:
            await self.connect()

        if self._client is None:
            return

        key = self._progress_key(job_id)
        progress = {
            "stage": stage,
            "processed": processed,
            "total": total,
            "failed": failed,
            "progress_pct": (processed / total * 100) if total > 0 else 0,
            "updated_at": datetime.now().isoformat(),
        }

        await self._client.hset(key, mapping=progress)
        await self._client.expire(key, 86400)  # 24 hour TTL

    async def get_progress(self, job_id: str | UUID) -> dict[str, Any] | None:
        """Get job progress from Redis."""
        if self._client is None:
            await self.connect()

        if self._client is None:
            return None

        key = self._progress_key(job_id)
        data = await self._client.hgetall(key)

        if not data:
            return None

        # Convert numeric fields
        return {
            "stage": data.get("stage"),
            "processed": int(data.get("processed", 0)),
            "total": int(data.get("total", 0)),
            "failed": int(data.get("failed", 0)),
            "progress_pct": float(data.get("progress_pct", 0)),
            "updated_at": data.get("updated_at"),
        }

    async def enqueue_job(self, job_id: str | UUID) -> None:
        """Add job to queue."""
        if self._client is None:
            await self.connect()

        if self._client is None:
            return

        await self._client.rpush(self.QUEUE_KEY, str(job_id))

    async def dequeue_job(self, timeout: int = 0) -> str | None:
        """Get next job from queue.

        Args:
            timeout: Seconds to wait for job (0 = no wait)

        Returns:
            Job ID or None if queue is empty
        """
        if self._client is None:
            await self.connect()

        if self._client is None:
            return None

        if timeout > 0:
            result = await self._client.blpop(self.QUEUE_KEY, timeout=timeout)
            return result[1] if result else None
        else:
            return await self._client.lpop(self.QUEUE_KEY)

    async def list_jobs(self, limit: int = 100) -> list[str]:
        """List recent job IDs."""
        if self._client is None:
            await self.connect()

        if self._client is None:
            return []

        # Scan for job keys
        job_ids = []
        pattern = f"{self.JOB_PREFIX}*"

        async for key in self._client.scan_iter(match=pattern, count=100):
            job_id = key.replace(self.JOB_PREFIX, "")
            job_ids.append(job_id)
            if len(job_ids) >= limit:
                break

        return job_ids

    async def health_check(self) -> dict[str, Any]:
        """Check Redis health."""
        try:
            if self._client is None:
                await self.connect()

            if self._client is None:
                return {
                    "status": "unavailable",
                    "error": "Redis client not configured",
                }

            # Ping Redis
            await self._client.ping()

            # Get queue length
            queue_length = await self._client.llen(self.QUEUE_KEY)

            return {
                "status": "healthy",
                "queue_length": queue_length,
            }
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Singleton instance
_redis_client: RedisClient | None = None


async def get_redis_client() -> RedisClient:
    """Get or create Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    return _redis_client
