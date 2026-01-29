"""Checkpoint system for resumable pipelines."""

import json
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
import redis.asyncio as redis

from src.config import get_settings
from src.models.enums import JobStatus

logger = structlog.get_logger()


class CheckpointData:
    """Data stored in a checkpoint."""

    def __init__(
        self,
        job_id: str,
        stage: str,
        processed_count: int,
        total_count: int,
        last_processed_id: str | None = None,
        stage_data: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ):
        self.job_id = job_id
        self.stage = stage
        self.processed_count = processed_count
        self.total_count = total_count
        self.last_processed_id = last_processed_id
        self.stage_data = stage_data or {}
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "last_processed_id": self.last_processed_id,
            "stage_data": self.stage_data,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            job_id=data["job_id"],
            stage=data["stage"],
            processed_count=data["processed_count"],
            total_count=data["total_count"],
            last_processed_id=data.get("last_processed_id"),
            stage_data=data.get("stage_data", {}),
            created_at=created_at,
        )


class CheckpointManager:
    """Manages pipeline checkpoints for resume capability.

    Features:
    - Save checkpoint at configurable intervals
    - Resume from last checkpoint on failure
    - Automatic checkpoint cleanup
    - Stage-level granularity
    """

    def __init__(
        self,
        redis_url: str | None = None,
        prefix: str = "pipeline:checkpoint:",
        ttl_hours: int = 24,
        checkpoint_interval: int = 1000,
    ):
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._prefix = prefix
        self._ttl_seconds = ttl_hours * 3600
        self._checkpoint_interval = checkpoint_interval
        self._redis: redis.Redis | None = None
        self._counters: dict[str, int] = {}

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is None:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _get_key(self, job_id: str, stage: str | None = None) -> str:
        """Get Redis key for checkpoint."""
        if stage:
            return f"{self._prefix}{job_id}:{stage}"
        return f"{self._prefix}{job_id}"

    async def save_checkpoint(
        self,
        job_id: str | UUID,
        stage: str,
        processed_count: int,
        total_count: int,
        last_processed_id: str | None = None,
        stage_data: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Save a checkpoint.

        Args:
            job_id: Job identifier
            stage: Current stage name
            processed_count: Number of items processed
            total_count: Total items to process
            last_processed_id: ID of last processed item (for resume)
            stage_data: Additional stage-specific data
            force: Save even if interval not reached

        Returns:
            True if checkpoint was saved
        """
        job_id_str = str(job_id)

        # Check if we should save (based on interval)
        counter_key = f"{job_id_str}:{stage}"
        if counter_key not in self._counters:
            self._counters[counter_key] = 0

        self._counters[counter_key] += 1

        if not force and self._counters[counter_key] % self._checkpoint_interval != 0:
            return False

        if self._redis is None:
            await self.connect()

        checkpoint = CheckpointData(
            job_id=job_id_str,
            stage=stage,
            processed_count=processed_count,
            total_count=total_count,
            last_processed_id=last_processed_id,
            stage_data=stage_data,
        )

        key = self._get_key(job_id_str, stage)
        await self._redis.setex(
            key,
            self._ttl_seconds,
            json.dumps(checkpoint.to_dict()),
        )

        # Also save job-level checkpoint (latest stage)
        job_key = self._get_key(job_id_str)
        await self._redis.setex(
            job_key,
            self._ttl_seconds,
            json.dumps({
                "job_id": job_id_str,
                "current_stage": stage,
                "updated_at": datetime.now().isoformat(),
            }),
        )

        logger.debug(
            "checkpoint_saved",
            job_id=job_id_str,
            stage=stage,
            processed=processed_count,
            total=total_count,
        )

        return True

    async def get_checkpoint(
        self,
        job_id: str | UUID,
        stage: str,
    ) -> CheckpointData | None:
        """Get checkpoint for a specific stage.

        Args:
            job_id: Job identifier
            stage: Stage name

        Returns:
            CheckpointData or None if not found
        """
        if self._redis is None:
            await self.connect()

        key = self._get_key(str(job_id), stage)
        data = await self._redis.get(key)

        if not data:
            return None

        try:
            return CheckpointData.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("checkpoint_parse_error", key=key, error=str(e))
            return None

    async def get_job_checkpoint(
        self,
        job_id: str | UUID,
    ) -> dict[str, Any] | None:
        """Get job-level checkpoint info.

        Args:
            job_id: Job identifier

        Returns:
            Job checkpoint info or None
        """
        if self._redis is None:
            await self.connect()

        key = self._get_key(str(job_id))
        data = await self._redis.get(key)

        if not data:
            return None

        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    async def get_all_stage_checkpoints(
        self,
        job_id: str | UUID,
    ) -> dict[str, CheckpointData]:
        """Get all stage checkpoints for a job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary of stage name to CheckpointData
        """
        if self._redis is None:
            await self.connect()

        job_id_str = str(job_id)
        pattern = f"{self._prefix}{job_id_str}:*"

        checkpoints = {}
        async for key in self._redis.scan_iter(match=pattern):
            data = await self._redis.get(key)
            if data:
                try:
                    checkpoint = CheckpointData.from_dict(json.loads(data))
                    checkpoints[checkpoint.stage] = checkpoint
                except (json.JSONDecodeError, KeyError):
                    continue

        return checkpoints

    async def delete_checkpoint(
        self,
        job_id: str | UUID,
        stage: str | None = None,
    ) -> int:
        """Delete checkpoint(s).

        Args:
            job_id: Job identifier
            stage: Specific stage to delete, or None for all

        Returns:
            Number of checkpoints deleted
        """
        if self._redis is None:
            await self.connect()

        job_id_str = str(job_id)

        if stage:
            # Delete specific stage
            key = self._get_key(job_id_str, stage)
            return await self._redis.delete(key)
        else:
            # Delete all checkpoints for job
            pattern = f"{self._prefix}{job_id_str}*"
            deleted = 0
            async for key in self._redis.scan_iter(match=pattern):
                deleted += await self._redis.delete(key)

            # Clear counters
            keys_to_remove = [k for k in self._counters if k.startswith(job_id_str)]
            for k in keys_to_remove:
                del self._counters[k]

            return deleted

    async def cleanup_old_checkpoints(
        self,
        max_age_hours: int | None = None,
    ) -> int:
        """Clean up old checkpoints.

        Args:
            max_age_hours: Maximum age in hours (uses TTL if not specified)

        Returns:
            Number of checkpoints cleaned up
        """
        if self._redis is None:
            await self.connect()

        # Redis TTL handles cleanup automatically
        # This method is for manual cleanup if needed
        max_age = max_age_hours or (self._ttl_seconds / 3600)
        cutoff = datetime.now() - timedelta(hours=max_age)

        cleaned = 0
        pattern = f"{self._prefix}*"

        async for key in self._redis.scan_iter(match=pattern):
            data = await self._redis.get(key)
            if data:
                try:
                    checkpoint_data = json.loads(data)
                    created_at = checkpoint_data.get("created_at") or checkpoint_data.get("updated_at")
                    if created_at:
                        created = datetime.fromisoformat(created_at)
                        if created < cutoff:
                            await self._redis.delete(key)
                            cleaned += 1
                except (json.JSONDecodeError, ValueError):
                    # Invalid data, delete it
                    await self._redis.delete(key)
                    cleaned += 1

        logger.info("checkpoints_cleaned", count=cleaned)
        return cleaned

    async def can_resume(self, job_id: str | UUID) -> bool:
        """Check if a job can be resumed from checkpoint.

        Args:
            job_id: Job identifier

        Returns:
            True if checkpoint exists and is valid
        """
        checkpoint = await self.get_job_checkpoint(job_id)
        return checkpoint is not None

    async def get_resume_info(
        self,
        job_id: str | UUID,
    ) -> dict[str, Any] | None:
        """Get information needed to resume a job.

        Args:
            job_id: Job identifier

        Returns:
            Resume information or None if cannot resume
        """
        job_checkpoint = await self.get_job_checkpoint(job_id)
        if not job_checkpoint:
            return None

        stage_checkpoints = await self.get_all_stage_checkpoints(job_id)

        return {
            "job_id": str(job_id),
            "current_stage": job_checkpoint.get("current_stage"),
            "updated_at": job_checkpoint.get("updated_at"),
            "stages": {
                stage: {
                    "processed": cp.processed_count,
                    "total": cp.total_count,
                    "last_id": cp.last_processed_id,
                    "data": cp.stage_data,
                }
                for stage, cp in stage_checkpoints.items()
            },
        }


# Singleton instance
_checkpoint_manager: CheckpointManager | None = None


async def get_checkpoint_manager() -> CheckpointManager:
    """Get or create CheckpointManager singleton."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
        await _checkpoint_manager.connect()
    return _checkpoint_manager
