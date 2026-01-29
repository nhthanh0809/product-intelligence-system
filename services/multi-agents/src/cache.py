"""Redis caching for multi-agent system."""

import hashlib
import json
from datetime import datetime
from typing import Any

import structlog

from src.config import get_settings

logger = structlog.get_logger()

# Try to import redis
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("redis-py not installed, caching disabled")


class QueryCache:
    """Redis-based query cache for agent responses.

    Caches:
    - Search results
    - Agent responses
    - Analysis results

    Features:
    - TTL-based expiration
    - Query normalization for better cache hits
    - Cache key prefixing
    """

    # Key prefixes
    SEARCH_PREFIX = "cache:search:"
    AGENT_PREFIX = "cache:agent:"
    ANALYSIS_PREFIX = "cache:analysis:"

    # Default TTLs (seconds)
    SEARCH_TTL = 300      # 5 minutes
    AGENT_TTL = 600       # 10 minutes
    ANALYSIS_TTL = 900    # 15 minutes

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._enabled = HAS_REDIS

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if not self._enabled:
            return

        if self._client is not None:
            return

        try:
            url = f"redis://{self.settings.redis_host}:{self.settings.redis_port}/0"
            self._client = redis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            logger.info("query_cache_connected")
        except Exception as e:
            logger.warning("query_cache_connect_failed", error=str(e))
            self._enabled = False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    def _make_key(self, prefix: str, query: str, **kwargs) -> str:
        """Generate cache key from query and params."""
        # Normalize query
        normalized = query.lower().strip()

        # Include relevant kwargs in key
        key_parts = [normalized]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")

        key_content = "|".join(key_parts)

        # Hash for shorter key
        key_hash = hashlib.sha256(key_content.encode()).hexdigest()[:16]

        return f"{prefix}{key_hash}"

    async def get_search(
        self,
        query: str,
        **kwargs,
    ) -> list[dict[str, Any]] | None:
        """Get cached search results."""
        if not self._enabled or not self._client:
            return None

        key = self._make_key(self.SEARCH_PREFIX, query, **kwargs)

        try:
            data = await self._client.get(key)
            if data:
                logger.debug("cache_hit", type="search", key=key)
                return json.loads(data)
        except Exception as e:
            logger.warning("cache_get_failed", error=str(e))

        return None

    async def set_search(
        self,
        query: str,
        results: list[dict[str, Any]],
        ttl: int | None = None,
        **kwargs,
    ) -> None:
        """Cache search results."""
        if not self._enabled or not self._client:
            return

        key = self._make_key(self.SEARCH_PREFIX, query, **kwargs)
        ttl = ttl or self.SEARCH_TTL

        try:
            await self._client.set(
                key,
                json.dumps(results),
                ex=ttl,
            )
            logger.debug("cache_set", type="search", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("cache_set_failed", error=str(e))

    async def get_agent_response(
        self,
        agent: str,
        query: str,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get cached agent response."""
        if not self._enabled or not self._client:
            return None

        key = self._make_key(f"{self.AGENT_PREFIX}{agent}:", query, **kwargs)

        try:
            data = await self._client.get(key)
            if data:
                logger.debug("cache_hit", type="agent", agent=agent)
                return json.loads(data)
        except Exception as e:
            logger.warning("cache_get_failed", error=str(e))

        return None

    async def set_agent_response(
        self,
        agent: str,
        query: str,
        response: dict[str, Any],
        ttl: int | None = None,
        **kwargs,
    ) -> None:
        """Cache agent response."""
        if not self._enabled or not self._client:
            return

        key = self._make_key(f"{self.AGENT_PREFIX}{agent}:", query, **kwargs)
        ttl = ttl or self.AGENT_TTL

        try:
            await self._client.set(
                key,
                json.dumps(response),
                ex=ttl,
            )
            logger.debug("cache_set", type="agent", agent=agent, ttl=ttl)
        except Exception as e:
            logger.warning("cache_set_failed", error=str(e))

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        if not self._enabled or not self._client:
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self._client.delete(*keys)
                logger.info("cache_invalidated", pattern=pattern, count=deleted)
                return deleted
        except Exception as e:
            logger.warning("cache_invalidate_failed", error=str(e))

        return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self._enabled or not self._client:
            return {"enabled": False}

        try:
            info = await self._client.info(section="stats")
            return {
                "enabled": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": await self._client.dbsize(),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}


# Singleton
_cache: QueryCache | None = None


async def get_cache() -> QueryCache:
    """Get or create query cache singleton."""
    global _cache
    if _cache is None:
        _cache = QueryCache()
        await _cache.connect()
    return _cache


async def cached_search(
    query: str,
    search_func,
    ttl: int | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Execute search with caching.

    Args:
        query: Search query
        search_func: Async function to call if cache miss
        ttl: Cache TTL in seconds
        **kwargs: Additional search parameters

    Returns:
        Search results (from cache or fresh)
    """
    cache = await get_cache()

    # Try cache first
    cached = await cache.get_search(query, **kwargs)
    if cached is not None:
        return cached

    # Execute search
    results = await search_func(query, **kwargs)

    # Cache results
    await cache.set_search(query, results, ttl=ttl, **kwargs)

    return results
