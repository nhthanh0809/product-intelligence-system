"""Cache-aware tool wrappers.

Provides cached versions of search and aggregation tools
that automatically use Redis caching for improved performance.
"""

import functools
import hashlib
import json
from typing import Any, Callable, TypeVar

import structlog

from src.cache import get_cache, QueryCache

logger = structlog.get_logger()

T = TypeVar("T")


def cache_key_from_args(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_parts = [str(a) for a in args]
    for k, v in sorted(kwargs.items()):
        if v is not None:
            key_parts.append(f"{k}={v}")

    key_content = "|".join(key_parts)
    return hashlib.sha256(key_content.encode()).hexdigest()[:16]


def cached(
    prefix: str,
    ttl: int = 300,
    key_builder: Callable[..., str] | None = None,
):
    """Decorator to cache async function results.

    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        key_builder: Optional custom key builder function

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                key_suffix = key_builder(*args, **kwargs)
            else:
                key_suffix = cache_key_from_args(*args, **kwargs)

            cache_key = f"{prefix}:{key_suffix}"

            # Try cache
            cache = await get_cache()
            if cache._enabled and cache._client:
                try:
                    cached_data = await cache._client.get(cache_key)
                    if cached_data:
                        logger.debug("cache_hit", key=cache_key)
                        return json.loads(cached_data)
                except Exception as e:
                    logger.warning("cache_get_failed", key=cache_key, error=str(e))

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            if cache._enabled and cache._client:
                try:
                    # Handle dataclass/pydantic objects
                    if hasattr(result, "model_dump"):
                        cache_value = result.model_dump()
                    elif hasattr(result, "__dataclass_fields__"):
                        import dataclasses
                        cache_value = dataclasses.asdict(result)
                    elif isinstance(result, (list, dict)):
                        cache_value = result
                    else:
                        cache_value = result

                    await cache._client.setex(
                        cache_key,
                        ttl,
                        json.dumps(cache_value, default=str),
                    )
                    logger.debug("cache_set", key=cache_key, ttl=ttl)
                except Exception as e:
                    logger.warning("cache_set_failed", key=cache_key, error=str(e))

            return result

        return wrapper

    return decorator


class CachedSearchToolkit:
    """Cache-aware wrapper for SearchToolkit.

    Wraps SearchToolkit methods with caching for improved performance.
    """

    # Cache TTLs by operation type
    SEARCH_TTL = 300      # 5 minutes
    PRODUCT_TTL = 600     # 10 minutes
    STATS_TTL = 900       # 15 minutes

    def __init__(self, search_toolkit: Any):
        self._toolkit = search_toolkit
        self._cache: QueryCache | None = None

    async def initialize(self) -> None:
        """Initialize toolkit and cache."""
        await self._toolkit.initialize()
        self._cache = await get_cache()

    async def close(self) -> None:
        """Close connections."""
        await self._toolkit.close()

    def _make_key(self, operation: str, *args, **kwargs) -> str:
        """Make cache key for operation."""
        return f"search:{operation}:{cache_key_from_args(*args, **kwargs)}"

    async def _get_cached(self, key: str) -> Any | None:
        """Get cached value."""
        if not self._cache or not self._cache._enabled:
            return None

        try:
            data = await self._cache._client.get(key)
            if data:
                logger.debug("cache_hit", key=key)
                return json.loads(data)
        except Exception as e:
            logger.warning("cache_get_failed", key=key, error=str(e))

        return None

    async def _set_cached(self, key: str, value: Any, ttl: int) -> None:
        """Set cached value."""
        if not self._cache or not self._cache._enabled:
            return

        try:
            await self._cache._client.setex(key, ttl, json.dumps(value, default=str))
            logger.debug("cache_set", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("cache_set_failed", key=key, error=str(e))

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Cached hybrid search."""
        key = self._make_key("hybrid", query, limit=limit, **kwargs)

        # Check cache
        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        # Execute search
        results = await self._toolkit.hybrid_search(query, limit=limit, **kwargs)

        # Serialize results
        result_dicts = []
        for r in results:
            if hasattr(r, "model_dump"):
                result_dicts.append(r.model_dump())
            elif isinstance(r, dict):
                result_dicts.append(r)
            else:
                result_dicts.append(dict(r))

        # Cache results
        await self._set_cached(key, result_dicts, self.SEARCH_TTL)

        return result_dicts

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Cached semantic search."""
        key = self._make_key("semantic", query, limit=limit, **kwargs)

        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        results = await self._toolkit.semantic_search(query, limit=limit, **kwargs)

        result_dicts = []
        for r in results:
            if hasattr(r, "model_dump"):
                result_dicts.append(r.model_dump())
            elif isinstance(r, dict):
                result_dicts.append(r)
            else:
                result_dicts.append(dict(r))

        await self._set_cached(key, result_dicts, self.SEARCH_TTL)

        return result_dicts

    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Cached keyword search."""
        key = self._make_key("keyword", query, limit=limit, **kwargs)

        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        results = await self._toolkit.keyword_search(query, limit=limit, **kwargs)

        result_dicts = []
        for r in results:
            if hasattr(r, "model_dump"):
                result_dicts.append(r.model_dump())
            elif isinstance(r, dict):
                result_dicts.append(r)
            else:
                result_dicts.append(dict(r))

        await self._set_cached(key, result_dicts, self.SEARCH_TTL)

        return result_dicts

    async def section_search(
        self,
        query: str,
        section: str,
        limit: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Cached section search."""
        key = self._make_key("section", query, section=section, limit=limit, **kwargs)

        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        results = await self._toolkit.section_search(query, section=section, limit=limit, **kwargs)

        result_dicts = []
        for r in results:
            if hasattr(r, "model_dump"):
                result_dicts.append(r.model_dump())
            elif isinstance(r, dict):
                result_dicts.append(r)
            else:
                result_dicts.append(dict(r))

        await self._set_cached(key, result_dicts, self.SEARCH_TTL)

        return result_dicts

    async def find_similar(
        self,
        asin: str,
        limit: int = 10,
        **kwargs,
    ) -> list[dict]:
        """Cached similar product search."""
        key = self._make_key("similar", asin, limit=limit, **kwargs)

        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        results = await self._toolkit.find_similar(asin, limit=limit, **kwargs)

        result_dicts = []
        for r in results:
            if hasattr(r, "model_dump"):
                result_dicts.append(r.model_dump())
            elif isinstance(r, dict):
                result_dicts.append(r)
            else:
                result_dicts.append(dict(r))

        await self._set_cached(key, result_dicts, self.SEARCH_TTL)

        return result_dicts

    async def get_product_by_asin(self, asin: str) -> dict | None:
        """Cached product lookup."""
        key = self._make_key("product", asin)

        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        result = await self._toolkit.get_product_by_asin(asin)

        if result:
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = dict(result)

            await self._set_cached(key, result_dict, self.PRODUCT_TTL)
            return result_dict

        return None

    # Pass through non-cached methods
    def __getattr__(self, name: str):
        """Pass through to underlying toolkit."""
        return getattr(self._toolkit, name)


class CachedAggregationToolkit:
    """Cache-aware wrapper for AggregationToolkit."""

    STATS_TTL = 900  # 15 minutes

    def __init__(self, aggregation_toolkit: Any):
        self._toolkit = aggregation_toolkit
        self._cache: QueryCache | None = None

    async def initialize(self) -> None:
        """Initialize toolkit and cache."""
        await self._toolkit.initialize()
        self._cache = await get_cache()

    async def close(self) -> None:
        """Close connections."""
        await self._toolkit.close()

    def _make_key(self, operation: str, *args, **kwargs) -> str:
        """Make cache key for operation."""
        return f"agg:{operation}:{cache_key_from_args(*args, **kwargs)}"

    async def _get_cached(self, key: str) -> Any | None:
        """Get cached value."""
        if not self._cache or not self._cache._enabled:
            return None

        try:
            data = await self._cache._client.get(key)
            if data:
                logger.debug("cache_hit", key=key)
                return json.loads(data)
        except Exception as e:
            logger.warning("cache_get_failed", key=key, error=str(e))

        return None

    async def _set_cached(self, key: str, value: Any, ttl: int) -> None:
        """Set cached value."""
        if not self._cache or not self._cache._enabled:
            return

        try:
            await self._cache._client.setex(key, ttl, json.dumps(value, default=str))
            logger.debug("cache_set", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("cache_set_failed", key=key, error=str(e))

    async def get_brand_stats(self, brand: str):
        """Cached brand statistics."""
        key = self._make_key("brand_stats", brand)

        cached = await self._get_cached(key)
        if cached is not None:
            from src.tools.aggregation_tools import BrandStats
            return BrandStats(**cached)

        result = await self._toolkit.get_brand_stats(brand)

        if result:
            import dataclasses
            await self._set_cached(key, dataclasses.asdict(result), self.STATS_TTL)

        return result

    async def get_category_stats(self, category: str):
        """Cached category statistics."""
        key = self._make_key("category_stats", category)

        cached = await self._get_cached(key)
        if cached is not None:
            from src.tools.aggregation_tools import CategoryStats
            return CategoryStats(**cached)

        result = await self._toolkit.get_category_stats(category)

        if result:
            import dataclasses
            await self._set_cached(key, dataclasses.asdict(result), self.STATS_TTL)

        return result

    async def get_top_brands(self, category: str | None = None, limit: int = 10, sort_by: str = "product_count"):
        """Cached top brands."""
        key = self._make_key("top_brands", category=category, limit=limit, sort_by=sort_by)

        cached = await self._get_cached(key)
        if cached is not None:
            from src.tools.aggregation_tools import BrandStats
            return [BrandStats(**b) for b in cached]

        result = await self._toolkit.get_top_brands(category=category, limit=limit, sort_by=sort_by)

        if result:
            import dataclasses
            await self._set_cached(
                key,
                [dataclasses.asdict(b) for b in result],
                self.STATS_TTL,
            )

        return result

    async def get_top_categories(self, limit: int = 10, sort_by: str = "product_count"):
        """Cached top categories."""
        key = self._make_key("top_categories", limit=limit, sort_by=sort_by)

        cached = await self._get_cached(key)
        if cached is not None:
            from src.tools.aggregation_tools import CategoryStats
            return [CategoryStats(**c) for c in cached]

        result = await self._toolkit.get_top_categories(limit=limit, sort_by=sort_by)

        if result:
            import dataclasses
            await self._set_cached(
                key,
                [dataclasses.asdict(c) for c in result],
                self.STATS_TTL,
            )

        return result

    async def get_price_distribution(self, category: str | None = None, brand: str | None = None):
        """Cached price distribution."""
        key = self._make_key("price_dist", category=category, brand=brand)

        cached = await self._get_cached(key)
        if cached is not None:
            from src.tools.aggregation_tools import PriceDistribution
            return PriceDistribution(**cached)

        result = await self._toolkit.get_price_distribution(category=category, brand=brand)

        if result:
            import dataclasses
            await self._set_cached(key, dataclasses.asdict(result), self.STATS_TTL)

        return result

    # Pass through non-cached methods
    def __getattr__(self, name: str):
        """Pass through to underlying toolkit."""
        return getattr(self._toolkit, name)


# Factory functions
async def get_cached_search_toolkit() -> CachedSearchToolkit:
    """Get cached search toolkit."""
    from src.tools.search_tools import SearchToolkit

    toolkit = SearchToolkit()
    cached = CachedSearchToolkit(toolkit)
    await cached.initialize()
    return cached


async def get_cached_aggregation_toolkit() -> CachedAggregationToolkit:
    """Get cached aggregation toolkit."""
    from src.tools.aggregation_tools import AggregationToolkit

    toolkit = AggregationToolkit()
    cached = CachedAggregationToolkit(toolkit)
    await cached.initialize()
    return cached
