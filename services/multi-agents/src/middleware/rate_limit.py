"""Rate limiting middleware using token bucket algorithm."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger()


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self) -> float:
        """Time to wait until a token is available."""
        self._refill()
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.refill_rate


class RateLimiter:
    """Rate limiter with per-client buckets.

    Features:
    - Token bucket algorithm
    - Per-IP rate limiting
    - Per-endpoint rate limiting
    - Configurable limits
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
        cleanup_interval: int = 300,
    ):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.requests_per_second,
            )
        )
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()

    def _get_client_key(self, request: Request) -> str:
        """Get unique key for rate limiting."""
        # Use X-Forwarded-For if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"{client_ip}:{request.url.path}"

    async def is_allowed(self, request: Request) -> tuple[bool, dict]:
        """Check if request is allowed.

        Returns:
            Tuple of (allowed, headers)
        """
        key = self._get_client_key(request)

        async with self._lock:
            bucket = self._buckets[key]
            allowed = bucket.consume()

            headers = {
                "X-RateLimit-Limit": str(self.burst_size),
                "X-RateLimit-Remaining": str(int(bucket.tokens)),
                "X-RateLimit-Reset": str(int(bucket.wait_time() * 1000)),
            }

            if not allowed:
                headers["Retry-After"] = str(int(bucket.wait_time()) + 1)

            # Periodic cleanup of old buckets
            await self._cleanup()

            return allowed, headers

    async def _cleanup(self) -> None:
        """Remove stale buckets."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now
        stale_keys = []
        for key, bucket in self._buckets.items():
            if now - bucket.last_refill > self.cleanup_interval:
                stale_keys.append(key)

        for key in stale_keys:
            del self._buckets[key]

        if stale_keys:
            logger.debug("rate_limit_cleanup", removed=len(stale_keys))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        limiter: RateLimiter | None = None,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.limiter = limiter or RateLimiter()
        self.exclude_paths = exclude_paths or ["/health", "/"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        allowed, headers = await self.limiter.is_allowed(request)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                },
                headers=headers,
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Endpoint-specific rate limiters
_endpoint_limiters: dict[str, RateLimiter] = {}


def get_endpoint_limiter(
    endpoint: str,
    requests_per_second: float = 5.0,
    burst_size: int = 10,
) -> RateLimiter:
    """Get or create rate limiter for specific endpoint."""
    if endpoint not in _endpoint_limiters:
        _endpoint_limiters[endpoint] = RateLimiter(
            requests_per_second=requests_per_second,
            burst_size=burst_size,
        )
    return _endpoint_limiters[endpoint]
