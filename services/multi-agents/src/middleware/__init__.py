"""Middleware for Multi-Agent Service.

Provides:
- Rate limiting
- Correlation ID tracking
- Error handling
- Request/response logging
"""

from src.middleware.rate_limit import RateLimiter, RateLimitMiddleware
from src.middleware.correlation import CorrelationIdMiddleware, get_correlation_id
from src.middleware.error_handler import ErrorHandlerMiddleware, error_handler

__all__ = [
    "RateLimiter",
    "RateLimitMiddleware",
    "CorrelationIdMiddleware",
    "get_correlation_id",
    "ErrorHandlerMiddleware",
    "error_handler",
]
