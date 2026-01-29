"""Correlation ID middleware for request tracking."""

import uuid
from contextvars import ContextVar
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()

# Context variable to store correlation ID
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

# Header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"


def get_correlation_id() -> str:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    _correlation_id.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation IDs to all requests.

    Features:
    - Generates unique ID for each request
    - Accepts existing ID from header (for distributed tracing)
    - Adds ID to response headers
    - Binds ID to structlog context for logging
    """

    def __init__(self, app, header_name: str = CORRELATION_ID_HEADER):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Get existing correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = generate_correlation_id()

        # Set in context
        set_correlation_id(correlation_id)

        # Bind to structlog for all logs in this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )

        # Store in request state for access in route handlers
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id

        return response


class CorrelationIdFilter(structlog.BoundLogger):
    """Structlog filter that adds correlation ID to all log entries."""

    def _proxy_to_logger(self, method_name, event, **event_kw):
        correlation_id = get_correlation_id()
        if correlation_id:
            event_kw["correlation_id"] = correlation_id
        return super()._proxy_to_logger(method_name, event, **event_kw)


def add_correlation_id(logger, method_name, event_dict):
    """Structlog processor to add correlation ID."""
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict
