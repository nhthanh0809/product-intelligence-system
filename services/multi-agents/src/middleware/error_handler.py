"""Error handling middleware and utilities."""

import traceback
from datetime import datetime
from typing import Any, Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.middleware.correlation import get_correlation_id

logger = structlog.get_logger()


class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(AppError):
    """Request validation error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class NotFoundError(AppError):
    """Resource not found error."""

    def __init__(self, resource: str, identifier: str | None = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} '{identifier}' not found"
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": identifier},
        )


class ServiceUnavailableError(AppError):
    """External service unavailable."""

    def __init__(self, service: str, details: dict | None = None):
        super().__init__(
            message=f"Service '{service}' is unavailable",
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details={"service": service, **(details or {})},
        )


class RateLimitError(AppError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after": retry_after},
        )


def format_error_response(
    error_code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Format standardized error response."""
    response = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    }

    if details:
        response["error"]["details"] = details

    if correlation_id:
        response["error"]["correlation_id"] = correlation_id

    return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling.

    Features:
    - Catches all unhandled exceptions
    - Formats errors consistently
    - Logs errors with context
    - Includes correlation ID in error response
    """

    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        try:
            return await call_next(request)

        except AppError as e:
            # Application-level error
            correlation_id = get_correlation_id()
            logger.warning(
                "app_error",
                error_code=e.error_code,
                message=e.message,
                status_code=e.status_code,
                details=e.details,
            )

            return JSONResponse(
                status_code=e.status_code,
                content=format_error_response(
                    error_code=e.error_code,
                    message=e.message,
                    status_code=e.status_code,
                    details=e.details,
                    correlation_id=correlation_id,
                ),
            )

        except Exception as e:
            # Unexpected error
            correlation_id = get_correlation_id()
            error_id = correlation_id or "unknown"

            # Log full traceback
            logger.error(
                "unhandled_error",
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
            )

            # Don't expose internal details in production
            message = str(e) if self.debug else "An internal error occurred"
            details = {"error_id": error_id}
            if self.debug:
                details["traceback"] = traceback.format_exc()

            return JSONResponse(
                status_code=500,
                content=format_error_response(
                    error_code="INTERNAL_ERROR",
                    message=message,
                    status_code=500,
                    details=details,
                    correlation_id=correlation_id,
                ),
            )


def error_handler(func: Callable) -> Callable:
    """Decorator for route-level error handling.

    Wraps async route handlers to catch and format errors consistently.
    """
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AppError:
            raise  # Let middleware handle AppError
        except ValueError as e:
            raise ValidationError(str(e))
        except KeyError as e:
            raise NotFoundError("Resource", str(e))
        except Exception:
            raise  # Let middleware handle unexpected errors

    return wrapper
