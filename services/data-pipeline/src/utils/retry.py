"""Retry utilities for the pipeline."""

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)


def retry_async(
    config: RetryConfig | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
):
    """Decorator for async functions with retry logic.

    Args:
        config: RetryConfig object (or use individual params below)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        retryable_exceptions: Tuple of exception types to retry on

    Usage:
        @retry_async(max_retries=3, base_delay=1.0)
        async def my_function():
            ...

        # Or with config object
        @retry_async(config=RetryConfig(max_retries=5))
        async def my_function():
            ...
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries or 3,
            base_delay=base_delay or 1.0,
            retryable_exceptions=retryable_exceptions or (Exception,),
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = min(
                            config.base_delay * (config.backoff_factor ** attempt),
                            config.max_delay,
                        )

                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=config.max_retries,
                            delay=delay,
                            error=str(e),
                        )

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=config.max_retries + 1,
                            error=str(e),
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator


class RetryableError(Exception):
    """Base class for retryable errors."""

    pass


class NonRetryableError(Exception):
    """Base class for non-retryable errors."""

    pass


async def retry_with_callback(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: dict | None = None,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], Any] | None = None,
    on_failure: Callable[[Exception], Any] | None = None,
) -> T:
    """Execute a function with retry logic and callbacks.

    Args:
        func: Async function to execute
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        config: Retry configuration
        on_retry: Callback called on each retry (attempt, exception)
        on_failure: Callback called on final failure

    Returns:
        Result of func

    Raises:
        Exception: The last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()
    if kwargs is None:
        kwargs = {}

    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = min(
                    config.base_delay * (config.backoff_factor ** attempt),
                    config.max_delay,
                )

                if on_retry:
                    try:
                        result = on_retry(attempt + 1, e)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as callback_error:
                        logger.warning(
                            "retry_callback_failed",
                            error=str(callback_error),
                        )

                await asyncio.sleep(delay)

    if on_failure and last_exception:
        try:
            result = on_failure(last_exception)
            if asyncio.iscoroutine(result):
                await result
        except Exception as callback_error:
            logger.warning(
                "failure_callback_failed",
                error=str(callback_error),
            )

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry state")
