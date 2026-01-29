"""Base agent class with retry and circuit breaker patterns."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import structlog

from src.config import get_settings

logger = structlog.get_logger()

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.failure_count >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        """Open the circuit."""
        self.state = CircuitState.OPEN
        logger.warning("circuit_breaker_opened", failures=self.failure_count)

    def _close(self) -> None:
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info("circuit_breaker_closed")


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_exceptions: tuple = (Exception,)


@dataclass
class AgentMetrics:
    """Agent execution metrics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    last_call_time: datetime | None = None

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls * 100


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """Base agent class with retry and circuit breaker.

    Provides:
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Metrics tracking
    - Structured logging
    """

    name: str = "base"
    description: str = "Base agent"

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.settings = get_settings()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.metrics = AgentMetrics()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize agent resources. Override in subclasses."""
        self._initialized = True

    async def close(self) -> None:
        """Close agent resources. Override in subclasses."""
        pass

    async def execute(self, input_data: TInput) -> TOutput:
        """Execute agent with retry and circuit breaker.

        Args:
            input_data: Agent input

        Returns:
            Agent output

        Raises:
            Exception: If all retries fail or circuit is open
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(
                f"Circuit breaker open for {self.name}. "
                f"Retry after {self.circuit_breaker.recovery_timeout}s"
            )

        # Initialize if needed
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()
        self.metrics.total_calls += 1
        self.metrics.last_call_time = datetime.now()

        last_exception: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = await self._execute_internal(input_data)

                # Record success
                self.circuit_breaker.record_success()
                self.metrics.successful_calls += 1
                self.metrics.total_latency_ms += (
                    time.perf_counter() - start_time
                ) * 1000

                logger.debug(
                    "agent_executed",
                    agent=self.name,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    attempt=attempt + 1,
                )

                return result

            except self.retry_config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.retry_config.max_retries:
                    delay = min(
                        self.retry_config.base_delay
                        * (self.retry_config.backoff_factor ** attempt),
                        self.retry_config.max_delay,
                    )

                    logger.warning(
                        "agent_retry",
                        agent=self.name,
                        attempt=attempt + 1,
                        max_retries=self.retry_config.max_retries,
                        delay=delay,
                        error=str(e),
                    )

                    await asyncio.sleep(delay)

        # All retries failed
        self.circuit_breaker.record_failure()
        self.metrics.failed_calls += 1

        logger.error(
            "agent_failed",
            agent=self.name,
            attempts=self.retry_config.max_retries + 1,
            error=str(last_exception),
        )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Agent {self.name} failed without exception")

    @abstractmethod
    async def _execute_internal(self, input_data: TInput) -> TOutput:
        """Internal execution logic. Must be implemented by subclasses."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get agent metrics."""
        return {
            "name": self.name,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": self.metrics.success_rate,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "circuit_state": self.circuit_breaker.state.value,
            "last_call_time": (
                self.metrics.last_call_time.isoformat()
                if self.metrics.last_call_time
                else None
            ),
        }
