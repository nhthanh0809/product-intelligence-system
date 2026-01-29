"""Execution models for multi-agent orchestration.

This module provides models for:
- ExecutionStep: Individual agent execution step
- ExecutionPlan: Complete execution plan with steps
- ExecutionResult: Result of plan execution
- StepResult: Result of individual step execution
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of an execution step."""

    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied, can execute
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Skipped due to dependency failure
    TIMEOUT = "timeout"


class AgentType(str, Enum):
    """Types of agents available for execution."""

    # Core agents
    INTENT = "intent"
    GENERAL = "general"
    SUPERVISOR = "supervisor"

    # Product agents
    RETRIEVAL = "retrieval"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    PRICE = "price"
    TREND = "trend"
    RECOMMEND = "recommend"
    ATTRIBUTE = "attribute"

    # Output agents
    SYNTHESIS = "synthesis"

    @classmethod
    def product_agents(cls) -> set["AgentType"]:
        """Return set of product-related agents."""
        return {
            cls.RETRIEVAL,
            cls.COMPARISON,
            cls.ANALYSIS,
            cls.PRICE,
            cls.TREND,
            cls.RECOMMEND,
            cls.ATTRIBUTE,
        }

    @classmethod
    def from_intent(cls, intent_name: str) -> "AgentType":
        """Map intent name to agent type."""
        mapping = {
            "search": cls.RETRIEVAL,
            "compare": cls.COMPARISON,
            "analyze": cls.ANALYSIS,
            "price_check": cls.PRICE,
            "trend": cls.TREND,
            "recommend": cls.RECOMMEND,
            "greeting": cls.GENERAL,
            "farewell": cls.GENERAL,
            "help": cls.GENERAL,
            "small_talk": cls.GENERAL,
            "off_topic": cls.GENERAL,
            "clarification": cls.GENERAL,
        }
        return mapping.get(intent_name, cls.RETRIEVAL)


class ExecutionStep(BaseModel):
    """Individual step in an execution plan.

    Each step represents a single agent invocation with its
    inputs, dependencies, and execution constraints.
    """

    step_id: int = Field(description="Unique step identifier within plan")
    agent_type: AgentType = Field(description="Type of agent to execute")
    description: str = Field(description="Human-readable step description")

    # Input/output
    input_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the agent",
    )
    input_from_steps: list[int] = Field(
        default_factory=list,
        description="Step IDs whose output should be passed as input",
    )

    # Dependencies
    depends_on: list[int] = Field(
        default_factory=list,
        description="Step IDs that must complete before this step",
    )

    # Execution constraints
    timeout_seconds: float = Field(
        default=30.0,
        description="Maximum execution time for this step",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries on failure",
    )
    optional: bool = Field(
        default=False,
        description="If True, plan continues even if step fails",
    )

    # Execution state (updated during execution)
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current execution status",
    )
    started_at: datetime | None = Field(
        default=None,
        description="When step execution started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When step execution completed",
    )
    error: str | None = Field(
        default=None,
        description="Error message if step failed",
    )

    def is_ready(self, completed_steps: set[int]) -> bool:
        """Check if step is ready to execute.

        Args:
            completed_steps: Set of step IDs that have completed successfully

        Returns:
            True if all dependencies are satisfied
        """
        return all(dep in completed_steps for dep in self.depends_on)

    def can_run_parallel_with(self, other: "ExecutionStep") -> bool:
        """Check if this step can run in parallel with another.

        Args:
            other: Another execution step

        Returns:
            True if steps have no dependency relationship
        """
        # Neither step depends on the other
        return (
            other.step_id not in self.depends_on
            and self.step_id not in other.depends_on
        )


class ExecutionPlan(BaseModel):
    """Complete execution plan for a query.

    The plan contains ordered steps that agents should execute
    to fulfill a user query. Steps may have dependencies and
    can potentially run in parallel.
    """

    plan_id: str = Field(description="Unique plan identifier")
    query: str = Field(description="Original user query")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When plan was created",
    )

    # Steps
    steps: list[ExecutionStep] = Field(
        default_factory=list,
        description="Ordered list of execution steps",
    )

    # Plan metadata
    estimated_duration_seconds: float = Field(
        default=0.0,
        description="Estimated total execution time",
    )
    requires_context: bool = Field(
        default=False,
        description="Whether plan needs conversation context",
    )
    is_compound: bool = Field(
        default=False,
        description="Whether this is a compound (multi-intent) query",
    )

    # Execution state
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Overall plan status",
    )
    started_at: datetime | None = Field(
        default=None,
        description="When plan execution started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When plan execution completed",
    )

    def get_step(self, step_id: int) -> ExecutionStep | None:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_ready_steps(self) -> list[ExecutionStep]:
        """Get steps that are ready to execute.

        Returns:
            List of steps with all dependencies satisfied and status PENDING
        """
        completed_ids = {
            s.step_id for s in self.steps if s.status == StepStatus.COMPLETED
        }

        ready = []
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.is_ready(completed_ids):
                ready.append(step)

        return ready

    def get_parallel_groups(self) -> list[list[ExecutionStep]]:
        """Get groups of steps that can execute in parallel.

        Returns:
            List of step groups, where steps within each group can run in parallel
        """
        groups: list[list[ExecutionStep]] = []
        processed: set[int] = set()

        while len(processed) < len(self.steps):
            # Find all steps ready at this level
            group = []
            for step in self.steps:
                if step.step_id in processed:
                    continue
                # Check if all dependencies are in processed set
                if all(dep in processed for dep in step.depends_on):
                    group.append(step)

            if not group:
                break  # Circular dependency or error

            groups.append(group)
            processed.update(s.step_id for s in group)

        return groups

    def all_completed(self) -> bool:
        """Check if all steps are completed."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )

    def has_failures(self) -> bool:
        """Check if any required step failed."""
        return any(
            s.status in (StepStatus.FAILED, StepStatus.TIMEOUT)
            and not s.optional
            for s in self.steps
        )

    def get_execution_order(self) -> list[int]:
        """Get topologically sorted step execution order.

        Returns:
            List of step IDs in execution order
        """
        order = []
        for group in self.get_parallel_groups():
            order.extend(s.step_id for s in group)
        return order


class StepResult(BaseModel):
    """Result of executing a single step."""

    step_id: int = Field(description="Step that produced this result")
    agent_type: AgentType = Field(description="Agent that executed")
    success: bool = Field(description="Whether step succeeded")

    # Output
    output: Any = Field(
        default=None,
        description="Agent output data",
    )
    error: str | None = Field(
        default=None,
        description="Error message if failed",
    )

    # Metrics
    duration_ms: float = Field(
        default=0.0,
        description="Execution duration in milliseconds",
    )
    retries: int = Field(
        default=0,
        description="Number of retries before success/failure",
    )


class ExecutionResult(BaseModel):
    """Result of executing a complete plan."""

    plan_id: str = Field(description="Plan that was executed")
    query: str = Field(description="Original query")
    success: bool = Field(description="Whether plan succeeded overall")

    # Step results
    step_results: dict[int, StepResult] = Field(
        default_factory=dict,
        description="Results keyed by step_id",
    )

    # Final output
    final_output: Any = Field(
        default=None,
        description="Final synthesized output",
    )
    error: str | None = Field(
        default=None,
        description="Overall error if plan failed",
    )

    # Metrics
    total_duration_ms: float = Field(
        default=0.0,
        description="Total execution time in milliseconds",
    )
    steps_completed: int = Field(
        default=0,
        description="Number of steps completed successfully",
    )
    steps_failed: int = Field(
        default=0,
        description="Number of steps that failed",
    )
    steps_skipped: int = Field(
        default=0,
        description="Number of steps skipped",
    )

    # Execution trace
    execution_trace: list[str] = Field(
        default_factory=list,
        description="Ordered list of execution events for debugging",
    )

    def add_step_result(self, result: StepResult) -> None:
        """Add a step result and update metrics."""
        self.step_results[result.step_id] = result

        if result.success:
            self.steps_completed += 1
        else:
            self.steps_failed += 1

        self.total_duration_ms += result.duration_ms

    def get_output_for_step(self, step_id: int) -> Any:
        """Get output from a specific step."""
        result = self.step_results.get(step_id)
        return result.output if result else None
