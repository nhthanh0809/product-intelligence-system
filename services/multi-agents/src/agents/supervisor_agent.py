"""Supervisor Agent for orchestrating multi-agent execution.

The Supervisor Agent is responsible for:
1. Generating execution plans from intent analysis
2. Coordinating agent execution with dependency management
3. Handling parallel execution of independent steps
4. Managing errors, retries, and timeouts
5. Tracking execution progress
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine

import structlog

from src.agents.base import BaseAgent, RetryConfig
from src.models.intent import IntentAnalysis, QueryIntent, SubTask
from src.models.execution import (
    AgentType,
    ExecutionPlan,
    ExecutionStep,
    ExecutionResult,
    StepResult,
    StepStatus,
)

logger = structlog.get_logger()


# Type alias for agent executor functions
AgentExecutor = Callable[[dict[str, Any]], Coroutine[Any, Any, Any]]


class SupervisorAgent(BaseAgent[IntentAnalysis, ExecutionPlan]):
    """Supervisor agent for orchestrating multi-agent workflows.

    The supervisor takes an IntentAnalysis and creates an ExecutionPlan
    that specifies which agents to run and in what order. It handles:

    - Intent-to-agent mapping
    - Dependency resolution between steps
    - Parallel execution of independent steps
    - Error handling and fallback strategies
    - Timeout management per step
    - Progress tracking and logging
    """

    name = "supervisor"
    description = "Orchestrates multi-agent execution workflows"

    # Default timeouts per agent type (seconds)
    DEFAULT_TIMEOUTS: dict[AgentType, float] = {
        AgentType.INTENT: 5.0,
        AgentType.GENERAL: 2.0,
        AgentType.RETRIEVAL: 15.0,
        AgentType.COMPARISON: 20.0,
        AgentType.ANALYSIS: 20.0,
        AgentType.PRICE: 10.0,
        AgentType.TREND: 15.0,
        AgentType.RECOMMEND: 15.0,
        AgentType.ATTRIBUTE: 10.0,
        AgentType.SYNTHESIS: 10.0,
    }

    # Intent to agent type mapping
    INTENT_AGENT_MAP: dict[QueryIntent, AgentType] = {
        # Product intents
        QueryIntent.SEARCH: AgentType.RETRIEVAL,
        QueryIntent.COMPARE: AgentType.COMPARISON,
        QueryIntent.ANALYZE: AgentType.ANALYSIS,
        QueryIntent.PRICE_CHECK: AgentType.PRICE,
        QueryIntent.TREND: AgentType.TREND,
        QueryIntent.RECOMMEND: AgentType.RECOMMEND,
        # General intents
        QueryIntent.GREETING: AgentType.GENERAL,
        QueryIntent.FAREWELL: AgentType.GENERAL,
        QueryIntent.HELP: AgentType.GENERAL,
        QueryIntent.SMALL_TALK: AgentType.GENERAL,
        QueryIntent.OFF_TOPIC: AgentType.GENERAL,
        QueryIntent.CLARIFICATION: AgentType.GENERAL,
        QueryIntent.AMBIGUOUS: AgentType.GENERAL,
    }

    # Agents that require retrieval results as input
    RETRIEVAL_DEPENDENT_AGENTS: set[AgentType] = {
        AgentType.COMPARISON,
        AgentType.ANALYSIS,
        AgentType.PRICE,
        AgentType.RECOMMEND,
        AgentType.ATTRIBUTE,
    }

    def __init__(self):
        super().__init__(retry_config=RetryConfig(max_retries=1))
        self._agent_executors: dict[AgentType, AgentExecutor] = {}

    def register_executor(
        self,
        agent_type: AgentType,
        executor: AgentExecutor,
    ) -> None:
        """Register an executor function for an agent type.

        Args:
            agent_type: The type of agent
            executor: Async function that executes the agent
        """
        self._agent_executors[agent_type] = executor
        logger.debug("executor_registered", agent_type=agent_type.value)

    async def _execute_internal(self, analysis: IntentAnalysis) -> ExecutionPlan:
        """Generate execution plan from intent analysis.

        Args:
            analysis: IntentAnalysis from IntentAgent

        Returns:
            ExecutionPlan with steps to execute
        """
        return self.create_plan(analysis)

    def create_plan(self, analysis: IntentAnalysis) -> ExecutionPlan:
        """Create an execution plan from intent analysis.

        Args:
            analysis: IntentAnalysis with detected intents and entities

        Returns:
            ExecutionPlan with ordered steps
        """
        plan_id = str(uuid.uuid4())[:8]

        # Handle general chat intents (no product search needed)
        if analysis.primary_intent.is_general_chat():
            return self._create_general_chat_plan(plan_id, analysis)

        # Handle product-related intents
        return self._create_product_plan(plan_id, analysis)

    def _create_general_chat_plan(
        self,
        plan_id: str,
        analysis: IntentAnalysis,
    ) -> ExecutionPlan:
        """Create plan for general chat (greeting, help, etc.).

        Args:
            plan_id: Plan identifier
            analysis: IntentAnalysis

        Returns:
            Simple plan with just GeneralAgent
        """
        step = ExecutionStep(
            step_id=1,
            agent_type=AgentType.GENERAL,
            description=f"Handle {analysis.primary_intent.value} intent",
            input_params={
                "intent": analysis.primary_intent.value,
                "query": analysis.query,
            },
            timeout_seconds=self.DEFAULT_TIMEOUTS[AgentType.GENERAL],
        )

        return ExecutionPlan(
            plan_id=plan_id,
            query=analysis.query,
            steps=[step],
            estimated_duration_seconds=step.timeout_seconds,
            is_compound=False,
            requires_context=analysis.needs_context(),
        )

    def _create_product_plan(
        self,
        plan_id: str,
        analysis: IntentAnalysis,
    ) -> ExecutionPlan:
        """Create plan for product-related queries.

        Args:
            plan_id: Plan identifier
            analysis: IntentAnalysis

        Returns:
            ExecutionPlan with retrieval, processing, and synthesis steps
        """
        steps: list[ExecutionStep] = []
        step_id = 0

        # Collect all intents to process
        all_intents = analysis.get_all_intents()

        # Determine if we need retrieval
        needs_retrieval = self._needs_retrieval(all_intents)

        # Step 1: Retrieval (if needed)
        retrieval_step_id: int | None = None
        if needs_retrieval:
            step_id += 1
            retrieval_step_id = step_id

            retrieval_step = ExecutionStep(
                step_id=step_id,
                agent_type=AgentType.RETRIEVAL,
                description="Search for relevant products",
                input_params=self._build_retrieval_params(analysis),
                timeout_seconds=self.DEFAULT_TIMEOUTS[AgentType.RETRIEVAL],
            )
            steps.append(retrieval_step)

        # Step 2+: Processing agents based on intents
        processing_steps = self._create_processing_steps(
            all_intents,
            analysis,
            step_id,
            retrieval_step_id,
        )
        steps.extend(processing_steps)
        step_id += len(processing_steps)

        # Final step: Synthesis
        step_id += 1
        synthesis_deps = [s.step_id for s in steps]  # Depends on all previous steps

        synthesis_step = ExecutionStep(
            step_id=step_id,
            agent_type=AgentType.SYNTHESIS,
            description="Generate final response",
            input_params={"query": analysis.query},
            input_from_steps=synthesis_deps,
            depends_on=synthesis_deps,
            timeout_seconds=self.DEFAULT_TIMEOUTS[AgentType.SYNTHESIS],
        )
        steps.append(synthesis_step)

        # Calculate estimated duration
        estimated_duration = self._estimate_duration(steps)

        return ExecutionPlan(
            plan_id=plan_id,
            query=analysis.query,
            steps=steps,
            estimated_duration_seconds=estimated_duration,
            is_compound=analysis.is_compound(),
            requires_context=analysis.needs_context(),
        )

    def _needs_retrieval(self, intents: list[QueryIntent]) -> bool:
        """Check if any intent requires product retrieval.

        Args:
            intents: List of query intents

        Returns:
            True if retrieval is needed
        """
        for intent in intents:
            agent_type = self.INTENT_AGENT_MAP.get(intent)
            if agent_type and agent_type in self.RETRIEVAL_DEPENDENT_AGENTS:
                return True
            if intent == QueryIntent.SEARCH:
                return True
        return False

    def _build_retrieval_params(self, analysis: IntentAnalysis) -> dict[str, Any]:
        """Build parameters for retrieval agent.

        Args:
            analysis: IntentAnalysis with entities and constraints

        Returns:
            Dict of retrieval parameters
        """
        params: dict[str, Any] = {"query": analysis.query}

        if analysis.entities:
            entities = analysis.entities
            if entities.products:
                params["products"] = entities.products
            if entities.brands:
                params["brands"] = entities.brands
            if entities.categories:
                params["category"] = entities.categories[0]
            if entities.constraints:
                params["constraints"] = entities.constraints

        return params

    def _create_processing_steps(
        self,
        intents: list[QueryIntent],
        analysis: IntentAnalysis,
        current_step_id: int,
        retrieval_step_id: int | None,
    ) -> list[ExecutionStep]:
        """Create processing steps for each intent.

        Args:
            intents: List of intents to process
            analysis: Full intent analysis
            current_step_id: Current step counter
            retrieval_step_id: ID of retrieval step (if any)

        Returns:
            List of processing steps
        """
        steps: list[ExecutionStep] = []
        step_id = current_step_id

        # Filter out SEARCH intent (handled by retrieval)
        processing_intents = [i for i in intents if i != QueryIntent.SEARCH]

        # Group intents by whether they can run in parallel
        parallel_intents: list[QueryIntent] = []
        sequential_intents: list[QueryIntent] = []

        for intent in processing_intents:
            if intent in (QueryIntent.COMPARE, QueryIntent.RECOMMEND):
                # These may need results from other agents
                sequential_intents.append(intent)
            else:
                parallel_intents.append(intent)

        # Create steps for parallel intents
        for intent in parallel_intents:
            agent_type = self.INTENT_AGENT_MAP.get(intent)
            if not agent_type or agent_type == AgentType.GENERAL:
                continue

            step_id += 1
            deps = [retrieval_step_id] if retrieval_step_id else []

            step = ExecutionStep(
                step_id=step_id,
                agent_type=agent_type,
                description=f"Execute {intent.value} analysis",
                input_params=self._build_agent_params(intent, analysis),
                input_from_steps=[retrieval_step_id] if retrieval_step_id else [],
                depends_on=deps,
                timeout_seconds=self.DEFAULT_TIMEOUTS.get(agent_type, 15.0),
            )
            steps.append(step)

        # Track IDs of parallel steps for sequential dependencies
        parallel_step_ids = [s.step_id for s in steps]

        # Create steps for sequential intents
        for intent in sequential_intents:
            agent_type = self.INTENT_AGENT_MAP.get(intent)
            if not agent_type or agent_type == AgentType.GENERAL:
                continue

            step_id += 1

            # These depend on retrieval AND parallel steps
            deps = []
            if retrieval_step_id:
                deps.append(retrieval_step_id)
            deps.extend(parallel_step_ids)

            step = ExecutionStep(
                step_id=step_id,
                agent_type=agent_type,
                description=f"Execute {intent.value} analysis",
                input_params=self._build_agent_params(intent, analysis),
                input_from_steps=deps,
                depends_on=deps,
                timeout_seconds=self.DEFAULT_TIMEOUTS.get(agent_type, 15.0),
            )
            steps.append(step)

        return steps

    def _build_agent_params(
        self,
        intent: QueryIntent,
        analysis: IntentAnalysis,
    ) -> dict[str, Any]:
        """Build parameters for a specific agent.

        Args:
            intent: The intent being processed
            analysis: Full intent analysis

        Returns:
            Dict of agent parameters
        """
        params: dict[str, Any] = {
            "query": analysis.query,
            "intent": intent.value,
        }

        if analysis.entities:
            if analysis.entities.products:
                params["products"] = analysis.entities.products
            if analysis.entities.brands:
                params["brands"] = analysis.entities.brands
            if analysis.entities.constraints:
                params["constraints"] = analysis.entities.constraints

        return params

    def _estimate_duration(self, steps: list[ExecutionStep]) -> float:
        """Estimate total plan duration considering parallelism.

        Args:
            steps: List of execution steps

        Returns:
            Estimated duration in seconds
        """
        if not steps:
            return 0.0

        # Group steps by dependency level
        levels: list[list[ExecutionStep]] = []
        processed: set[int] = set()

        while len(processed) < len(steps):
            level = []
            for step in steps:
                if step.step_id in processed:
                    continue
                if all(dep in processed for dep in step.depends_on):
                    level.append(step)

            if not level:
                break

            levels.append(level)
            processed.update(s.step_id for s in level)

        # Duration is sum of max timeout per level (parallel within level)
        total = 0.0
        for level in levels:
            total += max(s.timeout_seconds for s in level)

        return total

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute a plan and return results.

        Args:
            plan: ExecutionPlan to execute
            context: Optional context from previous turns

        Returns:
            ExecutionResult with all step outputs
        """
        start_time = time.perf_counter()

        result = ExecutionResult(
            plan_id=plan.plan_id,
            query=plan.query,
            success=False,
        )

        plan.status = StepStatus.RUNNING
        plan.started_at = datetime.now()

        result.execution_trace.append(f"Plan {plan.plan_id} started")

        try:
            # Execute steps in dependency order
            for group in plan.get_parallel_groups():
                # Execute group in parallel
                group_results = await self._execute_step_group(
                    group, result, context
                )

                # Check for failures in required steps
                for step_result in group_results:
                    if not step_result.success:
                        step = plan.get_step(step_result.step_id)
                        if step and not step.optional:
                            # Required step failed, abort plan
                            result.error = f"Step {step_result.step_id} failed: {step_result.error}"
                            result.execution_trace.append(
                                f"Plan aborted due to step {step_result.step_id} failure"
                            )
                            return result

            # All steps completed
            plan.status = StepStatus.COMPLETED
            plan.completed_at = datetime.now()
            result.success = True

            # Get final output from synthesis step
            synthesis_steps = [
                s for s in plan.steps if s.agent_type == AgentType.SYNTHESIS
            ]
            if synthesis_steps:
                final_step_id = synthesis_steps[-1].step_id
                result.final_output = result.get_output_for_step(final_step_id)

            result.execution_trace.append(
                f"Plan completed successfully in {result.total_duration_ms:.1f}ms"
            )

        except Exception as e:
            plan.status = StepStatus.FAILED
            result.error = str(e)
            result.execution_trace.append(f"Plan failed with error: {e}")
            logger.error("plan_execution_failed", plan_id=plan.plan_id, error=str(e))

        result.total_duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _execute_step_group(
        self,
        steps: list[ExecutionStep],
        result: ExecutionResult,
        context: dict[str, Any] | None,
    ) -> list[StepResult]:
        """Execute a group of steps in parallel.

        Args:
            steps: Steps to execute (can run in parallel)
            result: ExecutionResult to update
            context: Optional context

        Returns:
            List of StepResults
        """
        tasks = []
        for step in steps:
            task = self._execute_single_step(step, result, context)
            tasks.append(task)

        step_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results: list[StepResult] = []
        for i, step_result in enumerate(step_results):
            if isinstance(step_result, Exception):
                # Task raised exception
                step = steps[i]
                error_result = StepResult(
                    step_id=step.step_id,
                    agent_type=step.agent_type,
                    success=False,
                    error=str(step_result),
                )
                processed_results.append(error_result)
                result.add_step_result(error_result)
            else:
                processed_results.append(step_result)
                result.add_step_result(step_result)

        return processed_results

    async def _execute_single_step(
        self,
        step: ExecutionStep,
        result: ExecutionResult,
        context: dict[str, Any] | None,
    ) -> StepResult:
        """Execute a single step with timeout.

        Args:
            step: Step to execute
            result: ExecutionResult for accessing previous outputs
            context: Optional context

        Returns:
            StepResult
        """
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()

        result.execution_trace.append(
            f"Step {step.step_id} ({step.agent_type.value}) started"
        )

        start_time = time.perf_counter()

        try:
            # Build input from previous step outputs
            input_data = dict(step.input_params)

            # Add outputs from dependent steps
            for dep_step_id in step.input_from_steps:
                dep_output = result.get_output_for_step(dep_step_id)
                if dep_output is not None:
                    input_data[f"step_{dep_step_id}_output"] = dep_output

            # Add context if needed
            if context:
                input_data["context"] = context

            # Get executor for this agent type
            executor = self._agent_executors.get(step.agent_type)
            if not executor:
                raise RuntimeError(f"No executor registered for {step.agent_type.value}")

            # Execute with timeout
            output = await asyncio.wait_for(
                executor(input_data),
                timeout=step.timeout_seconds,
            )

            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()

            duration_ms = (time.perf_counter() - start_time) * 1000
            result.execution_trace.append(
                f"Step {step.step_id} completed in {duration_ms:.1f}ms"
            )

            return StepResult(
                step_id=step.step_id,
                agent_type=step.agent_type,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            step.status = StepStatus.TIMEOUT
            step.completed_at = datetime.now()
            step.error = f"Timeout after {step.timeout_seconds}s"

            result.execution_trace.append(
                f"Step {step.step_id} timed out after {step.timeout_seconds}s"
            )

            return StepResult(
                step_id=step.step_id,
                agent_type=step.agent_type,
                success=False,
                error=step.error,
                duration_ms=step.timeout_seconds * 1000,
            )

        except Exception as e:
            step.status = StepStatus.FAILED
            step.completed_at = datetime.now()
            step.error = str(e)

            duration_ms = (time.perf_counter() - start_time) * 1000
            result.execution_trace.append(
                f"Step {step.step_id} failed: {e}"
            )

            logger.error(
                "step_execution_failed",
                step_id=step.step_id,
                agent_type=step.agent_type.value,
                error=str(e),
            )

            return StepResult(
                step_id=step.step_id,
                agent_type=step.agent_type,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )


# Singleton instance
_supervisor_agent: SupervisorAgent | None = None


async def get_supervisor_agent() -> SupervisorAgent:
    """Get or create supervisor agent singleton."""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = SupervisorAgent()
        await _supervisor_agent.initialize()
    return _supervisor_agent


async def create_execution_plan(analysis: IntentAnalysis) -> ExecutionPlan:
    """Create execution plan from intent analysis.

    Convenience function for quick plan creation.

    Args:
        analysis: IntentAnalysis from IntentAgent

    Returns:
        ExecutionPlan ready for execution
    """
    agent = await get_supervisor_agent()
    return await agent.execute(analysis)
