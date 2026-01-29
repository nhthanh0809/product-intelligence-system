"""Compound Workflow for Multi-Step Queries.

This workflow handles compound queries with multiple intents:
1. Analyze intent using IntentAgent
2. Generate execution plan using SupervisorAgent
3. Execute steps with dependency resolution and parallel execution
4. Aggregate results from multiple agents
5. Synthesize comprehensive response

Example compound queries:
- "Find headphones under $200, compare top 3, and recommend the best"
- "What are trending laptops? Show me the best value options"
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.models.intent import QueryIntent, IntentAnalysis, QueryComplexity
from src.models.execution import (
    AgentType,
    ExecutionPlan,
    ExecutionStep,
    ExecutionResult,
    StepResult,
    StepStatus,
)
from src.agents.intent_agent import IntentAgent, get_intent_agent
from src.agents.supervisor_agent import SupervisorAgent, get_supervisor_agent
from src.agents.general_agent import get_general_agent
from src.agents.search_agent import get_search_agent
from src.agents.compare_agent import CompareInput, get_compare_agent
from src.agents.analysis_agent import AnalysisInput, get_analysis_agent
from src.agents.price_agent import PriceInput, get_price_agent
from src.agents.trend_agent import TrendInput, get_trend_agent
from src.agents.recommend_agent import RecommendInput, get_recommend_agent
from src.agents.synthesis_agent import (
    SynthesisInput,
    SynthesisOutput,
    OutputFormat,
    get_synthesis_agent,
)
from src.workflows.simple_workflow import WorkflowResult

logger = structlog.get_logger()


@dataclass
class CompoundWorkflowResult(WorkflowResult):
    """Extended result for compound workflows."""

    # Execution details
    plan_id: str = ""
    steps_completed: int = 0
    steps_failed: int = 0
    parallel_groups: int = 0

    # Per-step results
    step_results: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        base = super().to_dict()
        base.update({
            "plan_id": self.plan_id,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "parallel_groups": self.parallel_groups,
            "step_results": self.step_results,
        })
        return base


class CompoundWorkflow:
    """Workflow for handling multi-intent compound queries.

    Uses SupervisorAgent to create execution plans and manages
    parallel execution of independent steps with proper dependency
    resolution.

    Example flow for "Find headphones, compare top 3":
    1. IntentAgent analyzes -> SEARCH + COMPARE intents
    2. SupervisorAgent creates plan:
       - Step 1: Retrieval (search headphones)
       - Step 2: Comparison (compare top 3) - depends on Step 1
       - Step 3: Synthesis (generate response) - depends on Steps 1-2
    3. Execute steps in order, respecting dependencies
    4. Return aggregated result
    """

    def __init__(self):
        """Initialize workflow with lazy agent loading."""
        self._intent_agent: IntentAgent | None = None
        self._supervisor_agent: SupervisorAgent | None = None
        # Agent executors for each type
        self._executors: dict[AgentType, Any] = {}

    async def _get_intent_agent(self) -> IntentAgent:
        """Get or create intent agent."""
        if self._intent_agent is None:
            self._intent_agent = await get_intent_agent()
        return self._intent_agent

    async def _get_supervisor_agent(self) -> SupervisorAgent:
        """Get or create supervisor agent."""
        if self._supervisor_agent is None:
            self._supervisor_agent = await get_supervisor_agent()
        return self._supervisor_agent

    async def _get_executor(self, agent_type: AgentType) -> Any:
        """Get agent executor for a given type.

        Args:
            agent_type: Type of agent needed

        Returns:
            Agent instance
        """
        if agent_type not in self._executors:
            if agent_type == AgentType.GENERAL:
                self._executors[agent_type] = await get_general_agent()
            elif agent_type == AgentType.RETRIEVAL:
                self._executors[agent_type] = await get_search_agent()
            elif agent_type == AgentType.COMPARISON:
                self._executors[agent_type] = await get_compare_agent()
            elif agent_type == AgentType.ANALYSIS:
                self._executors[agent_type] = await get_analysis_agent()
            elif agent_type == AgentType.PRICE:
                self._executors[agent_type] = await get_price_agent()
            elif agent_type == AgentType.TREND:
                self._executors[agent_type] = await get_trend_agent()
            elif agent_type == AgentType.RECOMMEND:
                self._executors[agent_type] = await get_recommend_agent()
            elif agent_type == AgentType.SYNTHESIS:
                self._executors[agent_type] = await get_synthesis_agent()

        return self._executors.get(agent_type)

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> CompoundWorkflowResult:
        """Execute compound workflow for a query.

        Args:
            query: User query
            context: Optional conversation context

        Returns:
            CompoundWorkflowResult with aggregated results
        """
        start_time = time.time()
        agents_used: list[str] = []

        try:
            # Step 1: Analyze intent
            intent_agent = await self._get_intent_agent()
            intent_analysis = await intent_agent.execute(query)
            agents_used.append("intent")

            logger.info(
                "compound_intent_analyzed",
                query=query[:50],
                primary=intent_analysis.primary_intent.value,
                secondary=[i.value for i in intent_analysis.secondary_intents],
                complexity=intent_analysis.complexity.value,
            )

            # Step 2: Create execution plan
            supervisor = await self._get_supervisor_agent()
            plan = supervisor.create_plan(intent_analysis)
            agents_used.append("supervisor")

            logger.info(
                "execution_plan_created",
                plan_id=plan.plan_id,
                steps=len(plan.steps),
                is_compound=plan.is_compound,
            )

            # Step 3: Execute plan
            execution_result = await self._execute_plan(
                plan, intent_analysis, context, agents_used
            )

            # Step 4: Build result
            result = self._build_result(
                query=query,
                intent_analysis=intent_analysis,
                plan=plan,
                execution_result=execution_result,
                agents_used=agents_used,
            )

            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            logger.error("compound_workflow_failed", query=query[:50], error=str(e))
            return CompoundWorkflowResult(
                query=query,
                response_text=f"I encountered an error processing your compound request: {str(e)}",
                intent=QueryIntent.SEARCH,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                agents_used=agents_used,
            )

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        intent_analysis: IntentAnalysis,
        context: dict[str, Any] | None,
        agents_used: list[str],
    ) -> ExecutionResult:
        """Execute the plan step by step.

        Args:
            plan: Execution plan from supervisor
            intent_analysis: Original intent analysis
            context: Optional conversation context
            agents_used: List to track agents

        Returns:
            ExecutionResult with all step outputs
        """
        result = ExecutionResult(
            plan_id=plan.plan_id,
            query=plan.query,
            success=False,
        )

        # Track step outputs for passing to dependent steps
        step_outputs: dict[int, Any] = {}

        # Execute steps in parallel groups
        for group in plan.get_parallel_groups():
            group_tasks = []

            for step in group:
                # Gather inputs from previous steps
                step_inputs = self._gather_step_inputs(step, step_outputs)

                # Create execution task
                task = self._execute_step(
                    step=step,
                    intent_analysis=intent_analysis,
                    context=context,
                    step_inputs=step_inputs,
                    agents_used=agents_used,
                )
                group_tasks.append((step, task))

            # Execute group in parallel
            tasks = [t[1] for t in group_tasks]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, (step, _) in enumerate(group_tasks):
                step_result = step_results[i]

                if isinstance(step_result, Exception):
                    # Step failed
                    error_result = StepResult(
                        step_id=step.step_id,
                        agent_type=step.agent_type,
                        success=False,
                        error=str(step_result),
                    )
                    result.add_step_result(error_result)

                    if not step.optional:
                        result.error = f"Required step {step.step_id} failed: {step_result}"
                        return result
                else:
                    # Step succeeded
                    success_result = StepResult(
                        step_id=step.step_id,
                        agent_type=step.agent_type,
                        success=True,
                        output=step_result,
                    )
                    result.add_step_result(success_result)
                    step_outputs[step.step_id] = step_result

        # All steps completed
        result.success = True

        # Get final synthesis output
        synthesis_steps = [s for s in plan.steps if s.agent_type == AgentType.SYNTHESIS]
        if synthesis_steps:
            final_step_id = synthesis_steps[-1].step_id
            result.final_output = step_outputs.get(final_step_id)

        return result

    def _gather_step_inputs(
        self,
        step: ExecutionStep,
        step_outputs: dict[int, Any],
    ) -> dict[str, Any]:
        """Gather inputs from previous steps.

        Args:
            step: Current step
            step_outputs: Outputs from completed steps

        Returns:
            Dict of inputs for this step
        """
        inputs: dict[str, Any] = {}

        for input_step_id in step.input_from_steps:
            if input_step_id in step_outputs:
                output = step_outputs[input_step_id]

                # Merge output based on type
                if isinstance(output, dict):
                    inputs.update(output)
                elif isinstance(output, list):
                    if "products" not in inputs:
                        inputs["products"] = []
                    inputs["products"].extend(output)
                else:
                    inputs[f"step_{input_step_id}_output"] = output

        return inputs

    async def _execute_step(
        self,
        step: ExecutionStep,
        intent_analysis: IntentAnalysis,
        context: dict[str, Any] | None,
        step_inputs: dict[str, Any],
        agents_used: list[str],
    ) -> Any:
        """Execute a single step.

        Args:
            step: Step to execute
            intent_analysis: Original intent analysis
            context: Optional conversation context
            step_inputs: Inputs from previous steps
            agents_used: List to track agents

        Returns:
            Step output
        """
        agent_type = step.agent_type
        step_start = time.time()

        logger.info(
            "step_execution_started",
            step_id=step.step_id,
            agent_type=agent_type.value,
            description=step.description,
            input_from_steps=step.input_from_steps,
        )

        agent = await self._get_executor(agent_type)

        if agent is None:
            logger.error("step_executor_not_found", agent_type=agent_type.value)
            raise ValueError(f"No executor for agent type: {agent_type}")

        agents_used.append(agent_type.value)

        # Execute based on agent type
        if agent_type == AgentType.GENERAL:
            result = await agent.execute(intent_analysis)
            return {"response_text": result.response_text, "suggestions": result.suggestions}

        elif agent_type == AgentType.RETRIEVAL:
            query = step.input_params.get("query", intent_analysis.query)
            state = await agent.search(query)
            return {
                "products": state.formatted_results,
                "total": state.total_results,
                "summary": state.summary,
            }

        elif agent_type == AgentType.COMPARISON:
            products = step_inputs.get("products", [])
            compare_input = CompareInput(
                query=intent_analysis.query,
                products=products[:5],  # Compare up to 5 products
            )
            result = await agent.execute(compare_input)

            # Build winner dict for synthesis agent (expects dict with "name" key)
            winner_dict = None
            if result.comparison.winner:
                winner_dict = {
                    "name": result.comparison.winner.title,
                    "asin": result.comparison.winner.asin,
                    "price": result.comparison.winner.price,
                }

            return {
                "comparison": {
                    "products": [
                        {
                            "asin": p.asin,
                            "title": p.title,
                            "brand": p.brand,
                            "price": p.price,
                            "stars": p.stars,
                            "pros": p.pros,
                            "cons": p.cons,
                        }
                        for p in result.comparison.products
                    ],
                    "winner": winner_dict,
                    "winner_reason": result.comparison.winner_reason,
                    "key_differences": result.comparison.key_differences,
                    "differences": result.comparison.key_differences,  # Alias for synthesis agent
                    "summary": result.comparison.comparison_summary,
                },
                "products": step_inputs.get("products", []),
            }

        elif agent_type == AgentType.ANALYSIS:
            analysis_input = AnalysisInput(
                query=intent_analysis.query,
                analysis_type="general",
            )
            result = await agent.execute(analysis_input)
            return {
                "analysis": {
                    "pros": result.analysis.pros,
                    "cons": result.analysis.cons,
                    "sentiment_score": result.analysis.sentiment_score,
                    "summary": result.analysis.summary,
                },
                "products": step_inputs.get("products", []),
            }

        elif agent_type == AgentType.PRICE:
            target_price = intent_analysis.entities.constraints.get("price_max")
            price_input = PriceInput(
                query=intent_analysis.query,
                target_price=target_price,
            )
            result = await agent.execute(price_input)
            return {
                "price_analysis": {
                    "products": [
                        {
                            "asin": p.asin,
                            "title": p.title,
                            "current_price": p.current_price,
                            "discount_pct": p.discount_pct,
                            "value_score": p.value_score,
                        }
                        for p in result.products
                    ],
                    "best_deal": result.best_deal.title if result.best_deal else None,
                    "recommendation": result.recommendation,
                    "summary": result.summary,
                },
                "products": step_inputs.get("products", []),
            }

        elif agent_type == AgentType.TREND:
            category = (
                intent_analysis.entities.categories[0]
                if intent_analysis.entities.categories
                else None
            )
            trend_input = TrendInput(
                query=intent_analysis.query,
                category=category,
            )
            result = await agent.execute(trend_input)
            return {
                "trends": {
                    "trending_products": [
                        {
                            "asin": p.asin,
                            "title": p.title,
                            "trend_score": p.trend_score,
                            "bought_in_last_month": p.bought_in_last_month,
                        }
                        for p in result.trending_products[:10]
                    ],
                    "hot_categories": result.hot_categories,
                    "insights": result.insights,
                    "summary": result.summary,
                },
                "products": step_inputs.get("products", []),
            }

        elif agent_type == AgentType.RECOMMEND:
            recommend_input = RecommendInput(
                query=intent_analysis.query,
                recommendation_type="similar",
            )
            result = await agent.execute(recommend_input)
            return {
                "recommendations": [
                    {
                        "asin": r.asin,
                        "title": r.title,
                        "brand": r.brand,
                        "price": r.price,
                        "overall_score": r.overall_score,
                        "rank": r.rank,
                        "reason": r.recommendation_reason,
                    }
                    for r in result.recommendations
                ],
                "top_pick": {
                    "asin": result.top_pick.asin,
                    "title": result.top_pick.title,
                } if result.top_pick else None,
                "reasoning": result.reasoning,
                "products": step_inputs.get("products", []),
            }

        elif agent_type == AgentType.SYNTHESIS:
            # Gather all data from previous steps
            products = step_inputs.get("products", [])
            comparison = step_inputs.get("comparison")
            analysis = step_inputs.get("analysis")
            price_analysis = step_inputs.get("price_analysis")
            trends = step_inputs.get("trends")
            recommendations = step_inputs.get("recommendations")

            logger.info(
                "synthesis_step_inputs",
                num_products=len(products),
                has_comparison=comparison is not None,
                has_analysis=analysis is not None,
                has_price_analysis=price_analysis is not None,
                has_trends=trends is not None,
                has_recommendations=recommendations is not None,
            )

            synthesis_input = SynthesisInput(
                query=intent_analysis.query,
                intent=intent_analysis.primary_intent,
                products=products,
                comparison=comparison,
                analysis=analysis,
                price_analysis=price_analysis,
                trends=trends,
                recommendations=recommendations,
            )

            logger.info("synthesis_agent_calling")
            result = await agent.execute(synthesis_input)
            logger.info(
                "synthesis_agent_completed",
                response_length=len(result.response_text),
                format=result.format.value,
                confidence=result.confidence,
            )

            return {
                "response_text": result.response_text,
                "format": result.format.value,
                "products": result.products,
                "confidence": result.confidence,
                "suggestions": result.suggestions,
            }

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _build_result(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        plan: ExecutionPlan,
        execution_result: ExecutionResult,
        agents_used: list[str],
    ) -> CompoundWorkflowResult:
        """Build final workflow result.

        Args:
            query: Original query
            intent_analysis: Intent analysis
            plan: Execution plan
            execution_result: Plan execution result
            agents_used: List of agents used

        Returns:
            CompoundWorkflowResult
        """
        # Get final output
        final_output = execution_result.final_output or {}

        # Extract response text
        response_text = final_output.get(
            "response_text",
            "I processed your request but couldn't generate a response."
        )

        # Extract products from final output or step results
        products = final_output.get("products", [])

        # Extract other data
        comparison = None
        analysis = None
        recommendations = None

        for step_id, step_result in execution_result.step_results.items():
            if step_result.output:
                output = step_result.output
                if "comparison" in output and not comparison:
                    comparison = output["comparison"]
                if "analysis" in output and not analysis:
                    analysis = output["analysis"]
                if "recommendations" in output and not recommendations:
                    recommendations = output["recommendations"]
                if "products" in output and not products:
                    products = output["products"]

        # Build step results summary
        step_results_summary = {}
        for step_id, step_result in execution_result.step_results.items():
            step_results_summary[step_id] = {
                "agent": step_result.agent_type.value,
                "success": step_result.success,
                "duration_ms": step_result.duration_ms,
                "error": step_result.error,
            }

        return CompoundWorkflowResult(
            query=query,
            response_text=response_text,
            intent=intent_analysis.primary_intent,
            products=products,
            comparison=comparison,
            analysis=analysis,
            recommendations=recommendations,
            format=OutputFormat(final_output.get("format", "text")),
            confidence=final_output.get("confidence", 0.0),
            suggestions=final_output.get("suggestions", []),
            agents_used=agents_used,
            error=execution_result.error,
            # Compound-specific fields
            plan_id=plan.plan_id,
            steps_completed=execution_result.steps_completed,
            steps_failed=execution_result.steps_failed,
            parallel_groups=len(plan.get_parallel_groups()),
            step_results=step_results_summary,
        )


# Singleton instance
_compound_workflow: CompoundWorkflow | None = None


async def get_compound_workflow() -> CompoundWorkflow:
    """Get or create compound workflow singleton."""
    global _compound_workflow
    if _compound_workflow is None:
        _compound_workflow = CompoundWorkflow()
    return _compound_workflow


async def execute_compound_query(
    query: str,
    context: dict[str, Any] | None = None,
) -> CompoundWorkflowResult:
    """Convenience function to execute a compound query.

    Args:
        query: User query
        context: Optional conversation context

    Returns:
        CompoundWorkflowResult
    """
    workflow = await get_compound_workflow()
    return await workflow.execute(query, context)
