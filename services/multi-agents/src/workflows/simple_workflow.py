"""Simple Workflow for Single-Intent Queries.

This workflow handles queries with a single, clear intent:
1. Analyze intent using IntentAgent
2. Route to appropriate agent based on intent
3. Synthesize response using SynthesisAgent
4. Return formatted result

Handles intents: search, compare, analyze, price_check, trend, recommend,
greeting, farewell, help, small_talk, off_topic
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.models.intent import QueryIntent, IntentAnalysis
from src.agents.intent_agent import IntentAgent, get_intent_agent
from src.agents.general_agent import GeneralAgent, GeneralResponse, get_general_agent
from src.agents.search_agent import get_search_agent
from src.agents.compare_agent import CompareAgent, CompareInput, get_compare_agent
from src.agents.analysis_agent import AnalysisAgent, AnalysisInput, get_analysis_agent
from src.agents.price_agent import PriceAgent, PriceInput, get_price_agent
from src.agents.trend_agent import TrendAgent, TrendInput, get_trend_agent
from src.agents.recommend_agent import RecommendAgent, RecommendInput, get_recommend_agent
from src.agents.synthesis_agent import (
    SynthesisAgent,
    SynthesisInput,
    SynthesisOutput,
    OutputFormat,
    get_synthesis_agent,
)

logger = structlog.get_logger()


@dataclass
class WorkflowResult:
    """Result from workflow execution."""

    # Core response
    query: str
    response_text: str
    intent: QueryIntent

    # Products and data
    products: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    recommendations: list[dict[str, Any]] | None = None

    # Metadata
    format: OutputFormat = OutputFormat.TEXT
    confidence: float = 0.0
    suggestions: list[str] = field(default_factory=list)

    # Execution info
    execution_time_ms: float = 0.0
    agents_used: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "query": self.query,
            "response": self.response_text,
            "intent": self.intent.value,
            "products": self.products,
            "comparison": self.comparison,
            "analysis": self.analysis,
            "recommendations": self.recommendations,
            "format": self.format.value,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "execution_time_ms": self.execution_time_ms,
            "agents_used": self.agents_used,
            "error": self.error,
        }


class SimpleWorkflow:
    """Workflow for handling single-intent queries.

    Routes queries to the appropriate agent based on intent:
    - General intents (greeting, help, etc.) -> GeneralAgent
    - Search intent -> SearchAgent
    - Compare intent -> CompareAgent
    - Analyze intent -> AnalysisAgent
    - Price intent -> PriceAgent
    - Trend intent -> TrendAgent
    - Recommend intent -> RecommendAgent

    All product-related queries get their results synthesized by SynthesisAgent.
    """

    def __init__(self):
        """Initialize workflow with lazy agent loading."""
        self._intent_agent: IntentAgent | None = None
        self._general_agent: GeneralAgent | None = None
        self._synthesis_agent: SynthesisAgent | None = None

    async def _get_intent_agent(self) -> IntentAgent:
        """Get or create intent agent."""
        if self._intent_agent is None:
            self._intent_agent = await get_intent_agent()
        return self._intent_agent

    async def _get_general_agent(self) -> GeneralAgent:
        """Get or create general agent."""
        if self._general_agent is None:
            self._general_agent = await get_general_agent()
        return self._general_agent

    async def _get_synthesis_agent(self) -> SynthesisAgent:
        """Get or create synthesis agent."""
        if self._synthesis_agent is None:
            self._synthesis_agent = await get_synthesis_agent()
        return self._synthesis_agent

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute workflow for a query.

        Args:
            query: User query
            context: Optional conversation context

        Returns:
            WorkflowResult with response and metadata
        """
        start_time = time.time()
        agents_used: list[str] = []

        try:
            # Step 1: Analyze intent
            intent_agent = await self._get_intent_agent()
            intent_analysis = await intent_agent.execute(query)
            agents_used.append("intent")

            logger.info(
                "intent_analyzed",
                query=query[:50],
                intent=intent_analysis.primary_intent.value,
                confidence=intent_analysis.confidence,
            )

            # Step 2: Route based on intent
            if intent_analysis.is_general_chat:
                result = await self._handle_general_chat(
                    query, intent_analysis, agents_used
                )
            else:
                result = await self._handle_product_query(
                    query, intent_analysis, context, agents_used
                )

            # Update execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            result.agents_used = agents_used

            return result

        except Exception as e:
            logger.error("workflow_failed", query=query[:50], error=str(e))
            return WorkflowResult(
                query=query,
                response_text=f"I encountered an error processing your request: {str(e)}",
                intent=QueryIntent.SEARCH,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                agents_used=agents_used,
            )

    async def _handle_general_chat(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> WorkflowResult:
        """Handle general conversation queries.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            WorkflowResult
        """
        general_agent = await self._get_general_agent()
        response = await general_agent.execute(intent_analysis)
        agents_used.append("general")

        return WorkflowResult(
            query=query,
            response_text=response.response_text,
            intent=intent_analysis.primary_intent,
            suggestions=response.suggestions,
            confidence=intent_analysis.confidence,
            format=OutputFormat.TEXT,
        )

    async def _handle_product_query(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        context: dict[str, Any] | None,
        agents_used: list[str],
    ) -> WorkflowResult:
        """Handle product-related queries.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            context: Optional conversation context
            agents_used: List to track agents

        Returns:
            WorkflowResult
        """
        intent = intent_analysis.primary_intent
        products: list[dict[str, Any]] = []
        comparison: dict[str, Any] | None = None
        analysis: dict[str, Any] | None = None
        price_analysis: dict[str, Any] | None = None
        trends: dict[str, Any] | None = None
        recommendations: list[dict[str, Any]] | None = None

        # Route to appropriate agent
        if intent == QueryIntent.SEARCH:
            products = await self._execute_search(query, intent_analysis, agents_used)

        elif intent == QueryIntent.COMPARE:
            products, comparison = await self._execute_compare(
                query, intent_analysis, agents_used
            )

        elif intent == QueryIntent.ANALYZE:
            products, analysis = await self._execute_analysis(
                query, intent_analysis, agents_used
            )

        elif intent == QueryIntent.PRICE_CHECK:
            products, price_analysis = await self._execute_price_check(
                query, intent_analysis, agents_used
            )

        elif intent == QueryIntent.TREND:
            products, trends = await self._execute_trend(
                query, intent_analysis, agents_used
            )

        elif intent == QueryIntent.RECOMMEND:
            products, recommendations = await self._execute_recommend(
                query, intent_analysis, agents_used
            )

        else:
            # Default to search
            products = await self._execute_search(query, intent_analysis, agents_used)

        # Step 3: Synthesize response
        synthesis_agent = await self._get_synthesis_agent()
        synthesis_input = SynthesisInput(
            query=query,
            intent=intent,
            products=products,
            comparison=comparison,
            analysis=analysis,
            price_analysis=price_analysis,
            trends=trends,
            recommendations=recommendations,
        )

        synthesis_output = await synthesis_agent.execute(synthesis_input)
        agents_used.append("synthesis")

        return WorkflowResult(
            query=query,
            response_text=synthesis_output.response_text,
            intent=intent,
            products=synthesis_output.products,
            comparison=comparison,
            analysis=analysis,
            recommendations=recommendations,
            format=synthesis_output.format,
            confidence=synthesis_output.confidence,
            suggestions=synthesis_output.suggestions,
        )

    async def _execute_search(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> list[dict[str, Any]]:
        """Execute search query.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            List of products
        """
        logger.info("simple_workflow_execute_search", query=query[:30])
        search_agent = await get_search_agent()
        state = await search_agent.search(query)
        agents_used.append("search")

        return state.formatted_results

    async def _execute_compare(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute comparison query.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            Tuple of (products, comparison_data)
        """
        compare_agent = await get_compare_agent()

        # Extract product names from entities
        product_names = intent_analysis.entities.products

        compare_input = CompareInput(
            query=query,
            product_names=product_names,
        )

        result = await compare_agent.execute(compare_input)
        agents_used.append("compare")

        # Format products
        products = [
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
        ]

        # Build winner/best_value dicts for synthesis agent (expects dict with "name" key)
        winner_dict = None
        if result.comparison.winner:
            winner_dict = {
                "name": result.comparison.winner.title,
                "asin": result.comparison.winner.asin,
                "price": result.comparison.winner.price,
            }

        best_value_dict = None
        if result.comparison.best_value:
            best_value_dict = {
                "name": result.comparison.best_value.title,
                "asin": result.comparison.best_value.asin,
                "price": result.comparison.best_value.price,
            }

        # Format comparison data
        comparison = {
            "winner": winner_dict,
            "winner_reason": result.comparison.winner_reason,
            "best_value": best_value_dict,
            "key_differences": result.comparison.key_differences,
            "differences": result.comparison.key_differences,  # Alias for synthesis agent
            "summary": result.comparison.comparison_summary,
            "matrix": result.comparison.comparison_matrix,
        }

        return products, comparison

    async def _execute_analysis(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute analysis query.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            Tuple of (products, analysis_data)
        """
        analysis_agent = await get_analysis_agent()

        analysis_input = AnalysisInput(
            query=query,
            analysis_type="general",
        )

        result = await analysis_agent.execute(analysis_input)
        agents_used.append("analysis")

        # Format analysis data
        analysis = {
            "pros": result.analysis.pros,
            "cons": result.analysis.cons,
            "sentiment_score": result.analysis.sentiment_score,
            "sentiment_label": result.analysis.sentiment_label,
            "common_themes": result.analysis.common_themes,
            "summary": result.analysis.summary,
        }

        # Get products that were analyzed
        products = [
            {
                "asin": p.get("asin", f"product_{i}"),
                "title": p.get("title", f"Product {i+1}"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "stars": p.get("stars"),
            }
            for i, p in enumerate(result.source_products[:5])
        ]

        return products, analysis

    async def _execute_price_check(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute price check query.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            Tuple of (products, price_analysis)
        """
        price_agent = await get_price_agent()

        # Extract target price from constraints
        target_price = intent_analysis.entities.constraints.get("price_max")

        price_input = PriceInput(
            query=query,
            target_price=target_price,
        )

        result = await price_agent.execute(price_input)
        agents_used.append("price")

        # Format products with price data
        products = [
            {
                "asin": p.asin,
                "title": p.title,
                "current_price": p.current_price,
                "list_price": p.list_price,
                "discount_pct": p.discount_pct,
                "price_rating": p.price_rating,
                "value_score": p.value_score,
            }
            for p in result.products
        ]

        # Format price analysis
        price_analysis = {
            "best_deal": {
                "title": result.best_deal.title,
                "discount": result.best_deal.discount_pct,
            } if result.best_deal else None,
            "recommendation": result.recommendation,
            "summary": result.summary,
        }

        return products, price_analysis

    async def _execute_trend(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute trend query.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            Tuple of (products, trends_data)
        """
        trend_agent = await get_trend_agent()

        # Extract category from entities
        category = (
            intent_analysis.entities.categories[0]
            if intent_analysis.entities.categories
            else None
        )

        trend_input = TrendInput(
            query=query,
            category=category,
        )

        result = await trend_agent.execute(trend_input)
        agents_used.append("trend")

        # Format trending products
        products = [
            {
                "asin": p.asin,
                "title": p.title,
                "brand": p.brand,
                "trend_score": p.trend_score,
                "bought_in_last_month": p.bought_in_last_month,
                "price": p.price,
                "stars": p.stars,
            }
            for p in result.trending_products[:10]
        ]

        # Format trends data
        trends = {
            "hot_categories": result.hot_categories,
            "insights": result.insights,
            "summary": result.summary,
        }

        return products, trends

    async def _execute_recommend(
        self,
        query: str,
        intent_analysis: IntentAnalysis,
        agents_used: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Execute recommendation query.

        Args:
            query: User query
            intent_analysis: Analyzed intent
            agents_used: List to track agents

        Returns:
            Tuple of (products, recommendations)
        """
        recommend_agent = await get_recommend_agent()

        recommend_input = RecommendInput(
            query=query,
            recommendation_type="similar",
        )

        result = await recommend_agent.execute(recommend_input)
        agents_used.append("recommend")

        # Format products
        products = [
            {
                "asin": r.asin,
                "title": r.title,
                "brand": r.brand,
                "price": r.price,
                "stars": r.stars,
                "similarity_score": r.similarity_score,
                "reason": r.recommendation_reason,
            }
            for r in result.recommendations
        ]

        # Format recommendations with picks
        recommendations = [
            {
                "asin": r.asin,
                "title": r.title,
                "brand": r.brand,
                "price": r.price,
                "stars": r.stars,
                "overall_score": r.overall_score,
                "rank": r.rank,
                "reason": r.recommendation_reason,
            }
            for r in result.recommendations
        ]

        return products, recommendations


# Singleton instance
_simple_workflow: SimpleWorkflow | None = None


async def get_simple_workflow() -> SimpleWorkflow:
    """Get or create simple workflow singleton."""
    global _simple_workflow
    if _simple_workflow is None:
        _simple_workflow = SimpleWorkflow()
    return _simple_workflow


async def execute_simple_query(
    query: str,
    context: dict[str, Any] | None = None,
) -> WorkflowResult:
    """Convenience function to execute a simple query.

    Args:
        query: User query
        context: Optional conversation context

    Returns:
        WorkflowResult
    """
    workflow = await get_simple_workflow()
    return await workflow.execute(query, context)
