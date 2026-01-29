"""
Agent implementations for Product Intelligence System.

This module provides:
1. SearchAgent - Product discovery using optimized search strategies
2. AnalysisAgent - Review and trend analysis
3. QAAgent - Question answering about products
4. Multi-agent workflow orchestration via LangGraph
"""

from typing import Any, TypedDict

import httpx
import structlog
from langgraph.graph import StateGraph, END

from src.config import get_settings
from src.tools import ALL_TOOLS

# Import search agent
from src.agents.search_agent import (
    SearchAgent,
    SearchAgentState,
    get_search_agent,
    search,
    search_products,
    FilterExtractor,
    IntentClassifier,
    SearchIntent,
)

logger = structlog.get_logger()
settings = get_settings()


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    """State passed between agents in the graph."""
    query: str
    query_type: str | None
    search_results: list[dict]
    analysis: str | None
    answer: str | None
    products: list[dict]
    error: str | None
    steps: list[str]
    # Extended fields from search agent
    search_state: dict | None
    latency_ms: float
    filters: dict | None


# =============================================================================
# Agent Nodes
# =============================================================================

async def classify_query(state: AgentState) -> AgentState:
    """Classify the user query to determine routing."""
    query = state["query"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_service_url}/generate",
            json={
                "prompt": f"""Classify this product query into one category:
- search: Looking for products (finding, browsing, discovering)
- comparison: Comparing products (vs, difference, which is better)
- analysis: Analyzing trends, statistics, or reviews
- qa: Asking a specific question about a product

Query: {query}

Output only the category name.""",
                "temperature": 0.1,
                "num_predict": 20,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        query_type = response.json().get("response", "search").strip().lower()

    # Validate query type
    if query_type not in ["search", "comparison", "analysis", "qa"]:
        query_type = "search"

    state["query_type"] = query_type
    state["steps"].append(f"classified_query:{query_type}")

    logger.debug("query_classified", query_type=query_type)
    return state


async def search_agent_node(state: AgentState) -> AgentState:
    """Execute search using the optimized SearchAgent."""
    query = state["query"]

    try:
        # Use the new SearchAgent
        agent = await get_search_agent()
        search_state = await agent.search(query)

        # Transfer results to main state
        state["search_results"] = [
            {"payload": r} for r in search_state.search_results
        ]
        state["products"] = search_state.formatted_results
        state["latency_ms"] = search_state.latency_ms
        state["filters"] = search_state.filters.model_dump() if search_state.filters else None
        state["search_state"] = {
            "intent": search_state.intent.value if search_state.intent else None,
            "query_type": search_state.query_type.value if search_state.query_type else None,
            "strategy": search_state.search_strategy,
            "summary": search_state.summary,
        }

        state["steps"].extend(search_state.steps)
        state["steps"].append(f"search_completed:{len(search_state.formatted_results)}_results")

        if search_state.error:
            state["error"] = search_state.error

    except Exception as e:
        logger.error("search_agent_failed", error=str(e))
        state["error"] = str(e)
        state["search_results"] = []
        state["products"] = []

    return state


async def analysis_agent(state: AgentState) -> AgentState:
    """Analyze search results or perform statistical analysis."""
    results = state.get("products", [])

    if not results:
        state["analysis"] = "No products found to analyze."
        return state

    # Build analysis prompt
    products_text = "\n".join([
        f"- {r.get('title', 'Unknown')[:60]} | ${r.get('price', 'N/A')} | {r.get('stars', 'N/A')}★ | {r.get('brand', 'Unknown')}"
        for r in results[:10]
    ])

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_service_url}/generate",
            json={
                "prompt": f"""Analyze these products based on the user query:

Query: {state['query']}

Products:
{products_text}

Provide a concise analysis including:
1. Price range and value assessment
2. Top-rated options
3. Brand representation
4. Key recommendation""",
                "temperature": 0.7,
                "num_predict": 400,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        state["analysis"] = response.json().get("response", "")

    state["steps"].append("analysis_completed")
    return state


async def qa_agent(state: AgentState) -> AgentState:
    """Answer questions about products."""
    results = state.get("products", [])
    query = state["query"]

    # Build context from results
    context_parts = []
    for r in results[:5]:
        context_parts.append(
            f"Product: {r.get('title', 'Unknown')}\n"
            f"Brand: {r.get('brand', 'Unknown')}\n"
            f"Price: ${r.get('price', 'N/A')}\n"
            f"Rating: {r.get('stars', 'N/A')}/5\n"
            f"Summary: {r.get('summary', 'N/A')}"
        )
    context = "\n\n".join(context_parts)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_service_url}/generate",
            json={
                "prompt": f"""Based on the following product information, answer the user's question.

Context:
{context}

Question: {query}

Provide a helpful, accurate answer based on the information provided. If the information is not sufficient, say so.""",
                "temperature": 0.7,
                "num_predict": 400,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        state["answer"] = response.json().get("response", "")

    state["steps"].append("qa_completed")
    return state


async def comparison_agent(state: AgentState) -> AgentState:
    """Compare products side by side."""
    results = state.get("products", [])

    if len(results) < 2:
        state["analysis"] = "Need at least 2 products to compare."
        return state

    # Build comparison prompt
    products_text = "\n\n".join([
        f"Product {i+1}: {r.get('title', 'Unknown')[:60]}\n"
        f"  Brand: {r.get('brand', 'Unknown')}\n"
        f"  Price: ${r.get('price', 'N/A')}\n"
        f"  Rating: {r.get('stars', 'N/A')}/5\n"
        f"  Best for: {r.get('best_for', 'N/A')}"
        for i, r in enumerate(results[:4])
    ])

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_service_url}/generate",
            json={
                "prompt": f"""Compare these products:

{products_text}

Query: {state['query']}

Provide:
1. Key differences between products
2. Best value option
3. Best overall option
4. Recommendation based on the query""",
                "temperature": 0.7,
                "num_predict": 500,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        state["analysis"] = response.json().get("response", "")

    state["steps"].append("comparison_completed")
    return state


async def synthesize_response(state: AgentState) -> AgentState:
    """Synthesize final response from agent outputs."""
    # Products should already be formatted from search agent
    if not state.get("products"):
        # Fallback: extract from search_results
        products = []
        for r in state.get("search_results", [])[:10]:
            payload = r.get("payload", {})
            if payload:
                products.append({
                    "asin": payload.get("asin"),
                    "title": payload.get("title"),
                    "brand": payload.get("brand"),
                    "price": payload.get("price"),
                    "stars": payload.get("stars"),
                    "category": payload.get("category_level1"),
                })
        state["products"] = products

    state["steps"].append("response_synthesized")
    return state


# =============================================================================
# Routing Functions
# =============================================================================

def route_by_query_type(state: AgentState) -> str:
    """Route to appropriate agent based on query type."""
    query_type = state.get("query_type", "search")

    if query_type == "comparison":
        return "search"  # Search first, then compare
    elif query_type == "analysis":
        return "search"  # Search first, then analyze
    elif query_type == "qa":
        return "search"  # Search first for context
    else:
        return "search"


def route_after_search(state: AgentState) -> str:
    """Route after search based on query type."""
    query_type = state.get("query_type", "search")

    if query_type == "comparison":
        return "comparison"
    elif query_type == "analysis":
        return "analysis"
    elif query_type == "qa":
        return "qa"
    else:
        return "synthesize"


def should_continue(state: AgentState) -> str:
    """Determine if we should continue processing."""
    if state.get("error"):
        return "end"
    if len(state.get("steps", [])) > settings.max_agent_steps:
        return "end"
    return "continue"


# =============================================================================
# Graph Construction
# =============================================================================

def create_agent_graph() -> StateGraph:
    """Create the multi-agent workflow graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("comparison", comparison_agent)
    workflow.add_node("qa", qa_agent)
    workflow.add_node("synthesize", synthesize_response)

    # Set entry point
    workflow.set_entry_point("classify")

    # Add edges from classify
    workflow.add_conditional_edges(
        "classify",
        route_by_query_type,
        {
            "search": "search",
        },
    )

    # Add edges after search
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {
            "analysis": "analysis",
            "comparison": "comparison",
            "qa": "qa",
            "synthesize": "synthesize",
        },
    )

    # Final edges
    workflow.add_edge("analysis", "synthesize")
    workflow.add_edge("comparison", "synthesize")
    workflow.add_edge("qa", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# Create global graph instance
agent_graph = create_agent_graph()


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_agent(query: str) -> dict[str, Any]:
    """
    Run the multi-agent system on a query.

    Args:
        query: User query

    Returns:
        Agent response with products, analysis, and answer
    """
    initial_state: AgentState = {
        "query": query,
        "query_type": None,
        "search_results": [],
        "analysis": None,
        "answer": None,
        "products": [],
        "error": None,
        "steps": [],
        "search_state": None,
        "latency_ms": 0.0,
        "filters": None,
    }

    try:
        result = await agent_graph.ainvoke(initial_state)

        return {
            "query": query,
            "query_type": result.get("query_type"),
            "products": result.get("products", []),
            "analysis": result.get("analysis"),
            "answer": result.get("answer"),
            "steps": result.get("steps", []),
            "error": result.get("error"),
            "search_state": result.get("search_state"),
            "latency_ms": result.get("latency_ms", 0.0),
            "filters": result.get("filters"),
        }
    except Exception as e:
        logger.error("agent_execution_failed", error=str(e))
        return {
            "query": query,
            "error": str(e),
            "products": [],
            "steps": ["error"],
        }


# =============================================================================
# Import new specialist agents
# =============================================================================

from src.agents.base import BaseAgent, RetryConfig, CircuitBreaker, AgentMetrics
from src.agents.orchestrator import (
    OrchestratorAgent,
    get_orchestrator,
    route_query,
    AgentType,
    RoutingResult,
)
from src.agents.analysis_agent import (
    AnalysisAgent,
    get_analysis_agent,
    analyze_reviews,
    AnalysisInput,
    AnalysisOutput,
)
from src.agents.compare_agent import (
    CompareAgent,
    get_compare_agent,
    compare_products,
    CompareInput,
    CompareOutput,
)
from src.agents.price_agent import (
    PriceAgent,
    get_price_agent,
    analyze_prices,
    PriceInput,
    PriceOutput,
)
from src.agents.trend_agent import (
    TrendAgent,
    get_trend_agent,
    get_trends,
    TrendInput,
    TrendOutput,
)
from src.agents.recommend_agent import (
    RecommendAgent,
    get_recommend_agent,
    get_recommendations,
    RecommendInput,
    RecommendOutput,
)

# =============================================================================
# Import Intent Agent
# =============================================================================

from src.agents.intent_agent import (
    IntentAgent,
    get_intent_agent,
    analyze_intent,
)
from src.agents.general_agent import (
    GeneralAgent,
    GeneralResponse,
    get_general_agent,
    handle_general_chat,
)
from src.agents.supervisor_agent import (
    SupervisorAgent,
    get_supervisor_agent,
    create_execution_plan,
)
from src.agents.synthesis_agent import (
    SynthesisAgent,
    SynthesisInput,
    SynthesisOutput,
    OutputFormat,
    get_synthesis_agent,
    synthesize_response,
)
from src.agents.attribute_agent import (
    AttributeAgent,
    AttributeType,
    ExtractedAttribute,
    ProductAttributes,
    AttributeExtractionInput,
    AttributeExtractionOutput,
    get_attribute_agent,
    extract_attributes,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main entry points
    "run_agent",
    "agent_graph",
    "create_agent_graph",
    # Intent agent (NEW)
    "IntentAgent",
    "get_intent_agent",
    "analyze_intent",
    # General agent
    "GeneralAgent",
    "GeneralResponse",
    "get_general_agent",
    "handle_general_chat",
    # Supervisor agent
    "SupervisorAgent",
    "get_supervisor_agent",
    "create_execution_plan",
    # Synthesis agent
    "SynthesisAgent",
    "SynthesisInput",
    "SynthesisOutput",
    "OutputFormat",
    "get_synthesis_agent",
    "synthesize_response",
    # Attribute agent
    "AttributeAgent",
    "AttributeType",
    "ExtractedAttribute",
    "ProductAttributes",
    "AttributeExtractionInput",
    "AttributeExtractionOutput",
    "get_attribute_agent",
    "extract_attributes",
    # Search agent
    "SearchAgent",
    "SearchAgentState",
    "get_search_agent",
    "search",
    "search_products",
    # Utilities
    "FilterExtractor",
    "IntentClassifier",
    "SearchIntent",
    # State
    "AgentState",
    # Base classes
    "BaseAgent",
    "RetryConfig",
    "CircuitBreaker",
    "AgentMetrics",
    # Orchestrator
    "OrchestratorAgent",
    "get_orchestrator",
    "route_query",
    "AgentType",
    "RoutingResult",
    # Analysis agent
    "AnalysisAgent",
    "get_analysis_agent",
    "analyze_reviews",
    "AnalysisInput",
    "AnalysisOutput",
    # Compare agent
    "CompareAgent",
    "get_compare_agent",
    "compare_products",
    "CompareInput",
    "CompareOutput",
    # Price agent
    "PriceAgent",
    "get_price_agent",
    "analyze_prices",
    "PriceInput",
    "PriceOutput",
    # Trend agent
    "TrendAgent",
    "get_trend_agent",
    "get_trends",
    "TrendInput",
    "TrendOutput",
    # Recommend agent
    "RecommendAgent",
    "get_recommend_agent",
    "get_recommendations",
    "RecommendInput",
    "RecommendOutput",
]
