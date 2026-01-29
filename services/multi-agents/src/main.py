"""Main FastAPI application for Multi-Agent Service."""

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.agents import run_agent
from src.middleware.correlation import CorrelationIdMiddleware, add_correlation_id
from src.middleware.rate_limit import RateLimitMiddleware, RateLimiter
from src.middleware.error_handler import ErrorHandlerMiddleware
from src.routers import config_router

# Configure Python's standard logging to allow INFO level
import logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,  # Add correlation ID to all logs
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Cache instance for lifespan management
_cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _cache
    settings = get_settings()
    logger.info("starting_service", service=settings.service_name, port=settings.service_port)

    # Initialize cache
    from src.cache import get_cache
    _cache = await get_cache()
    logger.info("cache_initialized")

    # Initialize database pool for config management
    import asyncpg
    from redis.asyncio import Redis
    from src.config import get_config_repository, get_config_manager

    db_pool = None
    redis_client = None
    try:
        db_pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            min_size=2,
            max_size=10,
        )
        logger.info("database_pool_initialized")

        redis_client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        logger.info("redis_client_initialized")

        # Initialize config repository and manager
        config_repo = await get_config_repository(db_pool, redis_client)
        await get_config_manager(db_pool, redis_client)
        logger.info("config_manager_initialized")

        # Initialize LLM Provider Manager
        from src.llm.manager import LLMProviderManager, get_llm_manager
        llm_manager = await get_llm_manager(config_repo)
        logger.info(
            "llm_manager_initialized",
            providers=len(llm_manager._providers),
            agents=len(llm_manager._agent_configs),
        )
    except Exception as e:
        logger.warning("config_db_init_failed", error=str(e))
        # Continue without config DB - will use defaults

    yield

    # Cleanup
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    if _cache:
        await _cache.close()
    logger.info("shutting_down_service", service=settings.service_name)


settings = get_settings()

app = FastAPI(
    title="Multi-Agent Service",
    description="LangGraph-based multi-agent orchestration for Product Intelligence System",
    version="0.1.0",
    lifespan=lifespan,
)

# Add middleware in order (last added = first executed)
# 1. Error handling (outermost - catches all errors)
app.add_middleware(ErrorHandlerMiddleware, debug=settings.log_level == "DEBUG")

# 2. Rate limiting
rate_limiter = RateLimiter(
    requests_per_second=10.0,  # 10 requests per second
    burst_size=20,             # Allow burst of 20 requests
)
app.add_middleware(
    RateLimitMiddleware,
    limiter=rate_limiter,
    exclude_paths=["/health", "/", "/docs", "/openapi.json"],
)

# 3. Correlation ID tracking
app.add_middleware(CorrelationIdMiddleware)

# 4. CORS (innermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_router)


class QueryRequest(BaseModel):
    """Query request."""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ProductInfo(BaseModel):
    """Product information."""

    asin: str | None = None
    title: str | None = None
    brand: str | None = None
    price: float | None = None
    stars: float | None = None
    category: str | None = None


class QueryResponse(BaseModel):
    """Query response."""

    query: str
    query_type: str | None
    products: list[ProductInfo]
    analysis: str | None
    answer: str | None
    steps: list[str]
    error: str | None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.service_name,
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    from src.cache import get_cache
    cache = await get_cache()
    stats = await cache.get_stats()
    return stats


@app.delete("/cache/clear")
async def cache_clear(pattern: str = "cache:*"):
    """Clear cache by pattern."""
    from src.cache import get_cache
    cache = await get_cache()
    deleted = await cache.invalidate_pattern(pattern)
    return {"deleted": deleted, "pattern": pattern}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the multi-agent system."""
    try:
        logger.info("processing_query", query=request.query[:50])

        result = await run_agent(request.query)

        products = [
            ProductInfo(**p) for p in result.get("products", [])
        ]

        return QueryResponse(
            query=result.get("query", request.query),
            query_type=result.get("query_type"),
            products=products,
            analysis=result.get("analysis"),
            answer=result.get("answer"),
            steps=result.get("steps", []),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error("query_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    """Search request with filters."""
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)
    category: str | None = None
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    min_rating: float | None = None
    rerank: bool | None = None  # Enable/disable reranking (default: use strategy setting)


class SearchStateInfo(BaseModel):
    """Search state information."""
    intent: str | None = None
    query_type: str | None = None
    strategy: str | None = None
    summary: str | None = None


class SearchProductResult(BaseModel):
    """Product result from search."""
    asin: str | None
    title: str | None
    brand: str | None
    price: float | None
    stars: float | None
    score: float | None = None
    img_url: str | None = None
    summary: str | None = None
    best_for: str | None = None


class SearchResultResponse(BaseModel):
    """Search result response."""
    query: str
    products: list[SearchProductResult]
    total: int
    latency_ms: float
    search_state: SearchStateInfo | None = None
    filters: dict | None = None


@app.post("/search", response_model=SearchResultResponse)
async def search_endpoint(request: SearchRequest):
    """
    Search products using optimized search strategies.

    Uses the SearchAgent with KeywordPriorityHybridStrategy (MRR 0.9126).
    Automatically detects query type and adjusts search strategy.
    """
    from src.agents import get_search_agent

    try:
        agent = await get_search_agent()

        # Build context with any filters
        context = {}
        if request.category:
            context["category"] = request.category
        if request.brand:
            context["brand"] = request.brand
        if request.price_min is not None:
            context["price_min"] = request.price_min
        if request.price_max is not None:
            context["price_max"] = request.price_max
        if request.min_rating is not None:
            context["min_rating"] = request.min_rating
        if request.rerank is not None:
            context["rerank"] = request.rerank

        state = await agent.search(request.query, context if context else None)

        products = [
            SearchProductResult(**p)
            for p in state.formatted_results[:request.limit]
        ]

        search_state = None
        if state.intent or state.query_type:
            search_state = SearchStateInfo(
                intent=state.intent.value if state.intent else None,
                query_type=state.query_type.value if state.query_type else None,
                strategy=state.search_strategy,
                summary=state.summary,
            )

        return SearchResultResponse(
            query=request.query,
            products=products,
            total=state.total_results,
            latency_ms=state.latency_ms,
            search_state=search_state,
            filters=state.filters.model_dump() if state.filters else None,
        )

    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/simple")
async def search_simple(request: QueryRequest):
    """Simplified search endpoint using SearchAgent."""
    from src.agents import search_products

    try:
        products = await search_products(request.query, limit=10)
        return {"query": request.query, "products": products}
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(request: QueryRequest):
    """Analysis endpoint."""
    try:
        result = await run_agent(f"Analyze: {request.query}")
        return {
            "query": request.query,
            "analysis": result.get("analysis"),
            "products": result.get("products", []),
        }
    except Exception as e:
        logger.error("analysis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa")
async def question_answer(request: QueryRequest):
    """Question answering endpoint."""
    try:
        result = await run_agent(request.query)
        return {
            "query": request.query,
            "answer": result.get("answer"),
            "products": result.get("products", []),
        }
    except Exception as e:
        logger.error("qa_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Specialist Agent Endpoints
# =============================================================================

class CompareRequest(BaseModel):
    """Compare request."""
    query: str = Field(..., min_length=1)
    product_names: list[str] = Field(default_factory=list)


@app.post("/compare")
async def compare_endpoint(request: CompareRequest):
    """Compare products endpoint."""
    from src.agents import compare_products

    try:
        result = await compare_products(
            request.query,
            product_names=request.product_names,
        )
        return {
            "query": request.query,
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
                "winner": result.comparison.winner.title if result.comparison.winner else None,
                "winner_reason": result.comparison.winner_reason,
                "best_value": result.comparison.best_value.title if result.comparison.best_value else None,
                "key_differences": result.comparison.key_differences,
                "summary": result.comparison.comparison_summary,
            },
        }
    except Exception as e:
        logger.error("compare_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class PriceRequest(BaseModel):
    """Price analysis request."""
    query: str = Field(..., min_length=1)
    target_price: float | None = None
    category: str | None = None


@app.post("/price")
async def price_endpoint(request: PriceRequest):
    """Price intelligence endpoint."""
    from src.agents import analyze_prices

    try:
        result = await analyze_prices(
            request.query,
            target_price=request.target_price,
        )
        return {
            "query": request.query,
            "products": [
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
            ],
            "best_deal": {
                "title": result.best_deal.title,
                "discount": result.best_deal.discount_pct,
            } if result.best_deal else None,
            "recommendation": result.recommendation,
            "summary": result.summary,
        }
    except Exception as e:
        logger.error("price_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class TrendRequest(BaseModel):
    """Trend analysis request."""
    query: str = Field(..., min_length=1)
    category: str | None = None
    time_range: str = "7d"


@app.post("/trends")
async def trends_endpoint(request: TrendRequest):
    """Market trends endpoint."""
    from src.agents import get_trends

    try:
        result = await get_trends(
            request.query,
            category=request.category,
            time_range=request.time_range,
        )
        return {
            "query": request.query,
            "trending_products": [
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
            ],
            "hot_categories": result.hot_categories,
            "insights": result.insights,
            "summary": result.summary,
        }
    except Exception as e:
        logger.error("trends_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class RecommendRequest(BaseModel):
    """Recommendation request."""
    query: str = Field(..., min_length=1)
    source_asin: str | None = None
    recommendation_type: str = "similar"  # similar, alternatives, accessories


@app.post("/recommend")
async def recommend_endpoint(request: RecommendRequest):
    """Product recommendations endpoint."""
    from src.agents import get_recommendations

    try:
        result = await get_recommendations(
            request.query,
            source_asin=request.source_asin,
            recommendation_type=request.recommendation_type,
        )
        return {
            "query": request.query,
            "recommendations": [
                {
                    "asin": r.asin,
                    "title": r.title,
                    "brand": r.brand,
                    "price": r.price,
                    "stars": r.stars,
                    "similarity_score": r.similarity_score,
                    "reason": r.recommendation_reason,
                    "match_type": r.match_type,
                }
                for r in result.recommendations
            ],
            "source_product": result.source_product,
            "summary": result.recommendation_summary,
        }
    except Exception as e:
        logger.error("recommend_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class ReviewAnalysisRequest(BaseModel):
    """Review analysis request."""
    query: str = Field(..., min_length=1)
    analysis_type: str = "general"  # general, pros_cons, sentiment, feature


@app.post("/reviews")
async def reviews_endpoint(request: ReviewAnalysisRequest):
    """Review analysis endpoint."""
    from src.agents import analyze_reviews

    try:
        result = await analyze_reviews(
            request.query,
            analysis_type=request.analysis_type,
        )
        return {
            "query": request.query,
            "analysis": {
                "pros": result.analysis.pros,
                "cons": result.analysis.cons,
                "sentiment_score": result.analysis.sentiment_score,
                "sentiment_label": result.analysis.sentiment_label,
                "common_themes": result.analysis.common_themes,
                "summary": result.analysis.summary,
            },
            "products_analyzed": result.products_analyzed,
        }
    except Exception as e:
        logger.error("reviews_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Conversation Endpoint
# =============================================================================

class ConversationRequest(BaseModel):
    """Conversation request."""
    query: str = Field(..., min_length=1)
    session_id: str | None = None


@app.post("/chat")
async def chat_endpoint(request: ConversationRequest):
    """Multi-turn conversation endpoint."""
    from src.conversation import get_conversation

    try:
        # Get or create conversation
        conversation = await get_conversation(request.session_id)

        # Add user message
        conversation.add_user_message(request.query)

        # Process query
        result = await run_agent(request.query)

        # Build response text
        response_text = ""
        if result.get("answer"):
            response_text = result["answer"]
        elif result.get("analysis"):
            response_text = result["analysis"]
        elif result.get("products"):
            products = result["products"][:5]
            response_text = f"Found {len(result['products'])} products:\n"
            for p in products:
                response_text += f"- {p.get('title', 'Unknown')[:50]} (${p.get('price', 'N/A')})\n"

        # Add assistant message
        conversation.add_assistant_message(
            response_text,
            metadata={"products": result.get("products", [])[:5]},
        )

        # Update context
        conversation.update_context(
            products=result.get("products", []),
            query_type=result.get("query_type"),
        )

        # Save conversation
        from src.conversation import get_conversation_manager
        manager = await get_conversation_manager()
        await manager.save(conversation)

        return {
            "session_id": conversation.session_id,
            "query": request.query,
            "response": response_text,
            "products": result.get("products", [])[:5],
            "query_type": result.get("query_type"),
            "context": {
                "recent_products": len(conversation.context.product_asins),
                "categories": conversation.context.categories[:3],
                "brands": conversation.context.brands[:3],
            },
        }
    except Exception as e:
        logger.error("chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Enhanced Workflow-Based Endpoints
# =============================================================================

class EnhancedChatRequest(BaseModel):
    """Enhanced chat request with workflow options."""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    use_compound_workflow: bool = False  # Force compound workflow


class EnhancedChatResponse(BaseModel):
    """Enhanced chat response with full metadata."""
    # Core response
    session_id: str
    query: str
    response: str
    intent: str

    # Products and data
    products: list[dict] = Field(default_factory=list)
    comparison: dict | None = None
    analysis: dict | None = None
    recommendations: list[dict] | None = None

    # Metadata
    format: str = "text"
    confidence: float = 0.0
    suggestions: list[str] = Field(default_factory=list)

    # Execution info
    execution_time_ms: float = 0.0
    agents_used: list[str] = Field(default_factory=list)
    turn_number: int = 1
    context_used: bool = False
    references_resolved: list[str] = Field(default_factory=list)

    # Error (if any)
    error: str | None = None


@app.post("/chat/v2", response_model=EnhancedChatResponse)
async def enhanced_chat_endpoint(request: EnhancedChatRequest):
    """Enhanced multi-turn conversation endpoint using workflows.

    Features:
    - Automatic intent detection and routing
    - Context resolution for references ("them", "compare those", "the first one")
    - Multi-step query execution
    - Comprehensive response with metadata
    - Follow-up suggestions
    """
    from src.workflows import execute_conversation_query

    logger.info(
        "chat_v2_request_received",
        query=request.query[:100],
        session_id=request.session_id,
        use_compound=request.use_compound_workflow,
    )

    try:
        result = await execute_conversation_query(
            query=request.query,
            session_id=request.session_id,
        )

        logger.info(
            "chat_v2_response_ready",
            intent=result.intent.value,
            num_products=len(result.products),
            execution_time_ms=result.execution_time_ms,
            agents_used=result.agents_used,
            error=result.error,
        )

        return EnhancedChatResponse(
            session_id=result.session_id,
            query=result.query,
            response=result.response_text,
            intent=result.intent.value,
            products=result.products,
            comparison=result.comparison,
            analysis=result.analysis,
            recommendations=result.recommendations,
            format=result.format.value,
            confidence=result.confidence,
            suggestions=result.suggestions,
            execution_time_ms=result.execution_time_ms,
            agents_used=result.agents_used,
            turn_number=result.turn_number,
            context_used=result.context_used,
            references_resolved=result.references_resolved,
            error=result.error,
        )

    except Exception as e:
        logger.error("enhanced_chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class LangGraphChatResponse(BaseModel):
    """LangGraph-powered chat response."""
    # Core response
    session_id: str
    query: str
    response: str
    intent: str

    # Products and data
    products: list[dict] = Field(default_factory=list)
    comparison: dict | None = None
    recommendations: list[dict] | None = None

    # Conversation metadata
    turn_number: int = 1
    action_taken: str = ""
    context_used: bool = False
    clarification_asked: bool = False
    preferences_learned: dict = Field(default_factory=dict)

    # Suggestions
    suggestions: list[str] = Field(default_factory=list)

    # Execution info
    execution_time_ms: float = 0.0
    error: str | None = None


@app.post("/chat/v3", response_model=LangGraphChatResponse)
async def langgraph_chat_endpoint(request: EnhancedChatRequest):
    """LangGraph-powered natural conversation endpoint.

    Features:
    - Full conversation context understanding via LLM
    - Automatic reference resolution for ANY type of reference
    - User preference learning (budget, brands, features)
    - Clarifying questions when truly ambiguous
    - Natural, ChatGPT-like responses
    - Persistent conversation state

    This is the most advanced chat endpoint, providing a natural
    conversation experience similar to ChatGPT.
    """
    from src.workflows.langgraph_conversation_workflow import (
        execute_langgraph_conversation,
    )

    logger.info(
        "chat_v3_request_received",
        query=request.query[:100],
        session_id=request.session_id,
    )

    try:
        result = await execute_langgraph_conversation(
            query=request.query,
            session_id=request.session_id,
        )

        logger.info(
            "chat_v3_response_ready",
            session_id=result.session_id,
            intent=result.intent.value if hasattr(result.intent, 'value') else str(result.intent),
            action=result.action_taken,
            turn=result.turn_number,
            clarification=result.clarification_asked,
            execution_time_ms=result.execution_time_ms,
            error=result.error,
        )

        return LangGraphChatResponse(
            session_id=result.session_id,
            query=result.query,
            response=result.response_text,
            intent=result.intent.value if hasattr(result.intent, 'value') else str(result.intent),
            products=result.products,
            comparison=result.comparison,
            recommendations=result.recommendations,
            turn_number=result.turn_number,
            action_taken=result.action_taken,
            context_used=result.context_used,
            clarification_asked=result.clarification_asked,
            preferences_learned=result.preferences_learned,
            suggestions=result.suggestions,
            execution_time_ms=result.execution_time_ms,
            error=result.error,
        )

    except Exception as e:
        logger.error("langgraph_chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/v3/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session."""
    from src.workflows.langgraph_conversation_workflow import (
        get_langgraph_conversation_workflow,
    )

    try:
        workflow = await get_langgraph_conversation_workflow()
        history = await workflow.get_history(session_id)

        return {
            "session_id": session_id,
            "messages": history,
            "total_messages": len(history),
        }
    except Exception as e:
        logger.error("get_history_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class WorkflowQueryRequest(BaseModel):
    """Request for workflow-based query processing."""
    query: str = Field(..., min_length=1, max_length=2000)
    workflow_type: str = "auto"  # auto, simple, compound


class WorkflowQueryResponse(BaseModel):
    """Response from workflow query processing."""
    query: str
    response: str
    intent: str
    products: list[dict] = Field(default_factory=list)
    comparison: dict | None = None
    analysis: dict | None = None
    format: str = "text"
    confidence: float = 0.0
    suggestions: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    agents_used: list[str] = Field(default_factory=list)
    workflow_used: str = "simple"
    error: str | None = None


@app.post("/query/v2", response_model=WorkflowQueryResponse)
async def workflow_query_endpoint(request: WorkflowQueryRequest):
    """Process query using workflow system.

    Workflow types:
    - auto: Automatically choose based on query complexity
    - simple: Single-intent workflow
    - compound: Multi-step workflow with execution planning
    """
    from src.workflows import (
        execute_simple_query,
        execute_compound_query,
    )
    from src.agents.intent_agent import get_intent_agent
    from src.models.intent import QueryComplexity

    try:
        workflow_used = request.workflow_type

        if request.workflow_type == "auto":
            # Analyze intent to determine workflow
            intent_agent = await get_intent_agent()
            analysis = await intent_agent.execute(request.query)

            if (
                analysis.complexity == QueryComplexity.COMPOUND
                or len(analysis.secondary_intents) > 0
            ):
                workflow_used = "compound"
            else:
                workflow_used = "simple"

        if workflow_used == "compound":
            result = await execute_compound_query(request.query)
        else:
            result = await execute_simple_query(request.query)

        return WorkflowQueryResponse(
            query=result.query,
            response=result.response_text,
            intent=result.intent.value,
            products=result.products,
            comparison=result.comparison,
            analysis=result.analysis,
            format=result.format.value,
            confidence=result.confidence,
            suggestions=result.suggestions,
            execution_time_ms=result.execution_time_ms,
            agents_used=result.agents_used,
            workflow_used=workflow_used,
            error=result.error,
        )

    except Exception as e:
        logger.error("workflow_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True,
    )
