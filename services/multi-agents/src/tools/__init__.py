"""
Agent tools for product search and analysis.

This module provides two sets of tools:

1. Simple HTTP client tools (original) - Make calls to vector-store service
2. Advanced search tools (new) - Direct implementations of optimized search strategies

Usage:
    # Simple HTTP tools (for basic use)
    from src.tools import semantic_search, keyword_search, hybrid_search

    # Advanced search tools (for agents)
    from src.tools.langchain_tools import get_search_tools
    tools = await get_search_tools()

    # Agent-specific tools
    from src.tools.langchain_tools import get_search_agent_tools
    search_tools = await get_search_agent_tools()
"""

from typing import Any

import httpx
from langchain_core.tools import tool

from src.config import get_settings

settings = get_settings()


# =============================================================================
# Simple HTTP Client Tools (Original)
# =============================================================================

@tool
async def semantic_search(query: str, limit: int = 10, filters: dict | None = None) -> list[dict]:
    """Search products using semantic similarity.

    Args:
        query: Natural language search query
        limit: Maximum number of results
        filters: Optional filters (category_level1, brand, price_min, price_max, min_rating)

    Returns:
        List of matching products with scores
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.vector_store_url}/search/semantic",
            json={"query": query, "limit": limit, "filters": filters},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("results", [])


@tool
async def keyword_search(query: str, limit: int = 10, filters: dict | None = None) -> list[dict]:
    """Search products using keyword matching.

    Args:
        query: Keyword search query
        limit: Maximum number of results
        filters: Optional filters

    Returns:
        List of matching products
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.vector_store_url}/search/keyword",
            json={"query": query, "limit": limit, "filters": filters},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("results", [])


@tool
async def hybrid_search(query: str, limit: int = 10, filters: dict | None = None) -> list[dict]:
    """Search products using hybrid semantic + keyword search.

    Args:
        query: Search query
        limit: Maximum number of results
        filters: Optional filters

    Returns:
        Fused search results
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.vector_store_url}/search/hybrid",
            json={"query": query, "limit": limit, "filters": filters},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("results", [])


@tool
async def product_lookup(asin: str) -> dict:
    """Get detailed product information by ASIN.

    Args:
        asin: Amazon Standard Identification Number

    Returns:
        Product details
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.vector_store_url}/product/{asin}",
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


@tool
async def compare_products(asins: list[str]) -> dict:
    """Compare multiple products by their ASINs.

    Args:
        asins: List of ASINs to compare

    Returns:
        Comparison of products
    """
    products = []
    async with httpx.AsyncClient() as client:
        for asin in asins[:5]:  # Limit to 5 products
            try:
                response = await client.get(
                    f"{settings.vector_store_url}/product/{asin}",
                    timeout=30.0,
                )
                if response.status_code == 200:
                    products.append(response.json())
            except Exception:
                pass

    return {
        "products": products,
        "count": len(products),
    }


@tool
async def aggregate_stats(field: str, query: str | None = None) -> dict:
    """Get aggregation statistics for a field.

    Args:
        field: Field to aggregate (category_level1, brand, etc.)
        query: Optional filter query

    Returns:
        Aggregation buckets
    """
    async with httpx.AsyncClient() as client:
        params = {"limit": 20}
        if query:
            params["q"] = query
        response = await client.get(
            f"{settings.vector_store_url}/aggregations/{field}",
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


# Simple HTTP tools (original)
ALL_TOOLS = [
    semantic_search,
    keyword_search,
    hybrid_search,
    product_lookup,
    compare_products,
    aggregate_stats,
]


# =============================================================================
# Advanced Search Tools Exports
# =============================================================================

# Export search toolkit and tools for direct use
from src.tools.search_tools import (
    SearchToolkit,
    SearchToolConfig,
    SearchMode,
    SearchClients,
    QueryAnalyzer,
    QueryAnalysis,
    QueryType,
    SearchResult,
    SearchResponse,
    SemanticSearchTool as SemanticSearchToolDirect,
    KeywordSearchTool as KeywordSearchToolDirect,
    HybridSearchTool as HybridSearchToolDirect,
    SectionSearchTool as SectionSearchToolDirect,
    SimilarProductsTool as SimilarProductsToolDirect,
)

# Export LangChain-compatible tools
from src.tools.langchain_tools import (
    get_search_tools,
    get_search_agent_tools,
    get_analysis_agent_tools,
    get_compare_agent_tools,
    get_recommend_agent_tools,
    get_toolkit,
    cleanup,
    HybridSearchTool,
    SemanticSearchTool,
    KeywordSearchTool,
    SectionSearchTool,
    SimilarProductsTool,
)

# Export aggregation tools
from src.tools.aggregation_tools import (
    AggregationToolkit,
    BrandStats,
    CategoryStats,
    PriceDistribution,
    get_aggregation_toolkit,
)

# Export cache-aware tools
from src.tools.cache_tools import (
    CachedSearchToolkit,
    CachedAggregationToolkit,
    get_cached_search_toolkit,
    get_cached_aggregation_toolkit,
    cached,
)

__all__ = [
    # Simple HTTP tools
    "semantic_search",
    "keyword_search",
    "hybrid_search",
    "product_lookup",
    "compare_products",
    "aggregate_stats",
    "ALL_TOOLS",
    # Search toolkit
    "SearchToolkit",
    "SearchToolConfig",
    "SearchMode",
    "SearchClients",
    "QueryAnalyzer",
    "QueryAnalysis",
    "QueryType",
    "SearchResult",
    "SearchResponse",
    # Direct tool classes
    "SemanticSearchToolDirect",
    "KeywordSearchToolDirect",
    "HybridSearchToolDirect",
    "SectionSearchToolDirect",
    "SimilarProductsToolDirect",
    # LangChain tools
    "get_search_tools",
    "get_search_agent_tools",
    "get_analysis_agent_tools",
    "get_compare_agent_tools",
    "get_recommend_agent_tools",
    "get_toolkit",
    "cleanup",
    "HybridSearchTool",
    "SemanticSearchTool",
    "KeywordSearchTool",
    "SectionSearchTool",
    "SimilarProductsTool",
    # Aggregation tools
    "AggregationToolkit",
    "BrandStats",
    "CategoryStats",
    "PriceDistribution",
    "get_aggregation_toolkit",
    # Cache-aware tools
    "CachedSearchToolkit",
    "CachedAggregationToolkit",
    "get_cached_search_toolkit",
    "get_cached_aggregation_toolkit",
    "cached",
]
