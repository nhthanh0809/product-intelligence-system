"""
LangChain-compatible tool wrappers for search tools.

These tools can be used directly with LangChain agents and LangGraph.

Usage:
    from src.tools.langchain_tools import get_search_tools

    tools = await get_search_tools()
    agent = create_react_agent(llm, tools)
"""

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.tools.search_tools import (
    SearchToolkit,
    SearchToolConfig,
    SearchResponse,
    QueryAnalyzer,
)


# =============================================================================
# Tool Input Schemas
# =============================================================================

class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""
    query: str = Field(description="Natural language search query for products")
    limit: int = Field(default=10, description="Maximum number of results to return")
    category: Optional[str] = Field(default=None, description="Filter by category (e.g., 'Electronics')")
    brand: Optional[str] = Field(default=None, description="Filter by brand (e.g., 'Sony')")
    price_min: Optional[float] = Field(default=None, description="Minimum price filter")
    price_max: Optional[float] = Field(default=None, description="Maximum price filter")
    min_rating: Optional[float] = Field(default=None, description="Minimum star rating (1-5)")


class SemanticSearchInput(BaseModel):
    """Input for semantic search tool."""
    query: str = Field(description="Conceptual or generic search query")
    limit: int = Field(default=10, description="Maximum number of results")
    category: Optional[str] = Field(default=None, description="Filter by category")
    brand: Optional[str] = Field(default=None, description="Filter by brand")


class KeywordSearchInput(BaseModel):
    """Input for keyword search tool."""
    query: str = Field(description="Keyword search query (best for brand/model names)")
    limit: int = Field(default=10, description="Maximum number of results")
    category: Optional[str] = Field(default=None, description="Filter by category")
    brand: Optional[str] = Field(default=None, description="Filter by brand")


class SectionSearchInput(BaseModel):
    """Input for section-targeted search tool."""
    query: str = Field(description="Search query for specific product aspect")
    section: str = Field(
        description="Section to search: 'reviews', 'specs', 'features', 'description', or 'use_cases'"
    )
    limit: int = Field(default=10, description="Maximum number of results")
    category: Optional[str] = Field(default=None, description="Filter by category")
    brand: Optional[str] = Field(default=None, description="Filter by brand")


class SimilarProductsInput(BaseModel):
    """Input for similar products tool."""
    asin: str = Field(description="ASIN of the source product to find similar products for")
    limit: int = Field(default=10, description="Maximum number of similar products")
    category: Optional[str] = Field(default=None, description="Filter by category")
    price_min: Optional[float] = Field(default=None, description="Minimum price filter")
    price_max: Optional[float] = Field(default=None, description="Maximum price filter")


class ProductLookupInput(BaseModel):
    """Input for product lookup tool."""
    asin: str = Field(description="ASIN of the product to look up")


class CompareProductsInput(BaseModel):
    """Input for product comparison tool."""
    asins: list[str] = Field(description="List of ASINs to compare (2-5 products)")


# =============================================================================
# LangChain Tool Implementations
# =============================================================================

class HybridSearchTool(BaseTool):
    """
    Search products using hybrid keyword+semantic search.

    This is the BEST performing search strategy (MRR 0.9126).
    Automatically adjusts keyword/semantic weights based on query type.

    Use for:
    - General product discovery
    - Any search query (adapts automatically)
    """

    name: str = "hybrid_search"
    description: str = """Search products using hybrid keyword+semantic search.
This is the best strategy for general product search. Use this as the default search tool.
It automatically detects query type (brand+model, generic, etc.) and adjusts accordingly.
Returns product results with title, price, rating, and summaries."""

    args_schema: Type[BaseModel] = HybridSearchInput
    toolkit: SearchToolkit | None = None

    def _build_filters(
        self,
        category: str | None,
        brand: str | None,
        price_min: float | None,
        price_max: float | None,
        min_rating: float | None,
    ) -> dict[str, Any] | None:
        """Build filters dict from optional params."""
        filters = {}
        if category:
            filters["category_level1"] = category
        if brand:
            filters["brand"] = brand
        if price_min is not None:
            filters["price_min"] = price_min
        if price_max is not None:
            filters["price_max"] = price_max
        if min_rating is not None:
            filters["min_rating"] = min_rating
        return filters if filters else None

    def _run(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        min_rating: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Synchronous execution (not supported)."""
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        min_rating: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Execute hybrid search."""
        if self.toolkit is None:
            self.toolkit = SearchToolkit()
            await self.toolkit.initialize()

        filters = self._build_filters(category, brand, price_min, price_max, min_rating)
        response = await self.toolkit.hybrid_search(query, limit, filters)
        return self._format_response(response)

    def _format_response(self, response: SearchResponse) -> str:
        """Format search response for agent consumption."""
        if not response.results:
            return f"No products found for query: {response.query}"

        lines = [
            f"Found {response.total_results} products (search type: {response.search_type}, "
            f"query type: {response.query_type}, latency: {response.latency_ms:.1f}ms)",
            ""
        ]

        for i, r in enumerate(response.results, 1):
            price_str = f"${r.price:.2f}" if r.price else "N/A"
            rating_str = f"{r.stars}★" if r.stars else "N/A"
            brand_str = f" ({r.brand})" if r.brand else ""
            lines.append(f"{i}. [{r.asin}] {r.title[:80]}{brand_str}")
            lines.append(f"   Price: {price_str} | Rating: {rating_str} | Score: {r.score:.4f}")
            if r.genAI_summary:
                lines.append(f"   Summary: {r.genAI_summary[:100]}...")
            lines.append("")

        return "\n".join(lines)


class SemanticSearchTool(BaseTool):
    """
    Search products using semantic similarity.

    Best for conceptual/generic queries where exact keyword matching
    is less important than understanding intent.

    Use for:
    - Generic queries ("good headphones for travel")
    - Use-case based searches
    - Conceptual product discovery
    """

    name: str = "semantic_search"
    description: str = """Search products using semantic similarity.
Best for conceptual or generic queries where understanding intent matters more than exact keywords.
Use this for queries like "good laptop for programming" or "camera for beginners".
Returns products ranked by semantic relevance to the query meaning."""

    args_schema: Type[BaseModel] = SemanticSearchInput
    toolkit: SearchToolkit | None = None

    def _run(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        if self.toolkit is None:
            self.toolkit = SearchToolkit()
            await self.toolkit.initialize()

        filters = {}
        if category:
            filters["category_level1"] = category
        if brand:
            filters["brand"] = brand

        response = await self.toolkit.semantic_search(
            query, limit, filters if filters else None
        )
        return self._format_response(response)

    def _format_response(self, response: SearchResponse) -> str:
        if not response.results:
            return f"No products found for query: {response.query}"

        lines = [f"Found {response.total_results} products (semantic search)", ""]
        for i, r in enumerate(response.results, 1):
            price_str = f"${r.price:.2f}" if r.price else "N/A"
            rating_str = f"{r.stars}★" if r.stars else "N/A"
            lines.append(f"{i}. [{r.asin}] {r.title[:80]}")
            lines.append(f"   Price: {price_str} | Rating: {rating_str} | Score: {r.score:.4f}")
            if r.genAI_summary:
                lines.append(f"   Summary: {r.genAI_summary[:100]}...")
            lines.append("")
        return "\n".join(lines)


class KeywordSearchTool(BaseTool):
    """
    Search products using keyword matching.

    Best for specific product searches with brand names or model numbers.
    Performance: R@1 87.7% on brand+model queries.

    Use for:
    - Brand + model searches ("Sony WH-1000XM5")
    - Model number lookups
    - Exact product name searches
    """

    name: str = "keyword_search"
    description: str = """Search products using keyword matching.
Best for specific searches with brand names or model numbers.
Use this for queries like "Sony WH-1000XM5" or "iPhone 15 Pro Max".
Has 87.7% accuracy on brand+model queries."""

    args_schema: Type[BaseModel] = KeywordSearchInput
    toolkit: SearchToolkit | None = None

    def _run(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        if self.toolkit is None:
            self.toolkit = SearchToolkit()
            await self.toolkit.initialize()

        filters = {}
        if category:
            filters["category_level1"] = category
        if brand:
            filters["brand"] = brand

        response = await self.toolkit.keyword_search(
            query, limit, filters if filters else None
        )
        return self._format_response(response)

    def _format_response(self, response: SearchResponse) -> str:
        if not response.results:
            return f"No products found for query: {response.query}"

        lines = [
            f"Found {response.total_results} products (keyword search, "
            f"query type: {response.query_type})",
            ""
        ]
        for i, r in enumerate(response.results, 1):
            price_str = f"${r.price:.2f}" if r.price else "N/A"
            rating_str = f"{r.stars}★" if r.stars else "N/A"
            brand_str = f" ({r.brand})" if r.brand else ""
            lines.append(f"{i}. [{r.asin}] {r.title[:80]}{brand_str}")
            lines.append(f"   Price: {price_str} | Rating: {rating_str} | Score: {r.score:.4f}")
            lines.append("")
        return "\n".join(lines)


class SectionSearchTool(BaseTool):
    """
    Search within specific product sections.

    Targets specific aspects of products like reviews, specs, or features.
    Uses child nodes in the vector store for precise retrieval.

    Use for:
    - Review analysis ("what do reviews say about battery life")
    - Specifications lookup ("technical specs of this laptop")
    - Feature queries ("features of this camera")
    """

    name: str = "section_search"
    description: str = """Search within specific product sections.
Use this to find products based on specific aspects like reviews, specs, or features.
Sections available: 'reviews', 'specs', 'features', 'description', 'use_cases'.
Best for queries like "products with good battery reviews" or "laptops with specific specs"."""

    args_schema: Type[BaseModel] = SectionSearchInput
    toolkit: SearchToolkit | None = None

    def _run(
        self,
        query: str,
        section: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        query: str,
        section: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        if self.toolkit is None:
            self.toolkit = SearchToolkit()
            await self.toolkit.initialize()

        filters = {}
        if category:
            filters["category_level1"] = category
        if brand:
            filters["brand"] = brand

        try:
            response = await self.toolkit.section_search(
                query, section, limit, filters if filters else None
            )
            return self._format_response(response, section)
        except ValueError as e:
            return f"Error: {str(e)}. Valid sections: reviews, specs, features, description, use_cases"

    def _format_response(self, response: SearchResponse, section: str) -> str:
        if not response.results:
            return f"No products found for section '{section}' query: {response.query}"

        lines = [
            f"Found {response.total_results} products matching '{section}' section",
            ""
        ]
        for i, r in enumerate(response.results, 1):
            price_str = f"${r.price:.2f}" if r.price else "N/A"
            rating_str = f"{r.stars}★" if r.stars else "N/A"
            lines.append(f"{i}. [{r.asin}] {r.title[:80]}")
            lines.append(f"   Price: {price_str} | Rating: {rating_str}")
            if r.content_preview:
                lines.append(f"   {section.capitalize()}: {r.content_preview[:150]}...")
            lines.append("")
        return "\n".join(lines)


class SimilarProductsTool(BaseTool):
    """
    Find products similar to a given product.

    Uses vector similarity to find products with similar characteristics.

    Use for:
    - "Show me similar products"
    - "Find alternatives to X"
    - Recommendation scenarios
    """

    name: str = "find_similar_products"
    description: str = """Find products similar to a given product by ASIN.
Use this when a user wants alternatives or similar products to one they're looking at.
Returns products with similar characteristics based on vector similarity."""

    args_schema: Type[BaseModel] = SimilarProductsInput
    toolkit: SearchToolkit | None = None

    def _run(
        self,
        asin: str,
        limit: int = 10,
        category: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        asin: str,
        limit: int = 10,
        category: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        if self.toolkit is None:
            self.toolkit = SearchToolkit()
            await self.toolkit.initialize()

        filters = {}
        if category:
            filters["category_level1"] = category
        if price_min is not None:
            filters["price_min"] = price_min
        if price_max is not None:
            filters["price_max"] = price_max

        response = await self.toolkit.find_similar(
            asin, limit, filters if filters else None
        )
        return self._format_response(response, asin)

    def _format_response(self, response: SearchResponse, source_asin: str) -> str:
        if not response.results:
            return f"No similar products found for ASIN: {source_asin}"

        lines = [
            f"Found {response.total_results} products similar to {source_asin}",
            ""
        ]
        for i, r in enumerate(response.results, 1):
            price_str = f"${r.price:.2f}" if r.price else "N/A"
            rating_str = f"{r.stars}★" if r.stars else "N/A"
            brand_str = f" ({r.brand})" if r.brand else ""
            lines.append(f"{i}. [{r.asin}] {r.title[:80]}{brand_str}")
            lines.append(f"   Price: {price_str} | Rating: {rating_str} | Similarity: {r.score:.4f}")
            if r.genAI_summary:
                lines.append(f"   Summary: {r.genAI_summary[:100]}...")
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# Tool Factory
# =============================================================================

# Shared toolkit instance
_toolkit: SearchToolkit | None = None


async def get_toolkit() -> SearchToolkit:
    """Get shared toolkit instance."""
    global _toolkit
    if _toolkit is None:
        _toolkit = SearchToolkit()
        await _toolkit.initialize()
    return _toolkit


async def get_search_tools() -> list[BaseTool]:
    """
    Get all search tools configured with shared toolkit.

    Returns:
        List of LangChain-compatible search tools
    """
    toolkit = await get_toolkit()

    # Create tool instances with shared toolkit
    hybrid = HybridSearchTool()
    hybrid.toolkit = toolkit

    semantic = SemanticSearchTool()
    semantic.toolkit = toolkit

    keyword = KeywordSearchTool()
    keyword.toolkit = toolkit

    section = SectionSearchTool()
    section.toolkit = toolkit

    similar = SimilarProductsTool()
    similar.toolkit = toolkit

    return [hybrid, semantic, keyword, section, similar]


async def get_search_agent_tools() -> list[BaseTool]:
    """
    Get tools specifically for the Search Agent.

    Returns hybrid, keyword, and semantic search tools.
    """
    toolkit = await get_toolkit()

    hybrid = HybridSearchTool()
    hybrid.toolkit = toolkit

    keyword = KeywordSearchTool()
    keyword.toolkit = toolkit

    semantic = SemanticSearchTool()
    semantic.toolkit = toolkit

    return [hybrid, keyword, semantic]


async def get_analysis_agent_tools() -> list[BaseTool]:
    """
    Get tools specifically for the Analysis Agent.

    Returns section search (for reviews) and hybrid search.
    """
    toolkit = await get_toolkit()

    section = SectionSearchTool()
    section.toolkit = toolkit

    hybrid = HybridSearchTool()
    hybrid.toolkit = toolkit

    return [section, hybrid]


async def get_compare_agent_tools() -> list[BaseTool]:
    """
    Get tools specifically for the Compare Agent.

    Returns section search (for specs/features) and similar products.
    """
    toolkit = await get_toolkit()

    section = SectionSearchTool()
    section.toolkit = toolkit

    similar = SimilarProductsTool()
    similar.toolkit = toolkit

    return [section, similar]


async def get_recommend_agent_tools() -> list[BaseTool]:
    """
    Get tools specifically for the Recommend Agent.

    Returns similar products and hybrid search.
    """
    toolkit = await get_toolkit()

    similar = SimilarProductsTool()
    similar.toolkit = toolkit

    hybrid = HybridSearchTool()
    hybrid.toolkit = toolkit

    return [similar, hybrid]


async def cleanup():
    """Cleanup shared resources."""
    global _toolkit
    if _toolkit:
        await _toolkit.close()
        _toolkit = None
