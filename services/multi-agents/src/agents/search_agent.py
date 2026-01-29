"""
Search Agent for Product Intelligence System.

This agent handles product discovery using optimized search strategies
based on search-flow experiments (Jan 2026).

Capabilities:
- Natural language to search parameters
- Query type detection and strategy routing
- Filter extraction from queries
- Result ranking and formatting

Tools:
- hybrid_search: Default for general queries (MRR 0.9126)
- keyword_search: For brand/model queries (R@1 87.7%)
- semantic_search: For generic/conceptual queries
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.config import get_settings
from src.config.repository import get_config_repository
from src.search import (
    SearchStrategyRegistry,
    get_search_registry,
    QueryAnalyzer,
    QueryType,
    SearchResponse,
    SearchFilters as StrategyFilters,
)

logger = structlog.get_logger()
settings = get_settings()


# =============================================================================
# State and Models
# =============================================================================

class SearchIntent(str, Enum):
    """Detected search intent."""
    PRODUCT_DISCOVERY = "product_discovery"  # Find products matching criteria
    BRAND_LOOKUP = "brand_lookup"            # Find specific brand products
    MODEL_LOOKUP = "model_lookup"            # Find specific model
    FEATURE_SEARCH = "feature_search"        # Find products with features
    USE_CASE_SEARCH = "use_case_search"      # Find products for use case
    PRICE_RANGE = "price_range"              # Find products in price range
    TOP_RATED = "top_rated"                  # Find highly rated products
    COMPARISON_PREP = "comparison_prep"      # Find products to compare


class SearchFilters(BaseModel):
    """Extracted search filters."""
    category: str | None = None
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    min_rating: float | None = None
    keywords: list[str] = Field(default_factory=list)


@dataclass
class SearchAgentState:
    """State for search agent workflow."""
    # Input
    query: str
    context: dict[str, Any] = field(default_factory=dict)

    # Analysis
    intent: SearchIntent | None = None
    query_type: QueryType | None = None
    filters: SearchFilters | None = None
    clean_query: str = ""

    # Search execution
    search_strategy: str = "hybrid"
    search_results: list[dict[str, Any]] = field(default_factory=list)
    search_response: SearchResponse | None = None

    # Output
    formatted_results: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    error: str | None = None
    steps: list[str] = field(default_factory=list)

    # Metrics
    latency_ms: float = 0.0
    total_results: int = 0


# =============================================================================
# Filter Extraction
# =============================================================================

class FilterExtractor:
    """Extract search filters from natural language queries."""

    # Price patterns
    PRICE_PATTERNS = [
        (r'under\s*\$?(\d+)', 'max'),
        (r'below\s*\$?(\d+)', 'max'),
        (r'less\s*than\s*\$?(\d+)', 'max'),
        (r'up\s*to\s*\$?(\d+)', 'max'),
        (r'over\s*\$?(\d+)', 'min'),
        (r'above\s*\$?(\d+)', 'min'),
        (r'more\s*than\s*\$?(\d+)', 'min'),
        (r'at\s*least\s*\$?(\d+)', 'min'),
        (r'\$(\d+)\s*-\s*\$?(\d+)', 'range'),
        (r'between\s*\$?(\d+)\s*and\s*\$?(\d+)', 'range'),
        (r'(\d+)\s*to\s*(\d+)\s*dollars?', 'range'),
    ]

    # Rating patterns
    RATING_PATTERNS = [
        (r'(\d+(?:\.\d+)?)\s*(?:star|stars|\+)\s*(?:and\s*(?:up|above|higher))?', 'min'),
        (r'at\s*least\s*(\d+(?:\.\d+)?)\s*stars?', 'min'),
        (r'(?:above|over)\s*(\d+(?:\.\d+)?)\s*stars?', 'min'),
        (r'highly\s*rated', 4.0),
        (r'top\s*rated', 4.5),
        (r'best\s*rated', 4.5),
    ]

    # Category keywords
    CATEGORY_KEYWORDS = {
        "electronics": ["electronics", "electronic", "gadget", "device", "tech"],
        "computers": ["computer", "laptop", "desktop", "pc", "notebook"],
        "headphones": ["headphones", "earbuds", "earphones", "headset"],
        "cameras": ["camera", "photography", "dslr", "mirrorless"],
        "home": ["home", "kitchen", "appliance", "furniture"],
        "tools": ["tool", "tools", "power tool", "hand tool"],
        "sports": ["sports", "fitness", "outdoor", "exercise"],
    }

    # Known brands
    KNOWN_BRANDS = [
        "Sony", "Bose", "Apple", "Samsung", "LG", "Anker", "JBL", "Sennheiser",
        "Dell", "HP", "Lenovo", "Microsoft", "Google", "Amazon", "Kindle",
        "DeWalt", "Milwaukee", "Makita", "Bosch", "Ryobi", "Black+Decker",
        "Canon", "Nikon", "Fujifilm", "Dyson", "Shark", "Roomba", "iRobot",
        "KitchenAid", "Ninja", "Instant Pot", "Vitamix", "Philips", "Panasonic",
    ]

    @classmethod
    def extract(cls, query: str) -> SearchFilters:
        """Extract all filters from query."""
        query_lower = query.lower()

        return SearchFilters(
            category=cls._extract_category(query_lower),
            brand=cls._extract_brand(query),
            price_min=cls._extract_price_min(query_lower),
            price_max=cls._extract_price_max(query_lower),
            min_rating=cls._extract_min_rating(query_lower),
            keywords=cls._extract_keywords(query_lower),
        )

    @classmethod
    def _extract_category(cls, query: str) -> str | None:
        """Extract category from query."""
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                return category
        return None

    @classmethod
    def _extract_brand(cls, query: str) -> str | None:
        """Extract brand from query."""
        query_lower = query.lower()
        for brand in cls.KNOWN_BRANDS:
            if brand.lower() in query_lower:
                return brand
        return None

    @classmethod
    def _extract_price_min(cls, query: str) -> float | None:
        """Extract minimum price from query."""
        for pattern, price_type in cls.PRICE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if price_type == 'min':
                    return float(match.group(1))
                elif price_type == 'range':
                    return float(match.group(1))
        return None

    @classmethod
    def _extract_price_max(cls, query: str) -> float | None:
        """Extract maximum price from query."""
        for pattern, price_type in cls.PRICE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if price_type == 'max':
                    return float(match.group(1))
                elif price_type == 'range':
                    return float(match.group(2))
        return None

    @classmethod
    def _extract_min_rating(cls, query: str) -> float | None:
        """Extract minimum rating from query."""
        for pattern, rating_type in cls.RATING_PATTERNS:
            if isinstance(rating_type, float):
                if re.search(pattern, query, re.IGNORECASE):
                    return rating_type
            else:
                match = re.search(pattern, query, re.IGNORECASE)
                if match and rating_type == 'min':
                    return float(match.group(1))
        return None

    @classmethod
    def _extract_keywords(cls, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'find', 'search', 'looking', 'show', 'get', 'want', 'need',
            'me', 'i', 'my', 'best', 'good', 'great', 'top',
        }
        words = re.findall(r'\b[a-z]+\b', query)
        return [w for w in words if len(w) > 2 and w not in stop_words]


# =============================================================================
# Intent Classification
# =============================================================================

class IntentClassifier:
    """Classify search intent from query."""

    @classmethod
    def classify(cls, query: str, analysis: Any) -> SearchIntent:
        """Classify the search intent."""
        query_lower = query.lower()

        # Check for comparison first (takes priority over brand/model lookup)
        compare_patterns = ['compare', ' vs ', 'versus', 'difference between', 'which is better']
        if any(p in query_lower for p in compare_patterns):
            return SearchIntent.COMPARISON_PREP

        # Check for specific patterns
        if analysis.model_numbers:
            return SearchIntent.MODEL_LOOKUP

        if analysis.has_brand and len(query.split()) <= 4:
            return SearchIntent.BRAND_LOOKUP

        # Check for price-related queries
        price_patterns = ['under', 'below', 'above', 'between', 'budget', 'cheap', 'affordable']
        if any(p in query_lower for p in price_patterns):
            return SearchIntent.PRICE_RANGE

        # Check for rating-related queries
        rating_patterns = ['top rated', 'highly rated', 'best rated', 'best reviewed']
        if any(p in query_lower for p in rating_patterns):
            return SearchIntent.TOP_RATED

        # Check for feature search
        feature_patterns = ['with', 'that has', 'featuring', 'includes']
        if any(p in query_lower for p in feature_patterns):
            return SearchIntent.FEATURE_SEARCH

        # Check for use case search
        use_case_patterns = ['for', 'good for', 'best for', 'suitable for']
        if any(p in query_lower for p in use_case_patterns):
            return SearchIntent.USE_CASE_SEARCH

        # Default to product discovery
        return SearchIntent.PRODUCT_DISCOVERY


# =============================================================================
# Search Agent
# =============================================================================

class SearchAgent:
    """
    Search Agent for product discovery.

    Uses optimized search strategies based on query analysis:
    - Hybrid search (default): MRR 0.9126
    - Keyword search: Best for brand/model (R@1 87.7%)
    - Semantic search: Best for generic queries

    Strategies are loaded dynamically from database configuration via
    SearchStrategyRegistry, allowing runtime configuration changes.
    """

    def __init__(self):
        self.registry: SearchStrategyRegistry | None = None
        self.settings = get_settings()

    async def initialize(self):
        """Initialize the search strategy registry."""
        if self.registry is None:
            config_repo = await get_config_repository()
            self.registry = await get_search_registry(config_repo)

    async def close(self):
        """Close the search registry."""
        if self.registry:
            await self.registry.shutdown()
            self.registry = None

    async def search(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> SearchAgentState:
        """
        Execute search with intelligent strategy selection.

        Args:
            query: User search query
            context: Optional context from previous interactions

        Returns:
            SearchAgentState with results and metadata
        """
        logger.info("search_agent_search_called", query=query[:30])
        await self.initialize()

        state = SearchAgentState(
            query=query,
            context=context or {},
        )

        try:
            # Step 1: Analyze query
            state = await self._analyze_query(state)

            # Step 2: Extract filters
            state = await self._extract_filters(state)

            # Step 3: Select search strategy
            state = await self._select_strategy(state)

            # Step 4: Execute search
            state = await self._execute_search(state)

            # Step 5: Format results
            state = await self._format_results(state)

            # Step 6: Generate summary
            state = await self._generate_summary(state)

        except Exception as e:
            logger.error("search_agent_failed", error=str(e))
            state.error = str(e)
            state.steps.append(f"error:{str(e)}")

        return state

    async def _analyze_query(self, state: SearchAgentState) -> SearchAgentState:
        """Analyze the query to determine type and characteristics."""
        analysis = QueryAnalyzer.analyze(state.query)

        state.query_type = analysis.query_type
        state.clean_query = analysis.clean_query
        state.intent = IntentClassifier.classify(state.query, analysis)
        state.steps.append(f"analyzed:type={analysis.query_type.value},intent={state.intent.value}")

        logger.debug(
            "query_analyzed",
            query_type=analysis.query_type.value,
            intent=state.intent.value,
            has_brand=analysis.has_brand,
            has_model=bool(analysis.model_numbers),
        )

        return state

    async def _extract_filters(self, state: SearchAgentState) -> SearchAgentState:
        """Extract filters from the query."""
        state.filters = FilterExtractor.extract(state.query)

        filter_str = []
        if state.filters.category:
            filter_str.append(f"category={state.filters.category}")
        if state.filters.brand:
            filter_str.append(f"brand={state.filters.brand}")
        if state.filters.price_min:
            filter_str.append(f"price_min=${state.filters.price_min}")
        if state.filters.price_max:
            filter_str.append(f"price_max=${state.filters.price_max}")
        if state.filters.min_rating:
            filter_str.append(f"rating>={state.filters.min_rating}")

        state.steps.append(f"filters:[{','.join(filter_str) or 'none'}]")

        return state

    async def _select_strategy(self, state: SearchAgentState) -> SearchAgentState:
        """Select the optimal search strategy based on analysis."""
        # Strategy selection based on query type and intent
        if state.query_type == QueryType.MODEL_NUMBER:
            state.search_strategy = "keyword"
        elif state.query_type == QueryType.BRAND_MODEL:
            state.search_strategy = "keyword"
        elif state.query_type == QueryType.GENERIC:
            state.search_strategy = "semantic"
        elif state.intent == SearchIntent.MODEL_LOOKUP:
            state.search_strategy = "keyword"
        elif state.intent == SearchIntent.BRAND_LOOKUP:
            state.search_strategy = "keyword"
        else:
            state.search_strategy = "hybrid"  # Default to best performer

        state.steps.append(f"strategy:{state.search_strategy}")

        return state

    async def _execute_search(self, state: SearchAgentState) -> SearchAgentState:
        """Execute the search using the registry-selected strategy."""
        # Build strategy filters (map local filter names to strategy filter names)
        strategy_filters = None
        if state.filters:
            strategy_filters = StrategyFilters(
                category=state.filters.category,
                brand=state.filters.brand,
                min_price=state.filters.price_min,  # Map price_min -> min_price
                max_price=state.filters.price_max,  # Map price_max -> max_price
                min_rating=state.filters.min_rating,
            )

        # Get the best strategy for this query from the registry
        query_to_use = state.clean_query or state.query
        strategy = await self.registry.get_strategy_for_query(query_to_use)

        logger.debug(
            "search_strategy_selected",
            query=query_to_use[:50],
            strategy=strategy.name,
            state_strategy=state.search_strategy,
        )

        # Get rerank setting from context, or fall back to database setting
        if "rerank" in state.context:
            use_rerank = state.context["rerank"]
            logger.info("rerank_from_context", enabled=use_rerank)
        else:
            # Check database setting for default rerank behavior
            try:
                from src.config.manager import get_config_manager
                config_manager = await get_config_manager()
                use_rerank = await config_manager.is_reranker_enabled()
                logger.info("rerank_from_db_setting", enabled=use_rerank)
            except Exception as e:
                logger.warning("rerank_db_setting_failed", error=str(e))
                use_rerank = False  # Default to False if DB fails

        # Execute search with the selected strategy
        response = await strategy.search(
            query=query_to_use,
            limit=10,
            filters=strategy_filters,
            rerank=use_rerank,
        )

        state.search_response = response
        state.search_results = [r.model_dump() for r in response.results]
        state.latency_ms = response.latency_ms
        state.total_results = len(response.results)
        state.search_strategy = strategy.name  # Update to actual strategy used

        # Reranking is now handled in the strategy via the rerank parameter
        # Log if reranking was used
        if use_rerank:
            state.steps.append(f"reranked:{len(state.search_results)}_results")

        state.steps.append(
            f"searched:{state.total_results}_results,{state.latency_ms:.0f}ms,strategy={strategy.name}"
        )

        return state

    async def _rerank_results(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rerank search results using configured reranker if enabled.

        Args:
            query: Original search query
            results: Search results to rerank

        Returns:
            Reranked results (or original if reranking disabled/failed)
        """
        logger.info("rerank_method_called", query=query[:30], num_results=len(results))
        try:
            from src.search.clients import get_search_clients

            clients = await get_search_clients()
            reranked = await clients.rerank_results(query, results)
            logger.debug("rerank_completed", original=len(results), reranked=len(reranked))
            return reranked
        except Exception as e:
            logger.warning("reranking_skipped", error=str(e))
            return results

    async def _format_results(self, state: SearchAgentState) -> SearchAgentState:
        """Format search results for output."""
        formatted = []

        for result in state.search_results:
            formatted.append({
                "asin": result.get("asin"),
                "title": result.get("title"),
                "brand": result.get("brand"),
                "price": result.get("price"),
                "stars": result.get("stars"),
                "score": result.get("score"),
                "img_url": result.get("img_url"),
                "summary": result.get("genAI_summary"),
                "best_for": result.get("genAI_best_for"),
            })

        state.formatted_results = formatted
        state.steps.append("formatted_results")

        return state

    async def _generate_summary(self, state: SearchAgentState) -> SearchAgentState:
        """Generate a brief summary of search results."""
        if not state.formatted_results:
            state.summary = "No products found matching your search criteria."
            return state

        # Build summary
        result_count = len(state.formatted_results)
        prices = [r["price"] for r in state.formatted_results if r.get("price")]
        ratings = [r["stars"] for r in state.formatted_results if r.get("stars")]

        summary_parts = [f"Found {result_count} products"]

        if prices:
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            summary_parts.append(f"priced ${min_price:.2f}-${max_price:.2f} (avg ${avg_price:.2f})")

        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            summary_parts.append(f"with average rating {avg_rating:.1f}★")

        # Add top brands
        brands = [r["brand"] for r in state.formatted_results if r.get("brand")]
        if brands:
            unique_brands = list(dict.fromkeys(brands))[:3]
            summary_parts.append(f"from {', '.join(unique_brands)}")

        state.summary = " ".join(summary_parts) + "."
        state.steps.append("summary_generated")

        return state


# =============================================================================
# LangGraph Integration
# =============================================================================

def create_search_agent_graph() -> StateGraph:
    """Create a LangGraph workflow for the search agent."""

    async def analyze_node(state: dict) -> dict:
        """Analyze query node."""
        analysis = QueryAnalyzer.analyze(state["query"])
        state["query_type"] = analysis.query_type.value
        state["clean_query"] = analysis.clean_query
        state["has_model"] = bool(analysis.model_numbers)
        state["has_brand"] = analysis.has_brand
        state["semantic_weight"] = analysis.semantic_weight
        state["keyword_weight"] = analysis.keyword_weight
        state["steps"].append(f"analyzed:{analysis.query_type.value}")
        return state

    async def extract_filters_node(state: dict) -> dict:
        """Extract filters node."""
        filters = FilterExtractor.extract(state["query"])
        state["filters"] = filters.model_dump()
        state["steps"].append("filters_extracted")
        return state

    async def search_node(state: dict) -> dict:
        """Execute search node using strategy registry."""
        from src.config.repository import get_config_repository
        from src.search import get_search_registry, SearchFilters as StrategyFilters

        config_repo = await get_config_repository()
        registry = await get_search_registry(config_repo)

        # Build strategy filters (map local filter names to strategy filter names)
        filters_dict = state.get("filters", {})
        strategy_filters = None
        if any(filters_dict.values()):
            strategy_filters = StrategyFilters(
                category=filters_dict.get("category"),
                brand=filters_dict.get("brand"),
                min_price=filters_dict.get("price_min"),  # Map price_min -> min_price
                max_price=filters_dict.get("price_max"),  # Map price_max -> max_price
                min_rating=filters_dict.get("min_rating"),
            )

        # Get best strategy for query and execute
        strategy = await registry.get_strategy_for_query(state["query"])

        # Get rerank setting from context
        context = state.get("context", {})
        rerank = context.get("rerank", True)  # Default to True

        response = await strategy.search(
            query=state["query"],
            limit=10,
            filters=strategy_filters,
            rerank=rerank,
        )

        state["search_results"] = [r.model_dump() for r in response.results]
        state["latency_ms"] = response.latency_ms
        state["total_results"] = len(response.results)
        state["strategy_used"] = strategy.name
        state["steps"].append(f"searched:{len(response.results)}_results,strategy={strategy.name},rerank={rerank}")

        return state

    async def format_node(state: dict) -> dict:
        """Format results node."""
        formatted = []
        for r in state.get("search_results", []):
            formatted.append({
                "asin": r.get("asin"),
                "title": r.get("title"),
                "brand": r.get("brand"),
                "price": r.get("price"),
                "stars": r.get("stars"),
                "score": r.get("score"),
            })
        state["products"] = formatted
        state["steps"].append("formatted")
        return state

    # Build graph
    from langgraph.graph import StateGraph, END
    from typing import TypedDict

    class SearchState(TypedDict):
        query: str
        query_type: str | None
        clean_query: str
        has_model: bool
        has_brand: bool
        semantic_weight: float
        keyword_weight: float
        filters: dict
        search_results: list
        latency_ms: float
        total_results: int
        strategy_used: str | None
        products: list
        steps: list
        error: str | None

    workflow = StateGraph(SearchState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("extract_filters", extract_filters_node)
    workflow.add_node("search", search_node)
    workflow.add_node("format", format_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "extract_filters")
    workflow.add_edge("extract_filters", "search")
    workflow.add_edge("search", "format")
    workflow.add_edge("format", END)

    return workflow.compile()


# =============================================================================
# Convenience Functions
# =============================================================================

# Singleton instance
_search_agent: SearchAgent | None = None


async def get_search_agent() -> SearchAgent:
    """Get or create search agent singleton."""
    global _search_agent
    if _search_agent is None:
        _search_agent = SearchAgent()
        await _search_agent.initialize()
    return _search_agent


async def search(query: str, context: dict[str, Any] | None = None) -> SearchAgentState:
    """
    Convenience function to execute a search.

    Args:
        query: User search query
        context: Optional context

    Returns:
        SearchAgentState with results
    """
    agent = await get_search_agent()
    return await agent.search(query, context)


async def search_products(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Simple function to search products.

    Args:
        query: Search query
        limit: Max results
        filters: Optional filters

    Returns:
        List of product dicts
    """
    agent = await get_search_agent()
    state = await agent.search(query)
    return state.formatted_results[:limit]
