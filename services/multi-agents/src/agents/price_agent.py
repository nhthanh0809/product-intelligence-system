"""Price agent for price intelligence."""

import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.agents.base import BaseAgent, RetryConfig
from src.tools.search_tools import SearchToolkit, SearchToolConfig

logger = structlog.get_logger()


@dataclass
class PriceInput:
    """Input for price agent."""
    query: str
    products: list[dict[str, Any]] = field(default_factory=list)
    target_price: float | None = None
    category: str | None = None


@dataclass
class PriceAnalysis:
    """Price analysis for a product."""
    asin: str
    title: str
    current_price: float | None
    list_price: float | None = None
    discount_pct: float | None = None
    price_rating: str = ""  # "great deal", "good", "fair", "overpriced"
    value_score: int = 0  # 1-10
    category_percentile: float | None = None  # Where it falls in category prices


@dataclass
class CategoryPriceStats:
    """Price statistics for a category."""
    category: str
    min_price: float | None = None
    max_price: float | None = None
    avg_price: float | None = None
    median_price: float | None = None
    budget_threshold: float | None = None  # Under this = budget
    premium_threshold: float | None = None  # Over this = premium


@dataclass
class PriceOutput:
    """Output from price agent."""
    query: str
    products: list[PriceAnalysis] = field(default_factory=list)
    category_stats: CategoryPriceStats | None = None
    best_deal: PriceAnalysis | None = None
    best_value: PriceAnalysis | None = None
    recommendation: str = ""
    summary: str = ""


class PriceAgent(BaseAgent[PriceInput, PriceOutput]):
    """Price agent for price intelligence.

    Handles scenarios P1-P5:
    - P1: Price range search
    - P2: Deal detection
    - P3: Value assessment
    - P4: Price comparison
    - P5: Budget recommendations
    """

    name = "price"
    description = "Price intelligence and deal detection"

    def __init__(self):
        super().__init__(retry_config=RetryConfig(max_retries=2))
        self._http_client: httpx.AsyncClient | None = None
        self._search_toolkit: SearchToolkit | None = None

    async def initialize(self) -> None:
        """Initialize clients."""
        self._http_client = httpx.AsyncClient(
            base_url=self.settings.ollama_service_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        )

        config = SearchToolConfig.from_settings()
        self._search_toolkit = SearchToolkit(config)
        await self._search_toolkit.initialize()

        await super().initialize()

    async def close(self) -> None:
        """Close clients."""
        if self._http_client:
            await self._http_client.aclose()
        if self._search_toolkit:
            await self._search_toolkit.close()

    async def _execute_internal(
        self,
        input_data: PriceInput,
    ) -> PriceOutput:
        """Execute price analysis."""
        products = input_data.products

        # Extract price constraints from query if not provided
        if input_data.target_price is None:
            input_data.target_price = self._extract_price_from_query(input_data.query)

        # If no products, search with price filters
        if not products:
            products = await self._search_with_price_filters(
                input_data.query,
                input_data.target_price,
                input_data.category,
            )

        if not products:
            return PriceOutput(
                query=input_data.query,
                summary="No products found matching price criteria.",
            )

        # Analyze prices
        analyzed_products = []
        category_prices = []

        for product in products[:10]:
            analysis = self._analyze_product_price(product)
            analyzed_products.append(analysis)
            if analysis.current_price:
                category_prices.append(analysis.current_price)

        # Calculate category stats
        category_stats = None
        if category_prices and input_data.category:
            category_stats = self._calculate_category_stats(
                input_data.category,
                category_prices,
            )

        # Find best deal and best value
        best_deal = self._find_best_deal(analyzed_products)
        best_value = self._find_best_value(analyzed_products)

        # Generate recommendation
        recommendation = await self._generate_recommendation(
            input_data.query,
            analyzed_products,
            input_data.target_price,
        )

        # Build summary
        summary = self._build_summary(
            analyzed_products,
            best_deal,
            best_value,
            input_data.target_price,
        )

        return PriceOutput(
            query=input_data.query,
            products=analyzed_products,
            category_stats=category_stats,
            best_deal=best_deal,
            best_value=best_value,
            recommendation=recommendation,
            summary=summary,
        )

    def _extract_price_from_query(self, query: str) -> float | None:
        """Extract target price from query."""
        patterns = [
            (r'under\s*\$?(\d+)', "max"),
            (r'below\s*\$?(\d+)', "max"),
            (r'around\s*\$?(\d+)', "target"),
            (r'about\s*\$?(\d+)', "target"),
            (r'\$(\d+)', "target"),
        ]

        for pattern, _ in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    async def _search_with_price_filters(
        self,
        query: str,
        target_price: float | None,
        category: str | None,
    ) -> list[dict[str, Any]]:
        """Search products with price filters."""
        filters = {}

        if target_price:
            # Set price range around target
            filters["price_max"] = target_price * 1.2
            filters["price_min"] = target_price * 0.5

        if category:
            filters["category_level1"] = category

        results = await self._search_toolkit.hybrid_search(
            query,
            limit=10,
            filters=filters if filters else None,
        )

        return [r.model_dump() for r in results.results]

    def _analyze_product_price(
        self,
        product: dict[str, Any],
    ) -> PriceAnalysis:
        """Analyze price for a single product."""
        current_price = product.get("price")
        list_price = product.get("list_price") or product.get("original_price")
        stars = product.get("stars")

        # Calculate discount
        discount_pct = None
        if current_price and list_price and list_price > current_price:
            discount_pct = ((list_price - current_price) / list_price) * 100

        # Determine price rating based on discount and reviews
        price_rating = "fair"
        if discount_pct:
            if discount_pct >= 30:
                price_rating = "great deal"
            elif discount_pct >= 15:
                price_rating = "good"

        # Calculate value score (simplified)
        value_score = 5
        if stars:
            value_score = min(10, int(stars * 2))
        if discount_pct and discount_pct >= 20:
            value_score = min(10, value_score + 2)

        return PriceAnalysis(
            asin=product.get("asin", ""),
            title=product.get("title", "")[:80],
            current_price=current_price,
            list_price=list_price,
            discount_pct=discount_pct,
            price_rating=price_rating,
            value_score=value_score,
        )

    def _calculate_category_stats(
        self,
        category: str,
        prices: list[float],
    ) -> CategoryPriceStats:
        """Calculate price statistics for category."""
        prices = sorted(prices)

        return CategoryPriceStats(
            category=category,
            min_price=min(prices),
            max_price=max(prices),
            avg_price=sum(prices) / len(prices),
            median_price=prices[len(prices) // 2],
            budget_threshold=prices[int(len(prices) * 0.25)] if len(prices) > 3 else None,
            premium_threshold=prices[int(len(prices) * 0.75)] if len(prices) > 3 else None,
        )

    def _find_best_deal(
        self,
        products: list[PriceAnalysis],
    ) -> PriceAnalysis | None:
        """Find product with best discount."""
        with_discount = [p for p in products if p.discount_pct and p.discount_pct > 0]
        if not with_discount:
            return None
        return max(with_discount, key=lambda p: p.discount_pct or 0)

    def _find_best_value(
        self,
        products: list[PriceAnalysis],
    ) -> PriceAnalysis | None:
        """Find product with best value score."""
        with_scores = [p for p in products if p.value_score > 0]
        if not with_scores:
            return None
        return max(with_scores, key=lambda p: p.value_score)

    async def _generate_recommendation(
        self,
        query: str,
        products: list[PriceAnalysis],
        target_price: float | None,
    ) -> str:
        """Generate price-based recommendation."""
        if not products:
            return "No products to recommend."

        # Build context
        context_parts = []
        for p in products[:5]:
            parts = [f"- {p.title}: ${p.current_price}"]
            if p.discount_pct:
                parts.append(f" ({p.discount_pct:.0f}% off)")
            parts.append(f" - Value: {p.value_score}/10, Rating: {p.price_rating}")
            context_parts.append("".join(parts))

        context = "\n".join(context_parts)

        prompt = f"""Based on the user's price query, provide a brief recommendation.

Query: {query}
Target Price: ${target_price if target_price else 'Not specified'}

Products:
{context}

Provide a 2-3 sentence recommendation focusing on price and value."""

        response = await self._http_client.post(
            "/generate",
            json={
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 150,
            },
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _build_summary(
        self,
        products: list[PriceAnalysis],
        best_deal: PriceAnalysis | None,
        best_value: PriceAnalysis | None,
        target_price: float | None,
    ) -> str:
        """Build price summary."""
        parts = [f"Found {len(products)} products"]

        prices = [p.current_price for p in products if p.current_price]
        if prices:
            parts.append(f"priced ${min(prices):.2f} - ${max(prices):.2f}")

        if best_deal and best_deal.discount_pct:
            parts.append(f"Best deal: {best_deal.discount_pct:.0f}% off")

        if best_value:
            parts.append(f"Best value score: {best_value.value_score}/10")

        return ". ".join(parts) + "."


# Singleton
_price_agent: PriceAgent | None = None


async def get_price_agent() -> PriceAgent:
    """Get or create price agent singleton."""
    global _price_agent
    if _price_agent is None:
        _price_agent = PriceAgent()
        await _price_agent.initialize()
    return _price_agent


async def analyze_prices(
    query: str,
    products: list[dict[str, Any]] | None = None,
    target_price: float | None = None,
) -> PriceOutput:
    """Convenience function for price analysis.

    Args:
        query: Price query
        products: Optional products to analyze
        target_price: Optional target price

    Returns:
        PriceOutput with analysis
    """
    agent = await get_price_agent()
    return await agent.execute(PriceInput(
        query=query,
        products=products or [],
        target_price=target_price,
    ))
