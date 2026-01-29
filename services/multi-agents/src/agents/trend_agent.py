"""Trend agent for market analysis."""

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.agents.base import BaseAgent, RetryConfig
from src.tools.postgres_tools import PostgresToolkit

logger = structlog.get_logger()


@dataclass
class TrendInput:
    """Input for trend agent."""
    query: str
    category: str | None = None
    time_range: str = "7d"  # 7d, 30d, 90d


@dataclass
class TrendingProduct:
    """A trending product."""
    asin: str
    title: str
    brand: str | None = None
    trend_score: float = 0.0
    review_velocity: int = 0  # Reviews per time period
    price: float | None = None
    stars: float | None = None
    bought_in_last_month: int | None = None
    rank_change: int | None = None  # Positive = improving


@dataclass
class CategoryTrend:
    """Trend data for a category."""
    category: str
    product_count: int = 0
    avg_price: float | None = None
    avg_rating: float | None = None
    top_brands: list[str] = field(default_factory=list)
    trending_keywords: list[str] = field(default_factory=list)


@dataclass
class TrendOutput:
    """Output from trend agent."""
    query: str
    trending_products: list[TrendingProduct] = field(default_factory=list)
    category_trends: list[CategoryTrend] = field(default_factory=list)
    hot_categories: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    summary: str = ""


class TrendAgent(BaseAgent[TrendInput, TrendOutput]):
    """Trend agent for market analysis.

    Handles scenarios T1-T5:
    - T1: Trending products
    - T2: Category trends
    - T3: Brand popularity
    - T4: Price trends
    - T5: Market insights
    """

    name = "trend"
    description = "Market trend analysis and trending products"

    def __init__(self):
        super().__init__(retry_config=RetryConfig(max_retries=2))
        self._http_client: httpx.AsyncClient | None = None
        self._pg_toolkit: PostgresToolkit | None = None

    async def initialize(self) -> None:
        """Initialize clients."""
        self._http_client = httpx.AsyncClient(
            base_url=self.settings.ollama_service_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        )

        self._pg_toolkit = PostgresToolkit()
        await self._pg_toolkit.initialize()

        await super().initialize()

    async def close(self) -> None:
        """Close clients."""
        if self._http_client:
            await self._http_client.aclose()
        if self._pg_toolkit:
            await self._pg_toolkit.close()

    async def _execute_internal(
        self,
        input_data: TrendInput,
    ) -> TrendOutput:
        """Execute trend analysis."""
        # Get trending products
        trending_products = await self._get_trending_products(
            input_data.category,
            input_data.time_range,
        )

        # Get category trends
        category_trends = await self._get_category_trends(input_data.category)

        # Get hot categories
        hot_categories = await self._get_hot_categories()

        # Generate insights
        insights = await self._generate_insights(
            input_data.query,
            trending_products,
            category_trends,
        )

        # Build summary
        summary = self._build_summary(
            trending_products,
            category_trends,
            input_data.category,
        )

        return TrendOutput(
            query=input_data.query,
            trending_products=trending_products,
            category_trends=category_trends,
            hot_categories=hot_categories,
            insights=insights,
            summary=summary,
        )

    async def _get_trending_products(
        self,
        category: str | None,
        time_range: str,
    ) -> list[TrendingProduct]:
        """Get trending products from PostgreSQL."""
        # Use PostgreSQL function or query
        days = {"7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)

        query = """
            SELECT
                p.asin,
                p.title,
                p.brand,
                p.price,
                p.stars,
                p.bought_in_last_month,
                p.reviews_count,
                COALESCE(pt.trend_score, p.popularity_score, 0) as trend_score,
                COALESCE(pt.review_velocity, 0) as review_velocity
            FROM products p
            LEFT JOIN product_trends pt ON p.asin = pt.asin
                AND pt.date >= CURRENT_DATE - INTERVAL '%s days'
            WHERE p.stars >= 3.5
                AND p.reviews_count >= 10
        """

        params = [days]

        if category:
            query += " AND p.category_level1 = $2"
            params.append(category)

        query += """
            ORDER BY
                p.bought_in_last_month DESC NULLS LAST,
                p.reviews_count DESC,
                p.stars DESC
            LIMIT 20
        """

        try:
            results = await self._pg_toolkit.fetch(query, *params)
        except Exception as e:
            logger.warning("trending_query_failed", error=str(e))
            # Fallback: simpler query
            results = await self._get_trending_fallback(category)

        trending = []
        for row in results:
            trending.append(TrendingProduct(
                asin=row.get("asin", ""),
                title=row.get("title", "")[:80],
                brand=row.get("brand"),
                trend_score=float(row.get("trend_score", 0)),
                review_velocity=int(row.get("review_velocity", 0)),
                price=row.get("price"),
                stars=row.get("stars"),
                bought_in_last_month=row.get("bought_in_last_month"),
            ))

        return trending

    async def _get_trending_fallback(
        self,
        category: str | None,
    ) -> list[dict]:
        """Fallback trending query."""
        query = """
            SELECT
                asin, title, brand, price, stars,
                bought_in_last_month, reviews_count,
                0 as trend_score, 0 as review_velocity
            FROM products
            WHERE stars >= 4.0
        """
        params = []

        if category:
            query += " AND category_level1 = $1"
            params.append(category)

        query += """
            ORDER BY bought_in_last_month DESC NULLS LAST, reviews_count DESC
            LIMIT 20
        """

        return await self._pg_toolkit.fetch(query, *params)

    async def _get_category_trends(
        self,
        category: str | None,
    ) -> list[CategoryTrend]:
        """Get category trend data."""
        query = """
            SELECT
                category_level1 as category,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating
            FROM products
            WHERE category_level1 IS NOT NULL
        """
        params = []

        if category:
            query += " AND category_level1 = $1"
            params.append(category)

        query += """
            GROUP BY category_level1
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """

        results = await self._pg_toolkit.fetch(query, *params)

        trends = []
        for row in results:
            # Get top brands for category
            brand_query = """
                SELECT brand, COUNT(*) as count
                FROM products
                WHERE category_level1 = $1 AND brand IS NOT NULL
                GROUP BY brand
                ORDER BY count DESC
                LIMIT 5
            """
            brands_result = await self._pg_toolkit.fetch(
                brand_query, row["category"]
            )
            top_brands = [b["brand"] for b in brands_result]

            trends.append(CategoryTrend(
                category=row["category"],
                product_count=int(row["product_count"]),
                avg_price=float(row["avg_price"]) if row["avg_price"] else None,
                avg_rating=float(row["avg_rating"]) if row["avg_rating"] else None,
                top_brands=top_brands,
            ))

        return trends

    async def _get_hot_categories(self) -> list[str]:
        """Get categories with most activity."""
        query = """
            SELECT category_level1
            FROM products
            WHERE category_level1 IS NOT NULL
                AND bought_in_last_month > 0
            GROUP BY category_level1
            ORDER BY SUM(bought_in_last_month) DESC
            LIMIT 5
        """

        results = await self._pg_toolkit.fetch(query)
        return [r["category_level1"] for r in results]

    async def _generate_insights(
        self,
        query: str,
        trending: list[TrendingProduct],
        categories: list[CategoryTrend],
    ) -> list[str]:
        """Generate market insights using LLM."""
        # Build context
        trending_text = "\n".join([
            f"- {p.title[:50]} ({p.brand}): {p.bought_in_last_month or 0} bought/month"
            for p in trending[:5]
        ])

        category_text = "\n".join([
            f"- {c.category}: {c.product_count} products, avg ${c.avg_price:.2f}"
            for c in categories[:5]
        ])

        prompt = f"""Based on this market data, provide 3-4 brief insights.

User Query: {query}

Trending Products:
{trending_text}

Category Data:
{category_text}

Provide actionable insights about:
1. What's popular and why
2. Price trends
3. Brand patterns
4. Recommendations

Format as a numbered list of insights."""

        response = await self._http_client.post(
            "/generate",
            json={
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 300,
            },
        )
        response.raise_for_status()

        raw = response.json().get("response", "")

        # Parse insights
        insights = []
        for line in raw.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                insight = line.lstrip('0123456789.-) ').strip()
                if insight:
                    insights.append(insight)

        return insights[:5]

    def _build_summary(
        self,
        trending: list[TrendingProduct],
        categories: list[CategoryTrend],
        category: str | None,
    ) -> str:
        """Build trend summary."""
        parts = []

        if trending:
            top = trending[0]
            parts.append(f"Top trending: {top.title[:40]} by {top.brand}")

        if categories:
            cat = categories[0]
            parts.append(f"Most active category: {cat.category} ({cat.product_count} products)")

        if category:
            cat_data = next((c for c in categories if c.category == category), None)
            if cat_data and cat_data.avg_price:
                parts.append(f"Average price in {category}: ${cat_data.avg_price:.2f}")

        return ". ".join(parts) + "." if parts else "No trend data available."


# Singleton
_trend_agent: TrendAgent | None = None


async def get_trend_agent() -> TrendAgent:
    """Get or create trend agent singleton."""
    global _trend_agent
    if _trend_agent is None:
        _trend_agent = TrendAgent()
        await _trend_agent.initialize()
    return _trend_agent


async def get_trends(
    query: str,
    category: str | None = None,
    time_range: str = "7d",
) -> TrendOutput:
    """Convenience function for trend analysis.

    Args:
        query: Trend query
        category: Optional category filter
        time_range: Time range for trends

    Returns:
        TrendOutput with analysis
    """
    agent = await get_trend_agent()
    return await agent.execute(TrendInput(
        query=query,
        category=category,
        time_range=time_range,
    ))
