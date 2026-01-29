"""PostgreSQL tools for multi-agent system."""

from typing import Any

import asyncpg
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class PostgresToolkit:
    """PostgreSQL toolkit for agent queries.

    Provides tools for:
    - Product queries by ASIN
    - Category statistics
    - Trend data retrieval
    - Price history
    - Brand analytics
    """

    def __init__(self):
        self.settings = get_settings()
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                database=self.settings.postgres_db,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            logger.info("postgres_toolkit_initialized")
        except Exception as e:
            logger.error("postgres_toolkit_init_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute query and return results as dicts."""
        if not self._pool:
            await self.initialize()

        records = await self._pool.fetch(query, *args)
        return [dict(r) for r in records]

    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        """Fetch a single row as dict."""
        if not self._pool:
            await self.initialize()

        record = await self._pool.fetchrow(query, *args)
        return dict(record) if record else None

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        if not self._pool:
            await self.initialize()

        return await self._pool.fetchval(query, *args)

    # =============================================================================
    # Product Tools
    # =============================================================================

    async def get_product(self, asin: str) -> dict[str, Any] | None:
        """Get product by ASIN."""
        query = """
            SELECT *
            FROM products
            WHERE asin = $1
        """
        return await self.fetchrow(query, asin)

    async def get_products_by_asins(
        self,
        asins: list[str],
    ) -> list[dict[str, Any]]:
        """Get multiple products by ASINs."""
        if not asins:
            return []

        query = """
            SELECT *
            FROM products
            WHERE asin = ANY($1)
        """
        return await self.fetch(query, asins)

    async def search_products(
        self,
        category: str | None = None,
        brand: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        min_rating: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search products with filters."""
        conditions = []
        params = []
        param_idx = 1

        if category:
            conditions.append(f"category_level1 = ${param_idx}")
            params.append(category)
            param_idx += 1

        if brand:
            conditions.append(f"brand = ${param_idx}")
            params.append(brand)
            param_idx += 1

        if price_min is not None:
            conditions.append(f"price >= ${param_idx}")
            params.append(price_min)
            param_idx += 1

        if price_max is not None:
            conditions.append(f"price <= ${param_idx}")
            params.append(price_max)
            param_idx += 1

        if min_rating is not None:
            conditions.append(f"stars >= ${param_idx}")
            params.append(min_rating)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT asin, title, brand, price, stars, reviews_count,
                   category_level1, is_best_seller, img_url
            FROM products
            WHERE {where_clause}
            ORDER BY reviews_count DESC NULLS LAST, stars DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit)

        return await self.fetch(query, *params)

    # =============================================================================
    # Trend Tools
    # =============================================================================

    async def get_trending_products(
        self,
        category: str | None = None,
        days: int = 7,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get trending products."""
        query = """
            SELECT
                p.asin,
                p.title,
                p.brand,
                p.price,
                p.stars,
                p.reviews_count,
                p.bought_in_last_month,
                COALESCE(pt.trend_score, 0) as trend_score,
                COALESCE(pt.review_velocity, 0) as review_velocity
            FROM products p
            LEFT JOIN product_trends pt ON p.asin = pt.asin
                AND pt.date >= CURRENT_DATE - $1::interval
            WHERE p.stars >= 3.5
                AND p.reviews_count >= 10
        """
        params = [f"{days} days"]

        if category:
            query += " AND p.category_level1 = $2"
            params.append(category)

        query += """
            ORDER BY
                p.bought_in_last_month DESC NULLS LAST,
                p.reviews_count DESC,
                p.stars DESC
            LIMIT $3
        """
        params.append(limit)

        return await self.fetch(query, *params)

    async def get_category_stats(
        self,
        category: str,
    ) -> dict[str, Any] | None:
        """Get statistics for a category."""
        query = """
            SELECT
                category_level1 as category,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(stars) as avg_rating,
                SUM(reviews_count) as total_reviews,
                COUNT(CASE WHEN is_best_seller THEN 1 END) as bestseller_count
            FROM products
            WHERE category_level1 = $1 AND price IS NOT NULL
            GROUP BY category_level1
        """
        return await self.fetchrow(query, category)

    async def get_top_categories(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get top categories by product count."""
        query = """
            SELECT
                category_level1 as category,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating
            FROM products
            WHERE category_level1 IS NOT NULL
            GROUP BY category_level1
            ORDER BY COUNT(*) DESC
            LIMIT $1
        """
        return await self.fetch(query, limit)

    # =============================================================================
    # Price Tools
    # =============================================================================

    async def get_price_history(
        self,
        asin: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get price history for a product."""
        query = """
            SELECT
                price,
                original_price,
                discount_percentage,
                recorded_at
            FROM price_history
            WHERE asin = $1
                AND recorded_at >= CURRENT_DATE - $2::interval
            ORDER BY recorded_at DESC
        """
        return await self.fetch(query, asin, f"{days} days")

    async def get_category_price_stats(
        self,
        category: str,
    ) -> dict[str, Any] | None:
        """Get price statistics for a category."""
        # This uses the PostgreSQL function defined in schema
        query = "SELECT * FROM get_category_price_stats($1)"
        return await self.fetchrow(query, category)

    # =============================================================================
    # Brand Tools
    # =============================================================================

    async def get_brand_stats(self, brand: str) -> dict[str, Any] | None:
        """Get statistics for a brand."""
        query = """
            SELECT
                brand,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating,
                SUM(reviews_count) as total_reviews,
                COUNT(DISTINCT category_level1) as category_count
            FROM products
            WHERE brand = $1
            GROUP BY brand
        """
        return await self.fetchrow(query, brand)

    async def get_top_brands_in_category(
        self,
        category: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top brands in a category."""
        query = """
            SELECT
                brand,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating,
                COUNT(CASE WHEN is_best_seller THEN 1 END) as bestseller_count
            FROM products
            WHERE category_level1 = $1 AND brand IS NOT NULL
            GROUP BY brand
            ORDER BY COUNT(*) DESC, AVG(stars) DESC
            LIMIT $2
        """
        return await self.fetch(query, category, limit)

    # =============================================================================
    # Review Tools
    # =============================================================================

    async def get_product_reviews(
        self,
        asin: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get reviews for a product."""
        query = """
            SELECT
                title,
                body,
                rating,
                author_name,
                verified_purchase,
                review_date,
                helpful_votes,
                sentiment_label
            FROM reviews
            WHERE asin = $1
            ORDER BY helpful_votes DESC, review_date DESC
            LIMIT $2
        """
        return await self.fetch(query, asin, limit)

    async def get_review_sentiment_summary(
        self,
        asin: str,
    ) -> dict[str, Any] | None:
        """Get sentiment summary for product reviews."""
        query = """
            SELECT
                COUNT(*) as total_reviews,
                AVG(rating) as avg_rating,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END) as positive_count,
                COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END) as negative_count,
                COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END) as neutral_count
            FROM reviews
            WHERE asin = $1
        """
        return await self.fetchrow(query, asin)


# Singleton
_pg_toolkit: PostgresToolkit | None = None


async def get_postgres_toolkit() -> PostgresToolkit:
    """Get or create PostgreSQL toolkit singleton."""
    global _pg_toolkit
    if _pg_toolkit is None:
        _pg_toolkit = PostgresToolkit()
        await _pg_toolkit.initialize()
    return _pg_toolkit
