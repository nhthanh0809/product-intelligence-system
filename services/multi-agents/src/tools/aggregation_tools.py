"""Aggregation tools for brand/category statistics.

Provides:
- Brand statistics and rankings
- Category analysis
- Price distribution
- Rating aggregation
- Trend aggregation
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import get_settings

logger = structlog.get_logger()


@dataclass
class BrandStats:
    """Brand statistics."""
    brand: str
    product_count: int = 0
    avg_price: float | None = None
    avg_rating: float | None = None
    total_reviews: int = 0
    price_range: tuple[float, float] | None = None
    top_categories: list[str] = field(default_factory=list)


@dataclass
class CategoryStats:
    """Category statistics."""
    category: str
    product_count: int = 0
    avg_price: float | None = None
    avg_rating: float | None = None
    total_reviews: int = 0
    price_range: tuple[float, float] | None = None
    top_brands: list[str] = field(default_factory=list)


@dataclass
class PriceDistribution:
    """Price distribution stats."""
    min_price: float
    max_price: float
    avg_price: float
    median_price: float
    percentiles: dict[int, float] = field(default_factory=dict)


class AggregationToolkit:
    """Tools for data aggregation and statistics."""

    def __init__(self, postgres_client: Any = None, search_toolkit: Any = None):
        self._postgres = postgres_client
        self._search = search_toolkit
        self.settings = get_settings()

    async def initialize(self) -> None:
        """Initialize connections."""
        if self._postgres is None:
            import asyncpg
            self._postgres = await asyncpg.create_pool(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                database=self.settings.postgres_db,
                min_size=1,
                max_size=5,
            )

    async def close(self) -> None:
        """Close connections."""
        if self._postgres:
            await self._postgres.close()

    async def get_brand_stats(self, brand: str) -> BrandStats | None:
        """Get statistics for a brand.

        Args:
            brand: Brand name

        Returns:
            BrandStats or None if not found
        """
        if not self._postgres:
            await self.initialize()

        query = """
            SELECT
                brand,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating,
                SUM(reviews_count) as total_reviews,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM products
            WHERE LOWER(brand) = LOWER($1)
            GROUP BY brand
        """

        try:
            row = await self._postgres.fetchrow(query, brand)
            if not row:
                return None

            # Get top categories for brand
            cat_query = """
                SELECT category, COUNT(*) as cnt
                FROM products
                WHERE LOWER(brand) = LOWER($1) AND category IS NOT NULL
                GROUP BY category
                ORDER BY cnt DESC
                LIMIT 5
            """
            cat_rows = await self._postgres.fetch(cat_query, brand)
            top_categories = [r["category"] for r in cat_rows]

            return BrandStats(
                brand=row["brand"] or brand,
                product_count=row["product_count"],
                avg_price=float(row["avg_price"]) if row["avg_price"] else None,
                avg_rating=float(row["avg_rating"]) if row["avg_rating"] else None,
                total_reviews=row["total_reviews"] or 0,
                price_range=(
                    float(row["min_price"]),
                    float(row["max_price"]),
                ) if row["min_price"] and row["max_price"] else None,
                top_categories=top_categories,
            )

        except Exception as e:
            logger.error("brand_stats_failed", brand=brand, error=str(e))
            return None

    async def get_category_stats(self, category: str) -> CategoryStats | None:
        """Get statistics for a category.

        Args:
            category: Category name

        Returns:
            CategoryStats or None if not found
        """
        if not self._postgres:
            await self.initialize()

        query = """
            SELECT
                category,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating,
                SUM(reviews_count) as total_reviews,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM products
            WHERE LOWER(category) LIKE LOWER($1)
            GROUP BY category
        """

        try:
            # Use LIKE for partial match
            row = await self._postgres.fetchrow(query, f"%{category}%")
            if not row:
                return None

            # Get top brands for category
            brand_query = """
                SELECT brand, COUNT(*) as cnt
                FROM products
                WHERE LOWER(category) LIKE LOWER($1) AND brand IS NOT NULL
                GROUP BY brand
                ORDER BY cnt DESC
                LIMIT 5
            """
            brand_rows = await self._postgres.fetch(brand_query, f"%{category}%")
            top_brands = [r["brand"] for r in brand_rows]

            return CategoryStats(
                category=row["category"] or category,
                product_count=row["product_count"],
                avg_price=float(row["avg_price"]) if row["avg_price"] else None,
                avg_rating=float(row["avg_rating"]) if row["avg_rating"] else None,
                total_reviews=row["total_reviews"] or 0,
                price_range=(
                    float(row["min_price"]),
                    float(row["max_price"]),
                ) if row["min_price"] and row["max_price"] else None,
                top_brands=top_brands,
            )

        except Exception as e:
            logger.error("category_stats_failed", category=category, error=str(e))
            return None

    async def get_top_brands(
        self,
        category: str | None = None,
        limit: int = 10,
        sort_by: str = "product_count",
    ) -> list[BrandStats]:
        """Get top brands by various metrics.

        Args:
            category: Optional category filter
            limit: Number of brands to return
            sort_by: Sort metric (product_count, avg_rating, total_reviews)

        Returns:
            List of BrandStats
        """
        if not self._postgres:
            await self.initialize()

        # Build query
        where_clause = ""
        params = []
        if category:
            where_clause = "WHERE LOWER(category) LIKE LOWER($1)"
            params.append(f"%{category}%")

        order_by = {
            "product_count": "product_count DESC",
            "avg_rating": "avg_rating DESC NULLS LAST",
            "total_reviews": "total_reviews DESC",
        }.get(sort_by, "product_count DESC")

        query = f"""
            SELECT
                brand,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating,
                SUM(reviews_count) as total_reviews
            FROM products
            {where_clause}
            GROUP BY brand
            HAVING brand IS NOT NULL
            ORDER BY {order_by}
            LIMIT {limit}
        """

        try:
            rows = await self._postgres.fetch(query, *params)
            return [
                BrandStats(
                    brand=row["brand"],
                    product_count=row["product_count"],
                    avg_price=float(row["avg_price"]) if row["avg_price"] else None,
                    avg_rating=float(row["avg_rating"]) if row["avg_rating"] else None,
                    total_reviews=row["total_reviews"] or 0,
                )
                for row in rows
            ]

        except Exception as e:
            logger.error("top_brands_failed", error=str(e))
            return []

    async def get_top_categories(
        self,
        limit: int = 10,
        sort_by: str = "product_count",
    ) -> list[CategoryStats]:
        """Get top categories by various metrics.

        Args:
            limit: Number of categories to return
            sort_by: Sort metric (product_count, avg_rating, total_reviews)

        Returns:
            List of CategoryStats
        """
        if not self._postgres:
            await self.initialize()

        order_by = {
            "product_count": "product_count DESC",
            "avg_rating": "avg_rating DESC NULLS LAST",
            "total_reviews": "total_reviews DESC",
        }.get(sort_by, "product_count DESC")

        query = f"""
            SELECT
                category,
                COUNT(*) as product_count,
                AVG(price) as avg_price,
                AVG(stars) as avg_rating,
                SUM(reviews_count) as total_reviews
            FROM products
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY {order_by}
            LIMIT {limit}
        """

        try:
            rows = await self._postgres.fetch(query)
            return [
                CategoryStats(
                    category=row["category"],
                    product_count=row["product_count"],
                    avg_price=float(row["avg_price"]) if row["avg_price"] else None,
                    avg_rating=float(row["avg_rating"]) if row["avg_rating"] else None,
                    total_reviews=row["total_reviews"] or 0,
                )
                for row in rows
            ]

        except Exception as e:
            logger.error("top_categories_failed", error=str(e))
            return []

    async def get_price_distribution(
        self,
        category: str | None = None,
        brand: str | None = None,
    ) -> PriceDistribution | None:
        """Get price distribution for products.

        Args:
            category: Optional category filter
            brand: Optional brand filter

        Returns:
            PriceDistribution or None
        """
        if not self._postgres:
            await self.initialize()

        # Build where clause
        conditions = ["price IS NOT NULL", "price > 0"]
        params = []
        param_idx = 1

        if category:
            conditions.append(f"LOWER(category) LIKE LOWER(${param_idx})")
            params.append(f"%{category}%")
            param_idx += 1

        if brand:
            conditions.append(f"LOWER(brand) = LOWER(${param_idx})")
            params.append(brand)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as p75,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY price) as p90
            FROM products
            WHERE {where_clause}
        """

        try:
            row = await self._postgres.fetchrow(query, *params)
            if not row or row["min_price"] is None:
                return None

            return PriceDistribution(
                min_price=float(row["min_price"]),
                max_price=float(row["max_price"]),
                avg_price=float(row["avg_price"]),
                median_price=float(row["median_price"]),
                percentiles={
                    25: float(row["p25"]),
                    50: float(row["median_price"]),
                    75: float(row["p75"]),
                    90: float(row["p90"]),
                },
            )

        except Exception as e:
            logger.error("price_distribution_failed", error=str(e))
            return None

    async def get_rating_distribution(
        self,
        category: str | None = None,
        brand: str | None = None,
    ) -> dict[str, int]:
        """Get rating distribution (1-5 stars).

        Args:
            category: Optional category filter
            brand: Optional brand filter

        Returns:
            Dictionary of star rating to count
        """
        if not self._postgres:
            await self.initialize()

        # Build where clause
        conditions = ["stars IS NOT NULL"]
        params = []
        param_idx = 1

        if category:
            conditions.append(f"LOWER(category) LIKE LOWER(${param_idx})")
            params.append(f"%{category}%")
            param_idx += 1

        if brand:
            conditions.append(f"LOWER(brand) = LOWER(${param_idx})")
            params.append(brand)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                FLOOR(stars) as star_rating,
                COUNT(*) as count
            FROM products
            WHERE {where_clause}
            GROUP BY FLOOR(stars)
            ORDER BY star_rating
        """

        try:
            rows = await self._postgres.fetch(query, *params)
            return {
                f"{int(row['star_rating'])} stars": row["count"]
                for row in rows
            }

        except Exception as e:
            logger.error("rating_distribution_failed", error=str(e))
            return {}

    async def aggregate_search_results(
        self,
        results: list[dict],
    ) -> dict[str, Any]:
        """Aggregate statistics from search results.

        Args:
            results: List of search result products

        Returns:
            Aggregated statistics
        """
        if not results:
            return {}

        # Collect values
        prices = [r.get("price") for r in results if r.get("price")]
        ratings = [r.get("stars") for r in results if r.get("stars")]
        brands = [r.get("brand") for r in results if r.get("brand")]
        categories = [r.get("category") for r in results if r.get("category")]

        # Calculate stats
        stats = {
            "total_results": len(results),
        }

        if prices:
            stats["price_stats"] = {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
            }

        if ratings:
            stats["rating_stats"] = {
                "min": min(ratings),
                "max": max(ratings),
                "avg": sum(ratings) / len(ratings),
            }

        if brands:
            from collections import Counter
            brand_counts = Counter(brands)
            stats["top_brands"] = [
                {"brand": b, "count": c}
                for b, c in brand_counts.most_common(5)
            ]

        if categories:
            from collections import Counter
            cat_counts = Counter(categories)
            stats["top_categories"] = [
                {"category": c, "count": cnt}
                for c, cnt in cat_counts.most_common(5)
            ]

        return stats


# Singleton instance
_aggregation_toolkit: AggregationToolkit | None = None


async def get_aggregation_toolkit() -> AggregationToolkit:
    """Get or create aggregation toolkit singleton."""
    global _aggregation_toolkit
    if _aggregation_toolkit is None:
        _aggregation_toolkit = AggregationToolkit()
        await _aggregation_toolkit.initialize()
    return _aggregation_toolkit
