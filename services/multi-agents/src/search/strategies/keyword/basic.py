"""
Basic keyword search strategy using Elasticsearch.

Provides standard full-text search with configurable field boosts.
"""

import time
from typing import Any

import structlog

from src.search.base import (
    BaseSearchStrategy,
    SearchFilters,
    SearchResponse,
    SearchResult,
    StrategySettings,
    StrategyType,
)
from src.search.query_analyzer import QueryAnalyzer

logger = structlog.get_logger()


class BasicKeywordStrategy(BaseSearchStrategy):
    """Basic keyword search strategy using Elasticsearch multi-match.

    This strategy performs full-text search across multiple fields
    with configurable boosting.

    Performance: R@1 ~75% (baseline)
    Best for: Simple keyword queries, known product names

    Settings:
        keyword_boost_title: Title field boost (default: 10.0)
        keyword_boost_short_title: Short title boost (default: 8.0)
        keyword_boost_brand: Brand field boost (default: 5.0)
        keyword_fuzziness: Fuzzy matching setting (default: "AUTO")
    """

    def __init__(
        self,
        name: str = "keyword_basic",
        settings: StrategySettings | None = None,
        clients: Any = None,
    ):
        """Initialize basic keyword strategy.

        Args:
            name: Strategy identifier
            settings: Configuration settings
            clients: SearchClients instance
        """
        super().__init__(
            name=name,
            strategy_type=StrategyType.KEYWORD,
            settings=settings,
            clients=clients,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
        **kwargs,
    ) -> SearchResponse:
        """Execute keyword search.

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional filters to apply
            **kwargs: Additional parameters (e.g., boost_overrides)

        Returns:
            SearchResponse with results
        """
        if not self.is_initialized:
            raise RuntimeError(f"Strategy {self.name} not initialized")

        start_time = time.time()

        # Analyze query
        analysis = QueryAnalyzer.analyze(query)

        # Build boost configuration
        boost_config = self._build_boost_config(kwargs.get("boost_overrides"))

        # Convert filters to dict
        filter_dict = filters.to_dict() if filters else None

        # Execute search
        try:
            raw_results = await self.clients.keyword_search(
                query=analysis.clean_query,
                limit=limit * self.settings.fetch_multiplier,
                filters=filter_dict,
                boost_config=boost_config,
            )

            # Convert to SearchResult objects
            results = self._process_results(raw_results, limit)

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "keyword_search_completed",
                strategy=self.name,
                query=query,
                results=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return self._create_response(
                results=results,
                query=query,
                query_type=analysis.query_type.value,
                latency_ms=latency_ms,
                filters=filters,
            )

        except Exception as e:
            logger.error(
                "keyword_search_failed",
                strategy=self.name,
                query=query,
                error=str(e),
            )
            raise

    def _build_boost_config(
        self,
        overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Build field boost configuration.

        Args:
            overrides: Optional boost overrides

        Returns:
            Field boost configuration dict
        """
        config = {
            "title": self.settings.keyword_boost_title,
            "title.autocomplete": self.settings.keyword_boost_title / 2,
            "short_title": self.settings.keyword_boost_short_title,
            "brand": self.settings.keyword_boost_brand,
            "product_type": 4.0,
            "genAI_summary": 2.0,
            "chunk_description": 1.0,
            "chunk_features": 1.0,
        }

        if overrides:
            config.update(overrides)

        return config

    def _process_results(
        self,
        raw_results: list[dict],
        limit: int,
    ) -> list[SearchResult]:
        """Process raw Elasticsearch results into SearchResult objects.

        Args:
            raw_results: Raw ES search results
            limit: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        results = []
        seen_asins = set()

        for hit in raw_results:
            source = hit.get("source", {})
            asin = source.get("asin") or source.get("parent_asin")

            if not asin or asin in seen_asins:
                continue

            seen_asins.add(asin)

            # Normalize score to 0-1 range (ES scores can be > 1)
            score = min(hit.get("score", 0) / 100.0, 1.0)

            results.append(SearchResult(
                asin=asin,
                title=source.get("title", ""),
                score=score,
                source=self.name,
                price=source.get("price"),
                stars=source.get("stars"),
                brand=source.get("brand"),
                category=source.get("category_name") or source.get("category_level1"),
                img_url=source.get("img_url") or source.get("imgUrl"),  # Handle both naming conventions
                genAI_summary=source.get("genAI_summary"),
                genAI_best_for=source.get("genAI_best_for"),
            ))

            if len(results) >= limit:
                break

        return results
