"""
Brand and model number priority keyword search strategy.

Optimized for queries containing brand names and model numbers.
Performance: R@1 87.7% (best keyword strategy)
"""

import re
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


class BrandModelPriorityStrategy(BaseSearchStrategy):
    """Brand and model number priority keyword search strategy.

    This strategy is optimized for queries containing brand names
    and model numbers. It applies significant boosts to:
    - Model number matches in title (20x boost)
    - Brand name matches (5x boost)
    - Title exact matches (10x boost)

    Performance: R@1 87.7% (best keyword strategy from experiments)
    Best for: "Sony WH-1000XM5", "MacBook Pro M2", "Dell XPS 15"

    Key features:
    - Model number extraction and boosting
    - Brand detection with priority boost
    - Phrase matching for exact title matches
    - Fallback to fuzzy matching for typo tolerance
    """

    def __init__(
        self,
        name: str = "keyword_brand_model_priority",
        settings: StrategySettings | None = None,
        clients: Any = None,
    ):
        """Initialize brand model priority strategy.

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
        """Execute brand/model optimized keyword search.

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional filters to apply
            **kwargs: Additional parameters

        Returns:
            SearchResponse with results
        """
        if not self.is_initialized:
            raise RuntimeError(f"Strategy {self.name} not initialized")

        start_time = time.time()

        # Analyze query
        analysis = QueryAnalyzer.analyze(query)

        # Build specialized query with brand/model boosts
        es_query = self._build_brand_model_query(
            query=analysis.clean_query,
            model_numbers=analysis.model_numbers,
            brand=analysis.detected_brand if analysis.has_brand else None,
            filters=filters,
        )

        # Execute search
        try:
            if not self.clients._elasticsearch:
                raise RuntimeError("Elasticsearch client not initialized")

            response = await self.clients._elasticsearch.search(
                index=self.clients.config.es_index,
                query=es_query,
                size=limit * self.settings.fetch_multiplier,
                _source=True,
            )

            raw_results = [
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"],
                }
                for hit in response["hits"]["hits"]
            ]

            # Apply post-processing boosts
            results = self._process_results(
                raw_results=raw_results,
                analysis=analysis,
                limit=limit,
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "brand_model_search_completed",
                strategy=self.name,
                query=query,
                model_numbers=analysis.model_numbers,
                brand=analysis.detected_brand,
                results=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return self._create_response(
                results=results,
                query=query,
                query_type=analysis.query_type.value,
                latency_ms=latency_ms,
                filters=filters,
                model_numbers=analysis.model_numbers,
                detected_brand=analysis.detected_brand,
            )

        except Exception as e:
            logger.error(
                "brand_model_search_failed",
                strategy=self.name,
                query=query,
                error=str(e),
            )
            raise

    def _build_brand_model_query(
        self,
        query: str,
        model_numbers: list[str],
        brand: str | None,
        filters: SearchFilters | None,
    ) -> dict[str, Any]:
        """Build Elasticsearch query with brand/model optimizations.

        Args:
            query: Cleaned search query
            model_numbers: Extracted model numbers
            brand: Detected brand name
            filters: Search filters

        Returns:
            Elasticsearch query dict
        """
        should_clauses = []
        boost_title = self.settings.keyword_boost_title
        boost_model = self.settings.keyword_boost_model
        boost_brand = self.settings.keyword_boost_brand

        # 1. Exact phrase match on title (highest priority)
        should_clauses.append({
            "match_phrase": {
                "title": {
                    "query": query,
                    "boost": boost_title * 2,
                }
            }
        })

        # 2. Model number exact matches (very high priority)
        for model in model_numbers:
            should_clauses.append({
                "match_phrase": {
                    "title": {
                        "query": model,
                        "boost": boost_model,
                    }
                }
            })
            should_clauses.append({
                "match_phrase": {
                    "short_title": {
                        "query": model,
                        "boost": boost_model * 0.8,
                    }
                }
            })

        # 3. Brand match (high priority)
        if brand:
            should_clauses.append({
                "match": {
                    "brand": {
                        "query": brand,
                        "boost": boost_brand,
                    }
                }
            })

        # 4. Standard multi-match with boosts
        should_clauses.append({
            "multi_match": {
                "query": query,
                "fields": [
                    f"title^{boost_title}",
                    f"title.autocomplete^{boost_title / 2}",
                    f"short_title^{self.settings.keyword_boost_short_title}",
                    f"brand^{boost_brand}",
                    "product_type^4",
                    "genAI_summary^2",
                ],
                "type": "best_fields",
                "fuzziness": self.settings.keyword_fuzziness,
            }
        })

        # Build final query
        es_query: dict[str, Any] = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        }

        # Add filters
        if filters:
            filter_clauses = self._build_filters(filters)
            if filter_clauses:
                es_query["bool"]["filter"] = filter_clauses

        return es_query

    def _build_filters(self, filters: SearchFilters) -> list[dict]:
        """Build Elasticsearch filter clauses.

        Args:
            filters: Search filters

        Returns:
            List of filter clauses
        """
        clauses = []

        if filters.brand:
            clauses.append({"term": {"brand.keyword": filters.brand}})

        if filters.category:
            clauses.append({"term": {"category_name.keyword": filters.category}})

        if filters.min_price is not None or filters.max_price is not None:
            range_filter: dict[str, Any] = {}
            if filters.min_price is not None:
                range_filter["gte"] = filters.min_price
            if filters.max_price is not None:
                range_filter["lte"] = filters.max_price
            clauses.append({"range": {"price": range_filter}})

        if filters.min_rating is not None:
            clauses.append({"range": {"stars": {"gte": filters.min_rating}}})

        if filters.exclude_asins:
            clauses.append({
                "bool": {
                    "must_not": [
                        {"terms": {"asin": filters.exclude_asins}}
                    ]
                }
            })

        return clauses

    def _process_results(
        self,
        raw_results: list[dict],
        analysis: Any,
        limit: int,
    ) -> list[SearchResult]:
        """Process results with brand/model post-processing boosts.

        Args:
            raw_results: Raw ES search results
            analysis: QueryAnalysis object
            limit: Maximum results to return

        Returns:
            List of SearchResult objects, re-ranked
        """
        results = []
        seen_asins = set()

        for hit in raw_results:
            source = hit.get("source", {})
            asin = source.get("asin") or source.get("parent_asin")

            if not asin or asin in seen_asins:
                continue

            seen_asins.add(asin)

            # Base score (normalized)
            base_score = min(hit.get("score", 0) / 100.0, 1.0)

            # Apply post-processing boosts
            title = source.get("title", "").lower()
            final_score = base_score

            # Boost for model number in title
            for model in analysis.model_numbers:
                if model.lower() in title:
                    final_score *= self.settings.model_boost_factor
                    break

            # Boost for brand in title
            if analysis.has_brand and analysis.detected_brand.lower() in title:
                final_score *= self.settings.brand_boost_factor

            results.append(SearchResult(
                asin=asin,
                title=source.get("title", ""),
                score=min(final_score, 1.0),
                source=self.name,
                price=source.get("price"),
                stars=source.get("stars"),
                brand=source.get("brand"),
                category=source.get("category_name") or source.get("category_level1"),
                img_url=source.get("img_url") or source.get("imgUrl"),  # Handle both naming conventions
                genAI_summary=source.get("genAI_summary"),
                genAI_best_for=source.get("genAI_best_for"),
            ))

        # Re-sort by final score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]
