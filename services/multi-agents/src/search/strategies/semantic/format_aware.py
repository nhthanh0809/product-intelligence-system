"""
Format-aware semantic search strategy.

Handles query formatting and the 0.99 similarity score issue
discovered in search-flow experiments.
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


class FormatAwareSemanticStrategy(BaseSearchStrategy):
    """Format-aware semantic search strategy.

    This strategy improves upon basic semantic search by:
    1. Cleaning and formatting queries before embedding
    2. Handling the 0.99 similarity score issue
    3. Applying query-type specific post-processing

    Performance: MRR 0.65 (best semantic strategy)
    Best for: Generic queries, conceptual searches

    The 0.99 similarity issue occurs when certain query formats
    produce artificially high similarity scores. This strategy
    detects and adjusts for this.
    """

    def __init__(
        self,
        name: str = "semantic_format_aware",
        settings: StrategySettings | None = None,
        clients: Any = None,
    ):
        """Initialize format-aware semantic strategy.

        Args:
            name: Strategy identifier
            settings: Configuration settings
            clients: SearchClients instance
        """
        super().__init__(
            name=name,
            strategy_type=StrategyType.SEMANTIC,
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
        """Execute format-aware semantic search.

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

        try:
            # Format query for better embedding
            formatted_query = self._format_query_for_embedding(
                analysis.clean_query,
                analysis,
            )

            # Get query embedding
            embedding = await self.clients.get_embedding(formatted_query)

            # Convert filters to dict
            filter_dict = filters.to_dict() if filters else None

            # Execute vector search with more results for post-processing
            raw_results = await self.clients.semantic_search(
                vector=embedding,
                limit=limit * self.settings.fetch_multiplier,
                filters=filter_dict,
                score_threshold=self.settings.semantic_score_threshold,
            )

            # Process results with format-aware scoring
            results = self._process_results(
                raw_results=raw_results,
                analysis=analysis,
                query=analysis.clean_query,
                limit=limit,
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "format_aware_semantic_search_completed",
                strategy=self.name,
                query=query,
                formatted_query=formatted_query,
                results=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return self._create_response(
                results=results,
                query=query,
                query_type=analysis.query_type.value,
                latency_ms=latency_ms,
                filters=filters,
                formatted_query=formatted_query,
            )

        except Exception as e:
            logger.error(
                "format_aware_semantic_search_failed",
                strategy=self.name,
                query=query,
                error=str(e),
            )
            raise

    def _format_query_for_embedding(
        self,
        query: str,
        analysis: Any,
    ) -> str:
        """Format query to produce better embeddings.

        Experiments showed that embedding "product search: [query]"
        format produces more accurate similarity scores than raw queries.

        Args:
            query: Cleaned query
            analysis: Query analysis result

        Returns:
            Formatted query string
        """
        # Add context prefix for better embedding
        if analysis.has_brand:
            return f"product by {analysis.detected_brand}: {query}"
        elif analysis.detected_section:
            return f"{analysis.detected_section} of product: {query}"
        else:
            return f"product search: {query}"

    def _process_results(
        self,
        raw_results: list[dict],
        analysis: Any,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Process results with format-aware scoring.

        Handles the 0.99 similarity issue by:
        1. Detecting suspiciously high scores
        2. Applying query-result relevance checks
        3. Re-scoring based on content matching

        Args:
            raw_results: Raw Qdrant search results
            analysis: Query analysis
            query: Original query
            limit: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        results = []
        seen_asins = set()
        query_words = set(query.lower().split())

        for hit in raw_results:
            payload = hit.get("payload", {})
            asin = payload.get("asin") or payload.get("parent_asin")

            if not asin or asin in seen_asins:
                continue

            seen_asins.add(asin)

            # Get base score
            score = hit.get("score", 0)
            title = payload.get("title", "")
            title_lower = title.lower()

            # Handle 0.99 similarity issue
            if score > 0.95:
                # Check if result actually matches query content
                title_words = set(title_lower.split())
                overlap = len(query_words & title_words)
                overlap_ratio = overlap / max(len(query_words), 1)

                if overlap_ratio < 0.3:
                    # Low content overlap - likely false positive
                    score *= 0.7
                    logger.debug(
                        "semantic_score_adjusted",
                        asin=asin,
                        original_score=hit.get("score"),
                        adjusted_score=score,
                        reason="low_content_overlap",
                    )

            # Apply brand boost if query contains brand and result matches
            if analysis.has_brand:
                result_brand = payload.get("brand", "").lower()
                if analysis.detected_brand.lower() in result_brand:
                    score = min(score * 1.1, 1.0)

            # Apply model number boost
            for model in analysis.model_numbers:
                if model.lower() in title_lower:
                    score = min(score * 1.15, 1.0)
                    break

            results.append(SearchResult(
                asin=asin,
                title=title,
                score=score,
                source=self.name,
                price=payload.get("price"),
                stars=payload.get("stars"),
                brand=payload.get("brand"),
                category=payload.get("category_name") or payload.get("category_level1"),
                img_url=payload.get("img_url") or payload.get("imgUrl"),  # Handle both naming conventions
                genAI_summary=payload.get("genAI_summary"),
                genAI_best_for=payload.get("genAI_best_for"),
            ))

        # Re-sort by adjusted score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]
