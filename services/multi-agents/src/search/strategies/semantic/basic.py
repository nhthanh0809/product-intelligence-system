"""
Basic semantic search strategy using Qdrant vector similarity.

Provides standard vector similarity search using embeddings.
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


class BasicSemanticStrategy(BaseSearchStrategy):
    """Basic semantic search strategy using Qdrant.

    This strategy performs vector similarity search using
    query embeddings. Good for conceptual and generic queries.

    Performance: MRR ~0.55 (baseline)
    Best for: "good laptop for students", "wireless earbuds for running"

    Settings:
        semantic_score_threshold: Minimum similarity score (default: 0.5)
    """

    def __init__(
        self,
        name: str = "semantic_basic",
        settings: StrategySettings | None = None,
        clients: Any = None,
    ):
        """Initialize basic semantic strategy.

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
        """Execute semantic vector search.

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
            # Get query embedding
            embedding = await self.clients.get_embedding(analysis.clean_query)

            # Convert filters to dict
            filter_dict = filters.to_dict() if filters else None

            # Execute vector search
            raw_results = await self.clients.semantic_search(
                vector=embedding,
                limit=limit * self.settings.fetch_multiplier,
                filters=filter_dict,
                score_threshold=self.settings.semantic_score_threshold,
            )

            logger.info(
                "semantic_search_raw_results",
                strategy=self.name,
                raw_count=len(raw_results),
                embedding_dims=len(embedding),
                fetch_limit=limit * self.settings.fetch_multiplier,
                score_threshold=self.settings.semantic_score_threshold,
            )

            # Convert to SearchResult objects
            results = self._process_results(raw_results, limit)

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "semantic_search_completed",
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
                "semantic_search_failed",
                strategy=self.name,
                query=query,
                error=str(e),
            )
            raise

    def _process_results(
        self,
        raw_results: list[dict],
        limit: int,
    ) -> list[SearchResult]:
        """Process raw Qdrant results into SearchResult objects.

        Args:
            raw_results: Raw Qdrant search results
            limit: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        results = []
        seen_asins = set()

        for hit in raw_results:
            payload = hit.get("payload", {})
            asin = payload.get("asin") or payload.get("parent_asin")

            if not asin or asin in seen_asins:
                continue

            seen_asins.add(asin)

            # Qdrant scores are already 0-1 similarity scores
            score = hit.get("score", 0)

            # Handle 0.99 similarity bug - treat very high scores with suspicion
            # (This was discovered in research-search-flows experiments)
            if score > 0.98:
                score = score * 0.95  # Slightly penalize suspiciously high scores

            results.append(SearchResult(
                asin=asin,
                title=payload.get("title", ""),
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

            if len(results) >= limit:
                break

        return results
