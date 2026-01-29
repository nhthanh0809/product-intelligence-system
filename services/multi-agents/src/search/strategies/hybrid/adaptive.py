"""
Adaptive fusion hybrid search strategy.

Automatically adjusts keyword/semantic weights based on query analysis.
"""

import asyncio
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
from src.search.query_analyzer import QueryAnalyzer, QueryAnalysis, QueryType

logger = structlog.get_logger()


class AdaptiveFusionStrategy(BaseSearchStrategy):
    """Adaptive fusion hybrid search strategy.

    This strategy automatically adjusts the keyword/semantic weights
    based on query analysis. It provides more aggressive weight
    adjustments than the keyword-priority strategy.

    Weight adjustments by query type:
    - BRAND_MODEL: keyword 0.80, semantic 0.20
    - MODEL_NUMBER: keyword 0.85, semantic 0.15
    - SHORT_TITLE: keyword 0.65, semantic 0.35
    - GENERIC: keyword 0.35, semantic 0.65
    - SECTION: keyword 0.30, semantic 0.70

    Best for: Mixed query workloads where query types vary significantly
    """

    # Aggressive weight mappings for different query types
    WEIGHT_MAP = {
        QueryType.BRAND_MODEL: (0.80, 0.20),
        QueryType.MODEL_NUMBER: (0.85, 0.15),
        QueryType.SHORT_TITLE: (0.65, 0.35),
        QueryType.GENERIC: (0.35, 0.65),
        QueryType.SECTION: (0.30, 0.70),
    }

    def __init__(
        self,
        name: str = "hybrid_adaptive",
        settings: StrategySettings | None = None,
        clients: Any = None,
    ):
        """Initialize adaptive fusion strategy.

        Args:
            name: Strategy identifier
            settings: Configuration settings
            clients: SearchClients instance
        """
        super().__init__(
            name=name,
            strategy_type=StrategyType.HYBRID,
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
        """Execute adaptive hybrid search.

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional filters to apply
            **kwargs: Additional parameters

        Returns:
            SearchResponse with adaptively fused results
        """
        if not self.is_initialized:
            raise RuntimeError(f"Strategy {self.name} not initialized")

        start_time = time.time()

        # Analyze query to determine optimal weights
        analysis = QueryAnalyzer.analyze(query)
        keyword_weight, semantic_weight = self._get_adaptive_weights(analysis)

        try:
            # Execute searches in parallel
            keyword_task = self._keyword_search(
                query=analysis.clean_query,
                analysis=analysis,
                limit=limit * self.settings.fetch_multiplier,
                filters=filters,
            )
            semantic_task = self._semantic_search(
                query=analysis.clean_query,
                analysis=analysis,
                limit=limit * self.settings.fetch_multiplier,
                filters=filters,
            )

            keyword_results, semantic_results = await asyncio.gather(
                keyword_task,
                semantic_task,
                return_exceptions=True,
            )

            # Handle errors gracefully
            if isinstance(keyword_results, Exception):
                logger.warning("keyword_search_failed", error=str(keyword_results))
                keyword_results = []
                # Shift all weight to semantic if keyword fails
                semantic_weight = 1.0
                keyword_weight = 0.0

            if isinstance(semantic_results, Exception):
                logger.warning("semantic_search_failed", error=str(semantic_results))
                semantic_results = []
                # Shift all weight to keyword if semantic fails
                keyword_weight = 1.0
                semantic_weight = 0.0

            # Fuse with adaptive weights
            fused_results = self._adaptive_fusion(
                keyword_results=keyword_results,
                semantic_results=semantic_results,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                analysis=analysis,
                limit=limit,
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "adaptive_search_completed",
                strategy=self.name,
                query=query,
                query_type=analysis.query_type.value,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                results=len(fused_results),
                latency_ms=round(latency_ms, 2),
            )

            return self._create_response(
                results=fused_results,
                query=query,
                query_type=analysis.query_type.value,
                latency_ms=latency_ms,
                filters=filters,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                adaptive_reason=f"query_type:{analysis.query_type.value}",
            )

        except Exception as e:
            logger.error(
                "adaptive_search_failed",
                strategy=self.name,
                query=query,
                error=str(e),
            )
            raise

    def _get_adaptive_weights(
        self,
        analysis: QueryAnalysis,
    ) -> tuple[float, float]:
        """Get adaptive weights based on query analysis.

        Args:
            analysis: Query analysis result

        Returns:
            Tuple of (keyword_weight, semantic_weight)
        """
        # Get base weights from query type
        base_kw, base_sem = self.WEIGHT_MAP.get(
            analysis.query_type,
            (self.settings.hybrid_keyword_weight, self.settings.hybrid_semantic_weight),
        )

        # Further adjust based on specific features
        kw_weight = base_kw
        sem_weight = base_sem

        # Model numbers strongly favor keyword
        if analysis.model_numbers:
            kw_weight = min(kw_weight + 0.1, 0.95)
            sem_weight = 1.0 - kw_weight

        # Long queries slightly favor semantic
        if len(analysis.clean_query.split()) > 8:
            sem_weight = min(sem_weight + 0.1, 0.80)
            kw_weight = 1.0 - sem_weight

        return kw_weight, sem_weight

    async def _keyword_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int,
        filters: SearchFilters | None,
    ) -> list[dict]:
        """Execute keyword search component."""
        boost_config = {
            "title": self.settings.keyword_boost_title,
            "title.autocomplete": self.settings.keyword_boost_title / 2,
            "short_title": self.settings.keyword_boost_short_title,
            "brand": self.settings.keyword_boost_brand,
            "product_type": 4.0,
            "genAI_summary": 2.0,
        }

        filter_dict = filters.to_dict() if filters else None

        raw_results = await self.clients.keyword_search(
            query=query,
            limit=limit,
            filters=filter_dict,
            boost_config=boost_config,
        )

        results = []
        for hit in raw_results:
            source = hit.get("source", {})
            asin = source.get("asin") or source.get("parent_asin")
            if asin:
                results.append({
                    "asin": asin,
                    "score": hit.get("score", 0),
                    "source": source,
                })

        return results

    async def _semantic_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int,
        filters: SearchFilters | None,
    ) -> list[dict]:
        """Execute semantic search component."""
        # Format query for better embedding
        formatted_query = query
        if analysis.has_brand:
            formatted_query = f"product by {analysis.detected_brand}: {query}"

        embedding = await self.clients.get_embedding(formatted_query)
        filter_dict = filters.to_dict() if filters else None

        raw_results = await self.clients.semantic_search(
            vector=embedding,
            limit=limit,
            filters=filter_dict,
            score_threshold=self.settings.semantic_score_threshold,
        )

        results = []
        for hit in raw_results:
            payload = hit.get("payload", {})
            asin = payload.get("asin") or payload.get("parent_asin")
            if asin:
                results.append({
                    "asin": asin,
                    "score": hit.get("score", 0),
                    "payload": payload,
                })

        return results

    def _adaptive_fusion(
        self,
        keyword_results: list[dict],
        semantic_results: list[dict],
        keyword_weight: float,
        semantic_weight: float,
        analysis: QueryAnalysis,
        limit: int,
    ) -> list[SearchResult]:
        """Fuse results with adaptive weighting and re-ranking.

        Args:
            keyword_results: Keyword search results
            semantic_results: Semantic search results
            keyword_weight: Weight for keyword results
            semantic_weight: Weight for semantic results
            analysis: Query analysis
            limit: Maximum results

        Returns:
            Fused and re-ranked SearchResult list
        """
        k = self.settings.hybrid_rrf_k
        asin_scores: dict[str, float] = {}
        asin_data: dict[str, dict] = {}
        asin_sources: dict[str, set] = {}  # Track which sources found each result

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            asin = result["asin"]
            rrf_score = keyword_weight * (1.0 / (k + rank + 1))
            asin_scores[asin] = asin_scores.get(asin, 0) + rrf_score

            if asin not in asin_sources:
                asin_sources[asin] = set()
            asin_sources[asin].add("keyword")

            if asin not in asin_data:
                source = result.get("source", {})
                asin_data[asin] = {
                    "title": source.get("title", ""),
                    "price": source.get("price"),
                    "stars": source.get("stars"),
                    "brand": source.get("brand"),
                    "category": source.get("category_name") or source.get("category_level1"),
                    "img_url": source.get("img_url") or source.get("imgUrl"),  # Handle both naming conventions
                    "genAI_summary": source.get("genAI_summary"),
                    "genAI_best_for": source.get("genAI_best_for"),
                }

        # Process semantic results
        for rank, result in enumerate(semantic_results):
            asin = result["asin"]
            rrf_score = semantic_weight * (1.0 / (k + rank + 1))
            asin_scores[asin] = asin_scores.get(asin, 0) + rrf_score

            if asin not in asin_sources:
                asin_sources[asin] = set()
            asin_sources[asin].add("semantic")

            if asin not in asin_data:
                payload = result.get("payload", {})
                asin_data[asin] = {
                    "title": payload.get("title", ""),
                    "price": payload.get("price"),
                    "stars": payload.get("stars"),
                    "brand": payload.get("brand"),
                    "category": payload.get("category_name") or payload.get("category_level1"),
                    "img_url": payload.get("img_url") or payload.get("imgUrl"),  # Handle both naming conventions
                    "genAI_summary": payload.get("genAI_summary"),
                    "genAI_best_for": payload.get("genAI_best_for"),
                }

        # Apply adaptive boosts
        for asin in asin_scores:
            data = asin_data.get(asin, {})
            title = data.get("title", "").lower()
            sources = asin_sources.get(asin, set())

            # Boost results found by both sources (higher confidence)
            if len(sources) == 2:
                asin_scores[asin] *= 1.15

            # Model number boost
            for model in analysis.model_numbers:
                if model.lower() in title:
                    asin_scores[asin] *= self.settings.model_boost_factor
                    break

            # Brand boost
            if analysis.has_brand and analysis.detected_brand.lower() in title:
                asin_scores[asin] *= self.settings.brand_boost_factor

        # Sort and create results
        sorted_asins = sorted(
            asin_scores.keys(),
            key=lambda x: asin_scores[x],
            reverse=True,
        )

        results = []
        for asin in sorted_asins[:limit]:
            data = asin_data.get(asin, {})
            score = min(asin_scores[asin], 1.0)

            results.append(SearchResult(
                asin=asin,
                title=data.get("title", ""),
                score=score,
                source=self.name,
                price=data.get("price"),
                stars=data.get("stars"),
                brand=data.get("brand"),
                category=data.get("category"),
                img_url=data.get("img_url"),
                genAI_summary=data.get("genAI_summary"),
                genAI_best_for=data.get("genAI_best_for"),
            ))

        return results
