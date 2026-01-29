"""
Keyword-priority hybrid search strategy.

Combines keyword and semantic search with RRF fusion,
giving priority to keyword results.

Performance: MRR 0.9126 (best overall from experiments)
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
from src.search.query_analyzer import QueryAnalyzer, QueryAnalysis

logger = structlog.get_logger()


class KeywordPriorityHybridStrategy(BaseSearchStrategy):
    """Keyword-priority hybrid search strategy.

    This strategy combines keyword (Elasticsearch) and semantic (Qdrant)
    search results using Reciprocal Rank Fusion (RRF). It prioritizes
    keyword results while incorporating semantic relevance.

    Performance: MRR 0.9126 (best overall from search-flow experiments)
    Best for: All query types, especially brand+model queries

    Key features:
    - Parallel keyword + semantic search execution
    - Weighted RRF fusion (keyword weight: 0.65, semantic: 0.35)
    - Dynamic boosts for model numbers and brands
    - Query-type aware weight adjustment

    RRF Formula:
        score = sum(weight_i * (1 / (k + rank_i + 1)))
        where k=40 provides balanced emphasis on top results

    Settings:
        hybrid_keyword_weight: Weight for keyword results (default: 0.65)
        hybrid_semantic_weight: Weight for semantic results (default: 0.35)
        hybrid_rrf_k: RRF k parameter (default: 40)
    """

    def __init__(
        self,
        name: str = "hybrid_keyword_priority",
        settings: StrategySettings | None = None,
        clients: Any = None,
    ):
        """Initialize keyword-priority hybrid strategy.

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
        """Execute hybrid search with keyword priority.

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional filters to apply
            **kwargs: Additional parameters (e.g., override_weights)

        Returns:
            SearchResponse with fused results
        """
        if not self.is_initialized:
            raise RuntimeError(f"Strategy {self.name} not initialized")

        start_time = time.time()

        # Analyze query
        analysis = QueryAnalyzer.analyze(query)

        # Get weights (can be overridden or use query-type defaults)
        keyword_weight, semantic_weight = self._get_weights(analysis, kwargs)

        try:
            # Execute keyword and semantic search in parallel
            keyword_task = self._keyword_search(
                query=analysis.clean_query,
                analysis=analysis,
                limit=limit * self.settings.fetch_multiplier,
                filters=filters,
            )
            semantic_task = self._semantic_search(
                query=analysis.clean_query,
                limit=limit * self.settings.fetch_multiplier,
                filters=filters,
            )

            keyword_results, semantic_results = await asyncio.gather(
                keyword_task,
                semantic_task,
                return_exceptions=True,
            )

            # Handle potential errors
            if isinstance(keyword_results, Exception):
                logger.warning("keyword_search_failed_in_hybrid", error=str(keyword_results))
                keyword_results = []
            if isinstance(semantic_results, Exception):
                logger.warning("semantic_search_failed_in_hybrid", error=str(semantic_results))
                semantic_results = []

            # Fuse results using weighted RRF
            fused_results = self._rrf_fusion(
                keyword_results=keyword_results,
                semantic_results=semantic_results,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                analysis=analysis,
                limit=limit * 2,  # Fetch more for reranking
            )

            # Apply reranking if enabled
            use_reranking = kwargs.get("rerank", self.settings.enable_reranking)
            logger.info(
                "reranking_check",
                use_reranking=use_reranking,
                rerank_kwarg=kwargs.get("rerank"),
                settings_enable=self.settings.enable_reranking,
                has_results=bool(fused_results),
                result_count=len(fused_results) if fused_results else 0,
            )
            if use_reranking and fused_results:
                fused_results = await self._rerank_results(
                    query=query,
                    results=fused_results,
                    limit=limit,
                )

            # Ensure we don't exceed limit
            fused_results = fused_results[:limit]

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "hybrid_search_completed",
                strategy=self.name,
                query=query,
                keyword_count=len(keyword_results) if isinstance(keyword_results, list) else 0,
                semantic_count=len(semantic_results) if isinstance(semantic_results, list) else 0,
                fused_count=len(fused_results),
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
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
                model_numbers=analysis.model_numbers,
            )

        except Exception as e:
            logger.error(
                "hybrid_search_failed",
                strategy=self.name,
                query=query,
                error=str(e),
            )
            raise

    def _get_weights(
        self,
        analysis: QueryAnalysis,
        kwargs: dict,
    ) -> tuple[float, float]:
        """Get keyword and semantic weights.

        Can be overridden via kwargs or uses query-type specific defaults.

        Args:
            analysis: Query analysis result
            kwargs: Additional parameters

        Returns:
            Tuple of (keyword_weight, semantic_weight)
        """
        if "keyword_weight" in kwargs and "semantic_weight" in kwargs:
            return kwargs["keyword_weight"], kwargs["semantic_weight"]

        # Use query-type specific weights from analysis
        return analysis.keyword_weight, analysis.semantic_weight

    async def _keyword_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        limit: int,
        filters: SearchFilters | None,
    ) -> list[dict]:
        """Execute keyword search component.

        Args:
            query: Cleaned query
            analysis: Query analysis
            limit: Result limit
            filters: Search filters

        Returns:
            List of results with ASIN and score
        """
        # Build boost configuration
        boost_config = {
            "title": self.settings.keyword_boost_title,
            "title.autocomplete": self.settings.keyword_boost_title / 2,
            "short_title": self.settings.keyword_boost_short_title,
            "brand": self.settings.keyword_boost_brand,
            "product_type": 4.0,
            "genAI_summary": 2.0,
        }

        # Add model number boost if detected
        if analysis.model_numbers:
            boost_config["title"] = self.settings.keyword_boost_model

        filter_dict = filters.to_dict() if filters else None

        raw_results = await self.clients.keyword_search(
            query=query,
            limit=limit,
            filters=filter_dict,
            boost_config=boost_config,
        )

        # Extract ASIN and score
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
        limit: int,
        filters: SearchFilters | None,
    ) -> list[dict]:
        """Execute semantic search component.

        Args:
            query: Cleaned query
            limit: Result limit
            filters: Search filters

        Returns:
            List of results with ASIN and score
        """
        # Get embedding
        embedding = await self.clients.get_embedding(query)

        filter_dict = filters.to_dict() if filters else None

        raw_results = await self.clients.semantic_search(
            vector=embedding,
            limit=limit,
            filters=filter_dict,
            score_threshold=self.settings.semantic_score_threshold,
        )

        # Extract ASIN and score
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

    def _rrf_fusion(
        self,
        keyword_results: list[dict],
        semantic_results: list[dict],
        keyword_weight: float,
        semantic_weight: float,
        analysis: QueryAnalysis,
        limit: int,
    ) -> list[SearchResult]:
        """Fuse results using weighted Reciprocal Rank Fusion.

        RRF scores each result based on its rank in each result set:
            score = sum(weight * (1 / (k + rank + 1)))

        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            keyword_weight: Weight for keyword results
            semantic_weight: Weight for semantic results
            analysis: Query analysis for post-processing
            limit: Maximum results to return

        Returns:
            Fused list of SearchResult objects
        """
        k = self.settings.hybrid_rrf_k
        asin_scores: dict[str, float] = {}
        asin_data: dict[str, dict] = {}

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            asin = result["asin"]
            rrf_score = keyword_weight * (1.0 / (k + rank + 1))
            asin_scores[asin] = asin_scores.get(asin, 0) + rrf_score

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

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            asin = result["asin"]
            rrf_score = semantic_weight * (1.0 / (k + rank + 1))
            asin_scores[asin] = asin_scores.get(asin, 0) + rrf_score

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

        # Apply post-processing boosts
        for asin, score in asin_scores.items():
            data = asin_data.get(asin, {})
            title = data.get("title", "").lower()

            # Model number boost
            for model in analysis.model_numbers:
                if model.lower() in title:
                    asin_scores[asin] *= self.settings.model_boost_factor
                    break

            # Brand boost
            if analysis.has_brand:
                if analysis.detected_brand.lower() in title:
                    asin_scores[asin] *= self.settings.brand_boost_factor

        # Sort by fused score and create SearchResult objects
        sorted_asins = sorted(
            asin_scores.keys(),
            key=lambda x: asin_scores[x],
            reverse=True,
        )

        results = []
        for asin in sorted_asins[:limit]:
            data = asin_data.get(asin, {})
            score = min(asin_scores[asin], 1.0)  # Normalize to 0-1

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

    async def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """Rerank results using the configured reranker model.

        Args:
            query: Original search query
            results: List of SearchResult objects to rerank
            limit: Maximum results to return

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return results

        try:
            # Convert SearchResult objects to dicts for reranking
            result_dicts = []
            for r in results:
                result_dicts.append({
                    "asin": r.asin,
                    "title": r.title,
                    "brand": r.brand,
                    "genAI_summary": r.genAI_summary,
                    "price": r.price,
                    "stars": r.stars,
                    "category": r.category,
                    "img_url": r.img_url,
                    "genAI_best_for": r.genAI_best_for,
                    "original_score": r.score,
                })

            # Call reranker (force_rerank=True bypasses database config check)
            reranked = await self.clients.rerank_results(
                query=query,
                results=result_dicts,
                force_rerank=True,
            )

            if not reranked:
                logger.warning("reranking_returned_empty", query=query[:50])
                return results[:limit]

            # Convert back to SearchResult objects
            reranked_results = []
            for r in reranked[:limit]:
                reranked_results.append(SearchResult(
                    asin=r.get("asin", ""),
                    title=r.get("title", ""),
                    score=r.get("rerank_score", r.get("original_score", 0)),
                    source=self.name,
                    price=r.get("price"),
                    stars=r.get("stars"),
                    brand=r.get("brand"),
                    category=r.get("category"),
                    img_url=r.get("img_url"),
                    genAI_summary=r.get("genAI_summary"),
                    genAI_best_for=r.get("genAI_best_for"),
                ))

            logger.info(
                "reranking_applied",
                query=query[:50],
                original_count=len(results),
                reranked_count=len(reranked_results),
                top_score=reranked_results[0].score if reranked_results else 0,
            )

            return reranked_results

        except Exception as e:
            logger.error("reranking_failed_in_strategy", error=str(e), query=query[:50])
            # Fall back to original results on error
            return results[:limit]
