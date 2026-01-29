#!/usr/bin/env python3
"""
Hybrid Search Flow Experiments

Experiments to optimize hybrid search metrics:
1. Basic RRF fusion (baseline)
2. RRF k parameter tuning (k=20, 40, 60, 80)
3. Weighted score fusion (instead of RRF)
4. Semantic-first with keyword boost
5. Adaptive fusion (based on query characteristics)
6. Three-way fusion (semantic + keyword + section)

Usage:
    python scripts/research-search-flows/hybrid_experiments.py --eval-data data/eval/datasets/level3_retrieval_evaluation.json
"""

import asyncio
import math
import sys
from pathlib import Path
from typing import Any

import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from base import (
    SearchConfig,
    SearchStrategy,
    DatabaseClients,
    QueryPreprocessor,
    load_evaluation_data,
    run_experiment,
    print_experiment_results,
    get_search_flows_config,
)
import config as cfg
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = structlog.get_logger()


# =============================================================================
# Helper: Semantic Search
# =============================================================================

async def semantic_search(
    clients: DatabaseClients,
    config: SearchConfig,
    query: str,
    limit: int,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute semantic search on Qdrant."""
    embedding = await clients.get_embedding(query)
    qdrant = clients.get_qdrant()

    response = qdrant.query_points(
        collection_name=config.qdrant_collection,
        query=embedding,
        limit=limit,
        with_payload=True,
    )

    results = []
    for point in response.points:
        results.append({
            "id": str(point.id),
            "score": point.score,
            "asin": point.payload.get("asin") or point.payload.get("parent_asin"),
            "payload": point.payload,
            "source": "semantic",
        })
    return results


# =============================================================================
# Helper: Keyword Search
# =============================================================================

def _get_keyword_fields(config: SearchConfig) -> list[str]:
    """Get mode-appropriate keyword search fields.

    Args:
        config: SearchConfig with pipeline_mode

    Returns:
        List of field names with boost values
    """
    if config.is_original_mode:
        # Original mode: only title and category available
        return [
            "title^10",
            "title.autocomplete^5",
            "category_name^2",
        ]
    else:
        # Enrich mode: full field set
        return [
            "title^10",
            "short_title^8",
            "brand^5",
            "product_type^4",
            "chunk_description^1",
            "chunk_features^1",
        ]


async def keyword_search(
    clients: DatabaseClients,
    config: SearchConfig,
    query: str,
    limit: int,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute keyword search on Elasticsearch.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """
    es = await clients.get_elasticsearch()

    es_query = {
        "multi_match": {
            "query": query,
            "fields": _get_keyword_fields(config),
            "type": "best_fields",
            "fuzziness": "AUTO",
        }
    }

    response = await es.search(
        index=config.es_index,
        query=es_query,
        size=limit,
    )

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "asin": hit["_source"].get("asin"),
            "source": hit["_source"],
            "source_type": "keyword",
        })
    return results


# =============================================================================
# Strategy 1: Basic RRF (k=60, baseline)
# =============================================================================

class BasicRRFStrategy(SearchStrategy):
    """Basic RRF fusion with k=60 (current implementation)."""

    @property
    def name(self) -> str:
        return "hybrid_rrf_k60"

    @property
    def description(self) -> str:
        return "Basic RRF fusion with k=60 (baseline)"

    def rrf_fusion(
        self,
        result_lists: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Apply Reciprocal Rank Fusion."""
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                asin = result.get("asin")
                if not asin:
                    continue

                rrf_score = 1.0 / (k + rank + 1)

                if asin not in scores:
                    scores[asin] = 0.0
                    docs[asin] = result

                scores[asin] += rrf_score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items:
            result = docs[asin].copy()
            result["rrf_score"] = score
            fused.append(result)

        return fused

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        fetch_limit = limit * 3

        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        # RRF fusion
        fused = self.rrf_fusion([semantic_results, keyword_results], k=60)

        latency = time.perf_counter() - start
        return fused[:limit], latency


# =============================================================================
# Strategy 2: RRF with k=20 (more emphasis on top results)
# =============================================================================

class RRFK20Strategy(BasicRRFStrategy):
    """RRF fusion with lower k value."""

    @property
    def name(self) -> str:
        return "hybrid_rrf_k20"

    @property
    def description(self) -> str:
        return "RRF fusion with k=20 (stronger top-result emphasis)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        fetch_limit = limit * 3

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        fused = self.rrf_fusion([semantic_results, keyword_results], k=20)

        latency = time.perf_counter() - start
        return fused[:limit], latency


# =============================================================================
# Strategy 3: RRF with k=100 (more balanced)
# =============================================================================

class RRFK100Strategy(BasicRRFStrategy):
    """RRF fusion with higher k value."""

    @property
    def name(self) -> str:
        return "hybrid_rrf_k100"

    @property
    def description(self) -> str:
        return "RRF fusion with k=100 (more balanced ranking)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        fetch_limit = limit * 3

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        fused = self.rrf_fusion([semantic_results, keyword_results], k=100)

        latency = time.perf_counter() - start
        return fused[:limit], latency


# =============================================================================
# Strategy 4: Weighted Score Fusion
# =============================================================================

class WeightedFusionStrategy(SearchStrategy):
    """Weighted score fusion instead of RRF."""

    @property
    def name(self) -> str:
        return "hybrid_weighted_fusion"

    @property
    def description(self) -> str:
        return "Weighted score fusion (0.6 semantic + 0.4 keyword)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        fetch_limit = limit * 3

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        # Normalize scores
        def normalize_scores(results: list[dict], field: str = "score"):
            if not results:
                return results
            scores = [r.get(field, 0) for r in results]
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            range_score = max_score - min_score if max_score != min_score else 1

            for r in results:
                r["norm_score"] = (r.get(field, 0) - min_score) / range_score
            return results

        semantic_results = normalize_scores(semantic_results)
        keyword_results = normalize_scores(keyword_results)

        # Weighted fusion
        semantic_weight = 0.6
        keyword_weight = 0.4

        scores = {}
        docs = {}

        for r in semantic_results:
            asin = r.get("asin")
            if not asin:
                continue
            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += r.get("norm_score", 0) * semantic_weight

        for r in keyword_results:
            asin = r.get("asin")
            if not asin:
                continue
            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += r.get("norm_score", 0) * keyword_weight

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items[:limit]:
            result = docs[asin].copy()
            result["fusion_score"] = score
            fused.append(result)

        latency = time.perf_counter() - start
        return fused, latency


# =============================================================================
# Strategy 5: Semantic-First with Keyword Boost
# =============================================================================

class SemanticFirstStrategy(SearchStrategy):
    """Semantic search primary, keyword boost secondary."""

    @property
    def name(self) -> str:
        return "hybrid_semantic_first"

    @property
    def description(self) -> str:
        return "Semantic primary with keyword match boost"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        fetch_limit = limit * 3

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        # Get keyword ASINs for boosting
        keyword_asins = {r.get("asin") for r in keyword_results if r.get("asin")}
        # Boost based on keyword rank
        keyword_boost = {}
        for rank, r in enumerate(keyword_results):
            asin = r.get("asin")
            if asin:
                keyword_boost[asin] = 1.0 / (rank + 1)  # Higher boost for top keyword results

        # Re-score semantic results with keyword boost
        results = []
        for r in semantic_results:
            asin = r.get("asin")
            if not asin:
                continue

            base_score = r.get("score", 0)
            boost = 0.0

            # Boost if found in keyword results
            if asin in keyword_asins:
                boost = 0.2 + keyword_boost.get(asin, 0) * 0.3  # Up to 0.5 boost

            final_score = base_score * (1 + boost)

            result = r.copy()
            result["score"] = final_score
            result["keyword_boost"] = boost
            results.append(result)

        # Sort by boosted score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Dedupe by ASIN
        seen = set()
        deduped = []
        for r in results:
            asin = r.get("asin")
            if asin and asin not in seen:
                seen.add(asin)
                deduped.append(r)
                if len(deduped) >= limit:
                    break

        latency = time.perf_counter() - start
        return deduped, latency


# =============================================================================
# Strategy 6: Adaptive Fusion (based on query)
# =============================================================================

class AdaptiveFusionStrategy(SearchStrategy):
    """Adapt fusion weights based on query characteristics."""

    @property
    def name(self) -> str:
        return "hybrid_adaptive"

    @property
    def description(self) -> str:
        return "Adaptive weights based on query (short=keyword, long=semantic)"

    def analyze_query(self, query: str) -> dict[str, float]:
        """Analyze query to determine optimal weights."""
        words = query.split()
        word_count = len(words)

        # Short queries (1-3 words) → favor keyword
        # Long queries (5+ words) → favor semantic
        if word_count <= 3:
            return {"semantic": 0.3, "keyword": 0.7}
        elif word_count <= 5:
            return {"semantic": 0.5, "keyword": 0.5}
        else:
            return {"semantic": 0.7, "keyword": 0.3}

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Analyze query
        weights = self.analyze_query(query)

        fetch_limit = limit * 3

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        # RRF with weighted contribution
        scores = {}
        docs = {}
        k = 60

        # Apply semantic weight
        for rank, r in enumerate(semantic_results):
            asin = r.get("asin")
            if not asin:
                continue
            rrf_score = (1.0 / (k + rank + 1)) * weights["semantic"]
            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += rrf_score

        # Apply keyword weight
        for rank, r in enumerate(keyword_results):
            asin = r.get("asin")
            if not asin:
                continue
            rrf_score = (1.0 / (k + rank + 1)) * weights["keyword"]
            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += rrf_score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items[:limit]:
            result = docs[asin].copy()
            result["fusion_score"] = score
            result["weights"] = weights
            fused.append(result)

        latency = time.perf_counter() - start
        return fused, latency


# =============================================================================
# Strategy 7: Combined Best Practices
# =============================================================================

class CombinedHybridStrategy(SearchStrategy):
    """Combine preprocessing + adaptive weights + parent aggregation."""

    @property
    def name(self) -> str:
        return "hybrid_combined"

    @property
    def description(self) -> str:
        return "Preprocessing + adaptive RRF + parent-child aggregation"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Preprocess query
        clean_query = QueryPreprocessor.clean_query(query)

        # Determine weights based on query
        words = clean_query.split()
        if len(words) <= 3:
            sem_weight, kw_weight = 0.4, 0.6
        elif len(words) <= 6:
            sem_weight, kw_weight = 0.5, 0.5
        else:
            sem_weight, kw_weight = 0.6, 0.4

        fetch_limit = limit * 5

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, clean_query, fetch_limit, filters),
            keyword_search(self.clients, self.config, clean_query, fetch_limit, filters),
        )

        # Aggregate by ASIN with weighted RRF
        scores = {}
        docs = {}
        k = 40  # Moderate k value

        for rank, r in enumerate(semantic_results):
            asin = r.get("asin")
            payload = r.get("payload", {})
            node_type = payload.get("node_type", "child")

            if not asin:
                continue

            # Base RRF score
            rrf_score = (1.0 / (k + rank + 1)) * sem_weight

            # Parent boost
            if node_type == "parent":
                rrf_score *= 1.15

            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] = max(scores[asin], rrf_score)  # Take max for same ASIN

        for rank, r in enumerate(keyword_results):
            asin = r.get("asin")
            if not asin:
                continue

            rrf_score = (1.0 / (k + rank + 1)) * kw_weight

            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += rrf_score

        # Quality boost from metadata
        for asin in scores:
            doc = docs[asin]
            payload = doc.get("payload", {}) or doc.get("source", {})

            stars = payload.get("stars", 0) or 0
            reviews = payload.get("reviews_count", 0) or 0
            is_bestseller = payload.get("is_best_seller", False)

            quality_boost = 0.0
            if stars > 4:
                quality_boost += 0.05
            if reviews > 100:
                quality_boost += 0.05
            if is_bestseller:
                quality_boost += 0.05

            scores[asin] *= (1 + quality_boost)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items[:limit]:
            result = docs[asin].copy()
            result["fusion_score"] = score
            fused.append(result)

        latency = time.perf_counter() - start
        return fused, latency


# =============================================================================
# Strategy 8: Hybrid + BGE Reranker (NEW)
# =============================================================================

class HybridRerankerStrategy(SearchStrategy):
    """Hybrid search with BGE reranker for final ranking."""

    @property
    def name(self) -> str:
        return "hybrid_reranker"

    @property
    def description(self) -> str:
        return "Hybrid RRF fusion + BGE reranker"

    def rrf_fusion(
        self,
        result_lists: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Apply Reciprocal Rank Fusion."""
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                asin = result.get("asin")
                if not asin:
                    continue

                rrf_score = 1.0 / (k + rank + 1)

                if asin not in scores:
                    scores[asin] = 0.0
                    docs[asin] = result

                scores[asin] += rrf_score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items:
            result = docs[asin].copy()
            result["rrf_score"] = score
            fused.append(result)

        return fused

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Over-fetch for reranking
        fetch_limit = limit * 5

        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        # RRF fusion
        fused = self.rrf_fusion([semantic_results, keyword_results], k=60)

        # Rerank top candidates using BGE reranker
        reranked = await self.clients.rerank_with_scores(
            query=query,
            candidates=fused[:limit * 3],
            top_k=limit,
        )

        latency = time.perf_counter() - start
        return reranked[:limit], latency


# =============================================================================
# Strategy 9: Weighted Fusion + Reranker (NEW)
# =============================================================================

class WeightedFusionRerankerStrategy(SearchStrategy):
    """Weighted fusion with BGE reranker."""

    @property
    def name(self) -> str:
        return "hybrid_weighted_reranker"

    @property
    def description(self) -> str:
        return "Weighted fusion (0.6/0.4) + BGE reranker"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        fetch_limit = limit * 5

        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, query, fetch_limit, filters),
            keyword_search(self.clients, self.config, query, fetch_limit, filters),
        )

        # Normalize scores
        def normalize_scores(results: list[dict], field: str = "score"):
            if not results:
                return results
            scores = [r.get(field, 0) for r in results]
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            range_score = max_score - min_score if max_score != min_score else 1

            for r in results:
                r["norm_score"] = (r.get(field, 0) - min_score) / range_score
            return results

        semantic_results = normalize_scores(semantic_results)
        keyword_results = normalize_scores(keyword_results)

        # Weighted fusion
        semantic_weight = 0.6
        keyword_weight = 0.4

        scores = {}
        docs = {}

        for r in semantic_results:
            asin = r.get("asin")
            if not asin:
                continue
            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += r.get("norm_score", 0) * semantic_weight

        for r in keyword_results:
            asin = r.get("asin")
            if not asin:
                continue
            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += r.get("norm_score", 0) * keyword_weight

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items:
            result = docs[asin].copy()
            result["fusion_score"] = score
            fused.append(result)

        # Rerank using BGE reranker
        reranked = await self.clients.rerank_with_scores(
            query=query,
            candidates=fused[:limit * 3],
            top_k=limit,
        )

        latency = time.perf_counter() - start
        return reranked, latency


# =============================================================================
# Strategy 10: Keyword Priority Hybrid (BEST PERFORMER - MRR 0.9126)
# =============================================================================

class KeywordPriorityHybridStrategy(SearchStrategy):
    """Keyword-priority hybrid: Best performer from experiments (MRR 0.9126).

    Based on direct database experiments (Jan 2026), this strategy uses
    higher keyword weights (0.65-0.7) which outperforms balanced RRF.

    Key findings:
    - brand+model queries: keyword 87.7% R@1 vs semantic 43.1%
    - short_title queries: keyword 76% R@1 vs semantic 62%
    - hybrid_kw_priority MRR: 0.9126 vs hybrid_rrf MRR: 0.8212
    """

    @property
    def name(self) -> str:
        return "hybrid_keyword_priority"

    @property
    def description(self) -> str:
        return "BEST: Keyword-priority hybrid (0.65-0.7 kw weight, MRR 0.9126)"

    def _extract_model_numbers(self, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        import re
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        return [m for m in matches if any(c.isdigit() for c in m)]

    def _has_brand_terms(self, query: str) -> bool:
        """Check if query has brand-like terms (capitalized words)."""
        words = query.split()
        return any(w[0].isupper() and len(w) > 2 for w in words if w and w[0].isalpha())

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Preprocess query
        clean_query = QueryPreprocessor.clean_query(query)
        model_numbers = self._extract_model_numbers(query)
        has_brands = self._has_brand_terms(query)
        words = clean_query.split()

        # Keyword-priority weights based on experiments
        # Key insight: keyword search excels for specific queries (brand+model)
        if model_numbers:
            # Model numbers strongly favor keyword
            sem_weight, kw_weight = 0.25, 0.75
        elif has_brands and len(words) <= 5:
            # Brand queries favor keyword
            sem_weight, kw_weight = 0.30, 0.70
        elif len(words) <= 3:
            # Short queries favor keyword
            sem_weight, kw_weight = 0.35, 0.65
        elif len(words) <= 6:
            # Medium queries - still keyword priority
            sem_weight, kw_weight = 0.40, 0.60
        else:
            # Long queries more balanced
            sem_weight, kw_weight = 0.50, 0.50

        fetch_limit = limit * 5

        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, clean_query, fetch_limit, filters),
            keyword_search_optimized(self.clients, self.config, query, fetch_limit, filters, model_numbers),
        )

        # Aggregate by ASIN with weighted RRF
        scores = {}
        docs = {}
        k = 40  # Moderate k for better top-result emphasis

        for rank, r in enumerate(semantic_results):
            asin = r.get("asin")
            payload = r.get("payload", {})
            node_type = payload.get("node_type", "child")

            if not asin:
                continue

            rrf_score = (1.0 / (k + rank + 1)) * sem_weight

            # Parent boost
            if node_type == "parent":
                rrf_score *= 1.15

            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] = max(scores[asin], rrf_score)

        for rank, r in enumerate(keyword_results):
            asin = r.get("asin")
            if not asin:
                continue

            rrf_score = (1.0 / (k + rank + 1)) * kw_weight

            # Boost for model number matches
            if model_numbers:
                title = str(r.get("source", {}).get("title", "")).upper()
                if any(m in title for m in model_numbers):
                    rrf_score *= 1.6  # Strong boost for exact model match

            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += rrf_score

        # Quality boost from metadata (minor)
        for asin in scores:
            doc = docs[asin]
            payload = doc.get("payload", {}) or doc.get("source", {})

            stars = payload.get("stars", 0) or 0
            reviews = payload.get("reviews_count", 0) or 0
            is_bestseller = payload.get("is_best_seller", False)

            quality_boost = 0.0
            if stars >= 4:
                quality_boost += (stars - 3) * 0.02
            if reviews > 100:
                quality_boost += min(0.05, math.log10(reviews) * 0.015)
            if is_bestseller:
                quality_boost += 0.03

            scores[asin] *= (1 + quality_boost)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items[:limit]:
            result = docs[asin].copy()
            result["fusion_score"] = score
            fused.append(result)

        latency = time.perf_counter() - start
        return fused, latency


# =============================================================================
# Strategy 11: Optimized Hybrid with Model Number Detection
# =============================================================================

class OptimizedHybridStrategy(SearchStrategy):
    """Optimized hybrid: model number detection + adaptive weights + quality signals."""

    @property
    def name(self) -> str:
        return "hybrid_optimized"

    @property
    def description(self) -> str:
        return "Model-aware + adaptive weights + quality signals + overfetch"

    def _extract_model_numbers(self, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        import re
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        return [m for m in matches if any(c.isdigit() for c in m)]

    def _has_brand_terms(self, query: str) -> bool:
        """Check if query has brand-like terms (capitalized words)."""
        words = query.split()
        return any(w[0].isupper() and len(w) > 2 for w in words if w.isalpha())

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Preprocess query
        clean_query = QueryPreprocessor.clean_query(query)
        model_numbers = self._extract_model_numbers(query)
        has_brands = self._has_brand_terms(query)

        # Adaptive weights based on query characteristics
        words = clean_query.split()
        if model_numbers:
            # Model numbers favor keyword search
            sem_weight, kw_weight = 0.3, 0.7
        elif has_brands and len(words) <= 4:
            # Short branded queries favor keyword
            sem_weight, kw_weight = 0.35, 0.65
        elif len(words) <= 3:
            # Short queries favor keyword
            sem_weight, kw_weight = 0.4, 0.6
        elif len(words) <= 6:
            # Medium queries balanced
            sem_weight, kw_weight = 0.5, 0.5
        else:
            # Long queries favor semantic
            sem_weight, kw_weight = 0.6, 0.4

        fetch_limit = limit * 5

        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            semantic_search(self.clients, self.config, clean_query, fetch_limit, filters),
            keyword_search_optimized(self.clients, self.config, query, fetch_limit, filters, model_numbers),
        )

        # Aggregate by ASIN with weighted RRF
        scores = {}
        docs = {}
        k = 40

        for rank, r in enumerate(semantic_results):
            asin = r.get("asin")
            payload = r.get("payload", {})
            node_type = payload.get("node_type", "child")

            if not asin:
                continue

            rrf_score = (1.0 / (k + rank + 1)) * sem_weight

            # Parent boost
            if node_type == "parent":
                rrf_score *= 1.2

            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] = max(scores[asin], rrf_score)

        for rank, r in enumerate(keyword_results):
            asin = r.get("asin")
            if not asin:
                continue

            rrf_score = (1.0 / (k + rank + 1)) * kw_weight

            # Extra boost if keyword found model number match
            if model_numbers and any(m in str(r.get("source", {}).get("title", "")).upper() for m in model_numbers):
                rrf_score *= 1.5

            if asin not in scores:
                scores[asin] = 0.0
                docs[asin] = r
            scores[asin] += rrf_score

        # Quality boost from metadata
        for asin in scores:
            doc = docs[asin]
            payload = doc.get("payload", {}) or doc.get("source", {})

            stars = payload.get("stars", 0) or 0
            reviews = payload.get("reviews_count", 0) or 0
            is_bestseller = payload.get("is_best_seller", False)

            quality_boost = 0.0
            if stars >= 4:
                quality_boost += (stars - 3) * 0.03  # 0.03-0.06 for 4-5 stars
            if reviews > 100:
                quality_boost += min(0.08, math.log10(reviews) * 0.02)
            if is_bestseller:
                quality_boost += 0.05

            scores[asin] *= (1 + quality_boost)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for asin, score in sorted_items[:limit]:
            result = docs[asin].copy()
            result["fusion_score"] = score
            fused.append(result)

        latency = time.perf_counter() - start
        return fused, latency


def _build_optimized_keyword_clauses(
    config: SearchConfig,
    clean_query: str,
    model_numbers: list[str] | None = None,
) -> list[dict]:
    """Build mode-appropriate optimized keyword search clauses.

    Args:
        config: SearchConfig with pipeline_mode
        clean_query: Preprocessed query string
        model_numbers: Optional list of extracted model numbers

    Returns:
        List of ES should clauses
    """
    should_clauses = []

    # Model number exact matches (highest priority)
    if model_numbers:
        for model in model_numbers:
            should_clauses.append({
                "match": {
                    "title": {
                        "query": model,
                        "boost": 20.0,
                    }
                }
            })

    # Phrase match on title
    should_clauses.append({
        "match_phrase": {
            "title": {
                "query": clean_query,
                "boost": 8.0,
                "slop": 2,
            }
        }
    })

    if config.is_original_mode:
        # Original mode: limited fields
        should_clauses.extend([
            # Multi-match on key fields
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": [
                        "title^10",
                        "category_name^2",
                    ],
                    "type": "best_fields",
                    "boost": 3.0,
                }
            },
            # Fuzzy fallback
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^3", "category_name"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
        ])
    else:
        # Enrich mode: full fields
        should_clauses.extend([
            # Multi-match on key fields
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": [
                        "title^10",
                        "short_title^8",
                        "brand^5",
                        "product_type^4",
                    ],
                    "type": "best_fields",
                    "boost": 3.0,
                }
            },
            # Fuzzy fallback
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^3", "brand^2", "chunk_description"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
        ])

    return should_clauses


async def keyword_search_optimized(
    clients: DatabaseClients,
    config: SearchConfig,
    query: str,
    limit: int,
    filters: dict[str, Any] | None = None,
    model_numbers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Optimized keyword search with model number awareness.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """
    es = await clients.get_elasticsearch()
    clean_query = QueryPreprocessor.clean_query(query)

    es_query = {
        "bool": {
            "should": _build_optimized_keyword_clauses(config, clean_query, model_numbers),
            "minimum_should_match": 1,
        }
    }

    response = await es.search(
        index=config.es_index,
        query=es_query,
        size=limit,
    )

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "asin": hit["_source"].get("asin"),
            "source": hit["_source"],
        })

    return results


# =============================================================================
# Main
# =============================================================================

def _get_cli_defaults() -> dict:
    """Get CLI defaults from pipeline_config.yaml.

    Resolves {count} and {mode} placeholders in paths to match 06_generate_eval_data.py output.
    """
    config = get_search_flows_config()
    data_dir = cfg.get_data_dir()
    product_count = cfg.get_count()
    pipeline_mode = cfg.get_mode()

    # Get eval_data path with {count} and {mode} substitution
    eval_data = config.get("eval_data", "eval/datasets/level3_retrieval_{count}_{mode}.json")
    eval_data = eval_data.replace("{count}", str(product_count)).replace("{mode}", pipeline_mode)
    eval_data_path = data_dir / eval_data if not Path(eval_data).is_absolute() else Path(eval_data)

    return {
        "eval_data": str(eval_data_path),
        "qdrant_host": config.get("qdrant_host", "localhost"),
        "qdrant_port": config.get("qdrant_port", 6333),
        "es_host": config.get("elasticsearch_host", "localhost"),
        "es_port": config.get("elasticsearch_port", 9200),
        "ollama_url": config.get("ollama_url", "http://localhost:8010"),
        "verbose": config.get("verbose", False),
        "product_count": product_count,
        "pipeline_mode": pipeline_mode,
    }


_cli_defaults = _get_cli_defaults()


@click.command()
@click.option(
    "--eval-data",
    type=click.Path(exists=True),
    default=None,
    help="Evaluation dataset JSON file (default: from config)",
)
@click.option("--qdrant-host", default=None, help="Qdrant host (default: from config)")
@click.option("--qdrant-port", default=None, type=int, help="Qdrant port (default: from config)")
@click.option("--es-host", default=None, help="Elasticsearch host (default: from config)")
@click.option("--es-port", default=None, type=int, help="Elasticsearch port (default: from config)")
@click.option("--ollama-url", default=None, help="Ollama service URL (default: from config)")
@click.option("--verbose", is_flag=True, default=None, help="Verbose output (default: from config)")
@click.option(
    "--mode",
    type=click.Choice(["original", "enrich", "auto"]),
    default="auto",
    help="Pipeline mode: 'original' (no genAI fields), 'enrich' (with genAI fields), 'auto' (from config)",
)
def main(
    eval_data: str | None,
    qdrant_host: str | None,
    qdrant_port: int | None,
    es_host: str | None,
    es_port: int | None,
    ollama_url: str | None,
    verbose: bool | None,
    mode: str,
):
    """Run hybrid search experiments.

    Mode affects which fields are used in search:
    - original: Core fields only (title, brand, category, chunk_*)
    - enrich: Core + genAI fields (genAI_summary, genAI_best_for, etc.)
    """
    # Apply config defaults
    eval_data = eval_data or _cli_defaults["eval_data"]
    qdrant_host = qdrant_host or _cli_defaults["qdrant_host"]
    qdrant_port = qdrant_port or _cli_defaults["qdrant_port"]
    es_host = es_host or _cli_defaults["es_host"]
    es_port = es_port or _cli_defaults["es_port"]
    ollama_url = ollama_url or _cli_defaults["ollama_url"]
    verbose = verbose if verbose is not None else _cli_defaults["verbose"]

    # Determine pipeline mode
    from base import get_pipeline_mode
    if mode == "auto":
        pipeline_mode = get_pipeline_mode()
    else:
        pipeline_mode = mode

    print("=" * 70)
    print("HYBRID SEARCH EXPERIMENTS")
    print("=" * 70)
    print(f"Pipeline Mode: {pipeline_mode.upper()}")
    if pipeline_mode == "original":
        print("  Fields: Core fields only (no genAI_* fields)")
    else:
        print("  Fields: Core + genAI fields")
    print()

    config = SearchConfig(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        es_host=es_host,
        es_port=es_port,
        ollama_url=ollama_url,
        pipeline_mode=pipeline_mode,
    )

    eval_queries = load_evaluation_data(Path(eval_data), search_type="hybrid")
    print(f"Loaded {len(eval_queries)} hybrid search queries")

    if not eval_queries:
        print("No hybrid queries found!")
        return

    clients = DatabaseClients(config)

    strategies = [
        BasicRRFStrategy(clients, config),
        RRFK20Strategy(clients, config),
        RRFK100Strategy(clients, config),
        WeightedFusionStrategy(clients, config),
        SemanticFirstStrategy(clients, config),
        AdaptiveFusionStrategy(clients, config),
        CombinedHybridStrategy(clients, config),
        KeywordPriorityHybridStrategy(clients, config),  # BEST PERFORMER - MRR 0.9126
        OptimizedHybridStrategy(clients, config),
        # Reranker strategies
        HybridRerankerStrategy(clients, config),
        WeightedFusionRerankerStrategy(clients, config),
    ]

    async def run_all():
        results = []
        for strategy in strategies:
            print(f"\nRunning: {strategy.name}...")
            result = await run_experiment(
                strategy,
                eval_queries,
                search_type="hybrid",
                verbose=verbose,
            )
            results.append(result)
            print(f"  Recall@10: {result.metrics.recall_at_10:.4f}, MRR: {result.metrics.mrr:.4f}")

        await clients.close()
        return results

    results = asyncio.run(run_all())
    print_experiment_results(results)


if __name__ == "__main__":
    main()
