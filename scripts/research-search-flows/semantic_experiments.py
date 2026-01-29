#!/usr/bin/env python3
"""
Semantic Search Flow Experiments

Experiments to optimize semantic search metrics:
1. Basic semantic search (baseline)
2. Parent-only search (filter node_type=parent)
3. Parent-child aggregation (search all, dedupe by ASIN, boost parent)
4. Multi-query retrieval (query expansion + merge)
5. Query preprocessing (clean + keyword extraction)
6. Score threshold tuning
7. Over-fetch and re-rank

Usage:
    python scripts/research-search-flows/semantic_experiments.py --eval-data data/eval/datasets/level3_retrieval_evaluation.json
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
import structlog

# Add to path
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
# Mode-Aware ES Keyword Boost Fields
# =============================================================================

def _get_es_boost_fields(config: SearchConfig) -> list[str]:
    """Get mode-appropriate ES keyword boost fields.

    Used by hybrid strategies that combine semantic search with ES keyword boost.

    Args:
        config: SearchConfig with pipeline_mode

    Returns:
        List of field names with boost values for ES multi_match
    """
    if config.is_original_mode:
        # Original mode: only title available for keyword boost
        return ["title^10", "category_name^2"]
    else:
        # Enrich mode: full field set
        return ["title^10", "brand^5", "product_type^3"]


# =============================================================================
# Strategy 1: Basic Semantic Search (Baseline)
# =============================================================================

class BasicSemanticStrategy(SearchStrategy):
    """Basic semantic search - current implementation baseline."""

    @property
    def name(self) -> str:
        return "semantic_basic"

    @property
    def description(self) -> str:
        return "Basic vector search without filters (baseline)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Get embedding
        embedding = await self.clients.get_embedding(query)

        # Search Qdrant
        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit,
            with_payload=True,
        )

        # Format results
        results = []
        for point in response.points:
            results.append({
                "id": str(point.id),
                "score": point.score,
                "asin": point.payload.get("asin") or point.payload.get("parent_asin"),
                "payload": point.payload,
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 2: Parent-Only Search
# =============================================================================

class ParentOnlySemanticStrategy(SearchStrategy):
    """Search only parent nodes for product-level results."""

    @property
    def name(self) -> str:
        return "semantic_parent_only"

    @property
    def description(self) -> str:
        return "Vector search filtered to parent nodes only"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        embedding = await self.clients.get_embedding(query)

        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
            ),
            with_payload=True,
        )

        results = []
        for point in response.points:
            results.append({
                "id": str(point.id),
                "score": point.score,
                "asin": point.payload.get("asin"),
                "payload": point.payload,
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 3: Parent-Child Aggregation
# =============================================================================

class ParentChildAggregationStrategy(SearchStrategy):
    """Search all nodes, aggregate by product, boost parent matches."""

    @property
    def name(self) -> str:
        return "semantic_parent_child_agg"

    @property
    def description(self) -> str:
        return "Search all nodes, aggregate by ASIN, boost parent matches"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        embedding = await self.clients.get_embedding(query)

        # Fetch more results for aggregation
        fetch_limit = limit * 5

        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            with_payload=True,
        )

        # Aggregate by ASIN with scoring
        asin_scores = {}
        asin_data = {}

        for point in response.points:
            payload = point.payload
            asin = payload.get("asin") or payload.get("parent_asin")
            if not asin:
                continue

            node_type = payload.get("node_type", "child")
            score = point.score

            # Boost parent matches
            if node_type == "parent":
                score *= 1.2

            if asin not in asin_scores:
                asin_scores[asin] = 0.0
                asin_data[asin] = {"payload": payload, "id": str(point.id)}

            # Accumulate score (could use max instead)
            asin_scores[asin] = max(asin_scores[asin], score)

        # Sort by aggregated score
        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 4: Multi-Query Retrieval
# =============================================================================

class MultiQuerySemanticStrategy(SearchStrategy):
    """Generate query variations, search each, merge results."""

    @property
    def name(self) -> str:
        return "semantic_multi_query"

    @property
    def description(self) -> str:
        return "Query expansion with multiple variations merged via RRF"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Generate query variations
        queries = self.preprocessor.expand_query(query)

        # Get embeddings for all queries
        embeddings = await self.clients.get_embeddings_batch(queries)

        qdrant = self.clients.get_qdrant()

        # Search with each embedding
        all_results = []
        for emb in embeddings:
            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=emb,
                limit=limit * 2,
                with_payload=True,
            )
            all_results.append(response.points)

        # RRF fusion
        asin_scores = {}
        asin_data = {}
        k = 60

        for results in all_results:
            for rank, point in enumerate(results):
                asin = point.payload.get("asin") or point.payload.get("parent_asin")
                if not asin:
                    continue

                rrf_score = 1.0 / (k + rank + 1)

                if asin not in asin_scores:
                    asin_scores[asin] = 0.0
                    asin_data[asin] = {"payload": point.payload, "id": str(point.id)}

                asin_scores[asin] += rrf_score

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 5: Query Preprocessing Enhanced
# =============================================================================

class PreprocessedSemanticStrategy(SearchStrategy):
    """Clean and preprocess query before embedding."""

    @property
    def name(self) -> str:
        return "semantic_preprocessed"

    @property
    def description(self) -> str:
        return "Query cleaning + keyword extraction before embedding"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Clean query
        clean_query = self.preprocessor.clean_query(query)

        # Extract and append keywords for emphasis
        keywords = self.preprocessor.extract_keywords(clean_query)
        if keywords:
            enhanced_query = f"{clean_query} {' '.join(keywords[:5])}"
        else:
            enhanced_query = clean_query

        embedding = await self.clients.get_embedding(enhanced_query)

        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
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
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 6: Over-fetch and Re-rank
# =============================================================================

class OverfetchRerankStrategy(SearchStrategy):
    """Fetch more results, re-rank by multiple signals."""

    @property
    def name(self) -> str:
        return "semantic_overfetch_rerank"

    @property
    def description(self) -> str:
        return "Fetch 3x results, re-rank using score + popularity + rating"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        embedding = await self.clients.get_embedding(query)

        # Over-fetch
        fetch_limit = limit * 3

        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            with_payload=True,
        )

        # Re-rank with multiple signals
        candidates = []
        for point in response.points:
            payload = point.payload

            # Base score from vector similarity
            base_score = point.score

            # Boost factors
            stars = payload.get("stars", 0) or 0
            reviews = payload.get("reviews_count", 0) or 0
            is_bestseller = payload.get("is_best_seller", False)

            # Composite score
            # Normalize: stars (0-5), reviews (log scale), bestseller bonus
            import math
            popularity_boost = min(math.log10(reviews + 1) / 5, 0.2) if reviews > 0 else 0
            rating_boost = (stars / 5) * 0.1 if stars > 0 else 0
            bestseller_boost = 0.05 if is_bestseller else 0

            final_score = base_score * (1 + popularity_boost + rating_boost + bestseller_boost)

            candidates.append({
                "id": str(point.id),
                "score": final_score,
                "original_score": base_score,
                "asin": payload.get("asin") or payload.get("parent_asin"),
                "payload": payload,
            })

        # Sort by re-ranked score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Dedupe by ASIN
        seen = set()
        results = []
        for c in candidates:
            if c["asin"] and c["asin"] not in seen:
                seen.add(c["asin"])
                results.append(c)
                if len(results) >= limit:
                    break

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 7: Combined Best Practices
# =============================================================================

class CombinedSemanticStrategy(SearchStrategy):
    """Combine preprocessing + parent-child aggregation + re-ranking."""

    @property
    def name(self) -> str:
        return "semantic_combined"

    @property
    def description(self) -> str:
        return "Preprocessing + parent-child agg + multi-signal re-ranking"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        import math
        start = time.perf_counter()

        # Step 1: Preprocess query
        clean_query = self.preprocessor.clean_query(query)
        keywords = self.preprocessor.extract_keywords(clean_query)

        # Step 2: Create enhanced query
        if keywords:
            enhanced_query = f"{clean_query} {' '.join(keywords[:3])}"
        else:
            enhanced_query = clean_query

        # Step 3: Get embedding
        embedding = await self.clients.get_embedding(enhanced_query)

        # Step 4: Over-fetch from Qdrant
        fetch_limit = limit * 5
        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            with_payload=True,
        )

        # Step 5: Aggregate by ASIN with smart scoring
        asin_scores = {}
        asin_data = {}

        for point in response.points:
            payload = point.payload
            asin = payload.get("asin") or payload.get("parent_asin")
            if not asin:
                continue

            node_type = payload.get("node_type", "child")
            base_score = point.score

            # Node type boost
            if node_type == "parent":
                base_score *= 1.15

            # Quality signals
            stars = payload.get("stars", 0) or 0
            reviews = payload.get("reviews_count", 0) or 0
            is_bestseller = payload.get("is_best_seller", False)

            popularity_boost = min(math.log10(reviews + 1) / 6, 0.15) if reviews > 0 else 0
            rating_boost = (stars / 5) * 0.08 if stars > 0 else 0
            bestseller_boost = 0.05 if is_bestseller else 0

            final_score = base_score * (1 + popularity_boost + rating_boost + bestseller_boost)

            if asin not in asin_scores:
                asin_scores[asin] = 0.0
                asin_data[asin] = {"payload": payload, "id": str(point.id)}

            # Take max score for this ASIN
            if final_score > asin_scores[asin]:
                asin_scores[asin] = final_score
                asin_data[asin] = {"payload": payload, "id": str(point.id)}

        # Step 6: Sort and return top results
        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 8: Aggressive Multi-Query with Parent Focus (NEW)
# =============================================================================

class AggressiveMultiQueryStrategy(SearchStrategy):
    """Enhanced multi-query with more query variants and parent prioritization."""

    @property
    def name(self) -> str:
        return "semantic_aggressive_multi"

    @property
    def description(self) -> str:
        return "Multiple query variants + parent focus + RRF fusion"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Clean query
        clean_query = self.preprocessor.clean_query(query)
        keywords = self.preprocessor.extract_keywords(clean_query)

        # Generate multiple query variants
        queries = [clean_query]

        # Add keyword-only version
        if keywords:
            queries.append(' '.join(keywords))
            # Add top 3 keywords only
            if len(keywords) >= 3:
                queries.append(' '.join(keywords[:3]))

        # Get embeddings for all queries (uses fallback if batch not available)
        embeddings = await self.clients.get_embeddings_batch(queries)

        qdrant = self.clients.get_qdrant()

        # Search with each embedding
        all_results = []
        for emb in embeddings:
            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=emb,
                limit=limit * 3,
                with_payload=True,
            )
            all_results.append(response.points)

        # RRF fusion with parent boosting
        asin_scores = {}
        asin_data = {}
        k = 40  # Lower k for more emphasis on top results

        for results in all_results:
            for rank, point in enumerate(results):
                payload = point.payload
                asin = payload.get("asin") or payload.get("parent_asin")
                if not asin:
                    continue

                # Base RRF score
                rrf_score = 1.0 / (k + rank + 1)

                # Boost for parent nodes
                if payload.get("node_type") == "parent":
                    rrf_score *= 1.3

                if asin not in asin_scores:
                    asin_scores[asin] = 0.0
                    asin_data[asin] = {"payload": payload, "id": str(point.id)}

                asin_scores[asin] += rrf_score

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 9: Hybrid Semantic (Qdrant + ES boost) (NEW)
# =============================================================================

class HybridBoostSemanticStrategy(SearchStrategy):
    """Semantic search with Elasticsearch title match boosting."""

    @property
    def name(self) -> str:
        return "semantic_hybrid_boost"

    @property
    def description(self) -> str:
        return "Semantic search boosted by ES title matches"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        import asyncio

        start = time.perf_counter()

        clean_query = self.preprocessor.clean_query(query)

        # Get embedding
        embedding = await self.clients.get_embedding(clean_query)

        # Run semantic and keyword search in parallel
        qdrant = self.clients.get_qdrant()

        semantic_response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 5,
            with_payload=True,
        )

        # Get ES keyword matches for boosting (mode-aware fields)
        es = await self.clients.get_elasticsearch()
        es_response = await es.search(
            index=self.config.es_index,
            query={
                "multi_match": {
                    "query": clean_query,
                    "fields": _get_es_boost_fields(self.config),
                    "type": "best_fields",
                }
            },
            size=limit * 3,
        )

        # Extract ES matches with scores
        es_scores = {}
        for rank, hit in enumerate(es_response["hits"]["hits"]):
            asin = hit["_source"].get("asin")
            if asin:
                # Higher boost for top ES matches
                es_scores[asin] = 1.0 / (rank + 1)

        # Score semantic results with ES boost
        asin_scores = {}
        asin_data = {}

        for point in semantic_response.points:
            payload = point.payload
            asin = payload.get("asin") or payload.get("parent_asin")
            if not asin:
                continue

            base_score = point.score

            # ES match boost
            if asin in es_scores:
                boost = min(es_scores[asin] * 0.5, 0.4)  # Up to 40% boost
                base_score *= (1 + boost)

            # Parent boost
            if payload.get("node_type") == "parent":
                base_score *= 1.15

            if asin not in asin_scores or base_score > asin_scores[asin]:
                asin_scores[asin] = base_score
                asin_data[asin] = {"payload": payload, "id": str(point.id)}

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 10: Format-Aware Semantic Search (Mode-Aware)
# =============================================================================

class FormatAwareSemanticStrategy(SearchStrategy):
    """Format query to match document embedding format for better similarity.

    Mode-aware behavior:
    - Original mode: Documents embedded as plain text (title + category)
                    Query is cleaned and focused on product keywords
    - Enrich mode: Documents embedded with structure (Product:, Brand:, etc.)
                   Query is formatted to match this structure

    In enrich mode: Plain queries get ~0.55 similarity, but formatted queries get ~0.99!
    """

    @property
    def name(self) -> str:
        return "semantic_format_aware"

    @property
    def description(self) -> str:
        return "Format query to match document embedding format (mode-aware)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Clean query
        clean_query = self.preprocessor.clean_query(query)

        # Extract brand if present in query
        brand, query_without_brand = self.preprocessor.extract_brand_from_query(clean_query)

        # Format query based on pipeline mode
        formatted_query = self.preprocessor.format_query_for_mode(
            clean_query,
            pipeline_mode=self.config.pipeline_mode,
            brand=brand,
        )

        # Get embedding for formatted query
        embedding = await self.clients.get_embedding(formatted_query)

        # Search with parent filter for best results
        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 3,  # Over-fetch for deduplication
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
            ),
            with_payload=True,
        )

        # Dedupe and format results
        seen = set()
        results = []
        for point in response.points:
            payload = point.payload
            asin = payload.get("asin")
            if asin and asin not in seen:
                seen.add(asin)
                results.append({
                    "id": str(point.id),
                    "score": point.score,
                    "asin": asin,
                    "payload": payload,
                })
                if len(results) >= limit:
                    break

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 11: Format-Aware + Multi-Query (NEW)
# =============================================================================

class FormatAwareMultiQueryStrategy(SearchStrategy):
    """Combine format-aware embedding with multi-query retrieval."""

    @property
    def name(self) -> str:
        return "semantic_format_multi"

    @property
    def description(self) -> str:
        return "Format-aware embedding + multi-query variants + RRF fusion"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        clean_query = self.preprocessor.clean_query(query)
        brand, query_without_brand = self.preprocessor.extract_brand_from_query(clean_query)
        keywords = self.preprocessor.extract_keywords(clean_query)

        # Generate multiple query variants in document format
        queries_to_embed = []

        # 1. Full query formatted
        queries_to_embed.append(
            self.preprocessor.format_for_semantic_search(clean_query, brand=brand)
        )

        # 2. Query without brand formatted (if brand was found)
        if brand and query_without_brand:
            queries_to_embed.append(
                self.preprocessor.format_for_semantic_search(query_without_brand, brand=brand)
            )

        # 3. Keywords only formatted
        if keywords:
            keyword_query = ' '.join(keywords[:5])
            queries_to_embed.append(
                self.preprocessor.format_for_semantic_search(keyword_query, brand=brand)
            )

        # Get embeddings for all variants
        embeddings = await self.clients.get_embeddings_batch(queries_to_embed)

        qdrant = self.clients.get_qdrant()

        # Search with each embedding (parent-only)
        all_results = []
        for emb in embeddings:
            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=emb,
                limit=limit * 2,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                ),
                with_payload=True,
            )
            all_results.append(response.points)

        # RRF fusion
        asin_scores = {}
        asin_data = {}
        k = 40

        for results in all_results:
            for rank, point in enumerate(results):
                asin = point.payload.get("asin")
                if not asin:
                    continue

                rrf_score = 1.0 / (k + rank + 1)

                if asin not in asin_scores:
                    asin_scores[asin] = 0.0
                    asin_data[asin] = {"payload": point.payload, "id": str(point.id)}

                asin_scores[asin] += rrf_score

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 12: Format-Aware + Hybrid Boost (NEW - BEST)
# =============================================================================

class FormatAwareHybridStrategy(SearchStrategy):
    """Format-aware semantic + ES keyword boost for best results."""

    @property
    def name(self) -> str:
        return "semantic_format_hybrid"

    @property
    def description(self) -> str:
        return "Format-aware semantic + ES keyword boost (RECOMMENDED)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        clean_query = self.preprocessor.clean_query(query)
        brand, _ = self.preprocessor.extract_brand_from_query(clean_query)

        # Format query for semantic search
        formatted_query = self.preprocessor.format_for_semantic_search(clean_query, brand=brand)
        embedding = await self.clients.get_embedding(formatted_query)

        # Run semantic search (parent-only)
        qdrant = self.clients.get_qdrant()
        semantic_response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 5,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
            ),
            with_payload=True,
        )

        # Run ES keyword search for boosting (mode-aware fields)
        es = await self.clients.get_elasticsearch()
        es_response = await es.search(
            index=self.config.es_index,
            query={
                "multi_match": {
                    "query": clean_query,
                    "fields": _get_es_boost_fields(self.config),
                    "type": "best_fields",
                }
            },
            size=limit * 3,
        )

        # Extract ES matches
        es_asins = {}
        for rank, hit in enumerate(es_response["hits"]["hits"]):
            asin = hit["_source"].get("asin")
            if asin:
                es_asins[asin] = 1.0 / (rank + 1)  # Rank-based score

        # Score semantic results with ES boost
        asin_scores = {}
        asin_data = {}

        for point in semantic_response.points:
            payload = point.payload
            asin = payload.get("asin")
            if not asin:
                continue

            base_score = point.score

            # Strong boost for ES matches (keyword confirms semantic)
            if asin in es_asins:
                # Up to 60% boost for top ES matches
                boost = min(es_asins[asin] * 0.6, 0.6)
                base_score *= (1 + boost)

            if asin not in asin_scores or base_score > asin_scores[asin]:
                asin_scores[asin] = base_score
                asin_data[asin] = {"payload": payload, "id": str(point.id)}

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 13: Optimized Semantic (BEST PERFORMER for semantic-only)
# =============================================================================

class OptimizedSemanticStrategy(SearchStrategy):
    """Optimized semantic search combining best practices.

    Based on experiments (Jan 2026):
    - Semantic search MRR: 0.6482 (vs hybrid_kw_priority 0.9126)
    - Works best for: generic/conceptual queries
    - Struggles with: brand+model specific queries (43.1% R@1 vs keyword 87.7%)

    This strategy optimizes for the use cases where semantic search excels.
    """

    @property
    def name(self) -> str:
        return "semantic_optimized"

    @property
    def description(self) -> str:
        return "BEST semantic: parent-only + quality boost + ES keyword assist"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        import math
        start = time.perf_counter()

        # Preprocess query
        clean_query = self.preprocessor.clean_query(query)
        keywords = self.preprocessor.extract_keywords(clean_query)

        # Get embedding
        embedding = await self.clients.get_embedding(clean_query)

        # Over-fetch from parent nodes
        fetch_limit = limit * 5
        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
            ),
            with_payload=True,
        )

        # Get ES keyword matches for boosting (mode-aware fields)
        es = await self.clients.get_elasticsearch()
        es_response = await es.search(
            index=self.config.es_index,
            query={
                "multi_match": {
                    "query": clean_query,
                    "fields": _get_es_boost_fields(self.config),
                    "type": "best_fields",
                }
            },
            size=limit * 3,
        )

        # Extract ES matches with rank scores
        es_ranks = {}
        for rank, hit in enumerate(es_response["hits"]["hits"]):
            asin = hit["_source"].get("asin")
            if asin:
                es_ranks[asin] = rank + 1

        # Score semantic results with multiple signals
        asin_scores = {}
        asin_data = {}

        for point in response.points:
            payload = point.payload
            asin = payload.get("asin")
            if not asin:
                continue

            base_score = point.score

            # Keyword boost: If ES found it in top results, boost
            if asin in es_ranks:
                kw_boost = min(0.4, 0.4 / es_ranks[asin])  # Up to 40% for rank 1
                base_score *= (1 + kw_boost)

            # Title keyword match boost
            title = (payload.get("title") or "").lower()
            title_boost = 0.0
            for kw in keywords:
                if kw.lower() in title:
                    title_boost += 0.08
            title_boost = min(title_boost, 0.3)
            base_score *= (1 + title_boost)

            # Quality signals
            stars = payload.get("stars", 0) or 0
            reviews = payload.get("reviews_count", 0) or 0
            is_bestseller = payload.get("is_best_seller", False)

            quality_boost = 0.0
            if stars >= 4:
                quality_boost += (stars - 3) * 0.02
            if reviews > 100:
                quality_boost += min(0.06, math.log10(reviews) * 0.015)
            if is_bestseller:
                quality_boost += 0.03

            final_score = base_score * (1 + quality_boost)

            if asin not in asin_scores or final_score > asin_scores[asin]:
                asin_scores[asin] = final_score
                asin_data[asin] = {"payload": payload, "id": str(point.id)}

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 14: Semantic + BGE Reranker (NEW)
# =============================================================================

class SemanticRerankerStrategy(SearchStrategy):
    """Semantic search with BGE reranker for improved ranking."""

    @property
    def name(self) -> str:
        return "semantic_reranker"

    @property
    def description(self) -> str:
        return "Semantic search + BGE reranker (qllama/bge-reranker-large)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Get embedding
        embedding = await self.clients.get_embedding(query)

        # Over-fetch for reranking
        fetch_limit = limit * 3

        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            with_payload=True,
        )

        # Prepare candidates for reranking
        candidates = []
        for point in response.points:
            payload = point.payload
            candidates.append({
                "id": str(point.id),
                "score": point.score,
                "asin": payload.get("asin") or payload.get("parent_asin"),
                "payload": payload,
            })

        # Rerank using BGE reranker
        reranked = await self.clients.rerank_with_scores(
            query=query,
            candidates=candidates,
            top_k=fetch_limit,
        )

        # Dedupe by ASIN and take top results
        seen = set()
        results = []
        for r in reranked:
            asin = r.get("asin")
            if asin and asin not in seen:
                seen.add(asin)
                results.append(r)
                if len(results) >= limit:
                    break

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 11: Parent-Only + Reranker (NEW)
# =============================================================================

class ParentOnlyRerankerStrategy(SearchStrategy):
    """Parent-only semantic search with BGE reranker."""

    @property
    def name(self) -> str:
        return "semantic_parent_reranker"

    @property
    def description(self) -> str:
        return "Parent-only search + BGE reranker"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        embedding = await self.clients.get_embedding(query)

        # Over-fetch for reranking
        fetch_limit = limit * 3

        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
            ),
            with_payload=True,
        )

        # Prepare candidates for reranking
        candidates = []
        for point in response.points:
            payload = point.payload
            candidates.append({
                "id": str(point.id),
                "score": point.score,
                "asin": payload.get("asin"),
                "payload": payload,
            })

        # Rerank using BGE reranker
        reranked = await self.clients.rerank_with_scores(
            query=query,
            candidates=candidates,
            top_k=limit,
        )

        latency = time.perf_counter() - start
        return reranked[:limit], latency


# =============================================================================
# Strategy 12: Combined + Reranker (NEW - Best of Both Worlds)
# =============================================================================

class CombinedRerankerStrategy(SearchStrategy):
    """Combined semantic strategies + BGE reranker for best results."""

    @property
    def name(self) -> str:
        return "semantic_combined_reranker"

    @property
    def description(self) -> str:
        return "Preprocessing + parent-child agg + BGE reranker"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Step 1: Preprocess query
        clean_query = self.preprocessor.clean_query(query)
        keywords = self.preprocessor.extract_keywords(clean_query)

        # Step 2: Create enhanced query for embedding
        if keywords:
            enhanced_query = f"{clean_query} {' '.join(keywords[:3])}"
        else:
            enhanced_query = clean_query

        # Step 3: Get embedding
        embedding = await self.clients.get_embedding(enhanced_query)

        # Step 4: Over-fetch from Qdrant
        fetch_limit = limit * 5
        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            with_payload=True,
        )

        # Step 5: Aggregate by ASIN (take best score per ASIN)
        asin_candidates = {}
        for point in response.points:
            payload = point.payload
            asin = payload.get("asin") or payload.get("parent_asin")
            if not asin:
                continue

            if asin not in asin_candidates or point.score > asin_candidates[asin]["score"]:
                asin_candidates[asin] = {
                    "id": str(point.id),
                    "score": point.score,
                    "asin": asin,
                    "payload": payload,
                }

        candidates = list(asin_candidates.values())

        # Step 6: Rerank using BGE reranker
        reranked = await self.clients.rerank_with_scores(
            query=clean_query,  # Use clean query for reranking
            candidates=candidates,
            top_k=limit,
        )

        latency = time.perf_counter() - start
        return reranked[:limit], latency


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
    ollama_url: str | None,
    verbose: bool | None,
    mode: str,
):
    """Run semantic search experiments.

    Mode affects which fields are used in search:
    - original: Core fields only (title, brand, category, chunk_*)
    - enrich: Core + genAI fields (genAI_summary, genAI_best_for, etc.)
    """
    # Apply config defaults
    eval_data = eval_data or _cli_defaults["eval_data"]
    qdrant_host = qdrant_host or _cli_defaults["qdrant_host"]
    qdrant_port = qdrant_port or _cli_defaults["qdrant_port"]
    ollama_url = ollama_url or _cli_defaults["ollama_url"]
    verbose = verbose if verbose is not None else _cli_defaults["verbose"]

    # Determine pipeline mode
    from base import get_pipeline_mode
    if mode == "auto":
        pipeline_mode = get_pipeline_mode()
    else:
        pipeline_mode = mode

    print("=" * 70)
    print("SEMANTIC SEARCH EXPERIMENTS")
    print("=" * 70)
    print(f"Pipeline Mode: {pipeline_mode.upper()}")
    if pipeline_mode == "original":
        print("  Fields: Core fields only (no genAI_* fields)")
    else:
        print("  Fields: Core + genAI fields")
    print()

    # Configuration
    config = SearchConfig(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        ollama_url=ollama_url,
        pipeline_mode=pipeline_mode,
    )

    # Load evaluation data
    eval_queries = load_evaluation_data(Path(eval_data), search_type="semantic")
    print(f"Loaded {len(eval_queries)} semantic search queries")

    if not eval_queries:
        print("No semantic queries found in evaluation data!")
        return

    # Initialize clients
    clients = DatabaseClients(config)

    # Define strategies to test
    strategies = [
        BasicSemanticStrategy(clients, config),
        ParentOnlySemanticStrategy(clients, config),
        ParentChildAggregationStrategy(clients, config),
        MultiQuerySemanticStrategy(clients, config),
        PreprocessedSemanticStrategy(clients, config),
        OverfetchRerankStrategy(clients, config),
        CombinedSemanticStrategy(clients, config),
        AggressiveMultiQueryStrategy(clients, config),
        HybridBoostSemanticStrategy(clients, config),
        # Format-aware strategies (CRITICAL FIX for embedding mismatch)
        FormatAwareSemanticStrategy(clients, config),
        FormatAwareMultiQueryStrategy(clients, config),
        FormatAwareHybridStrategy(clients, config),
        # Optimized strategy (BEST for semantic-only)
        OptimizedSemanticStrategy(clients, config),
        # Reranker strategies
        SemanticRerankerStrategy(clients, config),
        ParentOnlyRerankerStrategy(clients, config),
        CombinedRerankerStrategy(clients, config),
    ]

    # Run experiments
    async def run_all():
        results = []
        for strategy in strategies:
            print(f"\nRunning: {strategy.name}...")
            result = await run_experiment(
                strategy,
                eval_queries,
                search_type="semantic",
                verbose=verbose,
            )
            results.append(result)
            print(f"  Recall@10: {result.metrics.recall_at_10:.4f}, MRR: {result.metrics.mrr:.4f}")

        await clients.close()
        return results

    results = asyncio.run(run_all())

    # Print comparison
    print_experiment_results(results)


if __name__ == "__main__":
    main()
