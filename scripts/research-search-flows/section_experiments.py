#!/usr/bin/env python3
"""
Section Search Flow Experiments

Experiments to optimize section-targeted search:
1. Basic section filter (baseline)
2. Section embedding with parent fallback
3. Cross-section search with boost
4. Section-aware query expansion
5. Hierarchical search (children then parent)

Usage:
    python scripts/research-search-flows/section_experiments.py --eval-data data/eval/datasets/level3_retrieval_evaluation.json
"""

import asyncio
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
    EvalQuery,
    get_search_flows_config,
)
import config as cfg
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = structlog.get_logger()


# =============================================================================
# Section Keywords for Query Enhancement (Original Mode)
# =============================================================================

SECTION_KEYWORDS = {
    "reviews": ["review", "rating", "customer feedback", "opinion", "satisfaction"],
    "specs": ["specification", "technical", "dimensions", "capacity", "power"],
    "features": ["feature", "capability", "function", "benefit", "includes"],
    "use_cases": ["use case", "application", "suitable for", "ideal for", "perfect for"],
    "description": ["description", "overview", "about", "details", "summary"],
}


# =============================================================================
# Strategy 1: Basic Section Filter (Baseline)
# =============================================================================

class BasicSectionStrategy(SearchStrategy):
    """Basic section search - filter by section and search.

    Mode-aware behavior:
    - Original mode: Search parent nodes, enhance query with section keywords
    - Enrich mode: Search child nodes filtered by section
    """

    @property
    def name(self) -> str:
        return "section_basic"

    @property
    def description(self) -> str:
        return "Basic vector search filtered by section (baseline)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")
        embedding = await self.clients.get_embedding(query)
        qdrant = self.clients.get_qdrant()

        if self.config.pipeline_mode == "original":
            # ORIGINAL MODE: No child nodes exist, search parent nodes
            # Enhance query with section keywords for better matching
            section_keywords = SECTION_KEYWORDS.get(target_section, [])
            enhanced_query = f"{query} {' '.join(section_keywords[:2])}"
            enhanced_embedding = await self.clients.get_embedding(enhanced_query)

            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=enhanced_embedding,
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
                    "section": target_section,  # Requested section (not in data)
                    "payload": point.payload,
                })
        else:
            # ENRICH MODE: Search child nodes filtered by section
            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="node_type", match=MatchValue(value="child")),
                        FieldCondition(key="section", match=MatchValue(value=target_section)),
                    ]
                ),
                with_payload=True,
            )

            results = []
            for point in response.points:
                results.append({
                    "id": str(point.id),
                    "score": point.score,
                    "asin": point.payload.get("parent_asin"),
                    "section": point.payload.get("section"),
                    "payload": point.payload,
                })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 2: Section with Parent Fallback
# =============================================================================

class SectionWithFallbackStrategy(SearchStrategy):
    """Search section first, fall back to parent if needed.

    Mode-aware behavior:
    - Original mode: Go directly to parent search with section-enhanced query
    - Enrich mode: Search child nodes first, fallback to parent
    """

    @property
    def name(self) -> str:
        return "section_with_fallback"

    @property
    def description(self) -> str:
        return "Section search with parent node fallback"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")
        embedding = await self.clients.get_embedding(query)
        qdrant = self.clients.get_qdrant()

        results = []
        seen_asins = set()

        if self.config.pipeline_mode == "original":
            # ORIGINAL MODE: No child nodes, search parent with enhanced query
            section_keywords = SECTION_KEYWORDS.get(target_section, [])
            enhanced_query = f"{query} {' '.join(section_keywords[:2])}"
            enhanced_embedding = await self.clients.get_embedding(enhanced_query)

            parent_response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=enhanced_embedding,
                limit=limit,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                ),
                with_payload=True,
            )

            for point in parent_response.points:
                asin = point.payload.get("asin")
                if asin and asin not in seen_asins:
                    seen_asins.add(asin)
                    results.append({
                        "id": str(point.id),
                        "score": point.score,
                        "asin": asin,
                        "section": target_section,
                        "payload": point.payload,
                        "match_type": "parent_enhanced",
                    })
        else:
            # ENRICH MODE: Search child nodes first
            section_response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="node_type", match=MatchValue(value="child")),
                        FieldCondition(key="section", match=MatchValue(value=target_section)),
                    ]
                ),
                with_payload=True,
            )

            for point in section_response.points:
                asin = point.payload.get("parent_asin")
                if asin and asin not in seen_asins:
                    seen_asins.add(asin)
                    results.append({
                        "id": str(point.id),
                        "score": point.score,
                        "asin": asin,
                        "section": point.payload.get("section"),
                        "payload": point.payload,
                        "match_type": "section",
                    })

            # If not enough results, fall back to parent search
            if len(results) < limit:
                parent_response = qdrant.query_points(
                    collection_name=self.config.qdrant_collection,
                    query=embedding,
                    limit=limit - len(results),
                    query_filter=Filter(
                        must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                    ),
                    with_payload=True,
                )

                for point in parent_response.points:
                    asin = point.payload.get("asin")
                    if asin and asin not in seen_asins:
                        seen_asins.add(asin)
                        results.append({
                            "id": str(point.id),
                            "score": point.score * 0.9,  # Slight penalty for fallback
                            "asin": asin,
                            "section": None,
                            "payload": point.payload,
                            "match_type": "parent_fallback",
                        })

        latency = time.perf_counter() - start
        return results[:limit], latency


# =============================================================================
# Strategy 3: Cross-Section Search with Boost
# =============================================================================

class CrossSectionBoostStrategy(SearchStrategy):
    """Search all sections, boost target section.

    Mode-aware behavior:
    - Original mode: Search parent nodes with section-enhanced query
    - Enrich mode: Search all children, boost target section
    """

    @property
    def name(self) -> str:
        return "section_cross_boost"

    @property
    def description(self) -> str:
        return "Search all children, boost target section matches"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")
        qdrant = self.clients.get_qdrant()

        if self.config.pipeline_mode == "original":
            # ORIGINAL MODE: Search parent nodes with section keywords
            section_keywords = SECTION_KEYWORDS.get(target_section, [])
            enhanced_query = f"{query} {' '.join(section_keywords[:2])}"
            embedding = await self.clients.get_embedding(enhanced_query)

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
                    "section": target_section,
                    "payload": point.payload,
                })
        else:
            # ENRICH MODE: Search all child nodes
            embedding = await self.clients.get_embedding(query)
            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit * 5,  # Over-fetch for boosting
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="child"))]
                ),
                with_payload=True,
            )

            # Aggregate by ASIN with section boosting
            asin_scores = {}
            asin_data = {}

            for point in response.points:
                asin = point.payload.get("parent_asin")
                section = point.payload.get("section")
                if not asin:
                    continue

                base_score = point.score

                # Boost if matching target section
                if section == target_section:
                    base_score *= 1.5

                if asin not in asin_scores:
                    asin_scores[asin] = 0.0
                    asin_data[asin] = {
                        "payload": point.payload,
                        "id": str(point.id),
                        "section": section,
                    }

                if base_score > asin_scores[asin]:
                    asin_scores[asin] = base_score
                    asin_data[asin] = {
                        "payload": point.payload,
                        "id": str(point.id),
                        "section": section,
                    }

            sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for asin, score in sorted_asins[:limit]:
                results.append({
                    "id": asin_data[asin]["id"],
                    "score": score,
                    "asin": asin,
                    "section": asin_data[asin]["section"],
                    "payload": asin_data[asin]["payload"],
                })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 4: Section-Aware Query Expansion
# =============================================================================

class SectionAwareQueryStrategy(SearchStrategy):
    """Expand query with section-specific terms.

    Mode-aware behavior:
    - Original mode: Search parent nodes with expanded query
    - Enrich mode: Search child nodes filtered by section with expanded query
    """

    SECTION_KEYWORDS_EXT = {
        "reviews": ["reviews", "users say", "feedback", "rating", "opinion", "complaint", "praise"],
        "specs": ["specifications", "technical", "dimensions", "compatibility", "requirements"],
        "features": ["features", "capabilities", "functions", "technology", "what it does"],
        "use_cases": ["use cases", "scenarios", "best for", "ideal for", "suitable for"],
        "description": ["description", "overview", "about", "what is", "product details"],
    }

    @property
    def name(self) -> str:
        return "section_query_expansion"

    @property
    def description(self) -> str:
        return "Expand query with section-specific keywords"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")

        # Expand query with section keywords
        section_terms = self.SECTION_KEYWORDS_EXT.get(target_section, [])
        expanded_query = f"{query} {' '.join(section_terms[:3])}"

        embedding = await self.clients.get_embedding(expanded_query)
        qdrant = self.clients.get_qdrant()

        if self.config.pipeline_mode == "original":
            # ORIGINAL MODE: Search parent nodes with expanded query
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
                    "section": target_section,
                    "payload": point.payload,
                })
        else:
            # ENRICH MODE: Search child nodes filtered by section
            response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit * 2,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="node_type", match=MatchValue(value="child")),
                        FieldCondition(key="section", match=MatchValue(value=target_section)),
                    ]
                ),
                with_payload=True,
            )

            # Dedupe by ASIN
            seen = set()
            results = []
            for point in response.points:
                asin = point.payload.get("parent_asin")
                if asin and asin not in seen:
                    seen.add(asin)
                    results.append({
                        "id": str(point.id),
                        "score": point.score,
                        "asin": asin,
                        "section": point.payload.get("section"),
                        "payload": point.payload,
                    })

        latency = time.perf_counter() - start
        return results[:limit], latency


# =============================================================================
# Strategy 5: Hierarchical Search (Children + Parent)
# =============================================================================

class HierarchicalSectionStrategy(SearchStrategy):
    """Search children and parents, merge with section priority.

    Mode-aware behavior:
    - Original mode: Search parent nodes with section-enhanced query
    - Enrich mode: Search children first (section priority), then parents
    """

    @property
    def name(self) -> str:
        return "section_hierarchical"

    @property
    def description(self) -> str:
        return "Search children + parents, merge with section priority"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")
        qdrant = self.clients.get_qdrant()

        if self.config.pipeline_mode == "original":
            # ORIGINAL MODE: Search parent nodes with enhanced query
            section_keywords = SECTION_KEYWORDS.get(target_section, [])
            enhanced_query = f"{query} {' '.join(section_keywords[:2])}"
            embedding = await self.clients.get_embedding(enhanced_query)

            parent_response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                ),
                with_payload=True,
            )

            results = []
            for point in parent_response.points:
                results.append({
                    "id": str(point.id),
                    "score": point.score,
                    "asin": point.payload.get("asin"),
                    "section": target_section,
                    "payload": point.payload,
                    "match_type": "parent_enhanced",
                })
        else:
            # ENRICH MODE: Search children first, then parents
            embedding = await self.clients.get_embedding(query)

            # Search target section children
            section_task = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit * 2,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="node_type", match=MatchValue(value="child")),
                        FieldCondition(key="section", match=MatchValue(value=target_section)),
                    ]
                ),
                with_payload=True,
            )

            # Search other children
            other_children_task = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="child"))],
                    must_not=[FieldCondition(key="section", match=MatchValue(value=target_section))],
                ),
                with_payload=True,
            )

            # Search parents
            parent_task = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                ),
                with_payload=True,
            )

            section_response = section_task
            other_response = other_children_task
            parent_response = parent_task

            # Aggregate with priority: target section > other children > parents
            asin_scores = {}
            asin_data = {}

            # Target section (highest priority)
            for rank, point in enumerate(section_response.points):
                asin = point.payload.get("parent_asin")
                if not asin:
                    continue
                score = point.score * 1.3  # Section boost
                if asin not in asin_scores or score > asin_scores[asin]:
                    asin_scores[asin] = score
                    asin_data[asin] = {
                        "payload": point.payload,
                        "id": str(point.id),
                        "section": point.payload.get("section"),
                        "match_type": "target_section",
                    }

            # Other children (medium priority)
            for point in other_response.points:
                asin = point.payload.get("parent_asin")
                if not asin:
                    continue
                score = point.score
                if asin not in asin_scores or score > asin_scores[asin]:
                    asin_scores[asin] = score
                    asin_data[asin] = {
                        "payload": point.payload,
                        "id": str(point.id),
                        "section": point.payload.get("section"),
                        "match_type": "other_section",
                    }

            # Parents (fallback)
            for point in parent_response.points:
                asin = point.payload.get("asin")
                if not asin:
                    continue
                score = point.score * 0.8  # Parent penalty for section queries
                if asin not in asin_scores:
                    asin_scores[asin] = score
                    asin_data[asin] = {
                        "payload": point.payload,
                        "id": str(point.id),
                        "section": None,
                        "match_type": "parent",
                    }

            sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for asin, score in sorted_asins[:limit]:
                results.append({
                    "id": asin_data[asin]["id"],
                    "score": score,
                    "asin": asin,
                    "section": asin_data[asin]["section"],
                    "payload": asin_data[asin]["payload"],
                    "match_type": asin_data[asin]["match_type"],
                })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 6: Combined Best Practices
# =============================================================================

class CombinedSectionStrategy(SearchStrategy):
    """Combine preprocessing + query expansion + hierarchical search."""

    SECTION_KEYWORDS = {
        "reviews": ["reviews", "customer feedback", "ratings"],
        "specs": ["specifications", "technical details", "dimensions"],
        "features": ["features", "capabilities", "functions"],
        "use_cases": ["use cases", "ideal for", "best for"],
        "description": ["description", "overview", "about"],
    }

    @property
    def name(self) -> str:
        return "section_combined"

    @property
    def description(self) -> str:
        return "Preprocessing + query expansion + hierarchical + fallback"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")

        # Preprocess and expand query
        clean_query = QueryPreprocessor.clean_query(query)
        section_terms = self.SECTION_KEYWORDS.get(target_section, [])
        expanded_query = f"{clean_query} {' '.join(section_terms[:2])}"

        embedding = await self.clients.get_embedding(expanded_query)
        qdrant = self.clients.get_qdrant()

        # Over-fetch from target section
        section_response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 3,
            query_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="child")),
                    FieldCondition(key="section", match=MatchValue(value=target_section)),
                ]
            ),
            with_payload=True,
        )

        # Aggregate by ASIN with quality signals
        asin_scores = {}
        asin_data = {}

        for point in section_response.points:
            asin = point.payload.get("parent_asin")
            if not asin:
                continue

            base_score = point.score

            # Quality boost
            stars = point.payload.get("stars", 0) or 0
            if stars >= 4:
                base_score *= 1.05

            if asin not in asin_scores or base_score > asin_scores[asin]:
                asin_scores[asin] = base_score
                asin_data[asin] = {
                    "payload": point.payload,
                    "id": str(point.id),
                    "section": target_section,
                }

        # Fallback to parent if not enough
        if len(asin_scores) < limit:
            parent_response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                ),
                with_payload=True,
            )

            for point in parent_response.points:
                asin = point.payload.get("asin")
                if not asin or asin in asin_scores:
                    continue
                asin_scores[asin] = point.score * 0.85
                asin_data[asin] = {
                    "payload": point.payload,
                    "id": str(point.id),
                    "section": None,
                }

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "section": asin_data[asin]["section"],
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 7: Multi-Signal Section Search (NEW - for better MRR)
# =============================================================================

class MultiSignalSectionStrategy(SearchStrategy):
    """Combine semantic + keyword search for section with aggressive ranking."""

    @property
    def name(self) -> str:
        return "section_multi_signal"

    @property
    def description(self) -> str:
        return "Combined semantic + keyword with section filter and quality ranking"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        from elasticsearch import AsyncElasticsearch

        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")

        # Clean query
        clean_query = QueryPreprocessor.clean_query(query)

        # Get embedding for semantic search
        embedding = await self.clients.get_embedding(clean_query)
        qdrant = self.clients.get_qdrant()

        # Semantic search on section
        sem_response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 5,
            query_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="child")),
                    FieldCondition(key="section", match=MatchValue(value=target_section)),
                ]
            ),
            with_payload=True,
        )

        # Keyword search on Elasticsearch
        es = await self.clients.get_elasticsearch()
        kw_response = await es.search(
            index=self.config.es_index,
            query={
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": clean_query,
                                "fields": ["title^10", "brand^5", "chunk_description", "chunk_features"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }
                    ]
                }
            },
            size=limit * 3,
        )

        # Extract keyword ASINs with scores
        keyword_scores = {}
        for rank, hit in enumerate(kw_response["hits"]["hits"]):
            asin = hit["_source"].get("asin")
            if asin:
                keyword_scores[asin] = hit["_score"] / (rank + 1)  # Decay by rank

        # Aggregate semantic results with keyword boost
        asin_scores = {}
        asin_data = {}

        for rank, point in enumerate(sem_response.points):
            asin = point.payload.get("parent_asin")
            if not asin:
                continue

            # Base semantic score with rank decay
            base_score = point.score * (1.0 - rank * 0.02)  # Small penalty for lower ranks

            # Keyword match boost
            if asin in keyword_scores:
                base_score *= (1 + min(keyword_scores[asin] / 10, 0.5))  # Up to 50% boost

            # Quality boost
            stars = point.payload.get("stars", 0) or 0
            if stars >= 4.5:
                base_score *= 1.1
            elif stars >= 4.0:
                base_score *= 1.05

            if asin not in asin_scores or base_score > asin_scores[asin]:
                asin_scores[asin] = base_score
                asin_data[asin] = {
                    "payload": point.payload,
                    "id": str(point.id),
                    "section": target_section,
                }

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "section": asin_data[asin]["section"],
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 8: Title-Focused Section Search (NEW - improve MRR for title matches)
# =============================================================================

class TitleFocusedSectionStrategy(SearchStrategy):
    """Prioritize title matches within section search for better MRR."""

    @property
    def name(self) -> str:
        return "section_title_focused"

    @property
    def description(self) -> str:
        return "Section search with strong title match priority"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time

        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")
        clean_query = QueryPreprocessor.clean_query(query)
        keywords = QueryPreprocessor.extract_keywords(clean_query)

        embedding = await self.clients.get_embedding(clean_query)
        qdrant = self.clients.get_qdrant()

        # Search section
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 5,
            query_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="child")),
                    FieldCondition(key="section", match=MatchValue(value=target_section)),
                ]
            ),
            with_payload=True,
        )

        # Score with title matching boost
        asin_scores = {}
        asin_data = {}

        for point in response.points:
            asin = point.payload.get("parent_asin")
            if not asin:
                continue

            base_score = point.score
            title = (point.payload.get("title") or "").lower()

            # Title match boost - significant boost if query keywords in title
            title_boost = 0.0
            for kw in keywords:
                if kw.lower() in title:
                    title_boost += 0.15  # 15% boost per keyword match

            title_boost = min(title_boost, 0.6)  # Cap at 60% boost
            final_score = base_score * (1 + title_boost)

            if asin not in asin_scores or final_score > asin_scores[asin]:
                asin_scores[asin] = final_score
                asin_data[asin] = {
                    "payload": point.payload,
                    "id": str(point.id),
                    "section": target_section,
                    "title_boost": title_boost,
                }

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "section": asin_data[asin]["section"],
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 9: Optimized Section Search (BEST for section queries)
# =============================================================================

class OptimizedSectionStrategy(SearchStrategy):
    """Optimized section search combining best practices.

    Based on experiment findings:
    - Keyword search excels for product identification
    - Semantic search works for conceptual queries
    - Section filter + keyword assist provides best results
    """

    SECTION_KEYWORDS = {
        "reviews": ["reviews", "customer feedback", "ratings", "opinions"],
        "specs": ["specifications", "technical", "dimensions", "compatibility"],
        "features": ["features", "capabilities", "functions", "technology"],
        "use_cases": ["use cases", "ideal for", "best for", "scenarios"],
        "description": ["description", "overview", "about", "product details"],
    }

    @property
    def name(self) -> str:
        return "section_optimized"

    @property
    def description(self) -> str:
        return "BEST: Section search with keyword assist + title boost + quality signals"

    def _extract_model_numbers(self, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        import re
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        return [m for m in matches if any(c.isdigit() for c in m)]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        import math
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")

        # Preprocess and expand query
        clean_query = QueryPreprocessor.clean_query(query)
        keywords = QueryPreprocessor.extract_keywords(clean_query)
        model_numbers = self._extract_model_numbers(query)
        section_terms = self.SECTION_KEYWORDS.get(target_section, [])

        # Light expansion with section terms
        expanded_query = f"{clean_query} {' '.join(section_terms[:2])}"

        # Get embedding
        embedding = await self.clients.get_embedding(expanded_query)
        qdrant = self.clients.get_qdrant()

        # Over-fetch from target section
        fetch_limit = limit * 5
        section_response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            query_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="child")),
                    FieldCondition(key="section", match=MatchValue(value=target_section)),
                ]
            ),
            with_payload=True,
        )

        # Get ES keyword matches for boosting
        es = await self.clients.get_elasticsearch()

        es_should = [
            {
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^10", "brand^5", "product_type^3"],
                    "type": "best_fields",
                }
            }
        ]

        # Add model number exact matches
        if model_numbers:
            for model in model_numbers:
                es_should.append({
                    "match": {
                        "title": {
                            "query": model,
                            "boost": 20.0,
                        }
                    }
                })

        es_response = await es.search(
            index=self.config.es_index,
            query={"bool": {"should": es_should, "minimum_should_match": 1}},
            size=limit * 3,
        )

        # Extract ES matches with rank scores
        es_ranks = {}
        for rank, hit in enumerate(es_response["hits"]["hits"]):
            asin = hit["_source"].get("asin")
            if asin:
                es_ranks[asin] = rank + 1

        # Aggregate semantic results with multiple signals
        asin_scores = {}
        asin_data = {}

        for point in section_response.points:
            asin = point.payload.get("parent_asin")
            if not asin:
                continue

            base_score = point.score

            # Strong keyword boost for ES matches
            if asin in es_ranks:
                kw_boost = min(0.5, 0.5 / es_ranks[asin])  # Up to 50% for rank 1
                base_score *= (1 + kw_boost)

            # Title keyword match boost
            title = (point.payload.get("title") or "").lower()
            title_boost = 0.0
            for kw in keywords:
                if kw.lower() in title:
                    title_boost += 0.1
            title_boost = min(title_boost, 0.4)
            base_score *= (1 + title_boost)

            # Model number in title boost
            if model_numbers:
                for model in model_numbers:
                    if model.lower() in title:
                        base_score *= 1.5
                        break

            # Quality signals
            stars = point.payload.get("stars", 0) or 0
            reviews = point.payload.get("reviews_count", 0) or 0
            is_bestseller = point.payload.get("is_best_seller", False)

            quality_boost = 0.0
            if stars >= 4:
                quality_boost += (stars - 3) * 0.02
            if reviews > 100:
                quality_boost += min(0.05, math.log10(reviews) * 0.012)
            if is_bestseller:
                quality_boost += 0.02

            final_score = base_score * (1 + quality_boost)

            if asin not in asin_scores or final_score > asin_scores[asin]:
                asin_scores[asin] = final_score
                asin_data[asin] = {
                    "payload": point.payload,
                    "id": str(point.id),
                    "section": target_section,
                }

        # Fallback to parent if not enough
        if len(asin_scores) < limit:
            parent_response = qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=embedding,
                limit=limit,
                query_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
                ),
                with_payload=True,
            )

            for point in parent_response.points:
                asin = point.payload.get("asin")
                if not asin or asin in asin_scores:
                    continue

                # Apply ES boost to parent fallback too
                score = point.score * 0.85  # Slight penalty for fallback
                if asin in es_ranks:
                    score *= (1 + min(0.3, 0.3 / es_ranks[asin]))

                asin_scores[asin] = score
                asin_data[asin] = {
                    "payload": point.payload,
                    "id": str(point.id),
                    "section": None,
                }

        sorted_asins = sorted(asin_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for asin, score in sorted_asins[:limit]:
            results.append({
                "id": asin_data[asin]["id"],
                "score": score,
                "asin": asin,
                "section": asin_data[asin]["section"],
                "payload": asin_data[asin]["payload"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 10: Section + BGE Reranker (NEW)
# =============================================================================

class SectionRerankerStrategy(SearchStrategy):
    """Section search with BGE reranker for improved ranking."""

    @property
    def name(self) -> str:
        return "section_reranker"

    @property
    def description(self) -> str:
        return "Section search + BGE reranker"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")

        embedding = await self.clients.get_embedding(query)
        qdrant = self.clients.get_qdrant()

        # Over-fetch for reranking
        fetch_limit = limit * 5

        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            query_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="child")),
                    FieldCondition(key="section", match=MatchValue(value=target_section)),
                ]
            ),
            with_payload=True,
        )

        # Prepare candidates
        candidates = []
        seen_asins = set()
        for point in response.points:
            asin = point.payload.get("parent_asin")
            if asin and asin not in seen_asins:
                seen_asins.add(asin)
                candidates.append({
                    "id": str(point.id),
                    "score": point.score,
                    "asin": asin,
                    "section": point.payload.get("section"),
                    "payload": point.payload,
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
# Strategy 10: Multi-Signal + Reranker (NEW - Best combination)
# =============================================================================

class MultiSignalRerankerStrategy(SearchStrategy):
    """Multi-signal section search with BGE reranker for best results."""

    @property
    def name(self) -> str:
        return "section_multi_signal_reranker"

    @property
    def description(self) -> str:
        return "Multi-signal (semantic+keyword) + BGE reranker"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time

        start = time.perf_counter()

        target_section = (filters or {}).get("section", "reviews")
        clean_query = QueryPreprocessor.clean_query(query)

        # Get embedding for semantic search
        embedding = await self.clients.get_embedding(clean_query)
        qdrant = self.clients.get_qdrant()

        # Semantic search on section (over-fetch)
        fetch_limit = limit * 5
        sem_response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=fetch_limit,
            query_filter=Filter(
                must=[
                    FieldCondition(key="node_type", match=MatchValue(value="child")),
                    FieldCondition(key="section", match=MatchValue(value=target_section)),
                ]
            ),
            with_payload=True,
        )

        # Keyword search on Elasticsearch
        es = await self.clients.get_elasticsearch()
        kw_response = await es.search(
            index=self.config.es_index,
            query={
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": clean_query,
                                "fields": ["title^10", "brand^5", "chunk_description", "chunk_features"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }
                    ]
                }
            },
            size=limit * 3,
        )

        # Extract keyword ASINs for boosting
        keyword_asins = set()
        for hit in kw_response["hits"]["hits"]:
            asin = hit["_source"].get("asin")
            if asin:
                keyword_asins.add(asin)

        # Aggregate semantic results with keyword boost
        asin_candidates = {}
        for point in sem_response.points:
            asin = point.payload.get("parent_asin")
            if not asin:
                continue

            base_score = point.score

            # Keyword match boost
            if asin in keyword_asins:
                base_score *= 1.3

            if asin not in asin_candidates or base_score > asin_candidates[asin]["score"]:
                asin_candidates[asin] = {
                    "id": str(point.id),
                    "score": base_score,
                    "asin": asin,
                    "section": target_section,
                    "payload": point.payload,
                }

        candidates = list(asin_candidates.values())

        # Rerank using BGE reranker
        reranked = await self.clients.rerank_with_scores(
            query=clean_query,
            candidates=candidates,
            top_k=limit,
        )

        latency = time.perf_counter() - start
        return reranked[:limit], latency


# =============================================================================
# Main
# =============================================================================

def _get_cli_defaults() -> dict:
    """Get CLI defaults from pipeline_config.yaml."""
    config = get_search_flows_config()
    data_dir = cfg.get_data_dir()
    product_count = cfg.get_count()
    pipeline_mode = cfg.get_mode()

    # Resolve eval_data path with {count} and {mode} placeholders
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
    """Run section search experiments.

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
    print("SECTION SEARCH EXPERIMENTS")
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
        ollama_url=ollama_url,
        pipeline_mode=pipeline_mode,
    )

    # Load section queries
    all_queries = load_evaluation_data(Path(eval_data), search_type="section")

    # Add section filter from target_section field
    for q in all_queries:
        if q.target_section:
            q.filters = q.filters or {}
            q.filters["section"] = q.target_section

    print(f"Loaded {len(all_queries)} section search queries")

    if not all_queries:
        print("No section queries found!")
        return

    clients = DatabaseClients(config)

    strategies = [
        BasicSectionStrategy(clients, config),
        SectionWithFallbackStrategy(clients, config),
        CrossSectionBoostStrategy(clients, config),
        SectionAwareQueryStrategy(clients, config),
        HierarchicalSectionStrategy(clients, config),
        CombinedSectionStrategy(clients, config),
        MultiSignalSectionStrategy(clients, config),
        TitleFocusedSectionStrategy(clients, config),
        OptimizedSectionStrategy(clients, config),  # BEST for section queries
        # Reranker strategies
        SectionRerankerStrategy(clients, config),
        MultiSignalRerankerStrategy(clients, config),
    ]

    async def run_all():
        results = []
        for strategy in strategies:
            print(f"\nRunning: {strategy.name}...")
            result = await run_experiment(
                strategy,
                all_queries,
                search_type="section",
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
