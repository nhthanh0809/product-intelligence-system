"""
Search tools for multi-agent system using optimized strategies from research-search-flows experiments.

These tools wrap the best-performing search strategies discovered through experimentation:
- KeywordPriorityHybridStrategy: MRR 0.9126 (best overall)
- SectionSearchStrategy: Best for targeted section queries (reviews, specs)
- OptimizedSemanticStrategy: Best for generic/conceptual queries

Usage:
    from src.tools.search_tools import SearchToolkit

    toolkit = SearchToolkit()
    await toolkit.initialize()

    results = await toolkit.hybrid_search("Sony WH-1000XM5 headphones", limit=10)
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import structlog
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from src.config import get_settings

logger = structlog.get_logger()


# =============================================================================
# Configuration
# =============================================================================

class SearchMode(str, Enum):
    """Pipeline mode affecting available fields."""
    ORIGINAL = "original"  # Limited fields (title, category)
    ENRICH = "enrich"      # Full fields (genAI, chunks, brand, etc.)


@dataclass
class SearchToolConfig:
    """Configuration for search tools."""
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "products"

    # Elasticsearch
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "products"

    # Ollama (embedding service)
    ollama_url: str = "http://localhost:8010"
    embedding_model: str = "bge-large"

    # Search defaults
    default_limit: int = 10
    fetch_multiplier: int = 5

    # Pipeline mode
    pipeline_mode: SearchMode = SearchMode.ENRICH

    @classmethod
    def from_settings(cls) -> "SearchToolConfig":
        """Create config from service settings."""
        settings = get_settings()
        return cls(
            qdrant_host=getattr(settings, "qdrant_host", "qdrant"),
            qdrant_port=getattr(settings, "qdrant_port", 6333),
            qdrant_collection=getattr(settings, "qdrant_collection", "products"),
            es_host=getattr(settings, "elasticsearch_host", "elasticsearch"),
            es_port=getattr(settings, "elasticsearch_port", 9200),
            es_index=getattr(settings, "elasticsearch_index", "products"),
            ollama_url=getattr(settings, "ollama_service_url", "http://ollama-service:8010"),
            pipeline_mode=SearchMode.ENRICH,
        )


# =============================================================================
# Query Analysis
# =============================================================================

class QueryType(str, Enum):
    """Detected query type for weight adjustment."""
    BRAND_MODEL = "brand_model"      # "Sony WH-1000XM5"
    MODEL_NUMBER = "model_number"    # "WH-1000XM5"
    SHORT_TITLE = "short_title"      # "wireless headphones"
    GENERIC = "generic"              # "good headphones for travel"
    SECTION = "section"              # "reviews of Sony headphones"


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""
    query_type: QueryType
    clean_query: str
    model_numbers: list[str] = field(default_factory=list)
    has_brand: bool = False
    detected_brand: str = ""
    detected_section: str | None = None
    keyword_weight: float = 0.65
    semantic_weight: float = 0.35

    @property
    def model_boost(self) -> float:
        """Boost factor for model number detection."""
        return 1.6 if self.model_numbers else 1.0

    @property
    def brand_boost(self) -> float:
        """Boost factor for brand detection."""
        return 1.3 if self.has_brand else 1.0


class QueryAnalyzer:
    """Analyze queries to determine optimal search strategy."""

    # Known brands for detection
    KNOWN_BRANDS = [
        "Sony", "Bose", "Apple", "Samsung", "LG", "Anker", "JBL", "Sennheiser",
        "Dell", "HP", "Lenovo", "Microsoft", "Google", "Amazon", "Kindle",
        "DeWalt", "Milwaukee", "Makita", "Bosch", "Ryobi", "Black+Decker",
        "Canon", "Nikon", "Fujifilm", "Dyson", "Shark", "Roomba", "iRobot",
        "KitchenAid", "Ninja", "Instant Pot", "Vitamix", "Philips", "Panasonic",
    ]

    # Section keywords
    SECTION_KEYWORDS = {
        "reviews": ["review", "reviews", "rating", "ratings", "what do people say", "opinions"],
        "specs": ["specs", "specifications", "technical", "dimensions", "weight", "size"],
        "features": ["features", "capabilities", "what can it do", "functions"],
        "description": ["description", "about", "overview", "what is"],
        "use_cases": ["use case", "best for", "good for", "suited for", "who should buy"],
    }

    @classmethod
    def analyze(cls, query: str) -> QueryAnalysis:
        """Analyze query to determine type and optimal weights."""
        clean_query = cls._clean_query(query)
        model_numbers = cls._extract_model_numbers(query)
        has_brand, detected_brand = cls._detect_brand(query)
        detected_section = cls._detect_section(query)
        words = clean_query.split()

        # Determine query type and weights based on experiments
        if detected_section:
            query_type = QueryType.SECTION
            kw_weight, sem_weight = 0.40, 0.60
        elif model_numbers:
            query_type = QueryType.MODEL_NUMBER
            kw_weight, sem_weight = 0.75, 0.25
        elif has_brand and len(words) <= 5:
            query_type = QueryType.BRAND_MODEL
            kw_weight, sem_weight = 0.70, 0.30
        elif len(words) <= 3:
            query_type = QueryType.SHORT_TITLE
            kw_weight, sem_weight = 0.65, 0.35
        elif len(words) <= 6:
            query_type = QueryType.SHORT_TITLE
            kw_weight, sem_weight = 0.60, 0.40
        else:
            query_type = QueryType.GENERIC
            kw_weight, sem_weight = 0.40, 0.60

        return QueryAnalysis(
            query_type=query_type,
            clean_query=clean_query,
            model_numbers=model_numbers,
            has_brand=has_brand,
            detected_brand=detected_brand,
            detected_section=detected_section,
            keyword_weight=kw_weight,
            semantic_weight=sem_weight,
        )

    @classmethod
    def _clean_query(cls, query: str) -> str:
        """Clean common artifacts from queries."""
        # Remove markdown artifacts
        query = re.sub(r'\*+', ' ', query)
        query = re.sub(r'\*\*([^*]+)\*\*', r'\1', query)
        # Remove common question prefixes
        prefixes = [
            r'^(find|search for|looking for|show me|get me|i want|i need)\s+',
            r'^(what is|what are|how to|where can i find)\s+',
        ]
        for prefix in prefixes:
            query = re.sub(prefix, '', query, flags=re.IGNORECASE)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.strip()

    @classmethod
    def _extract_model_numbers(cls, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        return [m for m in matches if any(c.isdigit() for c in m)]

    @classmethod
    def _detect_brand(cls, query: str) -> tuple[bool, str]:
        """Detect if query contains a known brand."""
        query_lower = query.lower()
        for brand in cls.KNOWN_BRANDS:
            if brand.lower() in query_lower:
                return True, brand
        # Check for capitalized words that might be brands
        words = query.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2 and word[0].isalpha():
                return True, word
        return False, ""

    @classmethod
    def _detect_section(cls, query: str) -> str | None:
        """Detect if query targets a specific section."""
        query_lower = query.lower()
        for section, keywords in cls.SECTION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return section
        return None


# =============================================================================
# Database Clients
# =============================================================================

class SearchClients:
    """Manage database connections for search tools."""

    def __init__(self, config: SearchToolConfig):
        self.config = config
        self._qdrant: QdrantClient | None = None
        self._elasticsearch: AsyncElasticsearch | None = None
        self._http_client: httpx.AsyncClient | None = None

    def get_qdrant(self) -> QdrantClient:
        """Get Qdrant client."""
        if self._qdrant is None:
            self._qdrant = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
            )
        return self._qdrant

    async def get_elasticsearch(self) -> AsyncElasticsearch:
        """Get Elasticsearch client."""
        if self._elasticsearch is None:
            self._elasticsearch = AsyncElasticsearch(
                [f"http://{self.config.es_host}:{self.config.es_port}"]
            )
        return self._elasticsearch

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for embedding service."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.ollama_url,
                timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
            )
        return self._http_client

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        client = await self.get_http_client()
        response = await client.post(
            "/embed/single",
            json={"text": text, "model": self.config.embedding_model},
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    async def close(self):
        """Close all connections."""
        if self._qdrant:
            self._qdrant.close()
            self._qdrant = None
        if self._elasticsearch:
            await self._elasticsearch.close()
            self._elasticsearch = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# =============================================================================
# Search Result Models
# =============================================================================

class SearchResult(BaseModel):
    """Individual search result."""
    asin: str
    title: str
    score: float
    source: str  # "semantic", "keyword", "hybrid"
    price: float | None = None
    stars: float | None = None
    brand: str | None = None
    category: str | None = None  # Product category
    img_url: str | None = None
    genAI_summary: str | None = None
    genAI_best_for: str | None = None
    section: str | None = None  # For section search results
    content_preview: str | None = None  # Section content preview


class SearchResponse(BaseModel):
    """Search response with metadata."""
    results: list[SearchResult]
    query: str
    query_type: str
    search_type: str
    latency_ms: float
    total_results: int
    weights: dict[str, float] | None = None


# =============================================================================
# Search Tools
# =============================================================================

class SemanticSearchTool:
    """
    Semantic search using Qdrant vector similarity.

    Best for:
    - Generic/conceptual queries ("good headphones for travel")
    - Use-case based searches ("camera for beginners")

    Performance: MRR ~0.65 on generic queries
    """

    def __init__(self, clients: SearchClients, config: SearchToolConfig):
        self.clients = clients
        self.config = config

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        parent_only: bool = True,
    ) -> SearchResponse:
        """Execute semantic search."""
        start = time.perf_counter()

        # Get embedding
        embedding = await self.clients.get_embedding(query)

        # Build Qdrant filter
        qdrant_filter = self._build_filter(filters, parent_only)

        # Search
        qdrant = self.clients.get_qdrant()
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Format results
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(SearchResult(
                asin=payload.get("asin") or payload.get("parent_asin", ""),
                title=payload.get("title", ""),
                score=point.score,
                source="semantic",
                price=payload.get("price"),
                stars=payload.get("stars"),
                brand=payload.get("brand"),
                category=payload.get("category_name") or payload.get("category_level1"),
                img_url=payload.get("img_url") or payload.get("imgUrl"),  # Handle both naming conventions
                genAI_summary=payload.get("genAI_summary"),
                genAI_best_for=payload.get("genAI_best_for"),
            ))

        latency = (time.perf_counter() - start) * 1000

        return SearchResponse(
            results=results,
            query=query,
            query_type="generic",
            search_type="semantic",
            latency_ms=latency,
            total_results=len(results),
        )

    def _build_filter(
        self,
        filters: dict[str, Any] | None,
        parent_only: bool,
    ) -> Filter | None:
        """Build Qdrant filter."""
        conditions = []

        if parent_only:
            conditions.append(
                FieldCondition(key="node_type", match=MatchValue(value="parent"))
            )

        if filters:
            # Skip category filter - exact match is too strict
            # Let semantic search handle category matching through embeddings
            if "brand" in filters:
                conditions.append(
                    FieldCondition(
                        key="brand",
                        match=MatchValue(value=filters["brand"]),
                    )
                )
            if "price_min" in filters or "price_max" in filters:
                range_kwargs = {}
                if "price_min" in filters:
                    range_kwargs["gte"] = filters["price_min"]
                if "price_max" in filters:
                    range_kwargs["lte"] = filters["price_max"]
                conditions.append(
                    FieldCondition(key="price", range=Range(**range_kwargs))
                )
            if "min_rating" in filters:
                conditions.append(
                    FieldCondition(
                        key="stars",
                        range=Range(gte=filters["min_rating"]),
                    )
                )

        return Filter(must=conditions) if conditions else None


class KeywordSearchTool:
    """
    Keyword search using Elasticsearch.

    Best for:
    - Brand + model queries ("Sony WH-1000XM5")
    - Model number searches ("WH-1000XM5")
    - Exact product name searches

    Performance: R@1 87.7% on brand+model queries
    """

    def __init__(self, clients: SearchClients, config: SearchToolConfig):
        self.clients = clients
        self.config = config

    def _get_search_fields(self) -> list[str]:
        """Get search fields based on pipeline mode."""
        if self.config.pipeline_mode == SearchMode.ORIGINAL:
            return [
                "title^10",
                "title.autocomplete^5",
                "category_name^2",
            ]
        else:
            return [
                "title^10",
                "title.autocomplete^5",
                "short_title^8",
                "brand^5",
                "product_type^4",
                "product_type_keywords^3",
                "chunk_description^1",
                "chunk_features^1",
                "genAI_summary^2",
            ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        model_numbers: list[str] | None = None,
    ) -> SearchResponse:
        """Execute keyword search with optional model number boosting."""
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()
        analysis = QueryAnalyzer.analyze(query)

        # Build query with model number boosting
        if model_numbers or analysis.model_numbers:
            models = model_numbers or analysis.model_numbers
            # Use should clause to boost model numbers
            es_query = {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": self._get_search_fields(),
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        },
                        *[
                            {
                                "multi_match": {
                                    "query": model,
                                    "fields": ["title^10", "short_title^8"],
                                    "boost": 1.6,  # Model number boost
                                }
                            }
                            for model in models
                        ],
                    ],
                    "minimum_should_match": 1,
                }
            }
        else:
            es_query = {
                "multi_match": {
                    "query": query,
                    "fields": self._get_search_fields(),
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            }

        # Add filters
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
            if filter_clauses:
                es_query = {
                    "bool": {
                        "must": [es_query],
                        "filter": filter_clauses,
                    }
                }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        # Format results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(SearchResult(
                asin=source.get("asin", ""),
                title=source.get("title", ""),
                score=hit["_score"],
                source="keyword",
                price=source.get("price"),
                stars=source.get("stars"),
                brand=source.get("brand"),
                category=source.get("category_name") or source.get("category_level1"),
                img_url=source.get("img_url") or source.get("imgUrl"),  # Handle both naming conventions
                genAI_summary=source.get("genAI_summary"),
                genAI_best_for=source.get("genAI_best_for"),
            ))

        latency = (time.perf_counter() - start) * 1000

        return SearchResponse(
            results=results,
            query=query,
            query_type=analysis.query_type.value,
            search_type="keyword",
            latency_ms=latency,
            total_results=len(results),
        )

    def _build_filter_clauses(self, filters: dict[str, Any]) -> list[dict]:
        """Build Elasticsearch filter clauses."""
        clauses = []
        # Skip category and brand filters - data doesn't have these fields consistently
        # Let the text search handle category/brand matching through title search
        if "price_min" in filters or "price_max" in filters:
            range_query = {}
            if "price_min" in filters:
                range_query["gte"] = filters["price_min"]
            if "price_max" in filters:
                range_query["lte"] = filters["price_max"]
            clauses.append({"range": {"price": range_query}})
        if "min_rating" in filters:
            clauses.append({"range": {"stars": {"gte": filters["min_rating"]}}})
        return clauses


class HybridSearchTool:
    """
    Keyword-priority hybrid search using weighted RRF fusion.

    This is the best-performing strategy (MRR 0.9126) based on experiments.
    Dynamically adjusts keyword/semantic weights based on query analysis.

    Best for:
    - General product search
    - All query types (adapts weights automatically)
    """

    def __init__(self, clients: SearchClients, config: SearchToolConfig):
        self.clients = clients
        self.config = config
        self.semantic_tool = SemanticSearchTool(clients, config)
        self.keyword_tool = KeywordSearchTool(clients, config)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> SearchResponse:
        """
        Execute hybrid search with keyword priority.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters
            semantic_weight: Override semantic weight (auto-detected if None)
            keyword_weight: Override keyword weight (auto-detected if None)
        """
        start = time.perf_counter()

        # Analyze query to determine weights
        analysis = QueryAnalyzer.analyze(query)

        # Use provided weights or auto-detected
        sem_weight = semantic_weight if semantic_weight is not None else analysis.semantic_weight
        kw_weight = keyword_weight if keyword_weight is not None else analysis.keyword_weight

        fetch_limit = limit * self.config.fetch_multiplier

        # Run both searches in parallel
        semantic_task = self.semantic_tool.search(
            analysis.clean_query, fetch_limit, filters
        )
        keyword_task = self.keyword_tool.search(
            query, fetch_limit, filters, analysis.model_numbers
        )

        semantic_response, keyword_response = await asyncio.gather(
            semantic_task, keyword_task, return_exceptions=True
        )

        # Collect results for fusion
        result_lists = []
        weights = []

        if isinstance(semantic_response, SearchResponse):
            result_lists.append(semantic_response.results)
            weights.append(sem_weight)
        else:
            logger.warning("semantic_search_failed", error=str(semantic_response))

        if isinstance(keyword_response, SearchResponse):
            result_lists.append(keyword_response.results)
            weights.append(kw_weight)
        else:
            logger.warning("keyword_search_failed", error=str(keyword_response))

        if not result_lists:
            return SearchResponse(
                results=[],
                query=query,
                query_type=analysis.query_type.value,
                search_type="hybrid",
                latency_ms=(time.perf_counter() - start) * 1000,
                total_results=0,
            )

        # Apply weighted RRF fusion
        fused_results = self._weighted_rrf_fusion(
            result_lists,
            weights,
            k=40,  # Moderate k for better top-result emphasis
            analysis=analysis,
        )

        latency = (time.perf_counter() - start) * 1000

        return SearchResponse(
            results=fused_results[:limit],
            query=query,
            query_type=analysis.query_type.value,
            search_type="hybrid",
            latency_ms=latency,
            total_results=len(fused_results[:limit]),
            weights={"semantic": sem_weight, "keyword": kw_weight},
        )

    def _weighted_rrf_fusion(
        self,
        result_lists: list[list[SearchResult]],
        weights: list[float],
        k: int = 40,
        analysis: QueryAnalysis | None = None,
    ) -> list[SearchResult]:
        """Apply weighted Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        docs: dict[str, SearchResult] = {}

        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results):
                asin = result.asin
                if not asin:
                    continue

                # Calculate weighted RRF score
                rrf_score = weight * (1.0 / (k + rank + 1))

                # Apply boosts based on query analysis
                if analysis:
                    # Model number boost
                    if analysis.model_numbers and result.title:
                        for model in analysis.model_numbers:
                            if model.lower() in result.title.lower():
                                rrf_score *= analysis.model_boost
                                break

                    # Brand boost
                    if analysis.detected_brand and result.brand:
                        if analysis.detected_brand.lower() == result.brand.lower():
                            rrf_score *= analysis.brand_boost

                if asin not in scores:
                    scores[asin] = 0.0
                    docs[asin] = result

                scores[asin] += rrf_score

        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        fused = []
        for asin, score in sorted_items:
            result = docs[asin].model_copy()
            result.score = score
            result.source = "hybrid"
            fused.append(result)

        return fused


class SectionSearchTool:
    """
    Section-targeted search for specific product aspects.

    Best for:
    - Review analysis ("what do people say about battery life")
    - Specs comparison ("technical specifications of laptop")
    - Feature queries ("features of this camera")

    Uses child nodes in Qdrant for section-specific retrieval.
    """

    VALID_SECTIONS = ["description", "features", "specs", "reviews", "use_cases"]

    def __init__(self, clients: SearchClients, config: SearchToolConfig):
        self.clients = clients
        self.config = config

    async def search(
        self,
        query: str,
        section: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        include_keyword_boost: bool = True,
    ) -> SearchResponse:
        """
        Search within a specific section.

        Args:
            query: Search query
            section: Section to search (reviews, specs, features, description, use_cases)
            limit: Maximum results
            filters: Optional filters
            include_keyword_boost: Also run keyword search for better recall
        """
        start = time.perf_counter()

        if section not in self.VALID_SECTIONS:
            raise ValueError(f"Invalid section. Must be one of: {self.VALID_SECTIONS}")

        # Check pipeline mode - original mode has no child nodes
        if self.config.pipeline_mode == SearchMode.ORIGINAL:
            # Fall back to parent search
            logger.info("section_search_fallback", reason="original_mode", section=section)
            semantic_tool = SemanticSearchTool(self.clients, self.config)
            return await semantic_tool.search(query, limit, filters)

        # Get embedding
        embedding = await self.clients.get_embedding(query)

        # Build filter for section
        qdrant = self.clients.get_qdrant()
        conditions = [
            FieldCondition(key="node_type", match=MatchValue(value="child")),
            FieldCondition(key="section", match=MatchValue(value=section)),
        ]

        if filters:
            if "category_level1" in filters:
                conditions.append(
                    FieldCondition(
                        key="category_level1",
                        match=MatchValue(value=filters["category_level1"]),
                    )
                )
            if "brand" in filters:
                conditions.append(
                    FieldCondition(
                        key="brand",
                        match=MatchValue(value=filters["brand"]),
                    )
                )

        qdrant_filter = Filter(must=conditions)

        # Search child nodes
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=embedding,
            limit=limit * 2,  # Fetch more for deduplication
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Aggregate by parent ASIN (keep best per parent)
        seen_asins: set[str] = set()
        results = []

        for point in response.points:
            payload = point.payload or {}
            parent_asin = payload.get("parent_asin") or payload.get("asin", "")

            if parent_asin in seen_asins:
                continue
            seen_asins.add(parent_asin)

            results.append(SearchResult(
                asin=parent_asin,
                title=payload.get("title", ""),
                score=point.score,
                source="section",
                price=payload.get("price"),
                stars=payload.get("stars"),
                brand=payload.get("brand"),
                category=payload.get("category_name") or payload.get("category_level1"),
                img_url=payload.get("img_url") or payload.get("imgUrl"),  # Handle both naming conventions
                section=section,
                content_preview=(payload.get("content_preview") or "")[:200],
            ))

            if len(results) >= limit:
                break

        latency = (time.perf_counter() - start) * 1000

        return SearchResponse(
            results=results,
            query=query,
            query_type="section",
            search_type=f"section:{section}",
            latency_ms=latency,
            total_results=len(results),
        )


class SimilarProductsTool:
    """
    Find similar products using vector similarity.

    Best for:
    - "Show me similar products"
    - "Alternatives to X"
    - Recommendation scenarios
    """

    def __init__(self, clients: SearchClients, config: SearchToolConfig):
        self.clients = clients
        self.config = config

    async def search(
        self,
        asin: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        exclude_self: bool = True,
    ) -> SearchResponse:
        """
        Find products similar to the given ASIN.

        Args:
            asin: Source product ASIN
            limit: Maximum results
            filters: Optional filters
            exclude_self: Exclude the source product from results
        """
        start = time.perf_counter()

        qdrant = self.clients.get_qdrant()

        # First, find the source product to get its vector
        source_filter = Filter(
            must=[
                FieldCondition(key="asin", match=MatchValue(value=asin)),
                FieldCondition(key="node_type", match=MatchValue(value="parent")),
            ]
        )

        source_results = qdrant.scroll(
            collection_name=self.config.qdrant_collection,
            scroll_filter=source_filter,
            limit=1,
            with_vectors=True,
            with_payload=True,
        )

        if not source_results[0]:
            return SearchResponse(
                results=[],
                query=f"similar_to:{asin}",
                query_type="similar",
                search_type="similar",
                latency_ms=(time.perf_counter() - start) * 1000,
                total_results=0,
            )

        source_point = source_results[0][0]
        source_vector = source_point.vector
        source_payload = source_point.payload or {}

        # Build filter for similar search
        conditions = [
            FieldCondition(key="node_type", match=MatchValue(value="parent")),
        ]

        if filters:
            if "category_level1" in filters:
                conditions.append(
                    FieldCondition(
                        key="category_level1",
                        match=MatchValue(value=filters["category_level1"]),
                    )
                )
            if "brand" in filters:
                conditions.append(
                    FieldCondition(
                        key="brand",
                        match=MatchValue(value=filters["brand"]),
                    )
                )
            if "price_min" in filters or "price_max" in filters:
                range_kwargs = {}
                if "price_min" in filters:
                    range_kwargs["gte"] = filters["price_min"]
                if "price_max" in filters:
                    range_kwargs["lte"] = filters["price_max"]
                conditions.append(
                    FieldCondition(key="price", range=Range(**range_kwargs))
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        # Search for similar products
        search_limit = limit + 1 if exclude_self else limit
        response = qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=source_vector,
            limit=search_limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Format results
        results = []
        for point in response.points:
            payload = point.payload or {}
            result_asin = payload.get("asin", "")

            # Skip source product if requested
            if exclude_self and result_asin == asin:
                continue

            results.append(SearchResult(
                asin=result_asin,
                title=payload.get("title", ""),
                score=point.score,
                source="similar",
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

        latency = (time.perf_counter() - start) * 1000

        return SearchResponse(
            results=results,
            query=f"similar_to:{asin}",
            query_type="similar",
            search_type="similar",
            latency_ms=latency,
            total_results=len(results),
        )


# =============================================================================
# Search Toolkit (Main Interface)
# =============================================================================

class SearchToolkit:
    """
    Main interface for search tools.

    Provides access to all search strategies optimized from experiments.

    Usage:
        toolkit = SearchToolkit()
        await toolkit.initialize()

        # Use specific tools
        results = await toolkit.hybrid_search("Sony headphones")
        results = await toolkit.section_search("battery reviews", "reviews")

        # Auto-select best tool
        results = await toolkit.search("Sony WH-1000XM5")  # Uses hybrid

        await toolkit.close()
    """

    def __init__(self, config: SearchToolConfig | None = None):
        self.config = config or SearchToolConfig.from_settings()
        self.clients = SearchClients(self.config)

        # Initialize tools
        self.semantic = SemanticSearchTool(self.clients, self.config)
        self.keyword = KeywordSearchTool(self.clients, self.config)
        self.hybrid = HybridSearchTool(self.clients, self.config)
        self.section = SectionSearchTool(self.clients, self.config)
        self.similar = SimilarProductsTool(self.clients, self.config)

    async def initialize(self):
        """Initialize connections (called lazily, but can be called explicitly)."""
        # Connections are initialized lazily, but we can verify they work
        try:
            _ = self.clients.get_qdrant()
            _ = await self.clients.get_elasticsearch()
            logger.info("search_toolkit_initialized")
        except Exception as e:
            logger.error("search_toolkit_init_failed", error=str(e))
            raise

    async def close(self):
        """Close all connections."""
        await self.clients.close()

    # Convenience methods for direct tool access

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Semantic search (best for generic queries)."""
        return await self.semantic.search(query, limit, filters)

    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Keyword search (best for brand+model queries)."""
        return await self.keyword.search(query, limit, filters)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Hybrid search (best overall, auto-adjusts weights)."""
        return await self.hybrid.search(query, limit, filters)

    async def section_search(
        self,
        query: str,
        section: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Section-targeted search (reviews, specs, features)."""
        return await self.section.search(query, section, limit, filters)

    async def find_similar(
        self,
        asin: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Find similar products."""
        return await self.similar.search(asin, limit, filters)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """
        Auto-select best search strategy based on query analysis.

        This is the recommended entry point for general search.
        """
        analysis = QueryAnalyzer.analyze(query)

        # Route to appropriate tool based on query type
        if analysis.detected_section:
            return await self.section.search(
                query, analysis.detected_section, limit, filters
            )
        else:
            # Default to hybrid (best overall performance)
            return await self.hybrid.search(query, limit, filters)
