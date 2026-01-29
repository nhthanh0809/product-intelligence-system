#!/usr/bin/env python3
"""
Base utilities for search flow experiments.

Provides:
- Metrics calculation (Precision, Recall, MRR, NDCG, Hit Rate, Coverage)
- Direct database clients (Qdrant, Elasticsearch, Ollama)
- Evaluation data loading
- Results comparison and reporting
- Mode-aware search configuration (original vs enrich)
"""

import asyncio
import json
import math
import re
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import structlog
from elasticsearch import AsyncElasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Add parent directory for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg

def get_pipeline_mode() -> str:
    """Get the pipeline mode from config."""
    return cfg.get_mode()


def get_search_flows_config() -> dict:
    """Get search_flows configuration from pipeline_config.yaml."""
    return cfg.get_script("search_flows") or {}


logger = structlog.get_logger()


# =============================================================================
# Configuration
# =============================================================================

def _get_config_default(key: str, fallback: any) -> any:
    """Get a config value from search_flows section with fallback."""
    config = get_search_flows_config()
    value = config.get(key)
    return value if value is not None else fallback


@dataclass
class SearchConfig:
    """Configuration for search experiments.

    Values are loaded from pipeline_config.yaml under scripts.search_flows.
    CLI arguments can override these values.
    """
    # Qdrant
    qdrant_host: str = None
    qdrant_port: int = None
    qdrant_collection: str = None

    # Elasticsearch
    es_host: str = None
    es_port: int = None
    es_index: str = None

    # Ollama (embedding service)
    ollama_url: str = None
    embedding_model: str = None

    # Reranker model
    reranker_model: str = None

    # Search defaults
    default_limit: int = None
    fetch_multiplier: int = None

    # Pipeline mode (original vs enrich)
    # - original: No genAI_* fields available
    # - enrich: All genAI_* fields available for search
    pipeline_mode: str = None

    def __post_init__(self):
        """Load values from config if not explicitly set."""
        # Load from pipeline_config.yaml with hardcoded fallbacks
        if self.qdrant_host is None:
            self.qdrant_host = _get_config_default("qdrant_host", "localhost")
        if self.qdrant_port is None:
            self.qdrant_port = _get_config_default("qdrant_port", 6333)
        if self.qdrant_collection is None:
            self.qdrant_collection = _get_config_default("qdrant_collection", "products")

        if self.es_host is None:
            self.es_host = _get_config_default("elasticsearch_host", "localhost")
        if self.es_port is None:
            self.es_port = _get_config_default("elasticsearch_port", 9200)
        if self.es_index is None:
            self.es_index = _get_config_default("elasticsearch_index", "products")

        if self.ollama_url is None:
            self.ollama_url = _get_config_default("ollama_url", "http://localhost:8010")
        if self.embedding_model is None:
            self.embedding_model = _get_config_default("embedding_model", "bge-large")

        if self.reranker_model is None:
            self.reranker_model = _get_config_default("reranker_model", "qllama/bge-reranker-large")

        if self.default_limit is None:
            self.default_limit = _get_config_default("default_limit", 10)
        if self.fetch_multiplier is None:
            self.fetch_multiplier = _get_config_default("fetch_multiplier", 3)

        if self.pipeline_mode is None:
            try:
                self.pipeline_mode = get_pipeline_mode()
            except Exception:
                self.pipeline_mode = "enrich"

    @property
    def has_genai_fields(self) -> bool:
        """Check if genAI fields are expected based on mode."""
        return self.pipeline_mode == "enrich"

    @property
    def is_original_mode(self) -> bool:
        """Check if running in original mode (limited fields)."""
        return self.pipeline_mode == "original"

    @property
    def has_brand_fields(self) -> bool:
        """Check if brand/product_type fields are available."""
        return self.pipeline_mode == "enrich"

    @property
    def has_chunk_fields(self) -> bool:
        """Check if chunk_* fields (description, features, specs) are available."""
        return self.pipeline_mode == "enrich"

    def get_keyword_fields(self) -> list[str]:
        """Get flat list of all keyword search fields for current mode."""
        return get_all_keyword_fields(self.pipeline_mode)


# =============================================================================
# Mode-Specific Field Configuration
# =============================================================================

def get_keyword_search_fields(mode: str) -> dict[str, list[str]]:
    """Get keyword search fields based on pipeline mode.

    Args:
        mode: Pipeline mode ('original' or 'enrich')

    Returns:
        Dictionary with field categories and their boost values

    Note:
        Original mode only has: title, category_name (limited fields from CSV)
        Enrich mode has: title, short_title, brand, product_type, chunks, genAI fields
    """
    if mode == "original":
        # Original mode: only fields that exist in the DB
        # Original mode creates parent nodes only with limited CSV fields
        return {
            "high_priority": [
                "title^10",
                "title.autocomplete^5",
            ],
            "medium_priority": [
                "category_name^2",
            ],
            "low_priority": [],
            "genai_fields": [],
        }
    else:
        # Enrich mode: full field set with scraped and LLM-generated content
        return {
            "high_priority": [
                "title^10",
                "title.autocomplete^5",
                "short_title^8",
                "brand^5",
                "product_type^4",
            ],
            "medium_priority": [
                "category_name^2",
                "chunk_description^1",
                "chunk_features^1",
                "chunk_specs^1",
            ],
            "low_priority": [
                "product_type_keywords^3",
            ],
            "genai_fields": [
                "genAI_summary^3",
                "genAI_primary_function^2",
                "genAI_best_for^2",
            ],
        }


def get_all_keyword_fields(mode: str) -> list[str]:
    """Get flat list of all keyword search fields for a mode."""
    fields_config = get_keyword_search_fields(mode)
    all_fields = []
    for category_fields in fields_config.values():
        all_fields.extend(category_fields)
    return all_fields


# =============================================================================
# Metrics Calculator
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    hit_rate: float = 0.0
    coverage: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    # Detailed breakdown
    total_queries: int = 0
    successful_queries: int = 0
    queries_with_hits: int = 0
    unique_products_returned: int = 0
    total_products_in_corpus: int = 0


class MetricsCalculator:
    """Calculate retrieval evaluation metrics."""

    @staticmethod
    def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Precision@K: fraction of top-K results that are relevant."""
        if k == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        hits = sum(1 for item in retrieved_k if item in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Recall@K: fraction of relevant items found in top-K."""
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        hits = len(retrieved_k & relevant)
        return hits / len(relevant)

    @staticmethod
    def mrr(retrieved: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant result."""
        for i, item in enumerate(retrieved):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def dcg_at_k(relevances: list[float], k: int) -> float:
        """Discounted Cumulative Gain at K."""
        relevances = relevances[:k]
        if not relevances:
            return 0.0
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    @staticmethod
    def ndcg_at_k(
        retrieved: list[str],
        relevance_scores: dict[str, float],
        k: int,
    ) -> float:
        """Normalized DCG at K."""
        # Get relevances for retrieved items
        retrieved_relevances = [
            relevance_scores.get(item, 0) for item in retrieved[:k]
        ]

        # Calculate DCG
        dcg = MetricsCalculator.dcg_at_k(retrieved_relevances, k)

        # Calculate ideal DCG
        # IMPORTANT: Pad ideal relevances if fewer relevant items than k
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
        # Pad with zeros if needed
        while len(ideal_relevances) < k:
            ideal_relevances.append(0)
        idcg = MetricsCalculator.dcg_at_k(ideal_relevances[:k], k)

        if idcg == 0:
            return 0.0
        return min(dcg / idcg, 1.0)  # Cap at 1.0 to handle edge cases

    @staticmethod
    def hit_rate(retrieved: list[str], relevant: set[str]) -> float:
        """Hit rate: 1 if any relevant item found, else 0."""
        return 1.0 if any(item in relevant for item in retrieved) else 0.0

    @staticmethod
    def aggregate_metrics(
        all_metrics: list[dict[str, float]],
        latencies: list[float],
        unique_products: set[str],
        total_corpus_size: int,
    ) -> RetrievalMetrics:
        """Aggregate metrics from multiple queries."""
        if not all_metrics:
            return RetrievalMetrics()

        def safe_mean(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        def safe_percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        return RetrievalMetrics(
            precision_at_5=safe_mean([m["precision_at_5"] for m in all_metrics]),
            precision_at_10=safe_mean([m["precision_at_10"] for m in all_metrics]),
            recall_at_5=safe_mean([m["recall_at_5"] for m in all_metrics]),
            recall_at_10=safe_mean([m["recall_at_10"] for m in all_metrics]),
            mrr=safe_mean([m["mrr"] for m in all_metrics]),
            ndcg_at_10=safe_mean([m["ndcg_at_10"] for m in all_metrics]),
            hit_rate=safe_mean([m["hit_rate"] for m in all_metrics]),
            coverage=len(unique_products) / total_corpus_size if total_corpus_size > 0 else 0.0,
            latency_p50_ms=safe_percentile(latencies, 50) * 1000,
            latency_p95_ms=safe_percentile(latencies, 95) * 1000,
            total_queries=len(all_metrics),
            successful_queries=len(all_metrics),
            queries_with_hits=sum(1 for m in all_metrics if m["hit_rate"] > 0),
            unique_products_returned=len(unique_products),
            total_products_in_corpus=total_corpus_size,
        )


# =============================================================================
# Query Preprocessing
# =============================================================================

class QueryPreprocessor:
    """Preprocess queries for better search performance."""

    @staticmethod
    def clean_query(query: str) -> str:
        """Clean common artifacts from LLM-generated queries."""
        # Remove **** prefix/suffix patterns (LLM generation artifact)
        query = re.sub(r'^\*+\s*', '', query)
        query = re.sub(r'\s*\*+$', '', query)
        # Remove asterisks anywhere in the query
        query = re.sub(r'\*+', ' ', query)
        # Remove markdown artifacts
        query = re.sub(r'\*\*([^*]+)\*\*', r'\1', query)
        # Remove bullet points and list markers
        query = re.sub(r'^[-•·]\s*', '', query)
        # Remove numbering at start
        query = re.sub(r'^\d+[.)]\s*', '', query)
        # Remove quotes
        query = query.strip('"\'')
        # Remove common question prefixes that don't help search
        prefixes_to_remove = [
            r'^(find|search for|looking for|show me|get me|i want|i need)\s+',
            r'^(what is|what are|how to|where can i find)\s+',
        ]
        for prefix in prefixes_to_remove:
            query = re.sub(prefix, '', query, flags=re.IGNORECASE)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.strip()

    @staticmethod
    def extract_keywords(query: str, min_length: int = 3) -> list[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'although', 'though',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'am', 'about',
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        keywords = [w for w in words if len(w) >= min_length and w not in stop_words]
        return keywords

    @staticmethod
    def expand_query(query: str) -> list[str]:
        """Generate query variations for multi-query retrieval."""
        queries = [query]

        # Add keyword-only version
        keywords = QueryPreprocessor.extract_keywords(query)
        if keywords:
            queries.append(' '.join(keywords))

        # Add first N words (for long queries)
        words = query.split()
        if len(words) > 10:
            queries.append(' '.join(words[:10]))

        return queries

    @staticmethod
    def format_for_semantic_search(query: str, brand: str = "", category: str = "") -> str:
        """Format query to match document embedding format.

        IMPORTANT: Documents were embedded with format:
            Product: {title}
            Brand: {brand}
            Category: {category}
            ...

        To get high similarity, queries should match this format.
        Since we often don't have brand/category at query time, we format as:
            Product: {query}

        This improves semantic similarity from ~0.55 to ~0.65+

        For even better results, if brand is known, include it:
            Product: {query}
            Brand: {brand}
        """
        parts = [f"Product: {query}"]
        if brand:
            parts.append(f"Brand: {brand}")
        if category:
            parts.append(f"Category: {category}")
        return "\n".join(parts)

    @staticmethod
    def extract_brand_from_query(query: str, known_brands: list[str] | None = None) -> tuple[str, str]:
        """Extract brand name from query if present.

        Returns:
            Tuple of (brand_if_found, query_without_brand)
        """
        if not known_brands:
            # Common tool/equipment brands to check
            known_brands = [
                "Forney", "Lincoln", "Miller", "Hobart", "ESAB", "Hypertherm",
                "DeWalt", "Milwaukee", "Makita", "Bosch", "Ryobi", "Craftsman",
                "Stanley", "Klein", "Fluke", "Snap-on", "3M", "Honeywell",
                "Flameweld", "Lotos", "Everlast", "AHP", "Primeweld",
                "VEVOR", "YESWELDER", "Mophorn", "S SATC", "KAKA", "KEMIMOTO",
            ]

        query_lower = query.lower()
        for brand in known_brands:
            brand_lower = brand.lower()
            if brand_lower in query_lower:
                # Found brand, return it and query with brand removed
                pattern = re.compile(re.escape(brand), re.IGNORECASE)
                query_no_brand = pattern.sub('', query).strip()
                query_no_brand = ' '.join(query_no_brand.split())  # Normalize spaces
                return brand, query_no_brand

        return "", query

    @staticmethod
    def format_query_for_mode(query: str, pipeline_mode: str, brand: str = "", category: str = "") -> str:
        """Format query based on pipeline mode for optimal matching.

        Args:
            query: Original query text
            pipeline_mode: 'original' or 'enrich'
            brand: Optional brand name
            category: Optional category name

        Returns:
            Formatted query optimized for the pipeline mode's embedding format

        Mode-specific behavior:
        - Original mode: Documents embedded as plain text (title + category)
                        Query should be clean, focused on product keywords
        - Enrich mode: Documents embedded with structure (Product:, Brand:, etc.)
                       Query should match this format for better similarity
        """
        if pipeline_mode == "original":
            # ORIGINAL MODE: Plain text matching
            # Clean query, extract keywords, focus on product terms
            clean_query = QueryPreprocessor.clean_query(query)
            keywords = QueryPreprocessor.extract_keywords(clean_query)

            # Build simple query focused on product name/keywords
            if brand and brand.lower() not in clean_query.lower():
                clean_query = f"{brand} {clean_query}"
            if category and len(clean_query.split()) < 5:
                clean_query = f"{clean_query} {category}"

            return clean_query
        else:
            # ENRICH MODE: Structured format matching
            return QueryPreprocessor.format_for_semantic_search(query, brand, category)


# =============================================================================
# Database Clients
# =============================================================================

class DatabaseClients:
    """Manage database connections for experiments."""

    def __init__(self, config: SearchConfig):
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
        """Get embedding for text using configured model."""
        client = await self.get_http_client()
        response = await client.post(
            "/embed/single",
            json={"text": text, "model": self.config.embedding_model},
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts with fallback to sequential calls."""
        client = await self.get_http_client()

        # Try batch endpoint first
        try:
            response = await client.post(
                "/embed/batch",
                json={"texts": texts, "model": self.config.embedding_model},
            )
            response.raise_for_status()
            return response.json().get("embeddings", [])
        except (httpx.HTTPStatusError, httpx.RequestError):
            # Fallback to sequential calls if batch endpoint not available
            embeddings = []
            for text in texts:
                emb = await self.get_embedding(text)
                embeddings.append(emb)
            return embeddings

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank documents using BGE reranker model via dedicated /rerank endpoint.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = return all)

        Returns:
            List of dicts with 'index', 'score', 'text' sorted by relevance
        """
        if not documents:
            return []

        client = await self.get_http_client()

        try:
            # Use dedicated rerank endpoint
            response = await client.post(
                "/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "model": self.config.reranker_model,
                    "top_k": top_k,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

            # Convert response to expected format
            results = [
                {"index": r["index"], "score": r["score"], "text": r["text"]}
                for r in result.get("results", [])
            ]

            return results

        except Exception as e:
            logger.warning("rerank_endpoint_failed", error=str(e))
            # Fallback: return documents in original order with zero scores
            return [
                {"index": i, "score": 0.0, "text": doc}
                for i, doc in enumerate(documents)
            ]

    async def rerank_with_scores(
        self,
        query: str,
        candidates: list[dict],
        text_field: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidate results and return with reranker scores.

        Args:
            query: Search query
            candidates: List of result dicts containing text to rerank
            text_field: Field name containing the text to rerank
            top_k: Number of results to return

        Returns:
            Candidates sorted by reranker score with 'rerank_score' added
        """
        if not candidates:
            return []

        # Extract texts for reranking
        documents = []
        for c in candidates:
            # Try multiple fields to get text
            text = ""
            if text_field in c:
                text = c[text_field]
            elif "payload" in c:
                payload = c["payload"]
                text = payload.get("title", "") + " " + payload.get("chunk_description", "")
            elif "source" in c:
                source = c["source"]
                text = source.get("title", "") + " " + source.get("chunk_description", "")

            # Truncate to avoid token limits
            documents.append(text[:1500] if text else "")

        # Get reranker scores
        rerank_results = await self.rerank(query, documents, top_k=None)

        # Create index to score mapping
        score_map = {r["index"]: r["score"] for r in rerank_results}

        # Add rerank scores to candidates
        for i, c in enumerate(candidates):
            c["rerank_score"] = score_map.get(i, 0.0)

        # Sort by rerank score
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        if top_k:
            candidates = candidates[:top_k]

        return candidates

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

    def get_corpus_size(self) -> int:
        """Get total number of unique products in corpus."""
        qdrant = self.get_qdrant()
        # Count parent nodes only
        result = qdrant.count(
            collection_name=self.config.qdrant_collection,
            count_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="parent"))]
            ),
        )
        return result.count


# =============================================================================
# Base Search Strategy
# =============================================================================

class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, clients: DatabaseClients, config: SearchConfig):
        self.clients = clients
        self.config = config
        self.preprocessor = QueryPreprocessor()

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Execute search and return results with latency.

        Returns:
            Tuple of (results, latency_seconds)
        """
        pass

    def extract_asins(self, results: list[dict[str, Any]]) -> list[str]:
        """Extract ASINs from search results with deduplication preserving order."""
        asins = []
        seen = set()
        for r in results:
            # Try multiple locations for ASIN
            asin = None

            # Direct asin field
            if r.get("asin"):
                asin = r["asin"]
            # From payload (Qdrant)
            elif r.get("payload"):
                payload = r["payload"]
                asin = payload.get("asin") or payload.get("parent_asin")
            # From source (Elasticsearch)
            elif r.get("source"):
                source = r["source"]
                asin = source.get("asin") or source.get("parent_asin")
            # From _source (Elasticsearch raw)
            elif r.get("_source"):
                source = r["_source"]
                asin = source.get("asin") or source.get("parent_asin")

            if asin and asin not in seen:
                seen.add(asin)
                asins.append(asin)

        return asins


# =============================================================================
# Evaluation Data Loader
# =============================================================================

@dataclass
class EvalQuery:
    """Single evaluation query."""
    query_id: str
    query_text: str
    search_type: str
    target_node: str
    target_section: str | None
    source_asin: str
    relevant_asins: list[str]
    relevance_scores: dict[str, float]
    difficulty: str
    filters: dict[str, Any] | None = None


def load_evaluation_data(
    eval_file: Path,
    search_type: str | None = None,
) -> list[EvalQuery]:
    """Load evaluation queries from JSON file."""
    with open(eval_file) as f:
        data = json.load(f)

    queries = []
    for item in data:
        if search_type and item.get("search_type") != search_type:
            continue

        relevant_asins_data = item.get("relevant_asins", [])
        relevant_asins = [r["asin"] for r in relevant_asins_data if "asin" in r]
        relevance_scores = {
            r["asin"]: r.get("relevance_score", 1)
            for r in relevant_asins_data
            if "asin" in r
        }

        queries.append(EvalQuery(
            query_id=item.get("query_id", ""),
            query_text=item.get("query_text", ""),
            search_type=item.get("search_type", ""),
            target_node=item.get("target_node", "parent"),
            target_section=item.get("target_section"),
            source_asin=item.get("source_asin", ""),
            relevant_asins=relevant_asins,
            relevance_scores=relevance_scores,
            difficulty=item.get("difficulty", "medium"),
            filters=item.get("filters"),
        ))

    return queries


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    """Result of running a search strategy experiment."""
    strategy_name: str
    strategy_description: str
    search_type: str
    metrics: RetrievalMetrics
    sample_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def run_experiment(
    strategy: SearchStrategy,
    eval_queries: list[EvalQuery],
    search_type: str,
    verbose: bool = False,
) -> ExperimentResult:
    """Run experiment with a search strategy."""
    all_metrics = []
    latencies = []
    unique_products = set()
    errors = []
    sample_results = []

    corpus_size = strategy.clients.get_corpus_size()

    for i, eq in enumerate(eval_queries):
        try:
            # Clean query
            clean_query = QueryPreprocessor.clean_query(eq.query_text)
            if not clean_query or len(clean_query) < 3:
                continue

            # Execute search
            results, latency = await strategy.search(
                clean_query,
                limit=10,
                filters=eq.filters,
            )
            latencies.append(latency)

            # Extract ASINs
            retrieved_asins = strategy.extract_asins(results)
            unique_products.update(retrieved_asins)

            # Calculate metrics
            relevant_set = set(eq.relevant_asins)
            query_metrics = {
                "precision_at_5": MetricsCalculator.precision_at_k(retrieved_asins, relevant_set, 5),
                "precision_at_10": MetricsCalculator.precision_at_k(retrieved_asins, relevant_set, 10),
                "recall_at_5": MetricsCalculator.recall_at_k(retrieved_asins, relevant_set, 5),
                "recall_at_10": MetricsCalculator.recall_at_k(retrieved_asins, relevant_set, 10),
                "mrr": MetricsCalculator.mrr(retrieved_asins, relevant_set),
                "ndcg_at_10": MetricsCalculator.ndcg_at_k(retrieved_asins, eq.relevance_scores, 10),
                "hit_rate": MetricsCalculator.hit_rate(retrieved_asins, relevant_set),
            }
            all_metrics.append(query_metrics)

            # Save sample results
            if i < 5:
                sample_results.append({
                    "query": clean_query[:100],
                    "target": eq.source_asin,
                    "retrieved": retrieved_asins[:5],
                    "metrics": query_metrics,
                })

            if verbose and i % 20 == 0:
                logger.info("experiment_progress", strategy=strategy.name, completed=i, total=len(eval_queries))

        except Exception as e:
            errors.append(f"Query {eq.query_id}: {str(e)}")
            if verbose:
                logger.warning("query_failed", query_id=eq.query_id, error=str(e))

    # Aggregate metrics
    metrics = MetricsCalculator.aggregate_metrics(
        all_metrics, latencies, unique_products, corpus_size
    )

    return ExperimentResult(
        strategy_name=strategy.name,
        strategy_description=strategy.description,
        search_type=search_type,
        metrics=metrics,
        sample_results=sample_results,
        errors=errors,
    )


def print_experiment_results(results: list[ExperimentResult]):
    """Print comparison of experiment results."""
    print("\n" + "=" * 100)
    print("SEARCH STRATEGY EXPERIMENT RESULTS")
    print("=" * 100)

    # Group by search type
    by_type = {}
    for r in results:
        if r.search_type not in by_type:
            by_type[r.search_type] = []
        by_type[r.search_type].append(r)

    for search_type, type_results in by_type.items():
        print(f"\n{'='*50}")
        print(f"Search Type: {search_type.upper()}")
        print(f"{'='*50}")

        # Sort by composite score (weighted average of key metrics)
        def composite_score(r: ExperimentResult) -> float:
            m = r.metrics
            return (
                m.recall_at_10 * 0.25 +
                m.precision_at_10 * 0.15 +
                m.mrr * 0.25 +
                m.ndcg_at_10 * 0.20 +
                m.hit_rate * 0.15
            )

        sorted_results = sorted(type_results, key=composite_score, reverse=True)

        # Print header
        print(f"\n{'Strategy':<30} {'R@10':>8} {'P@10':>8} {'MRR':>8} {'NDCG':>8} {'Hit%':>8} {'P50ms':>8} {'Score':>8}")
        print("-" * 100)

        for r in sorted_results:
            m = r.metrics
            score = composite_score(r)
            print(
                f"{r.strategy_name:<30} "
                f"{m.recall_at_10:>8.4f} "
                f"{m.precision_at_10:>8.4f} "
                f"{m.mrr:>8.4f} "
                f"{m.ndcg_at_10:>8.4f} "
                f"{m.hit_rate:>8.4f} "
                f"{m.latency_p50_ms:>8.1f} "
                f"{score:>8.4f}"
            )

        # Best strategy
        best = sorted_results[0]
        print(f"\n✓ BEST: {best.strategy_name}")
        print(f"  Description: {best.strategy_description}")

    print("\n" + "=" * 100)
