"""
Shared database clients for search strategies.

Provides centralized access to:
- Qdrant (vector search)
- Elasticsearch (keyword search)
- Ollama (embeddings)
- Reranking (via Ollama)
"""

import asyncio
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import httpx
import structlog
from elasticsearch import AsyncElasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

if TYPE_CHECKING:
    from src.config.models import RerankerConfigWithModel

logger = structlog.get_logger()


@dataclass
class SearchClientsConfig:
    """Configuration for search clients."""
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

    @classmethod
    def from_settings(cls) -> "SearchClientsConfig":
        """Create config from service settings."""
        from src.config import get_settings

        settings = get_settings()
        return cls(
            qdrant_host=getattr(settings, "qdrant_host", "qdrant"),
            qdrant_port=getattr(settings, "qdrant_port", 6333),
            qdrant_collection=getattr(settings, "qdrant_collection", "products"),
            es_host=getattr(settings, "elasticsearch_host", "elasticsearch"),
            es_port=getattr(settings, "elasticsearch_port", 9200),
            es_index=getattr(settings, "elasticsearch_index", "products"),
            ollama_url=getattr(settings, "ollama_service_url", "http://ollama-service:8010"),
        )


class SearchClients:
    """Shared database clients for search strategies.

    Manages connections to:
    - Qdrant for vector search
    - Elasticsearch for keyword search
    - Ollama for embedding generation

    Usage:
        clients = SearchClients()
        await clients.initialize()

        # Use clients
        vector = await clients.get_embedding("query text")
        results = await clients.semantic_search(vector, limit=10)

        await clients.close()
    """

    def __init__(self, config: SearchClientsConfig | None = None):
        """Initialize clients with configuration.

        Args:
            config: Client configuration (uses defaults from settings if None)
        """
        self.config = config or SearchClientsConfig.from_settings()
        self._qdrant: QdrantClient | None = None
        self._elasticsearch: AsyncElasticsearch | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Check if clients are initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize all database connections."""
        async with self._lock:
            if self._initialized:
                return

            try:
                # Initialize Qdrant
                self._qdrant = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    timeout=30.0,
                )

                # Initialize Elasticsearch
                self._elasticsearch = AsyncElasticsearch(
                    hosts=[{
                        "host": self.config.es_host,
                        "port": self.config.es_port,
                        "scheme": "http",
                    }],
                    request_timeout=30.0,
                )

                # Initialize HTTP client for Ollama
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.ollama_url,
                    timeout=60.0,
                )

                self._initialized = True
                logger.info(
                    "search_clients_initialized",
                    qdrant=f"{self.config.qdrant_host}:{self.config.qdrant_port}",
                    elasticsearch=f"{self.config.es_host}:{self.config.es_port}",
                    ollama=self.config.ollama_url,
                )

            except Exception as e:
                logger.error("search_clients_init_failed", error=str(e))
                raise

    async def close(self) -> None:
        """Close all database connections."""
        async with self._lock:
            if self._elasticsearch:
                await self._elasticsearch.close()
                self._elasticsearch = None

            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

            if self._qdrant:
                self._qdrant.close()
                self._qdrant = None

            self._initialized = False
            logger.info("search_clients_closed")

    # =========================================================================
    # Embedding Operations
    # =========================================================================

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If clients not initialized
            httpx.HTTPError: If embedding service fails
        """
        if not self._initialized or not self._http_client:
            raise RuntimeError("Search clients not initialized")

        try:
            response = await self._http_client.post(
                "/embed/single",
                json={
                    "text": text,
                    "model": self.config.embedding_model,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

        except Exception as e:
            logger.error("embedding_failed", text=text[:50], error=str(e))
            raise

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized or not self._http_client:
            raise RuntimeError("Search clients not initialized")

        try:
            response = await self._http_client.post(
                "/embed/batch",
                json={
                    "texts": texts,
                    "model": self.config.embedding_model,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [])

        except Exception as e:
            logger.error("batch_embedding_failed", count=len(texts), error=str(e))
            raise

    # =========================================================================
    # Qdrant Operations
    # =========================================================================

    async def semantic_search(
        self,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Execute semantic search in Qdrant.

        Args:
            vector: Query embedding vector
            limit: Maximum results to return
            filters: Qdrant filter conditions
            score_threshold: Minimum similarity score

        Returns:
            List of search results with payload and score
        """
        if not self._initialized or not self._qdrant:
            raise RuntimeError("Search clients not initialized")

        try:
            # Build Qdrant filter
            qdrant_filter = self._build_qdrant_filter(filters) if filters else None

            # Use query_points for qdrant-client >= 1.7.0
            response = self._qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=vector,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )

            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in response.points
            ]

        except Exception as e:
            logger.error("semantic_search_failed", error=str(e))
            raise

    def _build_qdrant_filter(self, filters: dict[str, Any]) -> Filter | None:
        """Build Qdrant filter from filter dict."""
        conditions = []

        if filters.get("brand"):
            conditions.append(
                FieldCondition(
                    key="brand",
                    match=MatchValue(value=filters["brand"]),
                )
            )

        if filters.get("category"):
            conditions.append(
                FieldCondition(
                    key="category_name",
                    match=MatchValue(value=filters["category"]),
                )
            )

        # Handle both naming conventions: min_price/max_price and price_min/price_max
        min_price = filters.get("min_price") or filters.get("price_min")
        max_price = filters.get("max_price") or filters.get("price_max")
        if min_price is not None or max_price is not None:
            range_filter = {}
            if min_price is not None:
                range_filter["gte"] = min_price
            if max_price is not None:
                range_filter["lte"] = max_price
            conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(**range_filter),
                )
            )

        if filters.get("min_rating") is not None:
            conditions.append(
                FieldCondition(
                    key="stars",
                    range=Range(gte=filters["min_rating"]),
                )
            )

        if conditions:
            return Filter(must=conditions)
        return None

    # =========================================================================
    # Elasticsearch Operations
    # =========================================================================

    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        boost_config: dict[str, float] | None = None,
    ) -> list[dict]:
        """Execute keyword search in Elasticsearch.

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Filter conditions
            boost_config: Field boost configuration

        Returns:
            List of search results with score
        """
        if not self._initialized or not self._elasticsearch:
            raise RuntimeError("Search clients not initialized")

        try:
            # Default boost configuration
            boost = boost_config or {
                "title": 10.0,
                "title.autocomplete": 5.0,
                "short_title": 8.0,
                "brand": 5.0,
                "product_type": 4.0,
                "genAI_summary": 2.0,
            }

            # Build multi_match query with boosted fields
            fields = [f"{field}^{weight}" for field, weight in boost.items()]

            es_query: dict[str, Any] = {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": fields,
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }
                    ]
                }
            }

            # Add filters
            if filters:
                filter_clauses = self._build_es_filters(filters)
                if filter_clauses:
                    es_query["bool"]["filter"] = filter_clauses

            response = await self._elasticsearch.search(
                index=self.config.es_index,
                query=es_query,
                size=limit,
                _source=True,
            )

            return [
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"],
                }
                for hit in response["hits"]["hits"]
            ]

        except Exception as e:
            logger.error("keyword_search_failed", query=query, error=str(e))
            raise

    def _build_es_filters(self, filters: dict[str, Any]) -> list[dict]:
        """Build Elasticsearch filter clauses."""
        clauses = []

        if filters.get("brand"):
            clauses.append({"term": {"brand.keyword": filters["brand"]}})

        if filters.get("category"):
            clauses.append({"term": {"category_name.keyword": filters["category"]}})

        # Handle both naming conventions: min_price/max_price and price_min/price_max
        min_price = filters.get("min_price") or filters.get("price_min")
        max_price = filters.get("max_price") or filters.get("price_max")
        if min_price is not None or max_price is not None:
            range_filter: dict[str, Any] = {}
            if min_price is not None:
                range_filter["gte"] = min_price
            if max_price is not None:
                range_filter["lte"] = max_price
            clauses.append({"range": {"price": range_filter}})

        if filters.get("min_rating") is not None:
            clauses.append({"range": {"stars": {"gte": filters["min_rating"]}}})

        return clauses

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_product_by_asin(self, asin: str) -> dict | None:
        """Get product by ASIN from Elasticsearch.

        Args:
            asin: Product ASIN

        Returns:
            Product document or None if not found
        """
        if not self._initialized or not self._elasticsearch:
            raise RuntimeError("Search clients not initialized")

        try:
            response = await self._elasticsearch.search(
                index=self.config.es_index,
                query={"term": {"asin": asin}},
                size=1,
            )

            hits = response["hits"]["hits"]
            if hits:
                return hits[0]["_source"]
            return None

        except Exception as e:
            logger.error("get_product_failed", asin=asin, error=str(e))
            return None

    async def get_product_vector(self, asin: str) -> list[float] | None:
        """Get product embedding vector from Qdrant.

        Args:
            asin: Product ASIN

        Returns:
            Embedding vector or None if not found
        """
        if not self._initialized or not self._qdrant:
            raise RuntimeError("Search clients not initialized")

        try:
            results = self._qdrant.retrieve(
                collection_name=self.config.qdrant_collection,
                ids=[asin],
                with_vectors=True,
            )

            if results:
                return results[0].vector
            return None

        except Exception as e:
            logger.error("get_product_vector_failed", asin=asin, error=str(e))
            return None

    # =========================================================================
    # Reranking Operations
    # =========================================================================

    async def rerank_results(
        self,
        query: str,
        results: list[dict],
        reranker_config: "RerankerConfigWithModel | None" = None,
        force_rerank: bool = False,
    ) -> list[dict]:
        """Rerank search results using the configured reranker model.

        Args:
            query: Original search query
            results: List of search results (dicts with 'title', 'id', etc.)
            reranker_config: Reranker configuration (fetched if not provided)
            force_rerank: If True, bypass database config check and use defaults

        Returns:
            Reranked results list (sorted by relevance)
        """
        if not results:
            return results

        # Get reranker config if not provided
        if reranker_config is None and not force_rerank:
            reranker_config = await self._get_reranker_config()

        if not force_rerank and (reranker_config is None or not reranker_config.is_enabled):
            logger.debug("reranker_disabled_or_not_configured")
            return results

        try:
            # Get model info - use defaults if force_rerank with no config
            if reranker_config and reranker_config.model:
                model_name = reranker_config.model.model_name
                settings = reranker_config.settings or {}
            else:
                # Default reranker settings when force_rerank is used
                model_name = "qllama/bge-reranker-v2-m3"
                settings = {}
                logger.debug("using_default_reranker_model", model=model_name)

            top_k = settings.get("top_k", len(results))
            threshold = settings.get("threshold", 0.0)
            batch_size = settings.get("batch_size", 32)

            logger.info(
                "reranking_started",
                query=query[:50],
                num_results=len(results),
                model=model_name,
                top_k=top_k,
            )

            # Extract document texts for reranking
            documents = []
            for r in results:
                # Build document text from available fields
                text_parts = []
                if r.get("title"):
                    text_parts.append(r["title"])
                if r.get("brand"):
                    text_parts.append(f"Brand: {r['brand']}")
                if r.get("genAI_summary"):
                    text_parts.append(r["genAI_summary"][:200])
                documents.append(" | ".join(text_parts) if text_parts else str(r.get("id", "")))

            # Call Ollama service for reranking
            scores = await self._call_reranker(query, documents, model_name, batch_size)

            if not scores or len(scores) != len(results):
                logger.warning("reranker_score_mismatch", expected=len(results), got=len(scores) if scores else 0)
                return results

            # Combine results with scores and sort
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Filter by threshold and limit
            reranked = []
            for result, score in scored_results:
                if score >= threshold:
                    result["rerank_score"] = score
                    reranked.append(result)
                    if len(reranked) >= top_k:
                        break

            logger.info(
                "reranking_completed",
                original_count=len(results),
                reranked_count=len(reranked),
                top_score=reranked[0].get("rerank_score") if reranked else 0,
            )

            return reranked

        except Exception as e:
            logger.error("reranking_failed", error=str(e), query=query[:50])
            # Return original results on error
            return results

    async def _get_reranker_config(self) -> "RerankerConfigWithModel | None":
        """Get the active reranker configuration from database."""
        try:
            from src.config.manager import get_config_manager

            manager = await get_config_manager()
            return await manager.get_reranker_config()
        except Exception as e:
            logger.warning("reranker_config_fetch_failed", error=str(e))
            return None

    async def _call_reranker(
        self,
        query: str,
        documents: list[str],
        model: str,
        batch_size: int = 32,
    ) -> list[float]:
        """Call the reranker model via Ollama service.

        Args:
            query: Search query
            documents: List of document texts to score
            model: Reranker model name
            batch_size: Process documents in batches

        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")

        scores = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            try:
                # Call ollama-service rerank endpoint
                response = await self._http_client.post(
                    "/rerank",
                    json={
                        "query": query,
                        "documents": batch,
                        "model": model,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Parse results - response has "results" with score and index
                    results = data.get("results", [])
                    if results:
                        # Sort by original index to maintain document order
                        sorted_results = sorted(results, key=lambda x: x.get("index", 0))
                        batch_scores = [r.get("score", 0.0) for r in sorted_results]
                    else:
                        # Fallback to old format
                        batch_scores = data.get("scores", [])
                    scores.extend(batch_scores)
                else:
                    # Fallback: use LLM-based scoring
                    logger.warning(
                        "rerank_endpoint_failed",
                        status=response.status_code,
                        falling_back_to_llm=True,
                    )
                    batch_scores = await self._llm_based_rerank(query, batch, model)
                    scores.extend(batch_scores)

            except httpx.HTTPError as e:
                logger.warning("rerank_http_error", error=str(e), falling_back_to_llm=True)
                # Fallback to LLM-based scoring
                batch_scores = await self._llm_based_rerank(query, batch, model)
                scores.extend(batch_scores)

        return scores

    async def _llm_based_rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
    ) -> list[float]:
        """Fallback LLM-based reranking using generation API.

        Uses a prompt to ask the LLM to score relevance.
        """
        scores = []

        for doc in documents:
            try:
                # Use generation endpoint with scoring prompt
                prompt = f"""Rate the relevance of this document to the query on a scale of 0.0 to 1.0.
Only respond with a single number.

Query: {query}
Document: {doc[:500]}

Relevance score:"""

                response = await self._http_client.post(
                    "/generate",
                    json={
                        "prompt": prompt,
                        "model": model,
                        "max_tokens": 10,
                        "temperature": 0.0,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    text = data.get("response", "0.5").strip()
                    try:
                        score = float(text.split()[0])
                        score = max(0.0, min(1.0, score))
                    except (ValueError, IndexError):
                        score = 0.5
                    scores.append(score)
                else:
                    scores.append(0.5)  # Default score on error

            except Exception as e:
                logger.debug("llm_rerank_doc_failed", error=str(e))
                scores.append(0.5)

        return scores


# Singleton instance
_clients: SearchClients | None = None


async def get_search_clients(config: SearchClientsConfig | None = None) -> SearchClients:
    """Get or create the search clients singleton.

    Args:
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Initialized SearchClients instance
    """
    global _clients

    if _clients is None:
        _clients = SearchClients(config)

    if not _clients.is_initialized:
        await _clients.initialize()

    return _clients


async def close_search_clients() -> None:
    """Close the search clients singleton."""
    global _clients

    if _clients:
        await _clients.close()
        _clients = None
