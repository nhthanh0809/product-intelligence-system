"""Elasticsearch client for text search indexing."""

from typing import Any

import structlog
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from src.config import get_settings

logger = structlog.get_logger()


# Elasticsearch index mapping for products
PRODUCTS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "autocomplete": {
                    "type": "custom",
                    "tokenizer": "autocomplete_tokenizer",
                    "filter": ["lowercase"],
                },
                "autocomplete_search": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                },
            },
            "tokenizer": {
                "autocomplete_tokenizer": {
                    "type": "edge_ngram",
                    "min_gram": 2,
                    "max_gram": 20,
                    "token_chars": ["letter", "digit"],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            # Identity
            "asin": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "autocomplete": {
                        "type": "text",
                        "analyzer": "autocomplete",
                        "search_analyzer": "autocomplete_search",
                    },
                    "keyword": {"type": "keyword"},
                },
            },
            "short_title": {"type": "text"},
            "brand": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "product_type": {"type": "keyword"},
            "product_type_keywords": {"type": "keyword"},
            # Pricing
            "price": {"type": "float"},
            "list_price": {"type": "float"},
            # Ratings
            "stars": {"type": "float"},
            "reviews_count": {"type": "integer"},
            "bought_in_last_month": {"type": "integer"},
            # Categories
            "category_name": {"type": "text"},
            "category_level1": {"type": "keyword"},
            "category_level2": {"type": "keyword"},
            "category_level3": {"type": "keyword"},
            # Flags
            "is_best_seller": {"type": "boolean"},
            "is_amazon_choice": {"type": "boolean"},
            "prime_eligible": {"type": "boolean"},
            # Content
            "product_description": {"type": "text"},
            "features": {"type": "text"},
            # GenAI fields
            "genAI_summary": {"type": "text"},
            "genAI_best_for": {"type": "text"},
            "genAI_primary_function": {"type": "text"},
            "genAI_use_cases": {"type": "text"},
            "genAI_key_capabilities": {"type": "text"},
            # Chunk content (for enriched mode)
            "chunk_description": {"type": "text"},
            "chunk_features": {"type": "text"},
            "chunk_specs": {"type": "text"},
            "chunk_reviews": {"type": "text"},
            "chunk_use_cases": {"type": "text"},
            # URLs
            "product_url": {"type": "keyword", "index": False},
            "img_url": {"type": "keyword", "index": False},
            # Metadata
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    },
}


class ElasticsearchClientWrapper:
    """Elasticsearch client wrapper for product indexing."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client: AsyncElasticsearch | None = None

    async def connect(self) -> None:
        """Initialize Elasticsearch client."""
        if self._client is not None:
            return

        self._client = AsyncElasticsearch(
            [f"http://{self.settings.elasticsearch_host}:{self.settings.elasticsearch_port}"],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        logger.info(
            "elasticsearch_connected",
            host=self.settings.elasticsearch_host,
            port=self.settings.elasticsearch_port,
        )

    async def close(self) -> None:
        """Close Elasticsearch client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("elasticsearch_disconnected")

    @property
    def client(self) -> AsyncElasticsearch:
        """Get Elasticsearch client."""
        if self._client is None:
            raise RuntimeError("Elasticsearch client not connected. Call connect() first.")
        return self._client

    async def ensure_index(self, recreate: bool = False) -> None:
        """Ensure index exists with correct mapping."""
        index_name = self.settings.elasticsearch_index

        if self._client is None:
            await self.connect()

        exists = await self._client.indices.exists(index=index_name)

        if exists and recreate:
            await self._client.indices.delete(index=index_name)
            exists = False
            logger.info("elasticsearch_index_deleted", index=index_name)

        if not exists:
            await self._client.indices.create(
                index=index_name,
                body=PRODUCTS_MAPPING,
            )
            logger.info("elasticsearch_index_created", index=index_name)

    async def bulk_index(
        self,
        products: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> tuple[int, int]:
        """Bulk index products.

        Returns:
            Tuple of (success_count, error_count)
        """
        if self._client is None:
            await self.connect()

        index_name = self.settings.elasticsearch_index
        success_count = 0
        error_count = 0

        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]

            def generate_actions():
                for product in batch:
                    asin = product.get("asin")
                    if not asin:
                        continue

                    # Prepare document
                    doc = {k: v for k, v in product.items() if v is not None}

                    yield {
                        "_index": index_name,
                        "_id": asin,
                        "_source": doc,
                    }

            success, errors = await async_bulk(
                self._client,
                generate_actions(),
                raise_on_error=False,
                raise_on_exception=False,
            )

            success_count += success
            if errors:
                error_count += len(errors)
                for error in errors[:5]:  # Log first 5 errors
                    logger.warning("elasticsearch_index_error", error=error)

        logger.info(
            "elasticsearch_bulk_index_completed",
            success=success_count,
            errors=error_count,
        )
        return success_count, error_count

    async def index_product(self, product: dict[str, Any]) -> bool:
        """Index a single product."""
        if self._client is None:
            await self.connect()

        index_name = self.settings.elasticsearch_index
        asin = product.get("asin")

        if not asin:
            return False

        try:
            await self._client.index(
                index=index_name,
                id=asin,
                document=product,
            )
            return True
        except Exception as e:
            logger.error("elasticsearch_index_failed", asin=asin, error=str(e))
            return False

    async def delete_product(self, asin: str) -> bool:
        """Delete a product from the index."""
        if self._client is None:
            await self.connect()

        try:
            await self._client.delete(
                index=self.settings.elasticsearch_index,
                id=asin,
            )
            return True
        except Exception as e:
            logger.error("elasticsearch_delete_failed", asin=asin, error=str(e))
            return False

    async def get_index_info(self) -> dict[str, Any]:
        """Get index information."""
        if self._client is None:
            await self.connect()

        index_name = self.settings.elasticsearch_index

        try:
            stats = await self._client.indices.stats(index=index_name)
            index_stats = stats["indices"].get(index_name, {})

            return {
                "name": index_name,
                "docs_count": index_stats.get("primaries", {}).get("docs", {}).get("count", 0),
                "size_bytes": index_stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0),
            }
        except Exception as e:
            return {
                "name": index_name,
                "error": str(e),
            }

    async def health_check(self) -> dict[str, Any]:
        """Check Elasticsearch health."""
        try:
            if self._client is None:
                await self.connect()

            health = await self._client.cluster.health()
            index_info = await self.get_index_info()

            return {
                "status": "healthy" if health["status"] in ["green", "yellow"] else "unhealthy",
                "cluster_status": health["status"],
                "index": index_info,
            }
        except Exception as e:
            logger.error("elasticsearch_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Singleton instance
_es_client: ElasticsearchClientWrapper | None = None


async def get_elasticsearch_client() -> ElasticsearchClientWrapper:
    """Get or create Elasticsearch client singleton."""
    global _es_client
    if _es_client is None:
        _es_client = ElasticsearchClientWrapper()
        await _es_client.connect()
    return _es_client
