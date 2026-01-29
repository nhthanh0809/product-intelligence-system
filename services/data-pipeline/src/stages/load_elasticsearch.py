"""Load stage - Elasticsearch."""

from typing import Any

import structlog

from src.clients.elasticsearch_client import (
    ElasticsearchClientWrapper,
    get_elasticsearch_client,
)
from src.models.product import EmbeddedProduct
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


class LoadElasticsearchStage(BaseStage[EmbeddedProduct, dict[str, Any]]):
    """Load products into Elasticsearch.

    Performs bulk indexing for text search capabilities.
    """

    name = "load_elasticsearch"
    description = "Load products into Elasticsearch"

    def __init__(
        self,
        context: StageContext,
        es_client: ElasticsearchClientWrapper | None = None,
    ):
        super().__init__(context)
        self._client = es_client

    async def _ensure_client(self) -> ElasticsearchClientWrapper:
        """Ensure Elasticsearch client is initialized."""
        if self._client is None:
            self._client = await get_elasticsearch_client()
        return self._client

    async def process_batch(
        self,
        batch: list[EmbeddedProduct],
    ) -> list[dict[str, Any]]:
        """Process a batch of products."""
        client = await self._ensure_client()

        # Convert to Elasticsearch documents
        documents = []
        for product in batch:
            try:
                doc = self._product_to_document(product)
                documents.append(doc)
            except Exception as e:
                logger.warning(
                    "product_conversion_failed",
                    asin=product.asin,
                    error=str(e),
                )
                self.progress.failed += 1

        if not documents:
            return []

        # Bulk index
        try:
            success, errors = await client.bulk_index(
                documents,
                batch_size=len(documents),
            )

            # Update metrics
            self.context.job.metrics.products_loaded_elasticsearch += success

            # Track failures
            if errors > 0:
                self.progress.failed += errors

            return [{"asin": d["asin"], "status": "indexed"} for d in documents]

        except Exception as e:
            logger.error(
                "elasticsearch_batch_failed",
                count=len(documents),
                error=str(e),
            )
            self.progress.failed += len(documents)
            raise

    def _product_to_document(self, product: EmbeddedProduct) -> dict[str, Any]:
        """Convert EmbeddedProduct to Elasticsearch document."""
        doc = {
            "asin": product.asin,
            "title": product.title,
            "short_title": product.short_title,
            "brand": product.brand,
            "product_type": product.product_type,
            "price": product.price,
            "list_price": product.list_price,
            "stars": product.stars,
            "reviews_count": product.reviews_count,
            "bought_in_last_month": product.bought_in_last_month,
            "category_name": product.category_name,
            "category_level1": product.category_level1,
            "category_level2": product.category_level2,
            "category_level3": product.category_level3,
            "is_best_seller": product.is_best_seller,
            "is_amazon_choice": product.is_amazon_choice,
            "prime_eligible": product.prime_eligible,
            "product_description": product.product_description,
            "product_url": product.product_url,
            "img_url": product.img_url,
        }

        # Add features as text
        if product.features:
            doc["features"] = " | ".join(product.features)

        # Add product type keywords
        if product.product_type_keywords:
            doc["product_type_keywords"] = product.product_type_keywords

        # Remove None values
        return {k: v for k, v in doc.items() if v is not None}
