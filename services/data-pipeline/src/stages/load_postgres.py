"""Load stage - PostgreSQL."""

from typing import Any

import structlog

from src.clients.postgres_client import PostgresClient, get_postgres_client
from src.models.product import EmbeddedProduct
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


class LoadPostgresStage(BaseStage[EmbeddedProduct, dict[str, Any]]):
    """Load products into PostgreSQL.

    Performs bulk upserts into the products table.
    """

    name = "load_postgres"
    description = "Load products into PostgreSQL"

    def __init__(
        self,
        context: StageContext,
        postgres_client: PostgresClient | None = None,
    ):
        super().__init__(context)
        self._client = postgres_client

    async def _ensure_client(self) -> PostgresClient:
        """Ensure PostgreSQL client is initialized."""
        if self._client is None:
            self._client = await get_postgres_client()
        return self._client

    async def process_batch(
        self,
        batch: list[EmbeddedProduct],
    ) -> list[dict[str, Any]]:
        """Process a batch of products."""
        client = await self._ensure_client()

        # Convert to dicts for PostgreSQL
        product_dicts = []
        for product in batch:
            try:
                product_dict = self._product_to_dict(product)
                product_dicts.append(product_dict)
            except Exception as e:
                logger.warning(
                    "product_conversion_failed",
                    asin=product.asin,
                    error=str(e),
                )
                self.progress.failed += 1

        if not product_dicts:
            return []

        # Bulk upsert
        try:
            inserted, updated = await client.bulk_upsert_products(
                product_dicts,
                batch_size=len(product_dicts),
            )

            # Update metrics
            self.context.job.metrics.products_loaded_postgres += len(product_dicts)

            # Return loaded products as dicts for tracking
            return [{"asin": p["asin"], "status": "loaded"} for p in product_dicts]

        except Exception as e:
            logger.error(
                "postgres_batch_failed",
                count=len(product_dicts),
                error=str(e),
            )
            self.progress.failed += len(product_dicts)
            raise

    def _product_to_dict(self, product: EmbeddedProduct) -> dict[str, Any]:
        """Convert EmbeddedProduct to PostgreSQL-compatible dict."""
        import json

        data = {
            "asin": product.asin,
            "title": product.title,
            "brand": product.brand,
            "short_title": product.short_title,
            "product_type": product.product_type,
            "price": product.price,
            "list_price": product.list_price,
            "original_price": product.original_price,
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
            "availability": product.availability,
            "product_description": product.product_description,
            "product_url": product.product_url,
            "img_url": product.img_url,
            "embedding_model": product.embedding_model,
            "embedded_at": product.embedded_at,
        }

        # Convert JSONB fields
        if product.features:
            data["features"] = json.dumps(product.features)

        if product.specifications:
            data["specifications"] = json.dumps(product.specifications)

        if product.product_type_keywords:
            data["product_type_keywords"] = json.dumps(product.product_type_keywords)

        # GenAI text fields
        genai_text_fields = [
            "genai_summary", "genai_primary_function", "genai_best_for",
            "genai_unique_selling_points", "genai_detailed_description",
            "genai_how_it_works", "genai_materials", "genai_technology_explained",
            "genai_feature_comparison", "genai_specs_summary", "genai_durability_feedback",
            "genai_value_for_money_feedback", "genai_sentiment_label",
        ]
        for field in genai_text_fields:
            value = getattr(product, field, None)
            if value is not None:
                data[field] = value

        # GenAI numeric fields
        if product.genai_value_score is not None:
            data["genai_value_score"] = product.genai_value_score
        if product.genai_sentiment_score is not None:
            data["genai_sentiment_score"] = product.genai_sentiment_score
        if product.genai_enriched_at is not None:
            data["genai_enriched_at"] = product.genai_enriched_at

        # GenAI JSONB fields (need to be serialized to JSON strings)
        genai_jsonb_fields = [
            "genai_use_cases", "genai_target_audience", "genai_key_capabilities",
            "genai_whats_included", "genai_features_detailed", "genai_standout_features",
            "genai_specs_comparison_ready", "genai_specs_limitations",
            "genai_common_praises", "genai_common_complaints",
            "genai_use_case_scenarios", "genai_ideal_user_profiles",
            "genai_not_recommended_for", "genai_problems_solved",
            "genai_pros", "genai_cons",
        ]
        for field in genai_jsonb_fields:
            value = getattr(product, field, None)
            if value is not None:
                if isinstance(value, (list, dict)):
                    # Already a list or dict, serialize to JSON
                    data[field] = json.dumps(value)
                elif isinstance(value, str):
                    # Check if it's already valid JSON
                    try:
                        json.loads(value)
                        # It's valid JSON, use as-is
                        data[field] = value
                    except json.JSONDecodeError:
                        # Plain string, wrap as a single-element array
                        data[field] = json.dumps([value])

        # Remove None values
        return {k: v for k, v in data.items() if v is not None}
