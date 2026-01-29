"""Load stage - Qdrant vector store."""

from typing import Any

import structlog

from src.clients.qdrant_client import QdrantClientWrapper, get_qdrant_client
from src.models.product import EmbeddedProduct
from src.models.chunk import ProductChunk
from src.models.enums import IndexingStrategy, NodeType
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


class LoadQdrantStage(BaseStage[EmbeddedProduct, dict[str, Any]]):
    """Load products and chunks into Qdrant.

    Supports 3 indexing strategies:
    - parent_only: Only parent nodes (original mode)
    - enrich_existing: Update parents, add children
    - add_child_node: Fresh load with all nodes
    """

    name = "load_qdrant"
    description = "Load products into Qdrant vector store"

    def __init__(
        self,
        context: StageContext,
        qdrant_client: QdrantClientWrapper | None = None,
        strategy: IndexingStrategy | None = None,
    ):
        super().__init__(context)
        self._client = qdrant_client
        self.strategy = strategy or context.job.indexing_strategy
        self._chunks: list[ProductChunk] = []

    def set_chunks(self, chunks: list[ProductChunk]) -> None:
        """Set chunks to be loaded alongside products."""
        self._chunks = chunks

    def _ensure_client(self) -> QdrantClientWrapper:
        """Ensure Qdrant client is initialized."""
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    async def process_batch(
        self,
        batch: list[EmbeddedProduct],
    ) -> list[dict[str, Any]]:
        """Process a batch of products."""
        client = self._ensure_client()

        # Get GenAI fields from context (set by LLM extract stage)
        genai_fields = self.context.data.get("genai_fields", {})

        # Prepare points for Qdrant
        points = []
        for product in batch:
            if not product.embedding:
                logger.warning(
                    "product_missing_embedding",
                    asin=product.asin,
                )
                self.progress.failed += 1
                continue

            payload = product.to_payload().model_dump()

            # Add GenAI fields if available
            product_genai = genai_fields.get(product.asin, {})
            if product_genai:
                payload.update({
                    "genAI_summary": product_genai.get("genAI_summary"),
                    "genAI_best_for": product_genai.get("genAI_best_for"),
                    "genAI_primary_function": product_genai.get("genAI_primary_function"),
                    "genAI_use_cases": product_genai.get("genAI_use_cases"),
                    "genAI_key_capabilities": product_genai.get("genAI_key_capabilities"),
                })

            points.append({
                "embedding": product.embedding,
                "payload": payload,
            })

        if not points:
            return []

        # Upsert products
        try:
            upserted = await client.upsert_products(
                points,
                strategy=self.strategy,
                batch_size=len(points),
            )

            # Update metrics
            self.context.job.metrics.products_loaded_qdrant += upserted

            return [{"asin": p["payload"]["asin"], "status": "loaded"} for p in points]

        except Exception as e:
            logger.error(
                "qdrant_batch_failed",
                count=len(points),
                error=str(e),
            )
            self.progress.failed += len(points)
            raise

    async def load_chunks(self, chunks: list[ProductChunk]) -> int:
        """Load chunks as child nodes.

        Args:
            chunks: List of ProductChunk with embeddings

        Returns:
            Number of chunks loaded
        """
        if not chunks:
            return 0

        client = self._ensure_client()

        # Prepare chunk points
        chunk_points = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(
                    "chunk_missing_embedding",
                    chunk_id=chunk.chunk_id,
                )
                continue

            payload = chunk.to_payload().model_dump()
            chunk_points.append({
                "embedding": chunk.embedding,
                "payload": payload,
            })

        if not chunk_points:
            return 0

        # Upsert chunks
        try:
            upserted = await client.upsert_chunks(
                chunk_points,
                batch_size=100,
            )

            # Update metrics
            self.context.job.metrics.chunks_loaded += upserted

            logger.info(
                "chunks_loaded_to_qdrant",
                count=upserted,
                strategy=self.strategy.value,
            )

            return upserted

        except Exception as e:
            logger.error(
                "qdrant_chunks_failed",
                count=len(chunk_points),
                error=str(e),
            )
            raise

    async def run_with_chunks(
        self,
        products: list[EmbeddedProduct],
        chunks: list[ProductChunk] | None = None,
    ) -> dict[str, int]:
        """Load both products and chunks.

        Args:
            products: List of products with embeddings
            chunks: Optional list of chunks with embeddings

        Returns:
            Dict with loaded counts
        """
        result = {
            "products": 0,
            "chunks": 0,
        }

        # Load products using base run method
        loaded = await self.run(products)
        result["products"] = len(loaded)

        # Load chunks if provided and strategy supports it
        if chunks and self.strategy in (
            IndexingStrategy.ENRICH_EXISTING,
            IndexingStrategy.ADD_CHILD_NODE,
        ):
            result["chunks"] = await self.load_chunks(chunks)

        return result
