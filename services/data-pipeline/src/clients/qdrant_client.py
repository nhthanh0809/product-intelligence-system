"""Qdrant client with support for 3 indexing strategies."""

from typing import Any
from uuid import uuid4

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from src.config import get_settings
from src.models.enums import IndexingStrategy, NodeType

logger = structlog.get_logger()


class QdrantClientWrapper:
    """Qdrant client wrapper with indexing strategy support."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client: QdrantClient | None = None

    def connect(self) -> None:
        """Initialize Qdrant client."""
        if self._client is not None:
            return

        self._client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
        logger.info(
            "qdrant_connected",
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )

    def close(self) -> None:
        """Close Qdrant client."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("qdrant_disconnected")

    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client, connecting if necessary."""
        if self._client is None:
            self.connect()
        return self._client

    async def ensure_collection(self, recreate: bool = False) -> None:
        """Ensure collection exists with correct schema."""
        collection_name = self.settings.qdrant_collection

        exists = self.client.collection_exists(collection_name)

        if exists and recreate:
            self.client.delete_collection(collection_name)
            exists = False
            logger.info("qdrant_collection_deleted", collection=collection_name)

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            self._create_indexes(collection_name)

            logger.info("qdrant_collection_created", collection=collection_name)

    def _create_indexes(self, collection_name: str) -> None:
        """Create payload field indexes."""
        # Keyword indexes for exact match filtering
        keyword_fields = [
            "asin",
            "brand",
            "category_level1",
            "category_level2",
            "node_type",
            "section",
            "parent_asin",
        ]

        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        # Float indexes for range filtering
        float_fields = ["price", "stars"]
        for field in float_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.FLOAT,
            )

        # Integer indexes
        int_fields = ["reviews_count", "bought_in_last_month"]
        for field in int_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )

        # Boolean indexes
        bool_fields = ["is_best_seller", "is_amazon_choice", "prime_eligible"]
        for field in bool_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.BOOL,
            )

    async def upsert_products(
        self,
        products: list[dict[str, Any]],
        strategy: IndexingStrategy = IndexingStrategy.PARENT_ONLY,
        batch_size: int = 100,
    ) -> int:
        """Upsert product vectors based on indexing strategy.

        Args:
            products: List of dicts with 'embedding', 'payload' keys
            strategy: Indexing strategy to use
            batch_size: Batch size for upserting

        Returns:
            Number of points upserted
        """
        collection_name = self.settings.qdrant_collection
        total_upserted = 0

        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            points = []

            for product in batch:
                embedding = product.get("embedding")
                payload = product.get("payload", {})

                if not embedding:
                    continue

                # Generate point ID based on strategy
                asin = payload.get("asin", "")
                node_type = payload.get("node_type", "parent")

                if node_type == "parent":
                    point_id = self._generate_point_id(asin, "parent")
                else:
                    section = payload.get("section", "")
                    point_id = self._generate_point_id(asin, section)

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                total_upserted += len(points)

        logger.info(
            "qdrant_upsert_completed",
            strategy=strategy.value,
            total=total_upserted,
        )
        return total_upserted

    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Upsert chunk (child node) vectors.

        Args:
            chunks: List of dicts with 'embedding', 'payload' keys
            batch_size: Batch size for upserting

        Returns:
            Number of points upserted
        """
        collection_name = self.settings.qdrant_collection
        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            for chunk in batch:
                embedding = chunk.get("embedding")
                payload = chunk.get("payload", {})

                if not embedding:
                    continue

                parent_asin = payload.get("parent_asin", "")
                section = payload.get("section", "")
                point_id = self._generate_point_id(parent_asin, section)

                # Ensure node_type is set
                payload["node_type"] = NodeType.CHILD.value

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                total_upserted += len(points)

        logger.info("qdrant_chunks_upserted", total=total_upserted)
        return total_upserted

    def _generate_point_id(self, asin: str, suffix: str) -> str:
        """Generate a deterministic point ID from ASIN and suffix.

        Uses a UUID-like format derived from the ASIN+suffix hash
        for consistent point IDs across runs.
        """
        import hashlib

        # Create deterministic hash from ASIN + suffix
        content = f"{asin}_{suffix}"
        hash_bytes = hashlib.sha256(content.encode()).digest()[:16]

        # Format as UUID string
        return str(
            uuid4().__class__(bytes=hash_bytes, version=4)
        ).replace("-", "")[:32]

    async def delete_by_asin(self, asin: str) -> int:
        """Delete all points for a given ASIN (parent + children)."""
        collection_name = self.settings.qdrant_collection

        # Delete parent
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="asin", match=MatchValue(value=asin))]
            ),
        )

        # Delete children
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="parent_asin", match=MatchValue(value=asin))]
            ),
        )

        return 1  # Approximate

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection information."""
        collection_name = self.settings.qdrant_collection

        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }
        except Exception as e:
            return {
                "name": collection_name,
                "error": str(e),
            }

    async def health_check(self) -> dict[str, Any]:
        """Check Qdrant health."""
        try:
            # Check if we can connect and get collection info
            info = await self.get_collection_info()

            return {
                "status": "healthy",
                "collection": info,
            }
        except Exception as e:
            logger.error("qdrant_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Singleton instance
_qdrant_client: QdrantClientWrapper | None = None


def get_qdrant_client() -> QdrantClientWrapper:
    """Get or create Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClientWrapper()
        _qdrant_client.connect()
    return _qdrant_client
