"""Embed stage - generate embeddings for products."""

from datetime import datetime
from typing import Any

import structlog

from src.clients.ollama_client import OllamaClient, get_ollama_client
from src.models.product import CleanedProduct, EmbeddedProduct
from src.models.chunk import ProductChunk
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


class EmbedStage(BaseStage[CleanedProduct, EmbeddedProduct]):
    """Generate embeddings for products.

    Uses Ollama service to generate vector embeddings
    for product text content.
    """

    name = "embed"
    description = "Generate embeddings for products"

    def __init__(
        self,
        context: StageContext,
        ollama_client: OllamaClient | None = None,
        embedding_model: str | None = None,
    ):
        super().__init__(context)
        self._client = ollama_client
        self._initialized = False
        self._embedding_model = embedding_model  # Override model if specified

    async def _ensure_client(self) -> OllamaClient:
        """Ensure Ollama client is initialized."""
        if self._client is None:
            self._client = await get_ollama_client()
        return self._client

    async def process_batch(
        self,
        batch: list[CleanedProduct],
    ) -> list[EmbeddedProduct]:
        """Process a batch of cleaned products."""
        client = await self._ensure_client()

        # Extract texts for embedding
        texts = []
        valid_indices = []

        for i, product in enumerate(batch):
            text = product.embedding_text or product.build_embedding_text()
            if text:
                texts.append(text)
                valid_indices.append(i)

        if not texts:
            return []

        # Determine which model to use
        model = self._embedding_model or client.settings.embedding_model

        # Get embeddings in batch
        embeddings = await client.embed_batch(texts, model=model)

        # Create embedded products
        embedded_products = []
        now = datetime.now()

        for idx, embedding in zip(valid_indices, embeddings):
            product = batch[idx]

            # Create EmbeddedProduct with all fields
            embedded = EmbeddedProduct(
                **product.model_dump(),
                embedding=embedding,
                embedding_model=model,
                embedded_at=now,
            )
            embedded_products.append(embedded)

        return embedded_products

    async def embed_chunks(
        self,
        chunks: list[ProductChunk],
        batch_size: int | None = None,
    ) -> list[ProductChunk]:
        """Generate embeddings for chunks.

        Args:
            chunks: List of ProductChunk objects
            batch_size: Batch size for embedding

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return []

        client = await self._ensure_client()
        batch_size = batch_size or self.context.batch_size
        now = datetime.now()

        # Determine which model to use
        model = self._embedding_model or client.settings.embedding_model

        # Process in batches
        embedded_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Extract texts
            texts = [chunk.content for chunk in batch]

            # Get embeddings
            embeddings = await client.embed_batch(texts, model=model)

            # Update chunks with embeddings
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
                chunk.embedding_model = model
                chunk.embedded_at = now
                embedded_chunks.append(chunk)

            # Log progress
            processed = min(i + batch_size, len(chunks))
            logger.debug(
                "chunk_embedding_progress",
                processed=processed,
                total=len(chunks),
            )

        logger.info(
            "chunks_embedded",
            total=len(embedded_chunks),
            model=model,
        )

        return embedded_chunks
