"""Embedding service for Ollama."""

import asyncio
from typing import Sequence

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.services.model_manager import get_model_manager

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating embeddings via Ollama."""

    def __init__(self):
        """Initialize embedding service."""
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host
        self._client: httpx.AsyncClient | None = None
        self._model_manager = get_model_manager()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=self.settings.connect_timeout,
                    read=self.settings.read_timeout,
                    write=30.0,
                    pool=30.0,
                ),
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _embed_single(self, text: str, model: str) -> list[float]:
        """Generate embedding for a single text."""
        client = await self._get_client()

        # Truncate text if too long
        if len(text) > self.settings.embedding_max_length:
            text = text[: self.settings.embedding_max_length]

        response = await client.post(
            "/api/embeddings",
            json={"model": model, "prompt": text},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])

    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Model to use (defaults to config)

        Returns:
            Embedding vector
        """
        model = model or self.settings.default_embedding_model

        # Ensure model is available
        await self._model_manager.ensure_model(model)

        return await self._embed_single(text, model)

    async def embed_batch(
        self,
        texts: Sequence[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config)
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        model = model or self.settings.default_embedding_model
        batch_size = batch_size or self.settings.embedding_batch_size

        # Ensure model is available
        await self._model_manager.ensure_model(model)

        logger.info("embedding_batch", count=len(texts), model=model, batch_size=batch_size)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Process batch concurrently
            tasks = [self._embed_single(text, model) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            for j, emb in enumerate(batch_embeddings):
                if isinstance(emb, Exception):
                    logger.error(
                        "embedding_failed",
                        index=i + j,
                        error=str(emb),
                    )
                    # Return empty embedding for failed items
                    embeddings.append([0.0] * self.settings.embedding_dimensions)
                else:
                    embeddings.append(emb)

            logger.debug(
                "batch_completed",
                batch_start=i,
                batch_end=min(i + batch_size, len(texts)),
            )

        logger.info("embedding_batch_complete", total=len(embeddings))
        return embeddings

    def get_dimensions(self, model: str | None = None) -> int:
        """Get embedding dimensions for a model.

        Args:
            model: Model name (defaults to config)

        Returns:
            Number of dimensions
        """
        model = model or self.settings.default_embedding_model

        # Known dimensions for common models
        dimensions_map = {
            "bge-large": 1024,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
        }

        return dimensions_map.get(model, self.settings.embedding_dimensions)


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
