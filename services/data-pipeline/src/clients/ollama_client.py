"""Ollama embedding service client."""

import asyncio
from typing import Any

import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class OllamaClient:
    """Client for Ollama embedding service."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is not None and not self._client.is_closed:
            return

        self._client = httpx.AsyncClient(
            base_url=self.settings.ollama_service_url,
            timeout=httpx.Timeout(
                connect=self.settings.connect_timeout,
                read=self.settings.read_timeout,
                write=30.0,
                pool=30.0,
            ),
        )
        logger.info("ollama_client_connected", url=self.settings.ollama_service_url)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("ollama_client_disconnected")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None or self._client.is_closed:
            raise RuntimeError("Ollama client not connected. Call connect() first.")
        return self._client

    async def embed_single(self, text: str, model: str | None = None) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed
            model: Model to use (default from settings)

        Returns:
            Embedding vector
        """
        if self._client is None:
            await self.connect()

        response = await self._client.post(
            "/embed/single",
            json={
                "text": text,
                "model": model or self.settings.embedding_model,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data.get("embedding", [])

    async def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Model to use (default from settings)
            batch_size: Batch size (default from settings)

        Returns:
            List of embedding vectors
        """
        if self._client is None:
            await self.connect()

        batch_size = batch_size or self.settings.embedding_batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            response = await self._client.post(
                "/embed",
                json={
                    "texts": batch,
                    "model": model or self.settings.embedding_model,
                },
            )
            response.raise_for_status()

            data = response.json()
            batch_embeddings = data.get("embeddings", [])
            embeddings.extend(batch_embeddings)

            # Log progress for large batches
            if len(texts) > batch_size:
                logger.debug(
                    "embedding_batch_progress",
                    processed=min(i + batch_size, len(texts)),
                    total=len(texts),
                )

        return embeddings

    async def embed_products(
        self,
        products: list[dict[str, Any]],
        text_field: str = "embedding_text",
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Add embeddings to product dictionaries.

        Args:
            products: List of product dicts
            text_field: Field containing text to embed
            batch_size: Batch size for embedding

        Returns:
            Products with 'embedding' field added
        """
        # Extract texts
        texts = []
        valid_indices = []

        for i, product in enumerate(products):
            text = product.get(text_field)
            if text:
                texts.append(text)
                valid_indices.append(i)

        if not texts:
            return products

        # Get embeddings
        embeddings = await self.embed_batch(texts, batch_size=batch_size)

        # Add embeddings back to products
        for idx, embedding in zip(valid_indices, embeddings):
            products[idx]["embedding"] = embedding
            products[idx]["embedding_model"] = self.settings.embedding_model

        logger.info(
            "products_embedded",
            total=len(products),
            embedded=len(embeddings),
        )

        return products

    async def health_check(self) -> dict[str, Any]:
        """Check Ollama service health."""
        try:
            if self._client is None:
                await self.connect()

            response = await self._client.get("/health")

            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "ollama_status": data.get("status"),
                    "models": data.get("models", []),
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def list_models(self) -> dict[str, Any]:
        """List available models from Ollama service."""
        try:
            if self._client is None:
                await self.connect()

            response = await self._client.get("/models")

            if response.status_code == 200:
                return response.json()
            else:
                return {"models": [], "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error("ollama_list_models_failed", error=str(e))
            return {"models": [], "error": str(e)}


# Singleton instance
_ollama_client: OllamaClient | None = None


async def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
        await _ollama_client.connect()
    return _ollama_client
