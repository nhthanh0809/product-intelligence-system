"""Model management service for Ollama."""

import structlog
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = structlog.get_logger()


class ModelManager:
    """Manages Ollama models."""

    def __init__(self):
        """Initialize model manager."""
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host
        self._client: httpx.AsyncClient | None = None

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
    async def list_models(self) -> list[dict]:
        """List all available models."""
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get_model_info(self, name: str) -> dict | None:
        """Get information about a specific model."""
        client = await self._get_client()
        try:
            response = await client.post("/api/show", json={"name": name})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def pull_model(self, name: str, insecure: bool = False) -> dict:
        """Pull a model from Ollama library."""
        client = await self._get_client()
        logger.info("pulling_model", model=name)

        response = await client.post(
            "/api/pull",
            json={"name": name, "insecure": insecure},
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
        )
        response.raise_for_status()

        # Process streaming response
        result = {"status": "success", "name": name}
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "error" in data:
                    result = {"status": "error", "name": name, "error": data["error"]}
                    break
                if data.get("status") == "success":
                    result = {"status": "success", "name": name}

        logger.info("model_pulled", model=name, status=result["status"])
        return result

    async def delete_model(self, name: str) -> dict:
        """Delete a model."""
        client = await self._get_client()
        logger.info("deleting_model", model=name)

        response = await client.delete("/api/delete", json={"name": name})

        if response.status_code == 404:
            return {"status": "not_found", "name": name}

        response.raise_for_status()
        logger.info("model_deleted", model=name)
        return {"status": "success", "name": name}

    async def is_model_available(self, name: str) -> bool:
        """Check if a model is available."""
        info = await self.get_model_info(name)
        return info is not None

    async def ensure_model(self, name: str) -> bool:
        """Ensure a model is available, pulling if necessary."""
        if await self.is_model_available(name):
            return True

        logger.info("model_not_found_pulling", model=name)
        result = await self.pull_model(name)
        return result.get("status") == "success"

    async def health_check(self) -> dict:
        """Check Ollama service health."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return {
                "status": "healthy",
                "available_models": [m.get("name", "") for m in models],
            }
        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "available_models": [],
            }


# Singleton instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get model manager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
