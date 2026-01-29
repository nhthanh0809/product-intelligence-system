"""API client for multi-agent service."""

from typing import Any

import httpx

from config import get_settings


class APIClient:
    """HTTP client for multi-agent service."""

    def __init__(self):
        self.settings = get_settings()

    def _get_client(self) -> httpx.AsyncClient:
        """Create a new HTTP client for current event loop."""
        return httpx.AsyncClient(
            base_url=self.settings.multi_agent_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.settings.request_timeout,
                write=30.0,
                pool=30.0,
            ),
        )

    # ==========================================================================
    # Search Endpoints
    # ==========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        brand: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
    ) -> dict[str, Any]:
        """Search products."""
        payload = {"query": query, "limit": limit}
        if category:
            payload["category"] = category
        if brand:
            payload["brand"] = brand
        if price_min is not None:
            payload["price_min"] = price_min
        if price_max is not None:
            payload["price_max"] = price_max

        async with self._get_client() as client:
            response = await client.post("/search", json=payload)
            response.raise_for_status()
            return response.json()

    async def query(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        """Send query to multi-agent system."""
        payload = {"query": query}
        if session_id:
            payload["session_id"] = session_id

        async with self._get_client() as client:
            response = await client.post("/query", json=payload)
            response.raise_for_status()
            return response.json()

    # ==========================================================================
    # Specialist Agent Endpoints
    # ==========================================================================

    async def compare(
        self,
        query: str,
        product_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare products."""
        payload = {"query": query}
        if product_names:
            payload["product_names"] = product_names

        async with self._get_client() as client:
            response = await client.post("/compare", json=payload)
            response.raise_for_status()
            return response.json()

    async def analyze_price(
        self,
        query: str,
        target_price: float | None = None,
    ) -> dict[str, Any]:
        """Analyze prices."""
        payload = {"query": query}
        if target_price:
            payload["target_price"] = target_price

        async with self._get_client() as client:
            response = await client.post("/price", json=payload)
            response.raise_for_status()
            return response.json()

    async def get_trends(
        self,
        query: str,
        category: str | None = None,
        time_range: str = "7d",
    ) -> dict[str, Any]:
        """Get market trends."""
        payload = {
            "query": query,
            "time_range": time_range,
        }
        if category:
            payload["category"] = category

        async with self._get_client() as client:
            response = await client.post("/trends", json=payload)
            response.raise_for_status()
            return response.json()

    async def get_recommendations(
        self,
        query: str,
        source_asin: str | None = None,
        recommendation_type: str = "similar",
    ) -> dict[str, Any]:
        """Get product recommendations."""
        payload = {
            "query": query,
            "recommendation_type": recommendation_type,
        }
        if source_asin:
            payload["source_asin"] = source_asin

        async with self._get_client() as client:
            response = await client.post("/recommend", json=payload)
            response.raise_for_status()
            return response.json()

    async def analyze_reviews(
        self,
        query: str,
        analysis_type: str = "general",
    ) -> dict[str, Any]:
        """Analyze product reviews."""
        payload = {
            "query": query,
            "analysis_type": analysis_type,
        }

        async with self._get_client() as client:
            response = await client.post("/reviews", json=payload)
            response.raise_for_status()
            return response.json()

    # ==========================================================================
    # Conversation Endpoints
    # ==========================================================================

    async def chat(
        self,
        query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Send chat message (legacy endpoint)."""
        payload = {"query": query}
        if session_id:
            payload["session_id"] = session_id

        async with self._get_client() as client:
            response = await client.post("/chat", json=payload)
            response.raise_for_status()
            return response.json()

    async def chat_v2(
        self,
        query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Send chat message using enhanced v2 endpoint.

        Returns:
            Response with:
            - session_id: Session identifier
            - query: Original query
            - response: Natural language response
            - intent: Detected intent (search, compare, recommend, etc.)
            - products: List of products if applicable
            - comparison: Comparison data if applicable
            - suggestions: Follow-up suggestions
            - confidence: Response confidence score
            - execution_time_ms: Response time
        """
        payload = {"query": query}
        if session_id:
            payload["session_id"] = session_id

        async with self._get_client() as client:
            response = await client.post("/chat/v2", json=payload)
            response.raise_for_status()
            return response.json()

    # ==========================================================================
    # Health Check
    # ==========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check API health."""
        try:
            async with self._get_client() as client:
                response = await client.get("/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


def get_api_client() -> APIClient:
    """Get API client instance."""
    return APIClient()
