"""LLM Extract stage - use LLM to extract structured GenAI fields.

Uses the Ollama service to analyze product content and generate
structured fields like summary, best_for, use_cases, etc.
"""

import asyncio
import json
import re
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.models.product import CleanedProduct
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


# Extraction prompt template
EXTRACTION_PROMPT = """Analyze this product and extract structured information.

Product Information:
{content}

Extract the following fields in JSON format:
1. summary: A 2-3 sentence summary of what this product is and does
2. best_for: Who this product is best suited for (e.g., "home users", "professionals", "beginners")
3. primary_function: The main function/purpose in one phrase
4. use_cases: List of 3-5 specific use cases
5. key_capabilities: List of 3-5 key capabilities or features
6. pros: List of 2-3 advantages/strengths
7. cons: List of 1-2 potential drawbacks or limitations

Respond with ONLY valid JSON, no other text:
{{
  "summary": "...",
  "best_for": "...",
  "primary_function": "...",
  "use_cases": ["...", "..."],
  "key_capabilities": ["...", "..."],
  "pros": ["...", "..."],
  "cons": ["...", "..."]
}}"""


class LlmExtractStage(BaseStage[CleanedProduct, CleanedProduct]):
    """Extract GenAI fields using LLM.

    This stage:
    - Takes markdown content from previous stage
    - Sends to Ollama LLM for analysis
    - Extracts structured fields (summary, best_for, use_cases, etc.)
    - Updates products with GenAI enrichment
    """

    name = "llm_extract"
    description = "Extract GenAI fields using LLM"

    def __init__(
        self,
        context: StageContext,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        concurrent_requests: int = 2,
    ):
        super().__init__(context)
        self.settings = get_settings()
        # Use config values with parameter overrides
        self.model = model or self.settings.llm_model
        self.temperature = temperature if temperature is not None else self.settings.llm_temperature
        self.max_tokens = max_tokens or self.settings.llm_max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.concurrent_requests = concurrent_requests
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(concurrent_requests)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.ollama_service_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def _build_prompt(self, product: CleanedProduct, markdown: str | None) -> str:
        """Build the extraction prompt."""
        if markdown:
            content = markdown[:4000]  # Limit content length
        else:
            # Build content from product data
            parts = [f"Title: {product.title}"]
            if product.brand:
                parts.append(f"Brand: {product.brand}")
            if product.category_level1:
                parts.append(f"Category: {product.category_level1}")
            if product.product_description:
                parts.append(f"Description: {product.product_description[:1000]}")
            if product.features:
                parts.append(f"Features: {', '.join(product.features[:5])}")
            content = "\n".join(parts)

        return EXTRACTION_PROMPT.format(content=content)

    def _parse_llm_response(self, response_text: str) -> dict[str, Any] | None:
        """Parse LLM response to extract JSON."""
        try:
            # Try to find JSON in the response
            # First, try direct parse
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass

            # Look for JSON block in response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            logger.warning("no_json_in_response", response_preview=response_text[:200])
            return None

        except Exception as e:
            logger.warning("json_parse_failed", error=str(e))
            return None

    async def _call_llm(self, prompt: str) -> str | None:
        """Call the Ollama LLM service."""
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    client = await self._get_client()

                    response = await client.post(
                        "/generate",
                        json={
                            "prompt": prompt,
                            "model": self.model,
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return data.get("response", "")
                    else:
                        logger.warning(
                            "llm_call_failed",
                            status=response.status_code,
                            attempt=attempt + 1,
                        )

                except Exception as e:
                    logger.warning(
                        "llm_call_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        return None

    def _generate_fallback_extraction(self, product: CleanedProduct) -> dict[str, Any]:
        """Generate fallback extraction from product data when LLM is unavailable."""
        # Build summary from title and description
        summary = product.title
        if product.product_description:
            summary = f"{product.title}. {product.product_description[:200]}"

        # Infer best_for from category
        best_for = "general consumers"
        if product.category_level1:
            cat = product.category_level1.lower()
            if "industrial" in cat or "business" in cat:
                best_for = "business and industrial users"
            elif "office" in cat:
                best_for = "office workers and professionals"
            elif "home" in cat or "kitchen" in cat:
                best_for = "home users"
            elif "outdoor" in cat or "sports" in cat:
                best_for = "outdoor enthusiasts"

        # Extract use cases from features
        use_cases = []
        if product.features:
            use_cases = [f[:100] for f in product.features[:3]]

        # Key capabilities from features
        key_capabilities = []
        if product.features:
            key_capabilities = [f[:80] for f in product.features[:5]]

        return {
            "summary": summary[:500],
            "best_for": best_for,
            "primary_function": product.product_type or "general product",
            "use_cases": use_cases or ["general use"],
            "key_capabilities": key_capabilities or ["see product features"],
            "pros": ["quality product" if (product.stars or 0) >= 4 else "affordable option"],
            "cons": ["see reviews for details"],
        }

    async def _extract_for_product(
        self,
        product: CleanedProduct,
        markdown: str | None,
    ) -> dict[str, Any]:
        """Extract GenAI fields for a single product."""
        prompt = self._build_prompt(product, markdown)
        response = await self._call_llm(prompt)

        if response:
            extracted = self._parse_llm_response(response)
            if extracted:
                return extracted

        # Fallback: generate from product data
        logger.info("using_fallback_extraction", asin=product.asin)
        return self._generate_fallback_extraction(product)

    def _apply_extraction(
        self,
        product: CleanedProduct,
        extraction: dict[str, Any],
    ) -> CleanedProduct:
        """Apply extracted fields to product."""
        # Store extraction in context for later use
        genai_fields = {
            "genAI_summary": extraction.get("summary"),
            "genAI_best_for": extraction.get("best_for"),
            "genAI_primary_function": extraction.get("primary_function"),
            "genAI_use_cases": extraction.get("use_cases"),
            "genAI_key_capabilities": extraction.get("key_capabilities"),
            "genAI_pros": extraction.get("pros"),
            "genAI_cons": extraction.get("cons"),
        }

        # Store in context data for loading stages
        self.context.data.setdefault("genai_fields", {})[product.asin] = genai_fields

        return product

    async def process_batch(
        self,
        batch: list[CleanedProduct],
    ) -> list[CleanedProduct]:
        """Process a batch of products with concurrent LLM calls."""
        markdown_content = self.context.data.get("markdown_content", {})
        results = []

        # Process concurrently with semaphore limiting
        async def process_one(product: CleanedProduct) -> CleanedProduct:
            markdown = markdown_content.get(product.asin)
            extraction = await self._extract_for_product(product, markdown)
            return self._apply_extraction(product, extraction)

        tasks = [process_one(product) for product in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "product_extraction_failed",
                    asin=batch[i].asin,
                    error=str(result),
                )
                # Return original product on failure
                processed.append(batch[i])
            else:
                processed.append(result)

        return processed

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
