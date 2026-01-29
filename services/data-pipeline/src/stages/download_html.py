"""Download HTML stage - fetch product pages for enrichment.

Note: This stage is designed to work with cached HTML or mock data
for development/testing. In production, you would integrate with
a proper web scraping service or data provider API.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.models.product import CleanedProduct
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()

# Default cache directory
DEFAULT_CACHE_DIR = "/tmp/html_cache"


class DownloadHtmlStage(BaseStage[CleanedProduct, CleanedProduct]):
    """Download HTML content for products.

    This stage fetches product HTML pages for enrichment.
    It supports:
    - Local file caching to avoid re-downloading
    - Rate limiting to respect server limits
    - Retry logic with exponential backoff
    - Mock mode for testing without network access
    """

    name = "download"
    description = "Download product HTML pages"

    def __init__(
        self,
        context: StageContext,
        cache_dir: str | None = None,
        rate_limit: float = 1.0,  # Requests per second
        mock_mode: bool = True,  # Use mock data by default
        timeout: float = 30.0,
    ):
        super().__init__(context)
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.mock_mode = mock_mode
        self.timeout = timeout
        self._last_request_time = 0.0
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
            )
        return self._client

    def _get_cache_path(self, asin: str) -> Path:
        """Get cache file path for an ASIN."""
        return self.cache_dir / f"{asin}.html"

    def _load_from_cache(self, asin: str) -> str | None:
        """Load HTML from cache if available."""
        cache_path = self._get_cache_path(asin)
        if cache_path.exists():
            try:
                return cache_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("cache_read_failed", asin=asin, error=str(e))
        return None

    def _save_to_cache(self, asin: str, html: str) -> None:
        """Save HTML to cache."""
        try:
            cache_path = self._get_cache_path(asin)
            cache_path.write_text(html, encoding="utf-8")
        except Exception as e:
            logger.warning("cache_write_failed", asin=asin, error=str(e))

    def _generate_mock_html(self, product: CleanedProduct) -> str:
        """Generate mock HTML content from product data.

        This creates realistic HTML structure that can be processed
        by the html_to_markdown stage.
        """
        features_html = ""
        if product.features:
            features_list = "\n".join(f"<li>{f}</li>" for f in product.features[:10])
            features_html = f"<ul class='features'>{features_list}</ul>"

        specs_html = ""
        if product.specifications:
            specs_rows = "\n".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in list(product.specifications.items())[:20]
            )
            specs_html = f"<table class='specs'>{specs_rows}</table>"

        description = product.product_description or ""

        html = f"""
<!DOCTYPE html>
<html>
<head><title>{product.title}</title></head>
<body>
<div id="product-page">
    <h1 id="productTitle">{product.title}</h1>

    <div id="brand">
        <span>Brand: {product.brand or 'Unknown'}</span>
    </div>

    <div id="price">
        <span class="price">${product.price or 0:.2f}</span>
    </div>

    <div id="rating">
        <span class="stars">{product.stars or 0} out of 5 stars</span>
        <span class="reviews">{product.reviews_count or 0} reviews</span>
    </div>

    <div id="feature-bullets">
        <h2>About this item</h2>
        {features_html}
    </div>

    <div id="productDescription">
        <h2>Product Description</h2>
        <p>{description[:2000] if description else 'No description available.'}</p>
    </div>

    <div id="specifications">
        <h2>Technical Specifications</h2>
        {specs_html}
    </div>

    <div id="category">
        <span>Category: {product.category_level1 or ''}</span>
        {f' > {product.category_level2}' if product.category_level2 else ''}
    </div>
</div>
</body>
</html>
"""
        return html

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.rate_limit <= 0:
            return

        import time
        now = time.time()
        min_interval = 1.0 / self.rate_limit
        elapsed = now - self._last_request_time

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    async def _download_html(self, url: str, asin: str) -> str | None:
        """Download HTML from URL with retries."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                await self._rate_limit()
                client = await self._get_client()
                response = await client.get(url)

                if response.status_code == 200:
                    html = response.text
                    self._save_to_cache(asin, html)
                    return html
                elif response.status_code == 404:
                    logger.warning("product_not_found", asin=asin, url=url)
                    return None
                else:
                    logger.warning(
                        "download_failed",
                        asin=asin,
                        status=response.status_code,
                        attempt=attempt + 1,
                    )

            except Exception as e:
                logger.warning(
                    "download_error",
                    asin=asin,
                    error=str(e),
                    attempt=attempt + 1,
                )

            # Exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return None

    async def process_batch(
        self,
        batch: list[CleanedProduct],
    ) -> list[CleanedProduct]:
        """Process a batch of products."""
        results = []

        for product in batch:
            # Check cache first
            html = self._load_from_cache(product.asin)

            if html is None:
                if self.mock_mode:
                    # Generate mock HTML from product data
                    html = self._generate_mock_html(product)
                    self._save_to_cache(product.asin, html)
                elif product.product_url:
                    # Download from URL
                    html = await self._download_html(product.product_url, product.asin)

            if html:
                # Store HTML in context for next stage
                self.context.data.setdefault("html_content", {})[product.asin] = html
                results.append(product)
            else:
                logger.warning("no_html_available", asin=product.asin)
                # Still include product even without HTML
                results.append(product)

        return results

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
