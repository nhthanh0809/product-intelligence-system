"""HTML to Markdown stage - convert product HTML to clean markdown.

Extracts and structures product information from HTML pages
into clean markdown format for LLM processing.
"""

import re
from typing import Any

import structlog

from src.models.product import CleanedProduct
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


class HtmlToMarkdownStage(BaseStage[CleanedProduct, CleanedProduct]):
    """Convert HTML content to clean markdown.

    This stage:
    - Parses HTML product pages
    - Extracts key sections (description, features, specs)
    - Converts to clean markdown format
    - Stores structured content for LLM processing
    """

    name = "html_to_md"
    description = "Convert HTML to markdown"

    def __init__(
        self,
        context: StageContext,
        max_content_length: int = 8000,
    ):
        super().__init__(context)
        self.max_content_length = max_content_length

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def _extract_text_from_tag(self, html: str, tag_pattern: str) -> str:
        """Extract text content from HTML tag pattern."""
        match = re.search(tag_pattern, html, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1) if match.lastindex else match.group(0)
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            return self._clean_text(content)
        return ""

    def _extract_list_items(self, html: str, section_pattern: str) -> list[str]:
        """Extract list items from an HTML section."""
        section_match = re.search(section_pattern, html, re.DOTALL | re.IGNORECASE)
        if not section_match:
            return []

        section_html = section_match.group(0)
        items = re.findall(r'<li[^>]*>(.*?)</li>', section_html, re.DOTALL | re.IGNORECASE)

        cleaned_items = []
        for item in items:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', item)
            text = self._clean_text(text)
            if text and len(text) > 5:  # Filter out very short items
                cleaned_items.append(text)

        return cleaned_items[:20]  # Limit to 20 items

    def _extract_table_data(self, html: str, section_pattern: str) -> dict[str, str]:
        """Extract table data as key-value pairs."""
        section_match = re.search(section_pattern, html, re.DOTALL | re.IGNORECASE)
        if not section_match:
            return {}

        section_html = section_match.group(0)

        # Find table rows
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', section_html, re.DOTALL | re.IGNORECASE)

        data = {}
        for row in rows:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
            if len(cells) >= 2:
                key = self._clean_text(re.sub(r'<[^>]+>', '', cells[0]))
                value = self._clean_text(re.sub(r'<[^>]+>', '', cells[1]))
                if key and value:
                    data[key] = value

        return dict(list(data.items())[:30])  # Limit to 30 specs

    def _parse_html(self, html: str, product: CleanedProduct) -> dict[str, Any]:
        """Parse HTML and extract structured content."""
        content = {
            "title": "",
            "brand": "",
            "description": "",
            "features": [],
            "specifications": {},
            "category": "",
            "rating_info": "",
        }

        # Extract title
        content["title"] = (
            self._extract_text_from_tag(html, r'<h1[^>]*id="productTitle"[^>]*>(.*?)</h1>')
            or self._extract_text_from_tag(html, r'<h1[^>]*>(.*?)</h1>')
            or product.title
        )

        # Extract brand
        content["brand"] = (
            self._extract_text_from_tag(html, r'<div[^>]*id="brand"[^>]*>(.*?)</div>')
            or product.brand
            or ""
        )

        # Extract description
        description = self._extract_text_from_tag(
            html, r'<div[^>]*id="productDescription"[^>]*>(.*?)</div>'
        )
        if not description:
            description = self._extract_text_from_tag(
                html, r'<div[^>]*class="[^"]*description[^"]*"[^>]*>(.*?)</div>'
            )
        content["description"] = description or product.product_description or ""

        # Extract features
        features = self._extract_list_items(html, r'<div[^>]*id="feature-bullets"[^>]*>.*?<ul[^>]*>(.*?)</ul>')
        if not features:
            features = self._extract_list_items(html, r'<ul[^>]*class="[^"]*features[^"]*"[^>]*>(.*?)</ul>')
        content["features"] = features or product.features or []

        # Extract specifications
        specs = self._extract_table_data(html, r'<div[^>]*id="specifications"[^>]*>.*?<table[^>]*>(.*?)</table>')
        if not specs:
            specs = self._extract_table_data(html, r'<table[^>]*class="[^"]*specs[^"]*"[^>]*>(.*?)</table>')
        content["specifications"] = specs or product.specifications or {}

        # Extract category
        content["category"] = (
            self._extract_text_from_tag(html, r'<div[^>]*id="category"[^>]*>(.*?)</div>')
            or product.category_level1
            or ""
        )

        # Extract rating info
        content["rating_info"] = self._extract_text_from_tag(
            html, r'<div[^>]*id="rating"[^>]*>(.*?)</div>'
        )

        return content

    def _to_markdown(self, content: dict[str, Any]) -> str:
        """Convert structured content to markdown."""
        parts = []

        # Title
        if content.get("title"):
            parts.append(f"# {content['title']}")

        # Brand and Category
        meta = []
        if content.get("brand"):
            meta.append(f"**Brand:** {content['brand']}")
        if content.get("category"):
            meta.append(f"**Category:** {content['category']}")
        if content.get("rating_info"):
            meta.append(f"**Rating:** {content['rating_info']}")
        if meta:
            parts.append("\n".join(meta))

        # Description
        if content.get("description"):
            parts.append("## Description")
            # Truncate long descriptions
            desc = content["description"][:3000]
            parts.append(desc)

        # Features
        if content.get("features"):
            parts.append("## Key Features")
            for feature in content["features"]:
                parts.append(f"- {feature}")

        # Specifications
        if content.get("specifications"):
            parts.append("## Specifications")
            for key, value in content["specifications"].items():
                parts.append(f"- **{key}:** {value}")

        markdown = "\n\n".join(parts)

        # Truncate if too long
        if len(markdown) > self.max_content_length:
            markdown = markdown[:self.max_content_length] + "\n\n[Content truncated]"

        return markdown

    async def process_batch(
        self,
        batch: list[CleanedProduct],
    ) -> list[CleanedProduct]:
        """Process a batch of products."""
        results = []
        html_content = self.context.data.get("html_content", {})

        for product in batch:
            html = html_content.get(product.asin)

            if html:
                # Parse HTML and extract content
                content = self._parse_html(html, product)

                # Convert to markdown
                markdown = self._to_markdown(content)

                # Store markdown in context for LLM stage
                self.context.data.setdefault("markdown_content", {})[product.asin] = markdown

                # Also store structured content
                self.context.data.setdefault("structured_content", {})[product.asin] = content

                logger.debug(
                    "html_converted",
                    asin=product.asin,
                    markdown_length=len(markdown),
                )
            else:
                # No HTML available, create markdown from product data
                content = {
                    "title": product.title,
                    "brand": product.brand or "",
                    "description": product.product_description or "",
                    "features": product.features or [],
                    "specifications": product.specifications or {},
                    "category": product.category_level1 or "",
                    "rating_info": f"{product.stars or 0} stars, {product.reviews_count or 0} reviews",
                }
                markdown = self._to_markdown(content)
                self.context.data.setdefault("markdown_content", {})[product.asin] = markdown
                self.context.data.setdefault("structured_content", {})[product.asin] = content

            results.append(product)

        return results
