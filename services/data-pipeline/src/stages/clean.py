"""Clean stage - normalize and prepare products for embedding."""

import re
from datetime import datetime
from typing import Any

import structlog

from src.models.product import RawProduct, CleanedProduct
from src.models.chunk import ProductChunk, build_chunks_from_product
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()


class CleanStage(BaseStage[RawProduct, CleanedProduct]):
    """Clean and normalize product data.

    Performs:
    - Text normalization (whitespace, encoding)
    - Short title generation
    - Product type extraction
    - Embedding text preparation
    - Optional chunk building for enrich mode
    """

    name = "clean"
    description = "Clean and normalize product data"

    def __init__(self, context: StageContext, build_chunks: bool = False):
        super().__init__(context)
        self.build_chunks = build_chunks
        self.chunks: list[ProductChunk] = []

    async def process_batch(self, batch: list[RawProduct]) -> list[CleanedProduct]:
        """Process a batch of raw products."""
        cleaned = []

        for product in batch:
            try:
                cleaned_product = self._clean_product(product)
                if cleaned_product:
                    cleaned.append(cleaned_product)

                    # Build chunks if enabled
                    if self.build_chunks:
                        chunks = self._build_chunks(cleaned_product)
                        self.chunks.extend(chunks)
            except Exception as e:
                logger.warning(
                    "product_clean_failed",
                    asin=product.asin,
                    error=str(e),
                )
                self.progress.failed += 1

        return cleaned

    def _clean_product(self, product: RawProduct) -> CleanedProduct | None:
        """Clean a single product."""
        # Create cleaned product with inherited fields
        cleaned_data = product.model_dump()

        # Normalize text fields
        cleaned_data["title"] = self._normalize_text(product.title)

        if product.brand:
            cleaned_data["brand"] = self._normalize_text(product.brand)

        if product.product_description:
            cleaned_data["product_description"] = self._normalize_text(
                product.product_description
            )

        # Generate short title
        cleaned_data["short_title"] = self._generate_short_title(
            product.title,
            product.brand,
        )

        # Extract product type
        cleaned_data["product_type"] = self._extract_product_type(
            product.title,
            product.category_name,
        )

        # Extract product type keywords
        cleaned_data["product_type_keywords"] = self._extract_keywords(
            product.title,
            product.category_name,
        )

        # Normalize category names
        if product.category_name:
            cleaned_data["category_level1"] = self._extract_category_level(
                product.category_name, 1
            )
            cleaned_data["category_level2"] = self._extract_category_level(
                product.category_name, 2
            )
            cleaned_data["category_level3"] = self._extract_category_level(
                product.category_name, 3
            )

        # Add metadata
        cleaned_data["cleaned_at"] = datetime.now()

        # Create CleanedProduct
        cleaned = CleanedProduct(**cleaned_data)

        # Build embedding text
        cleaned.embedding_text = cleaned.build_embedding_text()

        return cleaned

    def _normalize_text(self, text: str | None) -> str | None:
        """Normalize text content."""
        if not text:
            return None

        # Decode HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        return text if text else None

    def _generate_short_title(
        self,
        title: str,
        brand: str | None,
    ) -> str:
        """Generate a concise short title."""
        if not title:
            return ""

        short = title

        # Remove brand from beginning if present
        if brand:
            pattern = rf"^{re.escape(brand)}\s*[-,:]?\s*"
            short = re.sub(pattern, "", short, flags=re.IGNORECASE)

        # Remove common suffix patterns
        suffix_patterns = [
            r"\s*[-,]\s*\d+\s*(pack|piece|count|ct).*$",
            r'\s*[-,]\s*\d+(\.\d+)?\s*(inch|in|"|cm|mm).*$',
            r"\s*[-,]\s*(black|white|silver|gold|blue|red|gray|grey).*$",
            r"\s*\([^)]*\)\s*$",
            r"\s*[-,]\s*model\s*#?\s*\w+.*$",
        ]

        for pattern in suffix_patterns:
            short = re.sub(pattern, "", short, flags=re.IGNORECASE)

        # Truncate if too long
        if len(short) > 100:
            short = short[:97] + "..."

        return short.strip()

    def _extract_product_type(
        self,
        title: str,
        category: str | None,
    ) -> str | None:
        """Extract product type from title and category."""
        # Common product type patterns
        type_patterns = [
            r"\b(headphones?|earbuds?|earphones?|headset)\b",
            r"\b(laptop|notebook|computer|pc|desktop)\b",
            r"\b(phone|smartphone|tablet|ipad)\b",
            r"\b(camera|dslr|mirrorless|camcorder)\b",
            r"\b(tv|television|monitor|display)\b",
            r"\b(speaker|soundbar|subwoofer)\b",
            r"\b(keyboard|mouse|webcam|microphone)\b",
            r"\b(charger|cable|adapter|hub)\b",
            r"\b(case|cover|protector|stand)\b",
            r"\b(router|modem|switch|extender)\b",
            r"\b(printer|scanner|copier)\b",
            r"\b(drill|saw|hammer|screwdriver)\b",
            r"\b(vacuum|cleaner|mop|broom)\b",
            r"\b(blender|mixer|processor|juicer)\b",
        ]

        text = f"{title} {category or ''}"

        for pattern in type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # Fall back to first category level
        if category:
            parts = category.split(">") if ">" in category else category.split("/")
            if parts:
                return parts[0].strip().lower()

        return None

    def _extract_keywords(
        self,
        title: str,
        category: str | None,
    ) -> list[str]:
        """Extract keywords from title and category."""
        text = f"{title} {category or ''}"

        # Remove common stop words
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall", "can",
            "this", "that", "these", "those", "it", "its", "new", "free",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        keywords = [w for w in words if w not in stop_words]

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:10]  # Limit to 10 keywords

    def _extract_category_level(
        self,
        category: str,
        level: int,
    ) -> str | None:
        """Extract specific level from category hierarchy."""
        # Try different separators
        if ">" in category:
            parts = category.split(">")
        elif "/" in category:
            parts = category.split("/")
        elif "|" in category:
            parts = category.split("|")
        else:
            parts = [category]

        parts = [p.strip() for p in parts if p.strip()]

        if level <= len(parts):
            return parts[level - 1]
        return None

    def _build_chunks(self, product: CleanedProduct) -> list[ProductChunk]:
        """Build chunks for child nodes."""
        return build_chunks_from_product(
            asin=product.asin,
            title=product.title,
            brand=product.brand,
            category_level1=product.category_level1,
            category_level2=product.category_level2,
            price=product.price,
            stars=product.stars,
            img_url=product.img_url,
            description=product.product_description,
            features=product.features,
            specs=product.specifications,
        )

    def get_chunks(self) -> list[ProductChunk]:
        """Get all built chunks."""
        return self.chunks
