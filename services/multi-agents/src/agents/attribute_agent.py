"""Attribute Agent for structured attribute extraction.

The Attribute Agent is responsible for:
1. Extracting structured attributes from product data
2. Parsing specifications (battery, weight, dimensions)
3. Calculating derived metrics (value score, discount %)
4. Normalizing units (currency, weight, time)
5. Handling missing values with defaults or estimates
"""

import re
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, RetryConfig

logger = structlog.get_logger()


class UnitType(str, Enum):
    """Types of units for normalization."""

    CURRENCY = "currency"
    WEIGHT = "weight"
    LENGTH = "length"
    TIME = "time"
    STORAGE = "storage"
    NONE = "none"


class AttributeType(str, Enum):
    """Types of product attributes."""

    # Basic attributes
    PRICE = "price"
    RATING = "rating"
    REVIEW_COUNT = "review_count"
    BRAND = "brand"
    TITLE = "title"

    # Specifications
    BATTERY_LIFE = "battery_life"
    WEIGHT = "weight"
    DIMENSIONS = "dimensions"
    STORAGE = "storage"
    SCREEN_SIZE = "screen_size"

    # Derived metrics
    VALUE_SCORE = "value_score"
    POPULARITY_SCORE = "popularity_score"
    DISCOUNT_PCT = "discount_pct"
    PRICE_PER_RATING = "price_per_rating"


class ExtractedAttribute(BaseModel):
    """A single extracted attribute with metadata."""

    name: AttributeType = Field(description="Attribute type")
    value: Any = Field(description="Extracted value")
    raw_value: str | None = Field(
        default=None,
        description="Original raw value before processing",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measurement",
    )
    normalized_value: float | None = Field(
        default=None,
        description="Value normalized to standard unit",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in extraction (0-1)",
    )
    is_estimated: bool = Field(
        default=False,
        description="Whether value was estimated",
    )


class AttributeExtractionInput(BaseModel):
    """Input for attribute extraction."""

    products: list[dict[str, Any]] = Field(
        description="Products to extract attributes from"
    )
    attributes_to_extract: list[AttributeType] | None = Field(
        default=None,
        description="Specific attributes to extract (None = all)",
    )
    calculate_derived: bool = Field(
        default=True,
        description="Whether to calculate derived metrics",
    )
    normalize_units: bool = Field(
        default=True,
        description="Whether to normalize units",
    )


class ProductAttributes(BaseModel):
    """Extracted attributes for a single product."""

    product_id: str = Field(description="Product identifier")
    attributes: dict[str, ExtractedAttribute] = Field(
        default_factory=dict,
        description="Extracted attributes by name",
    )
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Original product data",
    )

    def get_value(self, attr: AttributeType, default: Any = None) -> Any:
        """Get attribute value with optional default."""
        attr_data = self.attributes.get(attr.value)
        if attr_data:
            return attr_data.normalized_value or attr_data.value
        return default


class AttributeExtractionOutput(BaseModel):
    """Output from attribute extraction."""

    products: list[ProductAttributes] = Field(
        description="Products with extracted attributes"
    )
    extraction_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics of extraction",
    )


class AttributeAgent(BaseAgent[AttributeExtractionInput, AttributeExtractionOutput]):
    """Agent for extracting and normalizing product attributes.

    The attribute agent processes raw product data to extract
    structured attributes, parse specifications, calculate
    derived metrics, and normalize units for comparison.
    """

    name = "attribute"
    description = "Extracts and normalizes product attributes"

    # Unit conversion factors (to base unit)
    WEIGHT_CONVERSIONS = {
        "kg": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "g": 0.001,
        "gram": 0.001,
        "grams": 0.001,
        "lb": 0.453592,
        "lbs": 0.453592,
        "pound": 0.453592,
        "pounds": 0.453592,
        "oz": 0.0283495,
        "ounce": 0.0283495,
        "ounces": 0.0283495,
    }

    LENGTH_CONVERSIONS = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "cm": 0.01,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "mm": 0.001,
        "millimeter": 0.001,
        "millimeters": 0.001,
        "in": 0.0254,
        "inch": 0.0254,
        "inches": 0.0254,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
    }

    TIME_CONVERSIONS = {
        "h": 1.0,
        "hr": 1.0,
        "hrs": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "min": 1/60,
        "mins": 1/60,
        "minute": 1/60,
        "minutes": 1/60,
        "day": 24.0,
        "days": 24.0,
    }

    STORAGE_CONVERSIONS = {
        "gb": 1.0,
        "gigabyte": 1.0,
        "gigabytes": 1.0,
        "tb": 1024.0,
        "terabyte": 1024.0,
        "terabytes": 1024.0,
        "mb": 0.001,
        "megabyte": 0.001,
        "megabytes": 0.001,
    }

    # Patterns for spec extraction
    BATTERY_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(?:hour|hr|h)s?\s*(?:battery|life)?",
        r"battery[:\s]+(\d+(?:\.\d+)?)\s*(?:hour|hr|h)s?",
        r"(\d+(?:\.\d+)?)\s*mah",  # mAh capacity
        r"up\s+to\s+(\d+(?:\.\d+)?)\s*(?:hour|hr|h)s?",
    ]

    WEIGHT_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz|ounce|pound|gram)s?",
        r"weight[:\s]+(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|oz)s?",
    ]

    DIMENSION_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(in|cm|mm)?",
        r"(\d+(?:\.\d+)?)\s*(in|cm|mm)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(in|cm|mm)?",
    ]

    STORAGE_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)",
        r"storage[:\s]+(\d+(?:\.\d+)?)\s*(tb|gb|mb)",
        r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\s*(?:ssd|hdd|storage)",
    ]

    SCREEN_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(?:inch|in|\")\s*(?:screen|display)?",
        r"screen[:\s]+(\d+(?:\.\d+)?)\s*(?:inch|in|\")?",
    ]

    def __init__(self):
        super().__init__(retry_config=RetryConfig(max_retries=1))

    async def _execute_internal(
        self,
        input_data: AttributeExtractionInput,
    ) -> AttributeExtractionOutput:
        """Extract attributes from products.

        Args:
            input_data: Input with products and extraction options

        Returns:
            AttributeExtractionOutput with extracted attributes
        """
        extracted_products = []
        stats = {
            "total_products": len(input_data.products),
            "attributes_extracted": 0,
            "attributes_estimated": 0,
            "failed_extractions": 0,
        }

        for product in input_data.products:
            try:
                product_attrs = self._extract_product_attributes(
                    product,
                    input_data.attributes_to_extract,
                    input_data.calculate_derived,
                    input_data.normalize_units,
                )
                extracted_products.append(product_attrs)

                # Update stats
                stats["attributes_extracted"] += len(product_attrs.attributes)
                stats["attributes_estimated"] += sum(
                    1 for a in product_attrs.attributes.values() if a.is_estimated
                )

            except Exception as e:
                logger.warning(
                    "attribute_extraction_failed",
                    product_id=product.get("asin", "unknown"),
                    error=str(e),
                )
                stats["failed_extractions"] += 1

                # Add product with minimal attributes
                extracted_products.append(ProductAttributes(
                    product_id=product.get("asin", product.get("id", "unknown")),
                    raw_data=product,
                ))

        return AttributeExtractionOutput(
            products=extracted_products,
            extraction_summary=stats,
        )

    def _extract_product_attributes(
        self,
        product: dict[str, Any],
        attributes_to_extract: list[AttributeType] | None,
        calculate_derived: bool,
        normalize_units: bool,
    ) -> ProductAttributes:
        """Extract attributes from a single product."""
        product_id = product.get("asin", product.get("id", "unknown"))
        attributes: dict[str, ExtractedAttribute] = {}

        # Determine which attributes to extract
        if attributes_to_extract:
            attrs_to_extract = set(attributes_to_extract)
        else:
            attrs_to_extract = set(AttributeType)

        # Extract basic attributes
        if AttributeType.PRICE in attrs_to_extract:
            price_attr = self._extract_price(product)
            if price_attr:
                attributes[AttributeType.PRICE.value] = price_attr

        if AttributeType.RATING in attrs_to_extract:
            rating_attr = self._extract_rating(product)
            if rating_attr:
                attributes[AttributeType.RATING.value] = rating_attr

        if AttributeType.REVIEW_COUNT in attrs_to_extract:
            review_attr = self._extract_review_count(product)
            if review_attr:
                attributes[AttributeType.REVIEW_COUNT.value] = review_attr

        if AttributeType.BRAND in attrs_to_extract:
            brand_attr = self._extract_brand(product)
            if brand_attr:
                attributes[AttributeType.BRAND.value] = brand_attr

        if AttributeType.TITLE in attrs_to_extract:
            title_attr = self._extract_title(product)
            if title_attr:
                attributes[AttributeType.TITLE.value] = title_attr

        # Extract specifications (from title, description, specs)
        spec_text = self._get_spec_text(product)

        if AttributeType.BATTERY_LIFE in attrs_to_extract:
            battery_attr = self._extract_battery_life(spec_text, normalize_units)
            if battery_attr:
                attributes[AttributeType.BATTERY_LIFE.value] = battery_attr

        if AttributeType.WEIGHT in attrs_to_extract:
            weight_attr = self._extract_weight(spec_text, normalize_units)
            if weight_attr:
                attributes[AttributeType.WEIGHT.value] = weight_attr

        if AttributeType.STORAGE in attrs_to_extract:
            storage_attr = self._extract_storage(spec_text, normalize_units)
            if storage_attr:
                attributes[AttributeType.STORAGE.value] = storage_attr

        if AttributeType.SCREEN_SIZE in attrs_to_extract:
            screen_attr = self._extract_screen_size(spec_text, normalize_units)
            if screen_attr:
                attributes[AttributeType.SCREEN_SIZE.value] = screen_attr

        # Calculate derived metrics
        if calculate_derived:
            if AttributeType.VALUE_SCORE in attrs_to_extract:
                value_attr = self._calculate_value_score(attributes)
                if value_attr:
                    attributes[AttributeType.VALUE_SCORE.value] = value_attr

            if AttributeType.POPULARITY_SCORE in attrs_to_extract:
                popularity_attr = self._calculate_popularity_score(attributes)
                if popularity_attr:
                    attributes[AttributeType.POPULARITY_SCORE.value] = popularity_attr

            if AttributeType.DISCOUNT_PCT in attrs_to_extract:
                discount_attr = self._extract_discount(product)
                if discount_attr:
                    attributes[AttributeType.DISCOUNT_PCT.value] = discount_attr

            if AttributeType.PRICE_PER_RATING in attrs_to_extract:
                ppr_attr = self._calculate_price_per_rating(attributes)
                if ppr_attr:
                    attributes[AttributeType.PRICE_PER_RATING.value] = ppr_attr

        return ProductAttributes(
            product_id=product_id,
            attributes=attributes,
            raw_data=product,
        )

    def _get_spec_text(self, product: dict[str, Any]) -> str:
        """Get combined text for specification extraction."""
        parts = []

        # Title
        if product.get("title"):
            parts.append(product["title"])

        # Description
        if product.get("description"):
            parts.append(product["description"])

        # Specifications/features
        if product.get("specifications"):
            if isinstance(product["specifications"], dict):
                parts.extend(f"{k}: {v}" for k, v in product["specifications"].items())
            elif isinstance(product["specifications"], list):
                parts.extend(str(s) for s in product["specifications"])

        if product.get("features"):
            if isinstance(product["features"], list):
                parts.extend(str(f) for f in product["features"])

        # Bullet points
        if product.get("bullet_points"):
            parts.extend(str(b) for b in product["bullet_points"])

        return " ".join(parts).lower()

    def _extract_price(self, product: dict[str, Any]) -> ExtractedAttribute | None:
        """Extract price attribute."""
        price = product.get("price")

        if price is None:
            return None

        # Handle string prices
        if isinstance(price, str):
            # Remove currency symbols and parse
            price_str = re.sub(r"[^\d.]", "", price)
            try:
                price = float(price_str)
            except ValueError:
                return None

        return ExtractedAttribute(
            name=AttributeType.PRICE,
            value=price,
            unit="USD",
            normalized_value=float(price),
            confidence=1.0,
        )

    def _extract_rating(self, product: dict[str, Any]) -> ExtractedAttribute | None:
        """Extract rating attribute."""
        rating = product.get("stars", product.get("rating"))

        if rating is None:
            return None

        if isinstance(rating, str):
            try:
                rating = float(rating.split()[0])
            except (ValueError, IndexError):
                return None

        return ExtractedAttribute(
            name=AttributeType.RATING,
            value=rating,
            unit="stars",
            normalized_value=float(rating) / 5.0,  # Normalize to 0-1
            confidence=1.0,
        )

    def _extract_review_count(self, product: dict[str, Any]) -> ExtractedAttribute | None:
        """Extract review count attribute."""
        reviews = product.get("review_count", product.get("reviews", product.get("num_reviews")))

        if reviews is None:
            return None

        if isinstance(reviews, str):
            # Parse "1,234 reviews" format
            reviews_str = re.sub(r"[^\d]", "", reviews)
            try:
                reviews = int(reviews_str)
            except ValueError:
                return None

        return ExtractedAttribute(
            name=AttributeType.REVIEW_COUNT,
            value=reviews,
            normalized_value=float(reviews),
            confidence=1.0,
        )

    def _extract_brand(self, product: dict[str, Any]) -> ExtractedAttribute | None:
        """Extract brand attribute."""
        brand = product.get("brand")

        if not brand:
            return None

        return ExtractedAttribute(
            name=AttributeType.BRAND,
            value=brand,
            confidence=1.0,
        )

    def _extract_title(self, product: dict[str, Any]) -> ExtractedAttribute | None:
        """Extract title attribute."""
        title = product.get("title", product.get("name"))

        if not title:
            return None

        return ExtractedAttribute(
            name=AttributeType.TITLE,
            value=title,
            confidence=1.0,
        )

    def _extract_battery_life(
        self,
        spec_text: str,
        normalize: bool,
    ) -> ExtractedAttribute | None:
        """Extract battery life from specifications."""
        for pattern in self.BATTERY_PATTERNS:
            match = re.search(pattern, spec_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))

                    # Handle mAh specially (estimate hours)
                    if "mah" in match.group(0).lower():
                        # Very rough estimate: 100mAh ~ 1 hour for wireless earbuds
                        normalized = value / 100 if normalize else value
                        unit = "mAh"
                    else:
                        normalized = value if normalize else None
                        unit = "hours"

                    return ExtractedAttribute(
                        name=AttributeType.BATTERY_LIFE,
                        value=value,
                        raw_value=match.group(0),
                        unit=unit,
                        normalized_value=normalized,
                        confidence=0.9,
                    )
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_weight(
        self,
        spec_text: str,
        normalize: bool,
    ) -> ExtractedAttribute | None:
        """Extract weight from specifications."""
        for pattern in self.WEIGHT_PATTERNS:
            match = re.search(pattern, spec_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Normalize to kg
                    if normalize:
                        factor = self.WEIGHT_CONVERSIONS.get(unit, 1.0)
                        normalized = value * factor
                    else:
                        normalized = None

                    return ExtractedAttribute(
                        name=AttributeType.WEIGHT,
                        value=value,
                        raw_value=match.group(0),
                        unit=unit,
                        normalized_value=normalized,
                        confidence=0.9,
                    )
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_storage(
        self,
        spec_text: str,
        normalize: bool,
    ) -> ExtractedAttribute | None:
        """Extract storage capacity from specifications."""
        for pattern in self.STORAGE_PATTERNS:
            match = re.search(pattern, spec_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Normalize to GB
                    if normalize:
                        factor = self.STORAGE_CONVERSIONS.get(unit, 1.0)
                        normalized = value * factor
                    else:
                        normalized = None

                    return ExtractedAttribute(
                        name=AttributeType.STORAGE,
                        value=value,
                        raw_value=match.group(0),
                        unit=unit,
                        normalized_value=normalized,
                        confidence=0.9,
                    )
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_screen_size(
        self,
        spec_text: str,
        normalize: bool,
    ) -> ExtractedAttribute | None:
        """Extract screen size from specifications."""
        for pattern in self.SCREEN_PATTERNS:
            match = re.search(pattern, spec_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))

                    return ExtractedAttribute(
                        name=AttributeType.SCREEN_SIZE,
                        value=value,
                        raw_value=match.group(0),
                        unit="inches",
                        normalized_value=value if normalize else None,
                        confidence=0.9,
                    )
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_discount(self, product: dict[str, Any]) -> ExtractedAttribute | None:
        """Extract discount percentage."""
        # Check for explicit discount
        discount = product.get("discount", product.get("discount_pct"))
        if discount is not None:
            if isinstance(discount, str):
                discount_str = re.sub(r"[^\d.]", "", discount)
                try:
                    discount = float(discount_str)
                except ValueError:
                    discount = None

            if discount is not None:
                return ExtractedAttribute(
                    name=AttributeType.DISCOUNT_PCT,
                    value=discount,
                    unit="%",
                    normalized_value=discount / 100,
                    confidence=1.0,
                )

        # Calculate from original/current price
        current = product.get("price")
        original = product.get("original_price", product.get("list_price"))

        if current and original:
            try:
                current_val = float(re.sub(r"[^\d.]", "", str(current)))
                original_val = float(re.sub(r"[^\d.]", "", str(original)))

                if original_val > current_val:
                    discount_pct = ((original_val - current_val) / original_val) * 100
                    return ExtractedAttribute(
                        name=AttributeType.DISCOUNT_PCT,
                        value=round(discount_pct, 1),
                        unit="%",
                        normalized_value=discount_pct / 100,
                        confidence=0.95,
                    )
            except ValueError:
                pass

        return None

    def _calculate_value_score(
        self,
        attributes: dict[str, ExtractedAttribute],
    ) -> ExtractedAttribute | None:
        """Calculate value score (rating-to-price ratio)."""
        price_attr = attributes.get(AttributeType.PRICE.value)
        rating_attr = attributes.get(AttributeType.RATING.value)

        if not price_attr or not rating_attr:
            return None

        price = price_attr.value
        rating = rating_attr.value

        if not price or price <= 0:
            return None

        # Value score: rating per $10 spent, normalized
        value_score = (rating / (price / 10)) * 10
        value_score = min(value_score, 100)  # Cap at 100

        return ExtractedAttribute(
            name=AttributeType.VALUE_SCORE,
            value=round(value_score, 1),
            normalized_value=value_score / 100,
            confidence=0.8,
            is_estimated=True,
        )

    def _calculate_popularity_score(
        self,
        attributes: dict[str, ExtractedAttribute],
    ) -> ExtractedAttribute | None:
        """Calculate popularity score from rating and reviews."""
        rating_attr = attributes.get(AttributeType.RATING.value)
        reviews_attr = attributes.get(AttributeType.REVIEW_COUNT.value)

        if not rating_attr or not reviews_attr:
            return None

        rating = rating_attr.value
        reviews = reviews_attr.value

        if reviews <= 0:
            return None

        # Bayesian average approach
        # Weight rating by log of review count
        import math
        weight = math.log10(reviews + 1)
        popularity = rating * weight

        # Normalize to 0-100 scale (assuming max ~5 * 6 = 30)
        popularity_score = min(popularity / 30 * 100, 100)

        return ExtractedAttribute(
            name=AttributeType.POPULARITY_SCORE,
            value=round(popularity_score, 1),
            normalized_value=popularity_score / 100,
            confidence=0.85,
            is_estimated=True,
        )

    def _calculate_price_per_rating(
        self,
        attributes: dict[str, ExtractedAttribute],
    ) -> ExtractedAttribute | None:
        """Calculate price per rating point."""
        price_attr = attributes.get(AttributeType.PRICE.value)
        rating_attr = attributes.get(AttributeType.RATING.value)

        if not price_attr or not rating_attr:
            return None

        price = price_attr.value
        rating = rating_attr.value

        if not rating or rating <= 0:
            return None

        ppr = price / rating

        return ExtractedAttribute(
            name=AttributeType.PRICE_PER_RATING,
            value=round(ppr, 2),
            unit="$/star",
            confidence=0.9,
            is_estimated=True,
        )


# Singleton instance
_attribute_agent: AttributeAgent | None = None


async def get_attribute_agent() -> AttributeAgent:
    """Get or create attribute agent singleton."""
    global _attribute_agent
    if _attribute_agent is None:
        _attribute_agent = AttributeAgent()
        await _attribute_agent.initialize()
    return _attribute_agent


async def extract_attributes(
    products: list[dict[str, Any]],
    attributes: list[AttributeType] | None = None,
    calculate_derived: bool = True,
) -> AttributeExtractionOutput:
    """Extract attributes from products.

    Convenience function for quick attribute extraction.

    Args:
        products: Products to process
        attributes: Specific attributes to extract (None = all)
        calculate_derived: Whether to calculate derived metrics

    Returns:
        AttributeExtractionOutput with extracted attributes
    """
    agent = await get_attribute_agent()
    input_data = AttributeExtractionInput(
        products=products,
        attributes_to_extract=attributes,
        calculate_derived=calculate_derived,
    )
    return await agent.execute(input_data)
