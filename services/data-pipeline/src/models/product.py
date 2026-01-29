"""Product data models for the pipeline."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RawProduct(BaseModel):
    """Raw product data from CSV extraction.

    This represents the product as read from the source CSV file
    with minimal transformations.
    """

    # Identity
    asin: str = Field(..., description="Amazon Standard Identification Number")
    title: str = Field(..., description="Product title")
    brand: str | None = None

    # Pricing
    price: float | None = None
    list_price: float | None = None
    original_price: float | None = None

    # Ratings
    stars: float | None = None
    reviews_count: int | None = None
    bought_in_last_month: int | None = None

    # Categorization
    category_name: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    category_level3: str | None = None

    # Flags
    is_best_seller: bool = False
    is_amazon_choice: bool = False
    prime_eligible: bool = False
    availability: str | None = None

    # Content
    product_description: str | None = None
    features: list[str] | None = None
    specifications: dict[str, Any] | None = None

    # URLs
    product_url: str | None = None
    img_url: str | None = None

    # Metadata
    row_number: int | None = None

    @field_validator("price", "list_price", "original_price", mode="before")
    @classmethod
    def parse_price(cls, v: Any) -> float | None:
        """Parse price from various formats."""
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Remove currency symbols and whitespace
            cleaned = v.replace("$", "").replace(",", "").strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @field_validator("stars", mode="before")
    @classmethod
    def parse_stars(cls, v: Any) -> float | None:
        """Parse star rating."""
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @field_validator("reviews_count", "bought_in_last_month", mode="before")
    @classmethod
    def parse_int(cls, v: Any) -> int | None:
        """Parse integer fields."""
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            cleaned = v.replace(",", "").strip()
            if not cleaned:
                return None
            try:
                return int(float(cleaned))
            except ValueError:
                return None
        return None

    @field_validator("is_best_seller", "is_amazon_choice", "prime_eligible", mode="before")
    @classmethod
    def parse_bool(cls, v: Any) -> bool:
        """Parse boolean fields."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "yes", "1", "t", "y")
        if isinstance(v, (int, float)):
            return bool(v)
        return False

    @field_validator("features", mode="before")
    @classmethod
    def parse_features(cls, v: Any) -> list[str] | None:
        """Parse features from various formats."""
        if v is None:
            return None
        if isinstance(v, list):
            return [str(f) for f in v if f]
        if isinstance(v, str):
            # Try to parse as JSON-like list
            if v.startswith("["):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Split by newlines or bullets
            features = []
            for line in v.split("\n"):
                line = line.strip().lstrip("-").lstrip("*").lstrip("").strip()
                if line:
                    features.append(line)
            return features if features else None
        return None


class CleanedProduct(RawProduct):
    """Cleaned and normalized product data.

    Extends RawProduct with:
    - Normalized text fields
    - Generated short_title
    - Product type classification
    - Combined text for embedding
    - GenAI enrichment fields
    """

    # GenAI cleaned identity
    short_title: str | None = None
    product_type: str | None = None
    product_type_keywords: list[str] | None = None

    # Combined text for embedding (generated during cleaning)
    embedding_text: str | None = None

    # Cleaning metadata
    cleaned_at: datetime | None = None

    # GenAI enrichment fields (using lowercase to match database columns)
    # Text fields
    genai_summary: str | None = None
    genai_primary_function: str | None = None
    genai_best_for: str | None = None
    genai_unique_selling_points: str | None = None
    genai_detailed_description: str | None = None
    genai_how_it_works: str | None = None
    genai_materials: str | None = None
    genai_technology_explained: str | None = None
    genai_feature_comparison: str | None = None
    genai_specs_summary: str | None = None
    genai_durability_feedback: str | None = None
    genai_value_for_money_feedback: str | None = None
    genai_sentiment_label: str | None = None

    # Numeric fields
    genai_value_score: int | None = None
    genai_sentiment_score: float | None = None

    # JSONB fields (stored as Python objects, serialized to JSON)
    genai_use_cases: list[str] | str | None = None
    genai_target_audience: list[str] | str | None = None
    genai_key_capabilities: list[str] | str | None = None
    genai_whats_included: list[str] | str | None = None
    genai_features_detailed: list[str] | str | None = None
    genai_standout_features: list[str] | str | None = None
    genai_specs_comparison_ready: list[str] | str | None = None
    genai_specs_limitations: list[str] | str | None = None
    genai_common_praises: list[str] | str | None = None
    genai_common_complaints: list[str] | str | None = None
    genai_use_case_scenarios: list[str] | str | None = None
    genai_ideal_user_profiles: list[str] | str | None = None
    genai_not_recommended_for: list[str] | str | None = None
    genai_problems_solved: list[str] | str | None = None
    genai_pros: list[str] | str | None = None
    genai_cons: list[str] | str | None = None

    genai_enriched_at: datetime | None = None

    def build_embedding_text(self) -> str:
        """Build text for embedding generation."""
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.brand:
            parts.append(f"Brand: {self.brand}")

        if self.category_level1:
            category = self.category_level1
            if self.category_level2:
                category += f" > {self.category_level2}"
            parts.append(f"Category: {category}")

        if self.product_description:
            parts.append(f"Description: {self.product_description[:500]}")

        if self.features:
            features_text = ", ".join(self.features[:5])
            parts.append(f"Features: {features_text}")

        return "\n".join(parts)


class EmbeddedProduct(CleanedProduct):
    """Product with embedding vector.

    Extends CleanedProduct with the embedding vector
    ready for loading into Qdrant.
    """

    # Embedding
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedded_at: datetime | None = None

    def to_payload(self) -> "ProductPayload":
        """Convert to Qdrant payload format."""
        return ProductPayload(
            asin=self.asin,
            title=self.title,
            short_title=self.short_title,
            brand=self.brand,
            product_type=self.product_type,
            price=self.price,
            list_price=self.list_price,
            stars=self.stars,
            reviews_count=self.reviews_count,
            bought_in_last_month=self.bought_in_last_month,
            category_level1=self.category_level1,
            category_level2=self.category_level2,
            category_level3=self.category_level3,
            is_best_seller=self.is_best_seller,
            is_amazon_choice=self.is_amazon_choice,
            prime_eligible=self.prime_eligible,
            product_url=self.product_url,
            img_url=self.img_url,
            node_type="parent",
        )


class ProductPayload(BaseModel):
    """Qdrant payload for product vectors.

    This is the payload stored alongside vectors in Qdrant,
    containing searchable and filterable fields.
    """

    # Identity
    asin: str
    title: str
    short_title: str | None = None
    brand: str | None = None
    product_type: str | None = None

    # Pricing (indexed for filtering)
    price: float | None = None
    list_price: float | None = None

    # Ratings (indexed for filtering)
    stars: float | None = None
    reviews_count: int | None = None
    bought_in_last_month: int | None = None

    # Categories (indexed for filtering)
    category_level1: str | None = None
    category_level2: str | None = None
    category_level3: str | None = None

    # Flags (indexed for filtering)
    is_best_seller: bool = False
    is_amazon_choice: bool = False
    prime_eligible: bool = False

    # URLs
    product_url: str | None = None
    img_url: str | None = None

    # GenAI fields (for enriched mode)
    genAI_summary: str | None = None
    genAI_best_for: str | None = None
    genAI_primary_function: str | None = None
    genAI_use_cases: list[str] | None = None
    genAI_key_capabilities: list[str] | None = None

    # Node type for parent/child discrimination
    node_type: str = "parent"
