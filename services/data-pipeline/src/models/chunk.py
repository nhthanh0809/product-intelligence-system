"""Chunk models for child nodes in the pipeline."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.models.enums import NodeType, SectionType


class ProductChunk(BaseModel):
    """Product chunk for child node indexing.

    Each product can have up to 5 child nodes (one per section).
    Child nodes enable section-targeted search (e.g., searching
    specifically within reviews or specifications).
    """

    # Identity
    chunk_id: str = Field(..., description="Unique chunk ID: {asin}_{section}")
    parent_asin: str = Field(..., description="Parent product ASIN")

    # Section info
    section: SectionType = Field(..., description="Section type")
    node_type: NodeType = Field(default=NodeType.CHILD)

    # Content
    content: str = Field(..., description="Section content text")
    content_preview: str | None = Field(
        default=None,
        description="First 200 chars of content for display",
    )

    # Inherited from parent (for filtering)
    title: str | None = None
    brand: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    price: float | None = None
    stars: float | None = None
    img_url: str | None = None

    # Embedding
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedded_at: datetime | None = None

    @classmethod
    def from_product(
        cls,
        asin: str,
        section: SectionType,
        content: str,
        title: str | None = None,
        brand: str | None = None,
        category_level1: str | None = None,
        category_level2: str | None = None,
        price: float | None = None,
        stars: float | None = None,
        img_url: str | None = None,
    ) -> "ProductChunk":
        """Create a chunk from product data."""
        return cls(
            chunk_id=f"{asin}_{section.value}",
            parent_asin=asin,
            section=section,
            content=content,
            content_preview=content[:200] if content else None,
            title=title,
            brand=brand,
            category_level1=category_level1,
            category_level2=category_level2,
            price=price,
            stars=stars,
            img_url=img_url,
        )

    def to_payload(self) -> "ChunkPayload":
        """Convert to Qdrant payload format."""
        return ChunkPayload(
            chunk_id=self.chunk_id,
            parent_asin=self.parent_asin,
            section=self.section.value,
            node_type=self.node_type.value,
            content_preview=self.content_preview,
            title=self.title,
            brand=self.brand,
            category_level1=self.category_level1,
            category_level2=self.category_level2,
            price=self.price,
            stars=self.stars,
            img_url=self.img_url,
        )


class ChunkPayload(BaseModel):
    """Qdrant payload for chunk vectors.

    This is the payload stored alongside child vectors in Qdrant.
    """

    # Identity
    chunk_id: str
    parent_asin: str
    section: str
    node_type: str = "child"

    # Content preview for display
    content_preview: str | None = None

    # Inherited from parent (for filtering and display)
    title: str | None = None
    brand: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    price: float | None = None
    stars: float | None = None
    img_url: str | None = None


def build_chunks_from_product(
    asin: str,
    title: str | None,
    brand: str | None,
    category_level1: str | None,
    category_level2: str | None,
    price: float | None,
    stars: float | None,
    img_url: str | None,
    description: str | None = None,
    features: list[str] | None = None,
    specs: dict | None = None,
    reviews_summary: str | None = None,
    use_cases: list[str] | None = None,
) -> list[ProductChunk]:
    """Build all available chunks from product data.

    Creates up to 5 child nodes:
    - description: Product description
    - features: Product features
    - specs: Technical specifications
    - reviews: Review summary
    - use_cases: Use cases and recommendations
    """
    chunks = []
    common_kwargs = {
        "title": title,
        "brand": brand,
        "category_level1": category_level1,
        "category_level2": category_level2,
        "price": price,
        "stars": stars,
        "img_url": img_url,
    }

    # Description chunk
    if description and len(description) > 50:
        chunks.append(
            ProductChunk.from_product(
                asin=asin,
                section=SectionType.DESCRIPTION,
                content=description,
                **common_kwargs,
            )
        )

    # Features chunk
    if features and len(features) > 0:
        features_text = "\n".join(f"- {f}" for f in features)
        chunks.append(
            ProductChunk.from_product(
                asin=asin,
                section=SectionType.FEATURES,
                content=features_text,
                **common_kwargs,
            )
        )

    # Specs chunk
    if specs and len(specs) > 0:
        specs_text = "\n".join(f"{k}: {v}" for k, v in specs.items())
        chunks.append(
            ProductChunk.from_product(
                asin=asin,
                section=SectionType.SPECS,
                content=specs_text,
                **common_kwargs,
            )
        )

    # Reviews chunk
    if reviews_summary and len(reviews_summary) > 50:
        chunks.append(
            ProductChunk.from_product(
                asin=asin,
                section=SectionType.REVIEWS,
                content=reviews_summary,
                **common_kwargs,
            )
        )

    # Use cases chunk
    if use_cases and len(use_cases) > 0:
        use_cases_text = "\n".join(f"- {u}" for u in use_cases)
        chunks.append(
            ProductChunk.from_product(
                asin=asin,
                section=SectionType.USE_CASES,
                content=use_cases_text,
                **common_kwargs,
            )
        )

    return chunks
