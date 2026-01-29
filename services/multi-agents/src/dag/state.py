"""Typed state definitions for LangGraph workflows.

All state classes use Pydantic for validation and TypedDict for LangGraph compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class QueryType(str, Enum):
    """Query type classification."""
    SEARCH = "search"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    PRICE = "price"
    TREND = "trend"
    RECOMMENDATION = "recommendation"
    QA = "qa"


class SearchIntent(str, Enum):
    """Search intent classification."""
    PRODUCT_LOOKUP = "product_lookup"
    FEATURE_SEARCH = "feature_search"
    CATEGORY_BROWSE = "category_browse"
    BRAND_SEARCH = "brand_search"
    PRICE_SEARCH = "price_search"
    RATING_SEARCH = "rating_search"
    COMPARISON_SEARCH = "comparison_search"
    GENERAL = "general"


class ComparisonMode(str, Enum):
    """Comparison mode."""
    DIRECT = "direct"  # Compare specific products
    CATEGORY = "category"  # Best in category
    FEATURE = "feature"  # Compare by feature
    VALUE = "value"  # Best value comparison


class AnalysisType(str, Enum):
    """Analysis type."""
    GENERAL = "general"
    PROS_CONS = "pros_cons"
    SENTIMENT = "sentiment"
    FEATURES = "features"
    QUALITY = "quality"


# =============================================================================
# Product Models
# =============================================================================


class ProductInfo(BaseModel):
    """Basic product information."""
    asin: str
    title: str
    brand: str | None = None
    price: float | None = None
    list_price: float | None = None
    stars: float | None = None
    reviews_count: int | None = None
    category: str | None = None
    img_url: str | None = None


class ProductWithScore(ProductInfo):
    """Product with relevance score."""
    score: float = 0.0
    match_reason: str | None = None


class ComparedProduct(ProductInfo):
    """Product with comparison details."""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    key_features: dict[str, Any] = Field(default_factory=dict)
    value_score: float | None = None


class AnalyzedProduct(ProductInfo):
    """Product with analysis details."""
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    common_themes: list[str] = Field(default_factory=list)
    feature_ratings: dict[str, float] = Field(default_factory=dict)


# =============================================================================
# Main Agent State (TypedDict for LangGraph)
# =============================================================================


class AgentState(TypedDict, total=False):
    """Main orchestrator state for LangGraph.

    Used by the main workflow to coordinate between agents.
    """
    # Input
    query: str
    session_id: str | None

    # Classification
    query_type: str | None
    confidence: float

    # Search results
    search_results: list[dict]
    search_state: dict | None

    # Agent outputs
    analysis: str | None
    comparison: dict | None
    price_analysis: dict | None
    trends: dict | None
    recommendations: list[dict]

    # Final output
    answer: str | None
    products: list[dict]

    # Metadata
    error: str | None
    steps: list[str]
    latency_ms: float
    filters: dict | None


# =============================================================================
# Sub-workflow States (Dataclasses for internal use)
# =============================================================================


@dataclass
class SearchState:
    """State for search sub-workflow."""
    query: str
    context: dict = field(default_factory=dict)

    # Detection
    intent: SearchIntent | None = None
    query_type: QueryType | None = None

    # Filters
    category: str | None = None
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    min_rating: float | None = None

    # Strategy
    search_strategy: str = "hybrid"
    keyword_weight: float = 0.6
    semantic_weight: float = 0.4

    # Results
    raw_results: list[dict] = field(default_factory=list)
    formatted_results: list[ProductWithScore] = field(default_factory=list)
    total_results: int = 0

    # Output
    summary: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for state passing."""
        return {
            "query": self.query,
            "intent": self.intent.value if self.intent else None,
            "query_type": self.query_type.value if self.query_type else None,
            "strategy": self.search_strategy,
            "total_results": self.total_results,
            "summary": self.summary,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ComparisonState:
    """State for comparison sub-workflow."""
    query: str
    mode: ComparisonMode = ComparisonMode.DIRECT

    # Input products
    product_names: list[str] = field(default_factory=list)
    product_asins: list[str] = field(default_factory=list)

    # Fetched products
    products: list[ComparedProduct] = field(default_factory=list)

    # Analysis
    comparison_matrix: dict[str, dict[str, Any]] = field(default_factory=dict)
    key_differences: list[str] = field(default_factory=list)

    # Results
    winner: ComparedProduct | None = None
    winner_reason: str | None = None
    best_value: ComparedProduct | None = None
    comparison_summary: str | None = None

    # Metadata
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for state passing."""
        return {
            "mode": self.mode.value,
            "products": [p.model_dump() for p in self.products],
            "key_differences": self.key_differences,
            "winner": self.winner.model_dump() if self.winner else None,
            "winner_reason": self.winner_reason,
            "best_value": self.best_value.model_dump() if self.best_value else None,
            "summary": self.comparison_summary,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AnalysisState:
    """State for analysis sub-workflow."""
    query: str
    analysis_type: AnalysisType = AnalysisType.GENERAL

    # Input
    product_asin: str | None = None
    product_name: str | None = None

    # Fetched data
    product: AnalyzedProduct | None = None
    reviews: list[dict] = field(default_factory=list)
    review_count: int = 0

    # Analysis results
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    common_themes: list[str] = field(default_factory=list)
    feature_analysis: dict[str, Any] = field(default_factory=dict)

    # Output
    summary: str | None = None
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for state passing."""
        return {
            "analysis_type": self.analysis_type.value,
            "product": self.product.model_dump() if self.product else None,
            "review_count": self.review_count,
            "pros": self.pros,
            "cons": self.cons,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "common_themes": self.common_themes,
            "feature_analysis": self.feature_analysis,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "latency_ms": self.latency_ms,
        }


@dataclass
class PriceState:
    """State for price analysis sub-workflow."""
    query: str

    # Input
    target_price: float | None = None
    category: str | None = None

    # Products
    products: list[dict] = field(default_factory=list)

    # Analysis
    price_range: tuple[float, float] | None = None
    average_price: float | None = None
    deals: list[dict] = field(default_factory=list)
    best_deal: dict | None = None
    best_value: dict | None = None

    # Output
    recommendation: str | None = None
    summary: str | None = None

    # Metadata
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class TrendState:
    """State for trend analysis sub-workflow."""
    query: str
    category: str | None = None
    time_range: str = "7d"

    # Results
    trending_products: list[dict] = field(default_factory=list)
    hot_categories: list[str] = field(default_factory=list)
    rising_brands: list[str] = field(default_factory=list)

    # Analysis
    insights: list[str] = field(default_factory=list)
    summary: str | None = None

    # Metadata
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class RecommendState:
    """State for recommendation sub-workflow."""
    query: str

    # Input
    source_asin: str | None = None
    recommendation_type: str = "similar"  # similar, alternatives, accessories

    # Source product
    source_product: dict | None = None

    # Results
    recommendations: list[dict] = field(default_factory=list)

    # Output
    summary: str | None = None

    # Metadata
    latency_ms: float = 0.0
    error: str | None = None


# =============================================================================
# State Conversion Utilities
# =============================================================================


def merge_states(main_state: AgentState, sub_state: dict, prefix: str = "") -> AgentState:
    """Merge sub-workflow state into main state."""
    for key, value in sub_state.items():
        if prefix:
            main_state[f"{prefix}_{key}"] = value
        elif key not in main_state or main_state[key] is None:
            main_state[key] = value
    return main_state


def extract_products(state: AgentState) -> list[dict]:
    """Extract all products from state."""
    products = state.get("products", [])

    # Add from search results
    if state.get("search_results"):
        for result in state["search_results"]:
            if result not in products:
                products.append(result)

    # Add from recommendations
    if state.get("recommendations"):
        for rec in state["recommendations"]:
            if rec not in products:
                products.append(rec)

    return products[:20]  # Limit to 20 products
