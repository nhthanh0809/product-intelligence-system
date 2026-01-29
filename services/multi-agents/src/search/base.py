"""
Base classes for search strategies.

This module defines:
- BaseSearchStrategy: Abstract base class for all search strategies
- SearchResult: Individual search result
- SearchResponse: Search response with metadata
- SearchFilters: Filter parameters for search
- StrategySettings: Configuration schema for strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class StrategyType(str, Enum):
    """Types of search strategies."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SECTION = "section"


class SearchResult(BaseModel):
    """Individual search result."""
    asin: str
    title: str
    score: float
    source: str  # Strategy name that produced this result
    price: float | None = None
    stars: float | None = None
    brand: str | None = None
    category: str | None = None
    img_url: str | None = None
    genAI_summary: str | None = None
    genAI_best_for: str | None = None
    section: str | None = None  # For section search results
    content_preview: str | None = None  # Section content preview
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)


class SearchResponse(BaseModel):
    """Search response with metadata."""
    results: list[SearchResult]
    query: str
    query_type: str
    strategy_used: str
    total_found: int = 0
    latency_ms: float = 0.0
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "query_type": self.query_type,
            "strategy_used": self.strategy_used,
            "total_found": self.total_found,
            "latency_ms": self.latency_ms,
            "filters_applied": self.filters_applied,
            "metadata": self.metadata,
        }


@dataclass
class SearchFilters:
    """Filters to apply to search queries."""
    brand: str | None = None
    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None
    section: str | None = None
    exclude_asins: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.brand:
            result["brand"] = self.brand
        if self.category:
            result["category"] = self.category
        if self.min_price is not None:
            result["min_price"] = self.min_price
        if self.max_price is not None:
            result["max_price"] = self.max_price
        if self.min_rating is not None:
            result["min_rating"] = self.min_rating
        if self.section:
            result["section"] = self.section
        if self.exclude_asins:
            result["exclude_asins"] = self.exclude_asins
        return result

    def has_filters(self) -> bool:
        """Check if any filters are set."""
        return bool(self.to_dict())


@dataclass
class StrategySettings:
    """Configuration settings for a search strategy."""
    # General settings
    default_limit: int = 10
    fetch_multiplier: int = 5  # Fetch more than limit for post-filtering
    min_score: float = 0.0  # Minimum score threshold

    # Keyword search settings
    keyword_boost_title: float = 10.0
    keyword_boost_short_title: float = 8.0
    keyword_boost_brand: float = 5.0
    keyword_boost_model: float = 20.0
    keyword_fuzziness: str = "AUTO"

    # Semantic search settings
    semantic_score_threshold: float = 0.5

    # Hybrid settings
    hybrid_keyword_weight: float = 0.65
    hybrid_semantic_weight: float = 0.35
    hybrid_rrf_k: int = 40  # RRF parameter

    # Boost settings
    model_boost_factor: float = 1.6
    brand_boost_factor: float = 1.3

    # Reranking settings
    enable_reranking: bool = True  # Enable reranking by default

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategySettings":
        """Create settings from dictionary."""
        return cls(
            default_limit=data.get("default_limit", 10),
            fetch_multiplier=data.get("fetch_multiplier", 5),
            min_score=data.get("min_score", 0.0),
            keyword_boost_title=data.get("keyword_boost_title", 10.0),
            keyword_boost_short_title=data.get("keyword_boost_short_title", 8.0),
            keyword_boost_brand=data.get("keyword_boost_brand", 5.0),
            keyword_boost_model=data.get("keyword_boost_model", 20.0),
            keyword_fuzziness=data.get("keyword_fuzziness", "AUTO"),
            semantic_score_threshold=data.get("semantic_score_threshold", 0.5),
            hybrid_keyword_weight=data.get("hybrid_keyword_weight", 0.65),
            hybrid_semantic_weight=data.get("hybrid_semantic_weight", 0.35),
            hybrid_rrf_k=data.get("hybrid_rrf_k", 40),
            model_boost_factor=data.get("model_boost_factor", 1.6),
            brand_boost_factor=data.get("brand_boost_factor", 1.3),
            enable_reranking=data.get("enable_reranking", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_limit": self.default_limit,
            "fetch_multiplier": self.fetch_multiplier,
            "min_score": self.min_score,
            "keyword_boost_title": self.keyword_boost_title,
            "keyword_boost_short_title": self.keyword_boost_short_title,
            "keyword_boost_brand": self.keyword_boost_brand,
            "keyword_boost_model": self.keyword_boost_model,
            "keyword_fuzziness": self.keyword_fuzziness,
            "semantic_score_threshold": self.semantic_score_threshold,
            "hybrid_keyword_weight": self.hybrid_keyword_weight,
            "hybrid_semantic_weight": self.hybrid_semantic_weight,
            "hybrid_rrf_k": self.hybrid_rrf_k,
            "model_boost_factor": self.model_boost_factor,
            "brand_boost_factor": self.brand_boost_factor,
        }


class BaseSearchStrategy(ABC):
    """Abstract base class for search strategies.

    All search strategies must inherit from this class and implement
    the search() method.

    Attributes:
        name: Unique identifier for the strategy
        strategy_type: Type of strategy (keyword, semantic, hybrid, section)
        settings: Configuration settings
        clients: Shared database clients
    """

    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        settings: StrategySettings | None = None,
        clients: Any = None,  # SearchClients, but avoid circular import
    ):
        """Initialize the strategy.

        Args:
            name: Unique identifier for this strategy
            strategy_type: Type of search strategy
            settings: Configuration settings (uses defaults if None)
            clients: Shared database clients
        """
        self.name = name
        self.strategy_type = strategy_type
        self.settings = settings or StrategySettings()
        self._clients = clients
        self._initialized = False

    @property
    def clients(self):
        """Get the search clients."""
        return self._clients

    @clients.setter
    def clients(self, value):
        """Set the search clients."""
        self._clients = value

    @property
    def is_initialized(self) -> bool:
        """Check if strategy is initialized."""
        return self._initialized and self._clients is not None

    async def initialize(self, clients: Any = None) -> None:
        """Initialize the strategy.

        Args:
            clients: SearchClients instance (optional, uses existing if not provided)
        """
        if clients:
            self._clients = clients
        self._initialized = True
        logger.debug(
            "search_strategy_initialized",
            strategy=self.name,
            type=self.strategy_type.value,
        )

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
        **kwargs,
    ) -> SearchResponse:
        """Execute search with this strategy.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            filters: Optional filters to apply
            **kwargs: Additional strategy-specific parameters

        Returns:
            SearchResponse with results and metadata
        """
        pass

    def _create_response(
        self,
        results: list[SearchResult],
        query: str,
        query_type: str,
        latency_ms: float,
        filters: SearchFilters | None = None,
        **metadata,
    ) -> SearchResponse:
        """Create a standard search response.

        Args:
            results: List of search results
            query: Original query
            query_type: Detected query type
            latency_ms: Search latency in milliseconds
            filters: Applied filters
            **metadata: Additional metadata

        Returns:
            SearchResponse object
        """
        return SearchResponse(
            results=results,
            query=query,
            query_type=query_type,
            strategy_used=self.name,
            total_found=len(results),
            latency_ms=latency_ms,
            filters_applied=filters.to_dict() if filters else {},
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, type={self.strategy_type.value})"
