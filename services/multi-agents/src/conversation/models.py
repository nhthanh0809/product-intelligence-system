"""Enhanced conversation models with entity tracking.

Provides models for:
- Message storage with metadata
- Conversation context with product/entity tracking
- Entity references for pronoun resolution
- Resolved references for query rewriting
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class ReferenceType(str, Enum):
    """Types of references that can be resolved."""
    SINGULAR = "singular"      # it, this, that
    PLURAL = "plural"          # them, these, those
    POSSESSIVE = "possessive"  # its, their
    ORDINAL = "ordinal"        # first, second, last
    SUPERLATIVE = "superlative"  # cheapest, best, most expensive
    DEMONSTRATIVE = "demonstrative"  # this one, that product


@dataclass
class EntityReference:
    """A reference to an entity in conversation context."""
    reference_type: ReferenceType
    original_text: str  # The original pronoun/reference text
    position: int  # Position in query string

    # Resolution metadata
    resolved: bool = False
    resolved_to: str | None = None  # The resolved entity text
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_type": self.reference_type.value,
            "original_text": self.original_text,
            "position": self.position,
            "resolved": self.resolved,
            "resolved_to": self.resolved_to,
            "confidence": self.confidence,
        }


@dataclass
class ResolvedReference:
    """Result of resolving a reference."""
    original: str  # Original reference text
    resolved: str  # Resolved entity/product
    reference_type: ReferenceType
    confidence: float
    source: str  # "rule" or "llm"

    # Context used for resolution
    product_asin: str | None = None
    product_title: str | None = None


@dataclass
class Message:
    """A single conversation message with enhanced metadata."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Enhanced metadata
    message_id: str = field(default_factory=lambda: str(uuid4()))
    intent: str | None = None  # Detected intent for this message
    entities_mentioned: list[str] = field(default_factory=list)  # Products/brands mentioned
    references_resolved: list[dict] = field(default_factory=list)  # Resolved pronouns

    # Response metadata (for assistant messages)
    products_returned: list[str] = field(default_factory=list)  # ASINs returned
    agent_used: str | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "entities_mentioned": self.entities_mentioned,
            "references_resolved": self.references_resolved,
            "products_returned": self.products_returned,
            "agent_used": self.agent_used,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            message_id=data.get("message_id", str(uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            intent=data.get("intent"),
            entities_mentioned=data.get("entities_mentioned", []),
            references_resolved=data.get("references_resolved", []),
            products_returned=data.get("products_returned", []),
            agent_used=data.get("agent_used"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )


@dataclass
class ProductContext:
    """Context about a product mentioned in conversation."""
    asin: str
    title: str
    brand: str | None = None
    price: float | None = None
    stars: float | None = None

    # When this product was mentioned
    first_mentioned_turn: int = 0
    last_mentioned_turn: int = 0
    mention_count: int = 1

    # Position in last search results (for ordinal references)
    last_position: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asin": self.asin,
            "title": self.title,
            "brand": self.brand,
            "price": self.price,
            "stars": self.stars,
            "first_mentioned_turn": self.first_mentioned_turn,
            "last_mentioned_turn": self.last_mentioned_turn,
            "mention_count": self.mention_count,
            "last_position": self.last_position,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute by name (dict-like access for compatibility)."""
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductContext":
        return cls(
            asin=data["asin"],
            title=data["title"],
            brand=data.get("brand"),
            price=data.get("price"),
            stars=data.get("stars"),
            first_mentioned_turn=data.get("first_mentioned_turn", 0),
            last_mentioned_turn=data.get("last_mentioned_turn", 0),
            mention_count=data.get("mention_count", 1),
            last_position=data.get("last_position"),
        )


@dataclass
class ConversationContext:
    """Enhanced context extracted from conversation history."""

    # Product tracking with full context
    products: list[ProductContext] = field(default_factory=list)

    # Quick lookup by ASIN
    product_asins: list[str] = field(default_factory=list)

    # Last search results (ordered, for ordinal references)
    last_search_results: list[ProductContext] = field(default_factory=list)

    # Entity tracking
    brands: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    # User preferences (inferred)
    price_min: float | None = None
    price_max: float | None = None
    preferred_brands: list[str] = field(default_factory=list)

    # Last query context
    last_query: str | None = None
    last_intent: str | None = None
    last_agent: str | None = None

    # Turn counter
    current_turn: int = 0

    def add_product(self, product: dict[str, Any], position: int | None = None) -> ProductContext:
        """Add or update a product in context."""
        asin = product.get("asin") or product.get("id")
        if not asin:
            return None

        # Check if product already exists
        existing = self.get_product_by_asin(asin)
        if existing:
            existing.last_mentioned_turn = self.current_turn
            existing.mention_count += 1
            if position is not None:
                existing.last_position = position
            # Update price/rating if available
            if product.get("price"):
                existing.price = product["price"]
            if product.get("stars") or product.get("rating"):
                existing.stars = product.get("stars") or product.get("rating")
            return existing

        # Create new product context
        ctx = ProductContext(
            asin=asin,
            title=product.get("title", "Unknown"),
            brand=product.get("brand"),
            price=product.get("price"),
            stars=product.get("stars") or product.get("rating"),
            first_mentioned_turn=self.current_turn,
            last_mentioned_turn=self.current_turn,
            last_position=position,
        )

        self.products.append(ctx)
        self.product_asins.append(asin)

        # Track brand
        if ctx.brand and ctx.brand not in self.brands:
            self.brands.append(ctx.brand)

        # Keep only recent products (max 20)
        if len(self.products) > 20:
            self.products = self.products[-20:]
            self.product_asins = self.product_asins[-20:]

        return ctx

    def get_product_by_asin(self, asin: str) -> ProductContext | None:
        """Get product by ASIN."""
        for p in self.products:
            if p.asin == asin:
                return p
        return None

    def get_product_by_position(self, position: int) -> ProductContext | None:
        """Get product by position in last search results (1-indexed)."""
        if 0 < position <= len(self.last_search_results):
            return self.last_search_results[position - 1]
        return None

    def get_cheapest_product(self) -> ProductContext | None:
        """Get the cheapest product from last search results."""
        products_with_price = [p for p in self.last_search_results if p.price is not None]
        if products_with_price:
            return min(products_with_price, key=lambda p: p.price)
        return None

    def get_most_expensive_product(self) -> ProductContext | None:
        """Get the most expensive product from last search results."""
        products_with_price = [p for p in self.last_search_results if p.price is not None]
        if products_with_price:
            return max(products_with_price, key=lambda p: p.price)
        return None

    def get_best_rated_product(self) -> ProductContext | None:
        """Get the best rated product from last search results."""
        products_with_rating = [p for p in self.last_search_results if p.stars is not None]
        if products_with_rating:
            return max(products_with_rating, key=lambda p: p.stars)
        return None

    def get_last_mentioned_product(self) -> ProductContext | None:
        """Get the most recently mentioned product."""
        if self.products:
            return max(self.products, key=lambda p: p.last_mentioned_turn)
        return None

    def set_search_results(self, products: list[dict[str, Any]]) -> None:
        """Set the last search results for ordinal references."""
        self.last_search_results = []
        for i, p in enumerate(products[:10]):  # Keep top 10
            ctx = self.add_product(p, position=i + 1)
            if ctx:
                self.last_search_results.append(ctx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "products": [p.to_dict() for p in self.products],
            "product_asins": self.product_asins,
            "last_search_results": [p.to_dict() for p in self.last_search_results],
            "brands": self.brands,
            "categories": self.categories,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "preferred_brands": self.preferred_brands,
            "last_query": self.last_query,
            "last_intent": self.last_intent,
            "last_agent": self.last_agent,
            "current_turn": self.current_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        ctx = cls()
        ctx.products = [ProductContext.from_dict(p) for p in data.get("products", [])]
        ctx.product_asins = data.get("product_asins", [])
        ctx.last_search_results = [ProductContext.from_dict(p) for p in data.get("last_search_results", [])]
        ctx.brands = data.get("brands", [])
        ctx.categories = data.get("categories", [])
        ctx.price_min = data.get("price_min")
        ctx.price_max = data.get("price_max")
        ctx.preferred_brands = data.get("preferred_brands", [])
        ctx.last_query = data.get("last_query")
        ctx.last_intent = data.get("last_intent")
        ctx.last_agent = data.get("last_agent")
        ctx.current_turn = data.get("current_turn", 0)
        return ctx


@dataclass
class Conversation:
    """A conversation session with enhanced tracking."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Configuration
    max_history: int = 50  # Max messages to keep

    @property
    def turn_count(self) -> int:
        """Get number of conversation turns (user messages)."""
        return sum(1 for m in self.messages if m.role == "user")

    def add_user_message(
        self,
        content: str,
        intent: str | None = None,
        entities: list[str] | None = None,
        references_resolved: list[dict] | None = None,
    ) -> Message:
        """Add a user message with metadata."""
        self.context.current_turn += 1

        message = Message(
            role="user",
            content=content,
            intent=intent,
            entities_mentioned=entities or [],
            references_resolved=references_resolved or [],
        )
        self._add_message(message)

        # Update context
        self.context.last_query = content
        if intent:
            self.context.last_intent = intent

        return message

    def add_assistant_message(
        self,
        content: str,
        intent: str | None = None,
        products: list[dict] | None = None,
        agent: str | None = None,
        execution_time_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> Message:
        """Add an assistant message with response metadata.

        Args:
            content: Message content
            intent: Detected intent
            products: List of products to add (or use metadata["products"])
            agent: Agent used
            execution_time_ms: Execution time
            metadata: Additional metadata (can contain "products" key)
        """
        # Support both direct products and metadata["products"]
        if products is None and metadata and "products" in metadata:
            products = metadata["products"]

        product_asins = []
        if products:
            # Update context with products
            self.context.set_search_results(products)
            product_asins = [p.get("asin") or p.get("id") for p in products if p.get("asin") or p.get("id")]

        message = Message(
            role="assistant",
            content=content,
            intent=intent,
            products_returned=product_asins,
            agent_used=agent,
            execution_time_ms=execution_time_ms,
        )
        self._add_message(message)

        if agent:
            self.context.last_agent = agent

        return message

    def update_context(
        self,
        products: list[dict] | None = None,
        query_type: str | None = None,
        agent: str | None = None,
        brands: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> None:
        """Update conversation context with new information.

        Args:
            products: List of products to add to context
            query_type: Type of query (intent)
            agent: Agent that handled the query
            brands: Brands to track
            categories: Categories to track
        """
        if products:
            self.context.set_search_results(products)

            # Extract brands from products
            for p in products:
                brand = p.get("brand")
                if brand and brand not in self.context.brands:
                    self.context.brands.append(brand)

            # Extract categories
            for p in products:
                category = p.get("category")
                if category and category not in self.context.categories:
                    self.context.categories.append(category)

        if query_type:
            self.context.last_intent = query_type

        if agent:
            self.context.last_agent = agent

        if brands:
            for brand in brands:
                if brand not in self.context.brands:
                    self.context.brands.append(brand)

        if categories:
            for cat in categories:
                if cat not in self.context.categories:
                    self.context.categories.append(cat)

        # Keep lists bounded
        self.context.brands = self.context.brands[:20]
        self.context.categories = self.context.categories[:20]

    def _add_message(self, message: Message) -> None:
        """Add message and maintain history limit."""
        self.messages.append(message)
        self.updated_at = datetime.now()

        # Trim old messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_history(self, limit: int | None = None) -> list[Message]:
        """Get conversation history."""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def get_history_text(self, limit: int = 5) -> str:
        """Get conversation history as text."""
        history = self.get_history(limit)
        parts = []
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{prefix}: {msg.content}")
        return "\n".join(parts)

    def get_last_products(self, limit: int = 5) -> list[ProductContext]:
        """Get last mentioned products."""
        return self.context.last_search_results[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "max_history": self.max_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        conv = cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            max_history=data.get("max_history", 50),
        )
        conv.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        conv.context = ConversationContext.from_dict(data.get("context", {}))
        return conv
