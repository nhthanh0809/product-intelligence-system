"""Multi-turn conversation management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog

from src.config import get_settings

logger = structlog.get_logger()

# Try to import redis
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@dataclass
class Message:
    """A single conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """Context extracted from conversation history."""
    # Referenced products
    products: list[dict[str, Any]] = field(default_factory=list)
    product_asins: list[str] = field(default_factory=list)

    # Topics discussed
    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)

    # User preferences (inferred)
    price_range: tuple[float | None, float | None] = (None, None)
    preferred_brands: list[str] = field(default_factory=list)

    # Last query context
    last_query_type: str | None = None
    last_agent: str | None = None


@dataclass
class Conversation:
    """A conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Configuration
    max_history: int = 20  # Max messages to keep

    def add_user_message(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a user message."""
        message = Message(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        self._add_message(message)
        return message

    def add_assistant_message(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add an assistant message."""
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        self._add_message(message)
        return message

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

    def update_context(
        self,
        products: list[dict[str, Any]] | None = None,
        query_type: str | None = None,
        agent: str | None = None,
        **kwargs,
    ) -> None:
        """Update conversation context."""
        if products:
            # Add new products
            for p in products:
                asin = p.get("asin")
                if asin and asin not in self.context.product_asins:
                    self.context.product_asins.append(asin)
                    self.context.products.append(p)

                # Extract brand
                brand = p.get("brand")
                if brand and brand not in self.context.brands:
                    self.context.brands.append(brand)

                # Extract category
                cat = p.get("category_level1")
                if cat and cat not in self.context.categories:
                    self.context.categories.append(cat)

            # Keep only recent products
            if len(self.context.products) > 10:
                self.context.products = self.context.products[-10:]
                self.context.product_asins = self.context.product_asins[-10:]

        if query_type:
            self.context.last_query_type = query_type

        if agent:
            self.context.last_agent = agent

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "context": {
                "products": self.context.products,
                "product_asins": self.context.product_asins,
                "topics": self.context.topics,
                "categories": self.context.categories,
                "brands": self.context.brands,
                "last_query_type": self.context.last_query_type,
                "last_agent": self.context.last_agent,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        conv = cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
        conv.messages = [Message.from_dict(m) for m in data["messages"]]

        ctx = data.get("context", {})
        conv.context.products = ctx.get("products", [])
        conv.context.product_asins = ctx.get("product_asins", [])
        conv.context.topics = ctx.get("topics", [])
        conv.context.categories = ctx.get("categories", [])
        conv.context.brands = ctx.get("brands", [])
        conv.context.last_query_type = ctx.get("last_query_type")
        conv.context.last_agent = ctx.get("last_agent")

        return conv


class ConversationManager:
    """Manages conversation sessions with Redis persistence."""

    KEY_PREFIX = "conversation:"
    TTL = 3600 * 24  # 24 hours

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._conversations: dict[str, Conversation] = {}  # In-memory cache
        self._enabled = HAS_REDIS

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if not self._enabled:
            return

        if self._client is not None:
            return

        try:
            url = f"redis://{self.settings.redis_host}:{self.settings.redis_port}/0"
            self._client = redis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._client.ping()
            logger.info("conversation_manager_connected")
        except Exception as e:
            logger.warning("conversation_manager_connect_failed", error=str(e))
            self._enabled = False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.KEY_PREFIX}{session_id}"

    async def get_or_create(self, session_id: str | None = None) -> Conversation:
        """Get existing conversation or create new one."""
        if session_id:
            conv = await self.get(session_id)
            if conv:
                return conv

        # Create new conversation
        conv = Conversation()
        if session_id:
            conv.session_id = session_id

        self._conversations[conv.session_id] = conv
        await self.save(conv)

        return conv

    async def get(self, session_id: str) -> Conversation | None:
        """Get conversation by session ID."""
        # Check memory cache
        if session_id in self._conversations:
            return self._conversations[session_id]

        # Check Redis
        if self._enabled and self._client:
            try:
                import json
                data = await self._client.get(self._key(session_id))
                if data:
                    conv = Conversation.from_dict(json.loads(data))
                    self._conversations[session_id] = conv
                    return conv
            except Exception as e:
                logger.warning("conversation_get_failed", error=str(e))

        return None

    async def save(self, conversation: Conversation) -> None:
        """Save conversation to Redis."""
        self._conversations[conversation.session_id] = conversation

        if self._enabled and self._client:
            try:
                import json
                await self._client.set(
                    self._key(conversation.session_id),
                    json.dumps(conversation.to_dict()),
                    ex=self.TTL,
                )
            except Exception as e:
                logger.warning("conversation_save_failed", error=str(e))

    async def delete(self, session_id: str) -> None:
        """Delete conversation."""
        if session_id in self._conversations:
            del self._conversations[session_id]

        if self._enabled and self._client:
            try:
                await self._client.delete(self._key(session_id))
            except Exception as e:
                logger.warning("conversation_delete_failed", error=str(e))


# Singleton
_manager: ConversationManager | None = None


async def get_conversation_manager() -> ConversationManager:
    """Get or create conversation manager singleton."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
        await _manager.connect()
    return _manager


async def get_conversation(session_id: str | None = None) -> Conversation:
    """Get or create a conversation.

    Args:
        session_id: Optional session ID (creates new if None)

    Returns:
        Conversation object
    """
    manager = await get_conversation_manager()
    return await manager.get_or_create(session_id)
