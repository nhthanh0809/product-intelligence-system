"""Conversation Manager for session and context management.

Provides:
- Session management with Redis caching + PostgreSQL persistence
- Integration with PronounResolver and QueryRewriter
- Conversation lifecycle management
"""

import json
from datetime import datetime
from typing import Any

import structlog

from src.conversation.models import Conversation, ConversationContext, Message
from src.conversation.repository import ConversationRepository, get_conversation_repository
from src.conversation.pronoun_resolver import PronounResolver, get_pronoun_resolver
from src.conversation.query_rewriter import QueryRewriter, RewriteResult, get_query_rewriter

logger = structlog.get_logger()


class ConversationManager:
    """Manages conversation sessions with caching and persistence.

    Features:
    - Redis caching for fast session access
    - PostgreSQL persistence for durability
    - Pronoun resolution for multi-turn context
    - Query rewriting with entity injection
    """

    # Cache TTL in seconds (1 hour)
    CACHE_TTL = 3600

    def __init__(
        self,
        repository: ConversationRepository | None = None,
        redis_client: Any = None,
        pronoun_resolver: PronounResolver | None = None,
        query_rewriter: QueryRewriter | None = None,
    ):
        """Initialize manager.

        Args:
            repository: PostgreSQL repository
            redis_client: Redis client for caching
            pronoun_resolver: Pronoun resolver instance
            query_rewriter: Query rewriter instance
        """
        self._repository = repository
        self._redis = redis_client
        self._pronoun_resolver = pronoun_resolver
        self._query_rewriter = query_rewriter
        self._initialized = False

        # In-memory fallback for when Redis is unavailable
        self._memory_cache: dict[str, Conversation] = {}

    async def initialize(self) -> None:
        """Initialize manager components."""
        if self._initialized:
            return

        # Get repository
        if self._repository is None:
            self._repository = await get_conversation_repository()

        # Get Redis client
        if self._redis is None:
            try:
                from src.main import _redis_client
                self._redis = _redis_client
            except ImportError:
                logger.warning("conversation_manager_redis_unavailable")

        # Get pronoun resolver
        if self._pronoun_resolver is None:
            try:
                self._pronoun_resolver = await get_pronoun_resolver()
            except Exception as e:
                logger.warning("pronoun_resolver_init_failed", error=str(e))

        # Get query rewriter
        if self._query_rewriter is None:
            try:
                self._query_rewriter = await get_query_rewriter()
            except Exception as e:
                logger.warning("query_rewriter_init_failed", error=str(e))

        self._initialized = True
        logger.info("conversation_manager_initialized")

    async def get(self, session_id: str | None = None) -> Conversation:
        """Get or create a conversation.

        Args:
            session_id: Optional session ID. Creates new if None.

        Returns:
            Conversation instance
        """
        if not self._initialized:
            await self.initialize()

        # Create new conversation if no session_id
        if not session_id:
            conversation = Conversation()
            logger.info("conversation_created", session_id=conversation.session_id)
            return conversation

        # Try cache first
        conversation = await self._get_from_cache(session_id)
        if conversation:
            logger.debug("conversation_cache_hit", session_id=session_id)
            return conversation

        # Try database
        if self._repository:
            conversation = await self._repository.get(session_id)
            if conversation:
                # Cache it
                await self._set_cache(conversation)
                logger.debug("conversation_db_hit", session_id=session_id)
                return conversation

        # Create new with provided session_id
        conversation = Conversation(session_id=session_id)
        logger.info("conversation_created_with_id", session_id=session_id)
        return conversation

    async def save(self, conversation: Conversation) -> bool:
        """Save conversation to cache and database.

        Args:
            conversation: Conversation to save

        Returns:
            True if saved successfully
        """
        if not self._initialized:
            await self.initialize()

        conversation.updated_at = datetime.now()

        # Save to cache (always succeeds)
        await self._set_cache(conversation)

        # Try to save to database (optional persistence)
        if self._repository and self._repository._pool:
            success = await self._repository.save(conversation)
            if not success:
                logger.warning("conversation_db_save_failed", session_id=conversation.session_id)
                # Still return True since cache save succeeded

        logger.debug("conversation_saved", session_id=conversation.session_id)
        return True

    async def delete(self, session_id: str) -> bool:
        """Delete a conversation.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted
        """
        if not self._initialized:
            await self.initialize()

        # Remove from cache
        await self._delete_cache(session_id)

        # Remove from database
        if self._repository:
            return await self._repository.delete(session_id)

        return True

    async def rewrite_query(
        self,
        query: str,
        conversation: Conversation,
    ) -> RewriteResult:
        """Rewrite a query using conversation context.

        Args:
            query: Original query
            conversation: Conversation with context

        Returns:
            RewriteResult with rewritten query
        """
        if not self._initialized:
            await self.initialize()

        if self._query_rewriter:
            return await self._query_rewriter.rewrite(query, conversation)

        # Fallback: return unchanged
        return RewriteResult(
            original_query=query,
            rewritten_query=query,
        )

    async def _get_from_cache(self, session_id: str) -> Conversation | None:
        """Get conversation from cache."""
        # Try Redis
        if self._redis:
            try:
                key = f"conversation:{session_id}"
                data = await self._redis.get(key)
                if data:
                    return Conversation.from_dict(json.loads(data))
            except Exception as e:
                logger.warning("conversation_redis_get_failed", error=str(e))

        # Try memory cache
        return self._memory_cache.get(session_id)

    async def _set_cache(self, conversation: Conversation) -> None:
        """Set conversation in cache."""
        # Set in Redis
        if self._redis:
            try:
                key = f"conversation:{conversation.session_id}"
                data = json.dumps(conversation.to_dict())
                await self._redis.setex(key, self.CACHE_TTL, data)
            except Exception as e:
                logger.warning("conversation_redis_set_failed", error=str(e))

        # Set in memory cache (limited to 100 sessions)
        self._memory_cache[conversation.session_id] = conversation
        if len(self._memory_cache) > 100:
            # Remove oldest
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

    async def _delete_cache(self, session_id: str) -> None:
        """Delete conversation from cache."""
        # Delete from Redis
        if self._redis:
            try:
                key = f"conversation:{session_id}"
                await self._redis.delete(key)
            except Exception as e:
                logger.warning("conversation_redis_delete_failed", error=str(e))

        # Delete from memory cache
        self._memory_cache.pop(session_id, None)

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List recent conversation sessions.

        Args:
            limit: Max sessions to return
            offset: Offset for pagination

        Returns:
            List of session summaries
        """
        if not self._initialized:
            await self.initialize()

        if self._repository:
            return await self._repository.list_sessions(limit, offset)

        # Fallback to memory cache
        sessions = []
        for session_id, conv in list(self._memory_cache.items())[offset:offset+limit]:
            sessions.append({
                "session_id": session_id,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "message_count": len(conv.messages),
            })
        return sessions

    async def cleanup_old_sessions(self, max_age_hours: int = 72) -> int:
        """Clean up old sessions.

        Args:
            max_age_hours: Max age in hours

        Returns:
            Number of sessions deleted
        """
        if not self._initialized:
            await self.initialize()

        if self._repository:
            return await self._repository.cleanup_old_sessions(max_age_hours)

        return 0


# Singleton
_manager: ConversationManager | None = None


async def get_conversation_manager() -> ConversationManager:
    """Get or create conversation manager singleton."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
        await _manager.initialize()
    return _manager


async def get_conversation(session_id: str | None = None) -> Conversation:
    """Get or create a conversation.

    Convenience function that uses the singleton manager.

    Args:
        session_id: Optional session ID

    Returns:
        Conversation instance
    """
    manager = await get_conversation_manager()
    return await manager.get(session_id)
