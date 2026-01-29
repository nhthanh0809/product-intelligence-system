"""PostgreSQL repository for conversation storage.

Provides persistent storage for:
- Conversation sessions
- Message history
- Conversation context
"""

import json
from datetime import datetime
from typing import Any

import structlog

from src.conversation.models import Conversation, Message, ConversationContext

logger = structlog.get_logger()


class ConversationRepository:
    """PostgreSQL repository for conversation data."""

    def __init__(self, pool=None):
        """Initialize repository.

        Args:
            pool: Database connection pool
        """
        self._pool = pool

    def set_pool(self, pool) -> None:
        """Set database pool."""
        self._pool = pool

    async def get(self, session_id: str) -> Conversation | None:
        """Get conversation by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Conversation or None if not found
        """
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                # Get session
                row = await conn.fetchrow(
                    """
                    SELECT session_id, context, created_at, updated_at, metadata
                    FROM conversation_sessions
                    WHERE session_id = $1
                    """,
                    session_id,
                )

                if not row:
                    return None

                # Get messages
                message_rows = await conn.fetch(
                    """
                    SELECT message_id, role, content, intent, metadata, created_at
                    FROM conversation_messages
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                    """,
                    session_id,
                )

                # Build conversation
                conversation = Conversation(
                    session_id=row["session_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

                # Load context
                if row["context"]:
                    context_data = json.loads(row["context"]) if isinstance(row["context"], str) else row["context"]
                    conversation.context = ConversationContext.from_dict(context_data)

                # Load messages
                for msg_row in message_rows:
                    metadata = msg_row["metadata"] or {}
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    message = Message(
                        message_id=msg_row["message_id"],
                        role=msg_row["role"],
                        content=msg_row["content"],
                        timestamp=msg_row["created_at"],
                        intent=msg_row["intent"],
                        entities_mentioned=metadata.get("entities_mentioned", []),
                        references_resolved=metadata.get("references_resolved", []),
                        products_returned=metadata.get("products_returned", []),
                        agent_used=metadata.get("agent_used"),
                        execution_time_ms=metadata.get("execution_time_ms", 0.0),
                    )
                    conversation.messages.append(message)

                return conversation

        except Exception as e:
            logger.error("conversation_get_failed", session_id=session_id, error=str(e))
            return None

    async def save(self, conversation: Conversation) -> bool:
        """Save conversation to database.

        Args:
            conversation: Conversation to save

        Returns:
            True if saved successfully
        """
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Upsert session
                    await conn.execute(
                        """
                        INSERT INTO conversation_sessions (session_id, context, created_at, updated_at)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (session_id) DO UPDATE SET
                            context = EXCLUDED.context,
                            updated_at = EXCLUDED.updated_at
                        """,
                        conversation.session_id,
                        json.dumps(conversation.context.to_dict()),
                        conversation.created_at,
                        conversation.updated_at,
                    )

                    # Save messages (upsert each)
                    for message in conversation.messages:
                        metadata = {
                            "entities_mentioned": message.entities_mentioned,
                            "references_resolved": message.references_resolved,
                            "products_returned": message.products_returned,
                            "agent_used": message.agent_used,
                            "execution_time_ms": message.execution_time_ms,
                        }

                        await conn.execute(
                            """
                            INSERT INTO conversation_messages
                                (message_id, session_id, role, content, intent, metadata, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (message_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                intent = EXCLUDED.intent,
                                metadata = EXCLUDED.metadata
                            """,
                            message.message_id,
                            conversation.session_id,
                            message.role,
                            message.content,
                            message.intent,
                            json.dumps(metadata),
                            message.timestamp,
                        )

                return True

        except Exception as e:
            logger.error("conversation_save_failed", session_id=conversation.session_id, error=str(e))
            return False

    async def delete(self, session_id: str) -> bool:
        """Delete conversation.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted
        """
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Delete messages first (foreign key)
                    await conn.execute(
                        "DELETE FROM conversation_messages WHERE session_id = $1",
                        session_id,
                    )
                    # Delete session
                    await conn.execute(
                        "DELETE FROM conversation_sessions WHERE session_id = $1",
                        session_id,
                    )
                return True

        except Exception as e:
            logger.error("conversation_delete_failed", session_id=session_id, error=str(e))
            return False

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
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        s.session_id,
                        s.created_at,
                        s.updated_at,
                        COUNT(m.message_id) as message_count
                    FROM conversation_sessions s
                    LEFT JOIN conversation_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )

                return [
                    {
                        "session_id": row["session_id"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                        "message_count": row["message_count"],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error("conversation_list_failed", error=str(e))
            return []

    async def cleanup_old_sessions(self, max_age_hours: int = 72) -> int:
        """Delete sessions older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of sessions deleted
        """
        if not self._pool:
            return 0

        try:
            async with self._pool.acquire() as conn:
                # Get sessions to delete
                result = await conn.execute(
                    """
                    DELETE FROM conversation_sessions
                    WHERE updated_at < NOW() - INTERVAL '%s hours'
                    """,
                    max_age_hours,
                )
                # Parse result to get count
                count = int(result.split()[-1]) if result else 0
                logger.info("conversation_cleanup", deleted=count)
                return count

        except Exception as e:
            logger.error("conversation_cleanup_failed", error=str(e))
            return 0


# Singleton
_repository: ConversationRepository | None = None


async def get_conversation_repository() -> ConversationRepository:
    """Get or create conversation repository singleton."""
    global _repository
    if _repository is None:
        _repository = ConversationRepository()

        # Try to get pool from global state
        try:
            from src.main import _db_pool
            if _db_pool:
                _repository.set_pool(_db_pool)
        except ImportError:
            pass

    return _repository
