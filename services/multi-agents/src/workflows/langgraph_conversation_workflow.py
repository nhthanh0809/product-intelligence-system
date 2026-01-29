"""LangGraph-powered conversation workflow.

This workflow replaces the rule-based conversation workflow with
a LangGraph-powered engine for natural multi-turn conversations.

Features:
- Full conversation state persistence
- LLM-powered context understanding
- Automatic reference resolution
- User preference learning
- Natural response generation
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.conversation.langgraph_engine import (
    LangGraphConversationEngine,
    get_langgraph_engine,
)
from src.models.intent import QueryIntent

logger = structlog.get_logger()


@dataclass
class LangGraphConversationResult:
    """Result from LangGraph conversation workflow."""

    # Core response
    query: str
    response_text: str
    intent: QueryIntent

    # Products
    products: list[dict] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    recommendations: list[dict] = field(default_factory=list)

    # Conversation metadata
    session_id: str = ""
    turn_number: int = 0
    action_taken: str = ""

    # Context information
    context_used: bool = False
    clarification_asked: bool = False
    preferences_learned: dict = field(default_factory=dict)

    # Suggestions
    suggestions: list[str] = field(default_factory=list)

    # Performance
    execution_time_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "query": self.query,
            "response_text": self.response_text,
            "intent": self.intent.value if hasattr(self.intent, 'value') else str(self.intent),
            "products": self.products,
            "comparison": self.comparison,
            "recommendations": self.recommendations,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "action_taken": self.action_taken,
            "context_used": self.context_used,
            "clarification_asked": self.clarification_asked,
            "preferences_learned": self.preferences_learned,
            "suggestions": self.suggestions,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


class LangGraphConversationWorkflow:
    """Workflow using LangGraph for natural conversations.

    This workflow provides a ChatGPT-like experience where:
    1. The system understands full conversation context
    2. References are automatically resolved ("it", "them", "the first one")
    3. User preferences are learned and applied
    4. Responses are natural and conversational
    5. Clarifying questions are asked when needed
    """

    def __init__(self, engine: LangGraphConversationEngine | None = None):
        """Initialize workflow.

        Args:
            engine: LangGraph engine instance (will be created if None)
        """
        self._engine = engine
        self._initialized = False

    async def initialize(self):
        """Initialize the workflow."""
        if self._initialized:
            return

        if self._engine is None:
            self._engine = await get_langgraph_engine()

        self._initialized = True
        logger.info("langgraph_conversation_workflow_initialized")

    async def execute(
        self,
        query: str,
        session_id: str | None = None,
    ) -> LangGraphConversationResult:
        """Execute conversation turn.

        Args:
            query: User's message
            session_id: Session ID for conversation continuity

        Returns:
            LangGraphConversationResult with response and metadata
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        try:
            logger.info(
                "langgraph_conversation_started",
                query=query[:50],
                session_id=session_id,
            )

            # Use the LangGraph engine
            result = await self._engine.chat(query, session_id)

            # Map action to intent
            intent = self._map_action_to_intent(result.get("action", "search"))

            # Build result
            execution_time = (time.time() - start_time) * 1000

            logger.info(
                "langgraph_conversation_completed",
                session_id=result.get("session_id"),
                intent=result.get("intent"),
                action=result.get("action"),
                turn=result.get("turn_count"),
                execution_time_ms=execution_time,
            )

            return LangGraphConversationResult(
                query=query,
                response_text=result.get("response", ""),
                intent=intent,
                products=result.get("products", []),
                session_id=result.get("session_id", ""),
                turn_number=result.get("turn_count", 1),
                action_taken=result.get("action", ""),
                context_used=bool(result.get("products")),
                clarification_asked=result.get("clarification_asked", False),
                preferences_learned=result.get("preferences_learned", {}),
                suggestions=result.get("suggestions", []),
                execution_time_ms=execution_time,
                error=result.get("error"),
            )

        except Exception as e:
            logger.error("langgraph_conversation_failed", error=str(e))

            return LangGraphConversationResult(
                query=query,
                response_text=f"I encountered an error: {str(e)}",
                intent=QueryIntent.SEARCH,
                session_id=session_id or "",
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _map_action_to_intent(self, action: str) -> QueryIntent:
        """Map LangGraph action to QueryIntent."""
        action_intent_map = {
            "search": QueryIntent.SEARCH,
            "compare": QueryIntent.COMPARE,
            "recommend": QueryIntent.RECOMMEND,
            "price_check": QueryIntent.PRICE_CHECK,
            "product_details": QueryIntent.ANALYZE,
            "general_chat": QueryIntent.HELP,
            "clarify": QueryIntent.CLARIFICATION,
        }
        return action_intent_map.get(action, QueryIntent.SEARCH)

    async def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history.

        Args:
            session_id: Session ID

        Returns:
            List of messages
        """
        if not self._initialized:
            await self.initialize()

        return await self._engine.get_conversation_history(session_id)


# === Singleton ===

_workflow: LangGraphConversationWorkflow | None = None


async def get_langgraph_conversation_workflow() -> LangGraphConversationWorkflow:
    """Get or create the LangGraph conversation workflow singleton."""
    global _workflow
    if _workflow is None:
        _workflow = LangGraphConversationWorkflow()
        await _workflow.initialize()
    return _workflow


async def execute_langgraph_conversation(
    query: str,
    session_id: str | None = None,
) -> LangGraphConversationResult:
    """Convenience function to execute a LangGraph conversation query.

    Args:
        query: User's message
        session_id: Session ID for conversation continuity

    Returns:
        LangGraphConversationResult
    """
    workflow = await get_langgraph_conversation_workflow()
    return await workflow.execute(query, session_id)
