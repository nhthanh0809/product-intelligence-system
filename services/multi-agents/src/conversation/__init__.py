"""Enhanced conversation management module.

Provides:
- Conversation models with entity tracking
- PostgreSQL-based conversation storage
- Pronoun resolution for multi-turn context
- Query rewriting with entity injection
"""

from src.conversation.models import (
    Message,
    ConversationContext,
    Conversation,
    EntityReference,
    ResolvedReference,
    ProductContext,
    ReferenceType,
)
from src.conversation.repository import (
    ConversationRepository,
    get_conversation_repository,
)
from src.conversation.pronoun_resolver import (
    PronounResolver,
    PronounPattern,
    ResolutionResult,
)
from src.conversation.query_rewriter import (
    QueryRewriter,
    RewriteResult,
)
from src.conversation.manager import (
    ConversationManager,
    get_conversation_manager,
    get_conversation,
)
from src.conversation.langgraph_engine import (
    LangGraphConversationEngine,
    ConversationState,
    get_langgraph_engine,
)

__all__ = [
    # Models
    "Message",
    "ConversationContext",
    "Conversation",
    "EntityReference",
    "ResolvedReference",
    "ProductContext",
    "ReferenceType",
    # Repository
    "ConversationRepository",
    "get_conversation_repository",
    # Pronoun Resolver
    "PronounResolver",
    "PronounPattern",
    "ResolutionResult",
    # Query Rewriter
    "QueryRewriter",
    "RewriteResult",
    # Manager
    "ConversationManager",
    "get_conversation_manager",
    "get_conversation",
    # LangGraph Engine
    "LangGraphConversationEngine",
    "ConversationState",
    "get_langgraph_engine",
]
