"""Workflow implementations for Product Intelligence System.

This module provides three workflow types:

1. SimpleWorkflow: Single-intent queries
   - Search, compare, analyze, recommend, etc.
   - Direct routing to appropriate agent
   - Response synthesis

2. CompoundWorkflow: Multi-step queries
   - Multiple intents in one query
   - Execution plan with dependency resolution
   - Parallel execution of independent steps

3. ConversationWorkflow: Multi-turn context
   - Session management
   - Reference resolution ("compare them", "the first one")
   - Context-aware query enhancement
"""

from src.workflows.simple_workflow import (
    SimpleWorkflow,
    WorkflowResult,
    get_simple_workflow,
    execute_simple_query,
)
from src.workflows.compound_workflow import (
    CompoundWorkflow,
    CompoundWorkflowResult,
    get_compound_workflow,
    execute_compound_query,
)
from src.workflows.conversation_workflow import (
    ConversationWorkflow,
    ConversationWorkflowResult,
    get_conversation_workflow,
    execute_conversation_query,
)

__all__ = [
    # Simple workflow
    "SimpleWorkflow",
    "WorkflowResult",
    "get_simple_workflow",
    "execute_simple_query",
    # Compound workflow
    "CompoundWorkflow",
    "CompoundWorkflowResult",
    "get_compound_workflow",
    "execute_compound_query",
    # Conversation workflow
    "ConversationWorkflow",
    "ConversationWorkflowResult",
    "get_conversation_workflow",
    "execute_conversation_query",
]
