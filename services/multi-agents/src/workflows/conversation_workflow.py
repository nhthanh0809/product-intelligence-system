"""Conversation Workflow for Multi-Turn Context Management.

This workflow wraps simple and compound workflows to handle:
1. Conversation context preservation across turns
2. Reference resolution ("compare them", "show me more", "the first one")
3. Context-aware query enhancement
4. Conversation state updates

Example multi-turn conversation:
User: "Find wireless headphones under $200"
Assistant: [shows 5 products]
User: "Compare the top 3"
-> Resolves "top 3" to first 3 products from previous results
User: "Which is best for travel?"
-> Uses comparison context + travel use case
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.models.intent import QueryIntent, QueryComplexity, IntentAnalysis
from src.conversation import (
    Conversation,
    ConversationContext,
    get_conversation,
    get_conversation_manager,
    QueryRewriter,
    RewriteResult,
)
from src.agents.intent_agent import get_intent_agent
from src.workflows.simple_workflow import (
    SimpleWorkflow,
    WorkflowResult,
    get_simple_workflow,
)
from src.workflows.compound_workflow import (
    CompoundWorkflow,
    CompoundWorkflowResult,
    get_compound_workflow,
)
from src.agents.synthesis_agent import OutputFormat

logger = structlog.get_logger()


# Reference patterns for context resolution
REFERENCE_PATTERNS = {
    # Product references
    "them": r"\b(them|these|those|the products?)\b",
    "it": r"\b(it|this|that|the product)\b",
    "first": r"\b(first|1st|top)\s+(one|product|option)?\b",
    "second": r"\b(second|2nd)\s+(one|product|option)?\b",
    "third": r"\b(third|3rd)\s+(one|product|option)?\b",
    "last": r"\b(last|final)\s+(one|product|option)?\b",
    "cheapest": r"\b(cheapest|lowest\s+price|most\s+affordable)\b",
    "best": r"\b(best|top\s+rated|highest\s+rated)\b",

    # Quantity references
    "top_n": r"\btop\s+(\d+)\b",
    "first_n": r"\bfirst\s+(\d+)\b",

    # Comparison references
    "compare_them": r"\bcompare\s+(them|these|those)\b",
    "winner": r"\b(winner|best\s+one|recommended)\b",

    # Contextual
    "more": r"\b(more|similar|like\s+these?|alternatives?)\b",
    "details": r"\b(details?|more\s+info|tell\s+me\s+more)\b",
}


@dataclass
class ConversationWorkflowResult(WorkflowResult):
    """Extended result for conversation workflows."""

    session_id: str = ""
    turn_number: int = 0
    context_used: bool = False
    references_resolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        base = super().to_dict()
        base.update({
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "context_used": self.context_used,
            "references_resolved": self.references_resolved,
        })
        return base


class ConversationWorkflow:
    """Workflow for multi-turn conversations with context management.

    This workflow:
    1. Loads/creates conversation state
    2. Analyzes query for context references
    3. Resolves references using conversation history
    4. Enhances query with context
    5. Routes to simple or compound workflow
    6. Updates conversation state with results
    """

    def __init__(self):
        """Initialize workflow."""
        self._simple_workflow: SimpleWorkflow | None = None
        self._compound_workflow: CompoundWorkflow | None = None
        self._query_rewriter: QueryRewriter | None = None

    async def _get_query_rewriter(self) -> QueryRewriter:
        """Get query rewriter."""
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter()
            await self._query_rewriter.initialize()
        return self._query_rewriter

    async def _get_simple_workflow(self) -> SimpleWorkflow:
        """Get simple workflow."""
        if self._simple_workflow is None:
            self._simple_workflow = await get_simple_workflow()
        return self._simple_workflow

    async def _get_compound_workflow(self) -> CompoundWorkflow:
        """Get compound workflow."""
        if self._compound_workflow is None:
            self._compound_workflow = await get_compound_workflow()
        return self._compound_workflow

    async def execute(
        self,
        query: str,
        session_id: str | None = None,
    ) -> ConversationWorkflowResult:
        """Execute workflow with conversation context.

        Args:
            query: User query
            session_id: Optional session ID for multi-turn

        Returns:
            ConversationWorkflowResult with response and context
        """
        start_time = time.time()

        try:
            # Step 1: Get or create conversation
            conversation = await get_conversation(session_id)
            turn_number = len([m for m in conversation.messages if m.role == "user"]) + 1

            logger.info(
                "conversation_turn_started",
                session_id=conversation.session_id,
                turn=turn_number,
                query=query[:50],
            )

            # Step 2: Add user message
            conversation.add_user_message(query)

            # Step 3: Rewrite query using conversation context (pronoun resolution)
            references_resolved: list[str] = []
            context_used = False

            # Use QueryRewriter for advanced pronoun/reference resolution
            query_rewriter = await self._get_query_rewriter()
            rewrite_result = await query_rewriter.rewrite(query, conversation)

            resolved_query = rewrite_result.rewritten_query
            if rewrite_result.pronouns_resolved:
                for ref in rewrite_result.pronouns_resolved:
                    ref_str = f"{ref.get('original', '')} -> {ref.get('resolved', '')}"
                    references_resolved.append(ref_str)
                context_used = True

            logger.info(
                "conversation_query_rewritten",
                original=query[:50],
                rewritten=resolved_query[:50],
                pronouns_resolved=len(rewrite_result.pronouns_resolved),
                context_dependent=rewrite_result.is_context_dependent,
            )

            # Step 4: Additional context resolution (legacy patterns)
            _, context, resolved_refs = await self._resolve_references(
                resolved_query, conversation
            )

            if resolved_refs:
                references_resolved.extend(resolved_refs)
                context_used = True

            # Step 5: Analyze intent
            logger.info("conversation_analyzing_intent", query=resolved_query[:50])
            intent_agent = await get_intent_agent()
            intent_analysis = await intent_agent.execute(resolved_query)

            logger.info(
                "conversation_intent_analyzed",
                primary_intent=intent_analysis.primary_intent.value,
                complexity=intent_analysis.complexity.value,
                secondary_intents=[i.value for i in intent_analysis.secondary_intents],
                sub_tasks=intent_analysis.sub_tasks,
                confidence=intent_analysis.confidence,
            )

            # Inject context into analysis if needed
            if context_used and conversation.context.products:
                intent_analysis.conversation_context["products"] = conversation.context.products[:5]
                intent_analysis.conversation_context["brands"] = conversation.context.brands
                intent_analysis.conversation_context["categories"] = conversation.context.categories

            # Step 6: Route to appropriate workflow
            use_compound = self._should_use_compound_workflow(intent_analysis)
            logger.info("conversation_routing", use_compound_workflow=use_compound)

            if use_compound:
                logger.info("conversation_executing_compound_workflow")
                compound_workflow = await self._get_compound_workflow()
                base_result = await compound_workflow.execute(
                    resolved_query,
                    context={"conversation": conversation.context.__dict__},
                )
                logger.info("conversation_compound_workflow_completed", error=base_result.error)
            else:
                logger.info("conversation_executing_simple_workflow")
                simple_workflow = await self._get_simple_workflow()
                base_result = await simple_workflow.execute(
                    resolved_query,
                    context={"conversation": conversation.context.__dict__},
                )
                logger.info("conversation_simple_workflow_completed", error=base_result.error)

            # Step 7: Update conversation context
            conversation.add_assistant_message(
                base_result.response_text,
                metadata={"products": base_result.products[:5]},
            )

            conversation.update_context(
                products=base_result.products,
                query_type=base_result.intent.value,
                agent=",".join(base_result.agents_used),
            )

            # Save conversation
            manager = await get_conversation_manager()
            await manager.save(conversation)

            # Build result
            result = ConversationWorkflowResult(
                query=query,
                response_text=base_result.response_text,
                intent=base_result.intent,
                products=base_result.products,
                comparison=base_result.comparison,
                analysis=base_result.analysis,
                recommendations=base_result.recommendations,
                format=base_result.format,
                confidence=base_result.confidence,
                suggestions=self._generate_contextual_suggestions(
                    base_result, conversation
                ),
                execution_time_ms=(time.time() - start_time) * 1000,
                agents_used=base_result.agents_used,
                error=base_result.error,
                session_id=conversation.session_id,
                turn_number=turn_number,
                context_used=context_used,
                references_resolved=references_resolved,
            )

            return result

        except Exception as e:
            logger.error("conversation_workflow_failed", error=str(e))
            return ConversationWorkflowResult(
                query=query,
                response_text=f"I encountered an error: {str(e)}",
                intent=QueryIntent.SEARCH,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                session_id=session_id or "",
            )

    async def _resolve_references(
        self,
        query: str,
        conversation: Conversation,
    ) -> tuple[str, dict[str, Any], list[str]]:
        """Resolve references in query using conversation context.

        Args:
            query: Original query
            conversation: Conversation with history

        Returns:
            Tuple of (resolved_query, context, list_of_resolved_references)
        """
        resolved_query = query
        context: dict[str, Any] = {}
        resolved_refs: list[str] = []

        if not conversation.context.products:
            return resolved_query, context, resolved_refs

        products = conversation.context.products
        query_lower = query.lower()

        # Check for "them/these/those" references
        if re.search(REFERENCE_PATTERNS["them"], query_lower):
            # User is referring to previous products
            context["referenced_products"] = products[:5]
            resolved_refs.append("them -> previous products")

        # Check for ordinal references (first, second, third)
        if re.search(REFERENCE_PATTERNS["first"], query_lower):
            if products:
                context["referenced_products"] = [products[0]]
                product_name = products[0].get("title", "")[:30]
                resolved_refs.append(f"first -> {product_name}")

        if re.search(REFERENCE_PATTERNS["second"], query_lower):
            if len(products) > 1:
                context["referenced_products"] = [products[1]]
                product_name = products[1].get("title", "")[:30]
                resolved_refs.append(f"second -> {product_name}")

        if re.search(REFERENCE_PATTERNS["third"], query_lower):
            if len(products) > 2:
                context["referenced_products"] = [products[2]]
                product_name = products[2].get("title", "")[:30]
                resolved_refs.append(f"third -> {product_name}")

        # Check for "top N" references
        top_n_match = re.search(REFERENCE_PATTERNS["top_n"], query_lower)
        if top_n_match:
            n = int(top_n_match.group(1))
            context["referenced_products"] = products[:n]
            resolved_refs.append(f"top {n} -> first {n} products")

        # Check for "cheapest" reference
        if re.search(REFERENCE_PATTERNS["cheapest"], query_lower):
            priced = [p for p in products if p.get("price")]
            if priced:
                cheapest = min(priced, key=lambda p: p.get("price", float("inf")))
                context["referenced_products"] = [cheapest]
                resolved_refs.append(f"cheapest -> {cheapest.get('title', '')[:30]}")

        # Check for "best" reference (highest rated)
        if re.search(REFERENCE_PATTERNS["best"], query_lower):
            rated = [p for p in products if p.get("stars")]
            if rated:
                best = max(rated, key=lambda p: p.get("stars", 0))
                context["referenced_products"] = [best]
                resolved_refs.append(f"best -> {best.get('title', '')[:30]}")

        # Check for "compare them"
        if re.search(REFERENCE_PATTERNS["compare_them"], query_lower):
            context["referenced_products"] = products[:5]
            # Enhance query with product names for comparison
            product_names = [p.get("title", "")[:30] for p in products[:3]]
            if product_names:
                resolved_query = f"Compare {', '.join(product_names)}"
                resolved_refs.append(f"compare them -> compare {len(product_names)} products")

        # Check for "more" / "similar"
        if re.search(REFERENCE_PATTERNS["more"], query_lower):
            if products:
                # Use first product as reference for similarity
                context["reference_product"] = products[0]
                resolved_refs.append("more -> similar to previous results")

        return resolved_query, context, resolved_refs

    def _should_use_compound_workflow(self, intent_analysis: IntentAnalysis) -> bool:
        """Determine if compound workflow should be used.

        Args:
            intent_analysis: Analyzed intent

        Returns:
            True if compound workflow is appropriate
        """
        # Use compound workflow for:
        # 1. Compound complexity queries
        if intent_analysis.complexity == QueryComplexity.COMPOUND:
            return True

        # 2. Multiple intents detected
        if len(intent_analysis.secondary_intents) > 0:
            return True

        # 3. Multiple sub-tasks
        if len(intent_analysis.sub_tasks) > 1:
            return True

        return False

    def _generate_contextual_suggestions(
        self,
        result: WorkflowResult,
        conversation: Conversation,
    ) -> list[str]:
        """Generate follow-up suggestions based on context.

        Args:
            result: Workflow result
            conversation: Conversation context

        Returns:
            List of contextual suggestions
        """
        suggestions = list(result.suggestions)  # Start with existing suggestions

        # Add contextual suggestions based on intent and results
        if result.intent == QueryIntent.SEARCH and result.products:
            if len(result.products) >= 2:
                suggestions.append("Compare the top products")
            suggestions.append("Show me alternatives")

        elif result.intent == QueryIntent.COMPARE:
            suggestions.append("Tell me more about the winner")
            suggestions.append("Show me the best value option")

        elif result.intent == QueryIntent.RECOMMEND:
            suggestions.append("Why is this recommended?")
            suggestions.append("Show me alternatives")

        # Add brand-specific suggestions
        if conversation.context.brands:
            brand = conversation.context.brands[0]
            if f"{brand}" not in " ".join(suggestions):
                suggestions.append(f"More products from {brand}")

        # Limit suggestions
        return suggestions[:4]


# Singleton instance
_conversation_workflow: ConversationWorkflow | None = None


async def get_conversation_workflow() -> ConversationWorkflow:
    """Get or create conversation workflow singleton."""
    global _conversation_workflow
    if _conversation_workflow is None:
        _conversation_workflow = ConversationWorkflow()
    return _conversation_workflow


async def execute_conversation_query(
    query: str,
    session_id: str | None = None,
) -> ConversationWorkflowResult:
    """Convenience function to execute a conversation query.

    Args:
        query: User query
        session_id: Optional session ID

    Returns:
        ConversationWorkflowResult
    """
    workflow = await get_conversation_workflow()
    return await workflow.execute(query, session_id)
