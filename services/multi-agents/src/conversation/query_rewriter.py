"""Query rewriter for multi-turn conversation context.

Rewrites queries to:
1. Resolve pronouns to actual entities
2. Inject context from previous turns
3. Make queries self-contained for downstream agents
"""

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.conversation.models import ConversationContext, Conversation
from src.conversation.pronoun_resolver import PronounResolver, ResolutionResult

logger = structlog.get_logger()


@dataclass
class RewriteResult:
    """Result of query rewriting."""
    original_query: str
    rewritten_query: str

    # Resolution details
    pronouns_resolved: list[dict] = field(default_factory=list)
    context_injected: bool = False

    # Flags
    is_context_dependent: bool = False  # Query depends on previous context
    needs_products: bool = False  # Query needs products from context
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "pronouns_resolved": self.pronouns_resolved,
            "context_injected": self.context_injected,
            "is_context_dependent": self.is_context_dependent,
            "needs_products": self.needs_products,
            "confidence": self.confidence,
        }


class QueryRewriter:
    """Rewrites queries using conversation context.

    Handles:
    1. Pronoun resolution (it, them, these, first, cheapest)
    2. Ellipsis resolution ("compare them" → "compare [products]")
    3. Context injection for follow-up queries

    Example flow:
        Turn 1: "find wireless headphones"
        Turn 2: "compare the first two" → "compare [product1] and [product2]"
        Turn 3: "which is cheaper?" → "which is cheaper between [product1] and [product2]"
        Turn 4: "show me its reviews" → "show me [cheapest product] reviews"
    """

    # Patterns that indicate context dependency
    CONTEXT_PATTERNS = [
        r"\b(compare|versus|vs\.?)\s+(them|these|those|it)\b",
        r"\b(its|their)\s+(price|rating|review|spec)",
        r"\b(the|which)\s+(cheapest|best|first|second|last)",
        r"\bmore\s+about\s+(it|them|this|that)\b",
        r"\b(show|tell|give)\s+me\s+(it|them|this|that)\b",
        r"\bwhich\s+(is|one|are)\s+(cheap|cheaper|expensive|best|better|worst)\b",
        r"\b(is|are)\s+(it|they|this|these)\s+(cheap|cheaper|expensive|better|good)\b",
    ]

    # Patterns for comparison queries
    COMPARE_PATTERNS = [
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+(is|one|should)\b",
    ]

    def __init__(self, pronoun_resolver: PronounResolver | None = None):
        """Initialize query rewriter.

        Args:
            pronoun_resolver: Resolver for pronouns (created if None)
        """
        self._pronoun_resolver = pronoun_resolver
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the rewriter."""
        if self._pronoun_resolver is None:
            self._pronoun_resolver = PronounResolver()
            await self._pronoun_resolver.initialize()
        self._initialized = True

    async def rewrite(
        self,
        query: str,
        conversation: Conversation,
    ) -> RewriteResult:
        """Rewrite a query using conversation context.

        Args:
            query: Original user query
            conversation: Full conversation with context

        Returns:
            RewriteResult with rewritten query
        """
        if not self._initialized:
            await self.initialize()

        result = RewriteResult(
            original_query=query,
            rewritten_query=query,
        )

        context = conversation.context

        # Check if query is context-dependent
        result.is_context_dependent = self._is_context_dependent(query)

        # Step 1: Resolve pronouns
        if self._pronoun_resolver:
            resolution = await self._pronoun_resolver.resolve_with_llm_fallback(
                query, context
            )
            result.rewritten_query = resolution.resolved_query
            result.pronouns_resolved = [r.to_dict() if hasattr(r, 'to_dict') else {
                "original": r.original,
                "resolved": r.resolved,
                "confidence": r.confidence,
            } for r in resolution.references]
            result.confidence *= resolution.confidence

        # Step 2: Handle comparison queries without explicit products
        if self._is_comparison_query(query) and not self._has_explicit_products(query):
            rewritten = self._inject_comparison_products(result.rewritten_query, context)
            if rewritten != result.rewritten_query:
                result.rewritten_query = rewritten
                result.context_injected = True
                result.needs_products = True

        # Step 3: Handle price/review queries
        if self._is_price_or_review_query(query) and result.is_context_dependent:
            rewritten = self._inject_product_context(result.rewritten_query, context)
            if rewritten != result.rewritten_query:
                result.rewritten_query = rewritten
                result.context_injected = True

        logger.info(
            "query_rewritten",
            original=query[:50],
            rewritten=result.rewritten_query[:50],
            context_dependent=result.is_context_dependent,
            pronouns_resolved=len(result.pronouns_resolved),
        )

        return result

    def _is_context_dependent(self, query: str) -> bool:
        """Check if query depends on previous context."""
        query_lower = query.lower()

        # Check explicit context patterns
        for pattern in self.CONTEXT_PATTERNS:
            if re.search(pattern, query_lower):
                return True

        # Check for pronouns
        pronoun_patterns = [
            r"\b(it|its|this|that|them|their|these|those)\b",
            r"\b(first|second|third|last|cheapest|best)\b",
        ]
        for pattern in pronoun_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    def _is_comparison_query(self, query: str) -> bool:
        """Check if query is a comparison query."""
        query_lower = query.lower()
        for pattern in self.COMPARE_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _is_price_or_review_query(self, query: str) -> bool:
        """Check if query is about price or reviews."""
        patterns = [
            r"\bprice\b",
            r"\bcost\b",
            r"\bhow\s+much\b",
            r"\breview",
            r"\brating",
            r"\bspec",
            r"\bfeature",
        ]
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in patterns)

    def _has_explicit_products(self, query: str) -> bool:
        """Check if query has explicit product names."""
        # Simple heuristic: check for quoted strings or long capitalized words
        if '"' in query or "'" in query:
            return True

        # Check for brand/model patterns
        brand_pattern = r"\b[A-Z][a-z]+\s+[A-Z0-9]"
        if re.search(brand_pattern, query):
            return True

        return False

    def _inject_comparison_products(
        self,
        query: str,
        context: ConversationContext,
    ) -> str:
        """Inject products for comparison queries."""
        if not context.last_search_results:
            return query

        # Get products to compare (first 2-3 from results)
        products = context.last_search_results[:3]
        product_names = [f'"{p.title}"' for p in products]

        # Handle different comparison patterns
        query_lower = query.lower()

        if "compare them" in query_lower:
            product_list = " and ".join(product_names[:2])
            return query_lower.replace("compare them", f"compare {product_list}")

        if "compare these" in query_lower:
            product_list = ", ".join(product_names)
            return query_lower.replace("compare these", f"compare {product_list}")

        if re.search(r"\bcompare\b", query_lower) and not self._has_explicit_products(query):
            # Generic compare without explicit products
            product_list = " and ".join(product_names[:2])
            return f"compare {product_list}"

        return query

    def _inject_product_context(
        self,
        query: str,
        context: ConversationContext,
    ) -> str:
        """Inject product context for price/review queries."""
        if not context.last_search_results:
            return query

        # Get the focused product (first in results or last mentioned)
        product = context.get_last_mentioned_product()
        if not product and context.last_search_results:
            product = context.last_search_results[0]

        if not product:
            return query

        # Add product context
        query_lower = query.lower()

        # Handle "its price" / "their prices"
        if "its price" in query_lower:
            return query_lower.replace("its price", f'"{product.title}" price')

        if "their price" in query_lower:
            products = context.last_search_results[:3]
            product_names = " and ".join([f'"{p.title}"' for p in products])
            return query_lower.replace("their price", f"{product_names} prices")

        return query


# Singleton
_rewriter: QueryRewriter | None = None


async def get_query_rewriter() -> QueryRewriter:
    """Get or create query rewriter singleton."""
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
        await _rewriter.initialize()
    return _rewriter
