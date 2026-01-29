"""Pronoun resolver for multi-turn conversation context.

Resolves pronouns and references to entities in conversation context:
- Singular: it, this, that
- Plural: them, these, those
- Possessive: its, their
- Ordinal: first, second, third, last
- Superlative: cheapest, best, most expensive, highest rated
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.conversation.models import (
    ConversationContext,
    ProductContext,
    ReferenceType,
    ResolvedReference,
)

logger = structlog.get_logger()


class PronounPattern(str, Enum):
    """Patterns for pronoun detection."""
    # Singular references
    IT = r"\b(it)\b"
    THIS = r"\b(this)\b(?!\s+(?:one|product|item))"
    THAT = r"\b(that)\b(?!\s+(?:one|product|item))"
    THIS_ONE = r"\b(this\s+(?:one|product|item))\b"
    THAT_ONE = r"\b(that\s+(?:one|product|item))\b"

    # Plural references
    THEM = r"\b(them)\b"
    THESE = r"\b(these)\b(?!\s+(?:products|items))"
    THOSE = r"\b(those)\b(?!\s+(?:products|items))"
    THESE_PRODUCTS = r"\b(these\s+(?:products|items))\b"
    THOSE_PRODUCTS = r"\b(those\s+(?:products|items))\b"

    # Possessive references
    ITS = r"\b(its)\b"
    THEIR = r"\b(their)\b"

    # Ordinal references
    FIRST = r"\b(?:the\s+)?(first)\s*(?:one|product|item)?\b"
    SECOND = r"\b(?:the\s+)?(second)\s*(?:one|product|item)?\b"
    THIRD = r"\b(?:the\s+)?(third)\s*(?:one|product|item)?\b"
    FOURTH = r"\b(?:the\s+)?(fourth)\s*(?:one|product|item)?\b"
    FIFTH = r"\b(?:the\s+)?(fifth)\s*(?:one|product|item)?\b"
    LAST = r"\b(?:the\s+)?(last)\s*(?:one|product|item)?\b"
    NUMBER = r"\b(?:the\s+)?#?(\d+)(?:st|nd|rd|th)?\s*(?:one|product|item)?\b"

    # Superlative references
    CHEAPEST = r"\b(?:the\s+)?(cheapest|lowest\s+priced?|least\s+expensive)\b"
    MOST_EXPENSIVE = r"\b(?:the\s+)?(most\s+expensive|highest\s+priced?|priciest)\b"
    BEST = r"\b(?:the\s+)?(best|highest\s+rated?|top\s+rated?)\b"
    WORST = r"\b(?:the\s+)?(worst|lowest\s+rated?)\b"


@dataclass
class ResolutionResult:
    """Result of resolving references in a query."""
    original_query: str
    resolved_query: str
    references: list[ResolvedReference] = field(default_factory=list)
    has_unresolved: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "resolved_query": self.resolved_query,
            "references": [
                {
                    "original": r.original,
                    "resolved": r.resolved,
                    "type": r.reference_type.value,
                    "confidence": r.confidence,
                    "source": r.source,
                }
                for r in self.references
            ],
            "has_unresolved": self.has_unresolved,
            "confidence": self.confidence,
        }


class PronounResolver:
    """Resolves pronouns and references in user queries.

    Uses conversation context to resolve:
    - "it" / "this" / "that" → last mentioned or focused product
    - "them" / "these" / "those" → last search results
    - "its price" / "their prices" → product prices
    - "first" / "second" / "last" → ordinal position in results
    - "cheapest" / "best" → superlative from results

    Example:
        User: "find headphones"
        Assistant: [returns 5 products]
        User: "compare the first two"  → resolved to specific products
        User: "what's the cheapest?"  → resolved to cheapest product
        User: "tell me more about it"  → resolved to last focused product
    """

    # Priority order for pronoun patterns
    PATTERN_PRIORITY = [
        # Superlatives (most specific)
        (PronounPattern.CHEAPEST, ReferenceType.SUPERLATIVE),
        (PronounPattern.MOST_EXPENSIVE, ReferenceType.SUPERLATIVE),
        (PronounPattern.BEST, ReferenceType.SUPERLATIVE),
        (PronounPattern.WORST, ReferenceType.SUPERLATIVE),
        # Ordinals
        (PronounPattern.FIRST, ReferenceType.ORDINAL),
        (PronounPattern.SECOND, ReferenceType.ORDINAL),
        (PronounPattern.THIRD, ReferenceType.ORDINAL),
        (PronounPattern.FOURTH, ReferenceType.ORDINAL),
        (PronounPattern.FIFTH, ReferenceType.ORDINAL),
        (PronounPattern.LAST, ReferenceType.ORDINAL),
        (PronounPattern.NUMBER, ReferenceType.ORDINAL),
        # Demonstratives with noun
        (PronounPattern.THIS_ONE, ReferenceType.DEMONSTRATIVE),
        (PronounPattern.THAT_ONE, ReferenceType.DEMONSTRATIVE),
        (PronounPattern.THESE_PRODUCTS, ReferenceType.PLURAL),
        (PronounPattern.THOSE_PRODUCTS, ReferenceType.PLURAL),
        # Possessives
        (PronounPattern.ITS, ReferenceType.POSSESSIVE),
        (PronounPattern.THEIR, ReferenceType.POSSESSIVE),
        # Plural
        (PronounPattern.THEM, ReferenceType.PLURAL),
        (PronounPattern.THESE, ReferenceType.PLURAL),
        (PronounPattern.THOSE, ReferenceType.PLURAL),
        # Singular (least specific)
        (PronounPattern.IT, ReferenceType.SINGULAR),
        (PronounPattern.THIS, ReferenceType.SINGULAR),
        (PronounPattern.THAT, ReferenceType.SINGULAR),
    ]

    # Ordinal word to number mapping
    ORDINAL_MAP = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "last": -1,  # Special handling
    }

    def __init__(self, use_llm_fallback: bool = True):
        """Initialize resolver.

        Args:
            use_llm_fallback: Whether to use LLM for ambiguous cases
        """
        self.use_llm_fallback = use_llm_fallback
        self._llm_manager = None

    async def initialize(self) -> None:
        """Initialize LLM manager for fallback resolution."""
        if self.use_llm_fallback:
            try:
                from src.llm import get_llm_manager
                self._llm_manager = await get_llm_manager()
            except Exception as e:
                logger.warning("pronoun_resolver_llm_init_failed", error=str(e))

    def resolve(
        self,
        query: str,
        context: ConversationContext,
    ) -> ResolutionResult:
        """Resolve pronouns and references in a query.

        Args:
            query: User query with potential pronouns
            context: Conversation context with products

        Returns:
            ResolutionResult with resolved query and references
        """
        result = ResolutionResult(
            original_query=query,
            resolved_query=query,
        )

        if not context.products and not context.last_search_results:
            # No context to resolve against
            result.has_unresolved = self._has_pronouns(query)
            return result

        # Find and resolve all references
        resolved_query = query
        total_confidence = 1.0

        for pattern, ref_type in self.PATTERN_PRIORITY:
            matches = list(re.finditer(pattern.value, resolved_query, re.IGNORECASE))

            for match in matches:
                original_text = match.group(0)

                # Resolve based on type
                resolution = self._resolve_reference(
                    original_text=original_text,
                    ref_type=ref_type,
                    pattern=pattern,
                    context=context,
                    match=match,
                )

                if resolution:
                    result.references.append(resolution)
                    total_confidence *= resolution.confidence

                    # Replace in query
                    resolved_query = resolved_query.replace(
                        original_text,
                        resolution.resolved,
                        1,  # Only first occurrence
                    )
                else:
                    result.has_unresolved = True

        result.resolved_query = resolved_query
        result.confidence = total_confidence

        logger.debug(
            "pronoun_resolution_complete",
            original=query[:50],
            resolved=resolved_query[:50],
            num_references=len(result.references),
            confidence=result.confidence,
        )

        return result

    def _has_pronouns(self, query: str) -> bool:
        """Check if query contains any pronouns to resolve."""
        for pattern, _ in self.PATTERN_PRIORITY:
            if re.search(pattern.value, query, re.IGNORECASE):
                return True
        return False

    def _resolve_reference(
        self,
        original_text: str,
        ref_type: ReferenceType,
        pattern: PronounPattern,
        context: ConversationContext,
        match: re.Match,
    ) -> ResolvedReference | None:
        """Resolve a single reference.

        Args:
            original_text: The original reference text
            ref_type: Type of reference
            pattern: Pattern that matched
            context: Conversation context
            match: Regex match object

        Returns:
            ResolvedReference or None if unresolvable
        """
        product: ProductContext | None = None
        resolved_text: str = ""
        confidence: float = 0.8

        # Resolve based on type
        if ref_type == ReferenceType.SUPERLATIVE:
            product = self._resolve_superlative(pattern, context)
            if product:
                resolved_text = f'"{product.title}"'
                confidence = 0.9

        elif ref_type == ReferenceType.ORDINAL:
            position = self._get_ordinal_position(pattern, match, context)
            if position:
                product = context.get_product_by_position(position)
                if product:
                    resolved_text = f'"{product.title}"'
                    confidence = 0.95

        elif ref_type == ReferenceType.SINGULAR or ref_type == ReferenceType.DEMONSTRATIVE:
            # Resolve to last mentioned or focused product
            product = context.get_last_mentioned_product()
            if not product and context.last_search_results:
                product = context.last_search_results[0]
            if product:
                resolved_text = f'"{product.title}"'
                confidence = 0.7

        elif ref_type == ReferenceType.PLURAL:
            # Resolve to all products in last search
            if context.last_search_results:
                titles = [f'"{p.title}"' for p in context.last_search_results[:5]]
                resolved_text = ", ".join(titles)
                confidence = 0.85

        elif ref_type == ReferenceType.POSSESSIVE:
            # Handle possessive references like "its price"
            product = context.get_last_mentioned_product()
            if not product and context.last_search_results:
                product = context.last_search_results[0]
            if product:
                # Keep possessive context but add product reference
                if pattern == PronounPattern.ITS:
                    resolved_text = f'"{product.title}"\'s'
                else:  # THEIR
                    titles = [p.title for p in context.last_search_results[:3]]
                    resolved_text = " and ".join([f'"{t}"' for t in titles]) + "'s"
                confidence = 0.75

        if not resolved_text:
            return None

        return ResolvedReference(
            original=original_text,
            resolved=resolved_text,
            reference_type=ref_type,
            confidence=confidence,
            source="rule",
            product_asin=product.asin if product else None,
            product_title=product.title if product else None,
        )

    def _resolve_superlative(
        self,
        pattern: PronounPattern,
        context: ConversationContext,
    ) -> ProductContext | None:
        """Resolve superlative reference."""
        if pattern == PronounPattern.CHEAPEST:
            return context.get_cheapest_product()
        elif pattern == PronounPattern.MOST_EXPENSIVE:
            return context.get_most_expensive_product()
        elif pattern == PronounPattern.BEST:
            return context.get_best_rated_product()
        elif pattern == PronounPattern.WORST:
            # Get lowest rated
            products_with_rating = [p for p in context.last_search_results if p.stars is not None]
            if products_with_rating:
                return min(products_with_rating, key=lambda p: p.stars)
        return None

    def _get_ordinal_position(
        self,
        pattern: PronounPattern,
        match: re.Match,
        context: ConversationContext,
    ) -> int | None:
        """Get position from ordinal reference."""
        if pattern == PronounPattern.NUMBER:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return None

        ordinal_word = match.group(1).lower()

        if ordinal_word == "last":
            return len(context.last_search_results) if context.last_search_results else None

        return self.ORDINAL_MAP.get(ordinal_word)

    async def resolve_with_llm_fallback(
        self,
        query: str,
        context: ConversationContext,
    ) -> ResolutionResult:
        """Resolve with LLM fallback for ambiguous cases.

        Args:
            query: User query
            context: Conversation context

        Returns:
            ResolutionResult with resolved query
        """
        # First try rule-based resolution
        result = self.resolve(query, context)

        # If there are unresolved references and LLM is available
        if result.has_unresolved and self._llm_manager and self.use_llm_fallback:
            llm_result = await self._resolve_with_llm(query, context)
            if llm_result:
                return llm_result

        return result

    async def _resolve_with_llm(
        self,
        query: str,
        context: ConversationContext,
    ) -> ResolutionResult | None:
        """Use LLM to resolve ambiguous references."""
        if not self._llm_manager:
            return None

        # Build context for LLM
        products_context = []
        for i, p in enumerate(context.last_search_results[:5], 1):
            products_context.append(
                f"{i}. {p.title} (${p.price:.2f}, {p.stars}★)" if p.price else f"{i}. {p.title}"
            )

        prompt = f"""Given this conversation context:
Last search results:
{chr(10).join(products_context)}

User query: "{query}"

Rewrite the query to replace any pronouns (it, them, these, first, cheapest, etc.) with the actual product names from the context.
Only output the rewritten query, nothing else.
If no pronouns need resolution, output the original query unchanged."""

        try:
            from src.llm import GenerationConfig

            response = await self._llm_manager.generate_for_agent(
                agent_name="pronoun_resolver",
                prompt=prompt,
                config=GenerationConfig(
                    temperature=0.1,
                    max_tokens=200,
                ),
            )

            resolved_query = response.content.strip().strip('"\'')

            return ResolutionResult(
                original_query=query,
                resolved_query=resolved_query,
                references=[
                    ResolvedReference(
                        original=query,
                        resolved=resolved_query,
                        reference_type=ReferenceType.SINGULAR,
                        confidence=0.7,
                        source="llm",
                    )
                ],
                has_unresolved=False,
                confidence=0.7,
            )

        except Exception as e:
            logger.warning("pronoun_llm_resolution_failed", error=str(e))
            return None


# Singleton
_resolver: PronounResolver | None = None


async def get_pronoun_resolver() -> PronounResolver:
    """Get or create pronoun resolver singleton."""
    global _resolver
    if _resolver is None:
        _resolver = PronounResolver()
        await _resolver.initialize()
    return _resolver
