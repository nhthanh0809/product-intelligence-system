"""Synthesis Agent for generating natural language responses.

The Synthesis Agent is responsible for:
1. Combining outputs from multiple agents into coherent responses
2. Generating natural language summaries
3. Supporting multiple output formats (text, table, cards)
4. Adding citations and source attribution
5. Computing confidence scores
6. Suggesting follow-up queries

Enhanced with LLM support for more natural responses.
"""

import json
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, RetryConfig
from src.models.intent import QueryIntent

logger = structlog.get_logger()


class OutputFormat(str, Enum):
    """Supported output formats."""

    TEXT = "text"          # Natural language paragraph
    BULLET = "bullet"      # Bullet point list
    TABLE = "table"        # Structured table
    CARDS = "cards"        # Product cards for UI
    COMPARISON = "comparison"  # Side-by-side comparison


class SynthesisInput(BaseModel):
    """Input for synthesis agent."""

    query: str = Field(description="Original user query")
    intent: QueryIntent = Field(description="Primary intent")

    # Results from other agents
    products: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Products from retrieval",
    )
    analysis: dict[str, Any] | None = Field(
        default=None,
        description="Analysis results",
    )
    comparison: dict[str, Any] | None = Field(
        default=None,
        description="Comparison results",
    )
    price_analysis: dict[str, Any] | None = Field(
        default=None,
        description="Price analysis results",
    )
    trends: dict[str, Any] | None = Field(
        default=None,
        description="Trend data",
    )
    recommendations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Recommendations",
    )

    # Formatting preferences
    preferred_format: OutputFormat | None = Field(
        default=None,
        description="Preferred output format",
    )
    max_products: int = Field(
        default=5,
        description="Maximum products to include",
    )
    include_citations: bool = Field(
        default=True,
        description="Whether to include source citations",
    )


class Citation(BaseModel):
    """Source citation for a piece of information."""

    source_type: str = Field(description="Type of source (product, review, spec)")
    product_id: str | None = Field(default=None, description="Product ASIN if applicable")
    product_name: str | None = Field(default=None, description="Product name")
    field: str | None = Field(default=None, description="Specific field referenced")


class SynthesisOutput(BaseModel):
    """Output from synthesis agent."""

    # Main response
    response_text: str = Field(description="Main response text")
    format: OutputFormat = Field(description="Output format used")

    # Structured data for UI rendering
    products: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Formatted products for display",
    )
    table_data: dict[str, Any] | None = Field(
        default=None,
        description="Table data if format is TABLE",
    )
    comparison_data: dict[str, Any] | None = Field(
        default=None,
        description="Comparison data if format is COMPARISON",
    )

    # Metadata
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score (0-1)",
    )
    result_count: int = Field(
        default=0,
        description="Number of results found",
    )

    # Follow-up
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up queries",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or caveats",
    )


class SynthesisAgent(BaseAgent[SynthesisInput, SynthesisOutput]):
    """Agent for synthesizing responses from multi-agent outputs.

    The synthesis agent takes outputs from retrieval, analysis, comparison,
    and other agents, then generates a cohesive natural language response
    with appropriate formatting, citations, and follow-up suggestions.
    """

    name = "synthesis"
    description = "Generates natural language responses from agent outputs"

    # Templates for different intents
    SEARCH_TEMPLATES = {
        "found": "I found {count} products matching your search for \"{query}\". {summary}",
        "empty": "I couldn't find any products matching \"{query}\". {suggestions}",
        "filtered": "Here are {count} products matching your criteria. {highlights}",
    }

    COMPARE_TEMPLATES = {
        "intro": "Here's a comparison of {count} products based on your query:",
        "winner": "Based on the comparison, {winner} appears to be the best option for {criteria}.",
        "tie": "The products are quite comparable. Your choice depends on priorities.",
    }

    ANALYZE_TEMPLATES = {
        "summary": "Based on analysis of {count} reviews and specs:",
        "sentiment": "Overall sentiment: {sentiment} ({positive}% positive, {negative}% negative)",
        "key_points": "Key findings:",
    }

    RECOMMEND_TEMPLATES = {
        "intro": "Based on your needs, I recommend:",
        "top_pick": "Top recommendation: {product} - {reason}",
        "alternatives": "Alternatives to consider:",
    }

    # Follow-up suggestion templates by intent
    FOLLOW_UP_TEMPLATES = {
        QueryIntent.SEARCH: [
            "Compare top {count} results",
            "Show me more options under ${price}",
            "Filter by {brand} brand",
            "What do reviews say about these?",
        ],
        QueryIntent.COMPARE: [
            "Tell me more about {winner}",
            "Show similar products",
            "What are the pros and cons?",
            "Which has better reviews?",
        ],
        QueryIntent.ANALYZE: [
            "What are common complaints?",
            "Show me the best-rated features",
            "Compare with alternatives",
            "Is this a good value?",
        ],
        QueryIntent.RECOMMEND: [
            "Why this recommendation?",
            "Show me alternatives",
            "Compare with {alternative}",
            "What's the price history?",
        ],
        QueryIntent.PRICE_CHECK: [
            "Is this the best deal?",
            "Show price history",
            "Find cheaper alternatives",
            "Compare prices across brands",
        ],
        QueryIntent.TREND: [
            "Why are these trending?",
            "Compare trending products",
            "Show me hidden gems",
            "What's losing popularity?",
        ],
    }

    # LLM prompt templates for each intent type
    LLM_PROMPTS = {
        QueryIntent.SEARCH: """You are a helpful product assistant. Generate a natural, conversational response for a product search.

Query: "{query}"
Products found: {product_count}
Product summary: {product_summary}

Instructions:
1. Start with a brief, friendly acknowledgment of what was found
2. Highlight key product details (price range, ratings, brands)
3. Keep it concise (2-3 sentences max)
4. Don't list individual products - just summarize
5. Sound natural and helpful, not robotic

Response:""",

        QueryIntent.COMPARE: """You are a helpful product assistant. Generate a natural comparison summary.

Query: "{query}"
Products being compared: {product_names}
Comparison data: {comparison_summary}
Winner (if any): {winner}

Instructions:
1. Briefly introduce what you're comparing
2. Highlight the key differences that matter
3. If there's a clear winner, explain why
4. Keep it objective and helpful
5. 3-4 sentences max

Response:""",

        QueryIntent.RECOMMEND: """You are a helpful product assistant. Generate a natural recommendation response.

Query: "{query}"
Top recommendation: {top_pick}
Reason: {top_reason}
Alternatives: {alternatives}

Instructions:
1. Confidently present your top recommendation
2. Explain why it's the best fit
3. Briefly mention alternatives
4. Be helpful and conversational
5. 3-4 sentences max

Response:""",

        QueryIntent.ANALYZE: """You are a helpful product assistant. Generate a natural analysis summary.

Query: "{query}"
Analysis summary: {analysis_summary}
Key points: {key_points}
Sentiment: {sentiment}

Instructions:
1. Summarize the key findings naturally
2. Mention notable pros and cons
3. Give an overall assessment
4. Be balanced and objective
5. 3-4 sentences max

Response:""",
    }

    def __init__(self, use_llm: bool = True):
        """Initialize the Synthesis Agent.

        Args:
            use_llm: Whether to use LLM for natural language generation
        """
        super().__init__(retry_config=RetryConfig(max_retries=2))
        self.use_llm = use_llm
        self._llm_manager = None

    async def initialize(self) -> None:
        """Initialize LLM manager if enabled."""
        if self.use_llm:
            try:
                from src.llm import get_llm_manager
                self._llm_manager = await get_llm_manager()
                logger.info("synthesis_agent_llm_initialized")
            except Exception as e:
                logger.warning(
                    "synthesis_agent_llm_init_failed",
                    error=str(e),
                    fallback="template-based",
                )
                self._llm_manager = None
        await super().initialize()

    async def _execute_internal(self, input_data: SynthesisInput) -> SynthesisOutput:
        """Generate synthesized response from agent outputs.

        Uses LLM for natural language generation when available,
        with template-based fallback.

        Args:
            input_data: SynthesisInput with all agent outputs

        Returns:
            SynthesisOutput with formatted response
        """
        logger.info(
            "synthesis_execute_started",
            query=input_data.query[:50],
            intent=input_data.intent.value,
            num_products=len(input_data.products),
            has_comparison=input_data.comparison is not None,
        )

        # Determine best output format
        output_format = self._select_format(input_data)
        logger.info("synthesis_format_selected", format=output_format.value)

        # Try LLM synthesis first if enabled and available
        should_use_llm = self._should_use_llm()
        logger.info("synthesis_llm_check", should_use_llm=should_use_llm)

        if should_use_llm:
            llm_response = await self._llm_synthesize(input_data)
            if llm_response:
                # Enhance template response with LLM-generated text
                base_output = self._get_base_output(input_data, output_format)
                base_output.response_text = llm_response

                # Generate LLM follow-ups if we have context
                llm_suggestions = await self._llm_generate_follow_ups(input_data)
                if llm_suggestions:
                    base_output.suggestions = llm_suggestions

                return base_output

        # Fallback to template-based generation
        if input_data.intent == QueryIntent.COMPARE and input_data.comparison:
            return self._synthesize_comparison(input_data, output_format)
        elif input_data.intent == QueryIntent.ANALYZE and input_data.analysis:
            return self._synthesize_analysis(input_data, output_format)
        elif input_data.intent == QueryIntent.RECOMMEND and input_data.recommendations:
            return self._synthesize_recommendations(input_data, output_format)
        elif input_data.intent == QueryIntent.PRICE_CHECK and input_data.price_analysis:
            return self._synthesize_price_analysis(input_data, output_format)
        elif input_data.intent == QueryIntent.TREND and input_data.trends:
            return self._synthesize_trends(input_data, output_format)
        else:
            # Default: search/discovery response
            return self._synthesize_search(input_data, output_format)

    def _should_use_llm(self) -> bool:
        """Check if LLM should be used for synthesis."""
        return (
            self.use_llm
            and self._llm_manager is not None
            and self._llm_manager.is_initialized
        )

    async def _llm_synthesize(self, input_data: SynthesisInput) -> str | None:
        """Generate natural language response using LLM.

        Args:
            input_data: Input with query and agent outputs

        Returns:
            Generated response text or None if LLM fails
        """
        if not self._llm_manager:
            return None

        # Select prompt template based on intent
        prompt_template = self.LLM_PROMPTS.get(input_data.intent)
        if not prompt_template:
            # Use search template as default
            prompt_template = self.LLM_PROMPTS[QueryIntent.SEARCH]

        # Build context for prompt
        products = input_data.products[:input_data.max_products]
        context = self._build_llm_context(input_data, products)

        try:
            prompt = prompt_template.format(**context)
        except KeyError as e:
            logger.warning("llm_prompt_format_failed", error=str(e))
            return None

        try:
            from src.llm import GenerationConfig

            logger.info("synthesis_llm_calling", prompt_length=len(prompt))
            result = await self._llm_manager.generate_for_agent(
                agent_name="synthesis",
                prompt=prompt,
                config=GenerationConfig(
                    temperature=0.7,
                    max_tokens=300,
                ),
            )
            logger.info(
                "synthesis_llm_response_received",
                response_length=len(result.content),
                model_used=result.model if hasattr(result, 'model') else 'unknown',
            )

            response_text = result.content.strip()

            # Clean up response (remove quotes, extra whitespace)
            response_text = response_text.strip('"\'')
            response_text = " ".join(response_text.split())

            logger.debug(
                "llm_synthesis_complete",
                intent=input_data.intent.value,
                response_length=len(response_text),
            )

            return response_text

        except Exception as e:
            logger.warning("llm_synthesis_failed", error=str(e))
            return None

    def _build_llm_context(
        self, input_data: SynthesisInput, products: list[dict]
    ) -> dict[str, Any]:
        """Build context dictionary for LLM prompts.

        Args:
            input_data: Input with all agent outputs
            products: Filtered product list

        Returns:
            Context dict for prompt formatting
        """
        context = {
            "query": input_data.query,
            "product_count": len(products),
            "product_summary": self._build_product_summary(products),
        }

        # Add product names
        product_names = [p.get("title", "Unknown")[:50] for p in products[:5]]
        context["product_names"] = ", ".join(product_names) if product_names else "None"

        # Add comparison data
        if input_data.comparison:
            winner = input_data.comparison.get("winner") or {}
            context["winner"] = winner.get("name", "No clear winner")
            context["comparison_summary"] = json.dumps(
                input_data.comparison.get("differences", [])[:3]
            )
        else:
            context["winner"] = "N/A"
            context["comparison_summary"] = "N/A"

        # Add recommendation data
        if input_data.recommendations:
            top = input_data.recommendations[0] if input_data.recommendations else {}
            context["top_pick"] = top.get("name", top.get("title", "N/A"))
            context["top_reason"] = top.get("reason", "Best match")
            alternatives = [
                r.get("name", r.get("title", ""))
                for r in input_data.recommendations[1:4]
            ]
            context["alternatives"] = ", ".join(alternatives) if alternatives else "None"
        else:
            context["top_pick"] = "N/A"
            context["top_reason"] = "N/A"
            context["alternatives"] = "N/A"

        # Add analysis data
        if input_data.analysis:
            context["analysis_summary"] = input_data.analysis.get("summary", "N/A")
            context["key_points"] = ", ".join(
                input_data.analysis.get("key_points", [])[:3]
            )
            sentiment = input_data.analysis.get("sentiment") or {}
            context["sentiment"] = sentiment.get("overall", "mixed")
        else:
            context["analysis_summary"] = "N/A"
            context["key_points"] = "N/A"
            context["sentiment"] = "N/A"

        return context

    async def _llm_generate_follow_ups(
        self, input_data: SynthesisInput
    ) -> list[str] | None:
        """Generate follow-up suggestions using LLM.

        Args:
            input_data: Input with query and results

        Returns:
            List of follow-up suggestions or None
        """
        if not self._llm_manager:
            return None

        products = input_data.products[:3]
        product_context = ", ".join(
            p.get("title", "product")[:30] for p in products
        )

        prompt = f"""Given this product search context, suggest 3-4 natural follow-up questions a user might ask.

Query: "{input_data.query}"
Intent: {input_data.intent.value}
Products found: {product_context or "None"}

Generate short, natural questions. Output as JSON array only:
["question1", "question2", "question3"]"""

        try:
            from src.llm import GenerationConfig

            result = await self._llm_manager.generate_for_agent(
                agent_name="synthesis",
                prompt=prompt,
                config=GenerationConfig(
                    temperature=0.8,
                    max_tokens=150,
                    response_format="json",
                ),
            )

            # Parse JSON response
            import re
            json_match = re.search(r"\[[\s\S]*\]", result.content)
            if json_match:
                suggestions = json.loads(json_match.group())
                return suggestions[:4]

        except Exception as e:
            logger.debug("llm_follow_up_generation_failed", error=str(e))

        return None

    def _get_base_output(
        self, input_data: SynthesisInput, output_format: OutputFormat
    ) -> SynthesisOutput:
        """Get base output structure with formatted products and metadata.

        Args:
            input_data: Input with all agent outputs
            output_format: Selected output format

        Returns:
            SynthesisOutput with everything except response_text
        """
        products = input_data.products[:input_data.max_products]
        formatted_products = self._format_products(products, output_format)

        return SynthesisOutput(
            response_text="",  # To be filled by LLM
            format=output_format,
            products=formatted_products,
            table_data=self._build_table_data(products) if output_format == OutputFormat.TABLE else None,
            comparison_data=self._build_comparison_data(products, input_data.comparison or {}) if output_format == OutputFormat.COMPARISON else None,
            citations=self._build_citations(products) if input_data.include_citations else [],
            confidence=self._calculate_confidence(products, input_data.query),
            result_count=len(products),
            suggestions=self._generate_follow_ups(
                input_data.intent, products, input_data.query
            ),
        )

    def _select_format(self, input_data: SynthesisInput) -> OutputFormat:
        """Select the best output format based on intent and data.

        Args:
            input_data: Input with intent and preferences

        Returns:
            Best OutputFormat for this response
        """
        # Respect user preference if specified
        if input_data.preferred_format:
            return input_data.preferred_format

        # Auto-select based on intent
        intent = input_data.intent

        if intent == QueryIntent.COMPARE:
            return OutputFormat.COMPARISON
        elif intent == QueryIntent.ANALYZE:
            return OutputFormat.BULLET
        elif intent == QueryIntent.TREND:
            return OutputFormat.CARDS
        elif len(input_data.products) <= 3:
            return OutputFormat.CARDS
        elif len(input_data.products) > 5:
            return OutputFormat.TABLE
        else:
            return OutputFormat.TEXT

    def _synthesize_search(
        self,
        input_data: SynthesisInput,
        output_format: OutputFormat,
    ) -> SynthesisOutput:
        """Synthesize response for search/discovery queries."""
        products = input_data.products[:input_data.max_products]
        count = len(products)

        if count == 0:
            response_text = self.SEARCH_TEMPLATES["empty"].format(
                query=input_data.query,
                suggestions="Try broadening your search criteria or using different keywords.",
            )
            suggestions = [
                "Try a more general search",
                "Remove some filters",
                "Search for similar products",
            ]
        else:
            # Build summary
            summary = self._build_product_summary(products)
            response_text = self.SEARCH_TEMPLATES["found"].format(
                count=count,
                query=input_data.query,
                summary=summary,
            )
            suggestions = self._generate_follow_ups(
                QueryIntent.SEARCH, products, input_data.query
            )

        # Format products for display
        formatted_products = self._format_products(products, output_format)

        # Build citations
        citations = self._build_citations(products) if input_data.include_citations else []

        # Calculate confidence
        confidence = self._calculate_confidence(products, input_data.query)

        return SynthesisOutput(
            response_text=response_text,
            format=output_format,
            products=formatted_products,
            table_data=self._build_table_data(products) if output_format == OutputFormat.TABLE else None,
            citations=citations,
            confidence=confidence,
            result_count=count,
            suggestions=suggestions,
        )

    def _synthesize_comparison(
        self,
        input_data: SynthesisInput,
        output_format: OutputFormat,
    ) -> SynthesisOutput:
        """Synthesize response for comparison queries."""
        comparison = input_data.comparison or {}
        products = input_data.products[:input_data.max_products]
        count = len(products)

        # Build comparison intro
        response_parts = [
            self.COMPARE_TEMPLATES["intro"].format(count=count)
        ]

        # Add winner if determined
        winner = comparison.get("winner")
        if winner:
            response_parts.append(
                self.COMPARE_TEMPLATES["winner"].format(
                    winner=winner.get("name", "the top product"),
                    criteria=comparison.get("criteria", "your needs"),
                )
            )
        else:
            response_parts.append(self.COMPARE_TEMPLATES["tie"])

        # Add key differences
        differences = comparison.get("differences", [])
        if differences:
            response_parts.append("\nKey differences:")
            for diff in differences[:5]:
                response_parts.append(f"- {diff}")

        response_text = "\n".join(response_parts)

        # Build comparison data structure
        comparison_data = self._build_comparison_data(products, comparison)

        suggestions = self._generate_follow_ups(
            QueryIntent.COMPARE, products, input_data.query
        )

        return SynthesisOutput(
            response_text=response_text,
            format=OutputFormat.COMPARISON,
            products=self._format_products(products, OutputFormat.COMPARISON),
            comparison_data=comparison_data,
            citations=self._build_citations(products) if input_data.include_citations else [],
            confidence=self._calculate_confidence(products, input_data.query),
            result_count=count,
            suggestions=suggestions,
        )

    def _synthesize_analysis(
        self,
        input_data: SynthesisInput,
        output_format: OutputFormat,
    ) -> SynthesisOutput:
        """Synthesize response for analysis queries."""
        analysis = input_data.analysis or {}
        products = input_data.products[:input_data.max_products]

        response_parts = [
            self.ANALYZE_TEMPLATES["summary"].format(
                count=analysis.get("review_count", len(products))
            )
        ]

        # Add sentiment if available
        sentiment = analysis.get("sentiment") or {}
        if sentiment:
            response_parts.append(
                self.ANALYZE_TEMPLATES["sentiment"].format(
                    sentiment=sentiment.get("overall", "mixed"),
                    positive=sentiment.get("positive_pct", 50),
                    negative=sentiment.get("negative_pct", 50),
                )
            )

        # Add key points
        key_points = analysis.get("key_points", [])
        if key_points:
            response_parts.append(self.ANALYZE_TEMPLATES["key_points"])
            for point in key_points[:5]:
                response_parts.append(f"- {point}")

        # Add pros/cons if available
        pros = analysis.get("pros", [])
        cons = analysis.get("cons", [])
        if pros:
            response_parts.append("\nPros:")
            for pro in pros[:3]:
                response_parts.append(f"+ {pro}")
        if cons:
            response_parts.append("\nCons:")
            for con in cons[:3]:
                response_parts.append(f"- {con}")

        response_text = "\n".join(response_parts)

        suggestions = self._generate_follow_ups(
            QueryIntent.ANALYZE, products, input_data.query
        )

        return SynthesisOutput(
            response_text=response_text,
            format=OutputFormat.BULLET,
            products=self._format_products(products, output_format),
            citations=self._build_citations(products) if input_data.include_citations else [],
            confidence=self._calculate_confidence(products, input_data.query),
            result_count=len(products),
            suggestions=suggestions,
        )

    def _synthesize_recommendations(
        self,
        input_data: SynthesisInput,
        output_format: OutputFormat,
    ) -> SynthesisOutput:
        """Synthesize response for recommendation queries."""
        recommendations = input_data.recommendations or []
        products = input_data.products[:input_data.max_products]

        response_parts = [self.RECOMMEND_TEMPLATES["intro"]]

        # Add top recommendation
        if recommendations:
            top = recommendations[0]
            response_parts.append(
                self.RECOMMEND_TEMPLATES["top_pick"].format(
                    product=top.get("name", top.get("title", "Top Pick")),
                    reason=top.get("reason", "Best overall match for your needs"),
                )
            )

            # Add alternatives
            if len(recommendations) > 1:
                response_parts.append(self.RECOMMEND_TEMPLATES["alternatives"])
                for rec in recommendations[1:4]:
                    response_parts.append(
                        f"- {rec.get('name', rec.get('title', 'Alternative'))}: "
                        f"{rec.get('reason', 'Good alternative')}"
                    )

        response_text = "\n".join(response_parts)

        suggestions = self._generate_follow_ups(
            QueryIntent.RECOMMEND, products, input_data.query
        )

        return SynthesisOutput(
            response_text=response_text,
            format=OutputFormat.CARDS,
            products=self._format_products(products, OutputFormat.CARDS),
            citations=self._build_citations(products) if input_data.include_citations else [],
            confidence=self._calculate_confidence(products, input_data.query),
            result_count=len(recommendations),
            suggestions=suggestions,
        )

    def _synthesize_price_analysis(
        self,
        input_data: SynthesisInput,
        output_format: OutputFormat,
    ) -> SynthesisOutput:
        """Synthesize response for price analysis queries."""
        price_analysis = input_data.price_analysis or {}
        products = input_data.products[:input_data.max_products]

        response_parts = []

        # Price range
        price_range = price_analysis.get("price_range", {})
        if price_range:
            response_parts.append(
                f"Price range: ${price_range.get('min', 0):.2f} - "
                f"${price_range.get('max', 0):.2f} "
                f"(avg: ${price_range.get('avg', 0):.2f})"
            )

        # Best value
        best_value = price_analysis.get("best_value")
        if best_value:
            response_parts.append(
                f"\nBest value: {best_value.get('name', 'Product')} at "
                f"${best_value.get('price', 0):.2f}"
            )

        # Price verdict
        verdict = price_analysis.get("verdict")
        if verdict:
            response_parts.append(f"\nVerdict: {verdict}")

        # Deals
        deals = price_analysis.get("deals", [])
        if deals:
            response_parts.append("\nCurrent deals:")
            for deal in deals[:3]:
                response_parts.append(
                    f"- {deal.get('name', 'Product')}: "
                    f"{deal.get('discount', 0)}% off"
                )

        response_text = "\n".join(response_parts) if response_parts else "Price analysis complete."

        suggestions = self._generate_follow_ups(
            QueryIntent.PRICE_CHECK, products, input_data.query
        )

        return SynthesisOutput(
            response_text=response_text,
            format=OutputFormat.BULLET,
            products=self._format_products(products, output_format),
            citations=self._build_citations(products) if input_data.include_citations else [],
            confidence=self._calculate_confidence(products, input_data.query),
            result_count=len(products),
            suggestions=suggestions,
        )

    def _synthesize_trends(
        self,
        input_data: SynthesisInput,
        output_format: OutputFormat,
    ) -> SynthesisOutput:
        """Synthesize response for trend queries."""
        trends = input_data.trends or {}
        products = input_data.products[:input_data.max_products]

        response_parts = ["Here's what's trending:"]

        # Trending products
        trending = trends.get("trending", [])
        if trending:
            response_parts.append("\nTop trending products:")
            for i, product in enumerate(trending[:5], 1):
                name = product.get("name", product.get("title", f"Product {i}"))
                change = product.get("trend_score", product.get("change", 0))
                response_parts.append(f"{i}. {name} (+{change}%)")

        # Emerging categories
        categories = trends.get("emerging_categories", [])
        if categories:
            response_parts.append("\nEmerging categories:")
            for cat in categories[:3]:
                response_parts.append(f"- {cat}")

        # Declining
        declining = trends.get("declining", [])
        if declining:
            response_parts.append("\nLosing popularity:")
            for item in declining[:3]:
                response_parts.append(f"- {item.get('name', 'Product')}")

        response_text = "\n".join(response_parts)

        suggestions = self._generate_follow_ups(
            QueryIntent.TREND, products, input_data.query
        )

        return SynthesisOutput(
            response_text=response_text,
            format=OutputFormat.CARDS,
            products=self._format_products(products, OutputFormat.CARDS),
            citations=self._build_citations(products) if input_data.include_citations else [],
            confidence=self._calculate_confidence(products, input_data.query),
            result_count=len(trending) if trending else len(products),
            suggestions=suggestions,
        )

    def _build_product_summary(self, products: list[dict]) -> str:
        """Build a brief summary of products."""
        if not products:
            return ""

        # Extract key stats
        prices = [p.get("price", 0) for p in products if p.get("price")]
        ratings = [p.get("stars", p.get("rating", 0)) for p in products if p.get("stars") or p.get("rating")]
        brands = list(set(p.get("brand", "") for p in products if p.get("brand")))

        parts = []

        if prices:
            min_price = min(prices)
            max_price = max(prices)
            if min_price == max_price:
                parts.append(f"priced at ${min_price:.2f}")
            else:
                parts.append(f"ranging from ${min_price:.2f} to ${max_price:.2f}")

        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            parts.append(f"with an average rating of {avg_rating:.1f} stars")

        if brands and len(brands) <= 3:
            parts.append(f"from {', '.join(brands[:3])}")

        return "Products " + ", ".join(parts) + "." if parts else ""

    def _format_products(
        self,
        products: list[dict],
        output_format: OutputFormat,
    ) -> list[dict]:
        """Format products for the specified output format."""
        formatted = []

        for product in products:
            formatted_product = {
                "id": product.get("asin", product.get("id")),
                "title": product.get("title", "Unknown Product"),
                "brand": product.get("brand", "Unknown"),
                "price": product.get("price"),
                "rating": product.get("stars", product.get("rating")),
                "review_count": product.get("review_count", product.get("reviews")),
            }

            # Add format-specific fields
            if output_format in (OutputFormat.CARDS, OutputFormat.COMPARISON):
                formatted_product.update({
                    "image_url": product.get("image_url", product.get("image")),
                    "summary": product.get("summary", ""),
                    "pros": product.get("pros", []),
                    "cons": product.get("cons", []),
                })

            if output_format == OutputFormat.TABLE:
                formatted_product.update({
                    "category": product.get("category", product.get("category_level1")),
                    "availability": product.get("availability", "In Stock"),
                })

            formatted.append(formatted_product)

        return formatted

    def _build_table_data(self, products: list[dict]) -> dict[str, Any]:
        """Build table data structure for TABLE format."""
        if not products:
            return {"columns": [], "rows": []}

        columns = ["Product", "Brand", "Price", "Rating", "Reviews"]
        rows = []

        for product in products:
            rows.append([
                product.get("title", "Unknown")[:50],
                product.get("brand", "Unknown"),
                f"${product.get('price', 0):.2f}" if product.get("price") else "N/A",
                f"{product.get('stars', product.get('rating', 0)):.1f}" if product.get("stars") or product.get("rating") else "N/A",
                str(product.get("review_count", product.get("reviews", 0))),
            ])

        return {"columns": columns, "rows": rows}

    def _build_comparison_data(
        self,
        products: list[dict],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        """Build comparison data structure."""
        if not products:
            return {}

        # Attributes to compare
        attributes = ["price", "rating", "brand", "pros", "cons"]

        # Build comparison matrix
        matrix = {}
        for attr in attributes:
            matrix[attr] = {}
            for product in products:
                product_id = product.get("asin", product.get("id", "unknown"))
                if attr == "rating":
                    matrix[attr][product_id] = product.get("stars", product.get("rating", 0))
                elif attr == "pros":
                    matrix[attr][product_id] = product.get("pros", [])[:3]
                elif attr == "cons":
                    matrix[attr][product_id] = product.get("cons", [])[:3]
                else:
                    matrix[attr][product_id] = product.get(attr)

        return {
            "products": [p.get("asin", p.get("id")) for p in products],
            "product_names": {
                p.get("asin", p.get("id")): p.get("title", "Unknown")
                for p in products
            },
            "matrix": matrix,
            "winner": comparison.get("winner"),
            "differences": comparison.get("differences", []),
        }

    def _build_citations(self, products: list[dict]) -> list[Citation]:
        """Build citations for products."""
        citations = []

        for product in products:
            citations.append(Citation(
                source_type="product",
                product_id=product.get("asin", product.get("id")),
                product_name=product.get("title", "Unknown"),
            ))

        return citations

    def _calculate_confidence(self, products: list[dict], query: str) -> float:
        """Calculate confidence score for the response.

        Factors:
        - Number of results
        - Quality of matches
        - Rating coverage
        - Price coverage
        """
        if not products:
            return 0.2

        score = 0.5  # Base score

        # More results = higher confidence (up to a point)
        if len(products) >= 5:
            score += 0.2
        elif len(products) >= 3:
            score += 0.15
        elif len(products) >= 1:
            score += 0.1

        # Rating coverage
        rated = sum(1 for p in products if p.get("stars") or p.get("rating"))
        if rated == len(products):
            score += 0.15
        elif rated > len(products) / 2:
            score += 0.1

        # Price coverage
        priced = sum(1 for p in products if p.get("price"))
        if priced == len(products):
            score += 0.15
        elif priced > len(products) / 2:
            score += 0.1

        return min(score, 1.0)

    def _generate_follow_ups(
        self,
        intent: QueryIntent,
        products: list[dict],
        query: str,
    ) -> list[str]:
        """Generate follow-up suggestions based on context."""
        templates = self.FOLLOW_UP_TEMPLATES.get(intent, [])
        if not templates:
            return ["Show me more options", "Tell me more about these"]

        suggestions = []

        # Get context for template filling
        context = {
            "count": len(products),
            "query": query,
        }

        if products:
            context["product"] = products[0].get("title", "this product")[:30]
            context["brand"] = products[0].get("brand", "this brand")
            prices = [p.get("price", 0) for p in products if p.get("price")]
            if prices:
                context["price"] = int(sum(prices) / len(prices))

            if len(products) > 1:
                context["winner"] = products[0].get("title", "the top pick")[:30]
                context["alternative"] = products[1].get("title", "alternatives")[:30]

        # Fill templates
        for template in templates[:4]:
            try:
                suggestion = template.format(**context)
                suggestions.append(suggestion)
            except KeyError:
                # Skip templates that need unavailable context
                continue

        return suggestions[:4]


# Singleton instance
_synthesis_agent: SynthesisAgent | None = None


async def get_synthesis_agent(use_llm: bool = True) -> SynthesisAgent:
    """Get or create synthesis agent singleton."""
    global _synthesis_agent
    if _synthesis_agent is None:
        _synthesis_agent = SynthesisAgent(use_llm=use_llm)
        await _synthesis_agent.initialize()
    return _synthesis_agent


async def synthesize_response(
    query: str,
    intent: QueryIntent,
    products: list[dict],
    **kwargs,
) -> SynthesisOutput:
    """Synthesize response from agent outputs.

    Convenience function for quick synthesis.

    Args:
        query: Original user query
        intent: Primary intent
        products: Products from retrieval
        **kwargs: Additional agent outputs (analysis, comparison, etc.)

    Returns:
        SynthesisOutput with formatted response
    """
    agent = await get_synthesis_agent()
    input_data = SynthesisInput(
        query=query,
        intent=intent,
        products=products,
        **kwargs,
    )
    return await agent.execute(input_data)
