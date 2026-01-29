"""General Chat Agent.

This agent handles non-product related conversations including:
- Greetings and farewells
- Help requests and capability explanations
- Small talk
- Off-topic queries with polite redirects

Enhanced with LLM support for more natural, contextual responses.
"""

import random
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, RetryConfig
from src.models.intent import QueryIntent, IntentAnalysis

logger = structlog.get_logger()


class GeneralResponse(BaseModel):
    """Response from GeneralAgent."""

    intent: QueryIntent = Field(description="The detected intent")
    response_text: str = Field(description="The response message")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up queries",
    )
    show_capabilities: bool = Field(
        default=False,
        description="Whether to show capability list",
    )
    redirect_to_products: bool = Field(
        default=False,
        description="Whether response redirects to product queries",
    )


class GeneralAgent(BaseAgent[IntentAnalysis, GeneralResponse]):
    """Agent for handling general (non-product) conversations.

    Provides friendly, helpful responses for:
    - Greetings: Welcome users and introduce capabilities
    - Farewells: Thank users and invite them back
    - Help: Explain what the system can do
    - Small talk: Brief friendly responses with redirect
    - Off-topic: Polite explanation of scope with redirect

    All responses gently guide users toward product-related queries.
    """

    name = "general"
    description = "Handles general conversation and non-product queries"

    # Response templates with variations for natural conversation
    GREETING_RESPONSES = [
        "Hello! I'm your Product Intelligence Assistant. I can help you find, compare, and analyze products. What are you looking for today?",
        "Hi there! Ready to help you discover great products. Would you like to search for something specific, compare options, or get recommendations?",
        "Hey! Welcome to Product Intelligence. I can help you find the perfect product. What would you like to explore?",
        "Greetings! I'm here to help with all your product research needs. What can I help you find today?",
    ]

    FAREWELL_RESPONSES = [
        "Goodbye! Thanks for using Product Intelligence. Come back anytime you need help with product research!",
        "See you later! Feel free to return whenever you need to find or compare products.",
        "Thanks for chatting! I hope I was helpful. Come back anytime for more product insights!",
        "Take care! I'm always here when you need help finding the right products.",
    ]

    HELP_RESPONSE = """I'm your Product Intelligence Assistant! Here's what I can help you with:

**Product Discovery**
- "Find wireless headphones under $100"
- "Show me best-rated coffee makers"
- "Laptops with 16GB RAM for programming"

**Product Comparison**
- "Compare Sony WH-1000XM5 vs Bose QC45"
- "Which is better: MacBook Pro or Dell XPS?"
- "Difference between OLED and QLED TVs"

**Review Analysis**
- "What do people say about battery life?"
- "Pros and cons of iPhone 15"
- "Common complaints about this product"

**Price Intelligence**
- "Is $299 a good price for AirPods Pro?"
- "Find best deals on gaming monitors"
- "Budget laptops under $500"

**Trends & Recommendations**
- "Trending products in smart home"
- "Recommend headphones for travel"
- "Similar products to AirPods"

Just ask me anything about products, and I'll do my best to help!"""

    HELP_SUGGESTIONS = [
        "Find wireless headphones under $100",
        "Compare Sony vs Bose headphones",
        "What are the best-rated laptops?",
        "Recommend a camera for beginners",
        "Trending products in smart home",
    ]

    SMALL_TALK_RESPONSES = [
        "I'm doing great, thanks for asking! I'm always ready to help you find amazing products. What would you like to explore today?",
        "I'm here and ready to help! While I'm not great at small talk, I excel at finding products. What can I help you discover?",
        "All good here! I love helping people find the perfect products. Is there something specific you're looking for?",
    ]

    OFF_TOPIC_RESPONSES = {
        "weather": "I wish I could help with the weather, but I'm specialized in product intelligence. However, I can help you find weather-related gear! Would you like to see umbrellas, rain jackets, or weather stations?",
        "joke": "I'm not the best comedian, but I can find you some funny products! How about joke books, gag gifts, or party supplies?",
        "time": "I can't tell you the time, but I can help you find a great watch or clock! Would you like to see some options?",
        "news": "I don't have access to news, but I can help you find news-related products like e-readers, tablets, or smart displays. Interested?",
        "math": "Math isn't my specialty, but I can help you find calculators, math learning tools, or educational products!",
        "default": "That's outside my area of expertise. I'm specialized in helping you find, compare, and analyze products. Is there a product I can help you research today?",
    }

    OFF_TOPIC_SUGGESTIONS = {
        "weather": [
            "Find umbrellas under $30",
            "Best-rated rain jackets",
            "Weather stations for home",
        ],
        "joke": [
            "Funny gag gifts",
            "Best joke books",
            "Party supplies and games",
        ],
        "time": [
            "Best-rated smartwatches",
            "Wall clocks for home",
            "Fitness trackers with time display",
        ],
        "news": [
            "Best e-readers for reading",
            "Tablets for browsing",
            "Smart displays for home",
        ],
        "math": [
            "Scientific calculators",
            "Math learning tools for kids",
            "Educational STEM toys",
        ],
        "default": [
            "Find popular products",
            "Best-rated items this month",
            "Trending products in electronics",
        ],
    }

    CAPABILITY_CATEGORIES = [
        ("Product Discovery", "Find products by features, price, brand, or category"),
        ("Comparison", "Compare 2+ products side by side"),
        ("Review Analysis", "Understand what customers say"),
        ("Price Intelligence", "Evaluate prices and find deals"),
        ("Trends", "See what's popular and trending"),
        ("Recommendations", "Get personalized suggestions"),
    ]

    # LLM prompt for contextual responses
    LLM_RESPONSE_PROMPT = """You are a friendly, helpful Product Intelligence Assistant. Your role is to help users find, compare, and analyze products.

User query: "{query}"
Detected intent: {intent}
Context: {context}

Generate a natural, conversational response that:
1. Acknowledges the user appropriately for their intent ({intent})
2. Maintains a friendly, helpful personality
3. Gently guides them toward product-related queries if appropriate
4. Keeps the response concise (2-3 sentences)
5. Suggests relevant product queries they might want to try

Do not use markdown formatting. Keep it conversational.

Response:"""

    def __init__(self, use_llm: bool = True):
        """Initialize the General Agent.

        Args:
            use_llm: Whether to use LLM for contextual responses
        """
        super().__init__(retry_config=RetryConfig(max_retries=1))
        self.use_llm = use_llm
        self._llm_manager = None

    async def initialize(self) -> None:
        """Initialize LLM manager if enabled."""
        if self.use_llm:
            try:
                from src.llm import get_llm_manager
                self._llm_manager = await get_llm_manager()
                logger.info("general_agent_llm_initialized")
            except Exception as e:
                logger.warning(
                    "general_agent_llm_init_failed",
                    error=str(e),
                    fallback="canned responses",
                )
                self._llm_manager = None
        await super().initialize()

    async def _execute_internal(self, analysis: IntentAnalysis) -> GeneralResponse:
        """Generate response for general chat intent.

        Uses template-based responses with optional LLM enhancement
        for more natural, contextual replies.

        Args:
            analysis: IntentAnalysis from IntentAgent

        Returns:
            GeneralResponse with appropriate message and suggestions
        """
        intent = analysis.primary_intent
        query_lower = analysis.query.lower()

        # Get base template response
        if intent == QueryIntent.GREETING:
            base_response = self._handle_greeting()

        elif intent == QueryIntent.FAREWELL:
            base_response = self._handle_farewell()

        elif intent == QueryIntent.HELP:
            base_response = self._handle_help()

        elif intent == QueryIntent.SMALL_TALK:
            base_response = self._handle_small_talk()

        elif intent == QueryIntent.OFF_TOPIC:
            base_response = self._handle_off_topic(query_lower)

        elif intent == QueryIntent.CLARIFICATION:
            base_response = self._handle_clarification()

        else:
            # Fallback for any unhandled general intent
            base_response = self._handle_fallback()

        # Try LLM enhancement for more natural responses
        return await self._try_llm_enhancement(analysis, base_response)

    def _handle_greeting(self) -> GeneralResponse:
        """Handle greeting intent."""
        response = random.choice(self.GREETING_RESPONSES)

        logger.debug("general_agent_greeting")

        return GeneralResponse(
            intent=QueryIntent.GREETING,
            response_text=response,
            suggestions=[
                "Find wireless headphones",
                "Compare popular laptops",
                "What's trending in electronics?",
            ],
            show_capabilities=False,
            redirect_to_products=True,
        )

    def _handle_farewell(self) -> GeneralResponse:
        """Handle farewell intent."""
        response = random.choice(self.FAREWELL_RESPONSES)

        logger.debug("general_agent_farewell")

        return GeneralResponse(
            intent=QueryIntent.FAREWELL,
            response_text=response,
            suggestions=[],
            show_capabilities=False,
            redirect_to_products=False,
        )

    def _handle_help(self) -> GeneralResponse:
        """Handle help request."""
        logger.debug("general_agent_help")

        return GeneralResponse(
            intent=QueryIntent.HELP,
            response_text=self.HELP_RESPONSE,
            suggestions=self.HELP_SUGGESTIONS,
            show_capabilities=True,
            redirect_to_products=True,
        )

    def _handle_small_talk(self) -> GeneralResponse:
        """Handle small talk."""
        response = random.choice(self.SMALL_TALK_RESPONSES)

        logger.debug("general_agent_small_talk")

        return GeneralResponse(
            intent=QueryIntent.SMALL_TALK,
            response_text=response,
            suggestions=[
                "Show me popular products",
                "What's trending today?",
                "Help me find a gift",
            ],
            show_capabilities=False,
            redirect_to_products=True,
        )

    def _handle_off_topic(self, query_lower: str) -> GeneralResponse:
        """Handle off-topic queries with polite redirect."""
        # Detect topic for contextual response
        topic = "default"
        topic_keywords = {
            "weather": ["weather", "temperature", "rain", "sunny", "forecast"],
            "joke": ["joke", "funny", "laugh", "humor"],
            "time": ["time", "clock", "hour", "minute"],
            "news": ["news", "headline", "politics", "sports score"],
            "math": ["calculate", "math", "equation", "+", "-", "*", "/", "sum"],
        }

        for topic_name, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                topic = topic_name
                break

        response = self.OFF_TOPIC_RESPONSES.get(topic, self.OFF_TOPIC_RESPONSES["default"])
        suggestions = self.OFF_TOPIC_SUGGESTIONS.get(topic, self.OFF_TOPIC_SUGGESTIONS["default"])

        logger.debug("general_agent_off_topic", topic=topic)

        return GeneralResponse(
            intent=QueryIntent.OFF_TOPIC,
            response_text=response,
            suggestions=suggestions,
            show_capabilities=False,
            redirect_to_products=True,
        )

    def _handle_clarification(self) -> GeneralResponse:
        """Handle clarification requests."""
        response = """I'd be happy to clarify! I'm a Product Intelligence Assistant that helps you:

- **Find products** matching your needs
- **Compare** different options
- **Analyze reviews** to understand quality
- **Check prices** and find deals
- **Get recommendations** based on your preferences

Could you tell me more about what you're looking for? For example:
- What type of product?
- Any specific features or budget?
- Brand preferences?"""

        logger.debug("general_agent_clarification")

        return GeneralResponse(
            intent=QueryIntent.CLARIFICATION,
            response_text=response,
            suggestions=[
                "I'm looking for headphones",
                "Help me find a laptop",
                "What are good gift ideas?",
            ],
            show_capabilities=True,
            redirect_to_products=True,
        )

    def _handle_fallback(self) -> GeneralResponse:
        """Handle fallback for unknown intents."""
        response = "I'm not sure I understood that. I'm here to help you with product research - finding, comparing, and analyzing products. What would you like to explore?"

        logger.debug("general_agent_fallback")

        return GeneralResponse(
            intent=QueryIntent.AMBIGUOUS,
            response_text=response,
            suggestions=[
                "Find popular products",
                "Compare top-rated items",
                "What can you help with?",
            ],
            show_capabilities=False,
            redirect_to_products=True,
        )

    async def _llm_respond(
        self, query: str, intent: QueryIntent, base_response: GeneralResponse
    ) -> GeneralResponse | None:
        """Generate contextual response using LLM.

        Args:
            query: Original user query
            intent: Detected intent
            base_response: Fallback response to use if LLM fails

        Returns:
            Enhanced GeneralResponse or None if LLM unavailable
        """
        if not self._llm_manager or not self._llm_manager.is_initialized:
            return None

        # Build context based on intent
        context = self._build_llm_context(intent)

        prompt = self.LLM_RESPONSE_PROMPT.format(
            query=query,
            intent=intent.value,
            context=context,
        )

        try:
            from src.llm import GenerationConfig

            result = await self._llm_manager.generate_for_agent(
                agent_name="general",
                prompt=prompt,
                config=GenerationConfig(
                    temperature=0.8,  # More creative for conversational responses
                    max_tokens=200,
                ),
            )

            response_text = result.content.strip()

            # Clean up response
            response_text = response_text.strip('"\'')

            # Return enhanced response
            return GeneralResponse(
                intent=intent,
                response_text=response_text,
                suggestions=base_response.suggestions,
                show_capabilities=base_response.show_capabilities,
                redirect_to_products=base_response.redirect_to_products,
            )

        except Exception as e:
            logger.debug("llm_general_response_failed", error=str(e))
            return None

    def _build_llm_context(self, intent: QueryIntent) -> str:
        """Build context string for LLM prompt.

        Args:
            intent: Detected intent

        Returns:
            Context string describing the situation
        """
        context_map = {
            QueryIntent.GREETING: "User is greeting you. Welcome them and introduce your product assistance capabilities.",
            QueryIntent.FAREWELL: "User is saying goodbye. Thank them and invite them back.",
            QueryIntent.HELP: "User is asking what you can do. Explain your product intelligence capabilities briefly.",
            QueryIntent.SMALL_TALK: "User is making small talk. Be friendly but gently redirect to product queries.",
            QueryIntent.OFF_TOPIC: "User is asking about something outside your scope. Politely explain you specialize in products.",
            QueryIntent.CLARIFICATION: "User needs clarification. Ask what kind of product they're interested in.",
        }
        return context_map.get(
            intent,
            "User's intent is unclear. Offer to help with product research.",
        )

    async def _try_llm_enhancement(
        self, analysis: IntentAnalysis, base_response: GeneralResponse
    ) -> GeneralResponse:
        """Try to enhance response with LLM, falling back to base response.

        Args:
            analysis: Intent analysis with query
            base_response: Template-based response

        Returns:
            LLM-enhanced or base response
        """
        if not self.use_llm or not self._llm_manager:
            return base_response

        # Only use LLM for intents that benefit from context
        llm_intents = {
            QueryIntent.GREETING,
            QueryIntent.SMALL_TALK,
            QueryIntent.OFF_TOPIC,
        }

        if analysis.primary_intent not in llm_intents:
            return base_response

        llm_response = await self._llm_respond(
            analysis.query, analysis.primary_intent, base_response
        )

        return llm_response or base_response

    def get_capabilities_summary(self) -> str:
        """Get a short summary of capabilities."""
        lines = ["I can help you with:"]
        for name, desc in self.CAPABILITY_CATEGORIES:
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    def get_example_queries(self, category: str | None = None) -> list[str]:
        """Get example queries, optionally filtered by category."""
        examples = {
            "discovery": [
                "Find wireless headphones under $100",
                "Show me laptops with 16GB RAM",
                "Best-rated coffee makers",
            ],
            "comparison": [
                "Compare Sony vs Bose headphones",
                "MacBook Pro vs Dell XPS",
                "Which camera is better for beginners?",
            ],
            "analysis": [
                "What do reviews say about battery life?",
                "Pros and cons of iPhone 15",
                "Common complaints about this laptop",
            ],
            "price": [
                "Is $299 good for AirPods Pro?",
                "Best deals on monitors",
                "Budget gaming laptops",
            ],
            "trends": [
                "Trending in smart home",
                "Popular headphones this month",
                "Best sellers in electronics",
            ],
            "recommendations": [
                "Recommend headphones for travel",
                "Similar to AirPods but cheaper",
                "Accessories for Canon camera",
            ],
        }

        if category and category.lower() in examples:
            return examples[category.lower()]

        # Return mix of all categories
        all_examples = []
        for cat_examples in examples.values():
            all_examples.extend(cat_examples[:1])
        return all_examples


# Singleton instance
_general_agent: GeneralAgent | None = None


async def get_general_agent(use_llm: bool = True) -> GeneralAgent:
    """Get or create general agent singleton."""
    global _general_agent
    if _general_agent is None:
        _general_agent = GeneralAgent(use_llm=use_llm)
        await _general_agent.initialize()
    return _general_agent


async def handle_general_chat(analysis: IntentAnalysis) -> GeneralResponse:
    """Handle general chat query.

    Convenience function for quick general chat handling.

    Args:
        analysis: IntentAnalysis from IntentAgent

    Returns:
        GeneralResponse with appropriate message
    """
    agent = await get_general_agent()
    return await agent.execute(analysis)
