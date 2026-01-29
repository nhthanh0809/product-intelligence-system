"""LangGraph-powered conversation engine for natural multi-turn chat.

Features:
- Full conversation state management
- Automatic message history with persistence
- LLM-powered context understanding
- User preference learning
- Natural query rewriting
- Clarification handling
"""

import json
import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

logger = structlog.get_logger()


# === State Definitions ===

class ProductInfo(TypedDict, total=False):
    """Product information in conversation."""
    asin: str
    title: str
    brand: str
    price: float
    stars: float
    position: int


class UserPreferences(TypedDict, total=False):
    """Learned user preferences."""
    budget_min: float
    budget_max: float
    preferred_brands: list[str]
    disliked_brands: list[str]
    important_features: list[str]
    use_cases: list[str]


class ConversationState(TypedDict, total=False):
    """Full conversation state for LangGraph."""

    # Core conversation
    messages: Annotated[list[BaseMessage], operator.add]  # Auto-appends new messages
    session_id: str

    # Current turn info
    current_input: str
    current_intent: str
    rewritten_query: str

    # Context
    products_in_context: list[ProductInfo]
    last_search_results: list[ProductInfo]

    # User modeling
    user_preferences: UserPreferences
    facts_learned: list[str]

    # Flow control
    needs_clarification: bool
    clarification_question: str
    action_to_take: str  # search, compare, recommend, price, general, clarify

    # Results
    agent_results: dict[str, Any]
    final_response: str
    suggestions: list[str]

    # Metadata
    turn_count: int
    created_at: str
    updated_at: str


# === System Prompts ===

SYSTEM_PROMPT = """You are a helpful product assistant for an e-commerce platform specializing in product discovery, comparison, and recommendations.

Your capabilities:
- Search for products based on user requirements
- Compare products side-by-side
- Provide personalized recommendations
- Answer questions about products
- Track price information

Conversation Guidelines:
1. Be conversational and natural, like chatting with a knowledgeable friend
2. Remember everything from the conversation - user preferences, products discussed, etc.
3. When the user references something vague (it, them, the first one, the cheaper one), understand from context
4. If something is truly ambiguous, ask a brief clarifying question
5. Learn user preferences (budget, favorite brands, important features) and apply them
6. Proactively suggest relevant follow-ups
7. Keep responses concise but helpful

Current conversation context will be provided to help you understand references."""


CONTEXT_ANALYZER_PROMPT = """Analyze the user's message in the context of our conversation.

## Conversation History
{history}

## Products Currently Discussed
{products}

## User Preferences Known
{preferences}

## User's New Message
"{message}"

## Your Task
Analyze this message and respond with JSON:
```json
{{
    "understood_intent": "brief description of what user wants",
    "references_found": [
        {{"original": "the reference text", "refers_to": "what it refers to"}}
    ],
    "rewritten_query": "the message with all references replaced with actual names/values",
    "needs_clarification": false,
    "clarification_question": "only if needs_clarification is true",
    "action": "search|compare|recommend|price_check|product_details|general_chat|clarify",
    "new_preferences": {{
        "budget_max": null,
        "preferred_brand": null,
        "important_feature": null,
        "use_case": null
    }},
    "confidence": 0.95
}}
```

Important:
- "rewritten_query" should be self-contained - someone without context should understand it
- Only set needs_clarification=true if genuinely ambiguous
- Detect budget mentions like "under $300" → budget_max: 300
- Detect brand preferences like "I like Sony" → preferred_brand: "Sony"
- Detect features like "good battery life is important" → important_feature: "battery life"
"""


RESPONSE_GENERATOR_PROMPT = """Generate a natural, conversational response based on the results.

## Context
User asked: "{query}"
Intent understood: {intent}
Action taken: {action}

## Results
{results}

## User Preferences
{preferences}

## Guidelines
1. Be conversational, not robotic - talk like a helpful friend
2. If showing products, highlight what matters to THIS user based on their preferences
3. If comparing, focus on the differences that matter for their use case
4. Suggest logical next steps (but keep suggestions brief)
5. If results are limited or empty, acknowledge and offer alternatives
6. Use markdown formatting for better readability (bullet points, bold for emphasis)

Generate your response:"""


class LangGraphConversationEngine:
    """LangGraph-powered conversation engine.

    Uses a state graph to manage conversation flow:

    1. analyze_input: LLM analyzes user message in context
    2. route: Decide next action based on analysis
    3. execute_action: Run appropriate agent/handler
    4. generate_response: Create natural language response
    5. update_state: Update preferences and context
    """

    def __init__(self, llm_manager=None, checkpointer=None):
        """Initialize the conversation engine.

        Args:
            llm_manager: LLM manager instance (will be fetched if None)
            checkpointer: LangGraph checkpointer for persistence (uses MemorySaver if None)
        """
        self._llm_manager = llm_manager
        self._checkpointer = checkpointer or MemorySaver()
        self._graph = None
        self._initialized = False

        # Agent references (will be set during initialization)
        self._search_agent = None
        self._compare_agent = None
        self._recommend_agent = None
        self._price_agent = None
        self._synthesis_agent = None

    async def initialize(self):
        """Initialize the engine and build the graph."""
        if self._initialized:
            return

        # Get LLM manager
        if self._llm_manager is None:
            try:
                from src.llm import get_llm_manager
                self._llm_manager = await get_llm_manager()
            except Exception as e:
                logger.warning("llm_manager_init_failed", error=str(e))

        # Get agents
        try:
            from src.agents.search_agent import get_search_agent
            from src.agents.compare_agent import get_compare_agent
            from src.agents.recommend_agent import get_recommend_agent
            from src.agents.price_agent import get_price_agent
            from src.agents.synthesis_agent import get_synthesis_agent

            self._search_agent = await get_search_agent()
            self._compare_agent = await get_compare_agent()
            self._recommend_agent = await get_recommend_agent()
            self._price_agent = await get_price_agent()
            self._synthesis_agent = await get_synthesis_agent()
        except Exception as e:
            logger.warning("agents_init_partial", error=str(e))

        # Build the graph
        self._graph = self._build_graph()
        self._initialized = True

        logger.info("langgraph_conversation_engine_initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation graph."""

        # Create the graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("analyze_input", self._analyze_input)
        workflow.add_node("handle_clarification", self._handle_clarification)
        workflow.add_node("execute_search", self._execute_search)
        workflow.add_node("execute_compare", self._execute_compare)
        workflow.add_node("execute_recommend", self._execute_recommend)
        workflow.add_node("execute_price", self._execute_price)
        workflow.add_node("execute_general", self._execute_general)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_context", self._update_context)

        # Set entry point
        workflow.set_entry_point("analyze_input")

        # Add conditional routing after analysis
        workflow.add_conditional_edges(
            "analyze_input",
            self._route_action,
            {
                "clarify": "handle_clarification",
                "search": "execute_search",
                "compare": "execute_compare",
                "recommend": "execute_recommend",
                "price_check": "execute_price",
                "product_details": "execute_search",
                "general_chat": "execute_general",
            }
        )

        # All execution nodes go to response generation
        workflow.add_edge("handle_clarification", "generate_response")
        workflow.add_edge("execute_search", "generate_response")
        workflow.add_edge("execute_compare", "generate_response")
        workflow.add_edge("execute_recommend", "generate_response")
        workflow.add_edge("execute_price", "generate_response")
        workflow.add_edge("execute_general", "generate_response")

        # Response generation goes to context update
        workflow.add_edge("generate_response", "update_context")

        # Context update is the end
        workflow.add_edge("update_context", END)

        # Compile with checkpointer
        return workflow.compile(checkpointer=self._checkpointer)

    async def _analyze_input(self, state: ConversationState) -> dict:
        """Analyze user input with LLM to understand intent and resolve references."""

        current_input = state.get("current_input", "")
        messages = state.get("messages", [])
        products = state.get("products_in_context", [])
        preferences = state.get("user_preferences", {})

        # Format history for LLM
        history_text = self._format_history(messages)
        products_text = self._format_products(products)
        prefs_text = self._format_preferences(preferences)

        # Build analysis prompt
        prompt = CONTEXT_ANALYZER_PROMPT.format(
            history=history_text,
            products=products_text,
            preferences=prefs_text,
            message=current_input,
        )

        try:
            if self._llm_manager:
                from src.llm import GenerationConfig

                # Include system instruction in the prompt (no system_prompt param)
                full_prompt = f"""You are a conversation analyzer. Output only valid JSON.

{prompt}"""

                response = await self._llm_manager.generate_for_agent(
                    agent_name="conversation_analyzer",
                    prompt=full_prompt,
                    config=GenerationConfig(temperature=0.2, max_tokens=500),
                )

                # Parse JSON from response
                analysis = self._parse_json_response(response.content)
            else:
                # Fallback analysis
                analysis = {
                    "understood_intent": "search",
                    "rewritten_query": current_input,
                    "needs_clarification": False,
                    "action": "search",
                    "new_preferences": {},
                    "confidence": 0.5,
                }
        except Exception as e:
            logger.error("analyze_input_failed", error=str(e))
            analysis = {
                "understood_intent": "unknown",
                "rewritten_query": current_input,
                "needs_clarification": False,
                "action": "general_chat",
                "confidence": 0.3,
            }

        logger.info(
            "input_analyzed",
            intent=analysis.get("understood_intent"),
            action=analysis.get("action"),
            rewritten=analysis.get("rewritten_query", "")[:50],
            needs_clarification=analysis.get("needs_clarification"),
        )

        return {
            "current_intent": analysis.get("understood_intent", ""),
            "rewritten_query": analysis.get("rewritten_query", current_input),
            "needs_clarification": analysis.get("needs_clarification", False),
            "clarification_question": analysis.get("clarification_question", ""),
            "action_to_take": analysis.get("action", "general_chat"),
            # Store new preferences to merge later
            "_new_preferences": analysis.get("new_preferences", {}),
        }

    def _route_action(self, state: ConversationState) -> str:
        """Route to appropriate action based on analysis."""

        if state.get("needs_clarification"):
            return "clarify"

        action = state.get("action_to_take", "general_chat")

        # Map actions to node names
        action_map = {
            "search": "search",
            "compare": "compare",
            "recommend": "recommend",
            "price_check": "price_check",
            "product_details": "product_details",
            "general_chat": "general_chat",
            "clarify": "clarify",
        }

        return action_map.get(action, "general_chat")

    async def _handle_clarification(self, state: ConversationState) -> dict:
        """Handle clarification request."""
        return {
            "agent_results": {
                "type": "clarification",
                "question": state.get("clarification_question", "Could you please clarify?"),
            },
        }

    async def _execute_search(self, state: ConversationState) -> dict:
        """Execute search action."""
        query = state.get("rewritten_query", state.get("current_input", ""))
        preferences = state.get("user_preferences", {})

        try:
            if self._search_agent:
                # Use the search method which accepts query string
                result = await self._search_agent.search(query)

                # Get products from result
                products = []
                if hasattr(result, 'formatted_results'):
                    products = result.formatted_results
                elif hasattr(result, 'products'):
                    products = result.products

                return {
                    "agent_results": {
                        "type": "search",
                        "products": products[:10],
                        "total_found": len(products),
                    },
                    "last_search_results": [
                        {
                            "asin": p.get("asin", ""),
                            "title": p.get("title", ""),
                            "brand": p.get("brand", ""),
                            "price": p.get("price"),
                            "stars": p.get("stars"),
                            "position": i + 1,
                        }
                        for i, p in enumerate(products[:10])
                    ],
                }
        except Exception as e:
            logger.error("search_execution_failed", error=str(e))

        return {"agent_results": {"type": "search", "products": [], "error": "Search failed"}}

    async def _execute_compare(self, state: ConversationState) -> dict:
        """Execute compare action."""
        query = state.get("rewritten_query", state.get("current_input", ""))
        products = state.get("products_in_context", [])

        try:
            # If we have products in context, use them for comparison
            if products:
                return {
                    "agent_results": {
                        "type": "comparison",
                        "products_compared": products[:5],
                        "comparison_query": query,
                    },
                }

            # Otherwise search first
            if self._search_agent:
                result = await self._search_agent.search(query)

                found_products = []
                if hasattr(result, 'formatted_results'):
                    found_products = result.formatted_results
                elif hasattr(result, 'products'):
                    found_products = result.products

                return {
                    "agent_results": {
                        "type": "comparison",
                        "products_compared": found_products[:5],
                    },
                    "last_search_results": [
                        {
                            "asin": p.get("asin", ""),
                            "title": p.get("title", ""),
                            "brand": p.get("brand", ""),
                            "price": p.get("price"),
                            "stars": p.get("stars"),
                            "position": i + 1,
                        }
                        for i, p in enumerate(found_products[:10])
                    ],
                }
        except Exception as e:
            logger.error("compare_execution_failed", error=str(e))

        return {"agent_results": {"type": "comparison", "comparison": {}, "error": "Comparison failed"}}

    async def _execute_recommend(self, state: ConversationState) -> dict:
        """Execute recommend action."""
        query = state.get("rewritten_query", state.get("current_input", ""))

        try:
            # Use search agent for recommendations (search + rank)
            if self._search_agent:
                result = await self._search_agent.search(query)

                products = []
                if hasattr(result, 'formatted_results'):
                    products = result.formatted_results
                elif hasattr(result, 'products'):
                    products = result.products

                return {
                    "agent_results": {
                        "type": "recommendation",
                        "recommendations": products[:5],
                        "total_found": len(products),
                    },
                    "last_search_results": [
                        {
                            "asin": p.get("asin", ""),
                            "title": p.get("title", ""),
                            "brand": p.get("brand", ""),
                            "price": p.get("price"),
                            "stars": p.get("stars"),
                            "position": i + 1,
                        }
                        for i, p in enumerate(products[:10])
                    ],
                }
        except Exception as e:
            logger.error("recommend_execution_failed", error=str(e))

        return {"agent_results": {"type": "recommendation", "recommendations": []}}

    async def _execute_price(self, state: ConversationState) -> dict:
        """Execute price check action."""
        query = state.get("rewritten_query", state.get("current_input", ""))
        products = state.get("products_in_context", [])

        try:
            # If we have products, return their price info
            if products:
                price_info = []
                for p in products[:5]:
                    price_info.append({
                        "title": p.get("title", ""),
                        "price": p.get("price"),
                        "brand": p.get("brand", ""),
                    })

                return {
                    "agent_results": {
                        "type": "price",
                        "price_info": price_info,
                    },
                }

            # Otherwise search for products
            if self._search_agent:
                result = await self._search_agent.search(query)

                found_products = []
                if hasattr(result, 'formatted_results'):
                    found_products = result.formatted_results
                elif hasattr(result, 'products'):
                    found_products = result.products

                return {
                    "agent_results": {
                        "type": "price",
                        "price_info": [
                            {"title": p.get("title", ""), "price": p.get("price")}
                            for p in found_products[:5]
                        ],
                    },
                    "last_search_results": [
                        {
                            "asin": p.get("asin", ""),
                            "title": p.get("title", ""),
                            "brand": p.get("brand", ""),
                            "price": p.get("price"),
                            "stars": p.get("stars"),
                            "position": i + 1,
                        }
                        for i, p in enumerate(found_products[:10])
                    ],
                }
        except Exception as e:
            logger.error("price_execution_failed", error=str(e))

        return {"agent_results": {"type": "price", "price_info": {}}}

    async def _execute_general(self, state: ConversationState) -> dict:
        """Execute general chat action."""
        return {
            "agent_results": {
                "type": "general_chat",
                "message": state.get("current_input", ""),
            },
        }

    async def _generate_response(self, state: ConversationState) -> dict:
        """Generate natural language response."""

        agent_results = state.get("agent_results", {})
        result_type = agent_results.get("type", "general")

        # Handle clarification specially
        if result_type == "clarification":
            return {
                "final_response": agent_results.get("question", "Could you please clarify?"),
                "suggestions": ["Try being more specific", "Mention a product name"],
            }

        # Generate response with LLM
        try:
            if self._llm_manager:
                from src.llm import GenerationConfig

                prompt_content = RESPONSE_GENERATOR_PROMPT.format(
                    query=state.get("current_input", ""),
                    intent=state.get("current_intent", ""),
                    action=state.get("action_to_take", ""),
                    results=json.dumps(agent_results, indent=2, default=str),
                    preferences=self._format_preferences(state.get("user_preferences", {})),
                )

                # Include system prompt in the main prompt
                full_prompt = f"""{SYSTEM_PROMPT}

{prompt_content}"""

                response = await self._llm_manager.generate_for_agent(
                    agent_name="response_generator",
                    prompt=full_prompt,
                    config=GenerationConfig(temperature=0.7, max_tokens=1000),
                )

                return {
                    "final_response": response.content,
                    "suggestions": self._generate_suggestions(state, agent_results),
                }
        except Exception as e:
            logger.error("response_generation_failed", error=str(e))

        # Fallback response
        return {
            "final_response": self._generate_fallback_response(agent_results),
            "suggestions": [],
        }

    async def _update_context(self, state: ConversationState) -> dict:
        """Update conversation context with results."""

        updates = {
            "turn_count": state.get("turn_count", 0) + 1,
            "updated_at": datetime.now().isoformat(),
        }

        # Update products in context if we got new results
        agent_results = state.get("agent_results", {})
        if agent_results.get("type") == "search" and agent_results.get("products"):
            updates["products_in_context"] = state.get("last_search_results", [])

        # Merge new preferences
        new_prefs = state.get("_new_preferences", {})
        if new_prefs:
            current_prefs = state.get("user_preferences", {})

            if new_prefs.get("budget_max"):
                current_prefs["budget_max"] = new_prefs["budget_max"]

            if new_prefs.get("preferred_brand"):
                brands = current_prefs.get("preferred_brands", [])
                if new_prefs["preferred_brand"] not in brands:
                    brands.append(new_prefs["preferred_brand"])
                current_prefs["preferred_brands"] = brands

            if new_prefs.get("important_feature"):
                features = current_prefs.get("important_features", [])
                if new_prefs["important_feature"] not in features:
                    features.append(new_prefs["important_feature"])
                current_prefs["important_features"] = features

            if new_prefs.get("use_case"):
                use_cases = current_prefs.get("use_cases", [])
                if new_prefs["use_case"] not in use_cases:
                    use_cases.append(new_prefs["use_case"])
                current_prefs["use_cases"] = use_cases

            updates["user_preferences"] = current_prefs

        return updates

    # === Public API ===

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message.

        Args:
            message: User's message
            session_id: Session ID for conversation continuity

        Returns:
            Dict with response, products, suggestions, etc.
        """
        if not self._initialized:
            await self.initialize()

        # Generate session ID if needed
        if not session_id:
            session_id = str(uuid4())

        # Build initial state
        input_state = {
            "current_input": message,
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
        }

        # Config with thread ID for persistence
        config = {"configurable": {"thread_id": session_id}}

        try:
            # Run the graph
            result = await self._graph.ainvoke(input_state, config=config)

            # Add AI message to state
            final_response = result.get("final_response", "I'm not sure how to help with that.")

            return {
                "response": final_response,
                "session_id": session_id,
                "intent": result.get("current_intent", ""),
                "action": result.get("action_to_take", ""),
                "products": result.get("last_search_results", []),
                "suggestions": result.get("suggestions", []),
                "preferences_learned": result.get("user_preferences", {}),
                "turn_count": result.get("turn_count", 1),
                "clarification_asked": result.get("needs_clarification", False),
            }

        except Exception as e:
            logger.error("chat_failed", error=str(e), session_id=session_id)
            return {
                "response": f"I encountered an error: {str(e)}",
                "session_id": session_id,
                "error": str(e),
            }

    async def get_conversation_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of messages
        """
        config = {"configurable": {"thread_id": session_id}}

        try:
            state = await self._graph.aget_state(config)
            messages = state.values.get("messages", [])

            return [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in messages
            ]
        except Exception as e:
            logger.error("get_history_failed", error=str(e))
            return []

    # === Helper Methods ===

    def _format_history(self, messages: list[BaseMessage], max_turns: int = 10) -> str:
        """Format message history for LLM context."""
        if not messages:
            return "(No previous conversation)"

        recent = messages[-max_turns * 2:]  # 2 messages per turn
        lines = []

        for msg in recent:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content[:200]}...")

        return "\n".join(lines)

    def _format_products(self, products: list[ProductInfo]) -> str:
        """Format products for LLM context."""
        if not products:
            return "(No products discussed yet)"

        lines = []
        for p in products[:5]:
            price = f"${p.get('price', 'N/A')}" if p.get('price') else "N/A"
            stars = f"{p.get('stars', 'N/A')}★" if p.get('stars') else "N/A"
            lines.append(f"{p.get('position', '?')}. {p.get('title', 'Unknown')} - {price}, {stars}")

        return "\n".join(lines)

    def _format_preferences(self, prefs: UserPreferences) -> str:
        """Format user preferences for LLM context."""
        if not prefs:
            return "(No preferences learned yet)"

        lines = []

        if prefs.get("budget_max"):
            lines.append(f"- Budget: up to ${prefs['budget_max']}")

        if prefs.get("preferred_brands"):
            lines.append(f"- Preferred brands: {', '.join(prefs['preferred_brands'])}")

        if prefs.get("important_features"):
            lines.append(f"- Important features: {', '.join(prefs['important_features'])}")

        if prefs.get("use_cases"):
            lines.append(f"- Use cases: {', '.join(prefs['use_cases'])}")

        return "\n".join(lines) if lines else "(No preferences learned yet)"

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        try:
            # Try direct parse
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def _generate_suggestions(self, state: ConversationState, results: dict) -> list[str]:
        """Generate contextual follow-up suggestions."""
        suggestions = []
        action = state.get("action_to_take", "")
        products = results.get("products", [])

        if action == "search" and products:
            suggestions = [
                "Compare the top products",
                "Which one is best for my needs?",
                "Show me cheaper options",
            ]
        elif action == "compare":
            suggestions = [
                "Tell me more about the winner",
                "What about durability?",
                "Which has better reviews?",
            ]
        elif action == "recommend":
            suggestions = [
                "Why this recommendation?",
                "Show me alternatives",
                "What's the price history?",
            ]
        elif action == "price_check":
            suggestions = [
                "Is this a good deal?",
                "Show me price history",
                "Find cheaper alternatives",
            ]
        else:
            suggestions = [
                "Search for products",
                "Compare options",
                "Get recommendations",
            ]

        return suggestions[:3]

    def _generate_fallback_response(self, results: dict) -> str:
        """Generate a fallback response when LLM fails."""
        result_type = results.get("type", "")

        if result_type == "search":
            products = results.get("products", [])
            if products:
                lines = ["Here are some products I found:\n"]
                for i, p in enumerate(products[:5], 1):
                    price = f"${p.get('price', 'N/A')}" if p.get('price') else ""
                    lines.append(f"{i}. **{p.get('title', 'Unknown')}** {price}")
                return "\n".join(lines)
            return "I couldn't find any products matching your search."

        elif result_type == "comparison":
            return "Here's a comparison of the products you mentioned."

        elif result_type == "recommendation":
            return "Based on your preferences, here are my recommendations."

        return "I'm here to help! What would you like to know about products?"


# === Singleton ===

_engine: LangGraphConversationEngine | None = None


async def get_langgraph_engine() -> LangGraphConversationEngine:
    """Get or create the LangGraph conversation engine singleton."""
    global _engine
    if _engine is None:
        _engine = LangGraphConversationEngine()
        await _engine.initialize()
    return _engine
