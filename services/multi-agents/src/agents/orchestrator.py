"""Orchestrator agent for routing queries to specialist agents."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.agents.base import BaseAgent, RetryConfig

logger = structlog.get_logger()


class AgentType(str, Enum):
    """Available agent types."""
    SEARCH = "search"
    COMPARE = "compare"
    ANALYSIS = "analysis"
    PRICE = "price"
    TREND = "trend"
    RECOMMEND = "recommend"


@dataclass
class RoutingResult:
    """Result of query routing."""
    primary_agent: AgentType
    secondary_agents: list[AgentType] = field(default_factory=list)
    confidence: float = 0.0
    intent: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    requires_search: bool = True


class IntentPatterns:
    """Pattern-based intent detection."""

    # Pattern -> (AgentType, intent_name)
    PATTERNS: list[tuple[str, AgentType, str]] = [
        # Search patterns
        (r'\b(find|search|show|looking for|discover)\b', AgentType.SEARCH, "product_discovery"),
        (r'\b(what|which)\s+(products?|items?)\b', AgentType.SEARCH, "product_query"),

        # Comparison patterns
        (r'\b(compare|comparison|vs\.?|versus|difference between|which is better)\b',
         AgentType.COMPARE, "comparison"),
        (r'\b(between)\b.*\b(and|or)\b', AgentType.COMPARE, "comparison"),

        # Analysis patterns (reviews)
        (r'\b(reviews?|rating|rated|feedback|opinions?)\b', AgentType.ANALYSIS, "review_analysis"),
        (r'\b(pros?|cons?|advantages?|disadvantages?)\b', AgentType.ANALYSIS, "pros_cons"),
        (r'\b(sentiment|satisfaction|complaints?|issues?)\b', AgentType.ANALYSIS, "sentiment"),
        (r'\bwhat (do|are) (people|users?|customers?) (saying|think)\b', AgentType.ANALYSIS, "review_mining"),

        # Price patterns
        (r'\b(price|cost|cheap|expensive|affordable|budget|deal|discount|sale)\b',
         AgentType.PRICE, "price_query"),
        (r'\$\d+|\d+\s*dollars?', AgentType.PRICE, "price_range"),
        (r'\b(worth|value for money|good deal|overpriced)\b', AgentType.PRICE, "value_assessment"),
        (r'\b(price history|price drop|price change)\b', AgentType.PRICE, "price_tracking"),

        # Trend patterns
        (r'\b(trending|popular|hot|best sell(ing|er)|top)\b', AgentType.TREND, "trending"),
        (r'\b(market|category|segment)\s+(analysis|trends?|overview)\b', AgentType.TREND, "market_analysis"),
        (r'\b(most (bought|purchased|sold))\b', AgentType.TREND, "popularity"),

        # Recommendation patterns
        (r'\b(recommend|suggest|alternatives?|similar|like this)\b', AgentType.RECOMMEND, "recommendation"),
        (r'\b(accessories|compatible|works with|goes with)\b', AgentType.RECOMMEND, "accessories"),
        (r'\b(also (buy|bought|consider)|frequently bought)\b', AgentType.RECOMMEND, "bundle"),
    ]

    @classmethod
    def detect(cls, query: str) -> list[tuple[AgentType, str, float]]:
        """Detect intents from query.

        Returns list of (agent_type, intent, score) tuples.
        """
        query_lower = query.lower()
        matches = []

        for pattern, agent_type, intent in cls.PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Higher score for more specific patterns
                specificity = len(pattern) / 100  # Normalize
                matches.append((agent_type, intent, min(0.9, 0.5 + specificity)))

        return matches


class EntityExtractor:
    """Extract entities from queries."""

    KNOWN_BRANDS = [
        "Sony", "Bose", "Apple", "Samsung", "LG", "Anker", "JBL", "Sennheiser",
        "Dell", "HP", "Lenovo", "Microsoft", "Google", "Amazon", "Kindle",
        "DeWalt", "Milwaukee", "Makita", "Bosch", "Ryobi", "Black+Decker",
        "Canon", "Nikon", "Fujifilm", "Dyson", "Shark", "Roomba", "iRobot",
        "KitchenAid", "Ninja", "Instant Pot", "Vitamix", "Philips", "Panasonic",
    ]

    CATEGORY_KEYWORDS = {
        "electronics": ["electronics", "electronic", "gadget", "device", "tech"],
        "audio": ["headphones", "earbuds", "speakers", "soundbar", "audio"],
        "computers": ["computer", "laptop", "desktop", "pc", "notebook", "tablet"],
        "cameras": ["camera", "photography", "dslr", "mirrorless"],
        "home": ["home", "kitchen", "appliance", "furniture"],
        "tools": ["tool", "tools", "power tool", "hand tool"],
        "sports": ["sports", "fitness", "outdoor", "exercise"],
    }

    @classmethod
    def extract(cls, query: str) -> dict[str, Any]:
        """Extract entities from query."""
        entities = {}

        # Extract brands
        query_lower = query.lower()
        for brand in cls.KNOWN_BRANDS:
            if brand.lower() in query_lower:
                entities["brand"] = brand
                break

        # Extract categories
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                entities["category"] = category
                break

        # Extract price range
        price_patterns = [
            (r'under\s*\$?(\d+)', "max"),
            (r'below\s*\$?(\d+)', "max"),
            (r'over\s*\$?(\d+)', "min"),
            (r'above\s*\$?(\d+)', "min"),
            (r'\$(\d+)\s*[-to]+\s*\$?(\d+)', "range"),
        ]

        for pattern, price_type in price_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if price_type == "max":
                    entities["price_max"] = float(match.group(1))
                elif price_type == "min":
                    entities["price_min"] = float(match.group(1))
                elif price_type == "range":
                    entities["price_min"] = float(match.group(1))
                    entities["price_max"] = float(match.group(2))
                break

        # Extract product mentions for comparison
        vs_match = re.search(
            r'(.+?)\s+(?:vs\.?|versus|compared? to|or)\s+(.+)',
            query,
            re.IGNORECASE
        )
        if vs_match:
            entities["compare_items"] = [
                vs_match.group(1).strip(),
                vs_match.group(2).strip(),
            ]

        return entities


@dataclass
class OrchestratorInput:
    """Input for orchestrator."""
    query: str
    context: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None


@dataclass
class OrchestratorOutput:
    """Output from orchestrator."""
    routing: RoutingResult
    enriched_query: str
    context: dict[str, Any] = field(default_factory=dict)


class OrchestratorAgent(BaseAgent[OrchestratorInput, OrchestratorOutput]):
    """Orchestrator agent that routes queries to specialist agents.

    Routing Matrix:
    | Intent Pattern        | Primary Agent | Secondary  | Target Latency |
    |----------------------|---------------|------------|----------------|
    | "find", "search"     | SearchAgent   | -          | <200ms         |
    | "compare", "vs"      | CompareAgent  | Search     | <500ms         |
    | "reviews", "pros"    | AnalysisAgent | Search     | <400ms         |
    | "price", "deal"      | PriceAgent    | Search     | <400ms         |
    | "trending", "popular"| TrendAgent    | -          | <2s            |
    | "similar", "recommend"| RecommendAgent| Search    | <400ms         |
    """

    name = "orchestrator"
    description = "Routes queries to specialist agents"

    def __init__(self):
        super().__init__(retry_config=RetryConfig(max_retries=2))
        self._http_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._http_client = httpx.AsyncClient(
            base_url=self.settings.ollama_service_url,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=30.0),
        )
        await super().initialize()

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _execute_internal(
        self,
        input_data: OrchestratorInput,
    ) -> OrchestratorOutput:
        """Route query to appropriate agent."""
        query = input_data.query

        # Step 1: Pattern-based intent detection (fast)
        intent_matches = IntentPatterns.detect(query)

        # Step 2: Extract entities
        entities = EntityExtractor.extract(query)

        # Step 3: Determine primary agent
        if intent_matches:
            # Use highest confidence match
            primary_agent, intent, confidence = max(
                intent_matches, key=lambda x: x[2]
            )

            # Determine secondary agents
            secondary_agents = []
            other_agents = [a for a, _, _ in intent_matches if a != primary_agent]
            secondary_agents.extend(other_agents[:2])

            # Most agents need search first
            requires_search = primary_agent not in (AgentType.SEARCH, AgentType.TREND)

        else:
            # Default to search
            primary_agent = AgentType.SEARCH
            intent = "general_search"
            confidence = 0.5
            secondary_agents = []
            requires_search = False

        # Step 4: Build routing result
        routing = RoutingResult(
            primary_agent=primary_agent,
            secondary_agents=secondary_agents,
            confidence=confidence,
            intent=intent,
            entities=entities,
            requires_search=requires_search,
        )

        # Step 5: Enrich query with context
        enriched_query = self._enrich_query(query, entities, input_data.context)

        # Step 6: Build output context
        output_context = {
            **input_data.context,
            "entities": entities,
            "intent": intent,
            "routing": {
                "primary": primary_agent.value,
                "secondary": [a.value for a in secondary_agents],
            },
        }

        logger.info(
            "query_routed",
            query=query[:50],
            primary_agent=primary_agent.value,
            intent=intent,
            confidence=confidence,
            entities=entities,
        )

        return OrchestratorOutput(
            routing=routing,
            enriched_query=enriched_query,
            context=output_context,
        )

    def _enrich_query(
        self,
        query: str,
        entities: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """Enrich query with extracted context."""
        # For now, just return original query
        # Could be enhanced with query expansion
        return query


# Singleton
_orchestrator: OrchestratorAgent | None = None


async def get_orchestrator() -> OrchestratorAgent:
    """Get or create orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
        await _orchestrator.initialize()
    return _orchestrator


async def route_query(
    query: str,
    context: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> RoutingResult:
    """Route a query to the appropriate agent.

    Args:
        query: User query
        context: Optional context
        session_id: Optional session ID

    Returns:
        RoutingResult with agent assignments
    """
    orchestrator = await get_orchestrator()
    input_data = OrchestratorInput(
        query=query,
        context=context or {},
        session_id=session_id,
    )
    output = await orchestrator.execute(input_data)
    return output.routing
