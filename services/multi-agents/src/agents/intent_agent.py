"""Intent Understanding Agent.

This agent analyzes user queries to determine:
- Primary and secondary intents
- Entity extraction (products, brands, constraints)
- Query complexity (simple, compound, multi-turn)
- Whether it's product-related or general conversation

Enhanced with LLM support for complex/ambiguous queries.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.agents.base import BaseAgent, RetryConfig
from src.config import get_settings
from src.models.intent import (
    QueryIntent,
    QueryComplexity,
    EntityExtraction,
    SubTask,
    IntentAnalysis,
)

logger = structlog.get_logger()
settings = get_settings()


@dataclass
class LLMClassificationResult:
    """Result from LLM-based classification."""
    intents: list[QueryIntent]
    entities: dict[str, Any] = field(default_factory=dict)
    is_compound: bool = False
    confidence: float = 0.8
    reasoning: str = ""


class IntentAgent(BaseAgent[str, IntentAnalysis]):
    """Agent for understanding user query intent.

    Capabilities:
    - Intent classification (12+ intent types)
    - Entity extraction (products, brands, constraints)
    - Compound query detection and decomposition
    - General chat detection
    - Context resolution for references
    """

    name = "intent"
    description = "Analyzes user queries to determine intent and extract entities"

    # Patterns for rule-based detection (fallback)
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))(\s+there)?[\s!.,]*$",
        r"^(what'?s\s+up|howdy|yo)[\s!.,]*$",
    ]

    FAREWELL_PATTERNS = [
        r"^(bye|goodbye|see\s+you|thanks?|thank\s+you|cheers)[\s!.,]*$",
        r"^(have\s+a\s+(good|nice)\s+(day|one)|take\s+care)[\s!.,]*$",
    ]

    HELP_PATTERNS = [
        r"(what\s+can\s+you\s+do|help(\s+me)?|how\s+do\s+(i|you)|capabilities)",
        r"^(help|assist(ance)?|\?)[\s!.,]*$",
    ]

    SMALL_TALK_PATTERNS = [
        r"^how\s+are\s+you",
        r"^how('?s|\s+is)\s+(it\s+going|everything|life)",
        r"^what('?s|\s+is)\s+your\s+name",
    ]

    OFF_TOPIC_PATTERNS = [
        r"(weather|temperature|forecast)",
        r"(news|politics|sports\s+score)",
        r"(tell\s+(me\s+)?a\s+joke|funny)",
        r"(what\s+time|what\s+day|what\s+date)",
        r"(calculate|math|equation|\d+\s*[\+\-\*/]\s*\d+)",
    ]

    COMPARE_PATTERNS = [
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompare\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+(is|one)\s+(better|best)\b",
        r"\bcompared\s+to\b",
    ]

    RECOMMEND_PATTERNS = [
        r"\brecommend\b",
        r"\bsuggest\b",
        r"\bsimilar\s+(to|products?)\b",
        r"\balternatives?\b",
        r"\baccessories\s+for\b",
        r"\bwhat\s+should\s+i\s+(buy|get)\b",
    ]

    ANALYZE_PATTERNS = [
        r"\breviews?\b",
        r"\bpros?\s+(and\s+)?cons?\b",
        r"\bwhat\s+do\s+(people|users?|customers?)\s+say\b",
        r"\bsentiment\b",
        r"\bquality\b",
        r"\bdurable|durability\b",
    ]

    PRICE_PATTERNS = [
        r"\bprice\b",
        r"\bdeal[s]?\b",
        r"\bdiscount\b",
        r"\bbudget\b",
        r"\bcheap(er|est)?\b",
        r"\baffordable\b",
        r"\bworth\s+(it|the\s+(money|price))\b",
        r"\bis\s+\$?\d+\s+(a\s+)?good\b",
    ]

    TREND_PATTERNS = [
        r"\btrending\b",
        r"\bpopular\b",
        r"\bhot\s+products?\b",
        r"\bnew\s+releases?\b",
        r"\bgaining\s+popularity\b",
        r"\bbest\s+sell(er|ing)\b",
    ]

    # Constraint extraction patterns
    PRICE_CONSTRAINT_PATTERNS = [
        (r"under\s*\$?(\d+(?:\.\d+)?)", "price_max"),
        (r"below\s*\$?(\d+(?:\.\d+)?)", "price_max"),
        (r"less\s+than\s*\$?(\d+(?:\.\d+)?)", "price_max"),
        (r"up\s+to\s*\$?(\d+(?:\.\d+)?)", "price_max"),
        (r"(?:over|above|more\s+than)\s*\$?(\d+(?:\.\d+)?)", "price_min"),
        (r"\$(\d+(?:\.\d+)?)\s*(?:-|to)\s*\$?(\d+(?:\.\d+)?)", "price_range"),
        (r"between\s*\$?(\d+(?:\.\d+)?)\s*and\s*\$?(\d+(?:\.\d+)?)", "price_range"),
    ]

    RATING_CONSTRAINT_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*\+?\s*stars?", "rating_min"),
        (r"(?:above|over|at\s+least)\s*(\d+(?:\.\d+)?)\s*stars?", "rating_min"),
        (r"(?:top|best)\s*rated", 4.5),
        (r"highly\s*rated", 4.0),
    ]

    # Known brands for extraction
    KNOWN_BRANDS = {
        "sony", "bose", "apple", "samsung", "lg", "anker", "jbl", "sennheiser",
        "dell", "hp", "lenovo", "microsoft", "google", "amazon", "kindle",
        "dewalt", "milwaukee", "makita", "bosch", "ryobi", "black+decker",
        "canon", "nikon", "fujifilm", "dyson", "shark", "roomba", "irobot",
        "kitchenaid", "ninja", "instant pot", "vitamix", "philips", "panasonic",
        "logitech", "razer", "corsair", "asus", "acer", "msi", "nvidia", "amd",
        "intel", "epson", "brother", "beats", "jabra", "skullcandy", "audio-technica",
    }

    # Product categories
    CATEGORIES = {
        "headphones": ["headphones", "earbuds", "earphones", "headset", "airpods"],
        "laptops": ["laptop", "notebook", "macbook", "chromebook"],
        "phones": ["phone", "smartphone", "iphone", "android", "mobile"],
        "cameras": ["camera", "dslr", "mirrorless", "gopro", "webcam"],
        "tvs": ["tv", "television", "smart tv", "oled", "qled"],
        "speakers": ["speaker", "soundbar", "subwoofer", "bluetooth speaker"],
        "keyboards": ["keyboard", "mechanical keyboard", "wireless keyboard"],
        "mice": ["mouse", "gaming mouse", "wireless mouse", "trackpad"],
        "monitors": ["monitor", "display", "screen", "gaming monitor"],
        "tablets": ["tablet", "ipad", "android tablet", "e-reader"],
        "gaming": ["gaming", "console", "playstation", "xbox", "nintendo"],
        "smart home": ["smart home", "alexa", "google home", "smart speaker"],
        "wearables": ["watch", "smartwatch", "fitness tracker", "fitbit"],
        "appliances": ["appliance", "refrigerator", "washer", "dryer", "dishwasher"],
        "coffee": ["coffee", "espresso", "coffee maker", "keurig", "nespresso"],
    }

    # Reference words that indicate multi-turn context
    REFERENCE_WORDS = [
        "it", "them", "they", "these", "those", "this", "that",
        "the same", "similar ones", "more like", "other options",
    ]

    # LLM classification prompt template
    LLM_CLASSIFICATION_PROMPT = """You are an intent classification system for a product intelligence assistant.

Analyze this user query and classify it:

Query: "{query}"

## Instructions
1. Identify the primary intent from these options:
   - search: Finding/discovering products (e.g., "find headphones", "show me laptops")
   - compare: Comparing specific products (e.g., "compare X vs Y", "which is better")
   - analyze: Analyzing reviews/specs (e.g., "what do reviews say", "pros and cons")
   - price_check: Price evaluation (e.g., "is this a good price", "price history")
   - trend: Market trends (e.g., "what's popular", "trending products")
   - recommend: Product recommendations (e.g., "recommend headphones", "what should I buy")
   - greeting: Hello/Hi messages
   - farewell: Goodbye/Thanks messages
   - help: Questions about capabilities
   - small_talk: Casual conversation
   - off_topic: Non-product related

2. If the query has multiple intents, list them in execution order.

3. Extract any entities mentioned:
   - products: Specific product names or models
   - brands: Brand names mentioned
   - categories: Product categories
   - price_constraints: Price limits (e.g., "under $100")
   - use_cases: Intended use (e.g., "for gaming", "for work")

4. Determine if this is a compound query (multiple distinct actions).

## Output (JSON only, no other text):
{{
    "primary_intent": "intent_name",
    "secondary_intents": ["intent2", "intent3"],
    "entities": {{
        "products": [],
        "brands": [],
        "categories": [],
        "price_max": null,
        "price_min": null,
        "use_cases": []
    }},
    "is_compound": false,
    "confidence": 0.9,
    "reasoning": "Brief explanation of classification"
}}"""

    def __init__(
        self,
        use_llm: bool = True,
        llm_confidence_threshold: float = 0.7,
    ):
        """Initialize the Intent Agent.

        Args:
            use_llm: Whether to use LLM for complex/ambiguous queries
            llm_confidence_threshold: Minimum rule-based confidence to skip LLM
        """
        super().__init__(retry_config=RetryConfig(max_retries=2))
        self.use_llm = use_llm
        self.llm_confidence_threshold = llm_confidence_threshold
        self._llm_manager = None

    async def initialize(self) -> None:
        """Initialize LLM manager if LLM is enabled."""
        if self.use_llm:
            try:
                from src.llm import get_llm_manager
                self._llm_manager = await get_llm_manager()
                logger.info("intent_agent_llm_initialized")
            except Exception as e:
                logger.warning(
                    "intent_agent_llm_init_failed",
                    error=str(e),
                    fallback="rule-based only",
                )
                self._llm_manager = None
        await super().initialize()

    async def close(self) -> None:
        """Clean up resources."""
        self._llm_manager = None
        await super().close()

    async def _execute_internal(self, query: str) -> IntentAnalysis:
        """Analyze query intent.

        Uses a hybrid approach:
        1. Quick rule-based detection for obvious intents
        2. LLM classification for complex/ambiguous queries
        3. Confidence-based routing between rules and LLM

        Args:
            query: User query string

        Returns:
            IntentAnalysis with detected intents and entities
        """
        query_lower = query.lower().strip()
        llm_reasoning = ""

        # Step 1: Quick rule-based check for general chat
        general_intent = self._detect_general_intent(query_lower)
        if general_intent:
            return IntentAnalysis(
                query=query,
                primary_intent=general_intent,
                is_product_related=False,
                is_general_chat=True,
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                suggested_agents=["general"],
            )

        # Step 2: Extract entities with rules
        entities = self._extract_entities(query, query_lower)

        # Step 3: Detect product intents (rule-based)
        intents = self._detect_product_intents(query_lower)
        rule_confidence = self._calculate_confidence(intents, entities)

        # Step 4: Decide whether to use LLM
        # Use LLM if:
        # - LLM is enabled and available
        # - Rule-based confidence is below threshold OR query is complex
        should_use_llm = (
            self.use_llm
            and self._llm_manager is not None
            and self._llm_manager.is_initialized
            and (
                rule_confidence < self.llm_confidence_threshold
                or len(intents) > 2
                or not intents
            )
        )

        if should_use_llm:
            llm_result = await self._llm_classify(query, entities)
            if llm_result:
                # Merge LLM results if confidence is higher
                if llm_result.confidence > rule_confidence:
                    intents = llm_result.intents
                    llm_reasoning = llm_result.reasoning

                    # Merge entities from LLM
                    entities = self._merge_entities(entities, llm_result.entities)

                    logger.debug(
                        "llm_classification_used",
                        query=query[:50],
                        llm_confidence=llm_result.confidence,
                        rule_confidence=rule_confidence,
                    )

        # Step 5: Determine primary intent
        primary_intent = intents[0] if intents else QueryIntent.SEARCH
        secondary_intents = intents[1:] if len(intents) > 1 else []

        # Step 6: Check for compound query
        complexity = QueryComplexity.SIMPLE
        sub_tasks: list[SubTask] = []

        if len(intents) > 1:
            complexity = QueryComplexity.COMPOUND
            sub_tasks = self._create_sub_tasks(intents, entities)

        # Step 7: Check for context references
        if entities.references_previous:
            complexity = QueryComplexity.MULTI_TURN

        # Step 8: Build analysis result
        final_confidence = max(rule_confidence, llm_result.confidence if should_use_llm and llm_result else 0)

        analysis = IntentAnalysis(
            query=query,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            entities=entities,
            is_product_related=True,
            is_general_chat=False,
            requires_search=primary_intent in {QueryIntent.SEARCH, QueryIntent.TREND}
                or any(i in {QueryIntent.SEARCH, QueryIntent.TREND} for i in secondary_intents),
            requires_comparison=QueryIntent.COMPARE in intents,
            requires_recommendation=QueryIntent.RECOMMEND in intents,
            complexity=complexity,
            sub_tasks=sub_tasks,
            confidence=final_confidence,
            suggested_agents=self._suggest_agents(intents),
            reasoning=llm_reasoning,
        )

        logger.debug(
            "intent_analyzed",
            query=query[:50],
            primary_intent=primary_intent.value,
            complexity=complexity.value,
            entity_count=len(entities.products) + len(entities.brands),
            used_llm=should_use_llm and llm_result is not None,
        )

        return analysis

    def _detect_general_intent(self, query_lower: str) -> QueryIntent | None:
        """Detect general conversation intent using patterns."""
        # Check greeting
        for pattern in self.GREETING_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryIntent.GREETING

        # Check farewell
        for pattern in self.FAREWELL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryIntent.FAREWELL

        # Check help
        for pattern in self.HELP_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryIntent.HELP

        # Check small talk
        for pattern in self.SMALL_TALK_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryIntent.SMALL_TALK

        # Check off-topic (only if no product-related words)
        has_product_words = any(
            word in query_lower
            for word in ["product", "buy", "price", "review", "compare", "recommend"]
        )
        if not has_product_words:
            for pattern in self.OFF_TOPIC_PATTERNS:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return QueryIntent.OFF_TOPIC

        return None

    def _detect_product_intents(self, query_lower: str) -> list[QueryIntent]:
        """Detect product-related intents using patterns."""
        intents: list[QueryIntent] = []

        # Check compare (high priority)
        for pattern in self.COMPARE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                intents.append(QueryIntent.COMPARE)
                break

        # Check recommend
        for pattern in self.RECOMMEND_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if QueryIntent.RECOMMEND not in intents:
                    intents.append(QueryIntent.RECOMMEND)
                break

        # Check analyze
        for pattern in self.ANALYZE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if QueryIntent.ANALYZE not in intents:
                    intents.append(QueryIntent.ANALYZE)
                break

        # Check price
        for pattern in self.PRICE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if QueryIntent.PRICE_CHECK not in intents:
                    intents.append(QueryIntent.PRICE_CHECK)
                break

        # Check trend
        for pattern in self.TREND_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                if QueryIntent.TREND not in intents:
                    intents.append(QueryIntent.TREND)
                break

        # Default to search if looking for products
        search_words = ["find", "search", "show", "get", "looking for", "need", "want"]
        if any(word in query_lower for word in search_words):
            if QueryIntent.SEARCH not in intents:
                intents.insert(0, QueryIntent.SEARCH)

        # If no intents detected but has product words, default to search
        if not intents:
            intents.append(QueryIntent.SEARCH)

        return intents

    def _extract_entities(self, query: str, query_lower: str) -> EntityExtraction:
        """Extract entities from query."""
        entities = EntityExtraction()

        # Extract brands
        for brand in self.KNOWN_BRANDS:
            if brand.lower() in query_lower:
                entities.brands.append(brand.title())

        # Extract categories
        for category, keywords in self.CATEGORIES.items():
            if any(kw in query_lower for kw in keywords):
                entities.categories.append(category)

        # Extract product names (brand + model patterns)
        product_patterns = [
            r"([A-Z][a-z]+)\s+([A-Z]{2,}[\-]?\d+[A-Z]*\d*)",  # Sony WH-1000XM5
            r"([A-Z][a-z]+)\s+(Air\s*Pods?\s*(?:Pro|Max)?)",  # Apple AirPods Pro
            r"(MacBook|Surface|ThinkPad|XPS)\s*\w*",          # MacBook Pro
        ]
        for pattern in product_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    product = " ".join(match).strip()
                else:
                    product = match.strip()
                if product and product not in entities.products:
                    entities.products.append(product)

        # Extract ASINs
        asin_pattern = r"\b([A-Z0-9]{10})\b"
        asins = re.findall(asin_pattern, query)
        entities.asins = [a for a in asins if a.startswith("B0")]

        # Extract constraints
        entities.constraints = self._extract_constraints(query_lower)

        # Extract attributes
        attribute_words = [
            "price", "cost", "battery", "quality", "durability",
            "rating", "review", "spec", "feature", "performance",
            "size", "weight", "color", "warranty",
        ]
        entities.attributes = [w for w in attribute_words if w in query_lower]

        # Extract actions
        action_words = ["find", "search", "compare", "recommend", "analyze", "show"]
        entities.actions = [w for w in action_words if w in query_lower]

        # Extract use cases
        use_case_patterns = [
            (r"for\s+(travel|traveling|commut\w+)", "travel"),
            (r"for\s+(gaming|games)", "gaming"),
            (r"for\s+(work|office|business)", "work"),
            (r"for\s+(music|listening)", "music"),
            (r"for\s+(exercise|workout|gym|running)", "fitness"),
            (r"for\s+(study|school|college)", "education"),
        ]
        for pattern, use_case in use_case_patterns:
            if re.search(pattern, query_lower):
                entities.use_cases.append(use_case)

        # Check for references to previous context
        for ref_word in self.REFERENCE_WORDS:
            if re.search(rf"\b{ref_word}\b", query_lower):
                entities.references_previous = True
                entities.reference_type = "products"
                break

        return entities

    def _extract_constraints(self, query_lower: str) -> dict[str, Any]:
        """Extract price and rating constraints."""
        constraints: dict[str, Any] = {}

        # Price constraints
        for pattern, constraint_type in self.PRICE_CONSTRAINT_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                if constraint_type == "price_range":
                    constraints["price_min"] = float(match.group(1))
                    constraints["price_max"] = float(match.group(2))
                elif constraint_type == "price_max":
                    constraints["price_max"] = float(match.group(1))
                elif constraint_type == "price_min":
                    constraints["price_min"] = float(match.group(1))
                break

        # Rating constraints
        for item in self.RATING_CONSTRAINT_PATTERNS:
            if isinstance(item[1], float):
                pattern, rating = item
                if re.search(pattern, query_lower):
                    constraints["rating_min"] = rating
                    break
            else:
                pattern, constraint_type = item
                match = re.search(pattern, query_lower)
                if match:
                    constraints["rating_min"] = float(match.group(1))
                    break

        # Limit constraints
        limit_pattern = r"(?:top|first|best)\s*(\d+)"
        match = re.search(limit_pattern, query_lower)
        if match:
            constraints["limit"] = int(match.group(1))

        return constraints

    async def _llm_classify(
        self, query: str, entities: EntityExtraction
    ) -> LLMClassificationResult | None:
        """Use LLM for intent classification.

        Args:
            query: User query string
            entities: Pre-extracted entities from rules

        Returns:
            LLMClassificationResult or None if LLM fails
        """
        if not self._llm_manager or not self._llm_manager.is_initialized:
            return None

        prompt = self.LLM_CLASSIFICATION_PROMPT.format(query=query)

        try:
            from src.llm import GenerationConfig

            result = await self._llm_manager.generate_for_agent(
                agent_name="intent",
                prompt=prompt,
                config=GenerationConfig(
                    temperature=0.1,
                    max_tokens=500,
                    response_format="json",
                ),
            )

            # Parse JSON response
            response_text = result.content.strip()

            # Try to find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                data = json.loads(json_match.group())

                # Convert string intents to QueryIntent
                intent_map = {
                    "search": QueryIntent.SEARCH,
                    "compare": QueryIntent.COMPARE,
                    "analyze": QueryIntent.ANALYZE,
                    "price_check": QueryIntent.PRICE_CHECK,
                    "trend": QueryIntent.TREND,
                    "recommend": QueryIntent.RECOMMEND,
                    "greeting": QueryIntent.GREETING,
                    "farewell": QueryIntent.FAREWELL,
                    "help": QueryIntent.HELP,
                    "small_talk": QueryIntent.SMALL_TALK,
                    "off_topic": QueryIntent.OFF_TOPIC,
                }

                primary = intent_map.get(
                    data.get("primary_intent", "search"),
                    QueryIntent.SEARCH,
                )

                # Handle case where secondary_intents is null/None
                secondary_raw = data.get("secondary_intents") or []
                secondary = [
                    intent_map.get(i, QueryIntent.SEARCH)
                    for i in secondary_raw
                    if i in intent_map
                ]

                intents = [primary] + secondary

                return LLMClassificationResult(
                    intents=intents,
                    entities=data.get("entities", {}),
                    is_compound=data.get("is_compound", False),
                    confidence=data.get("confidence", 0.8),
                    reasoning=data.get("reasoning", ""),
                )

        except json.JSONDecodeError as e:
            logger.warning(
                "llm_intent_json_parse_failed",
                error=str(e),
                response=response_text[:200] if 'response_text' in locals() else None,
            )
        except Exception as e:
            logger.warning("llm_intent_analysis_failed", error=str(e))

        return None

    def _merge_entities(
        self, rule_entities: EntityExtraction, llm_entities: dict[str, Any]
    ) -> EntityExtraction:
        """Merge LLM-extracted entities with rule-based entities.

        Args:
            rule_entities: Entities from rule-based extraction
            llm_entities: Entities from LLM classification

        Returns:
            Merged EntityExtraction
        """
        # Start with rule-based entities
        merged = rule_entities

        # Add LLM products not already found (handle null values)
        llm_products = llm_entities.get("products") or []
        for product in llm_products:
            if product and product not in merged.products:
                merged.products.append(product)

        # Add LLM brands not already found (handle null values)
        llm_brands = llm_entities.get("brands") or []
        for brand in llm_brands:
            if brand and brand not in merged.brands:
                merged.brands.append(brand)

        # Add LLM categories not already found (handle null values)
        llm_categories = llm_entities.get("categories") or []
        for category in llm_categories:
            if category and category not in merged.categories:
                merged.categories.append(category)

        # Add LLM use cases not already found (handle null values)
        llm_use_cases = llm_entities.get("use_cases") or []
        for use_case in llm_use_cases:
            if use_case and use_case not in merged.use_cases:
                merged.use_cases.append(use_case)

        # Update price constraints if LLM found them and rules didn't
        if "price_max" not in merged.constraints and llm_entities.get("price_max"):
            merged.constraints["price_max"] = llm_entities["price_max"]
        if "price_min" not in merged.constraints and llm_entities.get("price_min"):
            merged.constraints["price_min"] = llm_entities["price_min"]

        return merged

    def _create_sub_tasks(
        self, intents: list[QueryIntent], entities: EntityExtraction
    ) -> list[SubTask]:
        """Create sub-tasks for compound queries."""
        sub_tasks: list[SubTask] = []
        task_id = 0

        for i, intent in enumerate(intents):
            task_id += 1
            depends_on = [task_id - 1] if i > 0 else []
            input_from = [task_id - 1] if i > 0 else []

            description = self._get_task_description(intent, entities)
            parameters = self._get_task_parameters(intent, entities)

            sub_tasks.append(
                SubTask(
                    task_id=task_id,
                    intent=intent,
                    description=description,
                    depends_on=depends_on,
                    input_from=input_from,
                    parameters=parameters,
                    can_parallelize=i == 0,  # First task can start immediately
                    estimated_latency_ms=self._estimate_latency(intent),
                )
            )

        return sub_tasks

    def _get_task_description(
        self, intent: QueryIntent, entities: EntityExtraction
    ) -> str:
        """Generate human-readable task description."""
        descriptions = {
            QueryIntent.SEARCH: "Search for products",
            QueryIntent.COMPARE: "Compare products",
            QueryIntent.ANALYZE: "Analyze product reviews/specs",
            QueryIntent.PRICE_CHECK: "Evaluate prices",
            QueryIntent.TREND: "Find trending products",
            QueryIntent.RECOMMEND: "Generate recommendations",
        }

        base = descriptions.get(intent, "Process query")

        # Add context
        if entities.categories:
            base += f" in {entities.categories[0]}"
        if entities.brands:
            base += f" ({', '.join(entities.brands[:2])})"
        if "price_max" in entities.constraints:
            base += f" under ${entities.constraints['price_max']}"

        return base

    def _get_task_parameters(
        self, intent: QueryIntent, entities: EntityExtraction
    ) -> dict[str, Any]:
        """Get task-specific parameters."""
        params: dict[str, Any] = {}

        if intent == QueryIntent.SEARCH:
            params["filters"] = entities.constraints
            if entities.categories:
                params["category"] = entities.categories[0]
            if entities.brands:
                params["brands"] = entities.brands

        elif intent == QueryIntent.COMPARE:
            params["products"] = entities.products
            params["attributes"] = entities.attributes or ["price", "rating"]

        elif intent == QueryIntent.ANALYZE:
            params["focus"] = entities.attributes or ["reviews"]

        elif intent == QueryIntent.RECOMMEND:
            params["type"] = "similar" if entities.products else "category"
            if entities.use_cases:
                params["use_case"] = entities.use_cases[0]

        return params

    def _estimate_latency(self, intent: QueryIntent) -> int:
        """Estimate task latency in milliseconds."""
        latency_map = {
            QueryIntent.SEARCH: 200,
            QueryIntent.COMPARE: 400,
            QueryIntent.ANALYZE: 300,
            QueryIntent.PRICE_CHECK: 200,
            QueryIntent.TREND: 500,
            QueryIntent.RECOMMEND: 300,
        }
        return latency_map.get(intent, 300)

    def _calculate_confidence(
        self, intents: list[QueryIntent], entities: EntityExtraction
    ) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.5

        # Higher confidence if we found specific entities
        if entities.products:
            confidence += 0.15
        if entities.brands:
            confidence += 0.1
        if entities.constraints:
            confidence += 0.1
        if entities.categories:
            confidence += 0.05

        # Lower confidence for compound queries
        if len(intents) > 2:
            confidence -= 0.1

        return min(max(confidence, 0.3), 0.95)

    def _suggest_agents(self, intents: list[QueryIntent]) -> list[str]:
        """Suggest agents to handle the query."""
        agent_map = {
            QueryIntent.SEARCH: "retrieval",
            QueryIntent.COMPARE: "comparison",
            QueryIntent.ANALYZE: "attribute",
            QueryIntent.PRICE_CHECK: "attribute",
            QueryIntent.TREND: "retrieval",
            QueryIntent.RECOMMEND: "recommendation",
            QueryIntent.GREETING: "general",
            QueryIntent.FAREWELL: "general",
            QueryIntent.HELP: "general",
            QueryIntent.SMALL_TALK: "general",
            QueryIntent.OFF_TOPIC: "general",
        }

        agents = []
        for intent in intents:
            agent = agent_map.get(intent)
            if agent and agent not in agents:
                agents.append(agent)

        # Always end with synthesis
        if agents and "synthesis" not in agents:
            agents.append("synthesis")

        return agents


# Singleton instance
_intent_agent: IntentAgent | None = None


async def get_intent_agent(use_llm: bool = True) -> IntentAgent:
    """Get or create intent agent singleton."""
    global _intent_agent
    if _intent_agent is None:
        _intent_agent = IntentAgent(use_llm=use_llm)
        await _intent_agent.initialize()
    return _intent_agent


async def analyze_intent(query: str) -> IntentAnalysis:
    """Analyze query intent.

    Convenience function for quick intent analysis.

    Args:
        query: User query string

    Returns:
        IntentAnalysis with detected intents and entities
    """
    agent = await get_intent_agent()
    return await agent.execute(query)
