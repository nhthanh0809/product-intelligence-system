"""Intent models for query understanding.

This module defines the data structures for intent analysis,
entity extraction, and query decomposition.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """All possible query intents.

    Product-related intents map to user scenarios:
    - SEARCH: D1-D6 (Discovery)
    - COMPARE: C1-C5 (Comparison)
    - ANALYZE: A1-A5 (Analysis)
    - PRICE_CHECK: P1-P5 (Price Intelligence)
    - TREND: T1-T5 (Trend Analysis)
    - RECOMMEND: R1-R5 (Recommendations)

    General conversation intents:
    - GREETING, FAREWELL, HELP, SMALL_TALK, OFF_TOPIC

    Complex intents:
    - COMPOUND: Multiple intents in one query
    - AMBIGUOUS: Unclear intent needing clarification
    """

    # Product-related intents
    SEARCH = "search"                    # D1-D6: Find/discover products
    COMPARE = "compare"                  # C1-C5: Compare products
    ANALYZE = "analyze"                  # A1-A5: Analyze reviews/specs
    PRICE_CHECK = "price_check"          # P1-P5: Price intelligence
    TREND = "trend"                      # T1-T5: Market trends
    RECOMMEND = "recommend"              # R1-R5: Recommendations

    # General conversation intents
    GREETING = "greeting"                # "Hello", "Hi there"
    FAREWELL = "farewell"                # "Bye", "Thanks", "Goodbye"
    HELP = "help"                        # "What can you do?", "Help me"
    SMALL_TALK = "small_talk"            # "How are you?", "What's up?"
    OFF_TOPIC = "off_topic"              # Weather, news, jokes, etc.
    CLARIFICATION = "clarification"      # "What do you mean?", "Can you explain?"

    # Complex intents
    COMPOUND = "compound"                # Multiple intents in one query
    AMBIGUOUS = "ambiguous"              # Unclear intent

    @classmethod
    def product_intents(cls) -> set["QueryIntent"]:
        """Return set of product-related intents."""
        return {
            cls.SEARCH,
            cls.COMPARE,
            cls.ANALYZE,
            cls.PRICE_CHECK,
            cls.TREND,
            cls.RECOMMEND,
        }

    @classmethod
    def general_intents(cls) -> set["QueryIntent"]:
        """Return set of general conversation intents."""
        return {
            cls.GREETING,
            cls.FAREWELL,
            cls.HELP,
            cls.SMALL_TALK,
            cls.OFF_TOPIC,
            cls.CLARIFICATION,
            cls.AMBIGUOUS,  # Needs clarification from user
        }

    def is_product_related(self) -> bool:
        """Check if intent is product-related."""
        return self in self.product_intents()

    def is_general_chat(self) -> bool:
        """Check if intent is general conversation."""
        return self in self.general_intents()


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"            # Single intent, single step
    COMPOUND = "compound"        # Multiple intents, multiple steps
    MULTI_TURN = "multi_turn"    # Requires conversation context


class EntityExtraction(BaseModel):
    """Extracted entities from user query.

    Examples:
        "Sony WH-1000XM5 under $300" ->
            products: ["Sony WH-1000XM5"]
            brands: ["Sony"]
            constraints: {"price_max": 300}

        "Compare headphones vs earbuds for travel" ->
            categories: ["headphones", "earbuds"]
            use_cases: ["travel"]
            actions: ["compare"]
    """

    # Product identifiers
    products: list[str] = Field(
        default_factory=list,
        description="Specific product names or models mentioned",
    )
    brands: list[str] = Field(
        default_factory=list,
        description="Brand names mentioned",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Product categories mentioned",
    )
    asins: list[str] = Field(
        default_factory=list,
        description="Amazon ASINs if mentioned",
    )

    # Attributes and constraints
    attributes: list[str] = Field(
        default_factory=list,
        description="Product attributes to focus on (price, battery, rating)",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints like price_max, rating_min, etc.",
    )

    # Context
    use_cases: list[str] = Field(
        default_factory=list,
        description="Use cases mentioned (travel, gaming, work)",
    )
    actions: list[str] = Field(
        default_factory=list,
        description="Actions to perform (compare, recommend, find)",
    )

    # References
    references_previous: bool = Field(
        default=False,
        description="Whether query references previous results (them, it, those)",
    )
    reference_type: str | None = Field(
        default=None,
        description="Type of reference (products, comparison, recommendation)",
    )

    def has_specific_products(self) -> bool:
        """Check if specific products are mentioned."""
        return len(self.products) > 0 or len(self.asins) > 0

    def has_constraints(self) -> bool:
        """Check if any constraints are specified."""
        return len(self.constraints) > 0

    def get_price_range(self) -> tuple[float | None, float | None]:
        """Get price range from constraints."""
        return (
            self.constraints.get("price_min"),
            self.constraints.get("price_max"),
        )


class SubTask(BaseModel):
    """Sub-task for compound query decomposition.

    Used when a query contains multiple intents that need
    to be executed in sequence or parallel.

    Example:
        "Find headphones under $100, compare top 3, recommend best"

        SubTasks:
        1. {intent: SEARCH, description: "Find headphones under $100"}
        2. {intent: COMPARE, description: "Compare top 3", depends_on: [1]}
        3. {intent: RECOMMEND, description: "Recommend best", depends_on: [2]}
    """

    task_id: int = Field(description="Unique task identifier")
    intent: QueryIntent = Field(description="Intent for this sub-task")
    description: str = Field(description="Human-readable task description")

    # Task dependencies
    depends_on: list[int] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task",
    )
    input_from: list[int] = Field(
        default_factory=list,
        description="Task IDs to get input data from",
    )

    # Task-specific parameters
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters specific to this task",
    )

    # Execution hints
    can_parallelize: bool = Field(
        default=False,
        description="Whether this task can run in parallel with siblings",
    )
    estimated_latency_ms: int = Field(
        default=500,
        description="Estimated execution time in milliseconds",
    )

    def is_ready(self, completed_tasks: set[int]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.depends_on)


class IntentAnalysis(BaseModel):
    """Complete result of intent analysis.

    This is the output of the IntentAgent and contains all
    information needed to route and execute the query.
    """

    # Original query
    query: str = Field(description="Original user query")

    # Primary classification
    primary_intent: QueryIntent = Field(description="Main detected intent")
    secondary_intents: list[QueryIntent] = Field(
        default_factory=list,
        description="Additional intents detected",
    )

    # Entity extraction
    entities: EntityExtraction = Field(
        default_factory=EntityExtraction,
        description="Extracted entities from query",
    )

    # Query classification flags
    is_product_related: bool = Field(
        default=True,
        description="Whether query is about products",
    )
    is_general_chat: bool = Field(
        default=False,
        description="Whether query is general conversation",
    )
    requires_search: bool = Field(
        default=False,
        description="Whether query requires product search",
    )
    requires_comparison: bool = Field(
        default=False,
        description="Whether query requires product comparison",
    )
    requires_recommendation: bool = Field(
        default=False,
        description="Whether query requires recommendation",
    )

    # Complexity assessment
    complexity: QueryComplexity = Field(
        default=QueryComplexity.SIMPLE,
        description="Query complexity level",
    )
    sub_tasks: list[SubTask] = Field(
        default_factory=list,
        description="Decomposed sub-tasks for compound queries",
    )

    # Context
    conversation_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant conversation context",
    )

    # Confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the analysis",
    )

    # Routing hints
    suggested_agents: list[str] = Field(
        default_factory=list,
        description="Suggested agents to handle this query",
    )

    # Debug info
    reasoning: str = Field(
        default="",
        description="LLM reasoning for classification (for debugging)",
    )

    def get_all_intents(self) -> list[QueryIntent]:
        """Get all detected intents (primary + secondary)."""
        return [self.primary_intent] + self.secondary_intents

    def is_compound(self) -> bool:
        """Check if query is compound (multiple intents)."""
        return (
            self.complexity == QueryComplexity.COMPOUND
            or len(self.secondary_intents) > 0
            or len(self.sub_tasks) > 1
        )

    def needs_context(self) -> bool:
        """Check if query needs conversation context."""
        return (
            self.complexity == QueryComplexity.MULTI_TURN
            or self.entities.references_previous
        )

    def get_execution_order(self) -> list[SubTask]:
        """Get sub-tasks in execution order (topological sort)."""
        if not self.sub_tasks:
            return []

        # Simple topological sort
        result = []
        completed: set[int] = set()
        remaining = list(self.sub_tasks)

        while remaining:
            # Find tasks with satisfied dependencies
            ready = [t for t in remaining if t.is_ready(completed)]

            if not ready:
                # Circular dependency or error - return remaining as-is
                result.extend(remaining)
                break

            # Add ready tasks and mark as completed
            for task in ready:
                result.append(task)
                completed.add(task.task_id)
                remaining.remove(task)

        return result


# Type aliases for convenience
IntentType = QueryIntent
Complexity = QueryComplexity
