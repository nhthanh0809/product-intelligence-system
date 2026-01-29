"""Compare agent for product comparison.

Enhanced with AttributeAgent integration for structured comparisons.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.agents.base import BaseAgent, RetryConfig
from src.tools.search_tools import SearchToolkit, SearchToolConfig

logger = structlog.get_logger()


# Comparison attributes with weights for scoring
COMPARISON_ATTRIBUTES = {
    "price": {"weight": 0.25, "lower_is_better": True},
    "rating": {"weight": 0.25, "lower_is_better": False},
    "value_score": {"weight": 0.20, "lower_is_better": False},
    "review_count": {"weight": 0.15, "lower_is_better": False},
    "popularity_score": {"weight": 0.15, "lower_is_better": False},
}


@dataclass
class CompareInput:
    """Input for comparison agent."""
    query: str
    products: list[dict[str, Any]] = field(default_factory=list)
    product_names: list[str] = field(default_factory=list)  # For searching


@dataclass
class ProductComparison:
    """Comparison data for a single product."""
    asin: str
    title: str
    brand: str | None = None
    price: float | None = None
    stars: float | None = None
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    best_for: str = ""
    score: float = 0.0  # Overall comparison score


@dataclass
class AttributeComparison:
    """Comparison of a single attribute across products."""
    attribute: str
    values: dict[str, float | None]  # product_id -> value
    normalized_values: dict[str, float]  # product_id -> 0-1 normalized
    winner_id: str | None = None
    unit: str | None = None


@dataclass
class ComparisonMatrix:
    """Structured comparison matrix."""
    product_ids: list[str]
    product_names: dict[str, str]  # id -> name
    attributes: list[AttributeComparison]
    overall_scores: dict[str, float]  # product_id -> weighted score
    winner_id: str | None = None
    best_value_id: str | None = None


@dataclass
class ComparisonResult:
    """Result of product comparison."""
    products: list[ProductComparison] = field(default_factory=list)
    winner: ProductComparison | None = None
    winner_reason: str = ""
    best_value: ProductComparison | None = None
    comparison_summary: str = ""
    key_differences: list[str] = field(default_factory=list)
    recommendation: str = ""
    # Enhanced fields
    comparison_matrix: ComparisonMatrix | None = None
    attribute_winners: dict[str, str] = field(default_factory=dict)  # attr -> product_id


@dataclass
class CompareOutput:
    """Output from comparison agent."""
    query: str
    comparison: ComparisonResult
    raw_response: str = ""


class CompareAgent(BaseAgent[CompareInput, CompareOutput]):
    """Compare agent for product comparisons.

    Handles scenarios C1-C5:
    - C1: Side-by-side comparison
    - C2: Feature comparison
    - C3: Price-value comparison
    - C4: Best for use-case
    - C5: Winner determination
    """

    name = "compare"
    description = "Product comparison and winner determination"

    def __init__(self):
        super().__init__(retry_config=RetryConfig(max_retries=2))
        self._http_client: httpx.AsyncClient | None = None
        self._search_toolkit: SearchToolkit | None = None

    async def initialize(self) -> None:
        """Initialize clients."""
        self._http_client = httpx.AsyncClient(
            base_url=self.settings.ollama_service_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        )

        config = SearchToolConfig.from_settings()
        self._search_toolkit = SearchToolkit(config)
        await self._search_toolkit.initialize()

        await super().initialize()

    async def close(self) -> None:
        """Close clients."""
        if self._http_client:
            await self._http_client.aclose()
        if self._search_toolkit:
            await self._search_toolkit.close()

    async def _execute_internal(
        self,
        input_data: CompareInput,
    ) -> CompareOutput:
        """Execute product comparison."""
        products = input_data.products

        # If no products, search for them
        if not products and input_data.product_names:
            products = await self._search_products(input_data.product_names)
        elif not products:
            # Extract product names from query
            product_names = self._extract_product_names(input_data.query)
            if product_names:
                products = await self._search_products(product_names)

        if len(products) < 2:
            return CompareOutput(
                query=input_data.query,
                comparison=ComparisonResult(
                    comparison_summary="Need at least 2 products to compare.",
                ),
            )

        # Build comparison context
        context = self._build_comparison_context(products[:4])

        # Generate comparison using LLM
        prompt = self._build_comparison_prompt(input_data.query, context)

        response = await self._http_client.post(
            "/generate",
            json={
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 1000,
            },
        )
        response.raise_for_status()
        raw_response = response.json().get("response", "")

        # Parse comparison result
        comparison = self._parse_comparison(raw_response, products[:4])

        # Build structured comparison matrix
        matrix = self.build_comparison_matrix(products[:4])
        comparison.comparison_matrix = matrix

        # Add attribute winners
        for attr in matrix.attributes:
            if attr.winner_id:
                comparison.attribute_winners[attr.attribute] = attr.winner_id

        # Generate key differences from matrix if not from LLM
        if not comparison.key_differences:
            comparison.key_differences = self.generate_key_differences(
                matrix, products[:4]
            )

        # Update winner from matrix if not determined by LLM
        if not comparison.winner and matrix.winner_id:
            for pc in comparison.products:
                if pc.asin == matrix.winner_id:
                    comparison.winner = pc
                    comparison.winner_reason = (
                        f"Highest overall score ({matrix.overall_scores.get(matrix.winner_id, 0):.1f}/100)"
                    )
                    break

        # Update best value from matrix
        if not comparison.best_value and matrix.best_value_id:
            for pc in comparison.products:
                if pc.asin == matrix.best_value_id:
                    comparison.best_value = pc
                    break

        # Add scores to product comparisons
        for pc in comparison.products:
            if pc.asin in matrix.overall_scores:
                pc.score = matrix.overall_scores[pc.asin]

        return CompareOutput(
            query=input_data.query,
            comparison=comparison,
            raw_response=raw_response,
        )

    async def _search_products(
        self,
        product_names: list[str],
    ) -> list[dict[str, Any]]:
        """Search for products by name."""
        all_products = []

        for name in product_names[:4]:
            results = await self._search_toolkit.keyword_search(name, limit=1)
            if results.results:
                all_products.append(results.results[0].model_dump())

        return all_products

    def _extract_product_names(self, query: str) -> list[str]:
        """Extract product names from comparison query."""
        # Pattern: "A vs B" or "A or B" or "compare A and B"
        patterns = [
            r'(.+?)\s+(?:vs\.?|versus|compared? to)\s+(.+)',
            r'compare\s+(.+?)\s+(?:and|with|to)\s+(.+)',
            r'(.+?)\s+or\s+(.+?)(?:\?|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                names = [match.group(1).strip(), match.group(2).strip()]
                # Clean up names
                names = [
                    re.sub(r'^(the|a|an)\s+', '', n, flags=re.IGNORECASE)
                    for n in names
                ]
                return names

        return []

    def _build_comparison_context(
        self,
        products: list[dict[str, Any]],
    ) -> str:
        """Build comparison context from products."""
        context_parts = []

        for i, product in enumerate(products):
            parts = [
                f"PRODUCT {i+1}:",
                f"  Name: {product.get('title', 'Unknown')[:80]}",
                f"  Brand: {product.get('brand', 'Unknown')}",
                f"  Price: ${product.get('price', 'N/A')}",
                f"  Rating: {product.get('stars', 'N/A')}/5",
            ]

            if product.get("genAI_best_for"):
                parts.append(f"  Best for: {product['genAI_best_for']}")

            if product.get("genAI_pros"):
                pros = product["genAI_pros"]
                if isinstance(pros, list):
                    parts.append(f"  Pros: {', '.join(pros[:3])}")

            if product.get("genAI_cons"):
                cons = product["genAI_cons"]
                if isinstance(cons, list):
                    parts.append(f"  Cons: {', '.join(cons[:3])}")

            if product.get("genAI_key_capabilities"):
                caps = product["genAI_key_capabilities"]
                if isinstance(caps, list):
                    parts.append(f"  Features: {', '.join(caps[:4])}")

            context_parts.append("\n".join(parts))

        return "\n\n".join(context_parts)

    def _build_comparison_prompt(self, query: str, context: str) -> str:
        """Build LLM prompt for comparison."""
        return f"""Compare these products based on the user's query.

User Query: {query}

Products to Compare:
{context}

Provide a structured comparison:

KEY DIFFERENCES:
- List main differences between products

COMPARISON TABLE:
For each product, rate (1-10):
- Value for money
- Features
- Quality/Durability
- User experience

WINNER: [Product name]
WINNER REASON: Why this product wins

BEST VALUE: [Product name]
VALUE REASON: Why this is the best value

RECOMMENDATION:
Specific recommendation based on the query. Who should buy which product?

SUMMARY:
Brief comparison summary (2-3 sentences)"""

    def _parse_comparison(
        self,
        response: str,
        products: list[dict[str, Any]],
    ) -> ComparisonResult:
        """Parse LLM response into comparison result."""
        result = ComparisonResult()

        # Create product comparisons
        for product in products:
            pc = ProductComparison(
                asin=product.get("asin", ""),
                title=product.get("title", ""),
                brand=product.get("brand"),
                price=product.get("price"),
                stars=product.get("stars"),
                pros=product.get("genAI_pros", [])[:3] if isinstance(product.get("genAI_pros"), list) else [],
                cons=product.get("genAI_cons", [])[:3] if isinstance(product.get("genAI_cons"), list) else [],
                best_for=product.get("genAI_best_for", ""),
            )
            result.products.append(pc)

        # Extract winner
        winner_match = re.search(r'WINNER[:\s]*(.+?)(?:\n|WINNER REASON)', response, re.IGNORECASE)
        if winner_match:
            winner_name = winner_match.group(1).strip()
            # Find matching product
            for pc in result.products:
                if winner_name.lower() in pc.title.lower():
                    result.winner = pc
                    break

        # Extract winner reason
        reason_match = re.search(r'WINNER REASON[:\s]*(.+?)(?:\n\n|BEST VALUE)', response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            result.winner_reason = reason_match.group(1).strip()

        # Extract best value
        value_match = re.search(r'BEST VALUE[:\s]*(.+?)(?:\n|VALUE REASON)', response, re.IGNORECASE)
        if value_match:
            value_name = value_match.group(1).strip()
            for pc in result.products:
                if value_name.lower() in pc.title.lower():
                    result.best_value = pc
                    break

        # Extract key differences
        diff_match = re.search(r'KEY DIFFERENCES[:\s]*((?:[-•]\s*.+\n?)+)', response, re.IGNORECASE)
        if diff_match:
            diff_text = diff_match.group(1)
            result.key_differences = [
                line.strip().lstrip('-•').strip()
                for line in diff_text.split('\n')
                if line.strip().lstrip('-•').strip()
            ]

        # Extract recommendation
        rec_match = re.search(r'RECOMMENDATION[:\s]*(.+?)(?:\n\n|SUMMARY)', response, re.IGNORECASE | re.DOTALL)
        if rec_match:
            result.recommendation = rec_match.group(1).strip()

        # Extract summary
        summary_match = re.search(r'SUMMARY[:\s]*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            result.comparison_summary = summary_match.group(1).strip()
        else:
            result.comparison_summary = response[:300]

        return result

    def build_comparison_matrix(
        self,
        products: list[dict[str, Any]],
    ) -> ComparisonMatrix:
        """Build structured comparison matrix from products.

        Args:
            products: Products to compare

        Returns:
            ComparisonMatrix with normalized values and scores
        """
        product_ids = [p.get("asin", str(i)) for i, p in enumerate(products)]
        product_names = {
            p.get("asin", str(i)): p.get("title", "Unknown")[:50]
            for i, p in enumerate(products)
        }

        attributes: list[AttributeComparison] = []

        # Extract and normalize each attribute
        for attr_name, config in COMPARISON_ATTRIBUTES.items():
            values: dict[str, float | None] = {}
            unit = None

            for p in products:
                pid = p.get("asin", str(products.index(p)))

                if attr_name == "price":
                    values[pid] = p.get("price")
                    unit = "USD"
                elif attr_name == "rating":
                    values[pid] = p.get("stars", p.get("rating"))
                    unit = "stars"
                elif attr_name == "review_count":
                    values[pid] = p.get("review_count", p.get("reviews"))
                elif attr_name == "value_score":
                    # Calculate value score if we have price and rating
                    price = p.get("price")
                    rating = p.get("stars", p.get("rating"))
                    if price and rating and price > 0:
                        values[pid] = (rating / (price / 10)) * 10
                    else:
                        values[pid] = None
                elif attr_name == "popularity_score":
                    # Calculate from rating and reviews
                    rating = p.get("stars", p.get("rating", 0))
                    reviews = p.get("review_count", p.get("reviews", 0))
                    if reviews and reviews > 0:
                        import math
                        values[pid] = rating * math.log10(reviews + 1)
                    else:
                        values[pid] = None

            # Normalize values to 0-1 range
            normalized = self._normalize_values(
                values,
                lower_is_better=config["lower_is_better"],
            )

            # Determine winner for this attribute
            winner_id = None
            if normalized:
                winner_id = max(normalized.items(), key=lambda x: x[1])[0]

            attributes.append(AttributeComparison(
                attribute=attr_name,
                values=values,
                normalized_values=normalized,
                winner_id=winner_id,
                unit=unit,
            ))

        # Calculate overall scores
        overall_scores = self._calculate_overall_scores(attributes)

        # Determine overall winner
        winner_id = None
        if overall_scores:
            winner_id = max(overall_scores.items(), key=lambda x: x[1])[0]

        # Determine best value (highest value_score)
        best_value_id = None
        for attr in attributes:
            if attr.attribute == "value_score" and attr.winner_id:
                best_value_id = attr.winner_id
                break

        return ComparisonMatrix(
            product_ids=product_ids,
            product_names=product_names,
            attributes=attributes,
            overall_scores=overall_scores,
            winner_id=winner_id,
            best_value_id=best_value_id,
        )

    def _normalize_values(
        self,
        values: dict[str, float | None],
        lower_is_better: bool = False,
    ) -> dict[str, float]:
        """Normalize values to 0-1 range.

        Args:
            values: Raw values by product ID
            lower_is_better: If True, lower values get higher normalized scores

        Returns:
            Normalized values (0-1) by product ID
        """
        valid_values = {k: v for k, v in values.items() if v is not None}
        if not valid_values:
            return {}

        min_val = min(valid_values.values())
        max_val = max(valid_values.values())

        # Handle case where all values are the same
        if max_val == min_val:
            return {k: 0.5 for k in valid_values}

        normalized = {}
        for pid, val in valid_values.items():
            if lower_is_better:
                # Invert so lower values become higher scores
                normalized[pid] = 1 - (val - min_val) / (max_val - min_val)
            else:
                normalized[pid] = (val - min_val) / (max_val - min_val)

        return normalized

    def _calculate_overall_scores(
        self,
        attributes: list[AttributeComparison],
    ) -> dict[str, float]:
        """Calculate weighted overall scores.

        Args:
            attributes: List of attribute comparisons

        Returns:
            Overall scores by product ID
        """
        scores: dict[str, float] = {}

        for attr in attributes:
            weight = COMPARISON_ATTRIBUTES.get(attr.attribute, {}).get("weight", 0.1)

            for pid, norm_val in attr.normalized_values.items():
                if pid not in scores:
                    scores[pid] = 0.0
                scores[pid] += norm_val * weight

        # Normalize to 0-100 scale
        if scores:
            max_possible = sum(c["weight"] for c in COMPARISON_ATTRIBUTES.values())
            scores = {k: (v / max_possible) * 100 for k, v in scores.items()}

        return scores

    def generate_key_differences(
        self,
        matrix: ComparisonMatrix,
        products: list[dict[str, Any]],
    ) -> list[str]:
        """Generate key differences from comparison matrix.

        Args:
            matrix: Comparison matrix
            products: Original products

        Returns:
            List of key difference statements
        """
        differences = []

        for attr in matrix.attributes:
            if not attr.values or len(attr.values) < 2:
                continue

            valid_values = {k: v for k, v in attr.values.items() if v is not None}
            if len(valid_values) < 2:
                continue

            min_id = min(valid_values.items(), key=lambda x: x[1])[0]
            max_id = max(valid_values.items(), key=lambda x: x[1])[0]

            if min_id == max_id:
                continue

            min_name = matrix.product_names.get(min_id, "Product")[:30]
            max_name = matrix.product_names.get(max_id, "Product")[:30]
            min_val = valid_values[min_id]
            max_val = valid_values[max_id]

            if attr.attribute == "price":
                diff_pct = ((max_val - min_val) / min_val) * 100
                differences.append(
                    f"{min_name} is ${max_val - min_val:.2f} ({diff_pct:.0f}%) cheaper than {max_name}"
                )
            elif attr.attribute == "rating":
                differences.append(
                    f"{max_name} has higher rating ({max_val:.1f}) vs {min_name} ({min_val:.1f})"
                )
            elif attr.attribute == "review_count":
                differences.append(
                    f"{max_name} has more reviews ({int(max_val):,}) vs {min_name} ({int(min_val):,})"
                )
            elif attr.attribute == "value_score":
                differences.append(
                    f"{max_name} offers better value for money"
                )

        return differences[:5]  # Limit to top 5 differences


# Singleton
_compare_agent: CompareAgent | None = None


async def get_compare_agent() -> CompareAgent:
    """Get or create compare agent singleton."""
    global _compare_agent
    if _compare_agent is None:
        _compare_agent = CompareAgent()
        await _compare_agent.initialize()
    return _compare_agent


async def compare_products(
    query: str,
    products: list[dict[str, Any]] | None = None,
    product_names: list[str] | None = None,
) -> CompareOutput:
    """Convenience function to compare products.

    Args:
        query: Comparison query
        products: Optional products to compare
        product_names: Optional product names to search

    Returns:
        CompareOutput with comparison results
    """
    agent = await get_compare_agent()
    return await agent.execute(CompareInput(
        query=query,
        products=products or [],
        product_names=product_names or [],
    ))
