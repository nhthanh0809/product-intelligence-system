"""Recommend agent for product recommendations.

Enhanced with context awareness and improved reasoning.
"""

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.agents.base import BaseAgent, RetryConfig
from src.tools.search_tools import SearchToolkit, SearchToolConfig

logger = structlog.get_logger()


# Recommendation scoring weights
RECOMMENDATION_WEIGHTS = {
    "relevance": 0.30,      # How well it matches the query
    "rating": 0.25,         # Product rating
    "value": 0.20,          # Value for money
    "popularity": 0.15,     # Review count / popularity
    "recency": 0.10,        # How recent the product is
}


@dataclass
class UserPreferences:
    """User preferences for personalized recommendations."""
    preferred_brands: list[str] = field(default_factory=list)
    excluded_brands: list[str] = field(default_factory=list)
    price_range: tuple[float | None, float | None] = (None, None)
    min_rating: float | None = None
    use_cases: list[str] = field(default_factory=list)
    previous_purchases: list[str] = field(default_factory=list)  # ASINs
    viewed_products: list[str] = field(default_factory=list)  # ASINs


@dataclass
class RecommendInput:
    """Input for recommendation agent."""
    query: str
    source_asin: str | None = None  # Product to find similar/alternatives for
    source_product: dict[str, Any] | None = None
    recommendation_type: str = "similar"  # similar, alternatives, accessories, bundle
    # Enhanced fields
    user_preferences: UserPreferences | None = None
    context: dict[str, Any] = field(default_factory=dict)  # Conversation context
    max_recommendations: int = 10


@dataclass
class Recommendation:
    """A single product recommendation."""
    asin: str
    title: str
    brand: str | None = None
    price: float | None = None
    stars: float | None = None
    similarity_score: float = 0.0
    recommendation_reason: str = ""
    match_type: str = "similar"  # similar, alternative, accessory, complement
    # Enhanced fields
    overall_score: float = 0.0  # Weighted recommendation score
    value_score: float = 0.0  # Value for money score
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    best_for: str = ""
    rank: int = 0


@dataclass
class RecommendOutput:
    """Output from recommendation agent."""
    query: str
    recommendations: list[Recommendation] = field(default_factory=list)
    source_product: dict[str, Any] | None = None
    recommendation_summary: str = ""
    # Enhanced fields
    top_pick: Recommendation | None = None
    budget_pick: Recommendation | None = None
    premium_pick: Recommendation | None = None
    alternatives: list[Recommendation] = field(default_factory=list)
    reasoning: str = ""  # Explanation of recommendation logic


class RecommendAgent(BaseAgent[RecommendInput, RecommendOutput]):
    """Recommendation agent for product suggestions.

    Handles scenarios R1-R5:
    - R1: Similar products
    - R2: Alternative products
    - R3: Accessories/compatible
    - R4: Frequently bought together
    - R5: Personalized recommendations
    """

    name = "recommend"
    description = "Product recommendations and alternatives"

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
        input_data: RecommendInput,
    ) -> RecommendOutput:
        """Execute recommendation."""
        source_product = input_data.source_product

        # Get source product if ASIN provided
        if input_data.source_asin and not source_product:
            source_product = await self._fetch_product(input_data.source_asin)

        # Get recommendations based on type
        if input_data.recommendation_type == "similar" and source_product:
            recommendations = await self._get_similar_products(
                source_product.get("asin", ""),
                source_product,
            )
        elif input_data.recommendation_type == "alternatives":
            recommendations = await self._get_alternatives(
                input_data.query,
                source_product,
            )
        elif input_data.recommendation_type == "accessories":
            recommendations = await self._get_accessories(
                input_data.query,
                source_product,
            )
        else:
            # Default: search-based recommendations
            recommendations = await self._get_search_recommendations(
                input_data.query,
            )

        # Apply user preferences if provided
        recommendations = self._apply_preferences(
            recommendations,
            input_data.user_preferences,
        )

        # Calculate recommendation scores
        recommendations = self._calculate_recommendation_scores(recommendations)

        # Limit to max_recommendations
        recommendations = recommendations[:input_data.max_recommendations]

        # Generate recommendation reasons
        recommendations = await self._generate_reasons(
            recommendations,
            source_product,
            input_data.query,
        )

        # Identify top, budget, and premium picks
        top_pick, budget_pick, premium_pick = self._identify_picks(recommendations)

        # Generate alternatives
        alternatives = self._generate_alternatives(recommendations, top_pick)

        # Build summary
        summary = self._build_summary(
            recommendations,
            source_product,
            input_data.recommendation_type,
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            top_pick,
            recommendations,
            source_product,
            input_data.query,
        )

        return RecommendOutput(
            query=input_data.query,
            recommendations=recommendations,
            source_product=source_product,
            recommendation_summary=summary,
            top_pick=top_pick,
            budget_pick=budget_pick,
            premium_pick=premium_pick,
            alternatives=alternatives,
            reasoning=reasoning,
        )

    async def _fetch_product(self, asin: str) -> dict[str, Any] | None:
        """Fetch product by ASIN."""
        results = await self._search_toolkit.keyword_search(asin, limit=1)
        if results.results:
            return results.results[0].model_dump()
        return None

    async def _get_similar_products(
        self,
        source_asin: str,
        source_product: dict[str, Any] | None,
    ) -> list[Recommendation]:
        """Get similar products using vector similarity."""
        results = await self._search_toolkit.find_similar(
            source_asin,
            limit=10,
        )

        recommendations = []
        for result in results.results:
            recommendations.append(Recommendation(
                asin=result.asin,
                title=result.title,
                brand=result.brand,
                price=result.price,
                stars=result.stars,
                similarity_score=result.score,
                match_type="similar",
            ))

        return recommendations

    async def _get_alternatives(
        self,
        query: str,
        source_product: dict[str, Any] | None,
    ) -> list[Recommendation]:
        """Get alternative products (same category, different brands)."""
        filters = {}

        if source_product:
            # Same category, different brand
            if source_product.get("category_level1"):
                filters["category_level1"] = source_product["category_level1"]

            # Exclude same brand for true alternatives
            search_query = query
            if source_product.get("product_type"):
                search_query = f"{source_product['product_type']} alternatives"

        results = await self._search_toolkit.hybrid_search(
            query,
            limit=10,
            filters=filters if filters else None,
        )

        recommendations = []
        source_brand = source_product.get("brand", "").lower() if source_product else ""

        for result in results.results:
            # Skip same brand
            if result.brand and result.brand.lower() == source_brand:
                continue

            recommendations.append(Recommendation(
                asin=result.asin,
                title=result.title,
                brand=result.brand,
                price=result.price,
                stars=result.stars,
                similarity_score=result.score,
                match_type="alternative",
            ))

        return recommendations[:8]

    async def _get_accessories(
        self,
        query: str,
        source_product: dict[str, Any] | None,
    ) -> list[Recommendation]:
        """Get accessories/compatible products."""
        # Build accessory search query
        if source_product:
            product_type = source_product.get("product_type", "")
            brand = source_product.get("brand", "")
            search_query = f"{brand} {product_type} accessories compatible"
        else:
            search_query = f"{query} accessories"

        results = await self._search_toolkit.hybrid_search(
            search_query,
            limit=10,
        )

        recommendations = []
        for result in results.results:
            recommendations.append(Recommendation(
                asin=result.asin,
                title=result.title,
                brand=result.brand,
                price=result.price,
                stars=result.stars,
                similarity_score=result.score,
                match_type="accessory",
            ))

        return recommendations

    async def _get_search_recommendations(
        self,
        query: str,
    ) -> list[Recommendation]:
        """Get recommendations based on search query."""
        results = await self._search_toolkit.hybrid_search(
            query,
            limit=10,
        )

        recommendations = []
        for result in results.results:
            recommendations.append(Recommendation(
                asin=result.asin,
                title=result.title,
                brand=result.brand,
                price=result.price,
                stars=result.stars,
                similarity_score=result.score,
                match_type="recommended",
            ))

        return recommendations

    async def _generate_reasons(
        self,
        recommendations: list[Recommendation],
        source_product: dict[str, Any] | None,
        query: str,
    ) -> list[Recommendation]:
        """Generate recommendation reasons using LLM."""
        if not recommendations:
            return recommendations

        # Build context
        source_info = ""
        if source_product:
            source_info = f"""
Source Product: {source_product.get('title', 'Unknown')[:50]}
Brand: {source_product.get('brand', 'Unknown')}
Price: ${source_product.get('price', 'N/A')}
"""

        rec_list = "\n".join([
            f"{i+1}. {r.title[:50]} ({r.brand}) - ${r.price}"
            for i, r in enumerate(recommendations[:5])
        ])

        prompt = f"""For each recommended product, provide a brief reason (1 sentence) why it's recommended.

{source_info}
User Query: {query}

Recommendations:
{rec_list}

Format:
1. [Reason for product 1]
2. [Reason for product 2]
..."""

        response = await self._http_client.post(
            "/generate",
            json={
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 300,
            },
        )
        response.raise_for_status()

        raw = response.json().get("response", "")

        # Parse reasons
        reasons = []
        for line in raw.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                reason = line.lstrip('0123456789.-) ').strip()
                reasons.append(reason)

        # Assign reasons to recommendations
        for i, rec in enumerate(recommendations[:len(reasons)]):
            if i < len(reasons):
                rec.recommendation_reason = reasons[i]

        return recommendations

    def _build_summary(
        self,
        recommendations: list[Recommendation],
        source_product: dict[str, Any] | None,
        recommendation_type: str,
    ) -> str:
        """Build recommendation summary."""
        if not recommendations:
            return "No recommendations found."

        parts = [f"Found {len(recommendations)} {recommendation_type} recommendations"]

        if source_product:
            parts.append(f"based on {source_product.get('title', 'Unknown')[:30]}")

        # Price range
        prices = [r.price for r in recommendations if r.price]
        if prices:
            parts.append(f"priced ${min(prices):.2f} - ${max(prices):.2f}")

        # Top brands
        brands = list(set(r.brand for r in recommendations if r.brand))[:3]
        if brands:
            parts.append(f"from {', '.join(brands)}")

        return ". ".join(parts) + "."

    def _apply_preferences(
        self,
        recommendations: list[Recommendation],
        preferences: UserPreferences | None,
    ) -> list[Recommendation]:
        """Apply user preferences to filter and rank recommendations.

        Args:
            recommendations: Initial recommendations
            preferences: User preferences

        Returns:
            Filtered and re-ranked recommendations
        """
        if not preferences:
            return recommendations

        filtered = []
        for rec in recommendations:
            # Skip excluded brands
            if rec.brand and rec.brand.lower() in [
                b.lower() for b in preferences.excluded_brands
            ]:
                continue

            # Check price range
            min_price, max_price = preferences.price_range
            if rec.price:
                if min_price and rec.price < min_price:
                    continue
                if max_price and rec.price > max_price:
                    continue

            # Check minimum rating
            if preferences.min_rating and rec.stars:
                if rec.stars < preferences.min_rating:
                    continue

            # Boost preferred brands
            if rec.brand and rec.brand.lower() in [
                b.lower() for b in preferences.preferred_brands
            ]:
                rec.overall_score *= 1.2  # 20% boost

            filtered.append(rec)

        return filtered

    def _calculate_recommendation_scores(
        self,
        recommendations: list[Recommendation],
    ) -> list[Recommendation]:
        """Calculate overall recommendation scores.

        Args:
            recommendations: Recommendations to score

        Returns:
            Recommendations with calculated scores
        """
        if not recommendations:
            return recommendations

        # Normalize each factor
        prices = [r.price for r in recommendations if r.price]
        ratings = [r.stars for r in recommendations if r.stars]
        similarities = [r.similarity_score for r in recommendations]

        for rec in recommendations:
            score = 0.0

            # Relevance (similarity)
            if similarities and max(similarities) > 0:
                relevance = rec.similarity_score / max(similarities)
                score += relevance * RECOMMENDATION_WEIGHTS["relevance"]

            # Rating
            if rec.stars and ratings:
                rating_norm = (rec.stars - min(ratings)) / (max(ratings) - min(ratings) + 0.01)
                score += rating_norm * RECOMMENDATION_WEIGHTS["rating"]

            # Value (inverse price normalized)
            if rec.price and prices:
                max_price = max(prices)
                min_price = min(prices)
                if max_price > min_price:
                    value_norm = 1 - (rec.price - min_price) / (max_price - min_price)
                else:
                    value_norm = 0.5
                rec.value_score = value_norm * 100
                score += value_norm * RECOMMENDATION_WEIGHTS["value"]

            rec.overall_score = score * 100  # Scale to 0-100

        # Sort by overall score and assign ranks
        recommendations.sort(key=lambda r: r.overall_score, reverse=True)
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1

        return recommendations

    def _identify_picks(
        self,
        recommendations: list[Recommendation],
    ) -> tuple[Recommendation | None, Recommendation | None, Recommendation | None]:
        """Identify top, budget, and premium picks.

        Args:
            recommendations: Scored recommendations

        Returns:
            Tuple of (top_pick, budget_pick, premium_pick)
        """
        if not recommendations:
            return None, None, None

        # Top pick is highest overall score
        top_pick = recommendations[0] if recommendations else None

        # Budget pick is best value among lower-priced items
        priced = [r for r in recommendations if r.price]
        if priced:
            avg_price = sum(r.price for r in priced) / len(priced)
            budget_options = [r for r in priced if r.price < avg_price]
            if budget_options:
                budget_pick = max(budget_options, key=lambda r: r.overall_score)
            else:
                budget_pick = min(priced, key=lambda r: r.price)
        else:
            budget_pick = None

        # Premium pick is highest rated among higher-priced items
        if priced:
            premium_options = [r for r in priced if r.price >= avg_price]
            if premium_options:
                premium_pick = max(premium_options, key=lambda r: r.stars or 0)
            else:
                premium_pick = max(priced, key=lambda r: r.price)
        else:
            premium_pick = None

        return top_pick, budget_pick, premium_pick

    def _generate_alternatives(
        self,
        recommendations: list[Recommendation],
        top_pick: Recommendation | None,
    ) -> list[Recommendation]:
        """Generate alternative suggestions different from top pick.

        Args:
            recommendations: All recommendations
            top_pick: The top recommendation

        Returns:
            List of alternatives with different characteristics (max 3)
        """
        if not recommendations or not top_pick:
            return []

        alternatives = []
        top_brand = top_pick.brand.lower() if top_pick.brand else ""
        top_price = top_pick.price or 0

        for rec in recommendations[1:]:  # Skip top pick
            # Check limit first
            if len(alternatives) >= 3:
                break

            if rec.asin == top_pick.asin:
                continue

            # Different brand alternative
            if rec.brand and rec.brand.lower() != top_brand:
                if not any(a.brand and a.brand.lower() == rec.brand.lower() for a in alternatives):
                    rec.recommendation_reason = f"Alternative from {rec.brand}"
                    alternatives.append(rec)
                    continue

            # Cheaper alternative
            if rec.price and rec.price < top_price * 0.8:
                if not any(a.price and a.price < top_price * 0.8 for a in alternatives):
                    rec.recommendation_reason = f"Budget-friendly alternative (${rec.price:.2f})"
                    alternatives.append(rec)
                    continue

            # Higher-rated alternative
            if rec.stars and top_pick.stars and rec.stars > top_pick.stars:
                if not any(a.stars and a.stars > (top_pick.stars or 0) for a in alternatives):
                    rec.recommendation_reason = f"Higher-rated alternative ({rec.stars:.1f} stars)"
                    alternatives.append(rec)
                    continue

        return alternatives

    def _build_reasoning(
        self,
        top_pick: Recommendation | None,
        recommendations: list[Recommendation],
        source_product: dict[str, Any] | None,
        query: str,
    ) -> str:
        """Build explanation for recommendations.

        Args:
            top_pick: Top recommendation
            recommendations: All recommendations
            source_product: Source product if any
            query: User query

        Returns:
            Explanation string
        """
        parts = []

        if source_product:
            parts.append(
                f"Based on {source_product.get('title', 'your selected product')[:40]}, "
                f"I've found {len(recommendations)} recommendations."
            )
        else:
            parts.append(f"Based on your search for '{query[:50]}', I've found {len(recommendations)} recommendations.")

        if top_pick:
            reasons = []
            if top_pick.stars and top_pick.stars >= 4.5:
                reasons.append(f"highly rated ({top_pick.stars:.1f} stars)")
            if top_pick.value_score and top_pick.value_score >= 70:
                reasons.append("excellent value")
            if top_pick.similarity_score and top_pick.similarity_score >= 0.8:
                reasons.append("closely matches your criteria")

            if reasons:
                parts.append(f"The top pick is {', '.join(reasons)}.")

        return " ".join(parts)


# Singleton
_recommend_agent: RecommendAgent | None = None


async def get_recommend_agent() -> RecommendAgent:
    """Get or create recommend agent singleton."""
    global _recommend_agent
    if _recommend_agent is None:
        _recommend_agent = RecommendAgent()
        await _recommend_agent.initialize()
    return _recommend_agent


async def get_recommendations(
    query: str,
    source_asin: str | None = None,
    recommendation_type: str = "similar",
) -> RecommendOutput:
    """Convenience function for recommendations.

    Args:
        query: Recommendation query
        source_asin: Optional source product ASIN
        recommendation_type: Type of recommendations

    Returns:
        RecommendOutput with recommendations
    """
    agent = await get_recommend_agent()
    return await agent.execute(RecommendInput(
        query=query,
        source_asin=source_asin,
        recommendation_type=recommendation_type,
    ))
