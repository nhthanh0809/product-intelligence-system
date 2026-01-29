"""Analysis agent for review mining and sentiment analysis."""

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from src.config import get_settings
from src.agents.base import BaseAgent, RetryConfig
from src.tools.search_tools import SearchToolkit, SearchToolConfig

logger = structlog.get_logger()


@dataclass
class AnalysisInput:
    """Input for analysis agent."""
    query: str
    products: list[dict[str, Any]] = field(default_factory=list)
    asin: str | None = None  # Specific product to analyze
    analysis_type: str = "general"  # general, pros_cons, sentiment, feature


@dataclass
class ReviewAnalysis:
    """Analyzed review data."""
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_label: str = "neutral"
    common_themes: list[str] = field(default_factory=list)
    feature_feedback: dict[str, str] = field(default_factory=dict)
    summary: str = ""


@dataclass
class AnalysisOutput:
    """Output from analysis agent."""
    query: str
    analysis: ReviewAnalysis
    products_analyzed: int = 0
    source_products: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


class AnalysisAgent(BaseAgent[AnalysisInput, AnalysisOutput]):
    """Analysis agent for review mining.

    Handles scenarios A1-A5:
    - A1: Pros and cons extraction
    - A2: Sentiment analysis
    - A3: Feature-specific feedback
    - A4: Common complaints/praises
    - A5: Review summarization
    """

    name = "analysis"
    description = "Review mining and sentiment analysis"

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

        # Initialize search toolkit for section search
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
        input_data: AnalysisInput,
    ) -> AnalysisOutput:
        """Execute review analysis."""
        # Get products if not provided
        products = input_data.products
        if not products and input_data.asin:
            # Fetch specific product
            products = await self._fetch_product(input_data.asin)
        elif not products:
            # Search for products using reviews section
            search_results = await self._search_toolkit.section_search(
                input_data.query,
                "reviews",
                limit=5,
            )
            products = [r.model_dump() for r in search_results.results]

        if not products:
            return AnalysisOutput(
                query=input_data.query,
                analysis=ReviewAnalysis(summary="No products found to analyze."),
            )

        # Build context for analysis
        context = self._build_analysis_context(products, input_data.analysis_type)

        # Generate analysis using LLM
        prompt = self._build_analysis_prompt(
            input_data.query,
            context,
            input_data.analysis_type,
        )

        response = await self._http_client.post(
            "/generate",
            json={
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 800,
            },
        )
        response.raise_for_status()
        raw_response = response.json().get("response", "")

        # Parse analysis from response
        analysis = self._parse_analysis(raw_response, input_data.analysis_type)

        return AnalysisOutput(
            query=input_data.query,
            analysis=analysis,
            products_analyzed=len(products),
            source_products=products[:5],
            raw_response=raw_response,
        )

    async def _fetch_product(self, asin: str) -> list[dict[str, Any]]:
        """Fetch product by ASIN using search."""
        results = await self._search_toolkit.keyword_search(asin, limit=1)
        return [r.model_dump() for r in results.results]

    def _build_analysis_context(
        self,
        products: list[dict[str, Any]],
        analysis_type: str,
    ) -> str:
        """Build context string from products."""
        context_parts = []

        for i, product in enumerate(products[:5]):
            parts = [f"Product {i+1}: {product.get('title', 'Unknown')[:60]}"]

            if product.get("stars"):
                parts.append(f"Rating: {product['stars']}/5")

            if product.get("genAI_common_praises"):
                praises = product["genAI_common_praises"]
                if isinstance(praises, list):
                    parts.append(f"Praises: {', '.join(praises[:3])}")

            if product.get("genAI_common_complaints"):
                complaints = product["genAI_common_complaints"]
                if isinstance(complaints, list):
                    parts.append(f"Complaints: {', '.join(complaints[:3])}")

            if product.get("genAI_pros"):
                pros = product["genAI_pros"]
                if isinstance(pros, list):
                    parts.append(f"Pros: {', '.join(pros[:3])}")

            if product.get("genAI_cons"):
                cons = product["genAI_cons"]
                if isinstance(cons, list):
                    parts.append(f"Cons: {', '.join(cons[:3])}")

            if product.get("content_preview"):
                parts.append(f"Review excerpt: {product['content_preview'][:150]}")

            context_parts.append("\n".join(parts))

        return "\n\n".join(context_parts)

    def _build_analysis_prompt(
        self,
        query: str,
        context: str,
        analysis_type: str,
    ) -> str:
        """Build LLM prompt for analysis."""
        base_prompt = f"""Analyze the following product reviews/data to answer the user's question.

User Query: {query}

Product Data:
{context}

"""

        if analysis_type == "pros_cons":
            return base_prompt + """
Provide a structured analysis with:

PROS:
- List the main advantages mentioned

CONS:
- List the main disadvantages mentioned

VERDICT:
Brief overall assessment"""

        elif analysis_type == "sentiment":
            return base_prompt + """
Analyze the sentiment:

SENTIMENT SCORE: (from -1.0 very negative to 1.0 very positive)
SENTIMENT: (Positive/Negative/Mixed/Neutral)

KEY POSITIVE THEMES:
- List positive aspects

KEY NEGATIVE THEMES:
- List concerns

SUMMARY:
Brief sentiment summary"""

        elif analysis_type == "feature":
            return base_prompt + """
Analyze feedback on specific features:

FEATURE ANALYSIS:
For each notable feature, provide:
- Feature name
- User sentiment (positive/negative/mixed)
- Common feedback

BEST FEATURES:
- List top-rated features

PROBLEMATIC FEATURES:
- List features with issues

SUMMARY:
Brief feature overview"""

        else:
            return base_prompt + """
Provide a comprehensive review analysis:

OVERALL SENTIMENT: (Positive/Negative/Mixed)

MAIN PROS:
- Key advantages

MAIN CONS:
- Key disadvantages

COMMON THEMES:
- Frequently mentioned topics

RECOMMENDATION:
Who should buy this and who should consider alternatives"""

    def _parse_analysis(
        self,
        response: str,
        analysis_type: str,
    ) -> ReviewAnalysis:
        """Parse LLM response into structured analysis."""
        analysis = ReviewAnalysis(summary=response)

        # Extract pros
        import re
        pros_match = re.search(r'(?:PROS|ADVANTAGES|POSITIVE)[:\s]*((?:[-•]\s*.+\n?)+)', response, re.IGNORECASE)
        if pros_match:
            pros_text = pros_match.group(1)
            analysis.pros = [
                line.strip().lstrip('-•').strip()
                for line in pros_text.split('\n')
                if line.strip().lstrip('-•').strip()
            ]

        # Extract cons
        cons_match = re.search(r'(?:CONS|DISADVANTAGES|NEGATIVE)[:\s]*((?:[-•]\s*.+\n?)+)', response, re.IGNORECASE)
        if cons_match:
            cons_text = cons_match.group(1)
            analysis.cons = [
                line.strip().lstrip('-•').strip()
                for line in cons_text.split('\n')
                if line.strip().lstrip('-•').strip()
            ]

        # Extract sentiment
        sentiment_match = re.search(r'SENTIMENT[:\s]*([+-]?\d*\.?\d+|Positive|Negative|Mixed|Neutral)', response, re.IGNORECASE)
        if sentiment_match:
            sentiment_val = sentiment_match.group(1)
            try:
                analysis.sentiment_score = float(sentiment_val)
            except ValueError:
                if sentiment_val.lower() == "positive":
                    analysis.sentiment_score = 0.7
                    analysis.sentiment_label = "positive"
                elif sentiment_val.lower() == "negative":
                    analysis.sentiment_score = -0.7
                    analysis.sentiment_label = "negative"
                elif sentiment_val.lower() == "mixed":
                    analysis.sentiment_score = 0.0
                    analysis.sentiment_label = "mixed"
                else:
                    analysis.sentiment_label = sentiment_val.lower()

        return analysis


# Singleton
_analysis_agent: AnalysisAgent | None = None


async def get_analysis_agent() -> AnalysisAgent:
    """Get or create analysis agent singleton."""
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = AnalysisAgent()
        await _analysis_agent.initialize()
    return _analysis_agent


async def analyze_reviews(
    query: str,
    products: list[dict[str, Any]] | None = None,
    analysis_type: str = "general",
) -> AnalysisOutput:
    """Convenience function to analyze reviews.

    Args:
        query: Analysis query
        products: Optional products to analyze
        analysis_type: Type of analysis

    Returns:
        AnalysisOutput with results
    """
    agent = await get_analysis_agent()
    return await agent.execute(AnalysisInput(
        query=query,
        products=products or [],
        analysis_type=analysis_type,
    ))
