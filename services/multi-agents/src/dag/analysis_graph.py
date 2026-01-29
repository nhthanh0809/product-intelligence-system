"""Analysis sub-workflow using LangGraph.

Handles product/review analysis with:
- Review fetching and processing
- Sentiment analysis
- Pros/cons extraction
- Theme identification
- Summary generation
"""

from typing import Any, Literal

import structlog

from src.dag.state import AnalysisState, AnalyzedProduct, AnalysisType

logger = structlog.get_logger()


# =============================================================================
# Workflow Nodes
# =============================================================================


async def fetch_product(state: AnalysisState, toolkit: Any) -> AnalysisState:
    """Fetch product for analysis."""
    import time
    start = time.time()

    product_data = None

    # Fetch by ASIN
    if state.product_asin:
        try:
            product_data = await toolkit.get_product_by_asin(state.product_asin)
        except Exception as e:
            logger.warning("product_fetch_failed", asin=state.product_asin, error=str(e))

    # Search by name
    if not product_data and state.product_name:
        try:
            results = await toolkit.hybrid_search(state.product_name, limit=1)
            if results:
                product_data = results[0]
                state.product_asin = product_data.get("asin")
        except Exception as e:
            logger.warning("product_search_failed", name=state.product_name, error=str(e))

    # Search from query
    if not product_data:
        try:
            results = await toolkit.hybrid_search(state.query, limit=1)
            if results:
                product_data = results[0]
                state.product_asin = product_data.get("asin")
        except Exception as e:
            logger.warning("product_search_failed", query=state.query, error=str(e))

    if product_data:
        state.product = AnalyzedProduct(
            asin=product_data.get("asin"),
            title=product_data.get("title", ""),
            brand=product_data.get("brand"),
            price=product_data.get("price"),
            stars=product_data.get("stars"),
            reviews_count=product_data.get("reviews_count"),
            category=product_data.get("category"),
            img_url=product_data.get("img_url"),
        )

    state.latency_ms += (time.time() - start) * 1000
    return state


async def fetch_reviews(state: AnalysisState, toolkit: Any) -> AnalysisState:
    """Fetch reviews for the product."""
    import time
    start = time.time()

    if not state.product_asin:
        return state

    try:
        # Try to get reviews from review section search
        reviews = await toolkit.section_search(
            query=state.product.title if state.product else state.query,
            section="reviews",
            limit=20,
        )

        if reviews:
            state.reviews = reviews
            state.review_count = len(reviews)

    except Exception as e:
        logger.warning("review_fetch_failed", asin=state.product_asin, error=str(e))

    state.latency_ms += (time.time() - start) * 1000
    return state


async def analyze_sentiment(state: AnalysisState) -> AnalysisState:
    """Analyze sentiment from reviews."""
    import time
    start = time.time()

    if not state.reviews:
        return state

    # Simple rule-based sentiment analysis
    positive_words = {
        "great", "excellent", "amazing", "love", "best", "perfect", "fantastic",
        "wonderful", "awesome", "good", "nice", "happy", "satisfied", "recommend",
    }
    negative_words = {
        "bad", "poor", "terrible", "worst", "hate", "disappointed", "broken",
        "defective", "useless", "waste", "awful", "horrible", "problem", "issue",
    }

    positive_count = 0
    negative_count = 0
    total_words = 0

    for review in state.reviews:
        text = review.get("text", "") or review.get("content", "")
        if not text:
            continue

        words = text.lower().split()
        total_words += len(words)

        for word in words:
            word = word.strip(".,!?\"'")
            if word in positive_words:
                positive_count += 1
            elif word in negative_words:
                negative_count += 1

    # Calculate sentiment score (-1 to 1)
    if positive_count + negative_count > 0:
        state.sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
    else:
        state.sentiment_score = 0.0

    # Determine sentiment label
    if state.sentiment_score > 0.3:
        state.sentiment_label = "positive"
    elif state.sentiment_score < -0.3:
        state.sentiment_label = "negative"
    else:
        state.sentiment_label = "neutral"

    # Update product if exists
    if state.product:
        state.product.sentiment_score = state.sentiment_score
        state.product.sentiment_label = state.sentiment_label

    state.latency_ms += (time.time() - start) * 1000
    return state


async def extract_pros_cons(state: AnalysisState) -> AnalysisState:
    """Extract pros and cons from reviews."""
    import time
    start = time.time()

    if not state.reviews:
        return state

    pros = []
    cons = []

    # Positive patterns
    positive_patterns = [
        "love", "great", "excellent", "best", "amazing", "perfect",
        "easy to", "works well", "highly recommend", "good quality",
    ]

    # Negative patterns
    negative_patterns = [
        "problem", "issue", "broken", "doesn't work", "poor quality",
        "disappointed", "waste", "returned", "defective", "stopped working",
    ]

    for review in state.reviews[:10]:  # Limit to 10 reviews
        text = review.get("text", "") or review.get("content", "")
        if not text:
            continue

        text_lower = text.lower()
        sentences = text.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) < 10 or len(sentence_lower) > 150:
                continue

            # Check for positive patterns
            for pattern in positive_patterns:
                if pattern in sentence_lower:
                    pros.append(sentence.strip().capitalize())
                    break

            # Check for negative patterns
            for pattern in negative_patterns:
                if pattern in sentence_lower:
                    cons.append(sentence.strip().capitalize())
                    break

    # Deduplicate and limit
    state.pros = list(set(pros))[:5]
    state.cons = list(set(cons))[:5]

    state.latency_ms += (time.time() - start) * 1000
    return state


async def identify_themes(state: AnalysisState) -> AnalysisState:
    """Identify common themes from reviews."""
    import time
    from collections import Counter
    start = time.time()

    if not state.reviews:
        return state

    # Feature keywords to look for
    feature_keywords = {
        "battery": "Battery Life",
        "screen": "Display",
        "display": "Display",
        "camera": "Camera",
        "sound": "Sound Quality",
        "audio": "Sound Quality",
        "build": "Build Quality",
        "quality": "Quality",
        "price": "Value",
        "value": "Value",
        "shipping": "Shipping",
        "delivery": "Shipping",
        "customer service": "Customer Service",
        "support": "Customer Service",
        "setup": "Setup/Installation",
        "install": "Setup/Installation",
        "performance": "Performance",
        "speed": "Performance",
        "comfort": "Comfort",
        "fit": "Fit",
        "size": "Size",
        "durability": "Durability",
        "design": "Design",
    }

    theme_counter = Counter()

    for review in state.reviews:
        text = review.get("text", "") or review.get("content", "")
        if not text:
            continue

        text_lower = text.lower()

        for keyword, theme in feature_keywords.items():
            if keyword in text_lower:
                theme_counter[theme] += 1

    # Get top themes
    state.common_themes = [theme for theme, _ in theme_counter.most_common(5)]

    if state.product:
        state.product.common_themes = state.common_themes

    state.latency_ms += (time.time() - start) * 1000
    return state


async def analyze_features(state: AnalysisState) -> AnalysisState:
    """Analyze specific features mentioned in reviews."""
    import time
    from collections import defaultdict
    start = time.time()

    if not state.reviews or state.analysis_type != AnalysisType.FEATURES:
        return state

    # Track feature mentions and sentiment
    feature_sentiments = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})

    feature_keywords = [
        "battery", "screen", "camera", "sound", "build",
        "price", "performance", "comfort", "design", "durability",
    ]

    positive_words = {"great", "excellent", "good", "love", "amazing", "best"}
    negative_words = {"bad", "poor", "terrible", "worst", "broken"}

    for review in state.reviews:
        text = review.get("text", "") or review.get("content", "")
        if not text:
            continue

        text_lower = text.lower()
        sentences = text.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower()

            for feature in feature_keywords:
                if feature in sentence_lower:
                    # Determine sentiment for this feature mention
                    words = set(sentence_lower.split())
                    if words & positive_words:
                        feature_sentiments[feature]["positive"] += 1
                    elif words & negative_words:
                        feature_sentiments[feature]["negative"] += 1
                    else:
                        feature_sentiments[feature]["neutral"] += 1

    # Calculate feature ratings (0-5 scale)
    feature_ratings = {}
    for feature, counts in feature_sentiments.items():
        total = counts["positive"] + counts["negative"] + counts["neutral"]
        if total > 0:
            # Convert to 0-5 scale
            score = (counts["positive"] * 5 + counts["neutral"] * 3 + counts["negative"] * 1) / total
            feature_ratings[feature] = round(score, 1)

    state.feature_analysis = dict(feature_sentiments)

    if state.product:
        state.product.feature_ratings = feature_ratings

    state.latency_ms += (time.time() - start) * 1000
    return state


async def generate_summary(state: AnalysisState, llm_client: Any | None = None) -> AnalysisState:
    """Generate analysis summary."""
    import time
    start = time.time()

    summary_parts = []

    # Product info
    if state.product:
        summary_parts.append(
            f"**{state.product.title[:80]}** ({state.product.stars or 'N/A'} stars)"
        )

    # Sentiment
    if state.sentiment_label:
        summary_parts.append(
            f"Overall sentiment is **{state.sentiment_label}** "
            f"(score: {state.sentiment_score:.2f})"
        )

    # Review count
    if state.review_count:
        summary_parts.append(f"Analyzed {state.review_count} reviews.")

    # Pros
    if state.pros:
        summary_parts.append(f"**Pros**: {'; '.join(state.pros[:3])}")

    # Cons
    if state.cons:
        summary_parts.append(f"**Cons**: {'; '.join(state.cons[:3])}")

    # Themes
    if state.common_themes:
        summary_parts.append(f"**Common themes**: {', '.join(state.common_themes)}")

    # Recommendations
    state.recommendations = _generate_recommendations(state)
    if state.recommendations:
        summary_parts.append(f"**Recommendation**: {state.recommendations[0]}")

    state.summary = " ".join(summary_parts)

    state.latency_ms += (time.time() - start) * 1000
    return state


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_recommendations(state: AnalysisState) -> list[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    # Based on sentiment
    if state.sentiment_label == "positive":
        recommendations.append("This product is well-received by customers and recommended for purchase.")
    elif state.sentiment_label == "negative":
        recommendations.append("Consider alternatives due to mixed customer feedback.")
    else:
        recommendations.append("Customer feedback is mixed - review specific concerns before purchasing.")

    # Based on pros/cons ratio
    if len(state.pros) > len(state.cons) * 2:
        recommendations.append("Strong positive feedback with few complaints.")
    elif len(state.cons) > len(state.pros):
        recommendations.append("Note the concerns raised by customers before purchasing.")

    return recommendations


# =============================================================================
# Workflow Router
# =============================================================================


def route_analysis(state: AnalysisState) -> Literal["fetch_reviews", "analyze", "error"]:
    """Route to next step based on state."""
    if state.error:
        return "error"
    if not state.product:
        return "error"
    return "fetch_reviews"


# =============================================================================
# Workflow Factory
# =============================================================================


def create_analysis_workflow(toolkit: Any, llm_client: Any | None = None):
    """Create analysis workflow.

    Args:
        toolkit: Search toolkit for product data
        llm_client: Optional LLM client for enhanced analysis

    Returns:
        Async function that runs the analysis workflow
    """

    async def run_analysis(query: str, **kwargs) -> AnalysisState:
        """Run the analysis workflow."""
        state = AnalysisState(
            query=query,
            product_asin=kwargs.get("product_asin"),
            product_name=kwargs.get("product_name"),
            analysis_type=AnalysisType(kwargs.get("analysis_type", "general")),
        )

        # Execute workflow steps
        state = await fetch_product(state, toolkit)

        if state.product:
            state = await fetch_reviews(state, toolkit)
            state = await analyze_sentiment(state)
            state = await extract_pros_cons(state)
            state = await identify_themes(state)

            if state.analysis_type == AnalysisType.FEATURES:
                state = await analyze_features(state)

            state = await generate_summary(state, llm_client)
        else:
            state.error = "Could not find product to analyze"

        return state

    return run_analysis
