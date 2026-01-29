"""Comparison sub-workflow using LangGraph.

Handles product comparison with:
- Product fetching and normalization
- Feature-by-feature comparison
- Winner determination
- Summary generation
"""

from typing import Any, Literal

import structlog

from src.dag.state import ComparisonState, ComparedProduct, ComparisonMode

logger = structlog.get_logger()


# =============================================================================
# Workflow Nodes
# =============================================================================


async def fetch_products(state: ComparisonState, toolkit: Any) -> ComparisonState:
    """Fetch products for comparison.

    Fetches by ASIN if provided, otherwise searches by name.
    """
    import time
    start = time.time()

    products = []

    # Fetch by ASIN
    for asin in state.product_asins:
        try:
            product_data = await toolkit.get_product_by_asin(asin)
            if product_data:
                products.append(ComparedProduct(
                    asin=product_data.get("asin"),
                    title=product_data.get("title", ""),
                    brand=product_data.get("brand"),
                    price=product_data.get("price"),
                    list_price=product_data.get("list_price"),
                    stars=product_data.get("stars"),
                    reviews_count=product_data.get("reviews_count"),
                    category=product_data.get("category"),
                    img_url=product_data.get("img_url"),
                ))
        except Exception as e:
            logger.warning("product_fetch_failed", asin=asin, error=str(e))

    # Search by name
    for name in state.product_names:
        if len(products) >= 5:  # Limit to 5 products
            break
        try:
            results = await toolkit.hybrid_search(name, limit=1)
            if results:
                p = results[0]
                if p.get("asin") not in [prod.asin for prod in products]:
                    products.append(ComparedProduct(
                        asin=p.get("asin"),
                        title=p.get("title", ""),
                        brand=p.get("brand"),
                        price=p.get("price"),
                        list_price=p.get("list_price"),
                        stars=p.get("stars"),
                        reviews_count=p.get("reviews_count"),
                        category=p.get("category"),
                        img_url=p.get("img_url"),
                    ))
        except Exception as e:
            logger.warning("product_search_failed", name=name, error=str(e))

    state.products = products
    state.latency_ms += (time.time() - start) * 1000

    return state


async def extract_features(state: ComparisonState, toolkit: Any) -> ComparisonState:
    """Extract key features from each product for comparison."""
    import time
    start = time.time()

    for product in state.products:
        try:
            # Get product details with features
            details = await toolkit.get_product_details(product.asin)
            if details:
                product.key_features = details.get("features", {})

            # Get pros/cons from reviews
            reviews = await toolkit.get_product_reviews(product.asin, limit=10)
            if reviews:
                product.pros = _extract_pros(reviews)
                product.cons = _extract_cons(reviews)

        except Exception as e:
            logger.warning("feature_extraction_failed", asin=product.asin, error=str(e))

    state.latency_ms += (time.time() - start) * 1000
    return state


async def compare_products(state: ComparisonState, llm_client: Any | None = None) -> ComparisonState:
    """Compare products and build comparison matrix."""
    import time
    start = time.time()

    if len(state.products) < 2:
        state.error = "Need at least 2 products for comparison"
        return state

    # Build comparison matrix
    comparison_matrix = {}
    comparison_fields = ["price", "stars", "reviews_count", "brand"]

    for field in comparison_fields:
        comparison_matrix[field] = {}
        for product in state.products:
            value = getattr(product, field, None)
            comparison_matrix[field][product.asin] = value

    # Add feature comparisons
    all_features = set()
    for product in state.products:
        all_features.update(product.key_features.keys())

    for feature in list(all_features)[:10]:  # Limit to 10 features
        comparison_matrix[feature] = {}
        for product in state.products:
            comparison_matrix[feature][product.asin] = product.key_features.get(feature)

    state.comparison_matrix = comparison_matrix

    # Find key differences
    state.key_differences = _find_key_differences(state.products, comparison_matrix)

    state.latency_ms += (time.time() - start) * 1000
    return state


async def determine_winner(state: ComparisonState) -> ComparisonState:
    """Determine winner and best value."""
    import time
    start = time.time()

    if not state.products:
        return state

    # Calculate value scores
    for product in state.products:
        score = 0.0
        if product.stars:
            score += product.stars * 0.3
        if product.reviews_count:
            score += min(product.reviews_count / 1000, 5) * 0.2
        if product.price and product.stars:
            value_ratio = product.stars / (product.price / 100 + 1)
            score += value_ratio * 0.5
        product.value_score = score

    # Winner by rating
    sorted_by_rating = sorted(
        state.products,
        key=lambda p: (p.stars or 0, p.reviews_count or 0),
        reverse=True,
    )
    if sorted_by_rating:
        state.winner = sorted_by_rating[0]
        state.winner_reason = f"Highest rated ({state.winner.stars} stars with {state.winner.reviews_count} reviews)"

    # Best value
    sorted_by_value = sorted(
        state.products,
        key=lambda p: p.value_score or 0,
        reverse=True,
    )
    if sorted_by_value:
        state.best_value = sorted_by_value[0]

    state.latency_ms += (time.time() - start) * 1000
    return state


async def generate_summary(state: ComparisonState, llm_client: Any | None = None) -> ComparisonState:
    """Generate comparison summary.

    Uses LLM if available, otherwise generates rule-based summary.
    """
    import time
    start = time.time()

    if not state.products:
        state.comparison_summary = "No products to compare."
        return state

    product_names = [p.title[:50] for p in state.products]

    # Rule-based summary
    summary_parts = []
    summary_parts.append(f"Comparing {len(state.products)} products: {', '.join(product_names)}.")

    if state.winner:
        summary_parts.append(f"**Winner**: {state.winner.title[:50]} - {state.winner_reason}.")

    if state.best_value and state.best_value != state.winner:
        summary_parts.append(f"**Best Value**: {state.best_value.title[:50]} at ${state.best_value.price:.2f}.")

    if state.key_differences:
        summary_parts.append(f"**Key Differences**: {'; '.join(state.key_differences[:3])}.")

    state.comparison_summary = " ".join(summary_parts)

    state.latency_ms += (time.time() - start) * 1000
    return state


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_pros(reviews: list[dict]) -> list[str]:
    """Extract common pros from reviews."""
    pros = []
    positive_keywords = ["great", "excellent", "love", "best", "amazing", "perfect"]

    for review in reviews[:5]:
        text = review.get("text", "").lower()
        for keyword in positive_keywords:
            if keyword in text:
                # Extract sentence containing keyword
                sentences = text.split(".")
                for sentence in sentences:
                    if keyword in sentence and len(sentence) < 100:
                        pros.append(sentence.strip().capitalize())
                        break
                break

    return list(set(pros))[:5]


def _extract_cons(reviews: list[dict]) -> list[str]:
    """Extract common cons from reviews."""
    cons = []
    negative_keywords = ["problem", "issue", "bad", "poor", "disappointing", "broken"]

    for review in reviews[:5]:
        text = review.get("text", "").lower()
        for keyword in negative_keywords:
            if keyword in text:
                sentences = text.split(".")
                for sentence in sentences:
                    if keyword in sentence and len(sentence) < 100:
                        cons.append(sentence.strip().capitalize())
                        break
                break

    return list(set(cons))[:5]


def _find_key_differences(products: list[ComparedProduct], matrix: dict) -> list[str]:
    """Find key differences between products."""
    differences = []

    # Price difference
    prices = [p.price for p in products if p.price]
    if prices and len(prices) > 1:
        price_diff = max(prices) - min(prices)
        if price_diff > 10:
            differences.append(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")

    # Rating difference
    ratings = [p.stars for p in products if p.stars]
    if ratings and len(ratings) > 1:
        rating_diff = max(ratings) - min(ratings)
        if rating_diff > 0.5:
            differences.append(f"Rating range: {min(ratings):.1f} - {max(ratings):.1f} stars")

    # Brand comparison
    brands = list(set(p.brand for p in products if p.brand))
    if len(brands) > 1:
        differences.append(f"Brands: {', '.join(brands)}")

    return differences


# =============================================================================
# Workflow Router
# =============================================================================


def route_comparison(state: ComparisonState) -> Literal["extract_features", "compare", "error"]:
    """Route to next step based on state."""
    if state.error:
        return "error"
    if not state.products:
        return "error"
    if len(state.products) < 2:
        return "error"
    return "extract_features"


# =============================================================================
# Workflow Factory
# =============================================================================


def create_comparison_workflow(toolkit: Any, llm_client: Any | None = None):
    """Create comparison workflow.

    Args:
        toolkit: Search toolkit for product data
        llm_client: Optional LLM client for enhanced summaries

    Returns:
        Async function that runs the comparison workflow
    """

    async def run_comparison(query: str, **kwargs) -> ComparisonState:
        """Run the comparison workflow."""
        state = ComparisonState(
            query=query,
            product_names=kwargs.get("product_names", []),
            product_asins=kwargs.get("product_asins", []),
            mode=ComparisonMode(kwargs.get("mode", "direct")),
        )

        # If no specific products, search for them
        if not state.product_names and not state.product_asins:
            # Extract product names from query
            state.product_names = _extract_product_names(query)

        # Execute workflow steps
        state = await fetch_products(state, toolkit)

        if len(state.products) >= 2:
            state = await extract_features(state, toolkit)
            state = await compare_products(state, llm_client)
            state = await determine_winner(state)
            state = await generate_summary(state, llm_client)
        else:
            state.error = "Could not find enough products for comparison"

        return state

    return run_comparison


def _extract_product_names(query: str) -> list[str]:
    """Extract product names from comparison query."""
    # Simple extraction: split by common comparison phrases
    query_lower = query.lower()

    # Common patterns
    for pattern in [" vs ", " versus ", " or ", " and ", " compared to "]:
        if pattern in query_lower:
            parts = query_lower.split(pattern)
            return [p.strip() for p in parts if p.strip()]

    return []
