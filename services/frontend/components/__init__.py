"""UI components for the frontend."""

from components.chat import render_chat_interface
from components.product_card import render_product_card, render_product_grid
from components.comparison_table import render_comparison_table
from components.charts import render_trend_chart, render_price_chart, render_review_sentiment_chart
from components.input import (
    render_query_input,
    render_search_input_with_filters,
    render_comparison_input,
    render_quick_actions,
    QUERY_SUGGESTIONS,
)
from components.review_analysis import (
    render_review_analysis,
    render_pros,
    render_cons,
    render_theme_tags,
    render_sentiment_gauge,
    render_review_highlights,
)
from components.recommendation_carousel import (
    render_recommendation_carousel,
    render_recommendation_card,
    render_horizontal_scroll_carousel,
    render_recommendation_summary,
    render_quick_recommendation_list,
)

__all__ = [
    # Chat
    "render_chat_interface",
    # Products
    "render_product_card",
    "render_product_grid",
    # Comparison
    "render_comparison_table",
    # Charts
    "render_trend_chart",
    "render_price_chart",
    "render_review_sentiment_chart",
    # Input
    "render_query_input",
    "render_search_input_with_filters",
    "render_comparison_input",
    "render_quick_actions",
    "QUERY_SUGGESTIONS",
    # Review Analysis
    "render_review_analysis",
    "render_pros",
    "render_cons",
    "render_theme_tags",
    "render_sentiment_gauge",
    "render_review_highlights",
    # Recommendations
    "render_recommendation_carousel",
    "render_recommendation_card",
    "render_horizontal_scroll_carousel",
    "render_recommendation_summary",
    "render_quick_recommendation_list",
]
