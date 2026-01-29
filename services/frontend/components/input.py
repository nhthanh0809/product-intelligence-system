"""Query input component with suggestions and autocomplete."""

import streamlit as st
from typing import Callable


# Common query suggestions by category
QUERY_SUGGESTIONS = {
    "discovery": [
        "best wireless headphones under $200",
        "laptop for programming",
        "gaming monitor 4K",
        "smart home devices",
        "noise cancelling earbuds",
    ],
    "comparison": [
        "Sony WH-1000XM5 vs Bose QC45",
        "MacBook Pro vs Dell XPS",
        "iPhone 15 vs Samsung S24",
        "AirPods Pro vs Galaxy Buds",
    ],
    "price": [
        "best deals on headphones",
        "budget laptops under $500",
        "premium monitors worth the price",
        "value smartphones",
    ],
    "analysis": [
        "what do people say about battery life",
        "common complaints about this product",
        "build quality reviews",
        "durability issues",
    ],
    "trends": [
        "trending gaming accessories",
        "popular smart home devices",
        "hot products this week",
        "rising brands in audio",
    ],
    "recommendations": [
        "similar to AirPods Pro",
        "alternatives to Sony headphones",
        "accessories for MacBook",
        "upgrade from budget laptop",
    ],
}


def render_query_input(
    label: str = "Search",
    placeholder: str = "Enter your query...",
    category: str = "discovery",
    key: str = "query_input",
    on_submit: Callable[[str], None] | None = None,
) -> str | None:
    """Render query input with suggestions.

    Args:
        label: Input label
        placeholder: Placeholder text
        category: Suggestion category
        key: Unique key for the input
        on_submit: Optional callback on submit

    Returns:
        Query string or None
    """
    # Initialize session state for suggestions
    if f"{key}_show_suggestions" not in st.session_state:
        st.session_state[f"{key}_show_suggestions"] = False

    col1, col2 = st.columns([5, 1])

    with col1:
        query = st.text_input(
            label,
            placeholder=placeholder,
            key=key,
            label_visibility="collapsed" if label == "Search" else "visible",
        )

    with col2:
        if st.button("💡", key=f"{key}_suggest_btn", help="Show suggestions"):
            st.session_state[f"{key}_show_suggestions"] = not st.session_state[f"{key}_show_suggestions"]

    # Show suggestions
    if st.session_state[f"{key}_show_suggestions"]:
        suggestions = QUERY_SUGGESTIONS.get(category, QUERY_SUGGESTIONS["discovery"])
        st.caption("Try one of these:")

        # Render suggestions as clickable chips
        suggestion_cols = st.columns(min(len(suggestions), 3))
        for idx, suggestion in enumerate(suggestions[:6]):
            col_idx = idx % 3
            with suggestion_cols[col_idx]:
                if st.button(
                    suggestion[:30] + ("..." if len(suggestion) > 30 else ""),
                    key=f"{key}_suggestion_{idx}",
                    use_container_width=True,
                ):
                    st.session_state[key] = suggestion
                    st.session_state[f"{key}_show_suggestions"] = False
                    if on_submit:
                        on_submit(suggestion)
                    st.rerun()

    return query if query else None


def render_search_input_with_filters(
    key: str = "search",
    show_filters: bool = True,
) -> dict:
    """Render search input with expandable filters.

    Args:
        key: Unique key prefix
        show_filters: Whether to show filter expander

    Returns:
        Dictionary with query and filter values
    """
    # Main search input
    query = render_query_input(
        label="Search products",
        placeholder="e.g., wireless headphones under $100",
        category="discovery",
        key=f"{key}_query",
    )

    filters = {}

    if show_filters:
        with st.expander("🔧 Filters", expanded=False):
            filter_cols = st.columns(4)

            with filter_cols[0]:
                category = st.text_input(
                    "Category",
                    key=f"{key}_category",
                    placeholder="Electronics",
                )
                if category:
                    filters["category"] = category

            with filter_cols[1]:
                brand = st.text_input(
                    "Brand",
                    key=f"{key}_brand",
                    placeholder="Sony",
                )
                if brand:
                    filters["brand"] = brand

            with filter_cols[2]:
                price_min = st.number_input(
                    "Min Price",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                    key=f"{key}_price_min",
                )
                if price_min > 0:
                    filters["price_min"] = price_min

            with filter_cols[3]:
                price_max = st.number_input(
                    "Max Price",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                    key=f"{key}_price_max",
                )
                if price_max > 0:
                    filters["price_max"] = price_max

            # Additional filters row
            filter_cols2 = st.columns(4)

            with filter_cols2[0]:
                min_rating = st.slider(
                    "Min Rating",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.5,
                    key=f"{key}_min_rating",
                )
                if min_rating > 0:
                    filters["min_rating"] = min_rating

            with filter_cols2[1]:
                sort_by = st.selectbox(
                    "Sort By",
                    ["relevance", "price_low", "price_high", "rating", "reviews"],
                    key=f"{key}_sort_by",
                )
                if sort_by != "relevance":
                    filters["sort_by"] = sort_by

    return {
        "query": query,
        "filters": filters,
    }


def render_comparison_input(key: str = "compare") -> dict:
    """Render comparison input with product name fields.

    Args:
        key: Unique key prefix

    Returns:
        Dictionary with query and product names
    """
    query = render_query_input(
        label="Compare products",
        placeholder="e.g., Sony WH-1000XM5 vs Bose QC45",
        category="comparison",
        key=f"{key}_query",
    )

    st.caption("Or enter specific products:")

    product_names = []
    cols = st.columns(2)

    for i in range(4):
        col_idx = i % 2
        with cols[col_idx]:
            name = st.text_input(
                f"Product {i + 1}",
                key=f"{key}_product_{i}",
                placeholder=f"Enter product name",
                label_visibility="collapsed",
            )
            if name:
                product_names.append(name)

    return {
        "query": query,
        "product_names": product_names,
    }


def render_quick_actions(
    actions: list[dict],
    key: str = "quick_actions",
) -> str | None:
    """Render quick action buttons.

    Args:
        actions: List of {"label": str, "query": str, "icon": str}
        key: Unique key prefix

    Returns:
        Selected query or None
    """
    st.caption("Quick Actions:")

    cols = st.columns(len(actions))

    for idx, action in enumerate(actions):
        with cols[idx]:
            if st.button(
                f"{action.get('icon', '🔍')} {action['label']}",
                key=f"{key}_{idx}",
                use_container_width=True,
            ):
                return action["query"]

    return None
