"""Recommendation carousel component for similar products."""

import streamlit as st
from typing import Any


def render_recommendation_carousel(
    recommendations: list[dict[str, Any]],
    title: str = "Recommended Products",
    show_reason: bool = True,
    items_per_row: int = 3,
):
    """Render recommendations as a carousel/grid.

    Args:
        recommendations: List of recommendation dictionaries
        title: Section title
        show_reason: Whether to show recommendation reasons
        items_per_row: Number of items per row
    """
    if not recommendations:
        st.info("No recommendations available")
        return

    st.subheader(f"💡 {title}")

    # Create rows of recommendations
    for row_start in range(0, len(recommendations), items_per_row):
        row_items = recommendations[row_start:row_start + items_per_row]
        cols = st.columns(items_per_row)

        for idx, rec in enumerate(row_items):
            with cols[idx]:
                render_recommendation_card(rec, show_reason=show_reason)


def render_recommendation_card(rec: dict[str, Any], show_reason: bool = True):
    """Render a single recommendation card.

    Args:
        rec: Recommendation dictionary
        show_reason: Whether to show recommendation reason
    """
    with st.container():
        # Card container with styling
        st.markdown(
            """
            <style>
            .rec-card {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 0.75rem;
                padding: 1rem;
                margin: 0.5rem 0;
                transition: box-shadow 0.2s;
            }
            .rec-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Product image placeholder
        img_url = rec.get("img_url")
        if img_url:
            st.image(img_url, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="
                    background: #f0f0f0;
                    height: 120px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 0.5rem;
                    color: #999;
                ">
                    📦
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Title
        title = rec.get("title", "Unknown Product")
        st.markdown(f"**{title[:50]}{'...' if len(title) > 50 else ''}**")

        # Brand
        brand = rec.get("brand")
        if brand:
            st.caption(brand)

        # Price and rating
        col1, col2 = st.columns(2)

        with col1:
            price = rec.get("price")
            if price:
                st.markdown(f"**${price:.2f}**")

        with col2:
            stars = rec.get("stars")
            if stars:
                st.markdown(f"⭐ {stars:.1f}")

        # Similarity score
        similarity = rec.get("similarity_score")
        if similarity:
            score_pct = int(similarity * 100)
            st.progress(similarity, text=f"{score_pct}% match")

        # Match type badge
        match_type = rec.get("match_type")
        if match_type:
            badge_color = {
                "similar": "#3498db",
                "alternative": "#9b59b6",
                "accessory": "#2ecc71",
                "upgrade": "#f39c12",
                "budget": "#1abc9c",
            }.get(match_type, "#95a5a6")

            st.markdown(
                f"""
                <span style="
                    background: {badge_color};
                    color: white;
                    padding: 0.2rem 0.5rem;
                    border-radius: 0.25rem;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                ">
                    {match_type}
                </span>
                """,
                unsafe_allow_html=True,
            )

        # Recommendation reason
        if show_reason:
            reason = rec.get("reason") or rec.get("recommendation_reason")
            if reason:
                st.caption(f"💡 {reason}")


def render_horizontal_scroll_carousel(
    recommendations: list[dict[str, Any]],
    title: str = "You Might Also Like",
):
    """Render recommendations as a horizontally scrolling carousel.

    Args:
        recommendations: List of recommendation dictionaries
        title: Section title
    """
    if not recommendations:
        return

    st.subheader(f"🎠 {title}")

    # CSS for horizontal scroll
    st.markdown(
        """
        <style>
        .horizontal-carousel {
            display: flex;
            overflow-x: auto;
            gap: 1rem;
            padding: 1rem 0;
            scroll-behavior: smooth;
        }
        .horizontal-carousel::-webkit-scrollbar {
            height: 8px;
        }
        .horizontal-carousel::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .horizontal-carousel::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        .carousel-item {
            flex: 0 0 250px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 0.75rem;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build carousel HTML
    carousel_html = '<div class="horizontal-carousel">'

    for rec in recommendations[:10]:
        title_text = rec.get("title", "Unknown")[:40]
        price = rec.get("price", 0)
        stars = rec.get("stars", 0)
        reason = rec.get("reason", "")[:60]

        carousel_html += f"""
        <div class="carousel-item">
            <div style="
                background: #f0f0f0;
                height: 100px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
            ">📦</div>
            <div style="font-weight: bold; font-size: 0.9rem;">{title_text}...</div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <span style="font-weight: bold;">${price:.2f}</span>
                <span>⭐ {stars:.1f}</span>
            </div>
            <div style="font-size: 0.75rem; color: #666; margin-top: 0.5rem;">{reason}</div>
        </div>
        """

    carousel_html += "</div>"
    st.markdown(carousel_html, unsafe_allow_html=True)


def render_recommendation_summary(
    recommendations: list[dict[str, Any]],
    source_product: dict[str, Any] | None = None,
):
    """Render recommendation summary with source product context.

    Args:
        recommendations: List of recommendations
        source_product: Optional source product for context
    """
    if source_product:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 0.75rem;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 0.9rem; opacity: 0.9;">Based on:</div>
                <div style="font-weight: bold; font-size: 1.1rem;">
                    {source_product.get('title', 'Selected Product')[:60]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not recommendations:
        st.info("No recommendations found for this product.")
        return

    # Categorize recommendations
    categories = {}
    for rec in recommendations:
        match_type = rec.get("match_type", "similar")
        if match_type not in categories:
            categories[match_type] = []
        categories[match_type].append(rec)

    # Display by category
    for category, items in categories.items():
        emoji = {
            "similar": "🔄",
            "alternative": "↔️",
            "accessory": "🔌",
            "upgrade": "⬆️",
            "budget": "💰",
        }.get(category, "📦")

        with st.expander(f"{emoji} {category.title()} ({len(items)})", expanded=True):
            render_recommendation_carousel(
                items[:6],
                title="",
                items_per_row=3,
            )


def render_quick_recommendation_list(
    recommendations: list[dict[str, Any]],
    limit: int = 5,
):
    """Render a compact list of recommendations.

    Args:
        recommendations: List of recommendations
        limit: Maximum items to show
    """
    for rec in recommendations[:limit]:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            title = rec.get("title", "Unknown")
            st.markdown(f"**{title[:50]}**{'...' if len(title) > 50 else ''}")
            reason = rec.get("reason", "")
            if reason:
                st.caption(reason[:80])

        with col2:
            price = rec.get("price")
            if price:
                st.metric("Price", f"${price:.2f}")

        with col3:
            stars = rec.get("stars")
            if stars:
                st.metric("Rating", f"⭐ {stars:.1f}")

        st.divider()
