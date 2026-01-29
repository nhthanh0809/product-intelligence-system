"""Review analysis UI component."""

import streamlit as st
from typing import Any


def render_review_analysis(analysis: dict[str, Any], products_analyzed: int = 0):
    """Render comprehensive review analysis.

    Args:
        analysis: Analysis results dictionary
        products_analyzed: Number of products analyzed
    """
    if not analysis:
        st.info("No analysis data available.")
        return

    # Header with stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Products Analyzed",
            products_analyzed,
            help="Number of products included in analysis",
        )

    with col2:
        sentiment_score = analysis.get("sentiment_score")
        if sentiment_score is not None:
            st.metric(
                "Sentiment Score",
                f"{sentiment_score:.2f}",
                delta=_sentiment_delta(sentiment_score),
                help="Overall sentiment (-1 to 1)",
            )

    with col3:
        sentiment_label = analysis.get("sentiment_label", "N/A")
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(
            sentiment_label, "❓"
        )
        st.metric(
            "Overall Sentiment",
            f"{emoji} {sentiment_label.capitalize()}",
        )

    st.divider()

    # Pros and Cons
    col1, col2 = st.columns(2)

    with col1:
        render_pros(analysis.get("pros", []))

    with col2:
        render_cons(analysis.get("cons", []))

    # Common themes
    themes = analysis.get("common_themes", [])
    if themes:
        st.subheader("📊 Common Themes")
        render_theme_tags(themes)

    # Feature analysis (if available)
    feature_analysis = analysis.get("feature_analysis", {})
    if feature_analysis:
        st.subheader("⭐ Feature Analysis")
        render_feature_breakdown(feature_analysis)

    # Summary
    summary = analysis.get("summary")
    if summary:
        st.subheader("📝 Summary")
        st.markdown(summary)

    # Recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.info(rec)


def render_pros(pros: list[str]):
    """Render pros list with styling.

    Args:
        pros: List of positive points
    """
    st.subheader("✅ Pros")

    if not pros:
        st.caption("No pros identified")
        return

    for pro in pros[:5]:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #d4edda 0%, transparent 100%);
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #28a745;
                margin: 0.5rem 0;
            ">
                ✓ {pro}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_cons(cons: list[str]):
    """Render cons list with styling.

    Args:
        cons: List of negative points
    """
    st.subheader("❌ Cons")

    if not cons:
        st.caption("No cons identified")
        return

    for con in cons[:5]:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #f8d7da 0%, transparent 100%);
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #dc3545;
                margin: 0.5rem 0;
            ">
                ✗ {con}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_theme_tags(themes: list[str]):
    """Render themes as colored tags.

    Args:
        themes: List of common themes
    """
    if not themes:
        return

    # Colors for different themes
    colors = [
        "#3498db", "#9b59b6", "#e74c3c", "#2ecc71", "#f39c12",
        "#1abc9c", "#e91e63", "#00bcd4", "#ff5722", "#607d8b",
    ]

    html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">'

    for idx, theme in enumerate(themes[:10]):
        color = colors[idx % len(colors)]
        html += f"""
            <span style="
                background: {color};
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 1rem;
                font-size: 0.9rem;
            ">
                {theme}
            </span>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_feature_breakdown(feature_analysis: dict[str, dict]):
    """Render feature-by-feature sentiment breakdown.

    Args:
        feature_analysis: Dictionary of feature -> sentiment counts
    """
    if not feature_analysis:
        return

    # Calculate scores for each feature
    feature_scores = []
    for feature, counts in feature_analysis.items():
        if isinstance(counts, dict):
            positive = counts.get("positive", 0)
            negative = counts.get("negative", 0)
            neutral = counts.get("neutral", 0)
            total = positive + negative + neutral
            if total > 0:
                score = (positive - negative) / total
                feature_scores.append({
                    "feature": feature.replace("_", " ").title(),
                    "score": score,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "total": total,
                })

    # Sort by score
    feature_scores.sort(key=lambda x: x["score"], reverse=True)

    # Render as progress bars
    for item in feature_scores[:8]:
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            st.markdown(f"**{item['feature']}**")

        with col2:
            # Normalize score to 0-100 for progress bar
            progress = (item["score"] + 1) / 2  # Convert -1..1 to 0..1
            color = _score_to_color(item["score"])
            st.progress(progress, text=f"{item['score']:.2f}")

        with col3:
            st.caption(f"({item['total']} mentions)")


def render_sentiment_gauge(score: float, label: str = ""):
    """Render a sentiment gauge visualization.

    Args:
        score: Sentiment score (-1 to 1)
        label: Optional label
    """
    # Normalize to percentage
    percentage = (score + 1) * 50  # Convert -1..1 to 0..100

    # Determine color
    if score > 0.3:
        color = "#28a745"
        emoji = "😊"
    elif score < -0.3:
        color = "#dc3545"
        emoji = "😞"
    else:
        color = "#ffc107"
        emoji = "😐"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="
                width: 100%;
                height: 20px;
                background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
                border-radius: 10px;
                position: relative;
                margin: 1rem 0;
            ">
                <div style="
                    position: absolute;
                    left: {percentage}%;
                    top: -5px;
                    transform: translateX(-50%);
                    width: 10px;
                    height: 30px;
                    background: #333;
                    border-radius: 5px;
                "></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
                <span>Negative</span>
                <span>Neutral</span>
                <span>Positive</span>
            </div>
            {f'<div style="margin-top: 0.5rem; font-weight: bold;">{label}</div>' if label else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_highlights(reviews: list[dict], limit: int = 5):
    """Render highlighted review excerpts.

    Args:
        reviews: List of review dictionaries
        limit: Maximum reviews to show
    """
    if not reviews:
        st.info("No reviews available")
        return

    st.subheader("📖 Review Highlights")

    for review in reviews[:limit]:
        rating = review.get("rating", review.get("stars"))
        text = review.get("text", review.get("content", ""))[:200]
        helpful = review.get("helpful_votes", 0)

        # Rating stars
        stars = "⭐" * int(rating) if rating else ""

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: #f8f9fa;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 0.5rem 0;
                ">
                    <div style="margin-bottom: 0.5rem;">
                        {stars} {f'({helpful} found helpful)' if helpful else ''}
                    </div>
                    <div style="font-style: italic; color: #555;">
                        "{text}..."
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _sentiment_delta(score: float) -> str:
    """Get delta indicator for sentiment score."""
    if score > 0.3:
        return "Positive"
    elif score < -0.3:
        return "Negative"
    return "Neutral"


def _score_to_color(score: float) -> str:
    """Convert score to color."""
    if score > 0.3:
        return "#28a745"
    elif score < -0.3:
        return "#dc3545"
    return "#ffc107"
