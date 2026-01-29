"""Chart components for visualization."""

from typing import Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_trend_chart(trending_products: list[dict[str, Any]]):
    """Render trending products chart."""
    if not trending_products:
        st.info("No trend data available.")
        return

    # Create DataFrame
    df = pd.DataFrame(trending_products)

    # Bar chart of trending products by bought_in_last_month
    if "bought_in_last_month" in df.columns:
        # Truncate titles for display
        df["short_title"] = df["title"].str[:30] + "..."

        fig = px.bar(
            df.head(10),
            x="short_title",
            y="bought_in_last_month",
            color="brand",
            title="Top Trending Products (Bought in Last Month)",
            labels={
                "short_title": "Product",
                "bought_in_last_month": "Units Sold",
                "brand": "Brand",
            },
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Rating distribution
    if "stars" in df.columns:
        fig2 = px.histogram(
            df,
            x="stars",
            nbins=10,
            title="Rating Distribution of Trending Products",
            labels={"stars": "Rating", "count": "Number of Products"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Brand breakdown
    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts().head(10)
        fig3 = px.pie(
            values=brand_counts.values,
            names=brand_counts.index,
            title="Top Brands in Trending Products",
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_price_chart(products: list[dict[str, Any]]):
    """Render price analysis chart."""
    if not products:
        st.info("No price data available.")
        return

    # Create DataFrame
    df = pd.DataFrame(products)

    if "current_price" not in df.columns:
        df["current_price"] = df.get("price", 0)

    # Price distribution
    fig = px.histogram(
        df,
        x="current_price",
        nbins=20,
        title="Price Distribution",
        labels={"current_price": "Price ($)", "count": "Number of Products"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Price vs Value Score scatter
    if "value_score" in df.columns:
        df["short_title"] = df["title"].str[:25] + "..."

        fig2 = px.scatter(
            df,
            x="current_price",
            y="value_score",
            hover_name="short_title",
            color="price_rating" if "price_rating" in df.columns else None,
            title="Price vs Value Score",
            labels={
                "current_price": "Price ($)",
                "value_score": "Value Score (1-10)",
            },
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Price comparison bar
    df_sorted = df.sort_values("current_price")
    df_sorted["short_title"] = df_sorted["title"].str[:20] + "..."

    fig3 = go.Figure()

    # Current price
    fig3.add_trace(go.Bar(
        name="Current Price",
        x=df_sorted["short_title"].head(10),
        y=df_sorted["current_price"].head(10),
        marker_color="steelblue",
    ))

    # List price if available
    if "list_price" in df.columns:
        fig3.add_trace(go.Bar(
            name="List Price",
            x=df_sorted["short_title"].head(10),
            y=df_sorted["list_price"].head(10),
            marker_color="lightgray",
        ))

    fig3.update_layout(
        title="Price Comparison",
        barmode="group",
        xaxis_tickangle=-45,
        height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)


def render_category_chart(categories: list[dict[str, Any]]):
    """Render category trends chart."""
    if not categories:
        st.info("No category data available.")
        return

    df = pd.DataFrame(categories)

    # Products per category
    fig = px.bar(
        df,
        x="category",
        y="product_count",
        title="Products by Category",
        labels={
            "category": "Category",
            "product_count": "Number of Products",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Average price by category
    if "avg_price" in df.columns:
        fig2 = px.bar(
            df,
            x="category",
            y="avg_price",
            title="Average Price by Category",
            labels={
                "category": "Category",
                "avg_price": "Average Price ($)",
            },
            color="avg_rating" if "avg_rating" in df.columns else None,
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)


def render_review_sentiment_chart(analysis: dict[str, Any]):
    """Render review sentiment analysis chart."""
    # Sentiment gauge
    score = analysis.get("sentiment_score", 0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=(score + 1) * 50,  # Convert -1 to 1 range to 0-100
        title={"text": "Sentiment Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "steelblue"},
            "steps": [
                {"range": [0, 33], "color": "lightcoral"},
                {"range": [33, 66], "color": "lightyellow"},
                {"range": [66, 100], "color": "lightgreen"},
            ],
        },
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Pros vs Cons
    pros = analysis.get("pros", [])
    cons = analysis.get("cons", [])

    if pros or cons:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Pros")
            for pro in pros[:5]:
                st.markdown(f"✓ {pro}")

        with col2:
            st.markdown("### Cons")
            for con in cons[:5]:
                st.markdown(f"✗ {con}")
