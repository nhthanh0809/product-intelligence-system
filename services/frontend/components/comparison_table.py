"""Comparison table component."""

from typing import Any

import streamlit as st
import pandas as pd


def render_comparison_table(comparison: dict[str, Any]):
    """Render product comparison table."""
    products = comparison.get("products", [])

    if not products:
        st.info("No products to compare.")
        return

    # Winner and best value callouts
    col1, col2 = st.columns(2)

    with col1:
        winner = comparison.get("winner")
        if winner:
            st.success(f"🏆 **Winner:** {winner}")
            if comparison.get("winner_reason"):
                st.caption(comparison["winner_reason"])

    with col2:
        best_value = comparison.get("best_value")
        if best_value:
            st.info(f"💰 **Best Value:** {best_value}")

    st.divider()

    # Key differences
    differences = comparison.get("key_differences", [])
    if differences:
        st.subheader("Key Differences")
        for diff in differences:
            st.markdown(f"• {diff}")
        st.divider()

    # Comparison table
    st.subheader("Side-by-Side Comparison")

    # Build DataFrame
    table_data = {
        "Attribute": ["Title", "Brand", "Price", "Rating", "Pros", "Cons"]
    }

    for i, product in enumerate(products[:4]):
        col_name = f"Product {i + 1}"
        table_data[col_name] = [
            product.get("title", "N/A")[:40] + "...",
            product.get("brand", "N/A"),
            f"${product.get('price', 0):.2f}" if product.get("price") else "N/A",
            f"{product.get('stars', 0):.1f}/5" if product.get("stars") else "N/A",
            ", ".join(product.get("pros", [])[:2]) or "N/A",
            ", ".join(product.get("cons", [])[:2]) or "N/A",
        ]

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary
    if comparison.get("summary"):
        st.divider()
        st.subheader("Summary")
        st.markdown(comparison["summary"])


def render_mini_comparison(products: list[dict[str, Any]]):
    """Render a mini comparison view for 2-3 products."""
    if len(products) < 2:
        st.warning("Need at least 2 products to compare.")
        return

    cols = st.columns(len(products[:3]))

    for i, col in enumerate(cols):
        product = products[i]
        with col:
            st.markdown(f"**{product.get('title', 'Unknown')[:30]}...**")

            if product.get("price"):
                st.metric("Price", f"${product['price']:.2f}")

            if product.get("stars"):
                st.metric("Rating", f"{product['stars']:.1f}/5")

            if product.get("brand"):
                st.caption(f"by {product['brand']}")

            # Pros/Cons
            pros = product.get("pros", [])
            if pros:
                st.markdown("**Pros:**")
                for pro in pros[:2]:
                    st.markdown(f"✓ {pro[:30]}")

            cons = product.get("cons", [])
            if cons:
                st.markdown("**Cons:**")
                for con in cons[:2]:
                    st.markdown(f"✗ {con[:30]}")
