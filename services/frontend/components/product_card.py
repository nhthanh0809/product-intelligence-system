"""Product card component - Simple UI."""

from typing import Any

import streamlit as st


def render_product_card(product: dict[str, Any], key: str = ""):
    """Render a simple product card."""
    title = product.get("title", "Unknown Product")
    brand = product.get("brand", "")
    price = product.get("price")
    list_price = product.get("list_price")
    stars = product.get("stars")
    reviews = product.get("reviews_count", 0)
    img_url = product.get("img_url", "")
    summary = product.get("summary") or product.get("genai_summary", "")

    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            if img_url:
                st.image(img_url, width=120)
            else:
                st.write("No image")

        with col2:
            st.markdown(f"**{title[:100]}**")

            if brand:
                st.caption(f"by {brand}")

            # Price
            if price:
                price_text = f"**${price:.2f}**"
                if list_price and list_price > price:
                    discount = ((list_price - price) / list_price) * 100
                    price_text += f" ~~${list_price:.2f}~~ ({discount:.0f}% off)"
                st.markdown(price_text)

            # Rating
            if stars:
                stars_text = f"{'★' * int(stars)}{'☆' * (5 - int(stars))} {stars:.1f}/5"
                if reviews:
                    stars_text += f" ({reviews:,} reviews)"
                st.caption(stars_text)

            # Badges
            badges = []
            if product.get("is_best_seller"):
                badges.append("Best Seller")
            if product.get("is_amazon_choice"):
                badges.append("Top Rated")
            if product.get("is_prime") or product.get("prime_eligible"):
                badges.append("Prime")
            if badges:
                st.caption(" | ".join(badges))

            # Summary
            if summary:
                st.caption(f"{summary[:150]}...")

        st.divider()


def render_product_grid(products: list[dict[str, Any]], columns: int = 3):
    """Render products in a grid layout."""
    if not products:
        st.info("No products to display.")
        return

    for i in range(0, len(products), columns):
        cols = st.columns(columns)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(products):
                with col:
                    _render_compact_card(products[idx])


def _render_compact_card(product: dict[str, Any]):
    """Render a compact product card."""
    title = product.get("title", "Unknown")[:50]
    price = product.get("price")
    stars = product.get("stars")
    brand = product.get("brand", "")

    with st.container():
        st.markdown(f"**{title}...**")

        info = []
        if price:
            info.append(f"${price:.2f}")
        if stars:
            info.append(f"★ {stars:.1f}")
        if brand:
            info.append(brand)

        if info:
            st.caption(" | ".join(info))


def render_product_mini(product: dict[str, Any]):
    """Render minimal product info."""
    title = product.get("title", "Unknown")[:50]
    price = product.get("price")
    stars = product.get("stars")

    st.markdown(f"**{title}...**")
    info = []
    if price:
        info.append(f"${price:.2f}")
    if stars:
        info.append(f"{stars:.1f}/5")
    if info:
        st.caption(" | ".join(info))
