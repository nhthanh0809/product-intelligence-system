"""User favorites and comparison list management."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import streamlit as st


@dataclass
class FavoriteProduct:
    """A favorited product."""
    asin: str
    title: str
    brand: str | None = None
    price: float | None = None
    stars: float | None = None
    img_url: str | None = None
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


@dataclass
class ComparisonList:
    """A list of products for comparison."""
    name: str
    products: list[FavoriteProduct] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FavoritesManager:
    """Manages user favorites and comparison lists.

    Uses Streamlit session state for persistence during session.
    """

    SESSION_KEY = "favorites_manager"
    MAX_FAVORITES = 50
    MAX_COMPARISON_LIST = 5

    def __init__(self):
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Initialize session state if needed."""
        if "favorites" not in st.session_state:
            st.session_state.favorites = {}

        if "comparison_lists" not in st.session_state:
            st.session_state.comparison_lists = {}

        if "active_comparison" not in st.session_state:
            st.session_state.active_comparison = []

    def add_favorite(self, product: dict[str, Any]) -> bool:
        """Add a product to favorites.

        Args:
            product: Product dictionary

        Returns:
            True if added, False if already exists or limit reached
        """
        self._ensure_initialized()

        asin = product.get("asin")
        if not asin:
            return False

        if asin in st.session_state.favorites:
            return False

        if len(st.session_state.favorites) >= self.MAX_FAVORITES:
            return False

        favorite = FavoriteProduct(
            asin=asin,
            title=product.get("title", "Unknown"),
            brand=product.get("brand"),
            price=product.get("price"),
            stars=product.get("stars"),
            img_url=product.get("img_url"),
        )

        st.session_state.favorites[asin] = asdict(favorite)
        return True

    def remove_favorite(self, asin: str) -> bool:
        """Remove a product from favorites.

        Args:
            asin: Product ASIN

        Returns:
            True if removed, False if not found
        """
        self._ensure_initialized()

        if asin in st.session_state.favorites:
            del st.session_state.favorites[asin]
            return True
        return False

    def is_favorite(self, asin: str) -> bool:
        """Check if a product is in favorites.

        Args:
            asin: Product ASIN

        Returns:
            True if in favorites
        """
        self._ensure_initialized()
        return asin in st.session_state.favorites

    def get_favorites(self) -> list[dict]:
        """Get all favorites.

        Returns:
            List of favorite products
        """
        self._ensure_initialized()
        return list(st.session_state.favorites.values())

    def get_favorites_count(self) -> int:
        """Get number of favorites."""
        self._ensure_initialized()
        return len(st.session_state.favorites)

    def clear_favorites(self):
        """Clear all favorites."""
        self._ensure_initialized()
        st.session_state.favorites = {}

    # Comparison list methods

    def add_to_comparison(self, product: dict[str, Any]) -> bool:
        """Add product to active comparison list.

        Args:
            product: Product dictionary

        Returns:
            True if added
        """
        self._ensure_initialized()

        asin = product.get("asin")
        if not asin:
            return False

        # Check if already in comparison
        if any(p.get("asin") == asin for p in st.session_state.active_comparison):
            return False

        if len(st.session_state.active_comparison) >= self.MAX_COMPARISON_LIST:
            return False

        st.session_state.active_comparison.append({
            "asin": asin,
            "title": product.get("title", "Unknown"),
            "brand": product.get("brand"),
            "price": product.get("price"),
            "stars": product.get("stars"),
        })
        return True

    def remove_from_comparison(self, asin: str) -> bool:
        """Remove product from comparison list.

        Args:
            asin: Product ASIN

        Returns:
            True if removed
        """
        self._ensure_initialized()

        original_len = len(st.session_state.active_comparison)
        st.session_state.active_comparison = [
            p for p in st.session_state.active_comparison
            if p.get("asin") != asin
        ]
        return len(st.session_state.active_comparison) < original_len

    def get_comparison_list(self) -> list[dict]:
        """Get active comparison list."""
        self._ensure_initialized()
        return st.session_state.active_comparison

    def clear_comparison(self):
        """Clear comparison list."""
        self._ensure_initialized()
        st.session_state.active_comparison = []

    def is_in_comparison(self, asin: str) -> bool:
        """Check if product is in comparison list."""
        self._ensure_initialized()
        return any(p.get("asin") == asin for p in st.session_state.active_comparison)

    # Notes

    def update_notes(self, asin: str, notes: str) -> bool:
        """Update notes for a favorite.

        Args:
            asin: Product ASIN
            notes: Notes text

        Returns:
            True if updated
        """
        self._ensure_initialized()

        if asin in st.session_state.favorites:
            st.session_state.favorites[asin]["notes"] = notes
            return True
        return False


# Singleton instance
_favorites_manager: FavoritesManager | None = None


def get_favorites_manager() -> FavoritesManager:
    """Get favorites manager singleton."""
    global _favorites_manager
    if _favorites_manager is None:
        _favorites_manager = FavoritesManager()
    return _favorites_manager


# Convenience functions
def add_to_favorites(product: dict) -> bool:
    """Add product to favorites."""
    return get_favorites_manager().add_favorite(product)


def remove_from_favorites(asin: str) -> bool:
    """Remove product from favorites."""
    return get_favorites_manager().remove_favorite(asin)


def get_favorites() -> list[dict]:
    """Get all favorites."""
    return get_favorites_manager().get_favorites()


def is_favorite(asin: str) -> bool:
    """Check if product is favorite."""
    return get_favorites_manager().is_favorite(asin)


# UI Components
def render_favorites_sidebar():
    """Render favorites list in sidebar."""
    manager = get_favorites_manager()
    favorites = manager.get_favorites()
    comparison = manager.get_comparison_list()

    with st.sidebar:
        st.markdown("---")

        # Favorites section
        with st.expander(f"❤️ Favorites ({len(favorites)})", expanded=False):
            if not favorites:
                st.caption("No favorites yet")
            else:
                for fav in favorites[:10]:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(fav["title"][:30])
                    with col2:
                        if st.button("❌", key=f"rm_fav_{fav['asin']}", help="Remove"):
                            manager.remove_favorite(fav["asin"])
                            st.rerun()

                if len(favorites) > 10:
                    st.caption(f"...and {len(favorites) - 10} more")

        # Comparison section
        with st.expander(f"⚖️ Compare ({len(comparison)}/5)", expanded=False):
            if not comparison:
                st.caption("Add products to compare")
            else:
                for prod in comparison:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(prod["title"][:30])
                    with col2:
                        if st.button("❌", key=f"rm_cmp_{prod['asin']}", help="Remove"):
                            manager.remove_from_comparison(prod["asin"])
                            st.rerun()

                if len(comparison) >= 2:
                    if st.button("Compare Now", type="primary", use_container_width=True):
                        st.session_state.compare_mode_products = comparison
                        st.rerun()


def render_favorite_button(product: dict, key: str = ""):
    """Render favorite/compare buttons for a product.

    Args:
        product: Product dictionary
        key: Unique key suffix
    """
    manager = get_favorites_manager()
    asin = product.get("asin", "")

    if not asin:
        return

    col1, col2 = st.columns(2)

    with col1:
        is_fav = manager.is_favorite(asin)
        btn_label = "❤️" if is_fav else "🤍"
        btn_help = "Remove from favorites" if is_fav else "Add to favorites"

        if st.button(btn_label, key=f"fav_{asin}_{key}", help=btn_help):
            if is_fav:
                manager.remove_favorite(asin)
            else:
                manager.add_favorite(product)
            st.rerun()

    with col2:
        in_compare = manager.is_in_comparison(asin)
        btn_label = "⚖️✓" if in_compare else "⚖️"
        btn_help = "Remove from comparison" if in_compare else "Add to comparison"

        if st.button(btn_label, key=f"cmp_{asin}_{key}", help=btn_help):
            if in_compare:
                manager.remove_from_comparison(asin)
            else:
                if not manager.add_to_comparison(product):
                    st.warning("Comparison list is full (max 5)")
            st.rerun()
