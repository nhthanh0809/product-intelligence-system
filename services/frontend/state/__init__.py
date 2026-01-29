"""State management for the frontend."""

from state.favorites import (
    FavoritesManager,
    get_favorites_manager,
    add_to_favorites,
    remove_from_favorites,
    get_favorites,
    is_favorite,
    render_favorites_sidebar,
)

__all__ = [
    "FavoritesManager",
    "get_favorites_manager",
    "add_to_favorites",
    "remove_from_favorites",
    "get_favorites",
    "is_favorite",
    "render_favorites_sidebar",
]
