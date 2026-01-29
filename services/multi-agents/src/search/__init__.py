"""
Search strategy module for configurable search in the multi-agent system.

This module provides:
- Abstract base classes for search strategies
- Strategy registry for dynamic strategy loading from database
- Pre-built strategies: keyword, semantic, hybrid
- Query analysis utilities

Usage:
    from src.search import SearchStrategyRegistry, SearchResult

    registry = SearchStrategyRegistry(config_repository)
    await registry.initialize()

    strategy = await registry.get_strategy_for_query("Sony headphones")
    results = await strategy.search("Sony headphones", limit=10)
"""

from .base import (
    BaseSearchStrategy,
    SearchResult,
    SearchResponse,
    SearchFilters,
    StrategySettings,
    StrategyType,
)
from .query_analyzer import QueryAnalyzer, QueryType, QueryAnalysis
from .clients import SearchClients, get_search_clients, close_search_clients
from .registry import (
    SearchStrategyRegistry,
    get_search_registry,
    shutdown_search_registry,
    STRATEGY_CLASSES,
)

__all__ = [
    # Base classes
    "BaseSearchStrategy",
    "SearchResult",
    "SearchResponse",
    "SearchFilters",
    "StrategySettings",
    "StrategyType",
    # Query analysis
    "QueryAnalyzer",
    "QueryType",
    "QueryAnalysis",
    # Clients
    "SearchClients",
    "get_search_clients",
    "close_search_clients",
    # Registry
    "SearchStrategyRegistry",
    "get_search_registry",
    "shutdown_search_registry",
    "STRATEGY_CLASSES",
]
