"""
Search Strategy Registry for dynamic strategy loading and selection.

Provides:
- Strategy loading from database configuration
- Query-type to strategy mapping
- Fallback strategy logic
- Strategy caching
"""

import asyncio
from typing import Any, Type

import structlog

from src.config.models import SearchStrategy as SearchStrategyModel, StrategyType as DBStrategyType
from src.config.repository import ConfigRepository

from .base import BaseSearchStrategy, StrategySettings, StrategyType
from .clients import SearchClients, get_search_clients
from .query_analyzer import QueryAnalyzer, QueryType

# Import strategy implementations
from .strategies.keyword import BasicKeywordStrategy, BrandModelPriorityStrategy
from .strategies.semantic import BasicSemanticStrategy, FormatAwareSemanticStrategy
from .strategies.hybrid import KeywordPriorityHybridStrategy, AdaptiveFusionStrategy

logger = structlog.get_logger()


# Strategy class registry - maps names to implementation classes
STRATEGY_CLASSES: dict[str, Type[BaseSearchStrategy]] = {
    # Keyword strategies
    "keyword_basic": BasicKeywordStrategy,
    "keyword_brand_model_priority": BrandModelPriorityStrategy,
    # Semantic strategies
    "semantic_basic": BasicSemanticStrategy,
    "semantic_format_aware": FormatAwareSemanticStrategy,
    # Hybrid strategies
    "hybrid_keyword_priority": KeywordPriorityHybridStrategy,
    "hybrid_adaptive": AdaptiveFusionStrategy,
}


class SearchStrategyRegistry:
    """Registry for search strategies.

    Manages strategy loading from database, caching, and query-type mapping.

    Usage:
        registry = SearchStrategyRegistry(config_repository)
        await registry.initialize()

        # Get strategy for a specific query
        strategy = await registry.get_strategy_for_query("Sony headphones")
        results = await strategy.search("Sony headphones")

        # Get a specific strategy by name
        hybrid = await registry.get_strategy("hybrid_keyword_priority")
    """

    def __init__(self, config_repository: ConfigRepository | None = None):
        """Initialize the registry.

        Args:
            config_repository: Repository for loading configuration
        """
        self._config_repository = config_repository
        self._strategies: dict[str, BaseSearchStrategy] = {}
        self._query_type_mapping: dict[str, list[str]] = {}
        self._default_strategy: str | None = None
        self._clients: SearchClients | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Check if registry is initialized."""
        return self._initialized

    async def initialize(
        self,
        config_repository: ConfigRepository | None = None,
    ) -> None:
        """Initialize the registry.

        Loads strategies and mappings from database configuration.

        Args:
            config_repository: Optional repository override
        """
        async with self._lock:
            if config_repository:
                self._config_repository = config_repository

            if not self._config_repository:
                raise ValueError("Config repository is required")

            try:
                # Initialize search clients
                self._clients = await get_search_clients()

                # Load strategies from database
                await self._load_strategies()

                # Load query-type mappings
                await self._load_mappings()

                self._initialized = True

                logger.info(
                    "search_registry_initialized",
                    strategies=len(self._strategies),
                    mappings=len(self._query_type_mapping),
                    default=self._default_strategy,
                )

            except Exception as e:
                logger.error("search_registry_init_failed", error=str(e))
                raise

    async def _load_strategies(self) -> None:
        """Load strategies from database configuration."""
        strategy_models = await self._config_repository.get_search_strategies(
            enabled_only=True
        )

        for model in strategy_models:
            await self._register_strategy_from_model(model)

            if model.is_default and self._default_strategy is None:
                self._default_strategy = model.name

        # Ensure we have a default strategy
        if not self._default_strategy and self._strategies:
            # Use hybrid_keyword_priority as default if available
            if "hybrid_keyword_priority" in self._strategies:
                self._default_strategy = "hybrid_keyword_priority"
            else:
                # Use first available strategy
                self._default_strategy = next(iter(self._strategies.keys()))

    async def _register_strategy_from_model(
        self,
        model: SearchStrategyModel,
    ) -> None:
        """Register a strategy from database model.

        Args:
            model: Database strategy model
        """
        # Get the strategy class
        strategy_class = STRATEGY_CLASSES.get(model.name)

        if not strategy_class:
            # Try implementation_class field
            if model.implementation_class:
                strategy_class = self._import_strategy_class(model.implementation_class)

        if not strategy_class:
            logger.warning(
                "unknown_strategy_class",
                strategy_name=model.name,
                implementation_class=model.implementation_class,
            )
            return

        # Create settings from model
        settings = StrategySettings.from_dict(model.settings or {})

        # Create and initialize strategy
        strategy = strategy_class(
            name=model.name,
            settings=settings,
            clients=self._clients,
        )
        await strategy.initialize(self._clients)

        self._strategies[model.name] = strategy

        logger.debug(
            "strategy_registered",
            name=model.name,
            type=model.strategy_type.value,
        )

    def _import_strategy_class(self, class_path: str) -> Type[BaseSearchStrategy] | None:
        """Import a strategy class from a module path.

        Args:
            class_path: Full module path (e.g., "src.search.strategies.keyword.BasicKeywordStrategy")

        Returns:
            Strategy class or None if import fails
        """
        try:
            parts = class_path.rsplit(".", 1)
            if len(parts) != 2:
                return None

            module_path, class_name = parts
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name, None)

        except Exception as e:
            logger.warning(
                "strategy_class_import_failed",
                class_path=class_path,
                error=str(e),
            )
            return None

    async def _load_mappings(self) -> None:
        """Load query-type to strategy mappings from database."""
        mappings = await self._config_repository.get_query_strategy_mappings()

        for mapping in mappings:
            if not mapping.is_enabled:
                continue

            query_type = mapping.query_type
            strategy = mapping.strategy

            if not strategy or strategy.name not in self._strategies:
                continue

            if query_type not in self._query_type_mapping:
                self._query_type_mapping[query_type] = []

            self._query_type_mapping[query_type].append(strategy.name)

        # Sort each mapping by priority (already sorted from DB, but ensure)
        for query_type in self._query_type_mapping:
            # Keep order from DB (should be priority-sorted)
            pass

    async def get_strategy(self, name: str) -> BaseSearchStrategy | None:
        """Get a strategy by name.

        Args:
            name: Strategy name

        Returns:
            Strategy instance or None if not found
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized")

        return self._strategies.get(name)

    async def get_strategy_for_query(
        self,
        query: str,
        fallback: bool = True,
    ) -> BaseSearchStrategy:
        """Get the best strategy for a query.

        Analyzes the query and returns the appropriate strategy based on
        query-type mappings.

        Args:
            query: Search query string
            fallback: Use fallback strategy if no mapping found

        Returns:
            Strategy instance

        Raises:
            RuntimeError: If no strategy available
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized")

        # Analyze query to determine type
        analysis = QueryAnalyzer.analyze(query)
        query_type = analysis.query_type.value.upper()

        # Look up mapped strategies
        mapped_strategies = self._query_type_mapping.get(query_type, [])

        # Try mapped strategies in priority order
        for strategy_name in mapped_strategies:
            strategy = self._strategies.get(strategy_name)
            if strategy and strategy.is_initialized:
                logger.debug(
                    "strategy_selected_for_query",
                    query=query[:50],
                    query_type=query_type,
                    strategy=strategy_name,
                )
                return strategy

        # Fallback to default strategy
        if fallback and self._default_strategy:
            strategy = self._strategies.get(self._default_strategy)
            if strategy and strategy.is_initialized:
                logger.debug(
                    "default_strategy_used",
                    query=query[:50],
                    query_type=query_type,
                    strategy=self._default_strategy,
                )
                return strategy

        # No strategy available
        raise RuntimeError(
            f"No strategy available for query type '{query_type}'"
        )

    async def get_strategies_by_type(
        self,
        strategy_type: StrategyType,
    ) -> list[BaseSearchStrategy]:
        """Get all strategies of a specific type.

        Args:
            strategy_type: Type of strategies to get

        Returns:
            List of matching strategies
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized")

        return [
            s for s in self._strategies.values()
            if s.strategy_type == strategy_type
        ]

    def list_strategies(self) -> list[dict[str, Any]]:
        """List all registered strategies.

        Returns:
            List of strategy info dicts
        """
        return [
            {
                "name": s.name,
                "type": s.strategy_type.value,
                "initialized": s.is_initialized,
                "is_default": s.name == self._default_strategy,
            }
            for s in self._strategies.values()
        ]

    def list_mappings(self) -> dict[str, list[str]]:
        """List query-type to strategy mappings.

        Returns:
            Dict of query_type -> strategy names
        """
        return dict(self._query_type_mapping)

    async def reload(self) -> None:
        """Reload strategies and mappings from database."""
        logger.info("search_registry_reloading")

        async with self._lock:
            self._strategies.clear()
            self._query_type_mapping.clear()
            self._default_strategy = None
            self._initialized = False

        await self.initialize()

    async def shutdown(self) -> None:
        """Shutdown the registry and clean up resources."""
        async with self._lock:
            self._strategies.clear()
            self._query_type_mapping.clear()
            self._default_strategy = None
            self._initialized = False

            if self._clients:
                await self._clients.close()
                self._clients = None

            logger.info("search_registry_shutdown")


# Singleton instance
_registry: SearchStrategyRegistry | None = None


async def get_search_registry(
    config_repository: ConfigRepository | None = None,
) -> SearchStrategyRegistry:
    """Get or create the search registry singleton.

    Args:
        config_repository: Config repository for initialization

    Returns:
        Initialized SearchStrategyRegistry instance
    """
    global _registry

    if _registry is None:
        _registry = SearchStrategyRegistry(config_repository)

    if not _registry.is_initialized and config_repository:
        await _registry.initialize(config_repository)

    return _registry


async def shutdown_search_registry() -> None:
    """Shutdown the search registry singleton."""
    global _registry

    if _registry:
        await _registry.shutdown()
        _registry = None
