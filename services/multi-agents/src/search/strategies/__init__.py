"""
Search strategy implementations.

This module provides concrete implementations of search strategies:
- Keyword strategies: Full-text search with Elasticsearch
- Semantic strategies: Vector search with Qdrant
- Hybrid strategies: Combined keyword + semantic with RRF fusion
"""

from .keyword import BrandModelPriorityStrategy, BasicKeywordStrategy
from .semantic import FormatAwareSemanticStrategy, BasicSemanticStrategy
from .hybrid import KeywordPriorityHybridStrategy, AdaptiveFusionStrategy

__all__ = [
    # Keyword strategies
    "BrandModelPriorityStrategy",
    "BasicKeywordStrategy",
    # Semantic strategies
    "FormatAwareSemanticStrategy",
    "BasicSemanticStrategy",
    # Hybrid strategies
    "KeywordPriorityHybridStrategy",
    "AdaptiveFusionStrategy",
]
