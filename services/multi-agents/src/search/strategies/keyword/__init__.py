"""
Keyword search strategies using Elasticsearch.

Provides full-text search strategies optimized for different query types:
- BrandModelPriorityStrategy: Best for brand+model queries (R@1 87.7%)
- BasicKeywordStrategy: Standard multi-match query
"""

from .brand_model_priority import BrandModelPriorityStrategy
from .basic import BasicKeywordStrategy

__all__ = [
    "BrandModelPriorityStrategy",
    "BasicKeywordStrategy",
]
