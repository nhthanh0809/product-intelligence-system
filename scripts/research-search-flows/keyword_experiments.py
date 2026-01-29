#!/usr/bin/env python3
"""
Keyword Search Flow Experiments

Experiments to optimize keyword search metrics:
1. Basic multi-match (baseline)
2. Boosted fields (title^5, brand^3)
3. Phrase matching
4. Fuzzy + exact combined
5. BM25 tuning
6. All fields search

Usage:
    python scripts/research-search-flows/keyword_experiments.py --eval-data data/eval/datasets/level3_retrieval_evaluation.json
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from base import (
    SearchConfig,
    SearchStrategy,
    DatabaseClients,
    QueryPreprocessor,
    load_evaluation_data,
    run_experiment,
    print_experiment_results,
    get_keyword_search_fields,
    get_all_keyword_fields,
    get_search_flows_config,
)
import config as cfg

logger = structlog.get_logger()


# =============================================================================
# Strategy 1: Basic Multi-Match (Baseline)
# =============================================================================

class BasicKeywordStrategy(SearchStrategy):
    """Basic multi-match search - current implementation.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_basic"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Basic multi-match with default boosting (baseline, {mode} mode)"

    def _get_search_fields(self) -> list[str]:
        """Get mode-appropriate search fields."""
        if self.config.is_original_mode:
            # Original mode: only title and category available
            return [
                "title^3",
                "title.autocomplete^2",
                "category_name",
            ]
        else:
            # Enrich mode: full field set
            return [
                "title^3",
                "title.autocomplete^2",
                "brand^2",
                "chunk_description",
                "chunk_features",
                "chunk_specs",
                "category_name",
            ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        es_query = {
            "multi_match": {
                "query": query,
                "fields": self._get_search_fields(),
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 2: High Title Boost
# =============================================================================

class HighTitleBoostStrategy(SearchStrategy):
    """Boost title field significantly.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_high_title_boost"

    @property
    def description(self) -> str:
        if self.config.is_original_mode:
            return "Title^10, category_name^1 (original mode - no brand/chunks)"
        return "Title^10, brand^5, other fields lower"

    def _get_search_fields(self) -> list[str]:
        """Get mode-appropriate search fields."""
        if self.config.is_original_mode:
            return [
                "title^10",
                "title.autocomplete^5",
                "category_name^1",
            ]
        else:
            return [
                "title^10",
                "title.autocomplete^5",
                "brand^5",
                "product_type^3",
                "chunk_description^1",
                "chunk_features^1",
                "category_name^1",
            ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        es_query = {
            "multi_match": {
                "query": query,
                "fields": self._get_search_fields(),
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 3: Phrase Match Priority
# =============================================================================

class PhraseMatchStrategy(SearchStrategy):
    """Prioritize phrase matches over individual terms.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_phrase_match"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Phrase matching with high boost for exact phrases ({mode} mode)"

    def _build_query_clauses(self, query: str) -> list[dict]:
        """Build mode-appropriate query clauses."""
        if self.config.is_original_mode:
            return [
                # Exact phrase match (highest priority) - title only
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^10"],
                        "type": "phrase",
                        "boost": 3.0,
                    }
                },
                # Phrase prefix (for partial matches)
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^5"],
                        "type": "phrase_prefix",
                        "boost": 2.0,
                    }
                },
                # Regular multi-match (fallback)
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "category_name"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ]
        else:
            return [
                # Exact phrase match (highest priority)
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^10", "brand^5"],
                        "type": "phrase",
                        "boost": 3.0,
                    }
                },
                # Phrase prefix (for partial matches)
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^5", "brand^3"],
                        "type": "phrase_prefix",
                        "boost": 2.0,
                    }
                },
                # Regular multi-match (fallback)
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "title^3",
                            "brand^2",
                            "chunk_description",
                            "chunk_features",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        # Combine phrase and term matching
        es_query = {
            "bool": {
                "should": self._build_query_clauses(query),
                "minimum_should_match": 1,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 4: Fuzzy + Exact Combined
# =============================================================================

class FuzzyExactCombinedStrategy(SearchStrategy):
    """Combine exact matching with fuzzy for typo tolerance.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_fuzzy_exact"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Exact match boost + fuzzy fallback for typo tolerance ({mode} mode)"

    def _build_query_clauses(self, query: str) -> list[dict]:
        """Build mode-appropriate query clauses."""
        if self.config.is_original_mode:
            return [
                # Exact match (no fuzziness) - title only
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^10"],
                        "type": "best_fields",
                        "boost": 2.0,
                    }
                },
                # Fuzzy match
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "title^5",
                            "title.autocomplete^3",
                            "category_name^1",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ]
        else:
            return [
                # Exact match (no fuzziness)
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^10", "brand^5", "product_type^3"],
                        "type": "best_fields",
                        "boost": 2.0,
                    }
                },
                # Fuzzy match
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "title^5",
                            "title.autocomplete^3",
                            "brand^3",
                            "chunk_description^1",
                            "chunk_features^1",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        es_query = {
            "bool": {
                "should": self._build_query_clauses(query),
                "minimum_should_match": 1,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 5: Cross-Fields Search
# =============================================================================

class CrossFieldsStrategy(SearchStrategy):
    """Cross-fields search for terms spanning multiple fields.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_cross_fields"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Cross-fields matching for multi-term queries ({mode} mode)"

    def _get_search_fields(self) -> list[str]:
        """Get mode-appropriate search fields."""
        if self.config.is_original_mode:
            return [
                "title^3",
                "category_name",
            ]
        else:
            return [
                "title^3",
                "brand^2",
                "product_type^2",
                "category_name",
                "chunk_description",
                "chunk_features",
            ]

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        es_query = {
            "multi_match": {
                "query": query,
                "fields": self._get_search_fields(),
                "type": "cross_fields",
                "operator": "or",
                "minimum_should_match": "30%",
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 6: All Fields with Smart Boosting (Mode-Aware)
# =============================================================================

class AllFieldsSmartBoostStrategy(SearchStrategy):
    """Search all available fields with smart boosting.

    Mode-aware: In 'original' mode, genAI_* fields are excluded.
    """

    @property
    def name(self) -> str:
        mode_suffix = "_original" if not self.config.has_genai_fields else ""
        return f"keyword_all_fields{mode_suffix}"

    @property
    def description(self) -> str:
        if self.config.has_genai_fields:
            return "All text fields with smart boosting (including genAI fields)"
        return "All text fields with smart boosting (no genAI fields - original mode)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        # Get mode-appropriate fields
        search_fields = get_all_keyword_fields(self.config.pipeline_mode)

        es_query = {
            "multi_match": {
                "query": query,
                "fields": search_fields,
                "type": "best_fields",
                "fuzziness": "AUTO",
                "tie_breaker": 0.3,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 7: Query Preprocessed + Combined Matching
# =============================================================================

class CombinedKeywordStrategy(SearchStrategy):
    """Combine query preprocessing with multiple match types.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_combined"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Preprocessed query + phrase + fuzzy + cross-fields ({mode} mode)"

    def _build_should_clauses(self, clean_query: str, keywords: list[str]) -> list[dict]:
        """Build mode-appropriate should clauses."""
        clauses = [
            # Phrase match on title (highest priority)
            {
                "match_phrase": {
                    "title": {
                        "query": clean_query,
                        "boost": 5.0,
                    }
                }
            },
        ]

        if self.config.is_original_mode:
            # Original mode: limited fields
            clauses.extend([
                # Multi-match on key fields
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^10",
                            "category_name^2",
                        ],
                        "type": "best_fields",
                        "boost": 3.0,
                    }
                },
                # Fuzzy match for typo tolerance
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^5",
                            "category_name^1",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ])
        else:
            # Enrich mode: full fields
            clauses.extend([
                # Multi-match on key fields
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^10",
                            "short_title^8",
                            "brand^5",
                            "product_type^4",
                        ],
                        "type": "best_fields",
                        "boost": 3.0,
                    }
                },
                # Fuzzy match for typo tolerance
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^5",
                            "brand^3",
                            "chunk_description^1",
                            "chunk_features^1",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ])

            # Add keyword-specific matching if keywords extracted (enrich mode only)
            if keywords:
                clauses.append({
                    "terms": {
                        "product_type_keywords": keywords,
                        "boost": 2.0,
                    }
                })

        return clauses

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        # Preprocess query
        clean_query = QueryPreprocessor.clean_query(query)
        keywords = QueryPreprocessor.extract_keywords(clean_query)

        es = await self.clients.get_elasticsearch()

        es_query = {
            "bool": {
                "should": self._build_should_clauses(clean_query, keywords),
                "minimum_should_match": 1,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 8: Model Number Aware Search
# =============================================================================

class ModelNumberAwareStrategy(SearchStrategy):
    """Extract and prioritize model numbers for exact matching.

    Many product queries include model numbers like TRX500FA, B07Z1QTH89, etc.
    This strategy extracts these and gives them extremely high boost.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_model_aware"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Extract model numbers and boost exact matches ({mode} mode)"

    def _extract_model_numbers(self, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        import re
        # Pattern for model numbers: alphanumeric with optional hyphens, 3+ chars
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        # Filter to only those with at least one digit
        return [m for m in matches if any(c.isdigit() for c in m)]

    def _build_should_clauses(self, clean_query: str, model_numbers: list[str]) -> list[dict]:
        """Build mode-appropriate should clauses."""
        should_clauses = []

        # If model numbers found, use very high boost for exact match
        if model_numbers:
            for model in model_numbers:
                # Exact match on title for model number
                should_clauses.append({
                    "match": {
                        "title": {
                            "query": model,
                            "boost": 20.0,  # Very high boost for model numbers
                        }
                    }
                })
                # Also check in the full query text
                if self.config.is_original_mode:
                    should_clauses.append({
                        "query_string": {
                            "query": f"*{model}*",
                            "fields": ["title^15"],
                            "boost": 15.0,
                        }
                    })
                else:
                    should_clauses.append({
                        "query_string": {
                            "query": f"*{model}*",
                            "fields": ["title^15", "short_title^10"],
                            "boost": 15.0,
                        }
                    })

        # Phrase match on full query (high priority)
        should_clauses.append({
            "match_phrase": {
                "title": {
                    "query": clean_query,
                    "boost": 8.0,
                    "slop": 2,  # Allow some word reordering
                }
            }
        })

        if self.config.is_original_mode:
            # Original mode: limited fields
            should_clauses.extend([
                # Multi-match on key fields
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^10",
                            "category_name^2",
                        ],
                        "type": "best_fields",
                        "boost": 3.0,
                    }
                },
                # Fuzzy fallback
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": ["title^5", "category_name"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ])
        else:
            # Enrich mode: full fields
            should_clauses.extend([
                # Multi-match on key fields
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^10",
                            "short_title^8",
                            "brand^5",
                            "product_type^4",
                        ],
                        "type": "best_fields",
                        "boost": 3.0,
                    }
                },
                # Fuzzy fallback
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^5",
                            "brand^3",
                            "chunk_description",
                            "chunk_features",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ])

        return should_clauses

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        # Extract model numbers from query
        model_numbers = self._extract_model_numbers(query)
        clean_query = QueryPreprocessor.clean_query(query)

        es_query = {
            "bool": {
                "should": self._build_should_clauses(clean_query, model_numbers),
                "minimum_should_match": 1,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "asin": hit["_source"].get("asin"),
                "source": hit["_source"],
            })

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 9: Brand+Model Priority (BEST for specific queries - 87.7% R@1)
# =============================================================================

class BrandModelPriorityStrategy(SearchStrategy):
    """Prioritize brand+model queries - achieves 87.7% R@1 on these.

    Based on experiments (Jan 2026):
    - brand_model queries: 87.7% R@1 (vs semantic 43.1%)
    - model_numbers queries: 78.5% R@1
    - first_words queries: 76% R@1

    This is the BEST strategy for specific product queries.

    Mode-aware: adjusts fields based on original vs enrich mode.
    Note: In original mode, brand extraction still works but brand field matching
    is disabled since the brand field is not populated.
    """

    @property
    def name(self) -> str:
        return "keyword_brand_model_priority"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"BEST: Brand+model priority ({mode} mode)"

    def _extract_model_numbers(self, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        import re
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        return [m for m in matches if any(c.isdigit() for c in m)]

    def _extract_brand(self, query: str) -> str:
        """Extract brand name from query (first capitalized word)."""
        words = query.split()
        for word in words:
            if word and word[0].isupper() and word.isalpha() and len(word) > 2:
                return word
        return ""

    def _build_should_clauses(
        self,
        clean_query: str,
        model_numbers: list[str],
        brand: str,
    ) -> list[dict]:
        """Build mode-appropriate should clauses."""
        should_clauses = []

        # HIGHEST PRIORITY: Model number exact match (25x boost)
        if model_numbers:
            for model in model_numbers:
                # Exact match in title
                should_clauses.append({
                    "match": {
                        "title": {
                            "query": model,
                            "boost": 30.0,
                        }
                    }
                })
                # Also in short_title (enrich mode only)
                if not self.config.is_original_mode:
                    should_clauses.append({
                        "match": {
                            "short_title": {
                                "query": model,
                                "boost": 25.0,
                            }
                        }
                    })

        # HIGH PRIORITY: Brand + model combination (enrich mode only)
        if brand and model_numbers and not self.config.is_original_mode:
            should_clauses.append({
                "bool": {
                    "must": [
                        {"match": {"brand": {"query": brand, "boost": 10.0}}},
                        {"match": {"title": {"query": model_numbers[0], "boost": 15.0}}},
                    ],
                    "boost": 5.0,
                }
            })

        # Phrase match on full query (high priority)
        should_clauses.append({
            "match_phrase": {
                "title": {
                    "query": clean_query,
                    "boost": 12.0,
                    "slop": 2,
                }
            }
        })

        # Multi-match on key fields
        if self.config.is_original_mode:
            should_clauses.append({
                "multi_match": {
                    "query": clean_query,
                    "fields": [
                        "title^10",
                        "category_name^2",
                    ],
                    "type": "best_fields",
                    "boost": 4.0,
                }
            })
            # Fuzzy fallback
            should_clauses.append({
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^4", "category_name"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            })
        else:
            should_clauses.append({
                "multi_match": {
                    "query": clean_query,
                    "fields": [
                        "title^10",
                        "short_title^8",
                        "brand^6",
                        "product_type^4",
                    ],
                    "type": "best_fields",
                    "boost": 4.0,
                }
            })
            # Fuzzy fallback
            should_clauses.append({
                "multi_match": {
                    "query": clean_query,
                    "fields": ["title^4", "brand^2", "chunk_description"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            })

        return should_clauses

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        # Extract brand and model numbers
        model_numbers = self._extract_model_numbers(query)
        brand = self._extract_brand(query)
        clean_query = QueryPreprocessor.clean_query(query)

        # Overfetch for quality reranking
        fetch_limit = limit * 4

        es_query = {
            "bool": {
                "should": self._build_should_clauses(clean_query, model_numbers, brand),
                "minimum_should_match": 1,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=fetch_limit,
        )

        # Rerank by quality signals
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]

            # Minor quality boost
            stars = source.get("stars") or 0
            reviews = source.get("reviews_count") or 0
            is_bestseller = source.get("is_best_seller", False)

            quality_boost = 1.0
            if stars >= 4:
                quality_boost += (stars - 3) * 0.02
            if reviews > 100:
                quality_boost += min(0.08, (reviews / 5000) * 0.04)
            if is_bestseller:
                quality_boost += 0.03

            final_score = score * quality_boost

            results.append({
                "id": hit["_id"],
                "score": final_score,
                "asin": source.get("asin"),
                "source": source,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Strategy 10: Optimized Keyword Search
# =============================================================================

class OptimizedKeywordStrategy(SearchStrategy):
    """Best practices combined: model numbers + phrase + fuzzy + overfetch.

    Mode-aware: adjusts fields based on original vs enrich mode.
    """

    @property
    def name(self) -> str:
        return "keyword_optimized"

    @property
    def description(self) -> str:
        mode = "original" if self.config.is_original_mode else "enrich"
        return f"Model-aware + phrase + fuzzy + overfetch/rerank ({mode} mode)"

    def _extract_model_numbers(self, query: str) -> list[str]:
        """Extract potential model/part numbers from query."""
        import re
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, query.upper())
        return [m for m in matches if any(c.isdigit() for c in m)]

    def _build_should_clauses(self, clean_query: str, model_numbers: list[str]) -> list[dict]:
        """Build mode-appropriate should clauses."""
        should_clauses = []

        # Model number exact matches (highest priority)
        if model_numbers:
            for model in model_numbers:
                should_clauses.append({
                    "match": {
                        "title": {
                            "query": model,
                            "boost": 25.0,
                        }
                    }
                })

        # Phrase match on title
        should_clauses.append({
            "match_phrase": {
                "title": {
                    "query": clean_query,
                    "boost": 10.0,
                    "slop": 2,
                }
            }
        })

        if self.config.is_original_mode:
            # Original mode: limited fields
            should_clauses.extend([
                # Multi-match best fields
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^10",
                            "category_name^2",
                        ],
                        "type": "best_fields",
                        "boost": 5.0,
                    }
                },
                # Cross-fields for multi-word queries
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^5",
                            "category_name^2",
                        ],
                        "type": "cross_fields",
                        "operator": "or",
                    }
                },
                # Fuzzy fallback
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": ["title^3", "category_name"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ])
        else:
            # Enrich mode: full fields
            should_clauses.extend([
                # Multi-match best fields
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^10",
                            "short_title^8",
                            "brand^6",
                            "product_type^4",
                            "product_type_keywords^3",
                        ],
                        "type": "best_fields",
                        "boost": 5.0,
                    }
                },
                # Cross-fields for multi-word queries
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": [
                            "title^5",
                            "brand^3",
                            "chunk_description^2",
                            "chunk_features^2",
                        ],
                        "type": "cross_fields",
                        "operator": "or",
                    }
                },
                # Fuzzy fallback
                {
                    "multi_match": {
                        "query": clean_query,
                        "fields": ["title^3", "brand^2", "chunk_description"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
            ])

        return should_clauses

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        import time
        start = time.perf_counter()

        es = await self.clients.get_elasticsearch()

        # Extract model numbers
        model_numbers = self._extract_model_numbers(query)
        clean_query = QueryPreprocessor.clean_query(query)

        # Overfetch 3x for reranking
        fetch_limit = limit * 3

        es_query = {
            "bool": {
                "should": self._build_should_clauses(clean_query, model_numbers),
                "minimum_should_match": 1,
            }
        }

        response = await es.search(
            index=self.config.es_index,
            query=es_query,
            size=fetch_limit,
        )

        # Rerank by quality signals
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]

            # Boost by quality signals
            stars = source.get("stars") or 0
            reviews = source.get("reviews_count") or 0
            is_bestseller = source.get("is_best_seller", False)

            # Quality boost
            quality_boost = 1.0
            quality_boost += (stars / 5.0) * 0.1  # Up to 10% boost for 5-star
            quality_boost += min(0.15, (reviews / 1000) * 0.05)  # Up to 15% for popular
            if is_bestseller:
                quality_boost += 0.05

            final_score = score * quality_boost

            results.append({
                "id": hit["_id"],
                "score": final_score,
                "asin": source.get("asin"),
                "source": source,
            })

        # Sort by adjusted score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        latency = time.perf_counter() - start
        return results, latency


# =============================================================================
# Main
# =============================================================================

def _get_cli_defaults() -> dict:
    """Get CLI defaults from pipeline_config.yaml.

    Resolves {count} and {mode} placeholders in paths to match 06_generate_eval_data.py output.
    """
    config = get_search_flows_config()
    data_dir = cfg.get_data_dir()
    product_count = cfg.get_count()
    pipeline_mode = cfg.get_mode()

    # Get eval_data path with {count} and {mode} substitution
    eval_data = config.get("eval_data", "eval/datasets/level3_retrieval_{count}_{mode}.json")
    eval_data = eval_data.replace("{count}", str(product_count)).replace("{mode}", pipeline_mode)
    eval_data_path = data_dir / eval_data if not Path(eval_data).is_absolute() else Path(eval_data)

    return {
        "eval_data": str(eval_data_path),
        "es_host": config.get("elasticsearch_host", "localhost"),
        "es_port": config.get("elasticsearch_port", 9200),
        "es_index": config.get("elasticsearch_index", "products"),
        "verbose": config.get("verbose", False),
        "product_count": product_count,
        "pipeline_mode": pipeline_mode,
    }


_cli_defaults = _get_cli_defaults()


@click.command()
@click.option(
    "--eval-data",
    type=click.Path(exists=True),
    default=None,
    help="Evaluation dataset JSON file (default: from config)",
)
@click.option("--es-host", default=None, help="Elasticsearch host (default: from config)")
@click.option("--es-port", default=None, type=int, help="Elasticsearch port (default: from config)")
@click.option("--es-index", default=None, help="Elasticsearch index (default: from config)")
@click.option(
    "--mode",
    type=click.Choice(["original", "enrich", "auto"]),
    default="auto",
    help="Pipeline mode: 'original' (no genAI fields), 'enrich' (with genAI fields), 'auto' (from config)",
)
@click.option("--verbose", is_flag=True, default=None, help="Verbose output (default: from config)")
def main(
    eval_data: str | None,
    es_host: str | None,
    es_port: int | None,
    es_index: str | None,
    mode: str,
    verbose: bool | None,
):
    """Run keyword search experiments.

    Mode affects which fields are used in search:
    - original: Core fields only (title, brand, category, chunk_*)
    - enrich: Core + genAI fields (genAI_summary, genAI_best_for, etc.)
    """
    # Apply config defaults
    eval_data = eval_data or _cli_defaults["eval_data"]
    es_host = es_host or _cli_defaults["es_host"]
    es_port = es_port or _cli_defaults["es_port"]
    es_index = es_index or _cli_defaults["es_index"]
    verbose = verbose if verbose is not None else _cli_defaults["verbose"]

    # Determine pipeline mode
    from base import get_pipeline_mode
    if mode == "auto":
        pipeline_mode = get_pipeline_mode()
    else:
        pipeline_mode = mode

    print("=" * 70)
    print("KEYWORD SEARCH EXPERIMENTS")
    print("=" * 70)
    print(f"Pipeline Mode: {pipeline_mode.upper()}")
    if pipeline_mode == "original":
        print("  Fields: Core fields only (no genAI_* fields)")
    else:
        print("  Fields: Core + genAI fields")
    print()

    # Configuration
    config = SearchConfig(
        es_host=es_host,
        es_port=es_port,
        es_index=es_index,
        pipeline_mode=pipeline_mode,
    )

    # Load evaluation data
    eval_queries = load_evaluation_data(Path(eval_data), search_type="keyword")
    print(f"Loaded {len(eval_queries)} keyword search queries")

    if not eval_queries:
        print("No keyword queries found in evaluation data!")
        return

    # Initialize clients
    clients = DatabaseClients(config)

    # Define strategies
    strategies = [
        BasicKeywordStrategy(clients, config),
        HighTitleBoostStrategy(clients, config),
        PhraseMatchStrategy(clients, config),
        FuzzyExactCombinedStrategy(clients, config),
        CrossFieldsStrategy(clients, config),
        AllFieldsSmartBoostStrategy(clients, config),
        CombinedKeywordStrategy(clients, config),
        ModelNumberAwareStrategy(clients, config),
        BrandModelPriorityStrategy(clients, config),  # BEST for brand+model queries
        OptimizedKeywordStrategy(clients, config),
    ]

    # Run experiments
    async def run_all():
        results = []
        for strategy in strategies:
            print(f"\nRunning: {strategy.name}...")
            result = await run_experiment(
                strategy,
                eval_queries,
                search_type="keyword",
                verbose=verbose,
            )
            results.append(result)
            print(f"  Recall@10: {result.metrics.recall_at_10:.4f}, MRR: {result.metrics.mrr:.4f}")

        await clients.close()
        return results

    results = asyncio.run(run_all())

    # Print comparison
    print_experiment_results(results)


if __name__ == "__main__":
    main()
