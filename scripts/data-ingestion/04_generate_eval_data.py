#!/usr/bin/env python3
"""
Script 06: Generate Comprehensive Evaluation Data

Generate evaluation datasets aligned with DB Design and Multi-Agent Architecture.
This is step 06 in the data ingestion pipeline, taking input from 03_clean_data.

Input: Output of 03_clean_data (cleaned/mvp_{count}_{mode}_cleaned.csv)

GENERATION ORDER (user scenarios drive all other levels):
1. Level 6: E2E Scenarios (30 user scenarios: D1-D6, C1-C5, A1-A5, P1-P5, T1-T5, R1-R5)
2. Level 5: Agent Query Accuracy (node routing, tool selection - agent-ready)
3. Level 4: Storage Design Validation (QA from Qdrant, PostgreSQL fallback tests)
4. Level 3: Retrieval Performance (parent search, child/section search, hybrid search)

BASED ON DATA INGESTION PIPELINE (evaluate pipeline outputs):
5. Level 2: Indexing Correctness (parent-child hierarchy, GenAI field presence, filter indexes)
6. Level 1: Embedding Quality (similarity, triplets, clustering, product_type coherence)

Usage:
    python scripts/06_generate_eval_data.py --input data/cleaned/mvp_10000_original_cleaned.csv
"""

import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import httpx
import numpy as np
import pandas as pd
import structlog

# Add paths for imports
# scripts/data-ingestion/ -> scripts/ (for config.py)
# services/data-pipeline/ (for src.config)
sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "data-pipeline"))  # services/data-pipeline/

from src.config import get_settings
import config as cfg

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


# =============================================================================
# Constants aligned with DB Design Proposal
# =============================================================================

# Parent node required fields (from DB Design Part 3)
PARENT_REQUIRED_FIELDS = {
    "identity": ["asin", "title", "short_title", "brand", "product_type", "product_type_keywords"],
    "category": ["category_level1", "category_level2", "category_level3"],
    "display": ["price", "list_price", "stars", "reviews_count", "img_url", "is_best_seller", "availability"],
    "genai_quick": [
        "genAI_summary", "genAI_primary_function", "genAI_best_for", "genAI_use_cases",
        "genAI_target_audience", "genAI_key_capabilities", "genAI_unique_selling_points", "genAI_value_score"
    ],
    "scores": ["popularity_score", "trending_rank", "price_percentile"],
    "metadata": ["node_type", "indexed_at"],
}

# Child sections (5 sections per product from DB Design Part 4)
CHILD_SECTIONS = {
    "description": {
        "genai_fields": ["genAI_detailed_description", "genAI_how_it_works", "genAI_whats_included", "genAI_materials"],
        "questions": ["Tell me more about this product", "How does this work?", "What comes in the box?"],
    },
    "features": {
        "genai_fields": ["genAI_features_detailed", "genAI_standout_features", "genAI_technology_explained"],
        "questions": ["What are the key features?", "What makes this special?", "What technology does it use?"],
    },
    "specs": {
        "genai_fields": ["genAI_specs_summary", "genAI_specs_comparison_ready", "genAI_specs_limitations"],
        "questions": ["What are the specifications?", "What's the battery life?", "Any limitations?"],
    },
    "reviews": {
        "genai_fields": ["genAI_sentiment_score", "genAI_common_praises", "genAI_common_complaints", "genAI_durability_feedback"],
        "questions": ["What do users say?", "What do people like about it?", "What are the complaints?"],
    },
    "use_cases": {
        "genai_fields": ["genAI_use_case_scenarios", "genAI_ideal_user_profiles", "genAI_not_recommended_for", "genAI_problems_solved"],
        "questions": ["Is this good for travel?", "Who should buy this?", "What problems does it solve?"],
    },
}

# Child node inherited filter fields
CHILD_INHERITED_FIELDS = ["category_level1", "brand", "price", "stars", "parent_asin", "section", "node_type"]


# =============================================================================
# Mode-Specific Constants (original vs enrich)
# =============================================================================

def get_parent_required_fields(mode: str) -> dict:
    """Get parent required fields based on pipeline mode.

    Args:
        mode: Pipeline mode ('original' or 'enrich')

    Returns:
        Dictionary of required field categories

    Note:
        Original mode only has limited fields from CSV (no brand, short_title, product_type, genAI_*).
        Enrich mode has full field set from LLM enrichment.
    """
    if mode == "original":
        # Original mode: only CSV-derived fields available
        # See 04_generate_embeddings.py lines 622-639 for parent metadata in original mode
        return {
            "identity": ["asin", "title"],  # No brand, short_title, product_type in original mode
            "category": ["category_level1"],  # category_level2/3 may be empty
            "display": ["price", "list_price", "stars", "reviews_count", "img_url", "is_best_seller"],
            "scores": [],  # No computed scores in original mode
            "metadata": ["node_type"],
            "genai_quick": [],  # No genAI fields in original mode
        }
    else:
        # Enrich mode: full field set with LLM enrichment
        return {
            "identity": ["asin", "title", "short_title", "brand", "product_type", "product_type_keywords"],
            "category": ["category_level1", "category_level2", "category_level3"],
            "display": ["price", "list_price", "stars", "reviews_count", "img_url", "is_best_seller", "availability"],
            "scores": ["popularity_score", "trending_rank", "price_percentile"],
            "metadata": ["node_type", "indexed_at"],
            "genai_quick": [
                "genAI_summary", "genAI_primary_function", "genAI_best_for", "genAI_use_cases",
                "genAI_target_audience", "genAI_key_capabilities", "genAI_unique_selling_points", "genAI_value_score"
            ],
        }


def get_child_sections(mode: str) -> dict:
    """Get child section configuration based on pipeline mode.

    Args:
        mode: Pipeline mode ('original' or 'enrich')

    Returns:
        Dictionary of child sections with genai_fields and questions
    """
    if mode == "enrich":
        return {
            "description": {
                "genai_fields": ["genAI_detailed_description", "genAI_how_it_works", "genAI_whats_included", "genAI_materials"],
                "questions": ["Tell me more about this product", "How does this work?", "What comes in the box?"],
            },
            "features": {
                "genai_fields": ["genAI_features_detailed", "genAI_standout_features", "genAI_technology_explained"],
                "questions": ["What are the key features?", "What makes this special?", "What technology does it use?"],
            },
            "specs": {
                "genai_fields": ["genAI_specs_summary", "genAI_specs_comparison_ready", "genAI_specs_limitations"],
                "questions": ["What are the specifications?", "What's the battery life?", "Any limitations?"],
            },
            "reviews": {
                "genai_fields": ["genAI_sentiment_score", "genAI_common_praises", "genAI_common_complaints", "genAI_durability_feedback"],
                "questions": ["What do users say?", "What do people like about it?", "What are the complaints?"],
            },
            "use_cases": {
                "genai_fields": ["genAI_use_case_scenarios", "genAI_ideal_user_profiles", "genAI_not_recommended_for", "genAI_problems_solved"],
                "questions": ["Is this good for travel?", "Who should buy this?", "What problems does it solve?"],
            },
        }
    else:
        # Original mode: no genAI fields, simpler questions based on basic fields
        return {
            "description": {
                "genai_fields": [],
                "questions": ["What is this product?", "Tell me about this item", "Product overview"],
            },
            "features": {
                "genai_fields": [],
                "questions": ["What are the features?", "Product features", "Key capabilities"],
            },
            "specs": {
                "genai_fields": [],
                "questions": ["Technical specifications", "Product specs", "Dimensions and details"],
            },
            "reviews": {
                "genai_fields": [],
                "questions": ["Customer reviews", "User feedback", "Ratings"],
            },
            "use_cases": {
                "genai_fields": [],
                "questions": ["What is it for?", "Who should buy this?", "Best uses"],
            },
        }


def get_question_routing(mode: str) -> list[dict]:
    """Get question-to-node routing based on pipeline mode.

    Args:
        mode: Pipeline mode ('original' or 'enrich')

    Returns:
        List of routing dictionaries
    """
    if mode == "enrich":
        return [
            # Parent node questions (with genAI fields)
            {"pattern": "What is {product}?", "node": "parent", "field": "genAI_summary"},
            {"pattern": "What does {product} do?", "node": "parent", "field": "genAI_primary_function"},
            {"pattern": "What is {product} used for?", "node": "parent", "field": "genAI_use_cases"},
            {"pattern": "Who is {product} for?", "node": "parent", "field": "genAI_target_audience"},
            {"pattern": "Find {product_type} under ${price}", "node": "parent", "field": None},
            {"pattern": "Best {product_type}", "node": "parent", "field": "genAI_best_for"},
            # Child node questions
            {"pattern": "Tell me more about {product}", "node": "description", "field": "genAI_detailed_description"},
            {"pattern": "How does {feature} work?", "node": "features", "field": "genAI_technology_explained"},
            {"pattern": "What's the battery life of {product}?", "node": "specs", "field": "specs_key"},
            {"pattern": "Does {product} support {capability}?", "node": "specs", "field": "specs_key"},
            {"pattern": "What do users say about {product}?", "node": "reviews", "field": "genAI_common_praises"},
            {"pattern": "What are problems with {product}?", "node": "reviews", "field": "genAI_common_complaints"},
            {"pattern": "Is {product} good for {activity}?", "node": "use_cases", "field": "genAI_use_case_scenarios"},
            {"pattern": "Who should buy {product}?", "node": "use_cases", "field": "genAI_ideal_user_profiles"},
        ]
    else:
        # Original mode: route to basic fields
        return [
            # Parent node questions (basic fields only)
            {"pattern": "What is {product}?", "node": "parent", "field": "title"},
            {"pattern": "What does {product} do?", "node": "parent", "field": "product_type"},
            {"pattern": "What is {product} used for?", "node": "parent", "field": "category_level1"},
            {"pattern": "Who is {product} for?", "node": "parent", "field": "category_level1"},
            {"pattern": "Find {product_type} under ${price}", "node": "parent", "field": None},
            {"pattern": "Best {product_type}", "node": "parent", "field": "stars"},
            # Child node questions (chunk content)
            {"pattern": "Tell me more about {product}", "node": "description", "field": "chunk_description"},
            {"pattern": "How does {feature} work?", "node": "features", "field": "chunk_features"},
            {"pattern": "What's the battery life of {product}?", "node": "specs", "field": "chunk_specs"},
            {"pattern": "Does {product} support {capability}?", "node": "specs", "field": "chunk_specs"},
            {"pattern": "What do users say about {product}?", "node": "reviews", "field": "chunk_reviews"},
            {"pattern": "What are problems with {product}?", "node": "reviews", "field": "chunk_reviews"},
            {"pattern": "Is {product} good for {activity}?", "node": "use_cases", "field": "chunk_use_cases"},
            {"pattern": "Who should buy {product}?", "node": "use_cases", "field": "chunk_use_cases"},
        ]


# Question-to-Node routing (from DB Design Part 5)
QUESTION_ROUTING = [
    # Parent node questions
    {"pattern": "What is {product}?", "node": "parent", "field": "genAI_summary"},
    {"pattern": "What does {product} do?", "node": "parent", "field": "genAI_primary_function"},
    {"pattern": "What is {product} used for?", "node": "parent", "field": "genAI_use_cases"},
    {"pattern": "Who is {product} for?", "node": "parent", "field": "genAI_target_audience"},
    {"pattern": "Find {product_type} under ${price}", "node": "parent", "field": None},
    {"pattern": "Best {product_type}", "node": "parent", "field": "genAI_best_for"},
    # Child node questions
    {"pattern": "Tell me more about {product}", "node": "description", "field": "genAI_detailed_description"},
    {"pattern": "How does {feature} work?", "node": "features", "field": "genAI_technology_explained"},
    {"pattern": "What's the battery life of {product}?", "node": "specs", "field": "specs_key"},
    {"pattern": "Does {product} support {capability}?", "node": "specs", "field": "specs_key"},
    {"pattern": "What do users say about {product}?", "node": "reviews", "field": "genAI_common_praises"},
    {"pattern": "What are problems with {product}?", "node": "reviews", "field": "genAI_common_complaints"},
    {"pattern": "Is {product} good for {activity}?", "node": "use_cases", "field": "genAI_use_case_scenarios"},
    {"pattern": "Who should buy {product}?", "node": "use_cases", "field": "genAI_ideal_user_profiles"},
]

# Tool-to-Storage Mapping (from DB Design Part 13)
TOOL_STORAGE_MAPPING = {
    "vector_search": {
        "primary_storage": "qdrant_parent",
        "fields_used": ["genAI_summary_embedding"],
        "description": "Semantic product search",
    },
    "keyword_search": {
        "primary_storage": "elasticsearch",
        "fields_used": ["title", "product_type", "brand"],
        "description": "Exact/fuzzy text search",
    },
    "hybrid_search": {
        "primary_storage": "qdrant_parent + elasticsearch",
        "fields_used": ["combined"],
        "description": "General search (default)",
    },
    "search_by_section": {
        "primary_storage": "qdrant_child",
        "fields_used": ["section-specific embedding"],
        "description": "Deep questions",
    },
    "get_product_details": {
        "primary_storage": "postgresql",
        "fields_used": ["full product record"],
        "description": "Complete data retrieval",
    },
    "find_similar": {
        "primary_storage": "qdrant_parent",
        "fields_used": ["vector similarity"],
        "description": "'Similar to X' queries",
    },
    "faceted_search": {
        "primary_storage": "elasticsearch",
        "fields_used": ["category", "brand", "price"],
        "description": "Filter navigation",
    },
    "aggregate_stats": {
        "primary_storage": "elasticsearch + postgresql",
        "fields_used": ["aggregation fields"],
        "description": "Statistics queries",
    },
    "get_price_history": {
        "primary_storage": "postgresql",
        "fields_used": ["price_history table"],
        "description": "Price analysis",
    },
    "get_trending": {
        "primary_storage": "postgresql",
        "fields_used": ["categories", "brands tables"],
        "description": "Trend queries",
    },
}

# Agent-to-Tool Mapping (from DB Design Part 13 and multi-agent spec)
AGENT_TOOL_MAPPING = {
    "search": {
        "tools": ["vector_search", "keyword_search", "hybrid_search"],
        "primary_storage": "qdrant_parent",
        "scenarios": ["D1", "D2", "D3", "D4", "D5", "D6"],
    },
    "compare": {
        "tools": ["search_by_section", "get_product_details"],
        "primary_storage": "qdrant_child + postgresql",
        "scenarios": ["C1", "C2", "C3", "C4", "C5"],
    },
    "analysis": {
        "tools": ["search_by_section"],
        "primary_storage": "qdrant_child_reviews",
        "scenarios": ["A1", "A2", "A3", "A4", "A5"],
    },
    "price": {
        "tools": ["vector_search", "get_price_history", "aggregate_stats"],
        "primary_storage": "qdrant_parent + postgresql",
        "scenarios": ["P1", "P2", "P3", "P4", "P5"],
    },
    "trend": {
        "tools": ["get_trending", "aggregate_stats"],
        "primary_storage": "postgresql",
        "scenarios": ["T1", "T2", "T3", "T4", "T5"],
    },
    "recommend": {
        "tools": ["find_similar", "get_relationships"],
        "primary_storage": "qdrant_parent + postgresql",
        "scenarios": ["R1", "R2", "R3", "R4", "R5"],
    },
}

# User Scenarios (from DB Design Part 1 - Frequency × Depth Matrix)
USER_SCENARIOS = {
    # Discovery scenarios (D1-D6)
    "D1_basic_search": {"frequency": "HIGH", "depth": "SHALLOW", "node": "parent", "latency_target": 100},
    "D2_feature_search": {"frequency": "HIGH", "depth": "MEDIUM", "node": "features", "latency_target": 200},
    "D3_use_case_search": {"frequency": "HIGH", "depth": "MEDIUM", "node": "use_cases", "latency_target": 200},
    "D4_gift_finding": {"frequency": "LOW", "depth": "SHALLOW", "node": "parent", "latency_target": 500},
    "D5_brand_search": {"frequency": "HIGH", "depth": "SHALLOW", "node": "parent", "latency_target": 100},
    "D6_highly_rated": {"frequency": "HIGH", "depth": "SHALLOW", "node": "parent", "latency_target": 100},
    # Comparison scenarios (C1-C5)
    "C1_direct_compare": {"frequency": "HIGH", "depth": "DEEP", "node": "specs", "latency_target": 500},
    "C2_category_compare": {"frequency": "MEDIUM", "depth": "DEEP", "node": "specs", "latency_target": 800},
    "C3_value_analysis": {"frequency": "MEDIUM", "depth": "DEEP", "node": "reviews", "latency_target": 800},
    "C4_brand_compare": {"frequency": "MEDIUM", "depth": "DEEP", "node": "specs", "latency_target": 800},
    "C5_feature_compare": {"frequency": "MEDIUM", "depth": "MEDIUM", "node": "features", "latency_target": 500},
    # Analysis scenarios (A1-A5)
    "A1_review_summary": {"frequency": "MEDIUM", "depth": "MEDIUM", "node": "reviews", "latency_target": 400},
    "A2_sentiment_analysis": {"frequency": "MEDIUM", "depth": "MEDIUM", "node": "reviews", "latency_target": 400},
    "A3_common_issues": {"frequency": "LOW", "depth": "MEDIUM", "node": "reviews", "latency_target": 800},
    "A4_durability_check": {"frequency": "LOW", "depth": "MEDIUM", "node": "reviews", "latency_target": 800},
    "A5_quality_assessment": {"frequency": "LOW", "depth": "MEDIUM", "node": "reviews", "latency_target": 800},
    # Price scenarios (P1-P5)
    "P1_price_check": {"frequency": "MEDIUM", "depth": "SHALLOW", "node": "parent", "latency_target": 200},
    "P2_deal_finding": {"frequency": "MEDIUM", "depth": "SHALLOW", "node": "parent", "latency_target": 200},
    "P3_budget_search": {"frequency": "MEDIUM", "depth": "SHALLOW", "node": "parent", "latency_target": 200},
    "P4_price_range": {"frequency": "LOW", "depth": "SHALLOW", "node": "parent", "latency_target": 500},
    "P5_value_for_money": {"frequency": "MEDIUM", "depth": "MEDIUM", "node": "parent", "latency_target": 400},
    # Trend scenarios (T1-T5)
    "T1_category_trends": {"frequency": "LOW", "depth": "DEEP", "node": "postgresql", "latency_target": 2000},
    "T2_popular_products": {"frequency": "MEDIUM", "depth": "SHALLOW", "node": "parent", "latency_target": 200},
    "T3_brand_performance": {"frequency": "LOW", "depth": "DEEP", "node": "postgresql", "latency_target": 2000},
    "T4_emerging_products": {"frequency": "LOW", "depth": "MEDIUM", "node": "parent", "latency_target": 500},
    "T5_bestseller_analysis": {"frequency": "MEDIUM", "depth": "SHALLOW", "node": "parent", "latency_target": 200},
    # Recommendation scenarios (R1-R5)
    "R1_accessories": {"frequency": "MEDIUM", "depth": "MEDIUM", "node": "parent", "latency_target": 400},
    "R2_similar_products": {"frequency": "MEDIUM", "depth": "SHALLOW", "node": "parent", "latency_target": 200},
    "R3_alternatives": {"frequency": "MEDIUM", "depth": "MEDIUM", "node": "parent", "latency_target": 400},
    "R4_upgrade_options": {"frequency": "LOW", "depth": "MEDIUM", "node": "parent", "latency_target": 500},
    "R5_bundle_suggestions": {"frequency": "LOW", "depth": "MEDIUM", "node": "postgresql", "latency_target": 800},
}


class EvaluationDataGenerator:
    """Generate comprehensive evaluation datasets aligned with DB Design.

    Generation Flow (ensures logical consistency):
    1. Level 6: User Scenarios → LLM generates diverse user queries
    2. Level 5: Agent/Tool Selection → Derived from Level 6 + LLM variations
    3. Level 4: Storage Design → Derived from Level 5 tool mappings
    4. Level 3: Retrieval → LLM generates search query variations

    All levels follow: User Scenarios → Agents → Tools → Queries

    Pipeline Mode:
    - original: Basic pipeline without GenAI enrichment (no genAI_* fields)
    - enrich: Full pipeline with GenAI enrichment (includes genAI_* fields)
    """

    def __init__(
        self,
        ollama_url: str = "http://192.168.80.54:11434",
        llm_model: str = "gpt-oss:120b",
        llm_timeout: float = 300.0,
        llm_max_retries: int = 3,
        llm_enabled: bool = True,
        fallback_to_template: bool = True,
        pipeline_mode: str = "enrich",
    ):
        """Initialize generator with config.

        Args:
            ollama_url: Ollama service URL
            llm_model: LLM model name for query generation
            llm_timeout: Timeout for LLM calls in seconds
            llm_max_retries: Max retries for LLM calls
            llm_enabled: Whether to use LLM for generation
            fallback_to_template: Fall back to templates if LLM fails
            pipeline_mode: Pipeline mode ('original' or 'enrich')
        """
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        self.llm_max_retries = llm_max_retries
        self.llm_enabled = llm_enabled
        self.fallback_to_template = fallback_to_template
        self.pipeline_mode = pipeline_mode
        self.client: httpx.AsyncClient | None = None
        # Store generated data for cross-level consistency
        self._level6_data: list[dict] = []
        self._level5_data: list[dict] = []

        # Set mode-specific constants
        self.parent_required_fields = get_parent_required_fields(pipeline_mode)
        self.child_sections = get_child_sections(pipeline_mode)
        self.question_routing = get_question_routing(pipeline_mode)

        # Log mode configuration
        logger.info(
            "generator_initialized",
            pipeline_mode=pipeline_mode,
            genai_fields_expected=len(self.parent_required_fields.get("genai_quick", [])) > 0,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.ollama_url,
                timeout=httpx.Timeout(connect=30.0, read=self.llm_timeout, write=30.0, pool=30.0),
            )
        return self.client

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _call_llm(self, prompt: str, max_retries: int | None = None) -> str:
        """Call Ollama LLM for query generation."""
        if not self.llm_enabled:
            return ""

        max_retries = max_retries if max_retries is not None else self.llm_max_retries
        client = await self._get_client()

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    "/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 1024,
                        }
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
            except Exception as e:
                logger.warning("llm_call_failed", attempt=attempt + 1, error=str(e))
                if attempt == max_retries - 1:
                    return ""
                await asyncio.sleep(1)
        return ""

    async def _generate_query_variations(
        self,
        scenario_id: str,
        scenario_desc: str,
        product_info: dict,
        num_variations: int = 5,
    ) -> list[str]:
        """Generate diverse query variations for a scenario using LLM.

        Args:
            scenario_id: The scenario ID (e.g., D1, C1, A1)
            scenario_desc: Description of what the user wants to do
            product_info: Product context (title, brand, category, etc.)
            num_variations: Number of query variations to generate

        Returns:
            List of diverse query strings
        """
        prompt = f"""You are generating realistic user search queries for a product intelligence system.

SCENARIO: {scenario_id} - {scenario_desc}

PRODUCT CONTEXT:
- Product: {product_info.get('title', 'Unknown Product')}
- Brand: {product_info.get('brand', 'Unknown')}
- Category: {product_info.get('category', 'Electronics')}
- Type: {product_info.get('product_type', 'product')}
- Price: ${product_info.get('price', 100)}

Generate {num_variations} DIVERSE and REALISTIC user queries that someone might type when trying to accomplish this scenario.

REQUIREMENTS:
1. Vary the query style: some short (2-3 words), some detailed (full sentences)
2. Include different phrasings: questions, commands, keywords
3. Some queries should include product name, some should be generic
4. Include natural language variations (typos OK occasionally)
5. Mix formal and casual language

OUTPUT FORMAT: Return ONLY the queries, one per line, numbered 1-{num_variations}. No explanations.

QUERIES:"""

        response = await self._call_llm(prompt)

        # Parse response into list of queries
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering (1., 1), 1:, etc.)
            if line and line[0].isdigit():
                line = line.lstrip("0123456789").lstrip(".)]: ").strip()
            if line and len(line) > 3:
                queries.append(line)

        # Fallback if LLM fails
        if not queries:
            queries = [f"{scenario_desc} for {product_info.get('title', 'product')[:30]}"]

        return queries[:num_variations]

    async def _generate_agent_routing_queries(
        self,
        agent_type: str,
        tools: list[str],
        product_info: dict,
        num_queries: int = 5,
    ) -> list[dict]:
        """Generate queries that test agent routing and tool selection.

        Args:
            agent_type: The expected agent (search, compare, analysis, price, trend, recommend)
            tools: Expected tools the agent should use
            product_info: Product context
            num_queries: Number of queries to generate

        Returns:
            List of dicts with query and expected routing info
        """
        queries = []
        title = product_info.get('title', 'Unknown Product')
        brand = product_info.get('brand', 'Unknown')
        category = product_info.get('category', 'Electronics')
        product_type = product_info.get('product_type', 'product')

        # Try LLM generation if enabled
        if self.llm_enabled:
            tool_descriptions = {
                "vector_search": "semantic similarity search",
                "keyword_search": "exact text matching",
                "hybrid_search": "combined semantic + keyword search",
                "search_by_section": "search within specific sections (reviews, specs, features)",
                "get_product_details": "retrieve full product information",
                "find_similar": "find similar products",
                "get_price_history": "get price trends over time",
                "get_trending": "get trending products/categories",
                "aggregate_stats": "compute statistics across products",
                "get_relationships": "find related/bundled products",
            }

            tools_desc = ", ".join([tool_descriptions.get(t, t) for t in tools])

            prompt = f"""You are generating test queries for a multi-agent product intelligence system.

AGENT: {agent_type.upper()} Agent
TOOLS AVAILABLE: {', '.join(tools)}
TOOL PURPOSES: {tools_desc}

PRODUCT CONTEXT:
- Product: {title}
- Brand: {brand}
- Category: {category}

Generate {num_queries} user queries that should be routed to the {agent_type.upper()} agent.

REQUIREMENTS:
1. Queries must clearly require the {agent_type} agent's capabilities
2. Vary complexity: simple queries and complex multi-part queries
3. Include edge cases that might be ambiguous
4. Some should explicitly need specific tools ({', '.join(tools)})
5. Mix product-specific and category-level queries

OUTPUT FORMAT: Return ONLY the queries, one per line, numbered 1-{num_queries}. No explanations.

QUERIES:"""

            response = await self._call_llm(prompt)

            # Parse response
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    line = line.lstrip("0123456789").lstrip(".)]: ").strip()
                if line and len(line) > 3:
                    queries.append({
                        "query": line,
                        "expected_agent": agent_type,
                        "expected_tools": tools,
                    })

        # Template fallback if LLM disabled or failed
        if not queries and self.fallback_to_template:
            title_short = " ".join(title.split()[:4])
            agent_templates = {
                "search": [
                    f"Find {brand} {product_type}",
                    f"Search for {title_short}",
                    f"Looking for {product_type} in {category}",
                    f"Show me {brand} products",
                    f"I need a {product_type}",
                ],
                "compare": [
                    f"Compare {title_short} with alternatives",
                    f"What's the difference between {brand} models",
                    f"Which {product_type} is better",
                    f"Compare features of {brand} products",
                    f"Side by side comparison of {product_type}",
                ],
                "analysis": [
                    f"What do reviews say about {title_short}",
                    f"Analyze customer feedback for {brand}",
                    f"Sentiment analysis of {product_type} reviews",
                    f"Common complaints about {title_short}",
                    f"User satisfaction with {brand} {product_type}",
                ],
                "price": [
                    f"Price history for {title_short}",
                    f"Is {brand} {product_type} worth the price",
                    f"Best deals on {product_type}",
                    f"Price comparison for {brand}",
                    f"Budget options for {product_type}",
                ],
                "trend": [
                    f"Trending {product_type} products",
                    f"Popular {brand} items this month",
                    f"Best selling {product_type} in {category}",
                    f"What's hot in {category}",
                    f"Rising {product_type} brands",
                ],
                "recommend": [
                    f"Products similar to {title_short}",
                    f"Recommend {product_type} like {brand}",
                    f"What goes well with {title_short}",
                    f"Suggest alternatives to {brand}",
                    f"Related products to {product_type}",
                ],
            }
            templates = agent_templates.get(agent_type, [
                f"Help me with {title_short}",
                f"I have a question about {brand} {product_type}",
                f"Tell me about {product_type}",
            ])
            for query_text in templates[:num_queries]:
                queries.append({
                    "query": query_text,
                    "expected_agent": agent_type,
                    "expected_tools": tools,
                })

        return queries[:num_queries]

    async def _generate_retrieval_queries(
        self,
        search_type: str,
        target_node: str,
        target_section: str | None,
        product_info: dict,
        num_queries: int = 5,
    ) -> list[dict]:
        """Generate diverse search queries for retrieval evaluation.

        Args:
            search_type: Type of search (semantic, keyword, hybrid, section)
            target_node: Expected node type (parent, child)
            target_section: For child nodes, the section (reviews, specs, features, etc.)
            product_info: Product context
            num_queries: Number of queries to generate

        Returns:
            List of dicts with query and expected retrieval info
        """
        queries = []
        title = product_info.get('title', 'Unknown Product')
        brand = product_info.get('brand', 'Unknown')
        product_type = product_info.get('product_type', 'product')

        # Try LLM generation if enabled
        if self.llm_enabled:
            section_context = ""
            if target_section:
                section_hints = {
                    "reviews": "user opinions, complaints, praises, sentiment",
                    "specs": "technical specifications, dimensions, compatibility",
                    "features": "product features, capabilities, technology",
                    "use_cases": "usage scenarios, who it's for, activities",
                    "description": "product overview, what it is, primary function",
                }
                section_context = f"\nTARGET SECTION: {target_section} ({section_hints.get(target_section, '')})"

            prompt = f"""You are generating search queries to test a product retrieval system.

SEARCH TYPE: {search_type}
TARGET: {target_node} node{section_context}

PRODUCT CONTEXT:
- Product: {title}
- Brand: {brand}
- Category: {product_info.get('category', 'Electronics')}
- Type: {product_type}

Generate {num_queries} search queries optimized for {search_type} search.

REQUIREMENTS FOR {search_type.upper()} SEARCH:
{"- Use natural language, concepts, and synonyms" if search_type == "semantic" else ""}
{"- Use exact terms, brand names, model numbers" if search_type == "keyword" else ""}
{"- Mix natural language with specific terms" if search_type == "hybrid" else ""}
{"- Questions targeting " + (target_section or "specific") + " information" if search_type == "section" else ""}

QUERY DIFFICULTY MIX:
- 2 EASY: Direct, clear queries
- 2 MEDIUM: Require some inference
- 1 HARD: Ambiguous or complex

OUTPUT FORMAT: Return queries with difficulty, one per line:
[EASY] query text
[MEDIUM] query text
[HARD] query text

QUERIES:"""

            response = await self._call_llm(prompt)

            # Parse response with difficulty
            for line in response.strip().split("\n"):
                line = line.strip()
                difficulty = "medium"
                if "[EASY]" in line.upper():
                    difficulty = "easy"
                    line = line.upper().replace("[EASY]", "").strip()
                elif "[MEDIUM]" in line.upper():
                    difficulty = "medium"
                    line = line.upper().replace("[MEDIUM]", "").strip()
                elif "[HARD]" in line.upper():
                    difficulty = "hard"
                    line = line.upper().replace("[HARD]", "").strip()

                # Clean up numbering
                if line and line[0].isdigit():
                    line = line.lstrip("0123456789").lstrip(".)]: ").strip()

                if line and len(line) > 3:
                    queries.append({
                        "query": line,
                        "search_type": search_type,
                        "target_node": target_node,
                        "target_section": target_section,
                        "difficulty": difficulty,
                    })

        # Template fallback if LLM disabled or failed
        if not queries and self.fallback_to_template:
            title_short = " ".join(title.split()[:4])
            section_templates = {
                "reviews": [
                    (f"What do users say about {title_short}", "easy"),
                    (f"Are there complaints about {brand} {product_type}", "medium"),
                    (f"Customer satisfaction with {title_short}", "hard"),
                ],
                "specs": [
                    (f"What are the specifications of {title_short}", "easy"),
                    (f"{brand} {product_type} dimensions", "medium"),
                    (f"Technical details and compatibility", "hard"),
                ],
                "features": [
                    (f"What features does {title_short} have", "easy"),
                    (f"Key capabilities of {brand} {product_type}", "medium"),
                    (f"What technology does it use", "hard"),
                ],
                "use_cases": [
                    (f"Who should buy {title_short}", "easy"),
                    (f"Is {brand} {product_type} good for travel", "medium"),
                    (f"What problems does this solve", "hard"),
                ],
                "description": [
                    (f"Tell me about {title_short}", "easy"),
                    (f"What is {brand} {product_type}", "medium"),
                    (f"Explain how this works", "hard"),
                ],
            }
            templates = section_templates.get(target_section, [
                (f"Search for {title_short}", "easy"),
                (f"Find {brand} {product_type}", "medium"),
                (f"Looking for products like {title_short}", "hard"),
            ])
            for query_text, difficulty in templates[:num_queries]:
                queries.append({
                    "query": query_text,
                    "search_type": search_type,
                    "target_node": target_node,
                    "target_section": target_section,
                    "difficulty": difficulty,
                })

        return queries[:num_queries]

    def _safe_get(self, row: pd.Series, field: str, default: Any = None) -> Any:
        """Safely get a field value from a row."""
        value = row.get(field, default)
        if pd.isna(value):
            return default
        return value

    def _is_valid_product(self, product: pd.Series, required_fields: list[str] | None = None) -> bool:
        """Check if product has valid (non-NaN) values for required fields.

        Args:
            product: Product row
            required_fields: List of field names to check. Defaults to ['asin', 'title']

        Returns:
            True if all required fields have non-NaN values
        """
        if required_fields is None:
            required_fields = ['asin', 'title']

        for field in required_fields:
            value = product.get(field)
            if value is None or pd.isna(value):
                return False
            # Also check for empty strings
            if isinstance(value, str) and not value.strip():
                return False
        return True

    def _get_product_text(self, product: pd.Series) -> str:
        """Get combined text for embedding."""
        parts = []
        if pd.notna(product.get("title")):
            parts.append(str(product.get("title")))
        if pd.notna(product.get("brand")):
            parts.append(f"Brand: {product.get('brand')}")
        if pd.notna(product.get("product_type")):
            parts.append(f"Type: {product.get('product_type')}")
        if pd.notna(product.get("genAI_summary")):
            parts.append(str(product.get("genAI_summary"))[:300])
        elif pd.notna(product.get("chunk_description")):
            parts.append(str(product.get("chunk_description"))[:300])
        if pd.notna(product.get("chunk_features")):
            parts.append(str(product.get("chunk_features"))[:200])
        return " ".join(parts)

    def _get_title_keywords(self, title: str, num_words: int = 3, skip_brand: str = "") -> str:
        """Extract unique keywords from title, optionally skipping brand name."""
        words = []
        for w in str(title).split():
            # Skip if it's the brand name (avoid duplication)
            if skip_brand and w.lower() == skip_brand.lower():
                continue
            # Include words > 3 chars that are alphabetic
            if len(w) > 3 and w.isalpha():
                words.append(w)
        return " ".join(words[:num_words])

    def _extract_model_numbers(self, title: str) -> list[str]:
        """Extract model numbers and part numbers from title.

        Looks for patterns like: TRX500FA, B07Z1QTH89, 31200-HA7-315, WW3D, A19, etc.
        """
        import re
        # Pattern for model numbers: alphanumeric with optional hyphens, 3+ chars
        # Must contain at least one digit to be a model number
        pattern = r'\b([A-Z0-9][A-Z0-9\-]{2,}[A-Z0-9])\b'
        matches = re.findall(pattern, title.upper())
        # Filter to only those with at least one digit (to avoid plain words)
        return [m for m in matches if any(c.isdigit() for c in m)]

    def _get_unique_title_terms(self, title: str, brand: str = "") -> dict:
        """Extract unique identifying terms from title.

        Returns dict with:
        - brand: Brand name
        - model_numbers: List of model/part numbers
        - product_type: Main product type (first few content words)
        - specific_terms: Unique identifying terms (from middle/end of title)
        - full_short: First 50 chars of title
        """
        words = title.split()

        # Extract model numbers
        model_numbers = self._extract_model_numbers(title)

        # Get product type (first 2-3 content words after brand)
        content_words = []
        brand_lower = brand.lower() if brand else ""
        for w in words:
            if w.lower() == brand_lower:
                continue
            if len(w) > 2 and w.isalpha():
                content_words.append(w)
            if len(content_words) >= 3:
                break
        product_type = " ".join(content_words[:3])

        # Get specific terms from middle/end of title (unique identifiers)
        # Look for words that are likely specific: proper nouns, longer words
        specific_terms = []
        for w in words[3:]:  # Skip first 3 words
            w_clean = w.strip("[](),")
            if len(w_clean) > 4 and w_clean[0].isupper() and w_clean.isalpha():
                if w_clean.lower() not in ["with", "from", "this", "that", "these", "those", "pack", "piece"]:
                    specific_terms.append(w_clean)
            if len(specific_terms) >= 3:
                break

        return {
            "brand": brand,
            "model_numbers": model_numbers,
            "product_type": product_type,
            "specific_terms": specific_terms,
            "full_short": title[:50] if len(title) > 50 else title,
        }

    # =========================================================================
    # Level 1: Embedding Quality Evaluation Data
    # =========================================================================
    def generate_level1_embedding_data(self, df: pd.DataFrame, num_samples: int = 100) -> list[dict]:
        """Generate Level 1: Embedding Quality evaluation data.

        Tests:
        - Similarity pairs (same/different category)
        - Triplets (anchor, positive, negative)
        - Clustering by category
        - Product type coherence (NEW)
        """
        eval_data = []
        categories = df['category_level1'].dropna().unique()

        # 1. Similarity pairs by category
        for category in categories[:10]:
            cat_df = df[df['category_level1'] == category]
            if len(cat_df) < 2:
                continue

            sampled = cat_df.sample(n=min(10, len(cat_df)))
            for i, (_, p1) in enumerate(sampled.iterrows()):
                for j, (_, p2) in enumerate(sampled.iterrows()):
                    if i >= j:
                        continue
                    eval_data.append({
                        "type": "similarity_pair",
                        "level": 1,
                        "expected_similarity": "high",
                        "product_1": {
                            "asin": p1.get("asin"),
                            "title": p1.get("title"),
                            "category": p1.get("category_level1"),
                            "product_type": self._safe_get(p1, "product_type"),
                            "text": self._get_product_text(p1),
                        },
                        "product_2": {
                            "asin": p2.get("asin"),
                            "title": p2.get("title"),
                            "category": p2.get("category_level1"),
                            "product_type": self._safe_get(p2, "product_type"),
                            "text": self._get_product_text(p2),
                        },
                    })

        # 2. Dissimilar pairs (different categories)
        if len(categories) >= 2:
            for i in range(min(50, num_samples // 2)):
                cat1, cat2 = random.sample(list(categories), 2)
                cat1_df = df[df['category_level1'] == cat1]
                cat2_df = df[df['category_level1'] == cat2]
                if len(cat1_df) == 0 or len(cat2_df) == 0:
                    continue
                p1 = cat1_df.sample(n=1).iloc[0]
                p2 = cat2_df.sample(n=1).iloc[0]
                eval_data.append({
                    "type": "similarity_pair",
                    "level": 1,
                    "expected_similarity": "low",
                    "product_1": {
                        "asin": p1.get("asin"),
                        "title": p1.get("title"),
                        "category": p1.get("category_level1"),
                        "product_type": self._safe_get(p1, "product_type"),
                        "text": self._get_product_text(p1),
                    },
                    "product_2": {
                        "asin": p2.get("asin"),
                        "title": p2.get("title"),
                        "category": p2.get("category_level1"),
                        "product_type": self._safe_get(p2, "product_type"),
                        "text": self._get_product_text(p2),
                    },
                })

        # 3. Triplets (anchor, positive, negative)
        for i in range(min(50, num_samples // 2)):
            category = random.choice(list(categories))
            cat_df = df[df['category_level1'] == category]
            other_cats = [c for c in categories if c != category]
            if len(cat_df) < 2 or len(other_cats) == 0:
                continue

            samples = cat_df.sample(n=2)
            anchor = samples.iloc[0]
            positive = samples.iloc[1]
            neg_cat = random.choice(other_cats)
            neg_df = df[df['category_level1'] == neg_cat]
            if len(neg_df) == 0:
                continue
            negative = neg_df.sample(n=1).iloc[0]

            eval_data.append({
                "type": "triplet",
                "level": 1,
                "anchor": {
                    "asin": anchor.get("asin"),
                    "title": anchor.get("title"),
                    "category": anchor.get("category_level1"),
                    "text": self._get_product_text(anchor),
                },
                "positive": {
                    "asin": positive.get("asin"),
                    "title": positive.get("title"),
                    "category": positive.get("category_level1"),
                    "text": self._get_product_text(positive),
                },
                "negative": {
                    "asin": negative.get("asin"),
                    "title": negative.get("title"),
                    "category": negative.get("category_level1"),
                    "text": self._get_product_text(negative),
                },
            })

        # 4. Clustering evaluation data
        cluster_samples = df.sample(n=min(200, len(df)))
        for _, row in cluster_samples.iterrows():
            eval_data.append({
                "type": "clustering",
                "level": 1,
                "asin": row.get("asin"),
                "text": self._get_product_text(row),
                "true_cluster": row.get("category_level1"),
                "sub_cluster": self._safe_get(row, "category_level2"),
                "product_type": self._safe_get(row, "product_type"),
            })

        # 5. NEW: Product type coherence tests
        if "product_type" in df.columns:
            product_types = df['product_type'].dropna().unique()
            for pt in product_types[:10]:
                pt_df = df[df['product_type'] == pt]
                if len(pt_df) < 2:
                    continue
                sampled = pt_df.sample(n=min(5, len(pt_df)))
                eval_data.append({
                    "type": "product_type_coherence",
                    "level": 1,
                    "product_type": pt,
                    "expected_similarity": "high",
                    "products": [
                        {
                            "asin": row.get("asin"),
                            "title": row.get("title"),
                            "text": self._get_product_text(row),
                        }
                        for _, row in sampled.iterrows()
                    ],
                })

        return eval_data

    # =========================================================================
    # Level 2: Indexing Correctness Evaluation Data
    # =========================================================================
    def generate_level2_indexing_data(self, df: pd.DataFrame, num_samples: int = 100) -> list[dict]:
        """Generate Level 2: Indexing Correctness evaluation data.

        Tests:
        - Parent-child hierarchy (1 parent + 5 children per product)
        - Parent field presence (all required fields including GenAI)
        - Child section field presence (section-specific GenAI fields)
        - Child inherited fields (filter fields from parent)
        - Filter index coverage
        """
        eval_data = []

        # 1. Hierarchy tests (1 parent + 5 children)
        for _, row in df.sample(n=min(num_samples, len(df))).iterrows():
            eval_data.append({
                "type": "hierarchy",
                "level": 2,
                "parent_asin": row.get("asin"),
                "expected_structure": {
                    "parent_count": 1,
                    "child_count": 5,
                    "child_sections": list(CHILD_SECTIONS.keys()),
                },
                "expected_children": [
                    {
                        "section": section,
                        "has_content": any(
                            pd.notna(row.get(f"chunk_{section}")) or pd.notna(row.get(field))
                            for field in config["genai_fields"]
                        ),
                    }
                    for section, config in CHILD_SECTIONS.items()
                ],
            })

        # 2. Parent field presence tests (GenAI fields)
        for _, row in df.sample(n=min(num_samples // 2, len(df))).iterrows():
            field_presence = {}
            for group, fields in PARENT_REQUIRED_FIELDS.items():
                field_presence[group] = {
                    field: pd.notna(row.get(field)) or pd.notna(row.get(field.replace("genAI_", "")))
                    for field in fields
                }

            eval_data.append({
                "type": "parent_field_presence",
                "level": 2,
                "asin": row.get("asin"),
                "required_fields": PARENT_REQUIRED_FIELDS,
                "field_presence": field_presence,
                "critical_fields": [
                    "asin", "title", "price", "category_level1",
                    "genAI_summary", "genAI_primary_function", "genAI_best_for"
                ],
            })

        # 3. Child section field presence tests
        for _, row in df.sample(n=min(num_samples // 2, len(df))).iterrows():
            for section, config in CHILD_SECTIONS.items():
                eval_data.append({
                    "type": "child_field_presence",
                    "level": 2,
                    "parent_asin": row.get("asin"),
                    "section": section,
                    "expected_genai_fields": config["genai_fields"],
                    "field_presence": {
                        field: pd.notna(row.get(field))
                        for field in config["genai_fields"]
                    },
                })

        # 4. Child inherited fields tests
        for _, row in df.sample(n=min(num_samples // 4, len(df))).iterrows():
            eval_data.append({
                "type": "child_inheritance",
                "level": 2,
                "parent_asin": row.get("asin"),
                "expected_inherited_fields": CHILD_INHERITED_FIELDS,
                "parent_values": {
                    "category_level1": self._safe_get(row, "category_level1"),
                    "brand": self._safe_get(row, "brand"),
                    "price": self._safe_get(row, "price"),
                    "stars": self._safe_get(row, "stars"),
                },
            })

        # 5. Filter index tests
        filter_tests = [
            {"field": "category_level1", "type": "keyword"},
            {"field": "category_level2", "type": "keyword"},
            {"field": "category_level3", "type": "keyword"},
            {"field": "brand", "type": "keyword"},
            {"field": "product_type", "type": "keyword"},
            {"field": "price", "type": "range"},
            {"field": "stars", "type": "range"},
            {"field": "is_best_seller", "type": "boolean"},
            {"field": "availability", "type": "keyword"},
        ]

        for test in filter_tests:
            field = test["field"]
            if field not in df.columns:
                continue

            if test["type"] == "keyword":
                values = df[field].dropna().unique()[:10]
                for value in values:
                    count = len(df[df[field] == value])
                    eval_data.append({
                        "type": "filter_test",
                        "level": 2,
                        "field": field,
                        "filter_type": "keyword",
                        "filter": {field: str(value)},
                        "expected_count_approx": count,
                    })
            elif test["type"] == "range":
                if field == "price":
                    price_ranges = [(0, 50), (50, 100), (100, 500), (500, 2000)]
                    for min_val, max_val in price_ranges:
                        count = len(df[(df['price'] >= min_val) & (df['price'] <= max_val)])
                        eval_data.append({
                            "type": "filter_test",
                            "level": 2,
                            "field": field,
                            "filter_type": "range",
                            "filter": {"price_min": min_val, "price_max": max_val},
                            "expected_count_approx": count,
                        })
                elif field == "stars":
                    for min_rating in [3.0, 4.0, 4.5]:
                        count = len(df[df['stars'] >= min_rating])
                        eval_data.append({
                            "type": "filter_test",
                            "level": 2,
                            "field": field,
                            "filter_type": "range",
                            "filter": {"min_rating": min_rating},
                            "expected_count_approx": count,
                        })
            elif test["type"] == "boolean":
                for value in [True, False]:
                    count = len(df[df[field] == value]) if field in df.columns else 0
                    eval_data.append({
                        "type": "filter_test",
                        "level": 2,
                        "field": field,
                        "filter_type": "boolean",
                        "filter": {field: value},
                        "expected_count_approx": count,
                    })

        return eval_data

    # =========================================================================
    # Level 3: Retrieval Performance Evaluation Data (LLM-Enhanced)
    # =========================================================================
    async def generate_level3_retrieval_data_async(self, df: pd.DataFrame, num_samples: int = 300) -> list[dict]:
        """Generate Level 3: Retrieval Performance evaluation data using LLM.

        Uses LLM to generate diverse search queries for testing:
        - Keyword search (exact matching)
        - Semantic search (concept-based)
        - Hybrid search (combined)
        - Section-targeted search (child nodes)

        Derives from Level 5/6 data for cross-level consistency.
        """
        eval_data = []
        used_queries = set()
        logger.info("level3_llm_generation_start")

        # Filter for products with valid critical fields before sampling
        valid_df = df[
            df['asin'].notna() &
            df['title'].notna() &
            df['price'].notna()
        ]
        if len(valid_df) == 0:
            logger.warning("no_valid_products_for_level3", reason="All products have NaN in asin, title, or price")
            return eval_data

        sample_products = valid_df.sample(n=min(30, len(valid_df))).to_dict('records')
        samples_per_type = num_samples // 5

        # 1. Keyword Search - LLM generates variations
        logger.info("generating_keyword_search_queries")
        keyword_queries = await self._generate_keyword_search_queries(sample_products, samples_per_type)
        eval_data.extend(keyword_queries)

        # 2. Semantic Search - LLM generates natural language queries
        logger.info("generating_semantic_search_queries")
        semantic_queries = await self._generate_semantic_search_queries(sample_products, samples_per_type)
        eval_data.extend(semantic_queries)

        # 3. Hybrid Search - LLM generates mixed queries
        logger.info("generating_hybrid_search_queries")
        hybrid_queries = await self._generate_hybrid_search_queries(sample_products, samples_per_type)
        eval_data.extend(hybrid_queries)

        # 4. Section Search - LLM generates section-targeted queries
        # ONLY in enrich mode - original mode has no child nodes for section search
        if self.pipeline_mode == "enrich":
            logger.info("generating_section_search_queries")
            section_queries = await self._generate_section_search_queries(sample_products, samples_per_type)
            eval_data.extend(section_queries)
        else:
            logger.info("skipping_section_search_queries", reason="original mode has no child nodes")

        # 5. Derive from Level 5/6 for consistency
        if self._level5_data or self._level6_data:
            logger.info("deriving_retrieval_queries_from_higher_levels")
            derived_queries = self._derive_retrieval_from_higher_levels()
            eval_data.extend(derived_queries)

        logger.info("level3_generation_complete", total_queries=len(eval_data))
        return eval_data

    async def _generate_keyword_search_queries(
        self,
        sample_products: list[dict],
        num_queries: int,
    ) -> list[dict]:
        """Generate keyword search test queries using LLM or templates."""
        eval_data = []

        for product in random.sample(sample_products, min(num_queries // 3, len(sample_products))):
            title = str(product.get("title", ""))
            brand = product.get("brand", "")
            asin = product.get("asin")

            queries = []

            # Try LLM generation if enabled
            if self.llm_enabled:
                # Mode-aware prompt - original mode may not have brand
                brand_info = f"BRAND: {brand}" if brand and brand != "nan" else "BRAND: (not available)"
                prompt = f"""Generate 5 KEYWORD search queries for product retrieval testing.

PRODUCT: {title}
{brand_info}

REQUIREMENTS:
- Use exact product names, model numbers from the title
- Include partial matches (title keywords)
- Include misspellings or typos (1-2 queries)
- Mix short (2-3 words) and longer queries
- If brand is not available, focus on title-based queries

OUTPUT: Return 5 queries with difficulty level, one per line:
[EASY] exact match query
[MEDIUM] partial match query
[HARD] misspelled/typo query

QUERIES:"""

                response = await self._call_llm(prompt)
                queries = self._parse_difficulty_queries(response)

            # Template fallback if LLM disabled or failed
            # NOTE: Based on experiments (Jan 2026), keyword search excels for:
            # - brand+model queries: 87.7% R@1 (vs semantic 43.1%)
            # - first_words queries: 76% R@1 (vs semantic 62%)
            if not queries and self.fallback_to_template:
                # Extract unique identifying terms
                terms = self._get_unique_title_terms(title, brand)
                model_numbers = terms["model_numbers"]
                specific_terms = terms["specific_terms"]
                product_type = terms["product_type"]

                queries = []

                # Mode-aware query generation
                # Original mode: no brand field in Qdrant/ES, use title-based queries only
                # Enrich mode: full field set including brand
                has_brand = brand and brand != "nan" and self.pipeline_mode == "enrich"

                # EASY - TYPE: brand_model (highest priority for keyword search)
                # Model numbers are most unique identifiers - keyword excels here
                if model_numbers:
                    if has_brand:
                        queries.append({
                            "query": f"{brand} {model_numbers[0]}",
                            "difficulty": "easy",
                            "query_type": "brand_model"
                        })
                    else:
                        # Original mode: use model number with title words instead
                        queries.append({
                            "query": f"{model_numbers[0]} {' '.join(title.split()[:2])}",
                            "difficulty": "easy",
                            "query_type": "model_title"
                        })
                    # Also add just model number for specific searches
                    if len(model_numbers) > 1:
                        queries.append({
                            "query": f"{model_numbers[0]} {model_numbers[1]}",
                            "difficulty": "easy",
                            "query_type": "model_numbers"
                        })

                # EASY - TYPE: first_words (partial title match)
                # First 6-8 words of title are highly specific
                first_words = " ".join(title.split()[:7])
                queries.append({
                    "query": first_words,
                    "difficulty": "easy",
                    "query_type": "first_words"
                })

                # EASY - TYPE: short_title (first 3-4 meaningful words)
                # Very effective for keyword search
                short_title = " ".join(title.split()[:4])
                queries.append({
                    "query": short_title,
                    "difficulty": "easy",
                    "query_type": "short_title"
                })

                # MEDIUM - TYPE: brand_type_specific or title_keywords
                if has_brand:
                    if specific_terms:
                        queries.append({
                            "query": f"{brand} {product_type} {specific_terms[0]}",
                            "difficulty": "medium",
                            "query_type": "brand_type_specific"
                        })
                    else:
                        title_kw = self._get_title_keywords(title, 3, skip_brand=brand)
                        queries.append({
                            "query": f"{brand} {title_kw}",
                            "difficulty": "medium",
                            "query_type": "brand_keywords"
                        })
                else:
                    # Original mode: use category + title keywords
                    category = product.get("category_level1", "")
                    title_kw = self._get_title_keywords(title, 4)
                    if category and category != "nan":
                        queries.append({
                            "query": f"{category} {title_kw}",
                            "difficulty": "medium",
                            "query_type": "category_keywords"
                        })
                    else:
                        queries.append({
                            "query": title_kw,
                            "difficulty": "medium",
                            "query_type": "title_keywords"
                        })

                # HARD - TYPE: generic (harder to match specific product)
                if product_type:
                    queries.append({
                        "query": f"{product_type}",
                        "difficulty": "hard",
                        "query_type": "generic"
                    })
                else:
                    # Fallback: extract product type from title
                    queries.append({
                        "query": " ".join(title.split()[-3:]),
                        "difficulty": "hard",
                        "query_type": "generic_title_end"
                    })

            for idx, q in enumerate(queries[:6]):  # Allow 6 queries for more coverage
                eval_data.append({
                    "query_id": f"L3_keyword_{asin}_{idx}",
                    "level": 3,
                    "type": "parent_search",
                    "search_type": "keyword",
                    "query_text": q["query"],
                    "target_node": "parent",
                    "difficulty": q["difficulty"],
                    "query_type": q.get("query_type", "unknown"),  # Track query type for analysis
                    "source_asin": asin,
                    "relevant_asins": [{"asin": asin, "relevance_score": 3}],
                    "expected_in_top_k": {
                        "top_5": q["difficulty"] == "easy",
                        "top_10": q["difficulty"] in ["easy", "medium"],
                        "top_20": True,
                    },
                })

        return eval_data

    async def _generate_semantic_search_queries(
        self,
        sample_products: list[dict],
        num_queries: int,
    ) -> list[dict]:
        """Generate semantic search test queries using LLM or templates."""
        eval_data = []

        for product in random.sample(sample_products, min(num_queries // 3, len(sample_products))):
            title = str(product.get("title", ""))
            brand = product.get("brand", "")
            product_type = product.get("product_type", "product")
            genai_summary = product.get("genAI_summary", "")[:200]
            asin = product.get("asin")

            queries = []

            # Try LLM generation if enabled
            if self.llm_enabled:
                prompt = f"""Generate 5 SEMANTIC (natural language) search queries for product retrieval testing.

PRODUCT: {title}
TYPE: {product_type}
SUMMARY: {genai_summary}

REQUIREMENTS:
- Use natural language, NOT exact product names
- Describe what the user WANTS or NEEDS
- Include use-case based queries ("headphones for working out")
- Include feature-based queries ("noise cancelling wireless")
- Vary from simple to complex

OUTPUT: Return 5 queries with difficulty level:
[EASY] direct description query
[MEDIUM] use-case based query
[HARD] abstract/conceptual query

QUERIES:"""

                response = await self._call_llm(prompt)
                queries = self._parse_difficulty_queries(response)

            # Template fallback if LLM disabled or failed
            # NOTE: Semantic search works better for natural language queries
            # but struggles with brand+model queries (43.1% R@1 vs keyword 87.7%)
            if not queries and self.fallback_to_template:
                # Extract unique identifying terms for more specific queries
                terms = self._get_unique_title_terms(title, brand)
                model_numbers = terms["model_numbers"]
                specific_terms = terms["specific_terms"]
                title_kw = self._get_title_keywords(title, 4, skip_brand=brand)
                category = product.get("category_level1", "")

                # Mode-aware: check if brand is available (enrich mode only)
                has_brand = brand and brand != "nan" and self.pipeline_mode == "enrich"
                has_product_type = product_type and product_type != "product" and self.pipeline_mode == "enrich"

                queries = []

                # EASY - TYPE: natural_specific (natural language with specific terms)
                if model_numbers:
                    if has_brand:
                        queries.append({
                            "query": f"Looking for {brand} {model_numbers[0]} {product_type if has_product_type else ''}".strip(),
                            "difficulty": "easy",
                            "query_type": "natural_specific"
                        })
                    else:
                        queries.append({
                            "query": f"Looking for {model_numbers[0]} {title_kw}",
                            "difficulty": "easy",
                            "query_type": "natural_model"
                        })
                else:
                    if has_brand:
                        queries.append({
                            "query": f"I need a {brand} {title_kw}",
                            "difficulty": "easy",
                            "query_type": "natural_brand"
                        })
                    else:
                        queries.append({
                            "query": f"I need a {title_kw}",
                            "difficulty": "easy",
                            "query_type": "natural_title"
                        })

                # EASY - TYPE: natural_brand_type or natural_category
                if has_brand:
                    if specific_terms:
                        queries.append({
                            "query": f"Find me {brand} {product_type if has_product_type else ''} {specific_terms[0]}".strip(),
                            "difficulty": "easy",
                            "query_type": "natural_specific"
                        })
                    else:
                        queries.append({
                            "query": f"Find me {brand} {product_type if has_product_type else title_kw}",
                            "difficulty": "easy",
                            "query_type": "natural_brand_type"
                        })
                else:
                    # Original mode: use category + title keywords
                    if category and category != "nan":
                        queries.append({
                            "query": f"Find me a {category} {title_kw}",
                            "difficulty": "easy",
                            "query_type": "natural_category"
                        })
                    else:
                        queries.append({
                            "query": f"Find me a good {title_kw}",
                            "difficulty": "easy",
                            "query_type": "natural_generic"
                        })

                # MEDIUM - TYPE: use_case (natural language use case queries)
                if has_brand and has_product_type:
                    queries.append({
                        "query": f"Best {brand} {product_type} for my needs",
                        "difficulty": "medium",
                        "query_type": "use_case"
                    })
                else:
                    queries.append({
                        "query": f"Best {category if category else 'product'} {title_kw} for my needs",
                        "difficulty": "medium",
                        "query_type": "use_case"
                    })
                queries.append({
                    "query": f"I'm looking for something like {terms['full_short']}",
                    "difficulty": "medium",
                    "query_type": "similar_to"
                })

                # HARD - TYPE: generic_question (vague queries)
                if has_product_type:
                    queries.append({
                        "query": f"What {product_type} would you recommend",
                        "difficulty": "hard",
                        "query_type": "generic_question"
                    })
                else:
                    queries.append({
                        "query": f"What {category if category else 'product'} would you recommend",
                        "difficulty": "hard",
                        "query_type": "generic_question"
                    })

            for idx, q in enumerate(queries[:5]):
                eval_data.append({
                    "query_id": f"L3_semantic_{asin}_{idx}",
                    "level": 3,
                    "type": "parent_search",
                    "search_type": "semantic",
                    "query_text": q["query"],
                    "target_node": "parent",
                    "difficulty": q["difficulty"],
                    "query_type": q.get("query_type", "unknown"),  # Track query type for analysis
                    "source_asin": asin,
                    "relevant_asins": [{"asin": asin, "relevance_score": 3}],
                    "expected_in_top_k": {
                        "top_5": q["difficulty"] == "easy",
                        "top_10": q["difficulty"] in ["easy", "medium"],
                        "top_20": True,
                    },
                })

        return eval_data

    async def _generate_hybrid_search_queries(
        self,
        sample_products: list[dict],
        num_queries: int,
    ) -> list[dict]:
        """Generate hybrid search test queries using LLM or templates."""
        eval_data = []

        for product in random.sample(sample_products, min(num_queries // 3, len(sample_products))):
            # Skip products with NaN in critical fields
            title = product.get("title")
            asin = product.get("asin")
            price = product.get("price")
            if not title or pd.isna(title) or not asin or pd.isna(asin) or price is None or pd.isna(price):
                continue

            title = str(title)
            brand = product.get("brand", "") or ""
            product_type = product.get("product_type", "product") or "product"
            price = int(price)

            queries = []

            # Try LLM generation if enabled
            if self.llm_enabled:
                prompt = f"""Generate 5 HYBRID search queries that combine keywords with natural language.

PRODUCT: {title}
BRAND: {brand}
TYPE: {product_type}
PRICE: ${price}

REQUIREMENTS:
- Combine brand/product names WITH natural language
- Include filter-like queries ("{brand} headphones under $300")
- Mix specific terms with descriptive language
- Include comparison-style queries

OUTPUT: Return 5 queries with difficulty level:
[EASY] brand + simple description
[MEDIUM] brand + features + constraints
[HARD] complex multi-part query

QUERIES:"""

                response = await self._call_llm(prompt)
                queries = self._parse_difficulty_queries(response)

            # Template fallback if LLM disabled or failed
            # NOTE: Hybrid search combines keyword + semantic
            # Based on experiments, hybrid with keyword priority achieves best MRR (0.9126)
            if not queries and self.fallback_to_template:
                # Extract unique identifying terms
                terms = self._get_unique_title_terms(title, brand)
                model_numbers = terms["model_numbers"]
                specific_terms = terms["specific_terms"]
                title_kw = self._get_title_keywords(title, 3, skip_brand=brand)
                category = product.get("category_level1", "")

                # Mode-aware: check if brand is available (enrich mode only)
                has_brand = brand and brand != "nan" and self.pipeline_mode == "enrich"
                has_product_type = product_type and product_type != "product" and self.pipeline_mode == "enrich"

                queries = []

                # EASY - TYPE: brand_model or title_model (works great with hybrid_kw_priority)
                if model_numbers:
                    if has_brand:
                        queries.append({
                            "query": f"{brand} {model_numbers[0]}",
                            "difficulty": "easy",
                            "query_type": "brand_model"
                        })
                    else:
                        queries.append({
                            "query": f"{model_numbers[0]} {' '.join(title.split()[:2])}",
                            "difficulty": "easy",
                            "query_type": "model_title"
                        })
                else:
                    if has_brand:
                        queries.append({
                            "query": f"{brand} {title_kw}",
                            "difficulty": "easy",
                            "query_type": "brand_keywords"
                        })
                    else:
                        queries.append({
                            "query": title_kw,
                            "difficulty": "easy",
                            "query_type": "title_keywords"
                        })

                # EASY - TYPE: first_words (partial title match)
                first_words = " ".join(title.split()[:7])
                queries.append({
                    "query": first_words,
                    "difficulty": "easy",
                    "query_type": "first_words"
                })

                # EASY - TYPE: exact_title (short form)
                short_title = " ".join(title.split()[:4])
                queries.append({
                    "query": short_title,
                    "difficulty": "easy",
                    "query_type": "short_title"
                })

                # MEDIUM - TYPE: brand_type_price or category_type_price (natural constraint query)
                if has_brand and has_product_type:
                    queries.append({
                        "query": f"Best {brand} {terms['product_type']} under ${int(price * 1.5)}",
                        "difficulty": "medium",
                        "query_type": "brand_type_price"
                    })
                else:
                    cat_str = category if category and category != "nan" else "product"
                    queries.append({
                        "query": f"Best {cat_str} {title_kw} under ${int(price * 1.5)}",
                        "difficulty": "medium",
                        "query_type": "category_type_price"
                    })

                # MEDIUM - TYPE: brand_keywords_specific or title_specific
                if specific_terms:
                    if has_brand:
                        queries.append({
                            "query": f"{brand} {title_kw} {specific_terms[0]}",
                            "difficulty": "medium",
                            "query_type": "brand_keywords_specific"
                        })
                    else:
                        queries.append({
                            "query": f"{title_kw} {specific_terms[0]}",
                            "difficulty": "medium",
                            "query_type": "title_specific"
                        })
                else:
                    if has_brand:
                        queries.append({
                            "query": f"{brand} {title_kw} with good reviews",
                            "difficulty": "medium",
                            "query_type": "brand_keywords_quality"
                        })
                    else:
                        queries.append({
                            "query": f"{title_kw} with good reviews",
                            "difficulty": "medium",
                            "query_type": "title_keywords_quality"
                        })

                # HARD - TYPE: generic_comparison (harder to match specific product)
                pt = terms['product_type'] if has_product_type else (category if category and category != "nan" else "product")
                queries.append({
                    "query": f"Compare {pt} features and price",
                    "difficulty": "hard",
                    "query_type": "generic_comparison"
                })

            for idx, q in enumerate(queries[:6]):  # Allow 6 queries for more coverage
                eval_data.append({
                    "query_id": f"L3_hybrid_{asin}_{idx}",
                    "level": 3,
                    "type": "parent_search",
                    "search_type": "hybrid",
                    "query_text": q["query"],
                    "target_node": "parent",
                    "difficulty": q["difficulty"],
                    "query_type": q.get("query_type", "unknown"),  # Track query type for analysis
                    "source_asin": asin,
                    "relevant_asins": [{"asin": asin, "relevance_score": 3}],
                    "expected_in_top_k": {
                        "top_5": q["difficulty"] == "easy",
                        "top_10": True,
                        "top_20": True,
                    },
                })

        return eval_data

    async def _generate_section_search_queries(
        self,
        sample_products: list[dict],
        num_queries: int,
    ) -> list[dict]:
        """Generate section-targeted search queries using LLM."""
        eval_data = []
        sections = ["reviews", "specs", "features", "use_cases", "description"]

        for section in sections:
            section_hints = {
                "reviews": "user opinions, complaints, praises, satisfaction",
                "specs": "technical specifications, dimensions, weight, compatibility",
                "features": "product features, capabilities, technology",
                "use_cases": "usage scenarios, who it's for, activities",
                "description": "product overview, what it is, primary function",
            }

            for product in random.sample(sample_products, min(num_queries // (len(sections) * 3), len(sample_products))):
                title = str(product.get("title", ""))
                title_short = " ".join(title.split()[:4])
                asin = product.get("asin")

                # Generate section-specific queries using LLM
                queries = await self._generate_retrieval_queries(
                    search_type="section",
                    target_node="child",
                    target_section=section,
                    product_info={
                        "title": title,
                        "brand": product.get("brand", ""),
                        "category": product.get("category_level1", ""),
                        "product_type": product.get("product_type", ""),
                    },
                    num_queries=3,
                )

                for idx, q in enumerate(queries):
                    eval_data.append({
                        "query_id": f"L3_section_{section}_{asin}_{idx}",
                        "level": 3,
                        "type": "section_search",
                        "search_type": "section",
                        "query_text": q["query"],
                        "target_node": "child",
                        "target_section": section,
                        "difficulty": q.get("difficulty", "medium"),
                        "source_asin": asin,
                        "relevant_asins": [{"asin": asin, "relevance_score": 3, "section": section}],
                        "expected_in_top_k": {
                            "top_5": q.get("difficulty") == "easy",
                            "top_10": True,
                            "top_20": True,
                        },
                    })

        return eval_data

    def _derive_retrieval_from_higher_levels(self) -> list[dict]:
        """Derive retrieval queries from Level 5/6 for consistency."""
        eval_data = []

        # From Level 6 E2E scenarios
        if self._level6_data:
            for l6_item in random.sample(self._level6_data, min(30, len(self._level6_data))):
                storage = l6_item.get("storage", "")
                search_type = "semantic"
                target_node = "parent"

                if "child" in storage:
                    target_node = "child"
                if "elasticsearch" in storage:
                    search_type = "hybrid"

                eval_data.append({
                    "query_id": f"L3_from_L6_{l6_item.get('scenario_id')}",
                    "level": 3,
                    "type": "derived_from_scenario",
                    "search_type": search_type,
                    "query_text": l6_item.get("query"),
                    "target_node": target_node,
                    "target_section": l6_item.get("storage", "").split("_")[-1] if "child" in storage else None,
                    "difficulty": "medium",
                    "source_asin": l6_item.get("source_asin"),
                    "source_scenario": l6_item.get("scenario_id"),
                })

        # From Level 5 tool selection
        if self._level5_data:
            for l5_item in random.sample(self._level5_data, min(20, len(self._level5_data))):
                if l5_item.get("type") == "tool_selection":
                    tool = l5_item.get("expected_tool", "")
                    search_type = "hybrid"
                    if "keyword" in tool:
                        search_type = "keyword"
                    elif "vector" in tool or "semantic" in tool:
                        search_type = "semantic"
                    elif "section" in tool:
                        search_type = "section"

                    eval_data.append({
                        "query_id": f"L3_from_L5_{l5_item.get('query_id')}",
                        "level": 3,
                        "type": "derived_from_tool",
                        "search_type": search_type,
                        "query_text": l5_item.get("query_text"),
                        "target_node": "parent" if "parent" in l5_item.get("expected_storage", "") else "child",
                        "difficulty": "medium",
                        "source_asin": l5_item.get("source_asin"),
                        "source_tool": tool,
                    })

        return eval_data

    def _parse_difficulty_queries(self, response: str) -> list[dict]:
        """Parse LLM response with difficulty tags."""
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            difficulty = "medium"

            if "[EASY]" in line.upper():
                difficulty = "easy"
                line = line.replace("[EASY]", "").replace("[easy]", "").strip()
            elif "[MEDIUM]" in line.upper():
                difficulty = "medium"
                line = line.replace("[MEDIUM]", "").replace("[medium]", "").strip()
            elif "[HARD]" in line.upper():
                difficulty = "hard"
                line = line.replace("[HARD]", "").replace("[hard]", "").strip()

            # Clean up numbering
            if line and line[0].isdigit():
                line = line.lstrip("0123456789").lstrip(".)]: ").strip()

            if line and len(line) > 3:
                queries.append({"query": line, "difficulty": difficulty})

        return queries

    def generate_level3_retrieval_data(self, df: pd.DataFrame, num_samples: int = 300) -> list[dict]:
        """Sync wrapper for Level 3 generation."""
        return asyncio.run(self.generate_level3_retrieval_data_async(df, num_samples))

    # Keep old template-based method for fallback
    def _generate_level3_template_based(self, df: pd.DataFrame, num_samples: int = 200) -> list[dict]:
        """Template-based Level 3 generation (fallback method).

        Mode-aware: In original mode, uses only available fields (title, category, price, stars).
        In enrich mode, also uses brand, genAI_summary, etc.
        """
        eval_data = []
        used_queries = set()

        # Original mode: 3 search types (keyword, semantic, hybrid) - no section search
        # Enrich mode: 5 search types (keyword, semantic, hybrid, section, combined)
        if self.pipeline_mode == "original":
            samples_per_type = num_samples // 3
        else:
            samples_per_type = num_samples // 5

        # Check if brand column exists and has values
        has_brand = "brand" in df.columns and df["brand"].notna().any()

        # 1. Parent node keyword search
        keyword_count = 0
        for _, product in df.iterrows():
            if keyword_count >= samples_per_type:
                break

            title = str(product.get("title", ""))
            asin = product.get("asin")
            category = str(product.get("category_level1", ""))

            if len(title) < 10:
                continue

            title_kw = self._get_title_keywords(title, 4)
            queries = [
                {"query": title_kw, "difficulty": "easy"},
            ]

            # Add brand-based query only if brand exists (enrich mode)
            if has_brand:
                brand = str(product.get("brand", ""))
                if brand and brand != "nan":
                    queries.append({"query": f"{brand} {self._get_title_keywords(title, 2)}", "difficulty": "easy"})

            # Add category-based query (works in both modes)
            if category and category != "nan":
                queries.append({"query": f"{category} {self._get_title_keywords(title, 2)}", "difficulty": "medium"})

            for q in queries:
                query_text = q["query"].strip()
                if not query_text or len(query_text) < 5 or query_text in used_queries:
                    continue

                used_queries.add(query_text)
                eval_data.append({
                    "query_id": f"L3_parent_keyword_{asin}_{keyword_count}",
                    "level": 3,
                    "type": "parent_search",
                    "search_type": "keyword",
                    "query_text": query_text,
                    "target_node": "parent",
                    "difficulty": q["difficulty"],
                    "source_asin": asin,
                    "relevant_asins": [{"asin": asin, "relevance_score": 3}],
                    "expected_in_top_k": {"top_5": True, "top_10": True, "top_20": True},
                    "pipeline_mode": self.pipeline_mode,
                })
                keyword_count += 1
                if keyword_count >= samples_per_type:
                    break

        # 2. Parent node semantic search
        semantic_count = 0
        for _, product in df.iterrows():
            if semantic_count >= samples_per_type:
                break

            title = str(product.get("title", ""))
            asin = product.get("asin")
            category = str(product.get("category_level1", ""))
            price = product.get("price", 0)

            title_kw = self._get_title_keywords(title, 3)
            if not title_kw:
                continue

            queries = [
                {"query": f"I'm looking for a {title_kw}", "difficulty": "medium"},
                {"query": f"Find me a good {title_kw}", "difficulty": "medium"},
            ]

            # Add category + title query (works in both modes)
            if category and category != "nan":
                queries.append({"query": f"I need {category} products like {title_kw}", "difficulty": "medium"})

            # Add price-based query (works in both modes)
            if price and price > 0:
                queries.append({"query": f"{title_kw} under ${float(price) * 1.5:.0f}", "difficulty": "medium"})

            # Add brand-based query only if brand exists (enrich mode)
            if has_brand:
                brand = str(product.get("brand", ""))
                if brand and brand != "nan":
                    queries.append({"query": f"Find me {brand} {title_kw}", "difficulty": "easy"})

            # Add genAI_summary query only in enrich mode
            if self.pipeline_mode == "enrich":
                genai_summary = self._safe_get(product, "genAI_summary", "")
                if genai_summary:
                    queries.append({"query": genai_summary[:100], "difficulty": "easy"})

            for q in queries:
                query_text = q["query"].strip()
                if not query_text or len(query_text) < 10 or query_text in used_queries:
                    continue

                used_queries.add(query_text)
                eval_data.append({
                    "query_id": f"L3_parent_semantic_{asin}_{semantic_count}",
                    "level": 3,
                    "type": "parent_search",
                    "search_type": "semantic",
                    "query_text": query_text,
                    "target_node": "parent",
                    "difficulty": q["difficulty"],
                    "source_asin": asin,
                    "relevant_asins": [{"asin": asin, "relevance_score": 3}],
                    "expected_in_top_k": {
                        "top_5": q["difficulty"] == "easy",
                        "top_10": True,
                        "top_20": True
                    },
                    "pipeline_mode": self.pipeline_mode,
                })
                semantic_count += 1
                if semantic_count >= samples_per_type:
                    break

        # 3. Hybrid search (keyword + semantic combined)
        hybrid_count = 0
        for _, product in df.iterrows():
            if hybrid_count >= samples_per_type:
                break

            title = str(product.get("title", ""))
            asin = product.get("asin")
            category = str(product.get("category_level1", ""))
            stars = product.get("stars", 0)

            title_kw = self._get_title_keywords(title, 3)
            if not title_kw:
                continue

            queries = [
                {"query": f"best {title_kw}", "difficulty": "medium"},
                {"query": f"top rated {title_kw}", "difficulty": "medium"},
            ]

            # Add rating-based query (works in both modes)
            if stars and float(stars) >= 4.0:
                queries.append({"query": f"highly rated {title_kw}", "difficulty": "medium"})

            # Add category + quality query
            if category and category != "nan":
                queries.append({"query": f"best {category} {self._get_title_keywords(title, 2)}", "difficulty": "medium"})

            for q in queries:
                query_text = q["query"].strip()
                if not query_text or len(query_text) < 10 or query_text in used_queries:
                    continue

                used_queries.add(query_text)
                eval_data.append({
                    "query_id": f"L3_hybrid_{asin}_{hybrid_count}",
                    "level": 3,
                    "type": "hybrid_search",
                    "search_type": "hybrid",
                    "query_text": query_text,
                    "target_node": "parent",
                    "difficulty": q["difficulty"],
                    "source_asin": asin,
                    "relevant_asins": [{"asin": asin, "relevance_score": 3}],
                    "expected_in_top_k": {"top_5": True, "top_10": True, "top_20": True},
                    "pipeline_mode": self.pipeline_mode,
                })
                hybrid_count += 1
                if hybrid_count >= samples_per_type:
                    break

        return eval_data

    # =========================================================================
    # Level 4: Storage Design Validation Data
    # =========================================================================
    def generate_level4_storage_data(self, df: pd.DataFrame, num_samples: int = 100) -> list[dict]:
        """Generate Level 4: Storage Design Validation data.

        Mode-aware:
        - Original mode: Test only basic field retrieval (price, stars, category, is_best_seller)
        - Enrich mode: Test GenAI QA fields + child node tests

        Tests:
        - QA from Qdrant (question→field mapping)
        - Qdrant-only completion tests
        - PostgreSQL fallback tests
        - Field completeness tests
        """
        eval_data = []

        # Mode-aware QA tests
        if self.pipeline_mode == "original":
            # Original mode: only basic fields available
            qa_parent_tests = [
                {"question": "What is the price of {title_short}?", "answer_field": "price", "node": "parent"},
                {"question": "What is the rating of {title_short}?", "answer_field": "stars", "node": "parent"},
                {"question": "Is {title_short} a bestseller?", "answer_field": "is_best_seller", "node": "parent"},
                {"question": "What category is {title_short} in?", "answer_field": "category_level1", "node": "parent"},
                {"question": "How many reviews does {title_short} have?", "answer_field": "reviews_count", "node": "parent"},
            ]
        else:
            # Enrich mode: GenAI fields + basic fields
            qa_parent_tests = [
                {"question": "What is {title_short}?", "answer_field": "genAI_summary", "node": "parent"},
                {"question": "What does {title_short} do?", "answer_field": "genAI_primary_function", "node": "parent"},
                {"question": "What is {title_short} best for?", "answer_field": "genAI_best_for", "node": "parent"},
                {"question": "Who should buy {title_short}?", "answer_field": "genAI_target_audience", "node": "parent"},
                {"question": "What can {title_short} do?", "answer_field": "genAI_key_capabilities", "node": "parent"},
                {"question": "Why buy {title_short}?", "answer_field": "genAI_unique_selling_points", "node": "parent"},
                {"question": "What is the price of {title_short}?", "answer_field": "price", "node": "parent"},
                {"question": "What is the rating of {title_short}?", "answer_field": "stars", "node": "parent"},
            ]

        for _, product in df.sample(n=min(num_samples // 2, len(df))).iterrows():
            title = str(product.get("title", ""))
            title_short = " ".join(title.split()[:4])
            asin = product.get("asin")

            for test in qa_parent_tests:
                question = test["question"].replace("{title_short}", title_short)
                answer_field = test["answer_field"]
                expected_answer = self._safe_get(product, answer_field)

                eval_data.append({
                    "type": "qa_from_qdrant",
                    "level": 4,
                    "subtype": "parent_qa",
                    "question": question,
                    "answer_field": answer_field,
                    "expected_node": test["node"],
                    "source_asin": asin,
                    "expected_answer": str(expected_answer) if expected_answer else None,
                    "requires_postgresql": False,
                })

        # 2. QA from Qdrant child node tests (ONLY in enrich mode - no child nodes in original mode)
        if self.pipeline_mode == "enrich":
            qa_child_tests = [
                {"question": "Tell me more about {title_short}", "answer_field": "genAI_detailed_description", "section": "description"},
                {"question": "How does {title_short} work?", "answer_field": "genAI_how_it_works", "section": "description"},
                {"question": "What are the features of {title_short}?", "answer_field": "genAI_features_detailed", "section": "features"},
                {"question": "What makes {title_short} special?", "answer_field": "genAI_standout_features", "section": "features"},
                {"question": "What are the specs of {title_short}?", "answer_field": "genAI_specs_summary", "section": "specs"},
                {"question": "Any limitations of {title_short}?", "answer_field": "genAI_specs_limitations", "section": "specs"},
                {"question": "What do users say about {title_short}?", "answer_field": "genAI_common_praises", "section": "reviews"},
                {"question": "What are complaints about {title_short}?", "answer_field": "genAI_common_complaints", "section": "reviews"},
                {"question": "Is {title_short} good for travel?", "answer_field": "genAI_use_case_scenarios", "section": "use_cases"},
                {"question": "Who should NOT buy {title_short}?", "answer_field": "genAI_not_recommended_for", "section": "use_cases"},
            ]

            for _, product in df.sample(n=min(num_samples // 2, len(df))).iterrows():
                title = str(product.get("title", ""))
                title_short = " ".join(title.split()[:4])
                asin = product.get("asin")

                for test in qa_child_tests:
                    question = test["question"].replace("{title_short}", title_short)
                    answer_field = test["answer_field"]
                    expected_answer = self._safe_get(product, answer_field)

                    eval_data.append({
                        "type": "qa_from_qdrant",
                        "level": 4,
                        "subtype": "child_qa",
                        "question": question,
                        "answer_field": answer_field,
                        "expected_node": test["section"],
                        "expected_section": test["section"],
                        "source_asin": asin,
                        "expected_answer": str(expected_answer) if expected_answer else None,
                        "requires_postgresql": False,
                        "pipeline_mode": self.pipeline_mode,
                    })

        # 3. Qdrant-only completion tests (should NOT need PostgreSQL)
        qdrant_only_questions = [
            "Find wireless headphones under $100",
            "Show me bestselling laptops",
            "What is the best rated coffee maker?",
            "Find Sony products in Electronics",
            "Show me 4+ star products",
        ]

        # Mode-aware expected fields
        if self.pipeline_mode == "original":
            expected_fields = ["asin", "title", "price", "stars", "img_url", "category_level1"]
        else:
            expected_fields = ["asin", "title", "price", "stars", "img_url", "genAI_summary", "genAI_best_for"]

        for q in qdrant_only_questions:
            eval_data.append({
                "type": "qdrant_only",
                "level": 4,
                "question": q,
                "requires_postgresql": False,
                "expected_fields_in_response": expected_fields,
                "pipeline_mode": self.pipeline_mode,
            })

        # 4. PostgreSQL fallback tests (SHOULD need PostgreSQL)
        pg_required_questions = [
            {"question": "What is the price history of {title_short}?", "pg_table": "price_history"},
            {"question": "What are the trending features in {category}?", "pg_table": "categories"},
            {"question": "What is the market share of {brand}?", "pg_table": "brands"},
            {"question": "What accessories go with {title_short}?", "pg_table": "product_relationships"},
        ]

        for _, product in df.sample(n=min(20, len(df))).iterrows():
            title = str(product.get("title", ""))
            title_short = " ".join(title.split()[:4])
            brand = self._safe_get(product, "brand", "Sony")
            category = self._safe_get(product, "category_level1", "Electronics")

            for test in pg_required_questions:
                question = test["question"].replace("{title_short}", title_short)
                question = question.replace("{brand}", brand)
                question = question.replace("{category}", category)

                eval_data.append({
                    "type": "postgresql_required",
                    "level": 4,
                    "question": question,
                    "requires_postgresql": True,
                    "pg_table": test["pg_table"],
                    "source_asin": product.get("asin"),
                })

        return eval_data

    # =========================================================================
    # Level 5: Agent Query Accuracy Data (LLM-Enhanced)
    # =========================================================================
    async def generate_level5_agent_data_async(self, df: pd.DataFrame, num_samples: int = 200) -> list[dict]:
        """Generate Level 5: Agent Query Accuracy data using LLM.

        Uses LLM to generate diverse queries for testing:
        - Query classification accuracy (which agent handles it)
        - Node routing accuracy (parent vs child node)
        - Tool selection accuracy (which tools to use)
        - Agent-to-Tool flow tests

        Derives from Level 6 scenarios for cross-level consistency.
        """
        eval_data = []
        logger.info("level5_llm_generation_start", num_agents=len(AGENT_TOOL_MAPPING))

        # Sample products for query generation (filter out NaN values)
        valid_df = df[
            df['asin'].notna() &
            df['title'].notna() &
            df['price'].notna()
        ]
        if len(valid_df) == 0:
            logger.warning("no_valid_products_for_level5", reason="All products have NaN in asin, title, or price")
            return eval_data

        sample_products = valid_df.sample(n=min(30, len(valid_df))).to_dict('records')
        categories = df['category_level1'].dropna().unique().tolist()[:10] if 'category_level1' in df.columns else []
        brands = df['brand'].dropna().unique().tolist()[:15] if 'brand' in df.columns else []

        # Part 1: Generate LLM-based queries for each agent type
        queries_per_agent = num_samples // (len(AGENT_TOOL_MAPPING) * 2)

        for agent_type, agent_config in AGENT_TOOL_MAPPING.items():
            tools = agent_config["tools"]
            logger.info("generating_agent_queries", agent=agent_type, tools=tools)

            for product in random.sample(sample_products, min(queries_per_agent, len(sample_products))):
                product_info = {
                    "title": product.get("title", ""),
                    "brand": product.get("brand") or (random.choice(brands) if brands else "Generic"),
                    "category": product.get("category_level1") or (random.choice(categories) if categories else "Products"),
                    "product_type": product.get("product_type", "product"),
                    "price": product.get("price", 100),
                }

                # Generate agent-specific queries using LLM
                routing_queries = await self._generate_agent_routing_queries(
                    agent_type=agent_type,
                    tools=tools,
                    product_info=product_info,
                    num_queries=5,
                )

                for idx, query_data in enumerate(routing_queries):
                    eval_data.append({
                        "query_id": f"L5_agent_{agent_type}_{product.get('asin')}_{idx}",
                        "level": 5,
                        "type": "agent_routing",
                        "query_text": query_data["query"],
                        "expected_agent": agent_type,
                        "expected_tools": tools,
                        "expected_storage": agent_config["primary_storage"],
                        "scenarios": agent_config["scenarios"],
                        "source_asin": product.get("asin"),
                        "product_title": product_info["title"][:50],
                    })

        # Part 2: Derive from Level 6 data for consistency (if available)
        if self._level6_data:
            logger.info("deriving_from_level6", num_level6=len(self._level6_data))
            for l6_item in random.sample(self._level6_data, min(50, len(self._level6_data))):
                eval_data.append({
                    "query_id": f"L5_from_L6_{l6_item.get('scenario_id')}_{l6_item.get('source_asin')}",
                    "level": 5,
                    "type": "agent_routing_from_scenario",
                    "query_text": l6_item.get("query"),
                    "expected_agent": l6_item.get("expected_agent"),
                    "expected_tools": l6_item.get("tool", "").split(" + "),
                    "expected_storage": l6_item.get("storage"),
                    "source_scenario": l6_item.get("scenario_id"),
                    "source_asin": l6_item.get("source_asin"),
                })

        # Part 3: Tool selection tests (specific tool → storage mapping)
        tool_selection_tests = await self._generate_tool_selection_tests(df, sample_products)
        eval_data.extend(tool_selection_tests)

        # Part 4: Node routing tests (parent vs child)
        node_routing_tests = self._generate_node_routing_tests(df, sample_products)
        eval_data.extend(node_routing_tests)

        # Store for Level 3 consistency
        self._level5_data = eval_data
        logger.info("level5_generation_complete", total_queries=len(eval_data))

        return eval_data

    async def _generate_tool_selection_tests(
        self,
        df: pd.DataFrame,
        sample_products: list[dict],
    ) -> list[dict]:
        """Generate tests for tool selection accuracy."""
        eval_data = []

        # Tool-specific query patterns
        tool_patterns = {
            "vector_search": {
                "desc": "Semantic similarity search for concepts",
                "storage": "qdrant_parent",
                "examples": ["headphones for working out", "comfortable noise cancelling", "wireless audio devices"],
            },
            "keyword_search": {
                "desc": "Exact text matching for brands/models",
                "storage": "elasticsearch",
                "examples": ["Sony WH-1000XM5", "Apple AirPods Pro", "Samsung Galaxy Buds"],
            },
            "hybrid_search": {
                "desc": "Combined semantic + keyword search",
                "storage": "qdrant_parent + elasticsearch",
                "examples": ["Sony wireless headphones under $300", "Apple earbuds with noise cancelling"],
            },
            "search_by_section": {
                "desc": "Search within specific product sections",
                "storage": "qdrant_child",
                "sections": ["reviews", "specs", "features", "use_cases"],
            },
            "find_similar": {
                "desc": "Find similar products",
                "storage": "qdrant_parent",
                "examples": ["products like AirPods", "similar to Sony headphones"],
            },
            "get_price_history": {
                "desc": "Price trends and history",
                "storage": "postgresql",
                "examples": ["price history", "price drops", "historical prices"],
            },
            "get_trending": {
                "desc": "Trending products and categories",
                "storage": "postgresql",
                "examples": ["trending now", "what's popular", "hot products"],
            },
        }

        for product in random.sample(sample_products, min(15, len(sample_products))):
            title_short = " ".join(str(product.get("title", "")).split()[:4])
            brand = product.get("brand", "")
            product_type = product.get("product_type", "product")

            for tool_name, tool_info in tool_patterns.items():
                queries = []

                # Try LLM generation if enabled
                if self.llm_enabled:
                    prompt = f"""Generate 3 user queries that would require the {tool_name} tool.

TOOL: {tool_name}
PURPOSE: {tool_info['desc']}
PRODUCT CONTEXT: {title_short} by {brand}

OUTPUT: Return 3 queries, one per line. No numbering or explanations.
"""
                    response = await self._call_llm(prompt)
                    queries = [q.strip() for q in response.strip().split("\n") if q.strip() and len(q.strip()) > 3]

                # Template fallback if LLM disabled or failed
                if not queries and self.fallback_to_template:
                    tool_templates = {
                        "vector_search": [
                            f"Find {product_type} like {title_short}",
                            f"Looking for {brand} {product_type}",
                            f"Best {product_type} options",
                        ],
                        "keyword_search": [
                            f"{brand} {title_short}",
                            f"Search {title_short}",
                            f"{brand} {product_type}",
                        ],
                        "hybrid_search": [
                            f"{brand} {product_type} with good reviews",
                            f"Best {brand} {product_type} under $500",
                            f"Top rated {title_short}",
                        ],
                        "search_by_section": [
                            f"What do reviews say about {title_short}",
                            f"Specifications of {brand} {product_type}",
                            f"Features of {title_short}",
                        ],
                        "find_similar": [
                            f"Products similar to {title_short}",
                            f"Alternatives to {brand} {product_type}",
                            f"Like {title_short} but different brand",
                        ],
                        "get_price_history": [
                            f"Price history for {title_short}",
                            f"Has {brand} {product_type} price dropped",
                            f"Price trends for {title_short}",
                        ],
                        "get_trending": [
                            f"Trending {product_type} right now",
                            f"Popular {brand} products",
                            f"What {product_type} is hot",
                        ],
                    }
                    queries = tool_templates.get(tool_name, [
                        f"Help with {title_short}",
                        f"Information about {brand} {product_type}",
                        f"Tell me about {title_short}",
                    ])

                for query in queries[:3]:
                    eval_data.append({
                        "query_id": f"L5_tool_{tool_name}_{product.get('asin')}",
                        "level": 5,
                        "type": "tool_selection",
                        "query_text": query,
                        "expected_tool": tool_name,
                        "expected_storage": tool_info["storage"],
                        "tool_description": tool_info["desc"],
                        "source_asin": product.get("asin"),
                    })

        return eval_data

    def _generate_node_routing_tests(
        self,
        df: pd.DataFrame,
        sample_products: list[dict],
    ) -> list[dict]:
        """Generate tests for node routing (parent vs child)."""
        eval_data = []

        # Filter valid products (non-NaN title, asin, price)
        valid_products = [
            p for p in sample_products
            if p.get("title") and pd.notna(p.get("title"))
            and p.get("asin") and pd.notna(p.get("asin"))
            and p.get("price") is not None and pd.notna(p.get("price"))
        ]

        if not valid_products:
            return eval_data

        # Node routing patterns
        for routing in QUESTION_ROUTING:
            for product in random.sample(valid_products, min(3, len(valid_products))):
                title = str(product.get("title", ""))
                title_short = " ".join(title.split()[:4])
                product_type = product.get("product_type", "product")
                price = product.get("price")

                query = routing["pattern"].replace("{product}", title_short)
                query = query.replace("{product_type}", product_type or "product")
                query = query.replace("{price}", str(int(price)))
                query = query.replace("{feature}", "noise cancellation")
                query = query.replace("{capability}", "Bluetooth")
                query = query.replace("{activity}", "travel")

                eval_data.append({
                    "query_id": f"L5_routing_{product.get('asin')}_{routing['node']}",
                    "level": 5,
                    "type": "node_routing",
                    "query_text": query,
                    "expected_node": routing["node"],
                    "expected_field": routing["field"],
                    "source_asin": product.get("asin"),
                })

        return eval_data

    def generate_level5_agent_data(self, df: pd.DataFrame, num_samples: int = 200) -> list[dict]:
        """Sync wrapper for Level 5 generation."""
        return asyncio.run(self.generate_level5_agent_data_async(df, num_samples))

    # Keep old template-based method for reference/fallback
    def _generate_level5_template_based(self, df: pd.DataFrame, num_samples: int = 150) -> list[dict]:
        """Template-based Level 5 generation (fallback method)."""
        eval_data = []

        # Use AGENT_TOOL_MAPPING from constants
        agent_types = AGENT_TOOL_MAPPING

        # Query templates by agent type
        query_templates = {
            "search": [
                {"template": "Find {product_type} under ${max_price}", "classification": "search", "node": "parent"},
                {"template": "Show me {brand} {product_type}", "classification": "search", "node": "parent"},
                {"template": "Best {product_type} with {feature}", "classification": "search", "node": "parent"},
            ],
            "compare": [
                {"template": "Compare {product1} vs {product2}", "classification": "comparison", "node": "specs"},
                {"template": "What's the difference between {product1} and {product2}?", "classification": "comparison", "node": "features"},
                {"template": "Which is better: {product1} or {product2}?", "classification": "comparison", "node": "specs"},
            ],
            "analysis": [
                {"template": "What do users say about {title_short}?", "classification": "analysis", "node": "reviews"},
                {"template": "What are the pros and cons of {title_short}?", "classification": "analysis", "node": "reviews"},
                {"template": "Is {title_short} durable?", "classification": "analysis", "node": "reviews"},
            ],
            "price": [
                {"template": "Is ${price} a good price for {title_short}?", "classification": "price", "node": "parent"},
                {"template": "Any deals on {product_type}?", "classification": "price", "node": "parent"},
                {"template": "What's the typical price for {product_type}?", "classification": "price", "node": "parent"},
            ],
            "trend": [
                {"template": "What's trending in {category}?", "classification": "trend", "node": "postgresql"},
                {"template": "Popular {product_type} right now", "classification": "trend", "node": "parent"},
                {"template": "Best selling {category} brands", "classification": "trend", "node": "postgresql"},
            ],
            "recommend": [
                {"template": "Products similar to {title_short}", "classification": "recommendation", "node": "parent"},
                {"template": "Accessories for {title_short}", "classification": "recommendation", "node": "parent"},
                {"template": "Alternatives to {title_short}", "classification": "recommendation", "node": "parent"},
            ],
        }

        samples_per_agent = num_samples // len(agent_types)

        for agent_type, agent_config in agent_types.items():
            templates = query_templates.get(agent_type, [])
            count = 0

            for _, product in df.iterrows():
                if count >= samples_per_agent:
                    break

                # Skip products with NaN in critical fields
                if not self._is_valid_product(product, ['asin', 'title', 'price']):
                    continue

                title = str(product.get("title"))
                title_short = " ".join(title.split()[:4])
                brand = self._safe_get(product, "brand", "Sony")
                category = self._safe_get(product, "category_level1", "Electronics")
                product_type = self._safe_get(product, "product_type", "product")
                price = int(product.get("price"))
                asin = product.get("asin")

                for template_info in templates:
                    template = template_info["template"]
                    query = template.replace("{title_short}", title_short)
                    query = query.replace("{brand}", brand)
                    query = query.replace("{category}", category)
                    query = query.replace("{product_type}", product_type or "product")
                    query = query.replace("{max_price}", str(int(price * 1.5)))
                    query = query.replace("{price}", str(price))
                    query = query.replace("{feature}", "good reviews")

                    # For comparison, use a second product
                    if "{product1}" in template or "{product2}" in template:
                        other = df[df['asin'] != asin].sample(n=1).iloc[0] if len(df) > 1 else product
                        other_title = " ".join(str(other.get("title", "")).split()[:4])
                        query = query.replace("{product1}", title_short)
                        query = query.replace("{product2}", other_title)

                    eval_data.append({
                        "query_id": f"L5_{agent_type}_{asin}_{count}",
                        "level": 5,
                        "type": "agent_routing",
                        "query_text": query,
                        "expected_agent": agent_type,
                        "expected_classification": template_info["classification"],
                        "expected_node": template_info["node"],
                        "expected_tools": agent_config["tools"],
                        "primary_storage": agent_config["primary_storage"],
                        "source_asin": asin,
                    })
                    count += 1
                    if count >= samples_per_agent:
                        break

        # Node routing tests (question → correct node)
        for routing in QUESTION_ROUTING:
            for _, product in df.sample(n=min(5, len(df))).iterrows():
                # Skip products with NaN in critical fields
                if not self._is_valid_product(product, ['asin', 'title', 'price']):
                    continue

                title = str(product.get("title"))
                title_short = " ".join(title.split()[:4])
                product_type = self._safe_get(product, "product_type", "product")
                price = product.get("price")

                query = routing["pattern"].replace("{product}", title_short)
                query = query.replace("{product_type}", product_type or "product")
                query = query.replace("{price}", str(int(price)))
                query = query.replace("{feature}", "noise cancellation")
                query = query.replace("{capability}", "Bluetooth")
                query = query.replace("{activity}", "travel")

                eval_data.append({
                    "query_id": f"L5_routing_{product.get('asin')}",
                    "level": 5,
                    "type": "node_routing",
                    "query_text": query,
                    "expected_node": routing["node"],
                    "expected_field": routing["field"],
                    "source_asin": product.get("asin"),
                })

        # NEW: Tool selection tests (from DB Design Part 13)
        tool_selection_templates = [
            {
                "template": "{product_type} under ${max_price}",
                "expected_tool": "hybrid_search",
                "expected_storage": ["qdrant_parent", "elasticsearch"],
                "expected_fields": ["genAI_summary", "genAI_best_for", "price"],
            },
            {
                "template": "{brand} {product_type}",
                "expected_tool": "keyword_search",
                "expected_storage": ["elasticsearch"],
                "expected_fields": ["title", "brand", "asin"],
            },
            {
                "template": "What do users say about {title_short} battery life?",
                "expected_tool": "search_by_section",
                "expected_storage": ["qdrant_child_reviews"],
                "expected_section": "reviews",
                "expected_fields": ["genAI_common_praises", "genAI_durability_feedback"],
            },
            {
                "template": "Products similar to {title_short}",
                "expected_tool": "find_similar",
                "expected_storage": ["qdrant_parent"],
                "expected_fields": ["asin", "title", "genAI_summary"],
            },
            {
                "template": "What's trending in {category}?",
                "expected_tool": "get_trending",
                "expected_storage": ["postgresql"],
                "expected_fields": ["trending_features", "top_brands"],
            },
            {
                "template": "Price history for {title_short}",
                "expected_tool": "get_price_history",
                "expected_storage": ["postgresql"],
                "expected_fields": ["price_history"],
            },
            {
                "template": "Compare specs of {title_short} vs similar products",
                "expected_tool": "search_by_section",
                "expected_storage": ["qdrant_child_specs"],
                "expected_section": "specs",
                "expected_fields": ["specs_key", "genAI_specs_summary"],
            },
            {
                "template": "What are the features of {title_short}?",
                "expected_tool": "search_by_section",
                "expected_storage": ["qdrant_child_features"],
                "expected_section": "features",
                "expected_fields": ["genAI_features_detailed", "genAI_standout_features"],
            },
        ]

        for _, product in df.sample(n=min(20, len(df))).iterrows():
            # Skip products with NaN in critical fields
            if not self._is_valid_product(product, ['asin', 'title']):
                continue

            title = str(product.get("title"))
            title_short = " ".join(title.split()[:4])
            brand = self._safe_get(product, "brand", "Sony")
            category = self._safe_get(product, "category_level1", "Electronics")
            product_type = self._safe_get(product, "product_type", "product")
            price = self._safe_get(product, "price", 100)
            asin = product.get("asin")

            for template_info in tool_selection_templates:
                query = template_info["template"].replace("{title_short}", title_short)
                query = query.replace("{brand}", brand)
                query = query.replace("{category}", category)
                query = query.replace("{product_type}", product_type or "product")
                query = query.replace("{max_price}", str(int(price * 1.5)))

                eval_data.append({
                    "query_id": f"L5_tool_{template_info['expected_tool']}_{asin}",
                    "level": 5,
                    "type": "tool_selection",
                    "query_text": query,
                    "expected_tool": template_info["expected_tool"],
                    "expected_storage": template_info["expected_storage"],
                    "expected_fields": template_info["expected_fields"],
                    "expected_section": template_info.get("expected_section"),
                    "tool_config": TOOL_STORAGE_MAPPING.get(template_info["expected_tool"], {}),
                    "source_asin": asin,
                })

        return eval_data

    # =========================================================================
    # Level 6: End-to-End User Scenarios Data (LLM-Enhanced)
    # =========================================================================
    async def generate_level6_e2e_data_async(self, df: pd.DataFrame, num_samples: int = 150) -> list[dict]:
        """Generate Level 6: End-to-End User Scenarios data using LLM.

        Uses LLM to generate diverse, realistic user queries for all 30 scenarios.
        This is the TOP of the evaluation hierarchy - drives Levels 5, 4, 3.

        Generates tests for all 30 user scenarios from DB Design:
        - Discovery (D1-D6): Product search and browsing
        - Comparison (C1-C5): Product comparisons
        - Analysis (A1-A5): Review and sentiment analysis
        - Price (P1-P5): Price-related queries
        - Trend (T1-T5): Trending and popularity queries
        - Recommendation (R1-R5): Product recommendations
        """
        eval_data = []
        logger.info("level6_llm_generation_start", num_scenarios=len(USER_SCENARIOS))

        # Sample diverse products for scenario generation (filter out NaN values)
        # Filter for products with valid asin, title, and price before sampling
        valid_df = df[
            df['asin'].notna() &
            df['title'].notna() &
            df['price'].notna()
        ]
        sample_products = valid_df.sample(n=min(30, len(valid_df))).to_dict('records') if len(valid_df) > 0 else []
        categories = df['category_level1'].dropna().unique().tolist()[:10] if 'category_level1' in df.columns else []
        brands = df['brand'].dropna().unique().tolist()[:15] if 'brand' in df.columns else []

        if not sample_products:
            logger.warning("no_valid_products_for_level6", reason="All products have NaN in asin, title, or price")
            return eval_data

        # Scenario descriptions for LLM context
        scenario_descriptions = {
            # Discovery
            "D1_basic_search": "Basic product search by category or type",
            "D2_feature_search": "Search for products with specific features",
            "D3_use_case_search": "Find products for a specific use case or activity",
            "D4_gift_finding": "Find gift ideas for someone",
            "D5_brand_search": "Search for products from a specific brand",
            "D6_highly_rated": "Find highly rated or best products",
            # Comparison
            "C1_direct_compare": "Compare two specific products directly",
            "C2_category_compare": "Compare product categories or types",
            "C3_value_analysis": "Find best value products within budget",
            "C4_brand_compare": "Compare products from different brands",
            "C5_feature_compare": "Compare specific features across products",
            # Analysis
            "A1_review_summary": "Get summary of user reviews and opinions",
            "A2_sentiment_analysis": "Understand overall sentiment about a product",
            "A3_common_issues": "Find common complaints or problems",
            "A4_durability_check": "Check product durability and longevity",
            "A5_quality_assessment": "Assess overall quality of a product",
            # Price
            "P1_price_check": "Check if a price is good or fair",
            "P2_deal_finding": "Find deals and discounts",
            "P3_budget_search": "Find products within a budget",
            "P4_price_range": "Get typical price range for product type",
            "P5_value_for_money": "Assess if product is worth the price",
            # Trend
            "T1_category_trends": "What's trending in a category",
            "T2_popular_products": "Find most popular products",
            "T3_brand_performance": "How is a brand performing",
            "T4_emerging_products": "Find new and emerging products",
            "T5_bestseller_analysis": "Analyze bestselling products",
            # Recommendation
            "R1_accessories": "Find accessories for a product",
            "R2_similar_products": "Find similar products",
            "R3_alternatives": "Find alternatives to a product",
            "R4_upgrade_options": "Find upgrade options",
            "R5_bundle_suggestions": "Find products often bought together",
        }

        # Generate 5 query variations per scenario per product (comprehensive coverage)
        queries_per_scenario = max(3, num_samples // len(USER_SCENARIOS))

        for scenario_id, scenario_config in USER_SCENARIOS.items():
            scenario_desc = scenario_descriptions.get(scenario_id, scenario_id)
            logger.info("generating_scenario", scenario=scenario_id, desc=scenario_desc)

            # Select products relevant to scenario
            for product in random.sample(sample_products, min(queries_per_scenario, len(sample_products))):
                product_info = {
                    "title": product.get("title", ""),
                    "brand": product.get("brand") or (random.choice(brands) if brands else "Generic"),
                    "category": product.get("category_level1") or (random.choice(categories) if categories else "Products"),
                    "product_type": product.get("product_type", "product"),
                    "price": product.get("price", 100),
                    "asin": product.get("asin"),
                }

                # Generate diverse queries using LLM
                query_variations = await self._generate_query_variations(
                    scenario_id=scenario_id,
                    scenario_desc=scenario_desc,
                    product_info=product_info,
                    num_variations=5,
                )

                # Get tool mapping for this scenario
                tool_info = self._get_scenario_tool_mapping(scenario_id)

                for idx, query in enumerate(query_variations):
                    eval_data.append({
                        "scenario_id": scenario_id,
                        "level": 6,
                        "type": "e2e_scenario",
                        "query": query,
                        "query_variation_idx": idx,
                        "scenario_description": scenario_desc,
                        "product_title": product_info["title"][:50],
                        "product_brand": product_info["brand"],
                        "product_category": product_info["category"],
                        "source_asin": product_info["asin"],
                        "expected_agent": self._get_agent_for_scenario(scenario_id),
                        **tool_info,
                        **scenario_config,
                    })

        # Store for cross-level consistency
        self._level6_data = eval_data
        logger.info("level6_generation_complete", total_queries=len(eval_data))

        return eval_data

    def _get_agent_for_scenario(self, scenario_id: str) -> str:
        """Map scenario to expected agent."""
        prefix = scenario_id.split("_")[0]
        agent_map = {
            "D1": "search", "D2": "search", "D3": "search",
            "D4": "search", "D5": "search", "D6": "search",
            "C1": "compare", "C2": "compare", "C3": "compare",
            "C4": "compare", "C5": "compare",
            "A1": "analysis", "A2": "analysis", "A3": "analysis",
            "A4": "analysis", "A5": "analysis",
            "P1": "price", "P2": "price", "P3": "price",
            "P4": "price", "P5": "price",
            "T1": "trend", "T2": "trend", "T3": "trend",
            "T4": "trend", "T5": "trend",
            "R1": "recommend", "R2": "recommend", "R3": "recommend",
            "R4": "recommend", "R5": "recommend",
        }
        return agent_map.get(prefix, "search")

    def _get_scenario_tool_mapping(self, scenario_id: str) -> dict:
        """Get tool and storage mapping for a scenario."""
        mapping = {
            # Discovery scenarios
            "D1_basic_search": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "D2_feature_search": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "D3_use_case_search": {"tool": "search_by_section", "storage": "qdrant_child_use_cases"},
            "D4_gift_finding": {"tool": "semantic_search", "storage": "qdrant_parent"},
            "D5_brand_search": {"tool": "keyword_search", "storage": "elasticsearch"},
            "D6_highly_rated": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            # Comparison scenarios
            "C1_direct_compare": {"tool": "search_by_section + get_product_details", "storage": "qdrant_child_specs + postgresql"},
            "C2_category_compare": {"tool": "aggregate_stats + search_by_section", "storage": "postgresql + qdrant_child"},
            "C3_value_analysis": {"tool": "hybrid_search + aggregate", "storage": "qdrant_parent + postgresql"},
            "C4_brand_compare": {"tool": "aggregate_stats + search_by_section", "storage": "postgresql + qdrant_child"},
            "C5_feature_compare": {"tool": "search_by_section", "storage": "qdrant_child_features"},
            # Analysis scenarios
            "A1_review_summary": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A2_sentiment_analysis": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A3_common_issues": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A4_durability_check": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A5_quality_assessment": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            # Price scenarios
            "P1_price_check": {"tool": "vector_search + get_price_history", "storage": "qdrant_parent + postgresql"},
            "P2_deal_finding": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "P3_budget_search": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "P4_price_range": {"tool": "aggregate_stats", "storage": "postgresql"},
            "P5_value_for_money": {"tool": "vector_search + aggregate", "storage": "qdrant_parent + postgresql"},
            # Trend scenarios
            "T1_category_trends": {"tool": "get_trending", "storage": "postgresql"},
            "T2_popular_products": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "T3_brand_performance": {"tool": "aggregate_stats", "storage": "postgresql"},
            "T4_emerging_products": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "T5_bestseller_analysis": {"tool": "vector_search + search_by_section", "storage": "qdrant_parent + qdrant_child"},
            # Recommendation scenarios
            "R1_accessories": {"tool": "find_similar + get_relationships", "storage": "qdrant_parent + postgresql"},
            "R2_similar_products": {"tool": "find_similar", "storage": "qdrant_parent"},
            "R3_alternatives": {"tool": "find_similar", "storage": "qdrant_parent"},
            "R4_upgrade_options": {"tool": "find_similar + search_by_section", "storage": "qdrant_parent + qdrant_child"},
            "R5_bundle_suggestions": {"tool": "get_relationships", "storage": "postgresql"},
        }
        return mapping.get(scenario_id, {"tool": "hybrid_search", "storage": "qdrant_parent"})

    def generate_level6_e2e_data(self, df: pd.DataFrame, num_samples: int = 150) -> list[dict]:
        """Sync wrapper for Level 6 generation."""
        return asyncio.run(self.generate_level6_e2e_data_async(df, num_samples))

    def _generate_scenario_test(
        self,
        scenario_id: str,
        config: dict,
        title_short: str,
        brand: str,
        category: str,
        product_type: str,
        price: float,
        asin: str,
        all_products: list,
        all_brands: list,
    ) -> dict | None:
        """Generate a test case for a specific scenario with tool mapping."""
        # Guard against NaN price values
        if price is None or pd.isna(price):
            price = 100.0
        price = float(price)

        # Scenario-to-Tool mapping aligned with evaluation-proposal.md
        scenario_tool_mapping = {
            # Discovery scenarios
            "D1_basic_search": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "D2_feature_search": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "D3_use_case_search": {"tool": "search_by_section", "storage": "qdrant_child_use_cases"},
            "D4_gift_finding": {"tool": "semantic_search", "storage": "qdrant_parent"},
            "D5_brand_search": {"tool": "keyword_search", "storage": "elasticsearch"},
            "D6_highly_rated": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            # Comparison scenarios
            "C1_direct_compare": {"tool": "search_by_section + get_product_details", "storage": "qdrant_child_specs + postgresql"},
            "C2_category_compare": {"tool": "aggregate_stats + search_by_section", "storage": "postgresql + qdrant_child"},
            "C3_value_analysis": {"tool": "hybrid_search + aggregate", "storage": "qdrant_parent + postgresql"},
            "C4_brand_compare": {"tool": "aggregate_stats + search_by_section", "storage": "postgresql + qdrant_child"},
            "C5_feature_compare": {"tool": "search_by_section", "storage": "qdrant_child_features"},
            # Analysis scenarios
            "A1_review_summary": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A2_sentiment_analysis": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A3_common_issues": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A4_durability_check": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            "A5_quality_assessment": {"tool": "search_by_section", "storage": "qdrant_child_reviews"},
            # Price scenarios
            "P1_price_check": {"tool": "vector_search + get_price_history", "storage": "qdrant_parent + postgresql"},
            "P2_deal_finding": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "P3_budget_search": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "P4_price_range": {"tool": "aggregate_stats", "storage": "postgresql"},
            "P5_value_for_money": {"tool": "vector_search + aggregate", "storage": "qdrant_parent + postgresql"},
            # Trend scenarios
            "T1_category_trends": {"tool": "get_trending", "storage": "postgresql"},
            "T2_popular_products": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "T3_brand_performance": {"tool": "aggregate_stats", "storage": "postgresql"},
            "T4_emerging_products": {"tool": "hybrid_search", "storage": "qdrant_parent"},
            "T5_bestseller_analysis": {"tool": "vector_search + search_by_section", "storage": "qdrant_parent + qdrant_child"},
            # Recommendation scenarios
            "R1_accessories": {"tool": "find_similar + get_relationships", "storage": "qdrant_parent + postgresql"},
            "R2_similar_products": {"tool": "find_similar", "storage": "qdrant_parent"},
            "R3_alternatives": {"tool": "find_similar", "storage": "qdrant_parent"},
            "R4_upgrade_options": {"tool": "find_similar + search_by_section", "storage": "qdrant_parent + qdrant_child"},
            "R5_bundle_suggestions": {"tool": "get_relationships", "storage": "postgresql"},
        }

        tool_info = scenario_tool_mapping.get(scenario_id, {})

        # Discovery scenarios
        if scenario_id == "D1_basic_search":
            return {
                "query": f"{product_type or 'product'}",
                "expected_flow": ["embedding", "semantic_search", "response"],
                "validation": {"min_results": 1, "node_type": "parent"},
                **tool_info,
            }
        elif scenario_id == "D2_feature_search":
            return {
                "query": f"{product_type} with good reviews",
                "expected_flow": ["embedding", "semantic_search", "filter", "response"],
                "validation": {"min_results": 1, "node_type": "parent"},
                **tool_info,
            }
        elif scenario_id == "D3_use_case_search":
            return {
                "query": f"best {product_type} for travel",
                "expected_flow": ["embedding", "section_search", "response"],
                "validation": {"min_results": 1, "target_section": "use_cases"},
                **tool_info,
            }
        elif scenario_id == "D4_gift_finding":
            return {
                "query": f"gift ideas in {category}",
                "expected_flow": ["embedding", "semantic_search", "filter", "response"],
                "validation": {"min_results": 1, "node_type": "parent"},
                **tool_info,
            }
        elif scenario_id == "D5_brand_search":
            return {
                "query": f"{brand} products",
                "expected_flow": ["keyword_search", "filter", "response"],
                "validation": {"min_results": 1, "filter_applied": {"brand": brand}},
                **tool_info,
            }
        elif scenario_id == "D6_highly_rated":
            return {
                "query": f"best rated {product_type}",
                "expected_flow": ["semantic_search", "sort", "response"],
                "validation": {"min_results": 1, "sorted_by": "stars"},
                **tool_info,
            }

        # Comparison scenarios
        elif scenario_id == "C1_direct_compare":
            other = random.choice([p for p in all_products if p.get("asin") != asin])
            other_title = " ".join(str(other.get("title", "")).split()[:4])
            return {
                "query": f"Compare {title_short} vs {other_title}",
                "expected_flow": ["classification", "product_lookup", "compare", "response"],
                "validation": {"mentions_both": [title_short, other_title], "target_section": "specs"},
                "products": [asin, other.get("asin")],
                **tool_info,
            }
        elif scenario_id == "C2_category_compare":
            return {
                "query": f"Compare top {product_type} brands",
                "expected_flow": ["search", "aggregate", "compare", "response"],
                "validation": {"mentions_brands": True},
                **tool_info,
            }
        elif scenario_id == "C3_value_analysis":
            return {
                "query": f"Best value {product_type} under ${int(price * 1.5)}",
                "expected_flow": ["search", "filter", "analyze", "response"],
                "validation": {"mentions_price": True, "mentions_value": True},
                **tool_info,
            }
        elif scenario_id == "C4_brand_compare":
            other_brand = random.choice([b for b in all_brands if b != brand])
            return {
                "query": f"Compare {brand} vs {other_brand} {product_type}",
                "expected_flow": ["search", "filter", "compare", "response"],
                "validation": {"mentions_both": [brand, other_brand]},
                **tool_info,
            }
        elif scenario_id == "C5_feature_compare":
            return {
                "query": f"Compare features of top {product_type}",
                "expected_flow": ["search", "section_search", "compare", "response"],
                "validation": {"target_section": "features"},
                **tool_info,
            }

        # Analysis scenarios
        elif scenario_id == "A1_review_summary":
            return {
                "query": f"What do users say about {title_short}?",
                "expected_flow": ["product_lookup", "section_search", "response"],
                "validation": {"target_section": "reviews", "mentions_sentiment": True},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "A2_sentiment_analysis":
            return {
                "query": f"Overall sentiment for {title_short}",
                "expected_flow": ["product_lookup", "section_search", "analyze", "response"],
                "validation": {"target_section": "reviews", "has_sentiment_score": True},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "A3_common_issues":
            return {
                "query": f"What are common complaints about {title_short}?",
                "expected_flow": ["product_lookup", "section_search", "response"],
                "validation": {"target_section": "reviews", "answer_field": "genAI_common_complaints"},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "A4_durability_check":
            return {
                "query": f"Is {title_short} durable?",
                "expected_flow": ["product_lookup", "section_search", "response"],
                "validation": {"target_section": "reviews", "answer_field": "genAI_durability_feedback"},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "A5_quality_assessment":
            return {
                "query": f"Quality assessment of {title_short}",
                "expected_flow": ["product_lookup", "section_search", "analyze", "response"],
                "validation": {"target_section": "reviews"},
                "source_asin": asin,
                **tool_info,
            }

        # Price scenarios
        elif scenario_id == "P1_price_check":
            return {
                "query": f"What is the price of {title_short}?",
                "expected_flow": ["product_lookup", "response"],
                "validation": {"answer_field": "price", "response_contains_number": True},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "P2_deal_finding":
            return {
                "query": f"Any deals on {product_type}?",
                "expected_flow": ["search", "filter", "response"],
                "validation": {"filter_applied": {"has_discount": True}},
                **tool_info,
            }
        elif scenario_id == "P3_budget_search":
            return {
                "query": f"{product_type} under ${int(price * 0.8)}",
                "expected_flow": ["search", "filter", "response"],
                "validation": {"filter_applied": {"price_max": int(price * 0.8)}},
                **tool_info,
            }
        elif scenario_id == "P4_price_range":
            return {
                "query": f"Typical price range for {product_type}",
                "expected_flow": ["search", "aggregate", "response"],
                "validation": {"mentions_price_range": True},
                **tool_info,
            }
        elif scenario_id == "P5_value_for_money":
            return {
                "query": f"Is {title_short} worth the price?",
                "expected_flow": ["product_lookup", "analyze", "response"],
                "validation": {"answer_field": "genAI_value_score"},
                "source_asin": asin,
                **tool_info,
            }

        # Trend scenarios
        elif scenario_id == "T1_category_trends":
            return {
                "query": f"What's trending in {category}?",
                "expected_flow": ["aggregate", "response"],
                "validation": {"requires_postgresql": True, "pg_table": "categories"},
                **tool_info,
            }
        elif scenario_id == "T2_popular_products":
            return {
                "query": f"Most popular {product_type}",
                "expected_flow": ["search", "sort", "response"],
                "validation": {"sorted_by": "popularity_score"},
                **tool_info,
            }
        elif scenario_id == "T3_brand_performance":
            return {
                "query": f"How is {brand} performing in {category}?",
                "expected_flow": ["aggregate", "response"],
                "validation": {"requires_postgresql": True, "pg_table": "brands"},
                **tool_info,
            }
        elif scenario_id == "T4_emerging_products":
            return {
                "query": f"New and emerging {product_type}",
                "expected_flow": ["search", "filter", "response"],
                "validation": {"filter_applied": {"is_emerging": True}},
                **tool_info,
            }
        elif scenario_id == "T5_bestseller_analysis":
            return {
                "query": f"Bestselling {product_type} analysis",
                "expected_flow": ["search", "filter", "analyze", "response"],
                "validation": {"filter_applied": {"is_best_seller": True}},
                **tool_info,
            }

        # Recommendation scenarios
        elif scenario_id == "R1_accessories":
            return {
                "query": f"Accessories for {title_short}",
                "expected_flow": ["product_lookup", "find_related", "response"],
                "validation": {"relationship_type": "accessory"},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "R2_similar_products":
            return {
                "query": f"Products similar to {title_short}",
                "expected_flow": ["product_lookup", "vector_similarity", "response"],
                "validation": {"min_results": 3},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "R3_alternatives":
            return {
                "query": f"Alternatives to {title_short}",
                "expected_flow": ["product_lookup", "find_related", "response"],
                "validation": {"relationship_type": "alternative"},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "R4_upgrade_options":
            return {
                "query": f"Upgrade options from {title_short}",
                "expected_flow": ["product_lookup", "find_related", "filter", "response"],
                "validation": {"relationship_type": "upgrade", "price_higher": True},
                "source_asin": asin,
                **tool_info,
            }
        elif scenario_id == "R5_bundle_suggestions":
            return {
                "query": f"Bundle suggestions with {title_short}",
                "expected_flow": ["product_lookup", "find_related", "response"],
                "validation": {"requires_postgresql": True, "pg_table": "product_relationships"},
                "source_asin": asin,
                **tool_info,
            }

        return None


def _flatten_for_csv(data: list[dict], level: int) -> list[dict]:
    """Flatten nested evaluation data for CSV export.

    Each level has different key fields to extract for easy review.
    """
    flattened = []

    for item in data:
        row = {
            "level": level,
            "type": item.get("type", ""),
        }

        if level == 6:
            # E2E Scenarios - key fields for review
            row.update({
                "scenario_id": item.get("scenario_id", ""),
                "query": item.get("query", ""),
                "frequency": item.get("frequency", ""),
                "depth": item.get("depth", ""),
                "tool": item.get("tool", ""),
                "storage": item.get("storage", ""),
                "node": item.get("node", ""),
                "latency_target": item.get("latency_target", ""),
                "expected_flow": " → ".join(item.get("expected_flow", [])) if item.get("expected_flow") else "",
                "source_asin": item.get("source_asin", ""),
            })

        elif level == 5:
            # Agent Query Accuracy - key fields for review
            row.update({
                "query_id": item.get("query_id", ""),
                "query_text": item.get("query_text", ""),
                "expected_agent": item.get("expected_agent", ""),
                "expected_tool": item.get("expected_tool", ""),
                "expected_node": item.get("expected_node", ""),
                "expected_section": item.get("expected_section", ""),
                "expected_storage": ", ".join(item.get("expected_storage", [])) if isinstance(item.get("expected_storage"), list) else item.get("expected_storage", ""),
                "expected_classification": item.get("expected_classification", ""),
                "primary_storage": item.get("primary_storage", ""),
                "source_asin": item.get("source_asin", ""),
            })

        elif level == 4:
            # Storage Design - key fields for review
            row.update({
                "question": item.get("question", ""),
                "subtype": item.get("subtype", ""),
                "answer_field": item.get("answer_field", ""),
                "expected_node": item.get("expected_node", ""),
                "expected_section": item.get("expected_section", ""),
                "requires_postgresql": item.get("requires_postgresql", ""),
                "pg_table": item.get("pg_table", ""),
                "source_asin": item.get("source_asin", ""),
                "expected_answer": str(item.get("expected_answer", ""))[:100] if item.get("expected_answer") else "",
            })

        elif level == 3:
            # Retrieval Performance - key fields for review
            row.update({
                "query_id": item.get("query_id", ""),
                "query_text": item.get("query_text", ""),
                "search_type": item.get("search_type", ""),
                "target_node": item.get("target_node", ""),
                "target_section": item.get("target_section", ""),
                "difficulty": item.get("difficulty", ""),
                "source_asin": item.get("source_asin", ""),
                "filters": str(item.get("filters", "")) if item.get("filters") else "",
                "expected_in_top_5": item.get("expected_in_top_k", {}).get("top_5", "") if isinstance(item.get("expected_in_top_k"), dict) else "",
                "expected_in_top_10": item.get("expected_in_top_k", {}).get("top_10", "") if isinstance(item.get("expected_in_top_k"), dict) else "",
            })

        flattened.append(row)

    return flattened


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help="Input CSV file path (default: output of 03_clean_data)",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Output directory for evaluation datasets (default: from pipeline_config.yaml)",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama service URL (default: from pipeline_config.yaml)",
)
@click.option(
    "--llm-model",
    default=None,
    help="LLM model name (default: from pipeline_config.yaml)",
)
@click.option(
    "--llm-enabled/--no-llm",
    default=None,
    help="Enable/disable LLM for generation (default: from pipeline_config.yaml)",
)
@click.option(
    "--level1-samples",
    type=int,
    default=None,
    help="Number of samples for Level 1 (Embedding)",
)
@click.option(
    "--level2-samples",
    type=int,
    default=None,
    help="Number of samples for Level 2 (Indexing)",
)
@click.option(
    "--level3-samples",
    type=int,
    default=None,
    help="Number of samples for Level 3 (Retrieval)",
)
@click.option(
    "--level4-samples",
    type=int,
    default=None,
    help="Number of samples for Level 4 (Storage)",
)
@click.option(
    "--level5-samples",
    type=int,
    default=None,
    help="Number of samples for Level 5 (Agent)",
)
@click.option(
    "--level6-samples",
    type=int,
    default=None,
    help="Number of samples for Level 6 (E2E)",
)
@click.option(
    "--metrics-output",
    "metrics_path",
    type=click.Path(),
    default=None,
    help="Metrics output file path (default: from pipeline_config.yaml)",
)
def main(
    input_path: str | None,
    output_dir: str | None,
    ollama_url: str | None,
    llm_model: str | None,
    llm_enabled: bool | None,
    level1_samples: int | None,
    level2_samples: int | None,
    level3_samples: int | None,
    level4_samples: int | None,
    level5_samples: int | None,
    level6_samples: int | None,
    metrics_path: str | None,
):
    """Generate comprehensive evaluation datasets for all 6 levels.

    Configuration is loaded from pipeline_config.yaml. CLI options override config values.

    Input: Always from 03_clean_data output (cleaned/mvp_{count}_{mode}_cleaned.csv)

    The pipeline mode (original vs enrich) affects:
    - Which fields are expected to be populated
    - Question routing patterns
    - GenAI field expectations in Level 2/4 tests
    """
    start_time = time.time()

    # Load configuration from pipeline_config.yaml
    script_name = "06_generate_eval_data"
    script_config = cfg.get_script(script_name)
    llm_config = script_config.get("llm", {})
    samples_config = script_config.get("samples", {})

    settings = get_settings()
    product_count = cfg.get_count()

    # Get pipeline mode from config (determines field expectations)
    pipeline_mode = cfg.get_mode()
    logger.info("pipeline_mode_detected", mode=pipeline_mode)

    # Resolve paths from config with {count} and {mode} substitution
    def resolve_path_local(path_str: str) -> Path:
        """Resolve path with {count} and {mode} substitution."""
        resolved = path_str.replace("{count}", str(product_count)).replace("{mode}", pipeline_mode)
        return settings.data_dir / resolved if not Path(resolved).is_absolute() else Path(resolved)

    # Input is always from 03_clean_data output (cleaned CSV)
    # Both original and enrich modes use the same cleaned file format
    default_input = f"cleaned/mvp_{product_count}_{pipeline_mode}_cleaned.csv"

    config_input = script_config.get("input", default_input)
    config_output_dir = script_config.get("output_dir", "eval/datasets")
    config_metrics = script_config.get("metrics", "metrics/06_eval_data_generation_metrics.json")

    input_path = Path(input_path) if input_path else resolve_path_local(config_input)
    output_dir = Path(output_dir) if output_dir else resolve_path_local(config_output_dir)
    metrics_path = Path(metrics_path) if metrics_path else resolve_path_local(config_metrics)

    # LLM configuration (CLI > config > default)
    final_ollama_url = ollama_url or llm_config.get("ollama_url", "http://192.168.80.54:11434")
    final_llm_model = llm_model or llm_config.get("model", "gpt-oss:120b")
    final_llm_enabled = llm_enabled if llm_enabled is not None else llm_config.get("enabled", True)
    final_llm_timeout = llm_config.get("timeout", 300.0)
    final_llm_max_retries = llm_config.get("max_retries", 3)
    final_fallback_to_template = script_config.get("fallback_to_template", True)

    # Sample sizes (CLI > config > default)
    level1_samples = level1_samples if level1_samples is not None else samples_config.get("level1", 100)
    level2_samples = level2_samples if level2_samples is not None else samples_config.get("level2", 100)
    level3_samples = level3_samples if level3_samples is not None else samples_config.get("level3", 300)
    level4_samples = level4_samples if level4_samples is not None else samples_config.get("level4", 100)
    level5_samples = level5_samples if level5_samples is not None else samples_config.get("level5", 200)
    level6_samples = level6_samples if level6_samples is not None else samples_config.get("level6", 150)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "starting_eval_data_generation",
        input_path=str(input_path),
        output_dir=str(output_dir),
        pipeline_mode=pipeline_mode,
        ollama_url=final_ollama_url,
        llm_model=final_llm_model,
        llm_enabled=final_llm_enabled,
    )

    metrics = {
        "stage": "evaluation_data_generation",
        "started_at": datetime.utcnow().isoformat(),
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "pipeline_mode": pipeline_mode,
        "config": {
            "pipeline_mode": pipeline_mode,
            "ollama_url": final_ollama_url,
            "llm_model": final_llm_model,
            "llm_enabled": final_llm_enabled,
            "llm_timeout": final_llm_timeout,
            "fallback_to_template": final_fallback_to_template,
            "samples": {
                "level1": level1_samples,
                "level2": level2_samples,
                "level3": level3_samples,
                "level4": level4_samples,
                "level5": level5_samples,
                "level6": level6_samples,
            },
        },
    }

    try:
        # Load data
        df = pd.read_csv(input_path)
        logger.info("data_loaded", rows=len(df), pipeline_mode=pipeline_mode)

        # Initialize generator with LLM config and pipeline mode
        generator = EvaluationDataGenerator(
            ollama_url=final_ollama_url,
            llm_model=final_llm_model,
            llm_timeout=final_llm_timeout,
            llm_max_retries=final_llm_max_retries,
            llm_enabled=final_llm_enabled,
            fallback_to_template=final_fallback_to_template,
            pipeline_mode=pipeline_mode,
        )
        datasets = {}

        # =================================================================
        # GENERATION ORDER: Level 6 → 5 → 4 → 3 (user scenarios first)
        # Level 1 & 2 are based on data ingestion pipeline outputs
        # Output: JSON (full data) + CSV (for easy review)
        # =================================================================

        # Build output filename suffix with count and mode (matches research-search-flows config)
        output_suffix = f"{product_count}_{pipeline_mode}"

        # Level 6: E2E Scenarios (30 user scenarios - generate FIRST)
        # These define the complete user journeys and drive all other levels
        logger.info("generating_level6_e2e_data", note="User scenarios first")
        level6_data = generator.generate_level6_e2e_data(df, level6_samples)
        datasets["level6_e2e"] = level6_data
        with open(output_dir / f"level6_e2e_{output_suffix}.json", "w") as f:
            json.dump(level6_data, f, indent=2)
        # CSV for Level 6
        level6_csv = _flatten_for_csv(level6_data, level=6)
        pd.DataFrame(level6_csv).to_csv(output_dir / f"level6_e2e_{output_suffix}.csv", index=False)

        # Level 5: Agent Query Accuracy (tool selection, routing)
        # Tests agent classification and tool selection accuracy
        logger.info("generating_level5_agent_data", note="Agent routing and tools")
        level5_data = generator.generate_level5_agent_data(df, level5_samples)
        datasets["level5_agent"] = level5_data
        with open(output_dir / f"level5_agent_{output_suffix}.json", "w") as f:
            json.dump(level5_data, f, indent=2)
        # CSV for Level 5
        level5_csv = _flatten_for_csv(level5_data, level=5)
        pd.DataFrame(level5_csv).to_csv(output_dir / f"level5_agent_{output_suffix}.csv", index=False)

        # Level 4: Storage Design Validation
        # Tests QA from Qdrant vs PostgreSQL fallback
        logger.info("generating_level4_storage_data", note="Storage validation")
        level4_data = generator.generate_level4_storage_data(df, level4_samples)
        datasets["level4_storage"] = level4_data
        with open(output_dir / f"level4_storage_{output_suffix}.json", "w") as f:
            json.dump(level4_data, f, indent=2)
        # CSV for Level 4
        level4_csv = _flatten_for_csv(level4_data, level=4)
        pd.DataFrame(level4_csv).to_csv(output_dir / f"level4_storage_{output_suffix}.csv", index=False)

        # Level 3: Retrieval Performance
        # Tests search quality (Recall@K, MRR, Section Hit Rate)
        logger.info("generating_level3_retrieval_data", note="Search evaluation")
        level3_data = generator.generate_level3_retrieval_data(df, level3_samples)
        datasets["level3_retrieval"] = level3_data
        with open(output_dir / f"level3_retrieval_{output_suffix}.json", "w") as f:
            json.dump(level3_data, f, indent=2)
        # CSV for Level 3
        level3_csv = _flatten_for_csv(level3_data, level=3)
        pd.DataFrame(level3_csv).to_csv(output_dir / f"level3_retrieval_{output_suffix}.csv", index=False)

        # =================================================================
        # Level 1 & 2: Based on DATA INGESTION PIPELINE outputs
        # These evaluate the quality of embeddings and indexing from pipeline
        # =================================================================

        # Level 2: Indexing Correctness (based on ingested data structure)
        # Tests parent-child hierarchy and field presence in Qdrant
        logger.info("generating_level2_indexing_data", note="Based on ingestion pipeline")
        level2_data = generator.generate_level2_indexing_data(df, level2_samples)
        datasets["level2_indexing"] = level2_data
        with open(output_dir / f"level2_indexing_{output_suffix}.json", "w") as f:
            json.dump(level2_data, f, indent=2)

        # Level 1: Embedding Quality (based on ingested embeddings)
        # Tests embedding similarity and clustering from pipeline
        logger.info("generating_level1_embedding_data", note="Based on ingestion pipeline")
        level1_data = generator.generate_level1_embedding_data(df, level1_samples)
        datasets["level1_embedding"] = level1_data
        with open(output_dir / f"level1_embedding_{output_suffix}.json", "w") as f:
            json.dump(level1_data, f, indent=2)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate statistics
        dataset_stats = {name: len(data) for name, data in datasets.items()}

        # Count by type within each level
        type_counts = {}
        for name, data in datasets.items():
            type_counts[name] = {}
            for item in data:
                item_type = item.get("type", "unknown")
                type_counts[name][item_type] = type_counts[name].get(item_type, 0) + 1

        metrics.update({
            "completed_at": datetime.utcnow().isoformat(),
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "metrics": {
                "input_products": len(df),
                "datasets_generated": dataset_stats,
                "type_counts": type_counts,
                "total_eval_samples": sum(dataset_stats.values()),
            },
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("eval_data_generation_complete", total_samples=sum(dataset_stats.values()))

        print("\n" + "=" * 70)
        print("EVALUATION DATA GENERATION COMPLETE")
        print("=" * 70)
        print(f"Pipeline Mode:       {pipeline_mode.upper()}")
        print(f"Input products:      {len(df)}")
        print(f"Output directory:    {output_dir}")
        if pipeline_mode == "original":
            print("\nMode: ORIGINAL (no GenAI enrichment)")
            print("  - GenAI fields: NOT expected (empty)")
            print("  - Search focuses on: title, brand, category, chunk_* fields")
        else:
            print("\nMode: ENRICH (with GenAI enrichment)")
            print("  - GenAI fields: expected and evaluated")
            print("  - Search includes: genAI_summary, genAI_best_for, etc.")
        print("\nLLM Configuration (from pipeline_config.yaml):")
        print(f"  Ollama URL:        {final_ollama_url}")
        print(f"  Model:             {final_llm_model}")
        print(f"  LLM Enabled:       {final_llm_enabled}")
        print(f"  Fallback:          {final_fallback_to_template}")
        print("\nGeneration Order: Level 6 → 5 → 4 → 3 (user scenarios first)")
        print("Level 1 & 2: Based on data ingestion pipeline")
        print("\nDatasets generated:")
        for name, count in dataset_stats.items():
            print(f"  - {name}: {count} samples")
            for type_name, type_count in type_counts.get(name, {}).items():
                print(f"      • {type_name}: {type_count}")
        print(f"\nTotal samples:       {sum(dataset_stats.values())}")
        print(f"Processing time:     {processing_time:.2f}s")
        print("\nOutput files (for easy review):")
        print("  JSON (full data):")
        print(f"    - level6_e2e_{pipeline_mode}_evaluation.json")
        print(f"    - level5_agent_{pipeline_mode}_evaluation.json")
        print(f"    - level4_storage_{pipeline_mode}_evaluation.json")
        print(f"    - level3_retrieval_{pipeline_mode}_evaluation.json")
        print(f"    - level2_indexing_{pipeline_mode}_evaluation.json")
        print(f"    - level1_embedding_{pipeline_mode}_evaluation.json")
        print("  CSV (structured for review):")
        print(f"    - level6_e2e_{pipeline_mode}_evaluation.csv")
        print(f"    - level5_agent_{pipeline_mode}_evaluation.csv")
        print(f"    - level4_storage_{pipeline_mode}_evaluation.csv")
        print(f"    - level3_retrieval_{pipeline_mode}_evaluation.csv")
        print("=" * 70)

    except Exception as e:
        logger.error("eval_data_generation_failed", error=str(e))
        import traceback
        traceback.print_exc()
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
