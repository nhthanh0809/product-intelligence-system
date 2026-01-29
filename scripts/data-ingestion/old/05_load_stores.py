#!/usr/bin/env python3
"""
Script 05: Load Multi-Store

Load embedded data into PostgreSQL, Qdrant, and Elasticsearch.

Usage:
    python scripts/05_load_stores.py --input data/embedded/mvp_100k_embedded.parquet
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

# Add src and scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # data-pipeline/
sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/

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


class PostgresLoader:
    """Load data into PostgreSQL with extended schema support.

    Supports:
    - Products table (core product data)
    - Brands table (aggregated brand statistics)
    - Price history tracking
    - Category extraction and loading
    - Mode-aware loading (original vs enrich)
    """

    def __init__(self, dsn: str, schema_path: str | None = None, mode: str = "enrich"):
        """Initialize loader.

        Args:
            dsn: PostgreSQL connection string
            schema_path: Path to SQL schema file for initialization
            mode: Pipeline mode ('original' or 'enrich')
        """
        self.dsn = dsn
        self.schema_path = schema_path or str(
            Path(__file__).parent.parent / "schema" / "postgres_schema.sql"
        )
        self.mode = mode
        self.conn = None

    async def connect(self):
        """Connect to database."""
        import asyncpg
        self.conn = await asyncpg.connect(self.dsn)

    async def close(self):
        """Close connection."""
        if self.conn:
            await self.conn.close()

    async def init_schema(self):
        """Initialize database schema from SQL file."""
        schema_file = Path(self.schema_path)
        if schema_file.exists():
            schema_sql = schema_file.read_text()
            try:
                await self.conn.execute(schema_sql)
                logger.info("schema_initialized", schema_file=str(schema_file))
            except Exception as e:
                logger.warning("schema_init_error", error=str(e))
        else:
            logger.warning("schema_file_not_found", path=str(schema_file))

    def _clean_value(self, value: Any) -> Any:
        """Clean value for PostgreSQL - convert NaN to None."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    def _to_json(self, value: Any) -> str | None:
        """Convert value to JSON string for JSONB fields."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, str):
            # Already a JSON string, return as-is
            try:
                json.loads(value)
                return value
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON, wrap in JSON
                return json.dumps(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return json.dumps(value)

    async def load_products(self, df: pd.DataFrame) -> dict:
        """Load products to PostgreSQL. Mode-aware: only loads GenAI fields in enrich mode.

        Returns:
            Dict with counts for products, brands, price_history
        """
        # Get unique products from nodes
        products = df[df["node_type"] == "parent"].copy()
        results = {"products": 0, "brands": 0, "price_history": 0}

        if self.mode == "enrich":
            # Enrich mode: include all fields including GenAI
            product_records = []
            for _, row in products.iterrows():
                product_records.append((
                    # Core fields
                    self._clean_value(row.get("asin")),                          # $1
                    self._clean_value(row.get("title")),                         # $2
                    self._clean_value(row.get("brand")),                         # $3
                    # Identity (GenAI cleaned)
                    self._clean_value(row.get("short_title")),                   # $4
                    self._clean_value(row.get("product_type")),                  # $5
                    self._to_json(row.get("product_type_keywords")),             # $6
                    # Pricing
                    self._clean_value(row.get("price")),                         # $7
                    self._clean_value(row.get("list_price")),                    # $8
                    # Ratings
                    self._clean_value(row.get("stars")),                         # $9
                    self._clean_value(row.get("reviews_count")),                 # $10
                    # Categories
                    self._clean_value(row.get("category_level1")),               # $11
                    self._clean_value(row.get("category_level2")),               # $12
                    self._clean_value(row.get("category_level3")),               # $13
                    # Flags
                    bool(row.get("is_best_seller", False)),                      # $14
                    bool(row.get("is_amazon_choice", False)),                    # $15
                    bool(row.get("prime_eligible", False)),                      # $16
                    self._clean_value(row.get("availability")),                  # $17
                    # URLs
                    self._clean_value(row.get("product_url")),                   # $18
                    self._clean_value(row.get("img_url")),                       # $19
                    # GenAI Parent Fields (quick answers)
                    self._clean_value(row.get("genAI_summary")),                 # $20
                    self._clean_value(row.get("genAI_primary_function")),        # $21
                    self._clean_value(row.get("genAI_best_for")),                # $22
                    self._to_json(row.get("genAI_use_cases")),                   # $23
                    self._to_json(row.get("genAI_target_audience")),             # $24
                    self._to_json(row.get("genAI_key_capabilities")),            # $25
                    self._clean_value(row.get("genAI_unique_selling_points")),   # $26
                    self._clean_value(row.get("genAI_value_score")),             # $27
                    # GenAI Description Fields
                    self._clean_value(row.get("genAI_detailed_description")),    # $28
                    self._clean_value(row.get("genAI_how_it_works")),            # $29
                    self._to_json(row.get("genAI_whats_included")),              # $30
                    # GenAI Features Fields
                    self._to_json(row.get("genAI_features_detailed")),           # $31
                    self._to_json(row.get("genAI_standout_features")),           # $32
                    self._clean_value(row.get("genAI_technology_explained")),    # $33
                    # GenAI Specs Fields
                    self._clean_value(row.get("genAI_specs_summary")),           # $34
                    self._to_json(row.get("genAI_specs_comparison_ready")),      # $35
                    self._to_json(row.get("genAI_specs_limitations")),           # $36
                    # GenAI Review Analysis Fields
                    self._clean_value(row.get("genAI_sentiment_score")),         # $37
                    self._to_json(row.get("genAI_common_praises")),              # $38
                    self._to_json(row.get("genAI_common_complaints")),           # $39
                    self._clean_value(row.get("genAI_durability_feedback")),     # $40
                    self._clean_value(row.get("genAI_value_for_money_feedback")),# $41
                    # GenAI Use Cases Fields
                    self._to_json(row.get("genAI_use_case_scenarios")),          # $42
                    self._to_json(row.get("genAI_ideal_user_profiles")),         # $43
                    self._to_json(row.get("genAI_not_recommended_for")),         # $44
                    self._to_json(row.get("genAI_problems_solved")),             # $45
                    # GenAI Pros/Cons
                    self._to_json(row.get("genAI_pros")),                        # $46
                    self._to_json(row.get("genAI_cons")),                        # $47
                    # Computed Metrics
                    self._clean_value(row.get("popularity_score")),              # $48
                    self._clean_value(row.get("trending_rank")),                 # $49
                    self._clean_value(row.get("price_percentile")),              # $50
                ))

            # Bulk insert products with all fields (enrich mode)
            await self.conn.executemany(
                """
                INSERT INTO products (
                    asin, title, brand,
                    short_title, product_type, product_type_keywords,
                    price, list_price, stars, reviews_count,
                    category_level1, category_level2, category_level3,
                    is_best_seller, is_amazon_choice, prime_eligible, availability,
                    product_url, img_url,
                    genAI_summary, genAI_primary_function, genAI_best_for,
                    genAI_use_cases, genAI_target_audience, genAI_key_capabilities,
                    genAI_unique_selling_points, genAI_value_score,
                    genAI_detailed_description, genAI_how_it_works, genAI_whats_included,
                    genAI_features_detailed, genAI_standout_features, genAI_technology_explained,
                    genAI_specs_summary, genAI_specs_comparison_ready, genAI_specs_limitations,
                    genAI_sentiment_score, genAI_common_praises, genAI_common_complaints,
                    genAI_durability_feedback, genAI_value_for_money_feedback,
                    genAI_use_case_scenarios, genAI_ideal_user_profiles,
                    genAI_not_recommended_for, genAI_problems_solved,
                    genAI_pros, genAI_cons,
                    popularity_score, trending_rank, price_percentile
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                    $31, $32, $33, $34, $35, $36, $37, $38, $39, $40,
                    $41, $42, $43, $44, $45, $46, $47, $48, $49, $50
                )
                ON CONFLICT (asin) DO UPDATE SET
                    title = EXCLUDED.title,
                    brand = EXCLUDED.brand,
                    short_title = EXCLUDED.short_title,
                    product_type = EXCLUDED.product_type,
                    product_type_keywords = EXCLUDED.product_type_keywords,
                    price = EXCLUDED.price,
                    list_price = EXCLUDED.list_price,
                    stars = EXCLUDED.stars,
                    reviews_count = EXCLUDED.reviews_count,
                    category_level1 = EXCLUDED.category_level1,
                    category_level2 = EXCLUDED.category_level2,
                    category_level3 = EXCLUDED.category_level3,
                    is_best_seller = EXCLUDED.is_best_seller,
                    availability = EXCLUDED.availability,
                    product_url = EXCLUDED.product_url,
                    img_url = EXCLUDED.img_url,
                    genAI_summary = EXCLUDED.genAI_summary,
                    genAI_primary_function = EXCLUDED.genAI_primary_function,
                    genAI_best_for = EXCLUDED.genAI_best_for,
                    genAI_use_cases = EXCLUDED.genAI_use_cases,
                    genAI_target_audience = EXCLUDED.genAI_target_audience,
                    genAI_key_capabilities = EXCLUDED.genAI_key_capabilities,
                    genAI_unique_selling_points = EXCLUDED.genAI_unique_selling_points,
                    genAI_value_score = EXCLUDED.genAI_value_score,
                    genAI_detailed_description = EXCLUDED.genAI_detailed_description,
                    genAI_how_it_works = EXCLUDED.genAI_how_it_works,
                    genAI_whats_included = EXCLUDED.genAI_whats_included,
                    genAI_features_detailed = EXCLUDED.genAI_features_detailed,
                    genAI_standout_features = EXCLUDED.genAI_standout_features,
                    genAI_technology_explained = EXCLUDED.genAI_technology_explained,
                    genAI_specs_summary = EXCLUDED.genAI_specs_summary,
                    genAI_specs_comparison_ready = EXCLUDED.genAI_specs_comparison_ready,
                    genAI_specs_limitations = EXCLUDED.genAI_specs_limitations,
                    genAI_sentiment_score = EXCLUDED.genAI_sentiment_score,
                    genAI_common_praises = EXCLUDED.genAI_common_praises,
                    genAI_common_complaints = EXCLUDED.genAI_common_complaints,
                    genAI_durability_feedback = EXCLUDED.genAI_durability_feedback,
                    genAI_value_for_money_feedback = EXCLUDED.genAI_value_for_money_feedback,
                    genAI_use_case_scenarios = EXCLUDED.genAI_use_case_scenarios,
                    genAI_ideal_user_profiles = EXCLUDED.genAI_ideal_user_profiles,
                    genAI_not_recommended_for = EXCLUDED.genAI_not_recommended_for,
                    genAI_problems_solved = EXCLUDED.genAI_problems_solved,
                    genAI_pros = EXCLUDED.genAI_pros,
                    genAI_cons = EXCLUDED.genAI_cons,
                    popularity_score = EXCLUDED.popularity_score,
                    trending_rank = EXCLUDED.trending_rank,
                    price_percentile = EXCLUDED.price_percentile,
                    genAI_enriched_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                """,
                product_records,
            )
        else:
            # Original mode: only core fields, no GenAI
            product_records = []
            for _, row in products.iterrows():
                product_records.append((
                    # Core fields
                    self._clean_value(row.get("asin")),                          # $1
                    self._clean_value(row.get("title")),                         # $2
                    self._clean_value(row.get("brand")),                         # $3
                    # Pricing
                    self._clean_value(row.get("price")),                         # $4
                    self._clean_value(row.get("list_price")),                    # $5
                    # Ratings
                    self._clean_value(row.get("stars")),                         # $6
                    self._clean_value(row.get("reviews_count")),                 # $7
                    # Categories
                    self._clean_value(row.get("category_level1")),               # $8
                    self._clean_value(row.get("category_level2")),               # $9
                    self._clean_value(row.get("category_level3")),               # $10
                    # Flags
                    bool(row.get("is_best_seller", False)),                      # $11
                    bool(row.get("is_amazon_choice", False)),                    # $12
                    bool(row.get("prime_eligible", False)),                      # $13
                    self._clean_value(row.get("availability")),                  # $14
                    # URLs
                    self._clean_value(row.get("product_url")),                   # $15
                    self._clean_value(row.get("img_url")),                       # $16
                ))

            # Bulk insert products with core fields only (original mode)
            await self.conn.executemany(
                """
                INSERT INTO products (
                    asin, title, brand,
                    price, list_price, stars, reviews_count,
                    category_level1, category_level2, category_level3,
                    is_best_seller, is_amazon_choice, prime_eligible, availability,
                    product_url, img_url
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16
                )
                ON CONFLICT (asin) DO UPDATE SET
                    title = EXCLUDED.title,
                    brand = EXCLUDED.brand,
                    price = EXCLUDED.price,
                    list_price = EXCLUDED.list_price,
                    stars = EXCLUDED.stars,
                    reviews_count = EXCLUDED.reviews_count,
                    category_level1 = EXCLUDED.category_level1,
                    category_level2 = EXCLUDED.category_level2,
                    category_level3 = EXCLUDED.category_level3,
                    is_best_seller = EXCLUDED.is_best_seller,
                    availability = EXCLUDED.availability,
                    product_url = EXCLUDED.product_url,
                    img_url = EXCLUDED.img_url,
                    updated_at = CURRENT_TIMESTAMP
                """,
                product_records,
            )

        results["products"] = len(product_records)

        # Extract and upsert brands (only if brand column exists - enrich mode)
        if "brand" in products.columns:
            brands = products[["brand"]].dropna().drop_duplicates()
            brand_records = []
            for _, row in brands.iterrows():
                brand_name = row.get("brand")
                if brand_name:
                    brand_slug = brand_name.lower().replace(" ", "-").replace(".", "")
                    brand_records.append((brand_name, brand_slug))

            if brand_records:
                await self.conn.executemany(
                    """
                    INSERT INTO brands (name, slug)
                    VALUES ($1, $2)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    brand_records,
                )
                results["brands"] = len(brand_records)
        else:
            logger.info("skipping_brands", reason="brand column not present in original mode")

        # Insert price history records for current prices
        price_records = []
        for _, row in products.iterrows():
            asin = row.get("asin")
            price = self._clean_value(row.get("price"))
            list_price = self._clean_value(row.get("list_price"))

            if asin and price:
                discount = None
                if list_price and list_price > price:
                    discount = round((1 - price / list_price) * 100, 2)
                price_records.append((asin, price, list_price, discount))

        if price_records:
            await self.conn.executemany(
                """
                INSERT INTO price_history (asin, price, original_price, discount_percentage)
                VALUES ($1, $2, $3, $4)
                """,
                price_records,
            )
            results["price_history"] = len(price_records)

        # Update brand statistics
        await self.conn.execute(
            """
            UPDATE brands b SET
                product_count = (SELECT COUNT(*) FROM products WHERE brand = b.name),
                avg_rating = (SELECT AVG(stars) FROM products WHERE brand = b.name),
                avg_price = (SELECT AVG(price) FROM products WHERE brand = b.name),
                review_count = (SELECT SUM(reviews_count) FROM products WHERE brand = b.name),
                updated_at = CURRENT_TIMESTAMP
            """
        )

        return results


class QdrantLoader:
    """Load data into Qdrant with multiple indexing strategies."""

    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        vector_size: int = 768,
        strategy: str = "add_child_node",
        payload_config: dict | None = None,
    ):
        """Initialize loader.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection: Collection name
            vector_size: Embedding vector dimensions
            strategy: Indexing strategy:
                - 'parent_only': Insert parent nodes only (for original mode)
                - 'enrich_existing': Update existing parents + add child nodes (for enrich mode)
                - 'full_replace': Delete and re-insert all points
            payload_config: Lean payload configuration from config file
        """
        self.host = host
        self.port = port
        self.collection = collection
        self.vector_size = vector_size
        self.strategy = strategy
        self.payload_config = payload_config or get_qdrant_payload_fields()
        self.client = None

    async def connect(self):
        """Connect to Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        self.client = QdrantClient(host=self.host, port=self.port)

        # Create collection if not exists
        collections = self.client.get_collections().collections
        existing_collection = next((c for c in collections if c.name == self.collection), None)

        if existing_collection:
            # Check if dimensions match, recreate if different
            collection_info = self.client.get_collection(self.collection)
            existing_size = collection_info.config.params.vectors.size
            if existing_size != self.vector_size:
                logger.warning(
                    "collection_dimension_mismatch",
                    collection=self.collection,
                    existing_size=existing_size,
                    required_size=self.vector_size,
                )
                # Delete and recreate with correct dimensions
                self.client.delete_collection(self.collection)
                self._create_collection_with_indexes()
        else:
            self._create_collection_with_indexes()

    def _create_collection_with_indexes(self):
        """Create collection with payload indexes for efficient filtering."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        # Create collection
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("collection_created", collection=self.collection, vector_size=self.vector_size)

        # Create payload indexes for efficient filtering
        # These match the schema design and align with Elasticsearch filter fields
        payload_indexes = {
            # Core identification
            "node_type": PayloadSchemaType.KEYWORD,
            "asin": PayloadSchemaType.KEYWORD,
            "parent_asin": PayloadSchemaType.KEYWORD,
            # Filtering fields
            "brand": PayloadSchemaType.KEYWORD,
            "product_type": PayloadSchemaType.KEYWORD,
            "category_level1": PayloadSchemaType.KEYWORD,
            "category_level2": PayloadSchemaType.KEYWORD,
            "category_level3": PayloadSchemaType.KEYWORD,
            "section": PayloadSchemaType.KEYWORD,
            "availability": PayloadSchemaType.KEYWORD,
            # Numeric filters
            "price": PayloadSchemaType.FLOAT,
            "list_price": PayloadSchemaType.FLOAT,
            "stars": PayloadSchemaType.FLOAT,
            "reviews_count": PayloadSchemaType.INTEGER,
            # Boolean filters
            "is_best_seller": PayloadSchemaType.BOOL,
            "prime_eligible": PayloadSchemaType.BOOL,
        }

        for field_name, schema_type in payload_indexes.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug("payload_index_created", field=field_name, type=str(schema_type))
            except Exception as e:
                # Index may already exist
                logger.debug("payload_index_exists", field=field_name, error=str(e))

        logger.info("payload_indexes_created", count=len(payload_indexes))

    def close(self):
        """Close connection."""
        if self.client:
            self.client.close()

    def _build_lean_parent_payload(self, row: pd.Series) -> dict:
        """Build lean payload for parent node based on config."""
        parent_fields = self.payload_config.get("parent_fields", [])
        payload = {
            "node_type": "parent",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

        for field in parent_fields:
            value = row.get(field)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                payload[field] = value

        return payload

    def _build_lean_child_payload(self, row: pd.Series, parent_asin: str) -> dict:
        """Build lean payload for child node based on config."""
        max_length = self.payload_config.get("content_preview_max_length", 200)
        section = row.get("node_type", "unknown")

        # Get content for preview
        content = row.get("text", "") or ""
        content_preview = content[:max_length] + "..." if len(content) > max_length else content

        payload = {
            "node_type": "child",
            "parent_asin": parent_asin,
            "section": section,
            "content_preview": content_preview,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add inherited filter fields from parent for filtering
        for field in ["category_level1", "brand", "price", "stars"]:
            value = row.get(field)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                payload[field] = value

        return payload

    def _build_merge_payload(self, row: pd.Series) -> dict:
        """Build full payload for merge_payload strategy (single point per product)."""
        payload = {
            "asin": row.get("asin"),
            "title": row.get("title"),
            "short_title": row.get("short_title"),
            "brand": row.get("brand"),
            "product_type": row.get("product_type"),
            "product_type_keywords": row.get("product_type_keywords"),
            "price": row.get("price"),
            "list_price": row.get("list_price"),
            "stars": row.get("stars"),
            "reviews_count": row.get("reviews_count"),
            "category_level1": row.get("category_level1"),
            "category_level2": row.get("category_level2"),
            "category_level3": row.get("category_level3"),
            "img_url": row.get("img_url"),
            "is_best_seller": row.get("is_best_seller"),
            "availability": row.get("availability"),
            # GenAI quick-answer fields
            "genAI_summary": row.get("genAI_summary"),
            "genAI_primary_function": row.get("genAI_primary_function"),
            "genAI_best_for": row.get("genAI_best_for"),
            "genAI_use_cases": row.get("genAI_use_cases"),
            "genAI_target_audience": row.get("genAI_target_audience"),
            "genAI_key_capabilities": row.get("genAI_key_capabilities"),
            "genAI_unique_selling_points": row.get("genAI_unique_selling_points"),
            "genAI_value_score": row.get("genAI_value_score"),
            # Computed scores (if available)
            "popularity_score": row.get("popularity_score"),
            "trending_rank": row.get("trending_rank"),
            "price_percentile": row.get("price_percentile"),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Remove None/NaN values
        return {k: v for k, v in payload.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

    async def load_vectors_merge_payload(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Load vectors using merge_payload strategy - single point per product."""
        from qdrant_client.models import PointStruct

        logger.info("using_strategy", strategy="merge_payload")

        # Only use parent nodes (one per product)
        parents = df[df["node_type"] == "parent"].copy()
        total_loaded = 0

        for i in tqdm(range(0, len(parents), batch_size), desc="Loading (merge_payload)"):
            batch = parents.iloc[i:i + batch_size]
            points = []

            for _, row in batch.iterrows():
                embedding = row.get("embedding")
                if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
                    continue

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                payload = self._build_merge_payload(row)
                asin = row.get("asin", str(uuid4()))

                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(collection_name=self.collection, points=points)
                total_loaded += len(points)

        return total_loaded

    async def load_vectors_add_child_node(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Load vectors using add_child_node strategy - parent + child nodes."""
        from qdrant_client.models import PointStruct

        logger.info("using_strategy", strategy="add_child_node")

        total_loaded = 0

        for i in tqdm(range(0, len(df), batch_size), desc="Loading (add_child_node)"):
            batch = df.iloc[i:i + batch_size]
            points = []

            for _, row in batch.iterrows():
                embedding = row.get("embedding")
                if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
                    continue

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                node_type = row.get("node_type")
                asin = row.get("asin")

                if node_type == "parent":
                    payload = self._build_lean_parent_payload(row)
                else:
                    payload = self._build_lean_child_payload(row, asin)

                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(collection_name=self.collection, points=points)
                total_loaded += len(points)

        return total_loaded

    async def load_vectors_full_replace(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Load vectors using full_replace strategy - delete all and re-insert."""
        logger.info("using_strategy", strategy="full_replace")

        # Delete and recreate collection with indexes
        logger.info("deleting_collection_for_full_replace", collection=self.collection)
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass  # Collection may not exist

        # Create collection with payload indexes for filtering
        self._create_collection_with_indexes()
        logger.info("collection_recreated_for_full_replace", collection=self.collection)

        # Use add_child_node logic for insertion
        return await self.load_vectors_add_child_node(df, batch_size)

    async def load_vectors_parent_only(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Load vectors using parent_only strategy - insert parent nodes only.

        Used in original mode: creates parent nodes with basic product info.
        """
        from qdrant_client.models import PointStruct

        logger.info("using_strategy", strategy="parent_only")

        # Only use parent nodes
        parents = df[df["node_type"] == "parent"].copy()
        total_loaded = 0

        for i in tqdm(range(0, len(parents), batch_size), desc="Loading (parent_only)"):
            batch = parents.iloc[i:i + batch_size]
            points = []

            for _, row in batch.iterrows():
                embedding = row.get("embedding")
                if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
                    continue

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Build payload with basic product fields
                payload = {
                    "node_type": "parent",
                    "asin": row.get("asin"),
                    "title": row.get("title"),
                    "price": row.get("price"),
                    "list_price": row.get("list_price"),
                    "stars": row.get("stars"),
                    "reviews_count": row.get("reviews_count"),
                    "category_level1": row.get("category_level1"),
                    "is_best_seller": row.get("is_best_seller"),
                    "img_url": row.get("img_url"),
                    "product_url": row.get("product_url"),
                    "bought_in_last_month": row.get("bought_in_last_month"),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Remove None/NaN values
                payload = {k: v for k, v in payload.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(collection_name=self.collection, points=points)
                total_loaded += len(points)

        return total_loaded

    async def load_vectors_enrich_existing(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Load vectors using enrich_existing strategy - update parents + add children.

        Used in enrich mode:
        1. Find existing parent nodes by ASIN
        2. Update parent nodes with enriched fields (new embedding + payload)
        3. Insert new child nodes linked to existing parents
        """
        from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

        logger.info("using_strategy", strategy="enrich_existing")

        total_loaded = 0
        updated_parents = 0
        new_children = 0

        # Separate parent updates from child nodes
        parent_updates = df[df["node_type"] == "parent_update"].copy()
        child_nodes = df[df["node_type"].isin(["description", "features", "specs", "reviews", "use_cases"])].copy()

        # Step 1: Update existing parent nodes
        logger.info("updating_parent_nodes", count=len(parent_updates))
        for i in tqdm(range(0, len(parent_updates), batch_size), desc="Updating parents"):
            batch = parent_updates.iloc[i:i + batch_size]

            for _, row in batch.iterrows():
                asin = row.get("asin")
                embedding = row.get("embedding")

                if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
                    continue

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Find existing parent point by ASIN
                search_result = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="asin", match=MatchValue(value=asin)),
                            FieldCondition(key="node_type", match=MatchValue(value="parent")),
                        ]
                    ),
                    limit=1,
                )

                if search_result and search_result[0]:
                    # Update existing parent with enriched data
                    existing_point = search_result[0][0]
                    existing_payload = existing_point.payload or {}

                    # Merge enriched fields into existing payload
                    enriched_payload = {
                        # Basic fields from 02c
                        "brand": row.get("brand"),
                        "short_title": row.get("short_title"),
                        "product_type": row.get("product_type"),
                        "product_type_keywords": row.get("product_type_keywords"),
                        "availability": row.get("availability"),
                        # Parent fields from 02c
                        "genAI_summary": row.get("genAI_summary"),
                        "genAI_primary_function": row.get("genAI_primary_function"),
                        "genAI_best_for": row.get("genAI_best_for"),
                        "genAI_use_cases": row.get("genAI_use_cases"),
                        "genAI_target_audience": row.get("genAI_target_audience"),
                        "genAI_key_capabilities": row.get("genAI_key_capabilities"),
                        "genAI_unique_selling_points": row.get("genAI_unique_selling_points"),
                        "genAI_value_score": row.get("genAI_value_score"),
                        "enriched_at": datetime.now(timezone.utc).isoformat(),
                    }

                    # Remove None/NaN values
                    enriched_payload = {k: v for k, v in enriched_payload.items()
                                        if v is not None and not (isinstance(v, float) and np.isnan(v))}

                    # Merge existing + enriched
                    merged_payload = {**existing_payload, **enriched_payload}

                    # Update the point with new embedding and merged payload
                    self.client.upsert(
                        collection_name=self.collection,
                        points=[
                            PointStruct(
                                id=existing_point.id,
                                vector=embedding,
                                payload=merged_payload,
                            )
                        ],
                    )
                    updated_parents += 1
                else:
                    logger.warning("parent_not_found_for_update", asin=asin)

        # Step 2: Insert new child nodes
        logger.info("inserting_child_nodes", count=len(child_nodes))
        for i in tqdm(range(0, len(child_nodes), batch_size), desc="Loading children"):
            batch = child_nodes.iloc[i:i + batch_size]
            points = []

            for _, row in batch.iterrows():
                embedding = row.get("embedding")
                if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
                    continue

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                asin = row.get("asin")
                section = row.get("section") or row.get("node_type")
                text = row.get("text", "")

                # Build child node payload
                payload = {
                    "node_type": section,
                    "asin": asin,
                    "parent_asin": asin,
                    "section": section,
                    "content_preview": str(text)[:200] if text else "",
                    # Inherited fields for filtering
                    "title": row.get("title"),
                    "brand": row.get("brand"),
                    "price": row.get("price"),
                    "stars": row.get("stars"),
                    "category_level1": row.get("category_level1"),
                    "is_best_seller": row.get("is_best_seller"),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Remove None/NaN values
                payload = {k: v for k, v in payload.items()
                           if v is not None and not (isinstance(v, float) and np.isnan(v))}

                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(collection_name=self.collection, points=points)
                new_children += len(points)

        total_loaded = updated_parents + new_children
        logger.info("enrich_complete", updated_parents=updated_parents, new_children=new_children)
        return total_loaded

    async def load_vectors(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Load vectors using configured strategy."""
        if self.strategy == "parent_only":
            return await self.load_vectors_parent_only(df, batch_size)
        elif self.strategy == "enrich_existing":
            return await self.load_vectors_enrich_existing(df, batch_size)
        elif self.strategy == "full_replace":
            return await self.load_vectors_full_replace(df, batch_size)
        # Legacy strategies (for backwards compatibility)
        elif self.strategy == "merge_payload":
            return await self.load_vectors_merge_payload(df, batch_size)
        else:  # add_child_node (default)
            return await self.load_vectors_add_child_node(df, batch_size)


class ElasticsearchLoader:
    """Load data into Elasticsearch."""

    def __init__(self, host: str, port: int, index: str):
        """Initialize loader."""
        self.host = host
        self.port = port
        self.index = index
        self.client = None

    async def connect(self):
        """Connect to Elasticsearch."""
        from elasticsearch import AsyncElasticsearch

        self.client = AsyncElasticsearch([f"http://{self.host}:{self.port}"])

        # Create index if not exists
        if not await self.client.indices.exists(index=self.index):
            # Use a mapping that supports full-text search on product data
            mapping = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "autocomplete_analyzer": {
                                "type": "custom",
                                "tokenizer": "autocomplete_tokenizer",
                                "filter": ["lowercase"]
                            }
                        },
                        "tokenizer": {
                            "autocomplete_tokenizer": {
                                "type": "edge_ngram",
                                "min_gram": 2,
                                "max_gram": 20,
                                "token_chars": ["letter", "digit"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "asin": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "autocomplete": {
                                    "type": "text",
                                    "analyzer": "autocomplete_analyzer",
                                    "search_analyzer": "standard"
                                }
                            }
                        },
                        "short_title": {"type": "text"},
                        "brand": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "product_type": {"type": "keyword"},
                        "product_type_keywords": {"type": "text"},
                        "price": {"type": "float"},
                        "list_price": {"type": "float"},
                        "stars": {"type": "float"},
                        "reviews_count": {"type": "integer"},
                        "category_level1": {"type": "keyword"},
                        "category_level2": {"type": "keyword"},
                        "category_level3": {"type": "keyword"},
                        "category_name": {"type": "text"},
                        "is_best_seller": {"type": "boolean"},
                        "prime_eligible": {"type": "boolean"},
                        "availability": {"type": "keyword"},
                        "img_url": {"type": "keyword"},
                        "product_url": {"type": "keyword"},
                        "product_description": {"type": "text"},
                        # Chunk sections (5 sections for Multi-Chunk)
                        "chunk_description": {"type": "text"},
                        "chunk_features": {"type": "text"},
                        "chunk_specs": {"type": "text"},
                        "chunk_reviews": {"type": "text"},
                        "chunk_use_cases": {"type": "text"},
                        # GenAI enrichment fields for search
                        "genAI_summary": {"type": "text"},
                        "genAI_primary_function": {"type": "text"},
                        "genAI_best_for": {"type": "text"},
                        "genAI_use_cases": {"type": "text"},
                        "genAI_target_audience": {"type": "text"},
                        "genAI_key_capabilities": {"type": "text"},
                        "genAI_unique_selling_points": {"type": "text"},
                        "genAI_value_score": {"type": "float"},
                        "indexed_at": {"type": "date"},
                    }
                }
            }
            # ES 8.x uses separate settings and mappings parameters
            await self.client.indices.create(
                index=self.index,
                settings=mapping["settings"],
                mappings=mapping["mappings"]
            )
            logger.info("index_created", index=self.index)

    async def close(self):
        """Close connection."""
        if self.client:
            await self.client.close()

    def _clean_value(self, value: Any) -> Any:
        """Clean value for Elasticsearch - convert NaN to None."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    async def load_documents(self, df: pd.DataFrame, batch_size: int = 500) -> int:
        """Load documents to Elasticsearch."""
        from elasticsearch.helpers import async_bulk

        # Get unique products (parent nodes) and collect text content from child nodes
        products = df[df["node_type"] == "parent"].copy()

        # Build a map of asin -> text content from child nodes (5 sections)
        child_content = {}
        for _, row in df[df["node_type"] != "parent"].iterrows():
            asin = row.get("asin")
            node_type = row.get("node_type")
            text = row.get("text", "")

            if asin not in child_content:
                child_content[asin] = {}

            if node_type == "description":
                child_content[asin]["chunk_description"] = self._clean_value(text)
            elif node_type == "features":
                child_content[asin]["chunk_features"] = self._clean_value(text)
            elif node_type == "specs":
                child_content[asin]["chunk_specs"] = self._clean_value(text)
            elif node_type == "reviews":
                child_content[asin]["chunk_reviews"] = self._clean_value(text)
            elif node_type == "use_cases":
                child_content[asin]["chunk_use_cases"] = self._clean_value(text)

        async def generate_actions():
            for _, row in products.iterrows():
                asin = row.get("asin")
                chunks = child_content.get(asin, {})

                source = {
                    "asin": self._clean_value(asin),
                    "title": self._clean_value(row.get("title")),
                    "short_title": self._clean_value(row.get("short_title")),
                    "brand": self._clean_value(row.get("brand")),
                    "product_type": self._clean_value(row.get("product_type")),
                    "product_type_keywords": self._clean_value(row.get("product_type_keywords")),
                    "price": self._clean_value(row.get("price")),
                    "list_price": self._clean_value(row.get("list_price")),
                    "stars": self._clean_value(row.get("stars")),
                    "reviews_count": self._clean_value(row.get("reviews_count")),
                    "category_level1": self._clean_value(row.get("category_level1")),
                    "category_level2": self._clean_value(row.get("category_level2")),
                    "category_level3": self._clean_value(row.get("category_level3")),
                    "category_name": self._clean_value(row.get("category_level1")),  # Alias for search
                    "is_best_seller": bool(row.get("is_best_seller", False)),
                    "prime_eligible": bool(row.get("prime_eligible", False)),
                    "availability": self._clean_value(row.get("availability")),
                    "img_url": self._clean_value(row.get("img_url")),
                    "product_url": self._clean_value(row.get("product_url")),
                    # Chunk sections (5 sections for Multi-Chunk)
                    "chunk_description": chunks.get("chunk_description"),
                    "chunk_features": chunks.get("chunk_features"),
                    "chunk_specs": chunks.get("chunk_specs"),
                    "chunk_reviews": chunks.get("chunk_reviews"),
                    "chunk_use_cases": chunks.get("chunk_use_cases"),
                    # GenAI enrichment fields for full-text search
                    "genAI_summary": self._clean_value(row.get("genAI_summary")),
                    "genAI_primary_function": self._clean_value(row.get("genAI_primary_function")),
                    "genAI_best_for": self._clean_value(row.get("genAI_best_for")),
                    "genAI_use_cases": self._clean_value(row.get("genAI_use_cases")),
                    "genAI_target_audience": self._clean_value(row.get("genAI_target_audience")),
                    "genAI_key_capabilities": self._clean_value(row.get("genAI_key_capabilities")),
                    "genAI_unique_selling_points": self._clean_value(row.get("genAI_unique_selling_points")),
                    "genAI_value_score": self._clean_value(row.get("genAI_value_score")),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
                # Remove None values
                source = {k: v for k, v in source.items() if v is not None}

                doc = {
                    "_index": self.index,
                    "_id": asin,
                    "_source": source,
                }
                yield doc

        success, failed = await async_bulk(
            self.client,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False,
        )

        return success


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help="Input Parquet file path",
)
@click.option(
    "--postgres-dsn",
    default=None,
    help="PostgreSQL connection string",
)
@click.option(
    "--qdrant-host",
    default=None,
    help="Qdrant host",
)
@click.option(
    "--qdrant-port",
    type=int,
    default=None,
    help="Qdrant port",
)
@click.option(
    "--qdrant-collection",
    default=None,
    help="Qdrant collection name",
)
@click.option(
    "--elasticsearch-host",
    default=None,
    help="Elasticsearch host",
)
@click.option(
    "--elasticsearch-port",
    type=int,
    default=None,
    help="Elasticsearch port",
)
@click.option(
    "--elasticsearch-index",
    default=None,
    help="Elasticsearch index name",
)
@click.option(
    "--skip-postgres",
    is_flag=True,
    help="Skip PostgreSQL loading",
)
@click.option(
    "--skip-qdrant",
    is_flag=True,
    help="Skip Qdrant loading",
)
@click.option(
    "--skip-elasticsearch",
    is_flag=True,
    help="Skip Elasticsearch loading",
)
@click.option(
    "--init-schema",
    is_flag=True,
    help="Initialize PostgreSQL schema before loading",
)
@click.option(
    "--strategy",
    type=click.Choice(["parent_only", "enrich_existing", "full_replace", "merge_payload", "add_child_node"]),
    default=None,
    help="Qdrant indexing strategy. Auto-selected based on mode if not specified (parent_only for original, enrich_existing for enrich).",
)
@click.option(
    "--vector-size",
    type=int,
    default=None,
    help="Vector dimensions (auto-detected from data if not specified)",
)
@click.option(
    "--count",
    "row_count",
    type=int,
    default=None,
    help="Limit number of products to load (defaults to all)",
)
@click.option(
    "--metrics-output",
    "metrics_path",
    type=click.Path(),
    default=None,
    help="Metrics output file path",
)
def main(
    input_path: str | None,
    postgres_dsn: str | None,
    qdrant_host: str | None,
    qdrant_port: int | None,
    qdrant_collection: str | None,
    elasticsearch_host: str | None,
    elasticsearch_port: int | None,
    elasticsearch_index: str | None,
    skip_postgres: bool,
    skip_qdrant: bool,
    skip_elasticsearch: bool,
    init_schema: bool,
    strategy: str | None,
    vector_size: int | None,
    row_count: int | None,
    metrics_path: str | None,
):
    """Load data into multiple stores."""
    start_time = time.time()

    # Load settings
    settings = get_settings()
    script_name = "05_load_stores"
    pipeline_mode = cfg.get_mode(script_name)
    product_count = cfg.get_count()

    # Use config file values, then CLI overrides (with {mode} in filename)
    input_path = input_path or str(cfg.get_path(script_name, "input", str(settings.embedded_data_dir / f"mvp_{product_count}_{pipeline_mode}_embedded.parquet")))
    postgres_dsn = postgres_dsn or cfg.get_script(script_name, "postgres_dsn", settings.postgres_dsn)
    qdrant_host = qdrant_host or cfg.get_script(script_name, "qdrant_host", settings.qdrant_host)
    qdrant_port = qdrant_port or cfg.get_script(script_name, "qdrant_port", settings.qdrant_port)
    qdrant_collection = qdrant_collection or cfg.get_script(script_name, "qdrant_collection", settings.qdrant_collection)
    elasticsearch_host = elasticsearch_host or cfg.get_script(script_name, "elasticsearch_host", settings.elasticsearch_host)
    elasticsearch_port = elasticsearch_port or cfg.get_script(script_name, "elasticsearch_port", settings.elasticsearch_port)
    elasticsearch_index = elasticsearch_index or cfg.get_script(script_name, "elasticsearch_index", settings.elasticsearch_index)
    metrics_path = metrics_path or str(cfg.get_path(script_name, "metrics", str(settings.metrics_dir / "05_loading_metrics.json")))
    row_count = row_count if row_count is not None else cfg.get_script(script_name, "count")
    vector_size = vector_size if vector_size is not None else cfg.get_script(script_name, "vector_size")

    # Get indexing strategy - mode-aware auto-selection if not specified
    if strategy:
        indexing_strategy = strategy
    else:
        config_strategy = cfg.get_indexing_strategy(script_name)
        if config_strategy:
            indexing_strategy = config_strategy
        else:
            # Auto-select based on pipeline mode
            indexing_strategy = "parent_only" if pipeline_mode == "original" else "enrich_existing"

    payload_config = cfg.get_qdrant_payload_fields(script_name)

    # Get skip flags from config if not set via CLI
    skip_postgres = skip_postgres if skip_postgres else cfg.get_script(script_name, "skip_postgres", False)
    skip_qdrant = skip_qdrant if skip_qdrant else cfg.get_script(script_name, "skip_qdrant", False)
    skip_elasticsearch = skip_elasticsearch if skip_elasticsearch else cfg.get_script(script_name, "skip_elasticsearch", False)

    input_path = Path(input_path)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "starting_multi_store_loading",
        input_path=str(input_path),
        indexing_strategy=indexing_strategy,
        pipeline_mode=pipeline_mode,
    )

    metrics = {
        "stage": "loading",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "indexing_strategy": indexing_strategy,
        "pipeline_mode": pipeline_mode,
    }

    async def run():
        # Load data
        df = pd.read_parquet(input_path)
        total_nodes = len(df)

        # Count unique products based on mode
        if pipeline_mode == "original":
            unique_products = df[df["node_type"] == "parent"]["asin"].nunique()
        else:
            # Enrich mode: count parent_update nodes
            unique_products = df[df["node_type"] == "parent_update"]["asin"].nunique()

        logger.info("data_loaded", total_nodes=total_nodes, unique_products=unique_products, mode=pipeline_mode)

        # Limit by unique products if specified
        if row_count and row_count < unique_products:
            # Get first N unique ASINs based on mode
            if pipeline_mode == "original":
                limited_asins = df[df["node_type"] == "parent"]["asin"].head(row_count).tolist()
            else:
                limited_asins = df[df["node_type"] == "parent_update"]["asin"].head(row_count).tolist()
            df = df[df["asin"].isin(limited_asins)]
            total_nodes = len(df)
            unique_products = row_count
            logger.info("rows_limited", products=row_count, nodes=total_nodes)

        # Auto-detect vector dimensions from data if not specified
        detected_vector_size = vector_size
        if detected_vector_size is None:
            # Get dimensions from first valid embedding
            for embedding in df["embedding"]:
                if embedding is not None and len(embedding) > 0:
                    detected_vector_size = len(embedding)
                    logger.info("auto_detected_vector_size", vector_size=detected_vector_size)
                    break
            if detected_vector_size is None:
                detected_vector_size = 768  # Default fallback
                logger.warning("using_default_vector_size", vector_size=detected_vector_size)

        results = {
            "total_nodes": total_nodes,
            "unique_products": unique_products,
            "vector_size": detected_vector_size,
            "stores": {},
        }

        # Load to PostgreSQL
        if not skip_postgres:
            logger.info("loading_postgres", init_schema=init_schema, mode=pipeline_mode)
            pg_loader = PostgresLoader(postgres_dsn, mode=pipeline_mode)
            try:
                await pg_loader.connect()

                # Initialize schema if requested
                if init_schema:
                    logger.info("initializing_postgres_schema")
                    await pg_loader.init_schema()

                pg_result = await pg_loader.load_products(df)
                results["stores"]["postgres"] = {
                    "status": "success",
                    "records_loaded": pg_result["products"],
                    "brands_loaded": pg_result["brands"],
                    "price_history_records": pg_result["price_history"],
                }
                logger.info(
                    "postgres_loaded",
                    products=pg_result["products"],
                    brands=pg_result["brands"],
                    price_history=pg_result["price_history"],
                )
            except Exception as e:
                logger.error("postgres_failed", error=str(e))
                results["stores"]["postgres"] = {
                    "status": "failed",
                    "error": str(e),
                }
            finally:
                await pg_loader.close()

        # Load to Qdrant
        if not skip_qdrant:
            logger.info("loading_qdrant", vector_size=detected_vector_size, strategy=indexing_strategy)
            qdrant_loader = QdrantLoader(
                host=qdrant_host,
                port=qdrant_port,
                collection=qdrant_collection,
                vector_size=detected_vector_size,
                strategy=indexing_strategy,
                payload_config=payload_config,
            )
            try:
                await qdrant_loader.connect()
                qdrant_count = await qdrant_loader.load_vectors(df)
                results["stores"]["qdrant"] = {
                    "status": "success",
                    "vectors_loaded": qdrant_count,
                    "collection": qdrant_collection,
                    "strategy": indexing_strategy,
                }
                logger.info("qdrant_loaded", count=qdrant_count)
            except Exception as e:
                logger.error("qdrant_failed", error=str(e))
                results["stores"]["qdrant"] = {
                    "status": "failed",
                    "error": str(e),
                }
            finally:
                qdrant_loader.close()

        # Load to Elasticsearch
        if not skip_elasticsearch:
            logger.info("loading_elasticsearch")
            es_loader = ElasticsearchLoader(
                elasticsearch_host, elasticsearch_port, elasticsearch_index
            )
            try:
                await es_loader.connect()
                es_count = await es_loader.load_documents(df)
                results["stores"]["elasticsearch"] = {
                    "status": "success",
                    "documents_loaded": es_count,
                    "index": elasticsearch_index,
                }
                logger.info("elasticsearch_loaded", count=es_count)
            except Exception as e:
                import traceback
                logger.error("elasticsearch_failed", error=str(e), traceback=traceback.format_exc())
                results["stores"]["elasticsearch"] = {
                    "status": "failed",
                    "error": str(e),
                }
            finally:
                await es_loader.close()

        return results

    try:
        result = asyncio.run(run())

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate overall success
        store_statuses = [s.get("status") for s in result["stores"].values()]
        overall_status = "success" if all(s == "success" for s in store_statuses) else "partial"

        metrics.update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": overall_status,
            "processing_time_seconds": round(processing_time, 2),
            "metrics": result,
        })

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("loading_complete", status=overall_status)

        # Print summary
        print("\n" + "=" * 60)
        print("MULTI-STORE LOADING COMPLETE")
        print("=" * 60)
        print(f"Pipeline mode:       {pipeline_mode}")
        print(f"Indexing strategy:   {indexing_strategy}")
        print(f"Total nodes:         {result['total_nodes']:,}")
        print(f"Unique products:     {result['unique_products']:,}")
        print(f"Vector dimensions:   {result.get('vector_size', 'N/A')}")
        print(f"Processing time:     {processing_time:.2f}s")
        print(f"\nStore Results:")
        for store, info in result["stores"].items():
            status = info.get("status", "skipped")
            if status == "success":
                if store == "postgres":
                    products = info.get("records_loaded", 0)
                    brands = info.get("brands_loaded", 0)
                    price_hist = info.get("price_history_records", 0)
                    print(f"  - {store}: ✓ {products:,} products, {brands:,} brands, {price_hist:,} price records")
                else:
                    count = info.get("vectors_loaded") or info.get("documents_loaded") or 0
                    print(f"  - {store}: ✓ {count:,} loaded")
            elif status == "failed":
                print(f"  - {store}: ✗ {info.get('error', 'Unknown error')}")
        print("=" * 60)

    except Exception as e:
        logger.error("loading_failed", error=str(e))
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
