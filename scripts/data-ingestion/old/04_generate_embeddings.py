#!/usr/bin/env python3
"""
Script 04: Generate Embeddings

Generate embeddings for products using Ollama service.
Implements Multi-Chunk strategy with parent and child nodes.

Usage:
    python scripts/04_generate_embeddings.py --input data/cleaned/mvp_100k_cleaned.csv
    python scripts/04_generate_embeddings.py --input data/cleaned/mvp_100k_cleaned.csv --model nomic-embed-text
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


# Known dimensions for common embedding models
MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
    "bge-base": 768,
    "e5-large": 1024,
    "e5-base": 768,
}


def get_model_dimensions(model: str, default: int = 768) -> int:
    """Get embedding dimensions for a model.

    Args:
        model: Model name
        default: Default dimensions if model not found

    Returns:
        Number of dimensions
    """
    if model in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model]

    # Check for partial matches
    for known_model, dims in MODEL_DIMENSIONS.items():
        if known_model in model.lower():
            return dims

    return default


class EmbeddingGenerator:
    """Generate embeddings using Ollama service."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:8010",
        model: str = "nomic-embed-text",
        batch_size: int = 100,
        dimensions: int | None = None,
    ):
        """Initialize embedding generator.

        Args:
            ollama_url: URL of the ollama-service
            model: Embedding model to use
            batch_size: Batch size for embedding generation
            dimensions: Embedding dimensions (auto-detected if None)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions or get_model_dimensions(model)
        self.client: httpx.AsyncClient | None = None

        # Stats
        self.success_count = 0
        self.error_count = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.ollama_url,
                timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
            )
        return self.client

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def check_model_available(self) -> bool:
        """Check if the embedding model is available.

        Returns:
            True if model is available
        """
        client = await self._get_client()
        try:
            # Try Ollama API endpoint first
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            model_base = self.model.split(":")[0]
            return model_base in models or any(model_base in m for m in models)
        except httpx.HTTPStatusError:
            # Fallback to custom embedding service endpoint
            try:
                response = await client.get("/models")
                response.raise_for_status()
                data = response.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                model_base = self.model.split(":")[0]
                return model_base in models or any(model_base in m for m in models)
            except Exception as e:
                logger.warning("check_model_failed", error=str(e))
                return False
        except Exception as e:
            logger.warning("check_model_failed", error=str(e))
            return False

    async def pull_model(self) -> bool:
        """Pull the embedding model if not available.

        Returns:
            True if model is ready
        """
        client = await self._get_client()
        try:
            logger.info("pulling_model", model=self.model)
            # Try Ollama API endpoint first
            response = await client.post(
                "/api/pull",
                json={"name": self.model},
                timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
            )
            response.raise_for_status()
            logger.info("model_pulled", model=self.model)
            return True
        except httpx.HTTPStatusError:
            # Fallback to custom embedding service endpoint
            try:
                response = await client.post(
                    "/models/pull",
                    json={"name": self.model},
                    timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
                )
                response.raise_for_status()
                logger.info("model_pulled", model=self.model)
                return True
            except Exception as e:
                logger.error("pull_model_failed", model=self.model, error=str(e))
                return False
        except Exception as e:
            logger.error("pull_model_failed", model=self.model, error=str(e))
            return False

    async def ensure_model(self) -> bool:
        """Ensure the model is available, pulling if necessary.

        Returns:
            True if model is ready
        """
        if await self.check_model_available():
            return True
        return await self.pull_model()

    async def _embed_single_ollama(self, text: str) -> list[float]:
        """Generate embedding using native Ollama API (single text).

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        client = await self._get_client()
        response = await client.post(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])

    async def _embed_batch_custom(self, texts: list[str]) -> list[list[float]] | None:
        """Try custom batch embedding service.

        Returns:
            List of embeddings or None if service not available
        """
        client = await self._get_client()
        try:
            response = await client.post(
                "/embed",
                json={"texts": texts, "model": self.model},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None  # Endpoint not available
            raise

    async def get_dimensions_from_service(self) -> int:
        """Get actual dimensions from the service by generating a test embedding.

        Returns:
            Embedding dimensions
        """
        try:
            # Try custom batch endpoint first
            embeddings = await self._embed_batch_custom(["test"])
            if embeddings and embeddings[0]:
                actual_dims = len(embeddings[0])
                if actual_dims != self.dimensions:
                    logger.info(
                        "dimensions_updated",
                        model=self.model,
                        configured=self.dimensions,
                        actual=actual_dims,
                    )
                    self.dimensions = actual_dims
                return actual_dims
        except Exception:
            pass

        try:
            # Fallback to Ollama native API
            embedding = await self._embed_single_ollama("test")
            if embedding:
                actual_dims = len(embedding)
                if actual_dims != self.dimensions:
                    logger.info(
                        "dimensions_updated",
                        model=self.model,
                        configured=self.dimensions,
                        actual=actual_dims,
                    )
                    self.dimensions = actual_dims
                return actual_dims
        except Exception as e:
            logger.warning("get_dimensions_failed", error=str(e))

        return self.dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Try custom batch endpoint first (faster)
            embeddings = await self._embed_batch_custom(texts)
            if embeddings is not None:
                self.success_count += len(texts)
                return embeddings

            # Fallback to Ollama native API (one at a time)
            embeddings = []
            for text in texts:
                try:
                    embedding = await self._embed_single_ollama(text)
                    embeddings.append(embedding)
                    self.success_count += 1
                except Exception as e:
                    logger.warning("single_embedding_failed", error=str(e))
                    embeddings.append([0.0] * self.dimensions)
                    self.error_count += 1
            return embeddings

        except Exception as e:
            logger.error("embedding_batch_failed", error=str(e), batch_size=len(texts))
            self.error_count += len(texts)
            # Return zero vectors for failed batch
            return [[0.0] * self.dimensions for _ in texts]

    def build_parent_text(self, row: pd.Series, parent_text_fields: list[str] | None = None) -> str:
        """Build parent node text for embedding.

        IMPORTANT: Uses simple, natural text format (no "Product:", "Brand:" prefixes)
        to maximize query-document similarity. Plain queries like "plasma cutter"
        should match well with product titles.

        Args:
            row: DataFrame row
            parent_text_fields: List of fields to include (from config)

        Returns:
            Natural text for embedding (title + brand + optional fields)
        """
        # Default fields if not specified in config
        if parent_text_fields is None:
            parent_text_fields = [
                "title", "brand", "category_level1",
                "genAI_summary", "genAI_primary_function", "genAI_best_for"
            ]

        parts = []

        # Primary: Title (most important for matching queries)
        title = row.get('title', '')
        if pd.notna(title) and title:
            parts.append(str(title))

        # Secondary: Brand (prepend if not already in title)
        brand = row.get('brand', '')
        if "brand" in parent_text_fields and pd.notna(brand) and brand:
            # Only add brand if not already at start of title
            if not str(title).lower().startswith(str(brand).lower()):
                parts.insert(0, str(brand))

        # Tertiary: Category (helps with general queries like "welding equipment")
        if "category_level1" in parent_text_fields:
            category = row.get("category_level1") or row.get("categoryName")
            if pd.notna(category) and category:
                parts.append(str(category))

        # GenAI fields (enrich mode only) - use natural text, no labels
        if "genAI_summary" in parent_text_fields and pd.notna(row.get("genAI_summary")):
            parts.append(str(row['genAI_summary']))

        if "genAI_primary_function" in parent_text_fields and pd.notna(row.get("genAI_primary_function")):
            parts.append(str(row['genAI_primary_function']))

        if "genAI_best_for" in parent_text_fields and pd.notna(row.get("genAI_best_for")):
            parts.append(str(row['genAI_best_for']))

        # Join with space for natural text flow (not newlines)
        return " ".join(parts)

    async def generate_multi_chunk_embeddings(
        self,
        df: pd.DataFrame,
        checkpoint_path: Path | None = None,
        child_sections: list[str] | None = None,
        parent_text_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings using Multi-Chunk strategy.

        Creates 6 embeddings per product:
        - 1 parent node (summary with GenAI fields)
        - 5 child nodes (description, features, specs, reviews, use_cases)

        Args:
            df: Input DataFrame
            checkpoint_path: Path for checkpoint file
            child_sections: List of child sections from config
            parent_text_fields: List of fields for parent text from config

        Returns:
            Results dict with nodes and embeddings
        """
        # Default child sections if not specified
        if child_sections is None:
            child_sections = ["description", "features", "specs", "reviews", "use_cases"]

        results = {
            "nodes": [],
            "embeddings": [],
        }

        # Load checkpoint if exists
        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get("processed_count", 0)
                results = checkpoint.get("results", results)
                logger.info("resuming_from_checkpoint", start_idx=start_idx)

        # Process products
        total_products = len(df)
        batch_texts = []
        batch_metadata = []

        for idx, row in tqdm(
            df.iloc[start_idx:].iterrows(),
            total=total_products - start_idx,
            desc="Generating embeddings",
            initial=start_idx,
        ):
            asin = row.get("asin")

            # Build texts for each node type
            node_texts = {"parent": self.build_parent_text(row, parent_text_fields)}

            # Add child sections from config (5 sections: description, features, specs, reviews, use_cases)
            for section in child_sections:
                chunk_col = f"chunk_{section}"
                node_texts[section] = row.get(chunk_col, "")

            # Add to batch
            for node_type, text in node_texts.items():
                if text and len(str(text).strip()) > 10:
                    batch_texts.append(str(text))

                    # Build node metadata - all fields for parent nodes, minimal for child nodes
                    meta = {
                        "asin": asin,
                        "node_type": node_type,
                        "node_id": f"{asin}_{node_type}",
                        "parent_id": f"{asin}_parent" if node_type != "parent" else None,
                        "text": str(text),  # Store text for content_preview in Qdrant
                    }

                    # Add full product metadata only for parent nodes
                    if node_type == "parent":
                        meta.update({
                            # Core product fields
                            "title": row.get("title"),
                            "short_title": row.get("short_title"),
                            "brand": row.get("brand"),
                            "product_type": row.get("product_type"),
                            "product_type_keywords": row.get("product_type_keywords"),
                            # Pricing
                            "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                            "list_price": float(row["list_price"]) if pd.notna(row.get("list_price")) else None,
                            # Ratings
                            "stars": float(row["stars"]) if pd.notna(row.get("stars")) else None,
                            "reviews_count": int(row["reviews_count"]) if pd.notna(row.get("reviews_count")) else None,
                            # Categories
                            "category_level1": row.get("category_level1") or row.get("categoryName"),
                            "category_level2": row.get("category_level2"),
                            "category_level3": row.get("category_level3"),
                            # Flags
                            "is_best_seller": bool(row.get("is_best_seller") or row.get("isBestSeller")),
                            "prime_eligible": bool(row.get("prime_eligible")),
                            "availability": row.get("availability"),
                            # URLs
                            "img_url": row.get("img_url") or row.get("imgUrl"),
                            "product_url": row.get("product_url") or row.get("productURL"),
                            # GenAI Parent Fields (quick-answer)
                            "genAI_summary": row.get("genAI_summary"),
                            "genAI_primary_function": row.get("genAI_primary_function"),
                            "genAI_best_for": row.get("genAI_best_for"),
                            "genAI_use_cases": row.get("genAI_use_cases"),
                            "genAI_target_audience": row.get("genAI_target_audience"),
                            "genAI_key_capabilities": row.get("genAI_key_capabilities"),
                            "genAI_unique_selling_points": row.get("genAI_unique_selling_points"),
                            "genAI_value_score": row.get("genAI_value_score"),
                            # GenAI Description Fields
                            "genAI_detailed_description": row.get("genAI_detailed_description"),
                            "genAI_how_it_works": row.get("genAI_how_it_works"),
                            "genAI_whats_included": row.get("genAI_whats_included"),
                            "genAI_materials": row.get("genAI_materials"),
                            # GenAI Features Fields
                            "genAI_features_detailed": row.get("genAI_features_detailed"),
                            "genAI_standout_features": row.get("genAI_standout_features"),
                            "genAI_technology_explained": row.get("genAI_technology_explained"),
                            "genAI_feature_comparison": row.get("genAI_feature_comparison"),
                            # GenAI Specs Fields
                            "genAI_specs_summary": row.get("genAI_specs_summary"),
                            "genAI_specs_comparison_ready": row.get("genAI_specs_comparison_ready"),
                            "genAI_specs_limitations": row.get("genAI_specs_limitations"),
                            # GenAI Review Analysis Fields
                            "genAI_sentiment_score": row.get("genAI_sentiment_score"),
                            "genAI_sentiment_label": row.get("genAI_sentiment_label"),
                            "genAI_common_praises": row.get("genAI_common_praises"),
                            "genAI_common_complaints": row.get("genAI_common_complaints"),
                            "genAI_durability_feedback": row.get("genAI_durability_feedback"),
                            "genAI_value_for_money_feedback": row.get("genAI_value_for_money_feedback"),
                            # GenAI Use Cases Fields
                            "genAI_use_case_scenarios": row.get("genAI_use_case_scenarios"),
                            "genAI_ideal_user_profiles": row.get("genAI_ideal_user_profiles"),
                            "genAI_not_recommended_for": row.get("genAI_not_recommended_for"),
                            "genAI_problems_solved": row.get("genAI_problems_solved"),
                            # GenAI Pros/Cons
                            "genAI_pros": row.get("genAI_pros"),
                            "genAI_cons": row.get("genAI_cons"),
                        })
                    else:
                        # Child nodes: minimal metadata for filtering
                        meta.update({
                            "title": row.get("title"),
                            "brand": row.get("brand"),
                            "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                            "stars": float(row["stars"]) if pd.notna(row.get("stars")) else None,
                            "category_level1": row.get("category_level1") or row.get("categoryName"),
                            "is_best_seller": bool(row.get("is_best_seller") or row.get("isBestSeller")),
                        })

                    batch_metadata.append(meta)

            # Process batch
            if len(batch_texts) >= self.batch_size:
                embeddings = await self.embed_batch(batch_texts)

                for i, (meta, emb) in enumerate(zip(batch_metadata, embeddings)):
                    results["nodes"].append(meta)
                    results["embeddings"].append(emb)

                batch_texts = []
                batch_metadata = []

                # Save checkpoint periodically
                if checkpoint_path and idx % 1000 == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({
                            "processed_count": idx + 1,
                            "results": results,
                        }, f)

        # Process remaining batch
        if batch_texts:
            embeddings = await self.embed_batch(batch_texts)
            for meta, emb in zip(batch_metadata, embeddings):
                results["nodes"].append(meta)
                results["embeddings"].append(emb)

        return results

    async def generate_mode_aware_embeddings(
        self,
        df: pd.DataFrame,
        pipeline_mode: str = "original",
        checkpoint_path: Path | None = None,
        child_sections: list[str] | None = None,
        parent_text_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings based on pipeline mode.

        Mode-specific behavior:
        - original mode: Generate PARENT embeddings only (from basic product info)
        - enrich mode: Generate CHILD embeddings only (link to existing parents)

        Args:
            df: Input DataFrame
            pipeline_mode: 'original' or 'enrich'
            checkpoint_path: Path for checkpoint file
            child_sections: List of child sections from config
            parent_text_fields: List of fields for parent text from config

        Returns:
            Results dict with nodes and embeddings
        """
        # Default child sections if not specified
        if child_sections is None:
            child_sections = ["description", "features", "specs", "reviews", "use_cases"]

        results = {
            "nodes": [],
            "embeddings": [],
        }

        # Load checkpoint if exists
        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get("processed_count", 0)
                results = checkpoint.get("results", results)
                logger.info("resuming_from_checkpoint", start_idx=start_idx)

        # Process products
        total_products = len(df)
        batch_texts = []
        batch_metadata = []

        logger.info(
            "generating_embeddings",
            mode=pipeline_mode,
            total_products=total_products,
            generate_parent=pipeline_mode == "original",
            generate_children=pipeline_mode == "enrich",
        )

        for idx, row in tqdm(
            df.iloc[start_idx:].iterrows(),
            total=total_products - start_idx,
            desc=f"Generating embeddings ({pipeline_mode} mode)",
            initial=start_idx,
        ):
            asin = row.get("asin")

            if pipeline_mode == "original":
                # ORIGINAL MODE: Generate PARENT embeddings only
                parent_text = self.build_parent_text(row, parent_text_fields)
                if parent_text and len(str(parent_text).strip()) > 10:
                    batch_texts.append(str(parent_text))

                    # Build parent node metadata with basic product info
                    meta = {
                        "asin": asin,
                        "node_type": "parent",
                        "node_id": f"{asin}_parent",
                        "parent_id": None,
                        "text": str(parent_text),
                        # Core product fields from extract_columns
                        "title": row.get("title"),
                        "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                        "list_price": float(row["list_price"]) if pd.notna(row.get("list_price")) else None,
                        "stars": float(row["stars"]) if pd.notna(row.get("stars")) else None,
                        "reviews_count": int(row["reviews_count"]) if pd.notna(row.get("reviews_count")) else None,
                        "category_level1": row.get("category_level1") or row.get("categoryName"),
                        "is_best_seller": bool(row.get("is_best_seller") or row.get("isBestSeller")),
                        "img_url": row.get("img_url") or row.get("imgUrl"),
                        "product_url": row.get("product_url") or row.get("productURL"),
                        "bought_in_last_month": row.get("bought_in_last_month") or row.get("boughtInLastMonth"),
                    }
                    batch_metadata.append(meta)

            else:
                # ENRICH MODE: Generate CHILD embeddings only (link to existing parents)
                # Also generate enriched parent fields for update

                # First, build enriched parent data (for updating existing parent)
                parent_text = self.build_parent_text(row, parent_text_fields)
                if parent_text and len(str(parent_text).strip()) > 10:
                    batch_texts.append(str(parent_text))

                    # Enriched parent metadata (to update existing parent node)
                    meta = {
                        "asin": asin,
                        "node_type": "parent_update",  # Special type for updating existing parent
                        "node_id": f"{asin}_parent",
                        "parent_id": None,
                        "text": str(parent_text),
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
                    }
                    batch_metadata.append(meta)

                # Then, generate child node embeddings
                for section in child_sections:
                    chunk_col = f"chunk_{section}"
                    chunk_text = row.get(chunk_col, "")

                    if chunk_text and len(str(chunk_text).strip()) > 10:
                        batch_texts.append(str(chunk_text))

                        # Child node metadata
                        meta = {
                            "asin": asin,
                            "node_type": section,  # description, features, specs, reviews, use_cases
                            "node_id": f"{asin}_{section}",
                            "parent_id": f"{asin}_parent",
                            "parent_asin": asin,  # For linking to existing parent
                            "section": section,
                            "text": str(chunk_text),
                            # Inherited fields for filtering
                            "title": row.get("title"),
                            "brand": row.get("brand"),
                            "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                            "stars": float(row["stars"]) if pd.notna(row.get("stars")) else None,
                            "category_level1": row.get("category_level1") or row.get("categoryName"),
                            "is_best_seller": bool(row.get("is_best_seller") or row.get("isBestSeller")),
                        }
                        batch_metadata.append(meta)

            # Process batch
            if len(batch_texts) >= self.batch_size:
                embeddings = await self.embed_batch(batch_texts)

                for i, (meta, emb) in enumerate(zip(batch_metadata, embeddings)):
                    results["nodes"].append(meta)
                    results["embeddings"].append(emb)

                batch_texts = []
                batch_metadata = []

                # Save checkpoint periodically
                if checkpoint_path and idx % 1000 == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({
                            "processed_count": idx + 1,
                            "results": results,
                        }, f)

        # Process remaining batch
        if batch_texts:
            embeddings = await self.embed_batch(batch_texts)
            for meta, emb in zip(batch_metadata, embeddings):
                results["nodes"].append(meta)
                results["embeddings"].append(emb)

        return results


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help="Input CSV file path",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output Parquet file path",
)
@click.option(
    "--count",
    "row_count",
    type=int,
    default=None,
    help="Limit number of products to process (defaults to all)",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama service URL (defaults to config)",
)
@click.option(
    "--model",
    default=None,
    help="Embedding model to use (defaults to config)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for embedding generation (defaults to config)",
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(),
    default=None,
    help="Checkpoint file for resume support",
)
@click.option(
    "--metrics-output",
    "metrics_path",
    type=click.Path(),
    default=None,
    help="Metrics output file path",
)
@click.option(
    "--auto-pull",
    is_flag=True,
    default=True,
    help="Automatically pull model if not available",
)
def main(
    input_path: str | None,
    output_path: str | None,
    row_count: int | None,
    ollama_url: str | None,
    model: str | None,
    batch_size: int | None,
    checkpoint_path: str | None,
    metrics_path: str | None,
    auto_pull: bool,
):
    """Generate embeddings for products."""
    start_time = time.time()

    # Load settings
    settings = get_settings()
    script_name = "04_generate_embeddings"
    pipeline_mode = cfg.get_mode(script_name)
    product_count = cfg.get_count()

    # Use config file values, then CLI overrides (with {mode} in filenames)
    input_path = input_path or str(cfg.get_path(script_name, "input", str(settings.cleaned_data_dir / f"mvp_{product_count}_{pipeline_mode}_cleaned.csv")))
    output_path = output_path or str(cfg.get_path(script_name, "output", str(settings.embedded_data_dir / f"mvp_{product_count}_{pipeline_mode}_embedded.parquet")))
    metrics_path = metrics_path or str(cfg.get_path(script_name, "metrics", str(settings.metrics_dir / "04_embedding_metrics.json")))
    ollama_url = ollama_url or cfg.get_script(script_name, "ollama_url", settings.ollama_service_url)
    model = model or cfg.get_script(script_name, "model", settings.embedding_model)
    batch_size = batch_size or cfg.get_script(script_name, "batch_size", settings.embedding_batch_size)
    row_count = row_count if row_count is not None else cfg.get_script(script_name, "count")
    checkpoint_path = checkpoint_path or cfg.get_script(script_name, "checkpoint")
    auto_pull = auto_pull if auto_pull else cfg.get_script(script_name, "auto_pull", True)

    # Get child sections from config (only used in enrich mode)
    child_sections = cfg.get_script(script_name, "child_sections", [
        "description", "features", "specs", "reviews", "use_cases"
    ])

    # Get mode-specific parent text fields
    if pipeline_mode == "original":
        parent_text_fields = cfg.get_script(script_name, "parent_text_fields_original", [
            "title", "category_level1"
        ])
    else:
        parent_text_fields = cfg.get_script(script_name, "parent_text_fields_enrich", [
            "title", "brand", "category_level1",
            "genAI_summary", "genAI_primary_function", "genAI_best_for"
        ])

    input_path = Path(input_path)
    output_path = Path(output_path)
    metrics_path = Path(metrics_path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "starting_embedding_generation",
        input_path=str(input_path),
        ollama_url=ollama_url,
        model=model,
        batch_size=batch_size,
    )

    # Initialize metrics
    metrics = {
        "stage": "embedding",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_mode": pipeline_mode,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "model": model,
        "batch_size": batch_size,
        "ollama_url": ollama_url,
        "child_sections": child_sections,
    }

    async def run():
        generator = EmbeddingGenerator(
            ollama_url=ollama_url,
            model=model,
            batch_size=batch_size,
        )

        try:
            # Check/pull model
            if auto_pull:
                if not await generator.ensure_model():
                    raise RuntimeError(f"Failed to ensure model '{model}' is available")

            # Get actual dimensions from service
            actual_dimensions = await generator.get_dimensions_from_service()
            logger.info("using_dimensions", model=model, dimensions=actual_dimensions)

            # Load data
            df = pd.read_csv(input_path)
            total_products = len(df)
            logger.info("data_loaded", count=total_products)

            # Limit rows if specified
            if row_count and row_count < len(df):
                df = df.head(row_count)
                total_products = len(df)
                logger.info("rows_limited", count=row_count)

            # Generate embeddings with mode-aware strategy
            # - original mode: parent embeddings only
            # - enrich mode: child embeddings only (link to existing parents)
            results = await generator.generate_mode_aware_embeddings(
                df,
                pipeline_mode=pipeline_mode,
                checkpoint_path=checkpoint_path,
                child_sections=child_sections,
                parent_text_fields=parent_text_fields,
            )

            # Create DataFrame for output
            nodes_df = pd.DataFrame(results["nodes"])
            embeddings_array = np.array(results["embeddings"])

            # Add embeddings as a column
            nodes_df["embedding"] = list(embeddings_array)

            # Save as Parquet (efficient for large arrays)
            nodes_df.to_parquet(output_path, index=False)
            logger.info("output_saved", path=str(output_path))

            # Calculate metrics
            total_nodes = len(results["nodes"])
            node_types = nodes_df["node_type"].value_counts().to_dict()

            # Check embedding quality (non-zero vectors)
            non_zero_count = sum(1 for e in results["embeddings"] if any(v != 0 for v in e))
            embedding_success_rate = non_zero_count / total_nodes * 100 if total_nodes > 0 else 0

            return {
                "total_products": total_products,
                "total_nodes": total_nodes,
                "node_types": node_types,
                "success_count": generator.success_count,
                "error_count": generator.error_count,
                "embedding_success_rate": round(embedding_success_rate, 2),
                "embedding_dimensions": generator.dimensions,
            }

        finally:
            await generator.close()

    try:
        result = asyncio.run(run())

        end_time = time.time()
        processing_time = end_time - start_time

        metrics.update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "metrics": result,
        })

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            "embedding_generation_complete",
            total_nodes=result["total_nodes"],
            success_rate=result["embedding_success_rate"],
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 60)
        print(f"Pipeline mode:       {pipeline_mode}")
        print(f"Total products:      {result['total_products']:,}")
        print(f"Total nodes:         {result['total_nodes']:,}")
        print(f"Embedding success:   {result['embedding_success_rate']}%")
        print(f"Processing time:     {processing_time:.2f}s")
        print(f"Model:               {model}")
        print(f"Dimensions:          {result['embedding_dimensions']}")
        print(f"Node distribution:")
        for node_type, count in result["node_types"].items():
            print(f"  - {node_type}: {count:,}")
        print(f"Output saved to:     {output_path}")
        print("=" * 60)

    except Exception as e:
        logger.error("embedding_generation_failed", error=str(e))
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
