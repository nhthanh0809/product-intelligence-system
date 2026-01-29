"""Extract stage - read products from CSV or Parquet files."""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import structlog
import numpy as np

from src.models.product import RawProduct
from src.stages.base import BaseStage, StageContext

logger = structlog.get_logger()

# Try to import polars, fall back to pandas
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    logger.info("polars_not_available", fallback="pandas")

# Always import pandas for parquet support
import pandas as pd


class ExtractStage(BaseStage[None, RawProduct]):
    """Extract products from CSV or Parquet source.

    Reads CSV/Parquet file and converts rows to RawProduct objects.
    Supports both Polars (preferred) and Pandas for CSV reading.
    For Parquet files, uses pandas for full compatibility.
    """

    name = "extract"
    description = "Extract products from CSV or Parquet source"

    def __init__(
        self,
        context: StageContext,
        csv_path: str | None = None,
        parquet_path: str | None = None,
        file_type: Literal["csv", "parquet", "auto"] = "auto",
    ):
        super().__init__(context)
        self.csv_path = csv_path or context.job.csv_path
        self.parquet_path = parquet_path
        self.file_type = file_type

    def _detect_file_type(self, path: str) -> Literal["csv", "parquet"]:
        """Auto-detect file type from extension."""
        path_lower = path.lower()
        if path_lower.endswith(".parquet") or path_lower.endswith(".pq"):
            return "parquet"
        return "csv"

    async def run_extract(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RawProduct]:
        """Extract products from CSV or Parquet file.

        Args:
            limit: Maximum number of products to extract
            offset: Starting row (0-indexed)

        Returns:
            List of RawProduct objects
        """
        # Determine which file to use
        file_path = self.parquet_path or self.csv_path
        if not file_path:
            raise ValueError("No file path specified (csv_path or parquet_path)")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type
        if self.file_type == "auto":
            file_type = self._detect_file_type(file_path)
        else:
            file_type = self.file_type

        self._start(limit or 0)

        try:
            if file_type == "parquet":
                products = await self._extract_from_parquet(path, limit, offset)
            elif HAS_POLARS:
                products = await self._extract_with_polars(path, limit, offset)
            else:
                products = await self._extract_with_pandas(path, limit, offset)

            # Update total after reading
            self.progress.total = len(products)
            self.progress.processed = len(products)

            self._complete()
            return products

        except Exception as e:
            self._fail(str(e))
            raise

    async def _extract_with_polars(
        self,
        path: Path,
        limit: int | None,
        offset: int,
    ) -> list[RawProduct]:
        """Extract using Polars (faster for large files)."""
        logger.info("extracting_with_polars", path=str(path))

        # Read CSV with Polars
        df = pl.read_csv(
            path,
            infer_schema_length=10000,
            null_values=["", "N/A", "null", "None"],
        )

        # Apply offset and limit
        if offset > 0:
            df = df.slice(offset)
        if limit:
            df = df.head(limit)

        # Convert to products
        products = []
        for row_idx, row in enumerate(df.iter_rows(named=True)):
            try:
                product = self._row_to_product(row, row_idx + offset)
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(
                    "row_parse_failed",
                    row=row_idx + offset,
                    error=str(e),
                )
                self.progress.failed += 1

        logger.info("extraction_completed", count=len(products))
        return products

    async def _extract_with_pandas(
        self,
        path: Path,
        limit: int | None,
        offset: int,
    ) -> list[RawProduct]:
        """Extract using Pandas (fallback)."""
        logger.info("extracting_with_pandas", path=str(path))

        # Read CSV with Pandas
        df = pd.read_csv(
            path,
            skiprows=range(1, offset + 1) if offset > 0 else None,
            nrows=limit,
            na_values=["", "N/A", "null", "None"],
        )

        # Convert to products
        products = []
        for row_idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                product = self._row_to_product(row_dict, row_idx + offset)
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(
                    "row_parse_failed",
                    row=row_idx + offset,
                    error=str(e),
                )
                self.progress.failed += 1

        logger.info("extraction_completed", count=len(products))
        return products

    async def _extract_from_parquet(
        self,
        path: Path,
        limit: int | None,
        offset: int,
    ) -> list[RawProduct]:
        """Extract from Parquet file using Pandas."""
        logger.info("extracting_from_parquet", path=str(path))

        # Read parquet with pandas
        df = pd.read_parquet(path)

        # Apply offset and limit
        if offset > 0:
            df = df.iloc[offset:]
        if limit:
            df = df.head(limit)

        # Convert to products
        products = []
        for row_idx, (_, row) in enumerate(df.iterrows()):
            try:
                row_dict = row.to_dict()
                product = self._row_to_product(row_dict, row_idx + offset)
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(
                    "row_parse_failed",
                    row=row_idx + offset,
                    error=str(e),
                )
                self.progress.failed += 1

        logger.info("extraction_completed", count=len(products), source="parquet")
        return products

    def _row_to_product(
        self,
        row: dict[str, Any],
        row_number: int,
    ) -> RawProduct | None:
        """Convert a CSV row to RawProduct.

        Handles various column name formats and data types.
        """
        # Get ASIN (required)
        asin = row.get("asin") or row.get("ASIN")
        if not asin:
            return None

        # Get title (required)
        title = row.get("title") or row.get("Title") or row.get("product_title")
        if not title:
            return None

        # Build product dict with normalization
        product_data = {
            "asin": str(asin).strip(),
            "title": str(title).strip(),
            "row_number": row_number,
        }

        # Map various column names to standard fields
        field_mappings = {
            "brand": ["brand", "Brand", "manufacturer", "Manufacturer"],
            "price": ["price", "Price", "current_price"],
            "list_price": ["list_price", "listPrice", "original_price"],
            "original_price": ["original_price", "originalPrice", "was_price"],
            "stars": ["stars", "rating", "Rating", "star_rating"],
            "reviews_count": ["reviews_count", "reviewsCount", "review_count", "num_reviews"],
            "bought_in_last_month": ["bought_in_last_month", "boughtInLastMonth", "monthly_sold"],
            "category_name": ["category_name", "categoryName", "category"],
            "category_level1": ["category_level1", "main_category", "category1"],
            "category_level2": ["category_level2", "sub_category", "category2"],
            "category_level3": ["category_level3", "category3"],
            "is_best_seller": ["is_best_seller", "isBestSeller", "best_seller"],
            "is_amazon_choice": ["is_amazon_choice", "isAmazonChoice", "amazon_choice"],
            "prime_eligible": ["prime_eligible", "primeEligible", "prime"],
            "availability": ["availability", "stock_status"],
            "product_description": ["product_description", "description", "Description"],
            "features": ["features", "Features", "bullet_points"],
            "specifications": ["specifications", "specs", "Specifications"],
            "product_url": ["product_url", "productUrl", "url", "link"],
            "img_url": ["img_url", "imgUrl", "image_url", "image", "thumbnail"],
        }

        for field, possible_names in field_mappings.items():
            for name in possible_names:
                if name in row and row[name] is not None:
                    value = row[name]
                    # Handle NaN values
                    if isinstance(value, float) and value != value:  # NaN check
                        continue
                    product_data[field] = value
                    break

        # Parse JSON fields if they're strings
        for json_field in ["features", "specifications"]:
            if json_field in product_data and isinstance(product_data[json_field], str):
                try:
                    product_data[json_field] = json.loads(product_data[json_field])
                except json.JSONDecodeError:
                    pass

        # Create RawProduct with validation
        try:
            return RawProduct(**product_data)
        except Exception as e:
            logger.debug(
                "product_validation_failed",
                asin=asin,
                error=str(e),
            )
            return None

    async def process_batch(self, batch: list[None]) -> list[RawProduct]:
        """Not used for extract stage - use run_extract instead."""
        raise NotImplementedError("Use run_extract() for extraction")
