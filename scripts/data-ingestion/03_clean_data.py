#!/usr/bin/env python3
"""
Script 03: Data Cleaning

Clean and normalize scraped product data.
Handles text normalization, HTML stripping, and data validation.

Usage:
    python scripts/03_clean_data.py --input data/scraped/mvp_100k_enriched.csv
"""

import html
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd
import structlog

# Add src and scripts to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Product_intelligence_system/
sys.path.insert(0, str(PROJECT_ROOT / "services" / "data-pipeline"))  # for src.config
sys.path.insert(0, str(Path(__file__).parent))  # for config module

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


class DataCleaner:
    """Data cleaning utilities."""

    @staticmethod
    def _is_na(value: Any) -> bool:
        """Check if value is NA, handling lists/dicts safely."""
        if value is None:
            return True
        if isinstance(value, (list, dict)):
            return len(value) == 0
        try:
            return pd.isna(value)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _has_value(value: Any) -> bool:
        """Check if value has a valid non-empty value, handling lists/dicts safely."""
        if value is None:
            return False
        if isinstance(value, (list, dict)):
            return len(value) > 0
        if isinstance(value, str):
            return len(value.strip()) > 0
        try:
            return not pd.isna(value)
        except (ValueError, TypeError):
            return True  # If we can't check, assume it has value

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode text to NFKC form."""
        if not isinstance(text, str):
            return text
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not isinstance(text, str):
            return text
        # Unescape HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def clean_price(price: Any) -> float | None:
        """Clean and parse price values."""
        if price is None or (not isinstance(price, (list, dict)) and pd.isna(price)):
            return None
        if isinstance(price, (int, float)):
            return float(price) if price > 0 else None
        if isinstance(price, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r"[^\d.]", "", price)
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None
        return None

    @staticmethod
    def clean_rating(rating: Any) -> float | None:
        """Clean and validate star rating."""
        if rating is None or (not isinstance(rating, (list, dict)) and pd.isna(rating)):
            return None
        if isinstance(rating, (int, float)):
            val = float(rating)
            return val if 0 <= val <= 5 else None
        if isinstance(rating, str):
            try:
                val = float(rating.split()[0])
                return val if 0 <= val <= 5 else None
            except (ValueError, IndexError):
                return None
        return None

    @staticmethod
    def clean_review_count(count: Any) -> int | None:
        """Clean and parse review count."""
        if count is None or (not isinstance(count, (list, dict)) and pd.isna(count)):
            return None
        if isinstance(count, (int, float)):
            return int(count) if count >= 0 else None
        if isinstance(count, str):
            cleaned = re.sub(r"[^\d]", "", count)
            try:
                return int(cleaned) if cleaned else None
            except ValueError:
                return None
        return None

    @staticmethod
    def parse_category_hierarchy(category: str) -> tuple[str | None, str | None, str | None]:
        """Parse category string into hierarchy levels."""
        if not isinstance(category, str) or not category:
            return None, None, None

        # Common separators: |, >, /, -
        separators = [" | ", " > ", " / ", " - "]
        parts = None
        for sep in separators:
            if sep in category:
                parts = [p.strip() for p in category.split(sep)]
                break

        if not parts:
            return category, None, None

        level1 = parts[0] if len(parts) > 0 else None
        level2 = parts[1] if len(parts) > 1 else None
        level3 = parts[2] if len(parts) > 2 else None

        return level1, level2, level3

    @staticmethod
    def parse_json_field(value: Any) -> Any:
        """Parse JSON string to Python object."""
        if isinstance(value, (dict, list)):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def build_chunk_description(self, row: pd.Series) -> str:
        """Build description chunk from product data."""
        parts = []

        if self._has_value(row.get("title")):
            parts.append(f"Product: {row['title']}")

        if self._has_value(row.get("brand")):
            parts.append(f"Brand: {row['brand']}")

        if self._has_value(row.get("product_description")):
            desc = self.strip_html(str(row["product_description"]))
            parts.append(f"\nDescription:\n{desc}")

        return "\n".join(parts) if parts else ""

    def build_chunk_features(self, row: pd.Series) -> str:
        """Build features chunk from about_this_item."""
        parts = []

        if self._has_value(row.get("title")):
            parts.append(f"Product: {row['title']}")

        about = self.parse_json_field(row.get("about_this_item"))
        if about:
            if isinstance(about, list):
                features = "\n".join(f"• {item}" for item in about)
            else:
                features = str(about)
            parts.append(f"\nKey Features:\n{features}")

        return "\n".join(parts) if parts else ""

    def build_chunk_specs(self, row: pd.Series) -> str:
        """Build specs chunk from technical details."""
        parts = []

        if self._has_value(row.get("title")):
            parts.append(f"Product: {row['title']}")

        tech = self.parse_json_field(row.get("technical_details"))
        if tech and isinstance(tech, dict):
            specs_text = "\n".join(f"• {k}: {v}" for k, v in tech.items())
            parts.append(f"\nTechnical Details:\n{specs_text}")

        specs = self.parse_json_field(row.get("specifications"))
        if specs and isinstance(specs, dict):
            specs_text = "\n".join(f"• {k}: {v}" for k, v in specs.items())
            parts.append(f"\nSpecifications:\n{specs_text}")

        return "\n".join(parts) if parts else ""

    def build_chunk_reviews(self, row: pd.Series) -> str:
        """Build reviews chunk from top_reviews."""
        parts = []

        if self._has_value(row.get("title")):
            parts.append(f"Product: {row['title']}")

        if self._has_value(row.get("stars")):
            parts.append(f"Overall Rating: {row['stars']}/5")

        summary = self.parse_json_field(row.get("review_summary"))
        if summary and isinstance(summary, dict):
            summary_text = ", ".join(f"{k}: {v}" for k, v in summary.items())
            parts.append(f"Rating Distribution: {summary_text}")

        reviews = self.parse_json_field(row.get("top_reviews"))
        if reviews and isinstance(reviews, list):
            review_texts = []
            for i, rev in enumerate(reviews[:5], 1):
                if isinstance(rev, dict):
                    rating = rev.get("rating", "N/A")
                    title = rev.get("title", "")
                    text = rev.get("text", "")[:200]
                    review_texts.append(f"Review {i} ({rating}/5): {title}\n{text}")
            if review_texts:
                parts.append(f"\nTop Reviews:\n" + "\n\n".join(review_texts))

        return "\n".join(parts) if parts else ""

    def build_chunk_use_cases(self, row: pd.Series) -> str:
        """Build use_cases chunk from GenAI fields and product data."""
        parts = []

        if self._has_value(row.get("title")):
            parts.append(f"Product: {row['title']}")

        # GenAI use case fields (from 02c_extract_with_llm.py output)
        use_case_scenarios = self.parse_json_field(row.get("genAI_use_case_scenarios"))
        if use_case_scenarios and isinstance(use_case_scenarios, list):
            parts.append("\nUse Case Scenarios:")
            for scenario in use_case_scenarios[:5]:
                if isinstance(scenario, dict):
                    desc = scenario.get("scenario", str(scenario))
                    fit = scenario.get("fit_score", "")
                    fit_str = f" (fit: {fit})" if fit else ""
                    parts.append(f"• {desc}{fit_str}")
                else:
                    parts.append(f"• {scenario}")

        ideal_profiles = self.parse_json_field(row.get("genAI_ideal_user_profiles"))
        if ideal_profiles and isinstance(ideal_profiles, list):
            parts.append("\nIdeal Users:")
            for profile in ideal_profiles[:5]:
                if isinstance(profile, dict):
                    desc = profile.get("profile", str(profile))
                    why = profile.get("why", "")
                    why_str = f" - {why}" if why else ""
                    parts.append(f"• {desc}{why_str}")
                else:
                    parts.append(f"• {profile}")

        not_recommended = self.parse_json_field(row.get("genAI_not_recommended_for"))
        if not_recommended and isinstance(not_recommended, list):
            parts.append("\nNot Recommended For:")
            for item in not_recommended[:5]:
                parts.append(f"• {item}")

        problems_solved = self.parse_json_field(row.get("genAI_problems_solved"))
        if problems_solved and isinstance(problems_solved, list):
            parts.append("\nProblems Solved:")
            for problem in problems_solved[:5]:
                parts.append(f"• {problem}")

        # Fallback: use basic product info if no GenAI data
        if len(parts) <= 1:
            if self._has_value(row.get("genAI_best_for")):
                parts.append(f"\nBest For: {row['genAI_best_for']}")
            if self._has_value(row.get("genAI_target_audience")):
                parts.append(f"Target Audience: {row['genAI_target_audience']}")
            use_cases = self.parse_json_field(row.get("genAI_use_cases"))
            if use_cases and isinstance(use_cases, list):
                parts.append("\nUse Cases:")
                for uc in use_cases[:5]:
                    parts.append(f"• {uc}")

        return "\n".join(parts) if parts else ""

    def standardize_field_names(self, df: pd.DataFrame, field_mapping: dict) -> pd.DataFrame:
        """Standardize field names according to mapping."""
        rename_map = {}
        for old_name, new_name in field_mapping.items():
            if old_name in df.columns and old_name != new_name:
                rename_map[old_name] = new_name
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info("fields_standardized", mapping=rename_map)
        return df


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
    help="Output CSV file path",
)
@click.option(
    "--count",
    "row_count",
    type=int,
    default=None,
    help="Limit number of products to process (defaults to all)",
)
@click.option(
    "--metrics-output",
    "metrics_path",
    type=click.Path(),
    default=None,
    help="Metrics output file path",
)
def main(input_path: str | None, output_path: str | None, row_count: int | None, metrics_path: str | None):
    """Clean and normalize product data."""
    # Load settings
    settings = get_settings()
    script_name = "03_clean_data"
    pipeline_mode = cfg.get_mode(script_name)
    product_count = cfg.get_count()

    # Determine default input based on pipeline mode
    # - original mode: input from raw/mvp_{count}_products.csv (no scraping)
    # - enrich mode: input from scraped/mvp_{count}_{mode}_extracted.csv or .json (with scraping + LLM)
    if pipeline_mode == "original":
        default_input = str(cfg.get_path(script_name, "input_original", str(settings.raw_data_dir / f"mvp_{product_count}_products.csv")))
    else:
        # Enrich mode: try CSV first, then JSON
        csv_path = cfg.get_path(script_name, "input_enrich", str(settings.scraped_data_dir / f"mvp_{product_count}_{pipeline_mode}_extracted.csv"))
        json_path = str(csv_path).replace(".csv", ".json")
        if Path(json_path).exists():
            default_input = json_path
        else:
            default_input = str(csv_path)

    # Use config file values, then CLI overrides
    input_path = input_path or default_input
    output_path = output_path or str(cfg.get_path(script_name, "output", str(settings.cleaned_data_dir / f"mvp_{product_count}_{pipeline_mode}_cleaned.csv")))
    metrics_path = metrics_path or str(cfg.get_path(script_name, "metrics", str(settings.metrics_dir / "03_cleaning_metrics.json")))
    row_count = row_count if row_count is not None else cfg.get_script(script_name, "count")

    start_time = time.time()

    input_path = Path(input_path)
    output_path = Path(output_path)
    metrics_path = Path(metrics_path)

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("starting_cleaning", input_path=str(input_path), pipeline_mode=pipeline_mode)

    # Get field mapping from config
    field_mapping = cfg.get_script(script_name, "field_mapping", {
        "listPrice": "list_price",
        "categoryName": "category_level1",
        "isBestSeller": "is_best_seller",
        "reviews": "reviews_count",
        "imgUrl": "img_url",
    })

    # Get chunk sections from config
    chunk_sections = cfg.get_script(script_name, "chunk_sections", [
        "description", "features", "specs", "reviews", "use_cases"
    ])

    # Initialize metrics
    metrics = {
        "stage": "cleaning",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_mode": pipeline_mode,
        "input_file": str(input_path),
        "output_file": str(output_path),
    }

    cleaner = DataCleaner()

    # Get scraped field mapping from config (for enrich mode)
    scraped_field_mapping = cfg.get_script(script_name, "scraped_field_mapping", {})

    try:
        # Load data - support both CSV and JSON
        if str(input_path).endswith(".json"):
            df = pd.read_json(input_path)
            logger.info("data_loaded_from_json", count=len(df))
        else:
            df = pd.read_csv(input_path)
            logger.info("data_loaded_from_csv", count=len(df))

        initial_count = len(df)
        logger.info("data_loaded", count=initial_count)

        # Limit rows if specified
        if row_count and row_count < len(df):
            df = df.head(row_count)
            logger.info("rows_limited", count=row_count)

        # Apply scraped field mapping first (enrich mode: scraped_* → standard names)
        if pipeline_mode == "enrich" and scraped_field_mapping:
            df = cleaner.standardize_field_names(df, scraped_field_mapping)
            logger.info("scraped_fields_mapped", mapping=scraped_field_mapping)

        # Standardize field names from config (original field names → DB names)
        df = cleaner.standardize_field_names(df, field_mapping)

        # Clean text fields
        text_columns = ["title", "product_description", "brand", "availability"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: cleaner.strip_html(cleaner.normalize_unicode(str(x)))
                    if cleaner._has_value(x) and isinstance(x, str)
                    else x
                )

        # Clean numeric fields
        if "price" in df.columns:
            df["price"] = df["price"].apply(cleaner.clean_price)

        if "listPrice" in df.columns:
            df["listPrice"] = df["listPrice"].apply(cleaner.clean_price)

        if "stars" in df.columns:
            df["stars"] = df["stars"].apply(cleaner.clean_rating)

        if "reviews" in df.columns:
            df["reviews_count"] = df["reviews"].apply(cleaner.clean_review_count)

        # Parse category hierarchy
        if "categoryName" in df.columns:
            categories = df["categoryName"].apply(cleaner.parse_category_hierarchy)
            df["category_level1"] = categories.apply(lambda x: x[0])
            df["category_level2"] = categories.apply(lambda x: x[1])
            df["category_level3"] = categories.apply(lambda x: x[2])

        # Remove rows with missing required fields
        required_fields = ["asin", "title"]
        df = df.dropna(subset=required_fields)
        after_required = len(df)
        dropped_required = initial_count - after_required

        # Remove duplicates
        df = df.drop_duplicates(subset=["asin"], keep="first")
        after_dedup = len(df)
        dropped_dupes = after_required - after_dedup

        # Build chunks for Multi-Chunk strategy (ONLY in enrich mode)
        # In original mode, we skip chunk building - only parent nodes are created
        if pipeline_mode == "enrich":
            logger.info("building_chunks", sections=chunk_sections, mode=pipeline_mode)
            if "description" in chunk_sections:
                df["chunk_description"] = df.apply(cleaner.build_chunk_description, axis=1)
            if "features" in chunk_sections:
                df["chunk_features"] = df.apply(cleaner.build_chunk_features, axis=1)
            if "specs" in chunk_sections:
                df["chunk_specs"] = df.apply(cleaner.build_chunk_specs, axis=1)
            if "reviews" in chunk_sections:
                df["chunk_reviews"] = df.apply(cleaner.build_chunk_reviews, axis=1)
            if "use_cases" in chunk_sections:
                df["chunk_use_cases"] = df.apply(cleaner.build_chunk_use_cases, axis=1)
        else:
            logger.info("skipping_chunks", mode=pipeline_mode, reason="original mode - parent nodes only")

        # Calculate data quality metrics
        final_count = len(df)

        # Completeness check
        completeness_checks = {
            "title": df["title"].notna().sum() / final_count,
            "price": df["price"].notna().sum() / final_count if "price" in df.columns else 0,
            "stars": df["stars"].notna().sum() / final_count if "stars" in df.columns else 0,
            "category": df["category_level1"].notna().sum() / final_count if "category_level1" in df.columns else 0,
            "description": df["product_description"].notna().sum() / final_count if "product_description" in df.columns else 0,
        }
        avg_completeness = sum(completeness_checks.values()) / len(completeness_checks)

        # Chunk coverage (5 sections, only in enrich mode)
        chunk_coverage = {}
        if pipeline_mode == "enrich":
            for section in chunk_sections:
                col_name = f"chunk_{section}"
                if col_name in df.columns:
                    chunk_coverage[col_name] = (df[col_name].str.len() > 50).sum() / final_count

        # Validity check
        price_valid = df[df["price"].notna()]["price"].between(0, 100000).mean() if "price" in df.columns else 1
        stars_valid = df[df["stars"].notna()]["stars"].between(0, 5).mean() if "stars" in df.columns else 1
        avg_validity = (price_valid + stars_valid) / 2

        data_quality_score = (avg_completeness + avg_validity) / 2 * 100

        # Save cleaned data
        df.to_csv(output_path, index=False)
        logger.info("output_saved", path=str(output_path), count=final_count)

        end_time = time.time()
        processing_time = end_time - start_time

        metrics.update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "metrics": {
                "input_records": initial_count,
                "output_records": final_count,
                "records_retained_rate": round(final_count / initial_count * 100, 2) if initial_count > 0 else 0,
                "dropped_missing_required": dropped_required,
                "dropped_duplicates": dropped_dupes,
                "data_quality_score": round(data_quality_score, 2),
                "completeness": {k: round(v * 100, 2) for k, v in completeness_checks.items()},
                "avg_completeness": round(avg_completeness * 100, 2),
                "avg_validity": round(avg_validity * 100, 2),
                "chunk_coverage": {k: round(v * 100, 2) for k, v in chunk_coverage.items()},
            },
        })

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            "cleaning_complete",
            final_count=final_count,
            quality_score=round(data_quality_score, 2),
        )

        # Print summary
        print("\n" + "=" * 60)
        print("CLEANING COMPLETE")
        print("=" * 60)
        print(f"Pipeline mode:       {pipeline_mode}")
        print(f"Input file:          {input_path}")
        print(f"Input records:       {initial_count:,}")
        print(f"Output records:      {final_count:,}")
        print(f"Records retained:    {final_count / initial_count * 100:.2f}%")
        print(f"Data quality score:  {data_quality_score:.2f}%")
        if pipeline_mode == "enrich":
            print(f"Chunks built:        {len(chunk_sections)} sections")
        else:
            print(f"Chunks built:        SKIPPED (original mode - parent only)")
        print(f"Processing time:     {processing_time:.2f}s")
        print(f"Output saved to:     {output_path}")
        print("=" * 60)

    except Exception as e:
        logger.error("cleaning_failed", error=str(e))
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
