#!/usr/bin/env python3
"""
Script 02c: Extract Product Information with LLM + GenAI Enrichment

Uses Ollama LLM to extract structured product information from Markdown files
and generate GenAI enrichment fields for intelligent product understanding.

Pipeline Design:
- Uses ORIGINAL CSV as base (title, price, stars, reviews, category, etc.)
- Extracts ADDITIONAL fields from markdown (description, features, specs, reviews text)
- Generates GenAI enrichment fields using LLM

Field Configuration:
- All fields are configured in pipeline_config.yaml under genai_enrichment
- Set field to true to generate with LLM
- Set field to false to include as empty column in output

Usage:
    python scripts/data-ingestion/02c_extract_with_llm.py
    python scripts/data-ingestion/02c_extract_with_llm.py --model gpt-oss:120b --concurrency 2

Prerequisites:
    pip install httpx pandas structlog click tqdm
    # Ensure Ollama is running: ollama serve
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import httpx
import pandas as pd
import structlog
from tqdm import tqdm

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
# SCRAPED FIELDS - extracted directly from markdown
# =============================================================================
SCRAPED_FIELDS = [
    '"scraped_description": "Full product description text from the markdown"',
    '"scraped_features": ["List of feature bullet points from About This Item section"]',
    '"scraped_technical_details": {{"key": "value pairs of technical specifications"}}',
    '"scraped_additional_info": {{"key": "value pairs like dimensions, weight, manufacturer"}}',
    '"scraped_top_reviews": ["review 1 text", "review 2 text if available"]',
    '"scraped_materials": "Materials mentioned (e.g., Carbon Steel, Plastic)"',
    '"scraped_whats_included": "What comes in the box/package if mentioned"',
    '"scraped_brand": "Brand name if found in the content"',
    '"scraped_availability": "Availability status (In Stock, Out of Stock)"',
]


def build_combined_prompt(enabled_fields: dict) -> str:
    """Build a SINGLE prompt that extracts markdown data AND generates GenAI fields."""
    # Start with scraped fields
    json_fields = list(SCRAPED_FIELDS)

    # Add GenAI fields based on config
    basic = enabled_fields.get("basic_fields", {})
    if basic.get("short_title"):
        json_fields.append('"short_title": "Cleaned, concise product title (max 6-8 words)"')
    if basic.get("product_type"):
        json_fields.append('"product_type": "Clean product type (e.g., Wireless Headphones, Hole Saw Kit)"')
    if basic.get("product_type_keywords"):
        json_fields.append('"product_type_keywords": ["keyword1", "keyword2", "keyword3"]')
    if basic.get("brand"):
        json_fields.append('"brand": "Brand name extracted from content"')
    if basic.get("availability"):
        json_fields.append('"availability": "Availability status (In Stock, Out of Stock, etc.)"')

    parent = enabled_fields.get("parent_fields", {})
    if parent.get("genAI_summary"):
        json_fields.append('"genAI_summary": "2-3 sentence product summary"')
    if parent.get("genAI_primary_function"):
        json_fields.append('"genAI_primary_function": "Main purpose in 1 sentence"')
    if parent.get("genAI_best_for"):
        json_fields.append('"genAI_best_for": "Description of ideal user"')
    if parent.get("genAI_use_cases"):
        json_fields.append('"genAI_use_cases": ["use case 1", "use case 2", "use case 3"]')
    if parent.get("genAI_target_audience"):
        json_fields.append('"genAI_target_audience": "Who should buy this product"')
    if parent.get("genAI_key_capabilities"):
        json_fields.append('"genAI_key_capabilities": ["capability 1", "capability 2", "capability 3"]')
    if parent.get("genAI_unique_selling_points"):
        json_fields.append('"genAI_unique_selling_points": ["usp 1", "usp 2"]')
    if parent.get("genAI_value_score"):
        json_fields.append('"genAI_value_score": 7')

    desc = enabled_fields.get("child_description_fields", {})
    if desc.get("genAI_detailed_description"):
        json_fields.append('"genAI_detailed_description": "Expanded description (2-3 paragraphs)"')
    if desc.get("genAI_how_it_works"):
        json_fields.append('"genAI_how_it_works": "How the product functions or is used"')
    if desc.get("genAI_whats_included"):
        json_fields.append('"genAI_whats_included": ["item 1", "item 2", "item 3"]')
    if desc.get("genAI_materials"):
        json_fields.append('"genAI_materials": "Materials the product is made from"')

    feat = enabled_fields.get("child_features_fields", {})
    if feat.get("genAI_features_detailed"):
        json_fields.append('"genAI_features_detailed": [{{"feature": "name", "description": "desc", "benefit": "benefit"}}]')
    if feat.get("genAI_standout_features"):
        json_fields.append('"genAI_standout_features": ["standout feature 1", "feature 2"]')
    if feat.get("genAI_technology_explained"):
        json_fields.append('"genAI_technology_explained": "Key technologies used or null"')
    if feat.get("genAI_feature_comparison"):
        json_fields.append('"genAI_feature_comparison": "How it compares to alternatives"')

    specs = enabled_fields.get("child_specs_fields", {})
    if specs.get("genAI_specs_summary"):
        json_fields.append('"genAI_specs_summary": "Human-readable specs summary"')
    if specs.get("genAI_specs_comparison_ready"):
        json_fields.append('"genAI_specs_comparison_ready": {{"spec1": "value", "spec2": "value"}}')
    if specs.get("genAI_specs_limitations"):
        json_fields.append('"genAI_specs_limitations": ["limitation 1", "limitation 2"]')

    reviews = enabled_fields.get("child_reviews_fields", {})
    if reviews.get("genAI_sentiment_score"):
        json_fields.append('"genAI_sentiment_score": 0.75')
    if reviews.get("genAI_sentiment_label"):
        json_fields.append('"genAI_sentiment_label": "Positive"')
    if reviews.get("genAI_common_praises"):
        json_fields.append('"genAI_common_praises": [{{"theme": "theme", "example": "praise"}}]')
    if reviews.get("genAI_common_complaints"):
        json_fields.append('"genAI_common_complaints": [{{"theme": "theme", "example": "complaint"}}]')
    if reviews.get("genAI_durability_feedback"):
        json_fields.append('"genAI_durability_feedback": "Expected durability"')
    if reviews.get("genAI_value_for_money_feedback"):
        json_fields.append('"genAI_value_for_money_feedback": "Value assessment"')

    use_cases = enabled_fields.get("child_use_cases_fields", {})
    if use_cases.get("genAI_use_case_scenarios"):
        json_fields.append('"genAI_use_case_scenarios": [{{"scenario": "desc", "fit_score": 0.9}}]')
    if use_cases.get("genAI_ideal_user_profiles"):
        json_fields.append('"genAI_ideal_user_profiles": [{{"profile": "desc", "why": "reason"}}]')
    if use_cases.get("genAI_not_recommended_for"):
        json_fields.append('"genAI_not_recommended_for": ["user type 1", "situation 2"]')
    if use_cases.get("genAI_problems_solved"):
        json_fields.append('"genAI_problems_solved": ["problem 1", "problem 2"]')

    json_schema = "{{\n    " + ",\n    ".join(json_fields) + "\n}}"

    return f'''You are a product data extractor and intelligence AI. From the markdown content below, extract product information AND generate intelligent enrichment fields.

PRODUCT CONTEXT:
Title: {{title}}
Category: {{category}}
Price: {{price}}
Stars: {{stars}}
Reviews Count: {{reviews_count}}

MARKDOWN CONTENT:
{{content}}

Extract and generate a JSON object with ALL these fields (use null if not found):

{json_schema}

RULES:
1. Return ONLY valid JSON - no markdown, no explanation, no text before/after
2. Use null for missing fields
3. scraped_* fields: Extract ACTUAL content from markdown
4. genAI_* fields: Generate intelligent analysis based on extracted data
5. genAI_value_score: 1-10 integer
6. genAI_sentiment_score: 0.0-1.0 float
7. short_title: Clean readable version (max 6-8 words)
8. product_type: Clean category like "Wireless Headphones"

JSON:'''


def build_genai_enrichment_prompt(enabled_fields: dict) -> str:
    """Build GenAI enrichment prompt based on enabled fields."""
    # Build the JSON schema based on enabled fields
    json_fields = []

    # Basic fields
    basic = enabled_fields.get("basic_fields", {})
    if basic.get("short_title"):
        json_fields.append('"short_title": "Cleaned, concise product title (max 6-8 words)"')
    if basic.get("product_type"):
        json_fields.append('"product_type": "Clean product type (e.g., \'Wireless Headphones\', \'Hole Saw Kit\')"')
    if basic.get("product_type_keywords"):
        json_fields.append('"product_type_keywords": ["keyword1", "keyword2", "keyword3"]')
    if basic.get("brand"):
        json_fields.append('"brand": "Brand name extracted from content"')
    if basic.get("availability"):
        json_fields.append('"availability": "Availability status (In Stock, Out of Stock, etc.)"')

    # Parent fields
    parent = enabled_fields.get("parent_fields", {})
    if parent.get("genAI_summary"):
        json_fields.append('"genAI_summary": "2-3 sentence product summary answering \'what is this product?\'"')
    if parent.get("genAI_primary_function"):
        json_fields.append('"genAI_primary_function": "Main purpose in 1 sentence"')
    if parent.get("genAI_best_for"):
        json_fields.append('"genAI_best_for": "Description of ideal user (e.g., \'DIY enthusiasts who need...\')"')
    if parent.get("genAI_use_cases"):
        json_fields.append('"genAI_use_cases": ["specific use case 1", "specific use case 2", "specific use case 3"]')
    if parent.get("genAI_target_audience"):
        json_fields.append('"genAI_target_audience": "Who should buy this product"')
    if parent.get("genAI_key_capabilities"):
        json_fields.append('"genAI_key_capabilities": ["capability 1", "capability 2", "capability 3"]')
    if parent.get("genAI_unique_selling_points"):
        json_fields.append('"genAI_unique_selling_points": ["what makes this product stand out 1", "usp 2"]')
    if parent.get("genAI_value_score"):
        json_fields.append('"genAI_value_score": 7')

    # Child description fields
    desc = enabled_fields.get("child_description_fields", {})
    if desc.get("genAI_detailed_description"):
        json_fields.append('"genAI_detailed_description": "Expanded description explaining the product in depth (2-3 paragraphs)"')
    if desc.get("genAI_how_it_works"):
        json_fields.append('"genAI_how_it_works": "Brief explanation of how the product functions or is used"')
    if desc.get("genAI_whats_included"):
        json_fields.append('"genAI_whats_included": ["item 1", "item 2", "item 3"]')
    if desc.get("genAI_materials"):
        json_fields.append('"genAI_materials": "Materials the product is made from"')

    # Child features fields
    feat = enabled_fields.get("child_features_fields", {})
    if feat.get("genAI_features_detailed"):
        json_fields.append('"genAI_features_detailed": [{{"feature": "Feature name", "description": "Explanation", "benefit": "User benefit"}}]')
    if feat.get("genAI_standout_features"):
        json_fields.append('"genAI_standout_features": ["top differentiating feature 1", "feature 2"]')
    if feat.get("genAI_technology_explained"):
        json_fields.append('"genAI_technology_explained": "Explanation of key technologies used (or null if not applicable)"')
    if feat.get("genAI_feature_comparison"):
        json_fields.append('"genAI_feature_comparison": "How this compares to typical alternatives"')

    # Child specs fields
    specs = enabled_fields.get("child_specs_fields", {})
    if specs.get("genAI_specs_summary"):
        json_fields.append('"genAI_specs_summary": "Human-readable summary of key specifications"')
    if specs.get("genAI_specs_comparison_ready"):
        json_fields.append('"genAI_specs_comparison_ready": {{"key spec 1": "value", "key spec 2": "value"}}')
    if specs.get("genAI_specs_limitations"):
        json_fields.append('"genAI_specs_limitations": ["limitation 1", "limitation 2"]')

    # Child reviews fields
    reviews = enabled_fields.get("child_reviews_fields", {})
    if reviews.get("genAI_sentiment_score"):
        json_fields.append('"genAI_sentiment_score": 0.75')
    if reviews.get("genAI_sentiment_label"):
        json_fields.append('"genAI_sentiment_label": "Positive"')
    if reviews.get("genAI_common_praises"):
        json_fields.append('"genAI_common_praises": [{{"theme": "theme name", "example": "sample praise"}}]')
    if reviews.get("genAI_common_complaints"):
        json_fields.append('"genAI_common_complaints": [{{"theme": "theme name", "example": "potential issue"}}]')
    if reviews.get("genAI_durability_feedback"):
        json_fields.append('"genAI_durability_feedback": "Expected durability based on materials and specs"')
    if reviews.get("genAI_value_for_money_feedback"):
        json_fields.append('"genAI_value_for_money_feedback": "Assessment of value for the price point"')

    # Child use cases fields
    use_cases = enabled_fields.get("child_use_cases_fields", {})
    if use_cases.get("genAI_use_case_scenarios"):
        json_fields.append('"genAI_use_case_scenarios": [{{"scenario": "description", "fit_score": 0.9}}]')
    if use_cases.get("genAI_ideal_user_profiles"):
        json_fields.append('"genAI_ideal_user_profiles": [{{"profile": "description", "why": "reason"}}]')
    if use_cases.get("genAI_not_recommended_for"):
        json_fields.append('"genAI_not_recommended_for": ["user type 1", "use case that doesn\'t fit"]')
    if use_cases.get("genAI_problems_solved"):
        json_fields.append('"genAI_problems_solved": ["problem 1 this product solves", "problem 2"]')

    json_schema = "{{\n    " + ",\n    ".join(json_fields) + "\n}}"

    return f'''You are a product intelligence AI. Based on the product data below, generate intelligent enrichment fields that help users understand and make decisions about this product.

PRODUCT DATA:
Title: {{title}}
Brand: {{brand}}
Category: {{category}}
Price: {{price}}
Stars: {{stars}}
Reviews Count: {{reviews_count}}
Description: {{description}}
Features: {{features}}
Technical Specs: {{specs}}

Generate a JSON object with these fields:

{json_schema}

IMPORTANT:
1. Return ONLY valid JSON - no markdown, no explanation
2. Use null for fields where information is insufficient
3. Be specific and actionable in descriptions
4. Base analysis on the provided product data
5. genAI_value_score is 1-10 integer (if included)
6. genAI_sentiment_score is 0.0-1.0 float (if included)
7. genAI_sentiment_label is one of: "Very Negative", "Negative", "Neutral", "Positive", "Very Positive" (if included)
8. short_title should be a clean, readable version without excessive keywords (if included)
9. product_type should be a clean category like "Extension Cord Winder" not the full title (if included)

JSON:'''


def get_all_genai_fields(genai_config: dict) -> dict[str, bool]:
    """Get all GenAI fields with their enabled status."""
    all_fields = {}

    # Basic fields
    for field, enabled in genai_config.get("basic_fields", {}).items():
        all_fields[field] = enabled

    # Parent fields
    for field, enabled in genai_config.get("parent_fields", {}).items():
        all_fields[field] = enabled

    # Child fields
    for section in ["child_description_fields", "child_features_fields",
                    "child_specs_fields", "child_reviews_fields", "child_use_cases_fields"]:
        for field, enabled in genai_config.get(section, {}).items():
            all_fields[field] = enabled

    return all_fields


class LLMExtractor:
    """Extracts product information using Ollama LLM with configurable fields.

    Uses a SINGLE LLM call per file to extract markdown data AND generate GenAI fields.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        concurrency: int = 2,
        timeout: float = 180.0,
        max_content_length: int = 8000,
        genai_config: dict = None,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.genai_config = genai_config or {}
        self.semaphore = asyncio.Semaphore(concurrency)

        # Build COMBINED prompt for single-call extraction
        self.combined_prompt_template = build_combined_prompt(self.genai_config)

        # Get all fields for output
        self.all_fields = get_all_genai_fields(self.genai_config)
        self.enabled_fields = {k: v for k, v in self.all_fields.items() if v}
        self.disabled_fields = {k: v for k, v in self.all_fields.items() if not v}

        # Stats
        self.success_count = 0
        self.error_count = 0
        self.parse_errors = 0

    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit LLM context."""
        if self.max_content_length > 0 and len(content) > self.max_content_length:
            return content[:self.max_content_length] + "\n...[truncated]"
        return content

    def _parse_llm_response(self, response_text: str) -> dict | None:
        """Parse JSON from LLM response."""
        # First try direct parse
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response
        json_patterns = [
            r'```json\s*(\{[\s\S]*?\})\s*```',  # Markdown code block
            r'```\s*(\{[\s\S]*?\})\s*```',  # Generic code block
            r'\{[\s\S]*\}',  # Any JSON object (last resort)
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        return None

    async def _call_llm(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        num_predict: int = 2000,
    ) -> tuple[dict | None, str]:
        """Call Ollama LLM and return parsed response."""
        try:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": num_predict,
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return None, f"Ollama API error: {response.status_code}"

            result = response.json()
            llm_response = result.get("response", "")

            parsed = self._parse_llm_response(llm_response)
            if parsed:
                return parsed, ""
            else:
                return None, f"Parse error: {llm_response[:200]}"

        except httpx.TimeoutException:
            return None, "LLM timeout"
        except Exception as e:
            return None, str(e)

    async def _extract_from_markdown(
        self,
        client: httpx.AsyncClient,
        asin: str,
        content: str,
    ) -> dict:
        """Phase 1: Extract additional fields from markdown content."""
        content = self._truncate_content(content)
        prompt = MARKDOWN_EXTRACTION_PROMPT.format(content=content)

        extracted, error = await self._call_llm(client, prompt, num_predict=6000)

        if extracted:
            extracted["asin"] = asin
            extracted["markdown_extraction_status"] = "success"
            return extracted
        else:
            self.parse_errors += 1
            return {
                "asin": asin,
                "markdown_extraction_status": "failed",
                "markdown_extraction_error": error,
            }

    async def _extract_genai_enrichment(
        self,
        client: httpx.AsyncClient,
        asin: str,
        product_data: dict,
    ) -> dict:
        """Phase 2: GenAI enrichment based on combined data."""
        # Prepare data for GenAI prompt
        title = product_data.get("title", "Unknown")
        brand = product_data.get("scraped_brand") or product_data.get("brand", "Unknown")
        category = product_data.get("category_level1") or product_data.get("categoryName", "Unknown")
        price = product_data.get("price", "Unknown")
        stars = product_data.get("stars", "Unknown")
        reviews_count = product_data.get("reviews") or product_data.get("reviews_count", "Unknown")

        # Use scraped description or original
        description = product_data.get("scraped_description") or product_data.get("description", "")

        # Format features
        features = product_data.get("scraped_features", [])
        if isinstance(features, list):
            features = "\n".join([f"- {f}" for f in features[:10]])
        else:
            features = str(features) if features else ""

        # Format specs
        specs = product_data.get("scraped_technical_details", {})
        if isinstance(specs, dict):
            specs = "\n".join([f"- {k}: {v}" for k, v in list(specs.items())[:15]])
        else:
            specs = str(specs) if specs else ""

        prompt = self.genai_prompt_template.format(
            title=title,
            brand=brand,
            category=category,
            price=price,
            stars=stars,
            reviews_count=reviews_count,
            description=description[:1000] if description else "",
            features=features,
            specs=specs,
        )

        enrichment, error = await self._call_llm(client, prompt, num_predict=6000)

        if enrichment:
            enrichment["genai_enrichment_status"] = "success"
            self.genai_success_count += 1
            return enrichment
        else:
            self.genai_error_count += 1
            return {
                "genai_enrichment_status": "failed",
                "genai_enrichment_error": error,
            }

    def _add_empty_disabled_fields(self, result: dict) -> dict:
        """Add empty values for disabled fields."""
        for field in self.disabled_fields:
            if field not in result:
                result[field] = ""
        return result

    async def _extract_single(
        self,
        client: httpx.AsyncClient,
        asin: str,
        content: str,
        original_data: dict,
    ) -> dict:
        """Single-call extraction: extracts markdown data AND generates GenAI fields in ONE LLM call."""
        async with self.semaphore:
            try:
                # Start with original CSV data as base
                result = dict(original_data)
                result["asin"] = asin

                # Truncate content for LLM context
                content = self._truncate_content(content)

                # Prepare context from original data
                title = original_data.get("title", "Unknown")
                category = original_data.get("category_level1") or original_data.get("categoryName", "Unknown")
                price = original_data.get("price", "Unknown")
                stars = original_data.get("stars", "Unknown")
                reviews_count = original_data.get("reviews") or original_data.get("reviews_count", "Unknown")

                # Build prompt with SINGLE call for both extraction and enrichment
                prompt = self.combined_prompt_template.format(
                    title=title,
                    category=category,
                    price=price,
                    stars=stars,
                    reviews_count=reviews_count,
                    content=content,
                )

                # Single LLM call for everything
                extracted, error = await self._call_llm(client, prompt, num_predict=8000)

                if extracted:
                    # Merge all extracted/generated fields
                    for key, value in extracted.items():
                        result[key] = value
                    result["extraction_status"] = "success"
                    self.success_count += 1
                else:
                    self.parse_errors += 1
                    result["extraction_status"] = "failed"
                    result["error"] = error

                # Add empty values for disabled fields
                result = self._add_empty_disabled_fields(result)
                return result

            except Exception as e:
                self.error_count += 1
                logger.warning("extraction_error", asin=asin, error=str(e))
                result = {
                    **original_data,
                    "asin": asin,
                    "extraction_status": "failed",
                    "error": str(e),
                }
                # Add empty disabled fields even on error
                return self._add_empty_disabled_fields(result)

    async def _process_file(
        self,
        client: httpx.AsyncClient,
        md_file: Path,
        original_lookup: dict,
    ) -> dict:
        """Process a single markdown file."""
        asin = md_file.stem
        original_data = original_lookup.get(asin, {"asin": asin})

        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")

            if not content or len(content) < 50:
                self.error_count += 1
                return self._add_empty_disabled_fields({
                    **original_data,
                    "asin": asin,
                    "extraction_status": "failed",
                    "error": "Insufficient markdown content",
                })

            return await self._extract_single(client, asin, content, original_data)

        except Exception as e:
            self.error_count += 1
            return self._add_empty_disabled_fields({
                **original_data,
                "asin": asin,
                "extraction_status": "failed",
                "error": str(e),
            })

    async def extract_all(
        self,
        markdown_files: list[Path],
        original_df: pd.DataFrame,
    ) -> list[dict]:
        """Extract information from all markdown files with true concurrent processing."""
        # Create lookup dict from original data
        original_lookup = {}
        for _, row in original_df.iterrows():
            asin = str(row.get("asin", ""))
            if asin:
                original_lookup[asin] = row.to_dict()

        async with httpx.AsyncClient() as client:
            # First verify Ollama is accessible
            try:
                health = await client.get(f"{self.ollama_url}/api/tags", timeout=5.0)
                if health.status_code != 200:
                    raise Exception(f"Ollama not accessible: {health.status_code}")
            except Exception as e:
                logger.error("ollama_not_available", error=str(e))
                raise Exception(f"Cannot connect to Ollama at {self.ollama_url}: {e}")

            # Create all tasks upfront for true concurrent processing
            # The semaphore in _extract_single controls actual concurrency
            tasks = [
                self._process_file(client, md_file, original_lookup)
                for md_file in markdown_files
            ]

            # Process concurrently with progress bar
            results = []
            with tqdm(total=len(tasks), desc="Extracting with LLM") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)

        return results


def standardize_field_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize field names to match DB proposal."""
    rename_map = {
        "categoryName": "category_level1",
        "listPrice": "list_price",
        "imgUrl": "img_url",
        "productURL": "product_url",
        "isBestSeller": "is_best_seller",
        "boughtInLastMonth": "bought_in_last_month",
        "reviews": "reviews_count",
    }

    for old_name, new_name in rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    return df


def list_ollama_models(ollama_url: str) -> list[dict]:
    """Fetch available models from Ollama server."""
    import httpx
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except Exception as e:
        print(f"Error fetching models from {ollama_url}: {e}")
    return []


def print_available_models(ollama_url: str):
    """Print available models from Ollama server."""
    models = list_ollama_models(ollama_url)
    if not models:
        print(f"No models found or cannot connect to Ollama at {ollama_url}")
        return

    print(f"\nAvailable models on {ollama_url}:")
    print("-" * 60)
    print(f"{'Model Name':<30} {'Size':<12} {'Modified'}")
    print("-" * 60)
    for m in models:
        name = m.get("name", "unknown")
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024**3)
        modified = m.get("modified_at", "")[:10] if m.get("modified_at") else ""
        print(f"{name:<30} {size_gb:>8.1f} GB   {modified}")
    print("-" * 60)
    print(f"Total: {len(models)} models\n")


@click.command()
@click.option("--input", "input_dir", type=click.Path(exists=True), default=None)
@click.option("--output", "output_path", type=click.Path(), default=None)
@click.option("--original-csv", "original_csv", type=click.Path(exists=True), default=None,
              help="Original products CSV to use as base data")
@click.option("--model", type=str, default=None,
              help="LLM model name (e.g., qwen3:8b, llama3:8b, mistral:7b). Use --list-models to see available models")
@click.option("--ollama-url", type=str, default=None, help="Ollama server URL")
@click.option("--list-models", "list_models", is_flag=True, help="List available models from Ollama server and exit")
@click.option("--concurrency", type=int, default=None)
@click.option("--timeout", type=float, default=None)
@click.option("--start", "start_row", type=int, default=None, help="Start row (1-indexed, inclusive)")
@click.option("--end", "end_row", type=int, default=None, help="End row (1-indexed, inclusive)")
@click.option("--limit", type=int, default=None, help="Limit number of files (alternative to --end)")
@click.option("--no-genai", is_flag=True, help="Disable all GenAI enrichment")
@click.option("--metrics-output", "metrics_path", type=click.Path(), default=None)
def main(
    input_dir: str | None,
    output_path: str | None,
    original_csv: str | None,
    model: str | None,
    ollama_url: str | None,
    list_models: bool,
    concurrency: int | None,
    timeout: float | None,
    start_row: int | None,
    end_row: int | None,
    limit: int | None,
    no_genai: bool,
    metrics_path: str | None,
):
    """Extract product information from Markdown using LLM with GenAI enrichment.

    Uses original CSV as base data and enriches with:
    1. Additional fields extracted from markdown (description, features, specs)
    2. GenAI enrichment fields (summary, use_cases, best_for, etc.)

    Field configuration is in pipeline_config.yaml under genai_enrichment.
    Set individual fields to true/false to enable/disable LLM generation.
    Disabled fields are included in output as empty columns.
    """
    # Load settings
    settings = get_settings()
    script_name = "02c_extract_with_llm"

    # Get ollama_url first (needed for --list-models)
    ollama_url = ollama_url or cfg.get_script(script_name, "ollama_url", "http://localhost:11434")

    # Handle --list-models flag
    if list_models:
        print_available_models(ollama_url)
        sys.exit(0)

    pipeline_mode = cfg.get_mode()
    product_count = cfg.get_count()

    # Use config file values, then CLI overrides (with {mode} in output filename)
    input_dir = Path(input_dir or str(cfg.get_path(script_name, "input_dir", str(settings.scraped_data_dir / "markdown"))))
    output_path = Path(output_path or str(cfg.get_path(script_name, "output", str(settings.scraped_data_dir / f"mvp_{product_count}_{pipeline_mode}_extracted.csv"))))
    original_csv_path = Path(original_csv) if original_csv else cfg.get_path(script_name, "original_csv", str(settings.raw_data_dir / f"mvp_{product_count}_products.csv"))
    metrics_path = Path(metrics_path or str(cfg.get_path(script_name, "metrics", str(settings.metrics_dir / "02c_extraction_metrics.json"))))

    # Get config values with CLI overrides
    model = model or cfg.get_script(script_name, "model", "qwen3:8b")
    concurrency = concurrency if concurrency is not None else cfg.get_script(script_name, "concurrency", 2)
    timeout = timeout if timeout is not None else cfg.get_script(script_name, "timeout", 180.0)

    # Row range settings (1-indexed, inclusive)
    start_row = start_row if start_row is not None else cfg.get_script(script_name, "start_row", 1)
    end_row = end_row if end_row is not None else cfg.get_script(script_name, "end_row")
    limit = limit if limit is not None else cfg.get_script(script_name, "limit")

    # GenAI enrichment configuration
    genai_config = cfg.get_script(script_name, "genai_enrichment", {})
    master_enabled = genai_config.get("enabled", True) and not no_genai

    # If master switch is off, disable all fields
    if not master_enabled:
        for section in ["basic_fields", "parent_fields", "child_description_fields",
                       "child_features_fields", "child_specs_fields",
                       "child_reviews_fields", "child_use_cases_fields"]:
            if section in genai_config:
                genai_config[section] = {k: False for k in genai_config[section]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Count enabled/disabled fields
    all_fields = get_all_genai_fields(genai_config)
    enabled_count = sum(1 for v in all_fields.values() if v)
    disabled_count = sum(1 for v in all_fields.values() if not v)

    logger.info(
        "starting_llm_extraction",
        input_dir=str(input_dir),
        original_csv=str(original_csv_path),
        model=model,
        concurrency=concurrency,
        enabled_fields=enabled_count,
        disabled_fields=disabled_count,
    )

    metrics = {
        "stage": "llm_extraction",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "original_csv": str(original_csv_path),
        "output_file": str(output_path),
        "model": model,
        "genai_master_enabled": master_enabled,
        "enabled_fields": enabled_count,
        "disabled_fields": disabled_count,
    }

    try:
        # Load original CSV as base data
        if not original_csv_path.exists():
            print(f"ERROR: Original CSV not found: {original_csv_path}")
            print("The original CSV is required as the base data source.")
            sys.exit(1)

        original_df = pd.read_csv(original_csv_path)
        print(f"Loaded {len(original_df)} products from original CSV: {original_csv_path}")

        # Find markdown files
        markdown_files = sorted(input_dir.glob("*.md"))

        if not markdown_files:
            print(f"No markdown files found in {input_dir}")
            print("Run 02a_download_html.py and 02b_html_to_markdown.py first.")
            sys.exit(1)

        # Filter to only process ASINs that exist in original CSV
        original_asins = set(original_df["asin"].astype(str).tolist())
        markdown_files = [f for f in markdown_files if f.stem in original_asins]

        if not markdown_files:
            print("No markdown files match ASINs in original CSV.")
            sys.exit(1)

        total_available = len(markdown_files)

        # Apply row range filtering (1-indexed, inclusive)
        # Convert to 0-indexed for Python slicing
        start_idx = max(0, start_row - 1) if start_row else 0

        if end_row:
            # end_row is inclusive, so we use end_row (not end_row - 1) for slice
            end_idx = min(end_row, total_available)
            markdown_files = markdown_files[start_idx:end_idx]
        elif limit:
            # limit takes N files starting from start_idx
            markdown_files = markdown_files[start_idx:start_idx + limit]
        else:
            # Process all from start_idx
            markdown_files = markdown_files[start_idx:]

        # Calculate actual range for display
        actual_start = start_idx + 1
        actual_end = start_idx + len(markdown_files)

        # Update output paths to include range (only if not processing all files)
        if actual_start > 1 or actual_end < total_available:
            # Insert range before file extension: file.csv -> file_1-100.csv
            output_stem = output_path.stem
            output_path = output_path.with_name(f"{output_stem}_{actual_start}-{actual_end}{output_path.suffix}")
            # Also update metrics path
            metrics_stem = metrics_path.stem
            metrics_path = metrics_path.with_name(f"{metrics_stem}_{actual_start}-{actual_end}{metrics_path.suffix}")

        metrics["total_files"] = len(markdown_files)
        metrics["total_available"] = total_available
        metrics["start_row"] = actual_start
        metrics["end_row"] = actual_end
        metrics["original_products"] = len(original_df)

        print(f"\n{'='*60}")
        print("02c: LLM EXTRACTION + GENAI ENRICHMENT")
        print(f"{'='*60}")
        print(f"Original CSV:      {original_csv_path}")
        print(f"Original products: {len(original_df)}")
        print(f"Available files:   {total_available}")
        print(f"Processing range:  {actual_start} - {actual_end} ({len(markdown_files)} files)")
        print(f"Model:             {model}")
        print(f"Ollama URL:        {ollama_url}")
        print(f"Concurrency:       {concurrency}")
        print(f"GenAI master:      {'enabled' if master_enabled else 'DISABLED'}")
        print(f"Fields enabled:    {enabled_count}")
        print(f"Fields disabled:   {disabled_count} (will be empty in output)")
        print(f"{'='*60}\n")

        # Extract with configurable fields
        extractor = LLMExtractor(
            ollama_url=ollama_url,
            model=model,
            concurrency=concurrency,
            timeout=timeout,
            genai_config=genai_config,
        )

        results = asyncio.run(extractor.extract_all(markdown_files, original_df))

        # Create dataframe from results
        extracted_df = pd.DataFrame(results)

        # Standardize field names
        extracted_df = standardize_field_names(extracted_df)

        # Convert complex columns to JSON strings for CSV
        for col in extracted_df.columns:
            if extracted_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                extracted_df[col] = extracted_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )

        # Save output CSV
        extracted_df.to_csv(output_path, index=False)
        logger.info("output_saved", path=str(output_path))

        # Also save as JSON for easier inspection
        json_output = output_path.with_suffix(".json")
        with open(json_output, "w") as f:
            json.dump(results, f, indent=2, default=str)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate coverage
        def safe_coverage(col: str) -> float:
            if col in extracted_df.columns:
                non_null = extracted_df[col].notna() & (extracted_df[col] != "null") & (extracted_df[col] != "")
                return non_null.sum() / len(extracted_df) * 100
            return 0.0

        metrics.update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "metrics": {
                "original_products": len(original_df),
                "markdown_files_processed": len(markdown_files),
                "successful": extractor.success_count,
                "failed": extractor.error_count,
                "parse_errors": extractor.parse_errors,
                "success_rate": round(extractor.success_count / len(markdown_files) * 100, 2) if markdown_files else 0,
                # Field coverage (sample)
                "title_coverage": round(safe_coverage("title"), 2),
                "brand_coverage": round(safe_coverage("brand"), 2),
                "short_title_coverage": round(safe_coverage("short_title"), 2),
                "product_type_coverage": round(safe_coverage("product_type"), 2),
                "genAI_summary_coverage": round(safe_coverage("genAI_summary"), 2),
                "avg_time_per_file": round(processing_time / len(markdown_files), 2) if markdown_files else 0,
            },
            "enabled_fields_list": [k for k, v in all_fields.items() if v],
            "disabled_fields_list": [k for k, v in all_fields.items() if not v],
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("LLM EXTRACTION COMPLETE (Single-Call Mode)")
        print("=" * 60)
        print(f"Model:             {model}")
        print(f"Concurrency:       {concurrency}")
        print(f"Original products: {len(original_df):,}")
        print(f"Markdown files:    {len(markdown_files):,}")
        print(f"Successful:        {extractor.success_count:,}")
        print(f"Failed:            {extractor.error_count:,}")
        print(f"Parse errors:      {extractor.parse_errors:,}")
        print(f"Success rate:      {extractor.success_count / len(markdown_files) * 100:.2f}%" if markdown_files else "N/A")
        print(f"\nFields Configuration:")
        print(f"  Enabled:         {enabled_count}")
        print(f"  Disabled:        {disabled_count}")
        print(f"\nPerformance:")
        print(f"  Total time:      {processing_time:.2f}s")
        print(f"  Avg time/file:   {processing_time / len(markdown_files):.2f}s" if markdown_files else "N/A")
        print(f"  Throughput:      {len(markdown_files) / processing_time * 60:.1f} files/min" if processing_time > 0 else "N/A")
        print(f"\nSample Field Coverage:")
        print(f"  brand:           {safe_coverage('brand'):.2f}%")
        print(f"  short_title:     {safe_coverage('short_title'):.2f}%")
        print(f"  product_type:    {safe_coverage('product_type'):.2f}%")
        print(f"  genAI_summary:   {safe_coverage('genAI_summary'):.2f}%")
        print(f"\nOutput CSV:        {output_path}")
        print(f"Output JSON:       {json_output}")
        print("=" * 60)

    except Exception as e:
        logger.error("extraction_failed", error=str(e))
        import traceback
        traceback.print_exc()
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
