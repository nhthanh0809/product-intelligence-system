#!/usr/bin/env python3
"""
Script 02b: Convert HTML to Markdown

Converts downloaded HTML files to clean Markdown format.
Uses Docling or BeautifulSoup for conversion.

Usage:
    python scripts/02b_html_to_markdown.py
    python scripts/02b_html_to_markdown.py --input data/html --output data/markdown
    python scripts/02b_html_to_markdown.py --use-docling

Prerequisites:
    pip install beautifulsoup4 lxml markdownify
    # Optional: pip install docling docling-core
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
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


def html_to_markdown_simple(html_content: str) -> str:
    """Convert HTML to Markdown using BeautifulSoup + markdownify."""
    from bs4 import BeautifulSoup

    try:
        import markdownify
        has_markdownify = True
    except ImportError:
        has_markdownify = False

    soup = BeautifulSoup(html_content, "lxml")

    # Remove unwanted elements
    for element in soup([
        "script", "style", "nav", "footer", "header", "noscript",
        "iframe", "svg", "link", "meta", "img",
    ]):
        element.decompose()

    # Remove Amazon-specific non-content elements
    remove_selectors = [
        "#sp-cc", ".sp-cc", "[data-cel-widget*='sp-cc']",
        "#nav-main", "#navbar", "#skiplink", "#navFooter",
        ".a-popover", "#a-popover", ".nav-sprite",
        "#rhf", "#rhf-shoveler",  # Recently viewed
        "#sims-fbt",  # Frequently bought together
        "#sponsoredProducts",
        "#ad-feedback-text",
        ".a-carousel",
    ]

    for selector in remove_selectors:
        for elem in soup.select(selector):
            elem.decompose()

    # Extract main product content sections
    sections = []

    # Product title
    title_elem = soup.select_one("#productTitle, #title span")
    if title_elem:
        sections.append(f"# {title_elem.get_text(strip=True)}\n")

    # Price
    price_elem = soup.select_one(".a-price .a-offscreen, #priceblock_ourprice, #corePrice_feature_div .a-offscreen")
    if price_elem:
        sections.append(f"**Price:** {price_elem.get_text(strip=True)}\n")

    # Rating
    rating_elem = soup.select_one("#acrPopover, .a-icon-star span.a-icon-alt")
    if rating_elem:
        rating_text = rating_elem.get("title") or rating_elem.get_text(strip=True)
        sections.append(f"**Rating:** {rating_text}\n")

    # Brand
    brand_elem = soup.select_one("#bylineInfo")
    if brand_elem:
        brand_text = brand_elem.get_text(strip=True)
        brand_text = re.sub(r'^(Visit the|Brand:)\s*', '', brand_text)
        brand_text = re.sub(r'\s*Store$', '', brand_text)
        sections.append(f"**Brand:** {brand_text}\n")

    # About this item / Feature bullets
    bullets_elem = soup.select_one("#feature-bullets")
    if bullets_elem:
        sections.append("\n## About This Item\n")
        for li in bullets_elem.select("li span.a-list-item"):
            text = li.get_text(strip=True)
            if text and len(text) > 5 and not text.startswith("›"):
                sections.append(f"- {text}\n")

    # Product description
    desc_elem = soup.select_one("#productDescription")
    if desc_elem:
        sections.append("\n## Product Description\n")
        desc_text = desc_elem.get_text(separator="\n", strip=True)
        sections.append(f"{desc_text}\n")

    # Technical details
    tech_tables = soup.select("#productDetails_techSpec_section_1, #technicalSpecifications_section_1")
    if tech_tables:
        sections.append("\n## Technical Details\n")
        for table in tech_tables:
            for row in table.select("tr"):
                th = row.select_one("th")
                td = row.select_one("td")
                if th and td:
                    key = th.get_text(strip=True)
                    value = td.get_text(strip=True)
                    sections.append(f"- **{key}:** {value}\n")

    # Additional information
    detail_bullets = soup.select_one("#detailBullets_feature_div, #productDetails_detailBullets_sections1")
    if detail_bullets:
        sections.append("\n## Additional Information\n")
        for row in detail_bullets.select("tr, li"):
            text = row.get_text(separator=": ", strip=True)
            if text and ":" in text:
                sections.append(f"- {text}\n")

    # Product details table
    prod_details = soup.select_one("#productDetails_db_sections")
    if prod_details:
        if not any("Additional Information" in s for s in sections):
            sections.append("\n## Additional Information\n")
        for row in prod_details.select("tr"):
            th = row.select_one("th")
            td = row.select_one("td")
            if th and td:
                key = th.get_text(strip=True)
                value = td.get_text(strip=True)
                sections.append(f"- **{key}:** {value}\n")

    # Availability
    avail_elem = soup.select_one("#availability span, #availability, #outOfStock")
    if avail_elem:
        avail_text = avail_elem.get_text(strip=True)
        if avail_text and len(avail_text) > 2:
            sections.append(f"\n**Availability:** {avail_text}\n")

    # Customer Reviews
    reviews_section = soup.select_one("#reviewsMedley, #cm-cr-dp-review-list, #customer-reviews-content")
    if reviews_section:
        reviews = []
        # Find individual reviews
        review_items = soup.select("[data-hook='review'], .review, .a-section.review")
        for review in review_items[:5]:  # Limit to top 5 reviews
            review_data = {}

            # Review title
            title_elem = review.select_one("[data-hook='review-title'] span:last-child, .review-title-content span")
            if title_elem:
                review_data["title"] = title_elem.get_text(strip=True)

            # Star rating
            rating_elem = review.select_one("[data-hook='review-star-rating'] span, .review-rating span, .a-icon-star span")
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                review_data["rating"] = rating_text

            # Review date
            date_elem = review.select_one("[data-hook='review-date'], .review-date")
            if date_elem:
                review_data["date"] = date_elem.get_text(strip=True)

            # Review body text
            body_elem = review.select_one(".review-text-content span, [data-hook='review-body'] span, .reviewText span")
            if body_elem:
                body_text = body_elem.get_text(strip=True)
                if body_text and len(body_text) > 10:
                    review_data["text"] = body_text

            # Verified purchase
            verified_elem = review.select_one("[data-hook='avp-badge'], [data-hook='avp-badge-linkless']")
            if verified_elem:
                review_data["verified"] = True

            # Helpful votes
            helpful_elem = review.select_one("[data-hook='helpful-vote-statement']")
            if helpful_elem:
                review_data["helpful"] = helpful_elem.get_text(strip=True)

            if review_data.get("text") or review_data.get("title"):
                reviews.append(review_data)

        if reviews:
            sections.append("\n## Customer Reviews\n")
            for i, review in enumerate(reviews, 1):
                title = review.get("title", "Review")
                rating = review.get("rating", "")
                text = review.get("text", "")
                date = review.get("date", "")
                verified = " (Verified Purchase)" if review.get("verified") else ""
                helpful = review.get("helpful", "")

                sections.append(f"### Review {i}: {title}\n")
                if rating:
                    sections.append(f"**Rating:** {rating}\n")
                if date:
                    sections.append(f"**Date:** {date}{verified}\n")
                if text:
                    sections.append(f"{text}\n")
                if helpful:
                    sections.append(f"*{helpful}*\n")
                sections.append("\n")

    # Frequently Bought Together
    fbt_elem = soup.select_one("#sims-fbt, #frequently-bought-together")
    if fbt_elem:
        fbt_items = fbt_elem.select(".a-section a.a-link-normal[href*='/dp/']")
        if fbt_items:
            sections.append("\n## Frequently Bought Together\n")
            seen_asins = set()
            for item in fbt_items[:5]:
                href = item.get("href", "")
                asin_match = re.search(r'/dp/([A-Z0-9]{10})', href)
                if asin_match:
                    asin = asin_match.group(1)
                    if asin not in seen_asins:
                        seen_asins.add(asin)
                        title_span = item.select_one("span.a-size-small, span.a-text-normal")
                        title_text = title_span.get_text(strip=True) if title_span else asin
                        sections.append(f"- [{title_text}](https://www.amazon.com/dp/{asin})\n")

    # If we extracted sections, use them
    if len(sections) > 1:
        markdown = "".join(sections)
    else:
        # Fallback: convert the main content area
        main_content = soup.select_one("#dp-container, #ppd, #centerCol")
        if main_content:
            if has_markdownify:
                markdown = markdownify.markdownify(str(main_content), heading_style="ATX")
            else:
                markdown = main_content.get_text(separator="\n", strip=True)
        else:
            if has_markdownify:
                markdown = markdownify.markdownify(str(soup), heading_style="ATX")
            else:
                markdown = soup.get_text(separator="\n", strip=True)

    # Clean up the markdown
    lines = markdown.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 2:
            # Skip navigation-like short text
            if not re.match(r'^[\s\d\.\,\-\|]+$', line):
                cleaned_lines.append(line)

    markdown = "\n".join(cleaned_lines)

    # Remove excessive blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown


def html_to_markdown_docling(html_path: Path) -> str:
    """Convert HTML to Markdown using Docling."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(html_path))
        markdown = result.document.export_to_markdown()
        return markdown

    except Exception as e:
        logger.warning("docling_failed", path=str(html_path), error=str(e))
        # Fallback to simple conversion
        html_content = html_path.read_text(encoding="utf-8", errors="ignore")
        return html_to_markdown_simple(html_content)


def convert_single_file(args: tuple) -> dict:
    """Convert a single HTML file to Markdown (for multiprocessing)."""
    html_path, output_dir, use_docling = args
    html_path = Path(html_path)
    output_dir = Path(output_dir)

    asin = html_path.stem
    output_path = output_dir / f"{asin}.md"

    try:
        if use_docling:
            markdown = html_to_markdown_docling(html_path)
        else:
            html_content = html_path.read_text(encoding="utf-8", errors="ignore")
            markdown = html_to_markdown_simple(html_content)

        # Save markdown
        output_path.write_text(markdown, encoding="utf-8")

        return {
            "asin": asin,
            "status": "success",
            "input_path": str(html_path),
            "output_path": str(output_path),
            "size_bytes": len(markdown),
        }

    except Exception as e:
        return {
            "asin": asin,
            "status": "failed",
            "input_path": str(html_path),
            "error": str(e),
        }


@click.command()
@click.option("--input", "input_dir", type=click.Path(exists=True), default=None)
@click.option("--output", "output_dir", type=click.Path(), default=None)
@click.option("--use-docling", is_flag=True, help="Use Docling for conversion")
@click.option("--workers", type=int, default=4, help="Number of parallel workers")
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option("--metrics-output", "metrics_path", type=click.Path(), default=None)
def main(
    input_dir: str | None,
    output_dir: str | None,
    use_docling: bool,
    workers: int,
    skip_existing: bool,
    metrics_path: str | None,
):
    """Convert HTML files to Markdown."""
    # Load settings
    settings = get_settings()
    script_name = "02b_html_to_markdown"

    # Use config file values, then CLI overrides
    input_dir = Path(input_dir or str(cfg.get_path(script_name, "input_dir", str(settings.scraped_data_dir / "html"))))
    output_dir = Path(output_dir or str(cfg.get_path(script_name, "output_dir", str(settings.scraped_data_dir / "markdown"))))
    metrics_path = Path(metrics_path or str(cfg.get_path(script_name, "metrics", str(settings.metrics_dir / "02b_markdown_metrics.json"))))

    # Get other config values
    use_docling = use_docling if use_docling else cfg.get_script(script_name, "use_docling", False)
    workers = workers if workers != 4 else cfg.get_script(script_name, "workers", 4)
    skip_existing = skip_existing if skip_existing else cfg.get_script(script_name, "skip_existing", True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("starting_markdown_conversion", input_dir=str(input_dir), use_docling=use_docling)

    metrics = {
        "stage": "html_to_markdown",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "use_docling": use_docling,
    }

    try:
        # Find HTML files
        html_files = list(input_dir.glob("*.html"))

        if not html_files:
            print(f"No HTML files found in {input_dir}")
            sys.exit(1)

        # Filter existing if skip_existing
        if skip_existing:
            html_files = [
                f for f in html_files
                if not (output_dir / f"{f.stem}.md").exists()
            ]

        if not html_files:
            print("All files already converted. Use --no-skip-existing to reconvert.")
            sys.exit(0)

        metrics["total_files"] = len(html_files)

        # Prepare args for multiprocessing
        convert_args = [(str(f), str(output_dir), use_docling) for f in html_files]

        # Convert files
        results = []
        success_count = 0
        error_count = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(convert_single_file, args): args for args in convert_args}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting to Markdown"):
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                else:
                    error_count += 1
                    logger.warning("conversion_failed", asin=result["asin"], error=result.get("error"))

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)

        end_time = time.time()
        processing_time = end_time - start_time

        metrics.update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "metrics": {
                "total_files": len(html_files),
                "successful": success_count,
                "failed": error_count,
                "success_rate": round(success_count / len(html_files) * 100, 2) if html_files else 0,
            },
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("MARKDOWN CONVERSION COMPLETE")
        print("=" * 60)
        print(f"Total files:       {len(html_files):,}")
        print(f"Converted:         {success_count:,}")
        print(f"Failed:            {error_count:,}")
        print(f"Processing time:   {processing_time:.2f}s")
        print(f"Output directory:  {output_dir}")
        print(f"Manifest:          {manifest_path}")
        print("=" * 60)

    except Exception as e:
        logger.error("conversion_failed", error=str(e))
        import traceback
        traceback.print_exc()
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
