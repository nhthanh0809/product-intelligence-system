#!/usr/bin/env python3
"""
Script 02a: Download HTML from Product URLs

Downloads HTML content from Amazon product pages using Playwright.
Handles popups, consent dialogs, and lazy loading.

Usage:
    python scripts/02a_download_html.py --count 100
    python scripts/02a_download_html.py --input data/raw/products.csv --output data/html

Prerequisites:
    pip install playwright pandas structlog click tqdm
    playwright install chromium
"""

import asyncio
import random
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import click
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


class HTMLDownloader:
    """Downloads HTML from product URLs using Playwright."""

    def __init__(
        self,
        output_dir: Path,
        concurrency: int = 2,
        headless: bool = True,
        timeout: int = 30000,
        slow_mo: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency
        self.headless = headless
        self.timeout = timeout
        self.slow_mo = slow_mo  # Milliseconds to slow down operations (for debugging)
        self.semaphore = asyncio.Semaphore(concurrency)

        # Stats
        self.success_count = 0
        self.error_count = 0
        self.skipped_count = 0

    async def _dismiss_popups(self, page) -> None:
        """Dismiss common popups and consent dialogs."""
        # First handle the "Continue Shopping" interstitial page
        try:
            # Check for "Continue shopping" link/button (Amazon's consent interstitial)
            continue_selectors = [
                "a:has-text('Continue shopping')",
                "a:has-text('Continue Shopping')",
                "input[type='submit'][value*='Continue']",
                "button:has-text('Continue')",
                "a.a-button-text:has-text('Continue')",
                "[data-action-type='DISMISS']",
                "a[href*='ref=cs_503']",
            ]

            for selector in continue_selectors:
                try:
                    elem = await page.query_selector(selector)
                    if elem and await elem.is_visible():
                        logger.info("clicking_continue", selector=selector)
                        await elem.click()
                        await asyncio.sleep(2)  # Wait for page to load
                        break
                except Exception:
                    pass

            # Also try clicking by text content
            try:
                continue_link = await page.get_by_text("Continue shopping", exact=False).first
                if continue_link and await continue_link.is_visible():
                    await continue_link.click()
                    await asyncio.sleep(2)
            except Exception:
                pass

        except Exception as e:
            logger.debug("continue_shopping_not_found", error=str(e))

        # Handle "Choose your location" popup
        location_popup_close_selectors = [
            "#GLUXConfirmClose",                    # Close button on location popup
            "#a-popover-1 .a-popover-close",       # Generic popover close
            ".a-popover-close",                     # Any popover close button
            "[data-action='a-popover-close']",     # Close by data action
            "button[aria-label='Close']",          # Close by aria label
            "#GLUXZipUpdate",                       # "Apply" button (alternative)
            ".a-button-close",                      # Generic close button
        ]

        for selector in location_popup_close_selectors:
            try:
                elem = await page.query_selector(selector)
                if elem and await elem.is_visible():
                    logger.info("closing_location_popup", selector=selector)
                    await elem.click()
                    await asyncio.sleep(0.5)
                    break
            except Exception:
                pass

        # Then handle other popups (cookie consent, etc.)
        popup_selectors = [
            "#sp-cc-accept",
            "#sp-cc-rejectall-link",
            "[data-action='sp-cc-accept']",
            "input[data-cel-widget='sp-cc-accept']",
            "#nav-global-location-popover-link",
        ]

        for selector in popup_selectors:
            try:
                elem = await page.query_selector(selector)
                if elem and await elem.is_visible():
                    await elem.click()
                    await asyncio.sleep(0.5)
            except Exception:
                pass

    async def _check_is_product_page(self, page) -> bool:
        """Check if current page is a product page (not consent/interstitial)."""
        # Check for product-specific elements
        product_indicators = [
            "#productTitle",
            "#dp-container",
            "#ppd",
            "#feature-bullets",
        ]

        for selector in product_indicators:
            elem = await page.query_selector(selector)
            if elem:
                return True

        # Check page title - consent pages have generic titles
        title = await page.title()
        if title and ("Amazon.ca" == title.strip() or "Amazon.com" == title.strip()):
            return False  # Likely consent page

        return False

    async def _safe_goto(self, page, url: str, timeout: int = None) -> tuple[bool, str]:
        """Safely navigate to URL, returns (success, error_message)."""
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout or self.timeout,
            )
            if response and response.status == 200:
                return True, ""
            return False, f"HTTP {response.status if response else 'No response'}"
        except Exception as e:
            return False, str(e)

    async def _safe_evaluate(self, page, script: str) -> None:
        """Safely execute JavaScript on page."""
        try:
            await page.evaluate(script)
        except Exception as e:
            logger.debug("evaluate_error", error=str(e))

    async def _download_page(
        self,
        page,
        url: str,
        asin: str,
        skip_existing: bool = True,
    ) -> dict:
        """Download HTML from a single URL with comprehensive error handling."""
        output_path = self.output_dir / f"{asin}.html"

        # Skip if exists
        if skip_existing and output_path.exists():
            self.skipped_count += 1
            return {
                "asin": asin,
                "status": "skipped",
                "path": str(output_path),
            }

        async with self.semaphore:
            try:
                # Random delay to avoid rate limiting
                await asyncio.sleep(random.uniform(1, 3))

                # Navigate to product details
                detail_url = url.rstrip("/") + "#productDetails"
                success, error = await self._safe_goto(page, detail_url)

                if not success:
                    # Try once more with plain URL
                    success, error = await self._safe_goto(page, url)
                    if not success:
                        self.error_count += 1
                        return {
                            "asin": asin,
                            "status": "failed",
                            "error": f"Navigation failed: {error}",
                        }

                # Wait a moment for page to settle
                await asyncio.sleep(1)

                # Dismiss popups and handle consent page (with error protection)
                try:
                    await self._dismiss_popups(page)
                except Exception as e:
                    logger.debug("dismiss_popups_error", asin=asin, error=str(e))

                # Check if we're on a product page, retry navigation if not
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        is_product = await self._check_is_product_page(page)
                        if is_product:
                            break
                    except Exception:
                        is_product = False

                    logger.info("not_product_page_retrying", asin=asin, retry=retry + 1)

                    # Try dismissing popups again
                    try:
                        await self._dismiss_popups(page)
                        await asyncio.sleep(2)
                    except Exception:
                        pass

                    # If still not on product page, try navigating again
                    try:
                        is_product = await self._check_is_product_page(page)
                        if not is_product:
                            await self._safe_goto(page, detail_url)
                            await asyncio.sleep(2)
                            await self._dismiss_popups(page)
                    except Exception:
                        pass

                # Final check (don't fail if we can't verify)
                try:
                    is_product = await self._check_is_product_page(page)
                    if not is_product:
                        logger.warning("could_not_reach_product_page", asin=asin)
                except Exception:
                    pass

                # Wait for product content (don't fail if timeout)
                try:
                    await page.wait_for_selector(
                        "#productTitle, #title, #dp-container",
                        timeout=10000,
                    )
                except Exception:
                    pass

                await asyncio.sleep(1)

                # Dismiss any new popups
                try:
                    await self._dismiss_popups(page)
                except Exception:
                    pass

                # Scroll to load lazy content (safely) - scroll further for full page
                await self._safe_evaluate(page, """
                    async () => {
                        const scrollStep = 500;
                        const maxScroll = document.body.scrollHeight;
                        // First pass: scroll to bottom to trigger all lazy loading
                        for (let y = 0; y < Math.min(maxScroll, 8000); y += scrollStep) {
                            window.scrollTo(0, y);
                            await new Promise(r => setTimeout(r, 150));
                        }
                        // Scroll to product details section
                        const details = document.querySelector(
                            '#productDetails_feature_div, #detailBullets_feature_div, #productDescription, #feature-bullets, #prodDetails'
                        );
                        if (details) {
                            details.scrollIntoView({ behavior: 'instant', block: 'center' });
                        }
                    }
                """)

                await asyncio.sleep(1.5)

                # Click "See more product details" link if present
                try:
                    see_more_selectors = [
                        "a:has-text('See more product details')",
                        "a:has-text('See More')",
                        "#productDetails_expand_link",
                        "#poExpander .a-expander-prompt",
                        "[data-action='a-expander-toggle']:has-text('See more')",
                    ]
                    for selector in see_more_selectors:
                        try:
                            elem = await page.query_selector(selector)
                            if elem and await elem.is_visible():
                                await elem.click()
                                await asyncio.sleep(1)
                                logger.debug("clicked_see_more", asin=asin, selector=selector)
                                break
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug("see_more_click_error", asin=asin, error=str(e))

                # Expand ALL collapsed sections including product details
                await self._safe_evaluate(page, """
                    () => {
                        // Click all expander toggles
                        const expanders = document.querySelectorAll(
                            '[data-action="a-expander-toggle"], .a-expander-header, .a-expander-prompt'
                        );
                        expanders.forEach(exp => {
                            if (exp.getAttribute('aria-expanded') === 'false' ||
                                exp.closest('.a-expander-container')?.querySelector('.a-expander-content[aria-expanded="false"]')) {
                                try { exp.click(); } catch(e) {}
                            }
                        });

                        // Specifically expand product details sections
                        const detailExpanders = document.querySelectorAll(
                            '#productDetails_techSpec_section_1 .a-expander-header, ' +
                            '#productDetails_detailBullets_sections1 .a-expander-header, ' +
                            '#poExpander .a-expander-prompt, ' +
                            '#productDescription_expander, ' +
                            '.prodDetSectionEntry .a-expander-header'
                        );
                        detailExpanders.forEach(exp => {
                            try { exp.click(); } catch(e) {}
                        });

                        // Click on "Show more" buttons in product description
                        const showMoreButtons = document.querySelectorAll(
                            '.a-expander-partial-collapse-header, ' +
                            '[data-action="a-expander-toggle"]'
                        );
                        showMoreButtons.forEach(btn => {
                            try { btn.click(); } catch(e) {}
                        });
                    }
                """)

                await asyncio.sleep(1)

                # Click on Product Information tab if it exists (some layouts use tabs)
                try:
                    tab_selectors = [
                        "#product-details-tab",
                        "a[href='#productDetails']",
                        "#dp-product-details_feature_div a",
                        ".product-details-tab",
                    ]
                    for selector in tab_selectors:
                        try:
                            elem = await page.query_selector(selector)
                            if elem and await elem.is_visible():
                                await elem.click()
                                await asyncio.sleep(0.5)
                                break
                        except Exception:
                            pass
                except Exception:
                    pass

                # Scroll to Technical Details section and wait for it to load
                await self._safe_evaluate(page, """
                    async () => {
                        // Find and scroll to technical details
                        const techDetails = document.querySelector(
                            '#productDetails_techSpec_section_1, ' +
                            '#technicalSpecifications_section_1, ' +
                            '#prodDetails, ' +
                            '#detailBulletsWrapper_feature_div, ' +
                            '#productDescription_feature_div'
                        );
                        if (techDetails) {
                            techDetails.scrollIntoView({ behavior: 'instant', block: 'start' });
                            await new Promise(r => setTimeout(r, 500));
                        }

                        // Also scroll to Additional Information if present
                        const additionalInfo = document.querySelector(
                            '#productDetails_detailBullets_sections1, ' +
                            '#detailBullets_feature_div'
                        );
                        if (additionalInfo) {
                            additionalInfo.scrollIntoView({ behavior: 'instant', block: 'start' });
                            await new Promise(r => setTimeout(r, 500));
                        }
                    }
                """)

                await asyncio.sleep(0.5)

                # Final expansion pass for any remaining collapsed content
                await self._safe_evaluate(page, """
                    () => {
                        // Expand any remaining collapsed sections
                        document.querySelectorAll('.a-expander-header[aria-expanded="false"]').forEach(el => {
                            try { el.click(); } catch(e) {}
                        });

                        // Force display of hidden product details
                        document.querySelectorAll('.a-expander-content').forEach(el => {
                            el.style.display = 'block';
                            el.setAttribute('aria-expanded', 'true');
                        });
                    }
                """)

                await asyncio.sleep(0.5)

                # Scroll to and load Customer Reviews section
                await self._safe_evaluate(page, """
                    async () => {
                        // Find customer reviews section
                        const reviewsSection = document.querySelector(
                            '#reviewsMedley, ' +
                            '#customer-reviews-content, ' +
                            '#cm-cr-dp-review-list, ' +
                            '[data-hook="reviews-medley-widget"]'
                        );
                        if (reviewsSection) {
                            reviewsSection.scrollIntoView({ behavior: 'instant', block: 'start' });
                            await new Promise(r => setTimeout(r, 1000));
                        }

                        // Find and scroll to "Frequently bought together" section
                        const frequentlyBought = document.querySelector(
                            '#sims-fbt, ' +
                            '#frequently-bought-together, ' +
                            '[data-component-type="s-similar-products"]'
                        );
                        if (frequentlyBought) {
                            frequentlyBought.scrollIntoView({ behavior: 'instant', block: 'start' });
                            await new Promise(r => setTimeout(r, 500));
                        }
                    }
                """)

                await asyncio.sleep(0.5)

                # Expand "Read more" links in reviews
                await self._safe_evaluate(page, """
                    () => {
                        // Expand review text "Read more" links
                        document.querySelectorAll('[data-hook="expand-collapse-read-more-less"], .review-text-read-more-expander .a-expander-header').forEach(el => {
                            try { el.click(); } catch(e) {}
                        });

                        // Expand all review bodies
                        document.querySelectorAll('.review-text-content').forEach(el => {
                            el.style.maxHeight = 'none';
                            el.style.overflow = 'visible';
                        });
                    }
                """)

                await asyncio.sleep(0.3)

                # Get final URL (after redirects)
                try:
                    final_url = page.url
                except Exception:
                    final_url = url

                # Get HTML content
                try:
                    html = await page.content()
                except Exception as e:
                    self.error_count += 1
                    return {
                        "asin": asin,
                        "status": "failed",
                        "error": f"Failed to get page content: {str(e)}",
                    }

                # Verify we got actual product content
                has_product_content = "#productTitle" in html or "feature-bullets" in html
                if not has_product_content:
                    logger.warning("html_missing_product_content", asin=asin)

                # Save HTML
                try:
                    output_path.write_text(html, encoding="utf-8")
                except Exception as e:
                    self.error_count += 1
                    return {
                        "asin": asin,
                        "status": "failed",
                        "error": f"Failed to save HTML: {str(e)}",
                    }

                self.success_count += 1
                return {
                    "asin": asin,
                    "status": "success",
                    "path": str(output_path),
                    "final_url": final_url,
                    "size_bytes": len(html),
                    "has_product_content": has_product_content,
                }

            except asyncio.CancelledError:
                # Re-raise cancellation to allow graceful shutdown
                raise
            except Exception as e:
                self.error_count += 1
                logger.warning("download_error", asin=asin, error=str(e))
                return {
                    "asin": asin,
                    "status": "failed",
                    "error": str(e),
                }

    async def _worker(
        self,
        worker_id: int,
        queue: asyncio.Queue,
        page,
        results: list,
        pbar: tqdm,
        skip_existing: bool,
    ) -> None:
        """Worker that processes products from queue."""
        while True:
            try:
                product = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            url = product.get("productURL")
            asin = product.get("asin", "unknown")

            try:
                if url and asin:
                    result = await self._download_page(page, url, asin, skip_existing)
                else:
                    result = {"asin": asin, "status": "failed", "error": "Missing URL or ASIN"}
                    self.error_count += 1
            except Exception as e:
                # Catch ANY error and continue to next product
                logger.warning("worker_error", worker_id=worker_id, asin=asin, error=str(e))
                result = {"asin": asin, "status": "failed", "error": f"Worker error: {str(e)}"}
                self.error_count += 1

                # Try to recover the page if it crashed
                try:
                    await page.goto("about:blank", timeout=5000)
                except Exception:
                    pass

            results.append(result)
            pbar.update(1)

            try:
                queue.task_done()
            except Exception:
                pass

    async def download_all(
        self,
        products: list[dict],
        skip_existing: bool = True,
    ) -> list[dict]:
        """Download HTML for all products using parallel workers."""
        from playwright.async_api import async_playwright

        results = []
        num_workers = min(self.concurrency, len(products))

        if not products:
            logger.info("no_products_to_download")
            return results

        browser = None
        contexts = []
        pages = []
        pbar = None

        try:
            async with async_playwright() as p:
                try:
                    browser = await p.chromium.launch(
                        headless=self.headless,
                        slow_mo=self.slow_mo,
                        args=[
                            "--disable-blink-features=AutomationControlled",
                            "--no-sandbox",
                            "--disable-setuid-sandbox",
                            "--disable-dev-shm-usage",
                        ]
                    )
                except Exception as e:
                    logger.error("browser_launch_failed", error=str(e))
                    # Mark all products as failed
                    for product in products:
                        results.append({
                            "asin": product.get("asin", "unknown"),
                            "status": "failed",
                            "error": f"Browser launch failed: {str(e)}",
                        })
                        self.error_count += 1
                    return results

                # Create separate contexts for better isolation
                for i in range(num_workers):
                    try:
                        context = await browser.new_context(
                            viewport={"width": 1920, "height": 1080},
                            user_agent=f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{120 + i}.0.0.0 Safari/537.36",
                            locale="en-US",
                            timezone_id="America/New_York",
                        )

                        # Stealth
                        await context.add_init_script("""
                            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                        """)

                        page = await context.new_page()
                        contexts.append(context)
                        pages.append(page)
                    except Exception as e:
                        logger.warning("context_creation_failed", worker=i, error=str(e))
                        # Continue with fewer workers
                        continue

                if not pages:
                    logger.error("no_browser_pages_created")
                    for product in products:
                        results.append({
                            "asin": product.get("asin", "unknown"),
                            "status": "failed",
                            "error": "No browser pages could be created",
                        })
                        self.error_count += 1
                    return results

                actual_workers = len(pages)
                logger.info("workers_created", requested=num_workers, actual=actual_workers)

                # Create queue and fill with products
                queue = asyncio.Queue()
                for product in products:
                    await queue.put(product)

                # Create progress bar
                pbar = tqdm(total=len(products), desc=f"Downloading HTML ({actual_workers} workers)")

                # Start workers
                workers = [
                    asyncio.create_task(
                        self._worker(i, queue, pages[i], results, pbar, skip_existing)
                    )
                    for i in range(actual_workers)
                ]

                # Wait for all workers to complete (with error handling)
                try:
                    await asyncio.gather(*workers, return_exceptions=True)
                except Exception as e:
                    logger.error("workers_failed", error=str(e))

                # Handle any remaining items in queue (if workers crashed)
                remaining = queue.qsize()
                if remaining > 0:
                    logger.warning("unprocessed_products", count=remaining)
                    while not queue.empty():
                        try:
                            product = queue.get_nowait()
                            results.append({
                                "asin": product.get("asin", "unknown"),
                                "status": "failed",
                                "error": "Worker crashed before processing",
                            })
                            self.error_count += 1
                            if pbar:
                                pbar.update(1)
                        except asyncio.QueueEmpty:
                            break

                if pbar:
                    pbar.close()

                # Cleanup (with error handling)
                for page in pages:
                    try:
                        await page.close()
                    except Exception:
                        pass
                for context in contexts:
                    try:
                        await context.close()
                    except Exception:
                        pass
                try:
                    await browser.close()
                except Exception:
                    pass

        except Exception as e:
            logger.error("download_all_failed", error=str(e))
            # Mark remaining products as failed
            processed_asins = {r.get("asin") for r in results}
            for product in products:
                asin = product.get("asin", "unknown")
                if asin not in processed_asins:
                    results.append({
                        "asin": asin,
                        "status": "failed",
                        "error": f"Download process failed: {str(e)}",
                    })
                    self.error_count += 1

        return results


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True), default=None,
              help="Input CSV file with product URLs")
@click.option("--output", "output_dir", type=click.Path(), default=None,
              help="Output directory for HTML files")
@click.option("--start", "start_row", type=int, default=1,
              help="Start row number (1-indexed, default: 1)")
@click.option("--end", "end_row", type=int, default=None,
              help="End row number (1-indexed, inclusive, default: last row)")
@click.option("--count", "limit", type=int, default=None,
              help="Number of products to process (alternative to --end)")
@click.option("--concurrency", type=int, default=4,
              help="Number of parallel browser workers (default: 4, max recommended: 8)")
@click.option("--headless/--no-headless", default=True,
              help="Run browser in headless mode (use --no-headless to see browser)")
@click.option("--slow-mo", "slow_mo", type=int, default=0,
              help="Slow down browser operations by N milliseconds (for debugging)")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip files that already exist")
@click.option("--metrics-output", "metrics_path", type=click.Path(), default=None,
              help="Path to save metrics JSON")
def main(
    input_path: str | None,
    output_dir: str | None,
    start_row: int,
    end_row: int | None,
    limit: int | None,
    concurrency: int,
    headless: bool,
    slow_mo: int,
    skip_existing: bool,
    metrics_path: str | None,
):
    """Download HTML from product URLs.
    python3 02a_download_html.py --start 7501 --end 10000 --concurrency 4
    Examples:
        # Download first 10 products
        python scripts/02a_download_html.py --count 10

        # Download products from row 2 to row 8 (inclusive)
        python scripts/02a_download_html.py --start 2 --end 8

        # Download products starting from row 100
        python scripts/02a_download_html.py --start 100 --count 50

        # Download with visible browser
        python scripts/02a_download_html.py --start 1 --end 5 --no-headless
    """
    # Load settings
    settings = get_settings()
    script_name = "02a_download_html"

    # Use config file values, then CLI overrides
    input_path = Path(input_path or str(cfg.get_path(script_name, "input", str(settings.raw_data_dir / "mvp_products.csv"))))
    output_dir = Path(output_dir or str(cfg.get_path(script_name, "output_dir", str(settings.scraped_data_dir / "html"))))
    metrics_path = Path(metrics_path or str(cfg.get_path(script_name, "metrics", str(settings.metrics_dir / "02a_download_metrics.json"))))

    # Get other config values
    start_row = start_row if start_row != 1 else cfg.get_script(script_name, "start_row", 1)
    end_row = end_row if end_row is not None else cfg.get_script(script_name, "end_row")
    limit = limit if limit is not None else cfg.get_script(script_name, "count")
    concurrency = concurrency if concurrency != 4 else cfg.get_script(script_name, "concurrency", 4)
    headless = headless if headless else cfg.get_script(script_name, "headless", True)
    skip_existing = skip_existing if skip_existing else cfg.get_script(script_name, "skip_existing", True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("starting_html_download", input_path=str(input_path), output_dir=str(output_dir))

    metrics = {
        "stage": "download_html",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_dir": str(output_dir),
    }

    try:
        # Load products
        df = pd.read_csv(input_path)
        products = df.to_dict("records")
        total_in_file = len(products)

        # Apply row range selection (1-indexed, inclusive)
        start_idx = max(0, start_row - 1)  # Convert to 0-indexed

        if end_row is not None:
            # Use explicit end row (inclusive, so we use end_row as the slice end)
            end_idx = min(end_row, total_in_file)
            products = products[start_idx:end_idx]
        elif limit is not None:
            # Use count/limit from start position
            products = products[start_idx:start_idx + limit]
        else:
            # Just apply start position
            products = products[start_idx:]

        logger.info(
            "row_range_applied",
            total_in_file=total_in_file,
            start_row=start_row,
            end_row=end_row,
            limit=limit,
            selected_count=len(products),
        )

        metrics["total_products"] = len(products)
        metrics["row_range"] = {
            "start_row": start_row,
            "end_row": end_row,
            "limit": limit,
            "total_in_file": total_in_file,
        }

        # Download
        downloader = HTMLDownloader(
            output_dir=output_dir,
            concurrency=concurrency,
            headless=headless,
            slow_mo=slow_mo,
        )

        results = asyncio.run(downloader.download_all(products, skip_existing))

        # Save results manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)

        end_time = time.time()
        processing_time = end_time - start_time

        actual_processed = downloader.success_count + downloader.error_count
        speed = actual_processed / processing_time if processing_time > 0 else 0

        metrics.update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "processing_time_seconds": round(processing_time, 2),
            "concurrency": concurrency,
            "metrics": {
                "total_products": len(products),
                "successful": downloader.success_count,
                "failed": downloader.error_count,
                "skipped": downloader.skipped_count,
                "success_rate": round(downloader.success_count / len(products) * 100, 2) if products else 0,
                "pages_per_second": round(speed, 2),
            },
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("HTML DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Input file:        {input_path}")
        print(f"Total in file:     {total_in_file:,}")
        print(f"Row range:         {start_row} to {end_row or (start_row + len(products) - 1)}")
        print(f"Selected products: {len(products):,}")
        print(f"Parallel workers:  {concurrency}")
        print("-" * 60)
        print(f"Downloaded:        {downloader.success_count:,}")
        print(f"Failed:            {downloader.error_count:,}")
        print(f"Skipped:           {downloader.skipped_count:,}")
        print(f"Processing time:   {processing_time:.2f}s")
        print(f"Speed:             {speed:.2f} pages/sec")
        print("-" * 60)
        print(f"Output directory:  {output_dir}")
        print(f"Manifest:          {manifest_path}")
        print("=" * 60)

    except Exception as e:
        logger.error("download_failed", error=str(e))
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
