"""Pipeline runner - orchestrates stage execution."""

import asyncio
from typing import Any, Callable

import structlog

from src.config import get_settings
from src.models.job import PipelineJob
from src.models.enums import PipelineMode, IndexingStrategy, JobStatus
from src.stages.base import StageContext
from src.stages.extract import ExtractStage
from src.stages.clean import CleanStage
from src.stages.embed import EmbedStage
from src.stages.load_postgres import LoadPostgresStage
from src.stages.load_qdrant import LoadQdrantStage
from src.stages.load_elasticsearch import LoadElasticsearchStage
from src.stages.download_html import DownloadHtmlStage
from src.stages.html_to_markdown import HtmlToMarkdownStage
from src.stages.llm_extract import LlmExtractStage

logger = structlog.get_logger()


class PipelineRunner:
    """Orchestrates pipeline execution.

    Manages the flow of data through stages:
    - Original mode: extract -> clean -> embed -> load
    - Enrich mode: + download -> html_to_md -> llm_extract
    """

    def __init__(self, job: PipelineJob, on_progress: Callable | None = None):
        self.job = job
        self.settings = get_settings()
        self.on_progress = on_progress

        # Initialize stages based on mode
        self.context = StageContext(
            job=job,
            batch_size=job.batch_size,
            on_progress=on_progress,
        )

        # Initialize job stages
        job.initialize_stages()

    async def run(self) -> PipelineJob:
        """Run the complete pipeline.

        Returns:
            Updated job with results
        """
        logger.info(
            "pipeline_started",
            job_id=str(self.job.job_id),
            mode=self.job.mode.value,
            strategy=self.job.indexing_strategy.value,
            product_count=self.job.product_count,
        )

        self.job.start()

        try:
            if self.job.mode == PipelineMode.ORIGINAL:
                await self._run_original_pipeline()
            else:
                await self._run_enrich_pipeline()

            self.job.complete()

            logger.info(
                "pipeline_completed",
                job_id=str(self.job.job_id),
                duration_seconds=self.job.duration_seconds,
                metrics=self.job.metrics.model_dump(),
            )

        except Exception as e:
            self.job.fail(str(e))
            logger.error(
                "pipeline_failed",
                job_id=str(self.job.job_id),
                error=str(e),
            )
            raise

        return self.job

    async def _run_original_pipeline(self) -> None:
        """Run original mode pipeline.

        extract -> clean -> embed -> load (postgres, qdrant, elasticsearch)
        """
        # Stage 1: Extract
        extract_stage = ExtractStage(self.context, self.job.csv_path)
        products = await extract_stage.run_extract(
            limit=self.job.product_count,
            offset=self.job.offset,
        )

        if not products:
            logger.warning("no_products_extracted")
            return

        self.job.metrics.products_extracted = len(products)
        self.job.metrics.total_products = len(products)

        # Stage 2: Clean
        clean_stage = CleanStage(self.context, build_chunks=False)
        cleaned = await clean_stage.run(products)
        self.job.metrics.products_cleaned = len(cleaned)

        if not cleaned:
            logger.warning("no_products_cleaned")
            return

        # Stage 3: Embed (with optional model override)
        embedding_model = None
        if self.job.model_config_options:
            embedding_model = self.job.model_config_options.embedding_model
        embed_stage = EmbedStage(self.context, embedding_model=embedding_model)
        embedded = await embed_stage.run(cleaned)
        self.job.metrics.products_embedded = len(embedded)

        if not embedded:
            logger.warning("no_products_embedded")
            return

        # Stage 4: Load (parallel)
        await self._load_all(embedded)

    async def _run_enrich_pipeline(self) -> None:
        """Run enrich mode pipeline.

        Full enrichment flow:
        extract -> clean -> download -> html_to_md -> llm_extract -> embed -> load
        """
        # Stage 1: Extract
        extract_stage = ExtractStage(self.context, self.job.csv_path)
        products = await extract_stage.run_extract(
            limit=self.job.product_count,
            offset=self.job.offset,
        )

        if not products:
            logger.warning("no_products_extracted")
            return

        self.job.metrics.products_extracted = len(products)
        self.job.metrics.total_products = len(products)

        # Stage 2: Clean (with chunk building)
        clean_stage = CleanStage(self.context, build_chunks=True)
        cleaned = await clean_stage.run(products)
        chunks = clean_stage.get_chunks()

        self.job.metrics.products_cleaned = len(cleaned)
        self.job.metrics.chunks_created = len(chunks)

        if not cleaned:
            logger.warning("no_products_cleaned")
            return

        # Stage 3: Download HTML (mock mode by default)
        download_stage = DownloadHtmlStage(self.context, mock_mode=True)
        downloaded = await download_stage.run(cleaned)

        # Stage 4: HTML to Markdown
        markdown_stage = HtmlToMarkdownStage(self.context)
        markdown_processed = await markdown_stage.run(downloaded)

        # Stage 5: LLM Extract (GenAI fields) - with optional model override
        llm_model = None
        llm_temperature = None
        llm_max_tokens = None
        if self.job.model_config_options:
            llm_model = self.job.model_config_options.llm_model
            llm_temperature = self.job.model_config_options.llm_temperature
            llm_max_tokens = self.job.model_config_options.llm_max_tokens

        llm_stage = LlmExtractStage(
            self.context,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
        enriched = await llm_stage.run(markdown_processed)

        # Stage 6: Embed products - with optional model override
        embedding_model = None
        if self.job.model_config_options:
            embedding_model = self.job.model_config_options.embedding_model
        embed_stage = EmbedStage(self.context, embedding_model=embedding_model)
        embedded = await embed_stage.run(enriched)
        self.job.metrics.products_embedded = len(embedded)

        # Stage 6b: Embed chunks
        if chunks:
            embedded_chunks = await embed_stage.embed_chunks(chunks)
            self.job.metrics.chunks_embedded = len(embedded_chunks)
        else:
            embedded_chunks = []

        if not embedded:
            logger.warning("no_products_embedded")
            return

        # Stage 7: Load (with chunks and GenAI fields)
        await self._load_all(embedded, embedded_chunks)

    async def _load_all(
        self,
        products: list,
        chunks: list | None = None,
    ) -> None:
        """Load data to all stores in parallel."""
        # Create load stages
        postgres_stage = LoadPostgresStage(self.context)
        qdrant_stage = LoadQdrantStage(self.context)
        es_stage = LoadElasticsearchStage(self.context)

        # Run PostgreSQL and Elasticsearch loads in parallel
        # Qdrant needs special handling for chunks
        async def load_postgres():
            return await postgres_stage.run(products)

        async def load_elasticsearch():
            return await es_stage.run(products)

        async def load_qdrant():
            if chunks:
                return await qdrant_stage.run_with_chunks(products, chunks)
            else:
                return await qdrant_stage.run(products)

        # Execute loads
        try:
            results = await asyncio.gather(
                load_postgres(),
                load_elasticsearch(),
                load_qdrant(),
                return_exceptions=True,
            )

            # Check for errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    store_name = ["postgres", "elasticsearch", "qdrant"][i]
                    logger.error(
                        "load_stage_failed",
                        store=store_name,
                        error=str(result),
                    )
                    self.job.add_error(f"load_{store_name}", str(result))

        except Exception as e:
            logger.error("load_all_failed", error=str(e))
            raise


async def run_pipeline(
    mode: PipelineMode = PipelineMode.ORIGINAL,
    csv_path: str | None = None,
    product_count: int | None = None,
    batch_size: int = 100,
    indexing_strategy: IndexingStrategy | None = None,
    on_progress: Callable | None = None,
) -> PipelineJob:
    """Convenience function to run a pipeline.

    Args:
        mode: Pipeline mode (original or enrich)
        csv_path: Path to CSV file
        product_count: Number of products to process
        batch_size: Batch size for processing
        indexing_strategy: Qdrant indexing strategy
        on_progress: Progress callback

    Returns:
        Completed PipelineJob
    """
    settings = get_settings()

    # Determine indexing strategy
    if indexing_strategy is None:
        if mode == PipelineMode.ORIGINAL:
            indexing_strategy = IndexingStrategy.PARENT_ONLY
        else:
            indexing_strategy = IndexingStrategy.ADD_CHILD_NODE

    # Create job
    job = PipelineJob(
        mode=mode,
        csv_path=csv_path or settings.csv_path,
        product_count=product_count,
        batch_size=batch_size,
        indexing_strategy=indexing_strategy,
    )

    # Run pipeline
    runner = PipelineRunner(job, on_progress)
    return await runner.run()
