"""Individual stage API routes for granular pipeline control."""

from typing import Any, Literal
from datetime import datetime
from pathlib import Path

import structlog
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.models.enums import PipelineMode, IndexingStrategy
from src.models.product import RawProduct, CleanedProduct, EmbeddedProduct
from src.models.job import PipelineJob, StageProgress
from src.stages.base import StageContext
from src.stages.extract import ExtractStage
from src.stages.clean import CleanStage
from src.stages.embed import EmbedStage
from src.stages.download_html import DownloadHtmlStage
from src.stages.html_to_markdown import HtmlToMarkdownStage
from src.stages.llm_extract import LlmExtractStage
from src.stages.load_postgres import LoadPostgresStage
from src.stages.load_qdrant import LoadQdrantStage
from src.stages.load_elasticsearch import LoadElasticsearchStage

logger = structlog.get_logger()
router = APIRouter(prefix="/stages", tags=["stages"])


# =============================================================================
# Utility Functions for Data Conversion
# =============================================================================

def clean_for_json(obj: Any) -> Any:
    """Recursively clean an object for JSON serialization, handling NaN/Inf."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        # Replace NaN/Inf in arrays with 0
        arr = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def convert_numpy(obj: Any) -> Any:
    """Convert numpy types to Python native types, handling NaN/Inf."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        arr = np.where(np.isfinite(obj), obj, 0.0)
        return arr.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


# =============================================================================
# Request/Response Models
# =============================================================================

class ExtractRequest(BaseModel):
    """Request for extract stage."""
    csv_path: str | None = Field(default=None, description="Path to CSV file")
    parquet_path: str | None = Field(default=None, description="Path to Parquet file")
    file_type: Literal["csv", "parquet", "auto"] = Field(
        default="auto",
        description="File type: csv, parquet, or auto (detect from extension)"
    )
    limit: int | None = Field(default=None, ge=1, description="Number of products to extract")
    offset: int = Field(default=0, ge=0, description="Starting row offset")
    batch_size: int = Field(default=100, ge=1, le=1000)


class ExtractResponse(BaseModel):
    """Response from extract stage."""
    success: bool
    count: int
    products: list[dict[str, Any]]
    file_type: str | None = None
    duration_seconds: float


class CleanRequest(BaseModel):
    """Request for clean stage."""
    products: list[dict[str, Any]] = Field(..., description="Raw products to clean")
    mode: PipelineMode = Field(default=PipelineMode.ORIGINAL)
    build_chunks: bool = Field(default=False, description="Build chunks for enrich mode")
    batch_size: int = Field(default=100, ge=1, le=1000)


class CleanResponse(BaseModel):
    """Response from clean stage."""
    success: bool
    count: int
    products: list[dict[str, Any]]
    chunks: list[dict[str, Any]] | None = None
    duration_seconds: float


class DownloadRequest(BaseModel):
    """Request for download HTML stage (enrich mode)."""
    products: list[dict[str, Any]] = Field(..., description="Cleaned products")
    mock_mode: bool = Field(default=True, description="Use mock HTML generation")
    batch_size: int = Field(default=100, ge=1, le=1000)


class DownloadResponse(BaseModel):
    """Response from download stage."""
    success: bool
    count: int
    products: list[dict[str, Any]]
    html_content: dict[str, str]  # asin -> html
    duration_seconds: float


class HtmlToMdRequest(BaseModel):
    """Request for HTML to markdown stage (enrich mode)."""
    products: list[dict[str, Any]] = Field(..., description="Products with HTML")
    html_content: dict[str, str] = Field(..., description="ASIN to HTML mapping")
    batch_size: int = Field(default=100, ge=1, le=1000)


class HtmlToMdResponse(BaseModel):
    """Response from HTML to markdown stage."""
    success: bool
    count: int
    products: list[dict[str, Any]]
    markdown_content: dict[str, str]  # asin -> markdown
    duration_seconds: float


class LlmExtractRequest(BaseModel):
    """Request for LLM extraction stage (enrich mode)."""
    products: list[dict[str, Any]] = Field(..., description="Products to enrich")
    markdown_content: dict[str, str] | None = Field(default=None, description="ASIN to markdown mapping")
    batch_size: int = Field(default=10, ge=1, le=100)
    # Model configuration options
    llm_model: str | None = Field(default=None, description="LLM model (e.g., 'llama3.2:3b', 'qwen2.5:7b', 'mistral:7b')")
    llm_temperature: float | None = Field(default=None, ge=0.0, le=1.0, description="LLM temperature (0.0-1.0)")
    llm_max_tokens: int | None = Field(default=None, ge=100, le=4000, description="Max tokens for LLM response")


class LlmExtractResponse(BaseModel):
    """Response from LLM extraction stage."""
    success: bool
    count: int
    products: list[dict[str, Any]]
    genai_fields: dict[str, dict[str, Any]]  # asin -> genai fields
    llm_model: str | None = None
    duration_seconds: float


class EmbedRequest(BaseModel):
    """Request for embed stage."""
    products: list[dict[str, Any]] = Field(..., description="Cleaned products to embed")
    batch_size: int = Field(default=50, ge=1, le=500)
    # Model configuration options
    embedding_model: str | None = Field(default=None, description="Embedding model (e.g., 'bge-large', 'bge-large', 'mxbai-embed-large')")


class EmbedResponse(BaseModel):
    """Response from embed stage."""
    success: bool
    count: int
    products: list[dict[str, Any]]
    embedding_model: str | None = None
    duration_seconds: float


class LoadRequest(BaseModel):
    """Request for load stages."""
    products: list[dict[str, Any]] = Field(..., description="Embedded products to load")
    mode: PipelineMode = Field(default=PipelineMode.ORIGINAL)
    indexing_strategy: IndexingStrategy | None = Field(default=None)
    genai_fields: dict[str, dict[str, Any]] | None = Field(default=None, description="GenAI fields by ASIN")
    chunks: list[dict[str, Any]] | None = Field(default=None, description="Chunks for enrich mode")
    batch_size: int = Field(default=100, ge=1, le=1000)


class LoadResponse(BaseModel):
    """Response from load stage."""
    success: bool
    count: int
    failed: int
    duration_seconds: float


class LoadParquetRequest(BaseModel):
    """Request for loading pre-embedded parquet files directly to stores."""
    parquet_path: str = Field(..., description="Path to pre-embedded parquet file")
    mode: PipelineMode = Field(default=PipelineMode.ORIGINAL, description="Pipeline mode (original or enrich)")
    indexing_strategy: IndexingStrategy | None = Field(
        default=None,
        description="Indexing strategy for Qdrant (auto-selected based on mode if not specified)"
    )
    targets: list[Literal["postgres", "qdrant", "elasticsearch"]] = Field(
        default=["postgres", "qdrant", "elasticsearch"],
        description="Target stores to load data into"
    )
    batch_size: int = Field(default=500, ge=1, le=2000, description="Batch size for loading")
    limit: int | None = Field(default=None, ge=1, description="Limit number of products to load")
    offset: int = Field(default=0, ge=0, description="Starting row offset")


class LoadParquetResponse(BaseModel):
    """Response from loading parquet file."""
    success: bool
    mode: str
    indexing_strategy: str
    parquet_info: dict[str, Any]
    results: dict[str, dict[str, int]]
    duration_seconds: float


class ResetRequest(BaseModel):
    """Request for resetting/clearing database stores."""
    targets: list[Literal["postgres", "qdrant", "elasticsearch"]] = Field(
        default=["postgres", "qdrant", "elasticsearch"],
        description="Target stores to reset"
    )
    recreate_qdrant: bool = Field(
        default=True,
        description="Recreate Qdrant collection after deletion"
    )
    vector_size: int = Field(
        default=1024,
        description="Vector size for Qdrant collection (if recreating)"
    )
    recreate_elasticsearch: bool = Field(
        default=True,
        description="Recreate Elasticsearch index after deletion"
    )


class ResetResponse(BaseModel):
    """Response from reset operation."""
    success: bool
    results: dict[str, dict[str, Any]]
    duration_seconds: float


# =============================================================================
# Helper Functions
# =============================================================================

def create_stage_context(batch_size: int = 100, mode: PipelineMode = PipelineMode.ORIGINAL) -> StageContext:
    """Create a stage context for individual stage execution."""
    job = PipelineJob(
        mode=mode,
        batch_size=batch_size,
    )
    job.initialize_stages()
    return StageContext(job=job, batch_size=batch_size)


# =============================================================================
# Stage Endpoints
# =============================================================================

@router.post("/extract", response_model=ExtractResponse)
async def run_extract_stage(request: ExtractRequest) -> ExtractResponse:
    """
    Extract products from CSV or Parquet file.

    This is the first stage in both original and enrich modes.
    Returns raw product data that can be passed to the clean stage.

    Supports:
    - CSV files (default)
    - Parquet files (for pre-processed data)
    - Auto-detection of file type based on extension
    """
    start_time = datetime.now()

    try:
        settings = get_settings()
        csv_path = request.csv_path or settings.csv_path

        context = create_stage_context(request.batch_size)
        stage = ExtractStage(
            context,
            csv_path=csv_path,
            parquet_path=request.parquet_path,
            file_type=request.file_type,
        )

        products = await stage.run_extract(
            limit=request.limit,
            offset=request.offset,
        )

        # Determine file type used
        file_path = request.parquet_path or csv_path
        file_type = stage._detect_file_type(file_path) if file_path else None

        duration = (datetime.now() - start_time).total_seconds()

        return ExtractResponse(
            success=True,
            count=len(products),
            products=[p.model_dump() for p in products],
            file_type=file_type,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("extract_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/clean", response_model=CleanResponse)
async def run_clean_stage(request: CleanRequest) -> CleanResponse:
    """
    Clean and normalize products.

    For original mode: normalizes text, generates short_title, product_type.
    For enrich mode: also builds chunks for child node indexing.
    """
    start_time = datetime.now()

    try:
        # Convert dicts to RawProduct
        raw_products = [RawProduct(**p) for p in request.products]

        build_chunks = request.build_chunks or request.mode == PipelineMode.ENRICH

        context = create_stage_context(request.batch_size, request.mode)
        stage = CleanStage(context, build_chunks=build_chunks)

        cleaned = await stage.run(raw_products)
        chunks = stage.get_chunks() if build_chunks else None

        duration = (datetime.now() - start_time).total_seconds()

        return CleanResponse(
            success=True,
            count=len(cleaned),
            products=[p.model_dump() for p in cleaned],
            chunks=[c.model_dump() for c in chunks] if chunks else None,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("clean_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/download", response_model=DownloadResponse)
async def run_download_stage(request: DownloadRequest) -> DownloadResponse:
    """
    Download HTML content for products (enrich mode).

    In mock mode (default), generates HTML from product data.
    In real mode, would download from product URLs.
    """
    start_time = datetime.now()

    try:
        # Convert dicts to CleanedProduct
        products = [CleanedProduct(**p) for p in request.products]

        context = create_stage_context(request.batch_size, PipelineMode.ENRICH)
        stage = DownloadHtmlStage(context, mock_mode=request.mock_mode)

        processed = await stage.run(products)
        html_content = context.data.get("html_content", {})

        duration = (datetime.now() - start_time).total_seconds()

        return DownloadResponse(
            success=True,
            count=len(processed),
            products=[p.model_dump() for p in processed],
            html_content=html_content,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("download_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/html_to_md", response_model=HtmlToMdResponse)
async def run_html_to_md_stage(request: HtmlToMdRequest) -> HtmlToMdResponse:
    """
    Convert HTML to markdown (enrich mode).

    Extracts structured content (title, description, features, specs)
    and converts to clean markdown for LLM processing.
    """
    start_time = datetime.now()

    try:
        # Convert dicts to CleanedProduct
        products = [CleanedProduct(**p) for p in request.products]

        context = create_stage_context(request.batch_size, PipelineMode.ENRICH)
        # Pre-populate HTML content in context
        context.data["html_content"] = request.html_content

        stage = HtmlToMarkdownStage(context)
        processed = await stage.run(products)

        markdown_content = context.data.get("markdown_content", {})

        duration = (datetime.now() - start_time).total_seconds()

        return HtmlToMdResponse(
            success=True,
            count=len(processed),
            products=[p.model_dump() for p in processed],
            markdown_content=markdown_content,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("html_to_md_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/llm_extract", response_model=LlmExtractResponse)
async def run_llm_extract_stage(request: LlmExtractRequest) -> LlmExtractResponse:
    """
    Extract GenAI fields using LLM (enrich mode).

    Uses Ollama LLM to generate:
    - summary, best_for, primary_function
    - use_cases, key_capabilities
    - pros, cons

    Supported LLM models:
    - llama3.2:3b (default, fast)
    - llama3.2:1b (faster, less accurate)
    - llama3.1:8b (more accurate, slower)
    - qwen2.5:7b (good for structured output)
    - mistral:7b (balanced)
    """
    start_time = datetime.now()

    try:
        # Convert dicts to CleanedProduct
        products = [CleanedProduct(**p) for p in request.products]

        context = create_stage_context(request.batch_size, PipelineMode.ENRICH)
        # Pre-populate markdown content if provided
        if request.markdown_content:
            context.data["markdown_content"] = request.markdown_content

        # Determine actual model used
        settings = get_settings()
        actual_model = request.llm_model or settings.llm_model

        # Create stage with model configuration
        stage = LlmExtractStage(
            context,
            model=request.llm_model,
            temperature=request.llm_temperature,
            max_tokens=request.llm_max_tokens,
        )
        processed = await stage.run(products)

        genai_fields = context.data.get("genai_fields", {})

        duration = (datetime.now() - start_time).total_seconds()

        return LlmExtractResponse(
            success=True,
            count=len(processed),
            products=[p.model_dump() for p in processed],
            genai_fields=genai_fields,
            llm_model=actual_model,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("llm_extract_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/embed", response_model=EmbedResponse)
async def run_embed_stage(request: EmbedRequest) -> EmbedResponse:
    """
    Generate embeddings for products.

    Uses Ollama embedding model to create vector embeddings
    for semantic search in Qdrant.

    Supported embedding models:
    - bge-large (default, 768 dimensions)
    - mxbai-embed-large (1024 dimensions)
    - bge-large (1024 dimensions)
    - all-minilm (384 dimensions)
    """
    start_time = datetime.now()

    try:
        # Convert dicts to CleanedProduct
        products = [CleanedProduct(**p) for p in request.products]

        context = create_stage_context(request.batch_size)
        stage = EmbedStage(context, embedding_model=request.embedding_model)

        embedded = await stage.run(products)

        # Determine actual model used
        settings = get_settings()
        actual_model = request.embedding_model or settings.embedding_model

        duration = (datetime.now() - start_time).total_seconds()

        return EmbedResponse(
            success=True,
            count=len(embedded),
            products=[p.model_dump() for p in embedded],
            embedding_model=actual_model,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("embed_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/load/postgres", response_model=LoadResponse)
async def run_load_postgres_stage(request: LoadRequest) -> LoadResponse:
    """
    Load products to PostgreSQL.

    Performs bulk upsert of products into the products table.
    """
    start_time = datetime.now()

    try:
        # Convert dicts to EmbeddedProduct
        products = [EmbeddedProduct(**p) for p in request.products]

        context = create_stage_context(request.batch_size, request.mode)
        stage = LoadPostgresStage(context)

        results = await stage.run(products)

        duration = (datetime.now() - start_time).total_seconds()

        return LoadResponse(
            success=True,
            count=len(results),
            failed=stage.progress.failed,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("load_postgres_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/load/qdrant", response_model=LoadResponse)
async def run_load_qdrant_stage(request: LoadRequest) -> LoadResponse:
    """
    Load products to Qdrant vector store.

    Supports indexing strategies:
    - parent_only: Only parent product nodes (original mode)
    - add_child_node: Parent + child chunk nodes (enrich mode)
    """
    start_time = datetime.now()

    try:
        # Convert dicts to EmbeddedProduct
        products = [EmbeddedProduct(**p) for p in request.products]

        # Determine indexing strategy
        strategy = request.indexing_strategy
        if strategy is None:
            strategy = (
                IndexingStrategy.ADD_CHILD_NODE
                if request.mode == PipelineMode.ENRICH
                else IndexingStrategy.PARENT_ONLY
            )

        context = create_stage_context(request.batch_size, request.mode)
        context.job.indexing_strategy = strategy

        # Add GenAI fields to context if provided
        if request.genai_fields:
            context.data["genai_fields"] = request.genai_fields

        stage = LoadQdrantStage(context, strategy=strategy)

        # Handle chunks for enrich mode
        if request.chunks:
            from src.models.chunk import ProductChunk
            chunks = [ProductChunk(**c) for c in request.chunks]
            result = await stage.run_with_chunks(products, chunks)
            count = result["products"]
        else:
            results = await stage.run(products)
            count = len(results)

        duration = (datetime.now() - start_time).total_seconds()

        return LoadResponse(
            success=True,
            count=count,
            failed=stage.progress.failed,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("load_qdrant_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/load/elasticsearch", response_model=LoadResponse)
async def run_load_elasticsearch_stage(request: LoadRequest) -> LoadResponse:
    """
    Load products to Elasticsearch.

    Indexes products for full-text search capabilities.
    """
    start_time = datetime.now()

    try:
        # Convert dicts to EmbeddedProduct
        products = [EmbeddedProduct(**p) for p in request.products]

        context = create_stage_context(request.batch_size, request.mode)
        stage = LoadElasticsearchStage(context)

        results = await stage.run(products)

        duration = (datetime.now() - start_time).total_seconds()

        return LoadResponse(
            success=True,
            count=len(results),
            failed=stage.progress.failed,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("load_elasticsearch_stage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/load/parquet", response_model=LoadParquetResponse)
async def run_load_parquet(request: LoadParquetRequest) -> LoadParquetResponse:
    """
    Load pre-embedded parquet files directly to stores.

    This endpoint is designed for loading parquet files that already contain
    embeddings (e.g., from external embedding pipelines or pre-processed data).

    Supports two modes:
    - **original**: Loads parent products with parent_only strategy
    - **enrich**: Loads parent updates + child nodes with enrich_existing strategy

    The parquet file should have the following columns:
    - asin: Product identifier (required)
    - embedding: Pre-computed embedding vector (required)
    - title/short_title: Product title
    - node_type: For enrich mode - 'parent', 'parent_update', or child sections
    - Other product fields as needed

    Example usage:
    ```
    POST /stages/load/parquet
    {
        "parquet_path": "/app/data/embed/products_embedded.parquet",
        "mode": "original",
        "indexing_strategy": "parent_only",
        "targets": ["postgres", "qdrant", "elasticsearch"],
        "batch_size": 500
    }
    ```
    """
    start_time = datetime.now()

    try:
        # Validate file exists
        path = Path(request.parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {request.parquet_path}")

        # Read parquet file
        df = pd.read_parquet(request.parquet_path)
        total_rows = len(df)

        # Apply offset and limit
        if request.offset > 0:
            df = df.iloc[request.offset:]
        if request.limit:
            df = df.head(request.limit)

        logger.info(
            "loading_parquet",
            path=request.parquet_path,
            total_rows=total_rows,
            filtered_rows=len(df),
            mode=request.mode.value,
        )

        # Detect embedding dimensions
        sample_embedding = df.iloc[0]["embedding"]
        if isinstance(sample_embedding, np.ndarray):
            vector_dims = sample_embedding.shape[0]
        else:
            vector_dims = len(sample_embedding)

        # Determine indexing strategy
        strategy = request.indexing_strategy
        if strategy is None:
            strategy = (
                IndexingStrategy.ENRICH_EXISTING
                if request.mode == PipelineMode.ENRICH
                else IndexingStrategy.PARENT_ONLY
            )

        # Prepare parquet info
        parquet_info = {
            "total_rows": total_rows,
            "loaded_rows": len(df),
            "embedding_dimensions": vector_dims,
            "columns": list(df.columns),
        }

        # Check for node types in enrich mode
        if request.mode == PipelineMode.ENRICH and "node_type" in df.columns:
            node_types = df["node_type"].value_counts().to_dict()
            parquet_info["node_types"] = {str(k): int(v) for k, v in node_types.items()}

        # Convert to product dicts
        results = {"postgres": {"loaded": 0, "failed": 0}, "qdrant": {"loaded": 0, "failed": 0}, "elasticsearch": {"loaded": 0, "failed": 0}}

        # Process based on mode
        if request.mode == PipelineMode.ORIGINAL:
            await _load_original_parquet(df, request, strategy, results)
        else:
            await _load_enrich_parquet(df, request, strategy, results)

        duration = (datetime.now() - start_time).total_seconds()

        return LoadParquetResponse(
            success=True,
            mode=request.mode.value,
            indexing_strategy=strategy.value,
            parquet_info=parquet_info,
            results={k: v for k, v in results.items() if k in request.targets},
            duration_seconds=duration,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("load_parquet_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


async def _load_original_parquet(
    df: pd.DataFrame,
    request: LoadParquetRequest,
    strategy: IndexingStrategy,
    results: dict[str, dict[str, int]],
) -> None:
    """Load original mode parquet to stores."""
    batch_size = request.batch_size
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # Convert to product dicts
        products = []
        for _, row in batch_df.iterrows():
            product = _row_to_product_dict(row, mode="original")
            if product:
                products.append(product)

        if not products:
            continue

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(products)} products)")

        # Load to each target
        if "postgres" in request.targets:
            try:
                context = create_stage_context(batch_size, request.mode)
                embedded_products = [EmbeddedProduct(**p) for p in products]
                stage = LoadPostgresStage(context)
                await stage.run(embedded_products)
                results["postgres"]["loaded"] += len(embedded_products)
            except Exception as e:
                logger.error("postgres_batch_failed", batch=batch_num, error=str(e))
                results["postgres"]["failed"] += len(products)

        if "qdrant" in request.targets:
            try:
                context = create_stage_context(batch_size, request.mode)
                context.job.indexing_strategy = strategy
                embedded_products = [EmbeddedProduct(**p) for p in products]
                stage = LoadQdrantStage(context, strategy=strategy)
                await stage.run(embedded_products)
                results["qdrant"]["loaded"] += len(embedded_products)
            except Exception as e:
                logger.error("qdrant_batch_failed", batch=batch_num, error=str(e))
                results["qdrant"]["failed"] += len(products)

        if "elasticsearch" in request.targets:
            try:
                context = create_stage_context(batch_size, request.mode)
                embedded_products = [EmbeddedProduct(**p) for p in products]
                stage = LoadElasticsearchStage(context)
                await stage.run(embedded_products)
                results["elasticsearch"]["loaded"] += len(embedded_products)
            except Exception as e:
                logger.error("elasticsearch_batch_failed", batch=batch_num, error=str(e))
                results["elasticsearch"]["failed"] += len(products)


async def _load_enrich_parquet(
    df: pd.DataFrame,
    request: LoadParquetRequest,
    strategy: IndexingStrategy,
    results: dict[str, dict[str, int]],
) -> None:
    """Load enrich mode parquet to stores."""
    batch_size = request.batch_size

    # Separate parent updates and child nodes
    if "node_type" in df.columns:
        parent_df = df[df["node_type"].isin(["parent", "parent_update"])]
        child_df = df[~df["node_type"].isin(["parent", "parent_update"])]
    else:
        parent_df = df
        child_df = pd.DataFrame()

    total_batches = (len(parent_df) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(parent_df), batch_size):
        batch_df = parent_df.iloc[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # Convert parent rows to product dicts
        products = []
        genai_fields = {}
        for _, row in batch_df.iterrows():
            product = _row_to_product_dict(row, mode="enrich")
            if product:
                products.append(product)
                # Extract GenAI fields
                asin = product["asin"]
                genai_fields[asin] = _extract_genai_fields(row)

        if not products:
            continue

        # Get child chunks for this batch
        batch_asins = [p["asin"] for p in products]
        chunks = []
        if len(child_df) > 0:
            for _, row in child_df.iterrows():
                parent_asin = row.get("parent_asin") or row.get("asin")
                if parent_asin in batch_asins:
                    chunk = _row_to_chunk_dict(row)
                    if chunk:
                        chunks.append(chunk)

        logger.info(
            f"Processing batch {batch_num}/{total_batches} "
            f"({len(products)} products, {len(chunks)} chunks)"
        )

        # Load to PostgreSQL (products only, no chunks)
        if "postgres" in request.targets:
            try:
                context = create_stage_context(batch_size, request.mode)
                embedded_products = [EmbeddedProduct(**p) for p in products]
                stage = LoadPostgresStage(context)
                await stage.run(embedded_products)
                results["postgres"]["loaded"] += len(embedded_products)
            except Exception as e:
                logger.error("postgres_batch_failed", batch=batch_num, error=str(e))
                results["postgres"]["failed"] += len(products)

        # Load to Qdrant (products + chunks)
        if "qdrant" in request.targets:
            try:
                from src.models.chunk import ProductChunk
                context = create_stage_context(batch_size, request.mode)
                context.job.indexing_strategy = strategy
                context.data["genai_fields"] = genai_fields

                embedded_products = [EmbeddedProduct(**p) for p in products]
                stage = LoadQdrantStage(context, strategy=strategy)

                if chunks:
                    chunk_objs = [ProductChunk(**c) for c in chunks]
                    result = await stage.run_with_chunks(embedded_products, chunk_objs)
                    results["qdrant"]["loaded"] += result.get("products", 0) + result.get("chunks", 0)
                else:
                    await stage.run(embedded_products)
                    results["qdrant"]["loaded"] += len(embedded_products)
            except Exception as e:
                logger.error("qdrant_batch_failed", batch=batch_num, error=str(e))
                results["qdrant"]["failed"] += len(products) + len(chunks)

        # Load to Elasticsearch (products only)
        if "elasticsearch" in request.targets:
            try:
                context = create_stage_context(batch_size, request.mode)
                embedded_products = [EmbeddedProduct(**p) for p in products]
                stage = LoadElasticsearchStage(context)
                await stage.run(embedded_products)
                results["elasticsearch"]["loaded"] += len(embedded_products)
            except Exception as e:
                logger.error("elasticsearch_batch_failed", batch=batch_num, error=str(e))
                results["elasticsearch"]["failed"] += len(products)


def _row_to_product_dict(row: pd.Series, mode: str = "original") -> dict[str, Any] | None:
    """Convert a parquet row to a product dict for loading."""
    asin = row.get("asin")
    if not asin:
        return None

    # Get title from multiple sources
    title = row.get("title")
    if pd.isna(title) or title is None:
        title = row.get("short_title")
    if pd.isna(title) or title is None:
        text = row.get("text", "")
        if text and not pd.isna(text):
            title = str(text).split("\n")[0][:200]
        else:
            title = f"Product {asin}"

    product = {
        "asin": str(asin),
        "title": str(title) if title else f"Product {asin}",
        "embedding": convert_numpy(row.get("embedding")),
        "embedding_text": convert_numpy(row.get("text") or row.get("embedding_text")),
    }

    # Add optional fields
    optional_fields = [
        "brand", "price", "list_price", "stars", "reviews_count",
        "category_level1", "category_level2", "category_level3",
        "is_best_seller", "is_amazon_choice", "prime_eligible",
        "img_url", "product_url", "bought_in_last_month",
        "short_title", "product_type",
    ]

    for field in optional_fields:
        if field in row.index and not pd.isna(row.get(field)):
            product[field] = convert_numpy(row.get(field))

    # Add GenAI fields for enrich mode
    if mode == "enrich":
        genai_fields = [
            "genAI_summary", "genAI_primary_function", "genAI_best_for",
            "genAI_use_cases", "genAI_key_capabilities", "genAI_target_audience",
            "genAI_unique_selling_points", "genAI_value_score",
        ]
        for field in genai_fields:
            if field in row.index and not pd.isna(row.get(field)):
                product[field] = convert_numpy(row.get(field))

    return clean_for_json(product)


def _row_to_chunk_dict(row: pd.Series) -> dict[str, Any] | None:
    """Convert a parquet row to a chunk dict for loading."""
    from src.models.enums import SectionType

    asin = row.get("asin")
    parent_asin = row.get("parent_asin") or asin
    section_raw = row.get("section") or row.get("node_type", "content")

    if not asin:
        return None

    # Validate section is a valid SectionType
    valid_sections = {s.value for s in SectionType}
    section = str(section_raw).lower()
    if section not in valid_sections:
        # Try to map common variations
        section_mapping = {
            "desc": "description",
            "feature": "features",
            "spec": "specs",
            "specifications": "specs",
            "review": "reviews",
            "use_case": "use_cases",
            "usecases": "use_cases",
        }
        section = section_mapping.get(section, section)

        # If still not valid, skip this chunk
        if section not in valid_sections:
            logger.debug("skipping_invalid_section", section=section_raw, asin=asin)
            return None

    chunk = {
        "chunk_id": f"{parent_asin}_{section}",
        "parent_asin": str(parent_asin),
        "section": section,
        "content": convert_numpy(row.get("text") or row.get("content") or ""),
        "embedding": convert_numpy(row.get("embedding")),
    }

    # Add optional filter fields
    filter_fields = ["title", "brand", "category_level1", "price", "stars"]
    for field in filter_fields:
        if field in row.index and not pd.isna(row.get(field)):
            chunk[field] = convert_numpy(row.get(field))

    return clean_for_json(chunk)


def _extract_genai_fields(row: pd.Series) -> dict[str, Any]:
    """Extract GenAI fields from a parquet row."""
    genai_fields = {}
    field_names = [
        "genAI_summary", "genAI_primary_function", "genAI_best_for",
        "genAI_use_cases", "genAI_key_capabilities", "genAI_target_audience",
        "genAI_unique_selling_points", "genAI_value_score",
    ]

    for field in field_names:
        if field in row.index and not pd.isna(row.get(field)):
            genai_fields[field] = convert_numpy(row.get(field))

    return clean_for_json(genai_fields)


# =============================================================================
# Database Reset Endpoint
# =============================================================================

@router.post("/reset", response_model=ResetResponse)
async def reset_databases(request: ResetRequest) -> ResetResponse:
    """
    Reset/clear product data from database stores.

    This endpoint clears all product data from the specified stores:
    - **postgres**: Truncates the products table (cascades to related tables)
    - **qdrant**: Deletes and optionally recreates the products collection
    - **elasticsearch**: Deletes and optionally recreates the products index

    Use this endpoint to start fresh before loading new data.

    **WARNING**: This is a destructive operation. All product data in the
    specified stores will be permanently deleted.

    Example:
    ```
    POST /stages/reset
    {
        "targets": ["postgres", "qdrant", "elasticsearch"],
        "recreate_qdrant": true,
        "vector_size": 1024,
        "recreate_elasticsearch": true
    }
    ```
    """
    import httpx
    import asyncpg

    start_time = datetime.now()
    settings = get_settings()
    results = {}

    # Reset PostgreSQL
    if "postgres" in request.targets:
        try:
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
            )
            try:
                await conn.execute("TRUNCATE TABLE products RESTART IDENTITY CASCADE;")
                count = await conn.fetchval("SELECT COUNT(*) FROM products;")
                results["postgres"] = {
                    "success": True,
                    "action": "truncated",
                    "remaining_count": count,
                }
                logger.info("postgres_reset_complete", remaining=count)
            finally:
                await conn.close()
        except Exception as e:
            logger.error("postgres_reset_failed", error=str(e))
            results["postgres"] = {"success": False, "error": str(e)}

    # Reset Qdrant
    if "qdrant" in request.targets:
        try:
            async with httpx.AsyncClient() as client:
                qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"

                # Delete collection
                delete_resp = await client.delete(
                    f"{qdrant_url}/collections/products",
                    timeout=30.0,
                )

                if request.recreate_qdrant:
                    # Recreate collection
                    create_resp = await client.put(
                        f"{qdrant_url}/collections/products",
                        json={
                            "vectors": {
                                "size": request.vector_size,
                                "distance": "Cosine",
                            }
                        },
                        timeout=30.0,
                    )
                    results["qdrant"] = {
                        "success": True,
                        "action": "deleted_and_recreated",
                        "vector_size": request.vector_size,
                    }
                else:
                    results["qdrant"] = {
                        "success": True,
                        "action": "deleted",
                    }
                logger.info("qdrant_reset_complete", recreated=request.recreate_qdrant)
        except Exception as e:
            logger.error("qdrant_reset_failed", error=str(e))
            results["qdrant"] = {"success": False, "error": str(e)}

    # Reset Elasticsearch
    if "elasticsearch" in request.targets:
        try:
            async with httpx.AsyncClient() as client:
                es_url = f"http://{settings.elasticsearch_host}:{settings.elasticsearch_port}"

                # Delete index
                delete_resp = await client.delete(
                    f"{es_url}/products",
                    timeout=30.0,
                )

                if request.recreate_elasticsearch:
                    # Recreate index
                    create_resp = await client.put(
                        f"{es_url}/products",
                        json={
                            "settings": {
                                "number_of_shards": 1,
                                "number_of_replicas": 0,
                            }
                        },
                        timeout=30.0,
                    )
                    results["elasticsearch"] = {
                        "success": True,
                        "action": "deleted_and_recreated",
                    }
                else:
                    results["elasticsearch"] = {
                        "success": True,
                        "action": "deleted",
                    }
                logger.info("elasticsearch_reset_complete", recreated=request.recreate_elasticsearch)
        except Exception as e:
            logger.error("elasticsearch_reset_failed", error=str(e))
            results["elasticsearch"] = {"success": False, "error": str(e)}

    duration = (datetime.now() - start_time).total_seconds()

    # Determine overall success
    all_success = all(
        r.get("success", False) for r in results.values()
    )

    return ResetResponse(
        success=all_success,
        results=results,
        duration_seconds=duration,
    )


# =============================================================================
# Stage Info Endpoint
# =============================================================================

@router.get("/info")
async def get_stages_info() -> dict[str, Any]:
    """
    Get information about available stages, models, and configuration options.

    Returns comprehensive documentation for all pipeline options.
    """
    settings = get_settings()

    return {
        "modes": {
            "original": {
                "description": "Basic pipeline without LLM enrichment",
                "stages": ["extract", "clean", "embed", "load/postgres", "load/qdrant", "load/elasticsearch"],
                "default_indexing_strategy": "parent_only",
            },
            "enrich": {
                "description": "Full pipeline with LLM-based GenAI field extraction",
                "stages": [
                    "extract", "clean", "download", "html_to_md", "llm_extract",
                    "embed", "load/postgres", "load/qdrant", "load/elasticsearch"
                ],
                "default_indexing_strategy": "add_child_node",
            },
        },
        "indexing_strategies": {
            "parent_only": {
                "description": "Load only parent product nodes (fastest, for original mode)",
                "use_case": "Basic product search without section-level retrieval",
            },
            "enrich_existing": {
                "description": "Update existing parent nodes, add child nodes",
                "use_case": "Incremental enrichment of existing products",
            },
            "add_child_node": {
                "description": "Load parent + all child nodes (recommended for enrich mode)",
                "use_case": "Full section-level semantic search",
            },
            "full_replace": {
                "description": "Delete existing data and load fresh (destructive)",
                "use_case": "Complete data refresh, testing",
            },
        },
        "embedding_models": {
            "bge-large": {
                "dimensions": 768,
                "description": "Default model, good balance of speed and quality",
                "is_default": True,
            },
            "mxbai-embed-large": {
                "dimensions": 1024,
                "description": "Higher quality embeddings, slightly slower",
            },
            "bge-large": {
                "dimensions": 1024,
                "description": "BGE large model, good for semantic search",
            },
            "all-minilm": {
                "dimensions": 384,
                "description": "Fastest model, smaller embeddings",
            },
        },
        "llm_models": {
            "llama3.2:3b": {
                "description": "Default model, fast with good quality",
                "is_default": True,
            },
            "llama3.2:1b": {
                "description": "Fastest model, lower quality",
            },
            "llama3.1:8b": {
                "description": "Higher quality, slower",
            },
            "qwen2.5:7b": {
                "description": "Good for structured JSON output",
            },
            "mistral:7b": {
                "description": "Balanced speed and quality",
            },
        },
        "current_defaults": {
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens,
        },
        "file_formats": {
            "csv": {
                "description": "Standard CSV files with product data",
                "use_case": "Raw product data ingestion",
                "supported_stages": ["extract"],
            },
            "parquet": {
                "description": "Apache Parquet files, efficient for large datasets with embeddings",
                "use_case": "Pre-embedded data, checkpoint/resume, large batch processing",
                "supported_stages": ["extract", "load/parquet"],
            },
        },
        "stages": {
            "extract": {
                "description": "Extract products from CSV or Parquet file",
                "input": "CSV or Parquet file path",
                "output": "Raw products",
                "options": ["csv_path", "parquet_path", "file_type", "limit", "offset", "batch_size"],
            },
            "clean": {
                "description": "Clean and normalize product data",
                "input": "Raw products",
                "output": "Cleaned products (+ chunks for enrich mode)",
                "options": ["mode", "build_chunks", "batch_size"],
            },
            "download": {
                "description": "Download/generate HTML content (enrich mode only)",
                "input": "Cleaned products",
                "output": "Products + HTML content",
                "options": ["mock_mode", "batch_size"],
            },
            "html_to_md": {
                "description": "Convert HTML to markdown (enrich mode only)",
                "input": "Products + HTML content",
                "output": "Products + markdown content",
                "options": ["batch_size"],
            },
            "llm_extract": {
                "description": "Extract GenAI fields using LLM (enrich mode only)",
                "input": "Products + markdown content",
                "output": "Products + GenAI fields",
                "options": ["llm_model", "llm_temperature", "llm_max_tokens", "batch_size"],
            },
            "embed": {
                "description": "Generate vector embeddings",
                "input": "Cleaned products",
                "output": "Embedded products",
                "options": ["embedding_model", "batch_size"],
            },
            "load/postgres": {
                "description": "Load products to PostgreSQL",
                "input": "Embedded products",
                "output": "Load result",
                "options": ["mode", "batch_size"],
            },
            "load/qdrant": {
                "description": "Load products to Qdrant vector store",
                "input": "Embedded products (+ GenAI fields, chunks)",
                "output": "Load result",
                "options": ["mode", "indexing_strategy", "genai_fields", "chunks", "batch_size"],
            },
            "load/elasticsearch": {
                "description": "Load products to Elasticsearch",
                "input": "Embedded products",
                "output": "Load result",
                "options": ["mode", "batch_size"],
            },
            "load/parquet": {
                "description": "Load pre-embedded parquet files directly to stores",
                "input": "Parquet file with embeddings",
                "output": "Load result with per-store counts",
                "options": ["parquet_path", "mode", "indexing_strategy", "targets", "batch_size", "limit", "offset"],
                "notes": [
                    "Supports both original and enrich modes",
                    "Auto-detects embedding dimensions from data",
                    "Handles parent and child nodes for enrich mode",
                    "Can target specific stores (postgres, qdrant, elasticsearch)",
                ],
            },
            "reset": {
                "description": "Reset/clear product data from database stores",
                "input": "Target stores and options",
                "output": "Reset result per store",
                "options": ["targets", "recreate_qdrant", "vector_size", "recreate_elasticsearch"],
                "notes": [
                    "Destructive operation - deletes all product data",
                    "Can target specific stores (postgres, qdrant, elasticsearch)",
                    "Optionally recreates Qdrant collection and Elasticsearch index",
                    "PostgreSQL truncates products table with cascade",
                ],
            },
        },
    }
