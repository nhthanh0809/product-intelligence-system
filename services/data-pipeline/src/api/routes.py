"""Simplified API routes for data pipeline service.

Single /run endpoint: CSV -> Extract -> Clean -> Embed -> Load (Postgres, Qdrant, Elasticsearch)
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

from src.config import get_settings
from src.clients.postgres_client import get_postgres_client
from src.clients.qdrant_client import get_qdrant_client
from src.clients.elasticsearch_client import get_elasticsearch_client
from src.clients.ollama_client import get_ollama_client

logger = structlog.get_logger()
router = APIRouter()

# =============================================================================
# Background Job Storage
# =============================================================================

# In-memory storage for job status (consider Redis for production)
_jobs: dict[str, dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class RunRequest(BaseModel):
    """Request for pipeline run with pre-cleaned CSV.

    The CSV must be in cleaned format (output from clean stage) with required columns:
    - asin, title (embedding_text is auto-generated if not present)

    Pipeline will run: Embed → Load (stages 4-5 only)
    """

    csv_path: str = Field(
        ...,
        description="Path to pre-cleaned CSV file",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Number of products to process (all if not specified)",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Starting row in CSV (0-indexed)",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for processing",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model (e.g., 'bge-large', 'bge-large'). Uses default if not specified.",
    )
    mode: str = Field(
        default="original",
        description="Pipeline mode: 'original' (parent nodes only) or 'enrich' (parent + child nodes with GenAI)",
    )
    indexing_strategy: str = Field(
        default="parent_only",
        description="Indexing strategy: 'parent_only', 'enrich_existing', or 'full_replace'",
    )


class ValidateCSVRequest(BaseModel):
    """Request for CSV validation."""

    csv_path: str = Field(..., description="Path to CSV file")
    mode: str = Field(default="original", description="Pipeline mode to validate for")


class ValidateCSVResponse(BaseModel):
    """Response from CSV validation."""

    valid: bool
    mode: str
    total_rows: int
    columns_found: list[str]
    required_columns: list[str]
    missing_columns: list[str]
    optional_columns_found: list[str]
    sample_data: list[dict[str, Any]] | None = None
    errors: list[str]


class RunResponse(BaseModel):
    """Response from pipeline run."""

    success: bool
    message: str
    stats: dict[str, Any]
    duration_seconds: float


class JobStatus(BaseModel):
    """Status of a background pipeline job."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    stage: str | None = None  # current stage name
    message: str | None = None
    stats: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


class AsyncRunResponse(BaseModel):
    """Response from async pipeline run submission."""

    job_id: str
    status: str
    message: str


class CleanRequest(BaseModel):
    """Request for cleaning database data."""

    targets: list[str] = Field(
        default=["postgres", "qdrant", "elasticsearch"],
        description="Target stores to clean: postgres, qdrant, elasticsearch",
    )
    recreate_qdrant: bool = Field(
        default=True,
        description="Recreate Qdrant collection after deletion",
    )
    vector_size: int = Field(
        default=1024,
        description="Vector size for Qdrant collection (if recreating)",
    )
    recreate_elasticsearch: bool = Field(
        default=True,
        description="Recreate Elasticsearch index after deletion",
    )


class CleanResponse(BaseModel):
    """Response from clean operation."""

    success: bool
    message: str
    results: dict[str, dict[str, Any]]
    duration_seconds: float


class ServiceHealth(BaseModel):
    """Health status of a service/dependency."""

    status: str
    details: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    timestamp: str
    dependencies: dict[str, ServiceHealth]


class EmbeddingModel(BaseModel):
    """Embedding model info."""

    name: str
    size: int | None = None
    family: str | None = None
    parameter_size: str | None = None


class EmbeddingModelsResponse(BaseModel):
    """Response with available embedding models."""

    available: bool
    models: list[EmbeddingModel]
    default_model: str | None = None
    message: str | None = None


# =============================================================================
# CSV Column Definitions
# =============================================================================

# Required columns for raw/original mode (before cleaning)
RAW_REQUIRED_COLUMNS = ["asin", "title"]

# Optional columns for original mode
RAW_OPTIONAL_COLUMNS = [
    "brand", "price", "list_price", "stars", "reviews_count",
    "category_name", "category_level1", "category_level2", "category_level3",
    "is_best_seller", "is_amazon_choice", "prime_eligible",
    "product_description", "features", "product_url", "img_url",
    "bought_in_last_month", "availability",
]

# Required columns for cleaned mode (after clean stage)
# Must have asin and title, embedding_text is optional (will be auto-generated)
CLEANED_REQUIRED_COLUMNS = ["asin", "title"]

# Optional columns for cleaned mode (output from clean stage)
CLEANED_OPTIONAL_COLUMNS = [
    # From RawProduct
    "brand", "price", "list_price", "stars", "reviews_count",
    "category_level1", "category_level2", "category_level3",
    "is_best_seller", "is_amazon_choice", "prime_eligible",
    "product_description", "features", "product_url", "img_url",
    "bought_in_last_month", "availability",
    # From CleanedProduct (added by clean stage)
    "short_title", "product_type", "product_type_keywords",
    "cleaned_at", "embedding_text",
    # GenAI enriched fields
    "genAI_summary", "genAI_primary_function", "genAI_best_for",
    "genAI_use_cases", "genAI_key_capabilities",
]


# =============================================================================
# Embedding Models Endpoint
# =============================================================================

# Known embedding model patterns (by name or family)
EMBEDDING_MODEL_PATTERNS = [
    "embed", "bge", "e5", "minilm", "nomic", "snowflake", "arctic",
    "sentence-transformer", "gte", "instructor",
]

EMBEDDING_MODEL_FAMILIES = ["bert", "nomic-bert"]


@router.get("/embedding-models", response_model=EmbeddingModelsResponse)
async def get_embedding_models() -> EmbeddingModelsResponse:
    """
    Get available embedding models from the Ollama server.

    Returns a list of embedding models that can be used for the pipeline.
    Filters models by known embedding model patterns and families.
    """
    settings = get_settings()

    try:
        ollama = await get_ollama_client()
        models_data = await ollama.list_models()

        embedding_models = []
        for model in models_data.get("models", []):
            name = model.get("name", "")
            details = model.get("details", {})
            family = details.get("family", "")
            families = details.get("families", [])

            # Check if this is an embedding model
            is_embedding = False

            # Check by family
            for f in [family] + families:
                if f.lower() in EMBEDDING_MODEL_FAMILIES:
                    is_embedding = True
                    break

            # Check by name patterns
            if not is_embedding:
                name_lower = name.lower()
                for pattern in EMBEDDING_MODEL_PATTERNS:
                    if pattern in name_lower:
                        is_embedding = True
                        break

            if is_embedding:
                embedding_models.append(EmbeddingModel(
                    name=name.split(":")[0] if ":" in name else name,
                    size=model.get("size"),
                    family=family,
                    parameter_size=details.get("parameter_size"),
                ))

        if not embedding_models:
            return EmbeddingModelsResponse(
                available=False,
                models=[],
                default_model=None,
                message="No embedding models found on Ollama server. Please pull an embedding model first (e.g., 'ollama pull bge-large' or 'ollama pull bge-large').",
            )

        # Set default model (prefer bge-large or bge-large)
        default_model = None
        for preferred in ["bge-large", "bge-large", "mxbai-embed-large"]:
            for model in embedding_models:
                if preferred in model.name.lower():
                    default_model = model.name
                    break
            if default_model:
                break

        if not default_model and embedding_models:
            default_model = embedding_models[0].name

        return EmbeddingModelsResponse(
            available=True,
            models=embedding_models,
            default_model=default_model,
            message=f"Found {len(embedding_models)} embedding model(s)",
        )

    except Exception as e:
        logger.error("failed_to_get_embedding_models", error=str(e))
        return EmbeddingModelsResponse(
            available=False,
            models=[],
            default_model=None,
            message=f"Failed to connect to Ollama server: {str(e)}",
        )


# =============================================================================
# CSV Validation Endpoint
# =============================================================================

@router.post("/validate", response_model=ValidateCSVResponse)
async def validate_csv(request: ValidateCSVRequest) -> ValidateCSVResponse:
    """
    Validate a CSV file for the specified pipeline mode.

    **Modes:**
    - **cleaned**: Pre-cleaned CSV (output from clean stage). Requires: asin, title, embedding_text.
      Pipeline will only run Embed → Load stages.

    Checks:
    - File exists and is readable
    - Required columns are present based on mode
    - Reports optional columns found
    - Returns sample data for preview

    Example:
    ```
    POST /validate
    {
        "csv_path": "/app/data/cleaned_products.csv",
        "mode": "cleaned"
    }
    ```
    """
    import pandas as pd
    from pathlib import Path

    errors = []
    mode = request.mode.lower()

    # Determine required columns based on mode
    if mode == "cleaned":
        required_cols = CLEANED_REQUIRED_COLUMNS
        optional_cols = CLEANED_OPTIONAL_COLUMNS
    else:
        return ValidateCSVResponse(
            valid=False,
            mode=mode,
            total_rows=0,
            columns_found=[],
            required_columns=CLEANED_REQUIRED_COLUMNS,
            missing_columns=[],
            optional_columns_found=[],
            sample_data=None,
            errors=[f"Invalid mode: {mode}. Must be 'cleaned'"],
        )

    # Check file exists
    path = Path(request.csv_path)
    if not path.exists():
        return ValidateCSVResponse(
            valid=False,
            mode=mode,
            total_rows=0,
            columns_found=[],
            required_columns=required_cols,
            missing_columns=required_cols,
            optional_columns_found=[],
            sample_data=None,
            errors=[f"File not found: {request.csv_path}"],
        )

    try:
        # Read CSV header and sample
        df = pd.read_csv(request.csv_path, nrows=5)
        total_df = pd.read_csv(request.csv_path, usecols=[0])
        total_rows = len(total_df)

        columns_found = list(df.columns)

        # Check required columns based on mode
        missing_columns = [col for col in required_cols if col not in columns_found]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

        # Check optional columns
        optional_found = [col for col in optional_cols if col in columns_found]

        # Prepare sample data
        sample_data = df.head(3).to_dict(orient="records")

        return ValidateCSVResponse(
            valid=len(missing_columns) == 0,
            mode=mode,
            total_rows=total_rows,
            columns_found=columns_found,
            required_columns=required_cols,
            missing_columns=missing_columns,
            optional_columns_found=optional_found,
            sample_data=sample_data,
            errors=errors,
        )

    except Exception as e:
        return ValidateCSVResponse(
            valid=False,
            mode=mode,
            total_rows=0,
            columns_found=[],
            required_columns=required_cols,
            missing_columns=[],
            optional_columns_found=[],
            sample_data=None,
            errors=[f"Error reading CSV: {str(e)}"],
        )


# =============================================================================
# File Upload Endpoint
# =============================================================================

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a CSV file for processing.

    Returns the path to the uploaded file for use with /run endpoint.
    """
    import os
    from pathlib import Path

    # Validate file type
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed",
        )

    # Create upload directory
    upload_dir = Path("/app/data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = upload_dir / safe_filename

    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info("file_uploaded", filename=safe_filename, size=len(content))

        return {
            "success": True,
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": len(content),
        }

    except Exception as e:
        logger.error("file_upload_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )


# =============================================================================
# Background Job Endpoints
# =============================================================================

def update_job_status(
    job_id: str,
    status: str,
    progress: int = 0,
    stage: str | None = None,
    message: str | None = None,
    stats: dict | None = None,
    error: str | None = None,
):
    """Update job status in storage."""
    if job_id in _jobs:
        _jobs[job_id].update({
            "status": status,
            "progress": progress,
            "stage": stage,
            "message": message,
            "error": error,
        })
        if stats:
            _jobs[job_id]["stats"] = stats
        if status in ("completed", "failed"):
            _jobs[job_id]["completed_at"] = datetime.now().isoformat()
            started = datetime.fromisoformat(_jobs[job_id]["started_at"])
            _jobs[job_id]["duration_seconds"] = (datetime.now() - started).total_seconds()


async def run_pipeline_background(job_id: str, request: RunRequest):
    """Run the pipeline in the background and update job status."""
    import pandas as pd

    try:
        update_job_status(job_id, "running", 5, "initializing", "Loading CSV...")

        from src.models.job import PipelineJob
        from src.models.enums import PipelineMode, IndexingStrategy
        from src.models.product import CleanedProduct
        from src.stages.base import StageContext
        from src.stages.embed import EmbedStage
        from src.stages.load_postgres import LoadPostgresStage
        from src.stages.load_qdrant import LoadQdrantStage
        from src.stages.load_elasticsearch import LoadElasticsearchStage

        stats = {
            "mode": request.mode,
            "indexing_strategy": request.indexing_strategy,
            "loaded_from_csv": 0,
            "embedded": 0,
            "loaded_postgres": 0,
            "loaded_qdrant": 0,
            "loaded_elasticsearch": 0,
            "errors": [],
        }

        # Read CSV
        update_job_status(job_id, "running", 10, "loading_csv", "Reading CSV file...")
        df = pd.read_csv(request.csv_path)

        if request.offset > 0:
            df = df.iloc[request.offset:]
        if request.limit:
            df = df.head(request.limit)

        # Validate required columns
        required_cols = ["asin", "title"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            update_job_status(job_id, "failed", 0, None, None, stats,
                              f"Missing required columns: {', '.join(missing_cols)}")
            return

        update_job_status(job_id, "running", 15, "processing_csv", f"Processing {len(df)} rows...")

        # Convert DataFrame to CleanedProduct objects (same logic as sync endpoint)
        products = []
        has_embedding_text = "embedding_text" in df.columns

        def normalize_column_name(col: str) -> str:
            if col.startswith("genAI_"):
                return "genai_" + col[6:].lower()
            if col.startswith("genai_") and not col.startswith("genai_genai"):
                return col.lower()
            return col

        for _, row in df.iterrows():
            product_dict = row.to_dict()
            product_dict = {k: (None if pd.isna(v) else v) for k, v in product_dict.items()}

            # Normalize column names
            normalized_dict = {}
            for k, v in product_dict.items():
                norm_key = normalize_column_name(k)
                if norm_key not in normalized_dict:
                    normalized_dict[norm_key] = v
            product_dict = normalized_dict

            # Parse JSON fields
            import json
            json_list_fields = ["features", "product_type_keywords", "genai_use_cases",
                                "genai_key_capabilities", "genai_target_audience"]
            for json_field in json_list_fields:
                if json_field in product_dict and isinstance(product_dict[json_field], str):
                    try:
                        product_dict[json_field] = json.loads(product_dict[json_field])
                    except (json.JSONDecodeError, TypeError):
                        if product_dict[json_field].startswith("["):
                            product_dict[json_field] = None

            # Convert numeric fields
            if "genai_value_score" in product_dict and product_dict["genai_value_score"] is not None:
                try:
                    product_dict["genai_value_score"] = int(float(product_dict["genai_value_score"]))
                except (ValueError, TypeError):
                    product_dict["genai_value_score"] = None

            if "genai_sentiment_score" in product_dict and product_dict["genai_sentiment_score"] is not None:
                try:
                    product_dict["genai_sentiment_score"] = float(product_dict["genai_sentiment_score"])
                except (ValueError, TypeError):
                    product_dict["genai_sentiment_score"] = None

            # Auto-generate embedding_text
            if not has_embedding_text or not product_dict.get("embedding_text"):
                parts = []
                if product_dict.get("title"):
                    parts.append(f"Title: {product_dict['title']}")
                if product_dict.get("brand"):
                    parts.append(f"Brand: {product_dict['brand']}")
                if product_dict.get("category_level1"):
                    cat = product_dict["category_level1"]
                    if product_dict.get("category_level2"):
                        cat += f" > {product_dict['category_level2']}"
                    parts.append(f"Category: {cat}")
                if product_dict.get("genai_summary"):
                    parts.append(f"Summary: {str(product_dict['genai_summary'])[:500]}")
                elif product_dict.get("product_description"):
                    parts.append(f"Description: {str(product_dict['product_description'])[:500]}")
                if product_dict.get("genai_best_for"):
                    parts.append(f"Best for: {product_dict['genai_best_for']}")
                product_dict["embedding_text"] = "\n".join(parts)

            # Filter to valid fields
            valid_fields = {
                "asin", "title", "brand", "price", "list_price", "stars", "reviews_count",
                "category_level1", "category_level2", "category_level3", "category_name",
                "is_best_seller", "is_amazon_choice", "prime_eligible", "availability",
                "product_description", "features", "product_url", "img_url",
                "bought_in_last_month", "short_title", "product_type", "product_type_keywords",
                "embedding_text", "original_price", "currency", "specifications",
                "genai_summary", "genai_primary_function", "genai_best_for",
                "genai_use_cases", "genai_target_audience", "genai_key_capabilities",
                "genai_unique_selling_points", "genai_value_score", "genai_detailed_description",
                "genai_how_it_works", "genai_whats_included", "genai_materials",
                "genai_features_detailed", "genai_standout_features", "genai_technology_explained",
                "genai_feature_comparison", "genai_specs_summary", "genai_specs_comparison_ready",
                "genai_specs_limitations", "genai_sentiment_score", "genai_sentiment_label",
                "genai_common_praises", "genai_common_complaints", "genai_durability_feedback",
                "genai_value_for_money_feedback", "genai_use_case_scenarios", "genai_ideal_user_profiles",
                "genai_not_recommended_for", "genai_problems_solved", "genai_pros", "genai_cons",
                "genai_enriched_at",
            }
            filtered_dict = {k: v for k, v in product_dict.items() if k in valid_fields}

            try:
                products.append(CleanedProduct(**filtered_dict))
            except Exception as e:
                logger.warning(f"Skipping product {product_dict.get('asin')}: {str(e)}")

        stats["loaded_from_csv"] = len(products)

        if not products:
            update_job_status(job_id, "completed", 100, None, "No products found in CSV", stats)
            return

        # Create job context
        pipeline_mode = PipelineMode.ENRICH if request.mode == "enrich" else PipelineMode.ORIGINAL
        strategy_map = {
            "parent_only": IndexingStrategy.PARENT_ONLY,
            "enrich_existing": IndexingStrategy.ENRICH_EXISTING,
            "full_replace": IndexingStrategy.FULL_REPLACE,
        }
        indexing_strategy = strategy_map.get(request.indexing_strategy, IndexingStrategy.PARENT_ONLY)

        job = PipelineJob(
            mode=pipeline_mode,
            csv_path=request.csv_path,
            product_count=request.limit,
            batch_size=request.batch_size,
            indexing_strategy=indexing_strategy,
            offset=request.offset,
        )
        job.initialize_stages()
        context = StageContext(job=job, batch_size=request.batch_size)

        # Stage: Embed
        update_job_status(job_id, "running", 25, "embedding", f"Generating embeddings for {len(products)} products...")
        embed_stage = EmbedStage(context, embedding_model=request.embedding_model)
        embedded = await embed_stage.run(products)
        stats["embedded"] = len(embedded)

        if not embedded:
            update_job_status(job_id, "completed", 100, None, "No products after embedding", stats)
            return

        # Stage: Load PostgreSQL
        update_job_status(job_id, "running", 50, "loading_postgres", "Loading to PostgreSQL...")
        try:
            postgres_stage = LoadPostgresStage(context)
            postgres_count = await postgres_stage.run(embedded)
            stats["loaded_postgres"] = postgres_count
        except Exception as e:
            stats["errors"].append(f"PostgreSQL: {str(e)}")
            logger.error("postgres_load_failed", error=str(e))

        # Stage: Load Qdrant
        update_job_status(job_id, "running", 70, "loading_qdrant", "Loading to Qdrant...")
        try:
            qdrant_stage = LoadQdrantStage(context)
            qdrant_count = await qdrant_stage.run(embedded)
            stats["loaded_qdrant"] = qdrant_count
        except Exception as e:
            stats["errors"].append(f"Qdrant: {str(e)}")
            logger.error("qdrant_load_failed", error=str(e))

        # Stage: Load Elasticsearch
        update_job_status(job_id, "running", 85, "loading_elasticsearch", "Loading to Elasticsearch...")
        try:
            es_stage = LoadElasticsearchStage(context)
            es_count = await es_stage.run(embedded)
            stats["loaded_elasticsearch"] = es_count
        except Exception as e:
            stats["errors"].append(f"Elasticsearch: {str(e)}")
            logger.error("elasticsearch_load_failed", error=str(e))

        # Complete
        message = f"Pipeline completed: {stats['embedded']} products embedded and loaded"
        update_job_status(job_id, "completed", 100, None, message, stats)

    except Exception as e:
        logger.error("pipeline_background_failed", job_id=job_id, error=str(e))
        update_job_status(job_id, "failed", 0, None, None, None, str(e))


@router.post("/run/async", response_model=AsyncRunResponse)
async def run_pipeline_async(request: RunRequest, background_tasks: BackgroundTasks) -> AsyncRunResponse:
    """
    Submit a pipeline job to run in the background.

    Returns immediately with a job_id that can be used to poll for status.
    Use GET /run/status/{job_id} to check progress.
    """
    import os

    # Validate CSV file exists
    if not os.path.exists(request.csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CSV file not found: {request.csv_path}",
        )

    # Create job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "stage": None,
        "message": "Job submitted",
        "stats": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "request": request.model_dump(),
    }

    # Start background task
    background_tasks.add_task(run_pipeline_background, job_id, request)

    return AsyncRunResponse(
        job_id=job_id,
        status="pending",
        message="Pipeline job submitted. Use GET /run/status/{job_id} to check progress.",
    )


@router.get("/run/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """
    Get the status of a background pipeline job.

    Poll this endpoint to track progress of a job submitted via POST /run/async.
    """
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    job = _jobs[job_id]
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        stage=job.get("stage"),
        message=job.get("message"),
        stats=job.get("stats"),
        error=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        duration_seconds=job.get("duration_seconds"),
    )


@router.get("/run/jobs", response_model=list[JobStatus])
async def list_jobs() -> list[JobStatus]:
    """List all pipeline jobs (recent)."""
    return [
        JobStatus(
            job_id=job["job_id"],
            status=job["status"],
            progress=job["progress"],
            stage=job.get("stage"),
            message=job.get("message"),
            stats=job.get("stats"),
            error=job.get("error"),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            duration_seconds=job.get("duration_seconds"),
        )
        for job in list(_jobs.values())[-20:]  # Last 20 jobs
    ]


# =============================================================================
# Main Run Endpoint (Synchronous)
# =============================================================================

@router.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest) -> RunResponse:
    """
    Run pipeline on pre-cleaned CSV: Embed → Load (stages 4-5 only).

    **Note:** For long-running pipelines, use POST /run/async instead to avoid timeouts.

    **Input Requirements:**
    The CSV must be pre-cleaned (output from clean stage) with required columns:
    - `asin`: Product ASIN (unique identifier)
    - `title`: Product title
    - `embedding_text`: Pre-generated text for embedding (auto-generated if missing)

    **Stages:**
    1. Read cleaned products from CSV
    2. **Embed**: Generate vector embeddings using Ollama
    3. **Load**: Store to PostgreSQL, Qdrant, and Elasticsearch

    Example:
    ```
    POST /run
    {
        "csv_path": "/app/data/cleaned_products.csv",
        "limit": 1000,
        "batch_size": 100,
        "embedding_model": "bge-large"
    }
    ```
    """
    import pandas as pd

    start_time = datetime.now()

    stats = {
        "mode": request.mode,
        "indexing_strategy": request.indexing_strategy,
        "loaded_from_csv": 0,
        "embedded": 0,
        "loaded_postgres": 0,
        "loaded_qdrant": 0,
        "loaded_elasticsearch": 0,
        "errors": [],
    }

    try:
        from src.models.job import PipelineJob
        from src.models.enums import PipelineMode, IndexingStrategy
        from src.models.product import CleanedProduct
        from src.stages.base import StageContext
        from src.stages.embed import EmbedStage
        from src.stages.load_postgres import LoadPostgresStage
        from src.stages.load_qdrant import LoadQdrantStage
        from src.stages.load_elasticsearch import LoadElasticsearchStage

        logger.info(
            "pipeline_started",
            csv_path=request.csv_path,
            mode=request.mode,
            indexing_strategy=request.indexing_strategy,
            limit=request.limit,
            offset=request.offset,
        )

        # Read cleaned CSV
        logger.info("reading_cleaned_csv")
        df = pd.read_csv(request.csv_path)

        # Apply offset and limit
        if request.offset > 0:
            df = df.iloc[request.offset:]
        if request.limit:
            df = df.head(request.limit)

        # Validate required columns
        missing_cols = [col for col in CLEANED_REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns for cleaned mode: {', '.join(missing_cols)}. "
                       f"Required: {', '.join(CLEANED_REQUIRED_COLUMNS)}",
            )

        # Convert DataFrame to CleanedProduct objects
        products = []
        has_embedding_text = "embedding_text" in df.columns

        # Define column name normalization mapping (CSV column -> model field)
        def normalize_column_name(col: str) -> str:
            """Normalize column names to match model fields."""
            # Convert genAI_* variants to genai_*
            if col.startswith("genAI_"):
                # genAI_summary -> genai_summary
                return "genai_" + col[6:].lower()
            # Handle already lowercase genai_ columns
            if col.startswith("genai_") and not col.startswith("genai_genai"):
                return col.lower()
            return col

        for _, row in df.iterrows():
            product_dict = row.to_dict()
            # Handle NaN values
            product_dict = {k: (None if pd.isna(v) else v) for k, v in product_dict.items()}

            # Normalize column names (genAI_* -> genai_*)
            normalized_dict = {}
            for k, v in product_dict.items():
                norm_key = normalize_column_name(k)
                # Only keep the first occurrence of normalized key
                if norm_key not in normalized_dict:
                    normalized_dict[norm_key] = v
            product_dict = normalized_dict

            # Parse JSON fields if they're strings (list-like values)
            json_list_fields = [
                "features", "product_type_keywords", "genai_use_cases",
                "genai_key_capabilities", "genai_target_audience",
            ]
            for json_field in json_list_fields:
                if json_field in product_dict and isinstance(product_dict[json_field], str):
                    try:
                        import json
                        product_dict[json_field] = json.loads(product_dict[json_field])
                    except (json.JSONDecodeError, TypeError):
                        # If it looks like a list but can't be parsed, set to None
                        if product_dict[json_field].startswith("["):
                            product_dict[json_field] = None

            # Convert genai_value_score to int
            if "genai_value_score" in product_dict and product_dict["genai_value_score"] is not None:
                try:
                    product_dict["genai_value_score"] = int(float(product_dict["genai_value_score"]))
                except (ValueError, TypeError):
                    product_dict["genai_value_score"] = None

            # Convert genai_sentiment_score to float
            if "genai_sentiment_score" in product_dict and product_dict["genai_sentiment_score"] is not None:
                try:
                    product_dict["genai_sentiment_score"] = float(product_dict["genai_sentiment_score"])
                except (ValueError, TypeError):
                    product_dict["genai_sentiment_score"] = None

            # Auto-generate embedding_text if not present
            if not has_embedding_text or not product_dict.get("embedding_text"):
                parts = []
                if product_dict.get("title"):
                    parts.append(f"Title: {product_dict['title']}")
                if product_dict.get("brand"):
                    parts.append(f"Brand: {product_dict['brand']}")
                if product_dict.get("category_level1"):
                    cat = product_dict["category_level1"]
                    if product_dict.get("category_level2"):
                        cat += f" > {product_dict['category_level2']}"
                    parts.append(f"Category: {cat}")
                if product_dict.get("genai_summary"):
                    parts.append(f"Summary: {str(product_dict['genai_summary'])[:500]}")
                elif product_dict.get("product_description"):
                    parts.append(f"Description: {str(product_dict['product_description'])[:500]}")
                if product_dict.get("genai_best_for"):
                    parts.append(f"Best for: {product_dict['genai_best_for']}")
                product_dict["embedding_text"] = "\n".join(parts)

            # Filter to only valid CleanedProduct fields
            valid_fields = {
                # Base product fields
                "asin", "title", "brand", "price", "list_price", "stars", "reviews_count",
                "category_level1", "category_level2", "category_level3", "category_name",
                "is_best_seller", "is_amazon_choice", "prime_eligible", "availability",
                "product_description", "features", "product_url", "img_url",
                "bought_in_last_month", "short_title", "product_type", "product_type_keywords",
                "embedding_text", "original_price", "currency", "specifications",
                # GenAI enrichment fields (lowercase to match model)
                "genai_summary", "genai_primary_function", "genai_best_for",
                "genai_use_cases", "genai_target_audience", "genai_key_capabilities",
                "genai_unique_selling_points", "genai_value_score", "genai_detailed_description",
                "genai_how_it_works", "genai_whats_included", "genai_materials",
                "genai_features_detailed", "genai_standout_features", "genai_technology_explained",
                "genai_feature_comparison", "genai_specs_summary", "genai_specs_comparison_ready",
                "genai_specs_limitations", "genai_sentiment_score", "genai_sentiment_label",
                "genai_common_praises", "genai_common_complaints", "genai_durability_feedback",
                "genai_value_for_money_feedback", "genai_use_case_scenarios", "genai_ideal_user_profiles",
                "genai_not_recommended_for", "genai_problems_solved", "genai_pros", "genai_cons",
                "genai_enriched_at",
            }
            filtered_dict = {k: v for k, v in product_dict.items() if k in valid_fields}

            try:
                products.append(CleanedProduct(**filtered_dict))
            except Exception as e:
                logger.warning(f"Skipping product {product_dict.get('asin')}: {str(e)}")

        stats["loaded_from_csv"] = len(products)
        logger.info("cleaned_csv_loaded", count=len(products))

        if not products:
            return RunResponse(
                success=True,
                message="No products found in cleaned CSV",
                stats=stats,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

        # Create job context - map mode and strategy strings to enums
        pipeline_mode = PipelineMode.ENRICH if request.mode == "enrich" else PipelineMode.ORIGINAL

        strategy_map = {
            "parent_only": IndexingStrategy.PARENT_ONLY,
            "enrich_existing": IndexingStrategy.ENRICH_EXISTING,
            "full_replace": IndexingStrategy.FULL_REPLACE,
        }
        indexing_strategy = strategy_map.get(request.indexing_strategy, IndexingStrategy.PARENT_ONLY)

        job = PipelineJob(
            mode=pipeline_mode,
            csv_path=request.csv_path,
            product_count=request.limit,
            batch_size=request.batch_size,
            indexing_strategy=indexing_strategy,
            offset=request.offset,
        )
        job.initialize_stages()
        context = StageContext(job=job, batch_size=request.batch_size)

        # Stage 4: Embed
        logger.info("stage_embed_started", model=request.embedding_model)
        embed_stage = EmbedStage(context, embedding_model=request.embedding_model)
        embedded = await embed_stage.run(products)
        stats["embedded"] = len(embedded)
        logger.info("stage_embed_completed", count=len(embedded))

        if not embedded:
            return RunResponse(
                success=True,
                message="No products after embedding",
                stats=stats,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

        # Stage: Load (parallel to all stores)
        logger.info("stage_load_started")

        postgres_stage = LoadPostgresStage(context)
        qdrant_stage = LoadQdrantStage(context)
        es_stage = LoadElasticsearchStage(context)

        async def load_postgres():
            try:
                result = await postgres_stage.run(embedded)
                return len(result)
            except Exception as e:
                stats["errors"].append(f"postgres: {str(e)}")
                logger.error("load_postgres_failed", error=str(e))
                return 0

        async def load_qdrant():
            try:
                result = await qdrant_stage.run(embedded)
                return len(result)
            except Exception as e:
                stats["errors"].append(f"qdrant: {str(e)}")
                logger.error("load_qdrant_failed", error=str(e))
                return 0

        async def load_elasticsearch():
            try:
                result = await es_stage.run(embedded)
                return len(result)
            except Exception as e:
                stats["errors"].append(f"elasticsearch: {str(e)}")
                logger.error("load_elasticsearch_failed", error=str(e))
                return 0

        # Execute all loads in parallel
        results = await asyncio.gather(
            load_postgres(),
            load_qdrant(),
            load_elasticsearch(),
        )

        stats["loaded_postgres"] = results[0]
        stats["loaded_qdrant"] = results[1]
        stats["loaded_elasticsearch"] = results[2]

        logger.info(
            "stage_load_completed",
            postgres=results[0],
            qdrant=results[1],
            elasticsearch=results[2],
        )

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(
            "pipeline_completed",
            mode="cleaned",
            duration_seconds=duration,
            stats=stats,
        )

        return RunResponse(
            success=True,
            message=f"Pipeline completed: {stats['embedded']} products embedded and loaded",
            stats=stats,
            duration_seconds=duration,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CSV file not found: {request.csv_path}",
        )
    except Exception as e:
        logger.error("pipeline_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Clean Database Endpoint
# =============================================================================

@router.post("/clean", response_model=CleanResponse)
async def clean_databases(request: CleanRequest) -> CleanResponse:
    """
    Clean/reset product data from database stores.

    This endpoint clears **ONLY product-related data** from the specified stores.
    Configuration data (LLM providers, models, search strategies, etc.) is preserved.

    **PostgreSQL Tables Cleaned:**
    - products, brands, categories
    - reviews, price_history, product_trends
    - category_stats, comparison_logs, search_logs

    **PostgreSQL Tables Preserved:**
    - config_categories, config_settings
    - llm_providers, llm_models, agent_model_configs
    - search_strategies, query_strategy_mapping
    - reranker_configs, config_audit_log
    - conversation_sessions, conversation_messages

    **Other Stores:**
    - **qdrant**: Deletes and optionally recreates the products collection
    - **elasticsearch**: Deletes and optionally recreates the products index

    Use this endpoint to start fresh before loading new product data.

    **WARNING**: This is a destructive operation. All product data in the
    specified stores will be permanently deleted.

    Example:
    ```
    POST /clean
    {
        "targets": ["postgres", "qdrant", "elasticsearch"],
        "recreate_qdrant": true,
        "vector_size": 1024,
        "recreate_elasticsearch": true
    }
    ```
    """
    import httpx

    start_time = datetime.now()
    settings = get_settings()
    results = {}

    logger.info("clean_databases_started", targets=request.targets)

    # Clean PostgreSQL - Only product-related tables, preserve config tables
    if "postgres" in request.targets:
        try:
            postgres = await get_postgres_client()

            # Product-related tables to clean (in order due to FK constraints)
            product_tables = [
                "search_logs",
                "comparison_logs",
                "category_stats",
                "product_trends",
                "price_history",
                "reviews",
                "products",
                "brands",
                "categories",
            ]

            deleted_counts = {}
            total_deleted = 0

            for table in product_tables:
                try:
                    # Get count before delete
                    count = await postgres.fetchval(f"SELECT COUNT(*) FROM {table};")
                    if count and count > 0:
                        # Use DELETE instead of TRUNCATE to avoid CASCADE affecting config tables
                        await postgres.execute(f"DELETE FROM {table};")
                        # Reset sequence if exists
                        await postgres.execute(f"""
                            DO $$
                            BEGIN
                                IF EXISTS (SELECT 1 FROM pg_class WHERE relname = '{table}_id_seq') THEN
                                    PERFORM setval('{table}_id_seq', 1, false);
                                END IF;
                            END $$;
                        """)
                        deleted_counts[table] = count
                        total_deleted += count
                except Exception as table_error:
                    # Table might not exist, skip silently
                    logger.debug(f"Skipping table {table}: {str(table_error)}")

            results["postgres"] = {
                "success": True,
                "action": "deleted_product_data",
                "deleted_count": total_deleted,
                "tables_cleaned": deleted_counts,
                "preserved": "config tables (llm_providers, llm_models, config_settings, etc.)",
            }
            logger.info("postgres_cleaned", deleted=total_deleted, tables=deleted_counts)
        except Exception as e:
            logger.error("postgres_clean_failed", error=str(e))
            results["postgres"] = {"success": False, "error": str(e)}

    # Clean Qdrant
    if "qdrant" in request.targets:
        try:
            async with httpx.AsyncClient() as client:
                qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"

                # Get collection info before delete
                try:
                    info_resp = await client.get(
                        f"{qdrant_url}/collections/products",
                        timeout=10.0,
                    )
                    if info_resp.status_code == 200:
                        info = info_resp.json()
                        points_before = info.get("result", {}).get("points_count", 0)
                    else:
                        points_before = 0
                except Exception:
                    points_before = 0

                # Delete collection
                delete_resp = await client.delete(
                    f"{qdrant_url}/collections/products",
                    timeout=30.0,
                )

                if request.recreate_qdrant:
                    # Recreate collection with proper config
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
                        "deleted_points": points_before,
                        "vector_size": request.vector_size,
                    }
                else:
                    results["qdrant"] = {
                        "success": True,
                        "action": "deleted",
                        "deleted_points": points_before,
                    }
                logger.info("qdrant_cleaned", deleted=points_before, recreated=request.recreate_qdrant)
        except Exception as e:
            logger.error("qdrant_clean_failed", error=str(e))
            results["qdrant"] = {"success": False, "error": str(e)}

    # Clean Elasticsearch
    if "elasticsearch" in request.targets:
        try:
            async with httpx.AsyncClient() as client:
                es_url = f"http://{settings.elasticsearch_host}:{settings.elasticsearch_port}"

                # Get index info before delete
                try:
                    stats_resp = await client.get(
                        f"{es_url}/products/_stats",
                        timeout=10.0,
                    )
                    if stats_resp.status_code == 200:
                        stats = stats_resp.json()
                        docs_before = stats.get("_all", {}).get("primaries", {}).get("docs", {}).get("count", 0)
                    else:
                        docs_before = 0
                except Exception:
                    docs_before = 0

                # Delete index
                delete_resp = await client.delete(
                    f"{es_url}/products",
                    timeout=30.0,
                )

                if request.recreate_elasticsearch:
                    # Recreate index with basic settings
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
                        "deleted_docs": docs_before,
                    }
                else:
                    results["elasticsearch"] = {
                        "success": True,
                        "action": "deleted",
                        "deleted_docs": docs_before,
                    }
                logger.info("elasticsearch_cleaned", deleted=docs_before, recreated=request.recreate_elasticsearch)
        except Exception as e:
            logger.error("elasticsearch_clean_failed", error=str(e))
            results["elasticsearch"] = {"success": False, "error": str(e)}

    duration = (datetime.now() - start_time).total_seconds()

    # Determine overall success
    all_success = all(
        r.get("success", False) for r in results.values()
    )

    logger.info(
        "clean_databases_completed",
        success=all_success,
        duration_seconds=duration,
        results=results,
    )

    return CleanResponse(
        success=all_success,
        message="All databases cleaned successfully" if all_success else "Some databases failed to clean",
        results=results,
        duration_seconds=duration,
    )


# =============================================================================
# Health Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Deep health check of service and all dependencies."""
    dependencies = {}

    # Check PostgreSQL
    try:
        postgres = await get_postgres_client()
        pg_health = await postgres.health_check()
        dependencies["postgres"] = ServiceHealth(
            status=pg_health["status"],
            details=pg_health,
        )
    except Exception as e:
        dependencies["postgres"] = ServiceHealth(
            status="unhealthy",
            details={"error": str(e)},
        )

    # Check Qdrant
    try:
        qdrant = get_qdrant_client()
        qdrant_health = await qdrant.health_check()
        dependencies["qdrant"] = ServiceHealth(
            status=qdrant_health["status"],
            details=qdrant_health,
        )
    except Exception as e:
        dependencies["qdrant"] = ServiceHealth(
            status="unhealthy",
            details={"error": str(e)},
        )

    # Check Elasticsearch
    try:
        es = await get_elasticsearch_client()
        es_health = await es.health_check()
        dependencies["elasticsearch"] = ServiceHealth(
            status=es_health["status"],
            details=es_health,
        )
    except Exception as e:
        dependencies["elasticsearch"] = ServiceHealth(
            status="unhealthy",
            details={"error": str(e)},
        )

    # Check Ollama service
    try:
        ollama = await get_ollama_client()
        ollama_health = await ollama.health_check()
        dependencies["ollama"] = ServiceHealth(
            status=ollama_health["status"],
            details=ollama_health,
        )
    except Exception as e:
        dependencies["ollama"] = ServiceHealth(
            status="unhealthy",
            details={"error": str(e)},
        )

    # Determine overall status
    all_healthy = all(
        dep.status == "healthy" for dep in dependencies.values()
    )
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        service="data-pipeline",
        version="0.1.0",
        timestamp=datetime.now().isoformat(),
        dependencies=dependencies,
    )


@router.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Simple liveness check."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check - verifies critical dependencies."""
    try:
        postgres = await get_postgres_client()
        await postgres.fetchval("SELECT 1")

        qdrant = get_qdrant_client()
        qdrant.client  # Just verify connection

        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Not ready: {str(e)}",
        )
