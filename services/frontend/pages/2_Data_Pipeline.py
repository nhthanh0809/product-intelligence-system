"""Data Pipeline page for CSV upload and processing.

Supports two pipeline modes:
- Original: Basic pipeline (Embed → Load) with parent nodes only
- Enrich: Full pipeline with GenAI fields and child nodes

Pipeline runs stages 4-5: Embed → Load
"""

import httpx
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from config import get_settings

settings = get_settings()

st.set_page_config(
    page_title="Data Pipeline - Product Intelligence System",
    page_icon="📊",
    layout="wide",
)

# Initialize session state for tracking running jobs
if "pipeline_job_id" not in st.session_state:
    st.session_state.pipeline_job_id = None
if "pipeline_job_status" not in st.session_state:
    st.session_state.pipeline_job_status = None


# =============================================================================
# API Client
# =============================================================================

class DataPipelineAPI:
    """Client for data pipeline API."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = settings.request_timeout

    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request to data pipeline API."""
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = getattr(client, method)(url, **kwargs)
                response.raise_for_status()
                if response.status_code == 204 or not response.content:
                    return True
                return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"API Error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None

    def get(self, endpoint: str, **kwargs):
        return self._make_request("get", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs):
        return self._make_request("post", endpoint, **kwargs)

    def upload_file(self, file) -> dict | None:
        """Upload a CSV file."""
        url = f"{self.base_url}/upload"
        try:
            with httpx.Client(timeout=60.0) as client:
                files = {"file": (file.name, file.getvalue(), "text/csv")}
                response = client.post(url, files=files)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"Upload Error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            st.error(f"Upload Error: {str(e)}")
            return None

    def validate_csv(self, csv_path: str) -> dict | None:
        """Validate CSV file for cleaned mode."""
        return self.post("/validate", json={"csv_path": csv_path, "mode": "cleaned"})

    def run_pipeline(self, csv_path: str, limit: int | None = None,
                     batch_size: int = 100, embedding_model: str | None = None,
                     mode: str = "original", indexing_strategy: str = "parent_only") -> dict | None:
        """Run the pipeline synchronously (embed + load only)."""
        payload = {
            "csv_path": csv_path,
            "batch_size": batch_size,
            "mode": mode,
            "indexing_strategy": indexing_strategy,
        }
        if limit:
            payload["limit"] = limit
        if embedding_model:
            payload["embedding_model"] = embedding_model

        return self.post("/run", json=payload)

    def run_pipeline_async(self, csv_path: str, limit: int | None = None,
                           batch_size: int = 100, embedding_model: str | None = None,
                           mode: str = "original", indexing_strategy: str = "parent_only") -> dict | None:
        """Submit pipeline job to run in background."""
        payload = {
            "csv_path": csv_path,
            "batch_size": batch_size,
            "mode": mode,
            "indexing_strategy": indexing_strategy,
        }
        if limit:
            payload["limit"] = limit
        if embedding_model:
            payload["embedding_model"] = embedding_model

        return self.post("/run/async", json=payload)

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a background pipeline job."""
        return self.get(f"/run/status/{job_id}")

    def list_jobs(self) -> dict | None:
        """List recent pipeline jobs."""
        return self.get("/run/jobs")

    def clean_databases(self, targets: list[str], recreate_qdrant: bool = True,
                        vector_size: int = 1024, recreate_elasticsearch: bool = True) -> dict | None:
        """Clean database data."""
        return self.post("/clean", json={
            "targets": targets,
            "recreate_qdrant": recreate_qdrant,
            "vector_size": vector_size,
            "recreate_elasticsearch": recreate_elasticsearch,
        })

    def health_check(self) -> dict | None:
        """Check API health."""
        return self.get("/health")

    def get_embedding_models(self) -> dict | None:
        """Get available embedding models from Ollama."""
        return self.get("/embedding-models")


api = DataPipelineAPI(settings.data_pipeline_url)


# =============================================================================
# Page Header
# =============================================================================

st.title("📊 Data Pipeline")
st.markdown("""
Upload **pre-cleaned CSV** files to embed and load into the product database.

**Pipeline Stages:** Embed → Load (stages 4-5 only)
""")

# Check API availability
with st.spinner("Checking data pipeline service..."):
    health = api.health_check()
    if health is None:
        st.error(
            "Data Pipeline service is not available. "
            "Please ensure the service is running."
        )
        st.stop()
    elif health.get("status") != "healthy":
        st.warning(f"Data Pipeline service is degraded: {health.get('status')}")


# =============================================================================
# Sidebar Navigation
# =============================================================================

st.sidebar.title("Pipeline Actions")
action = st.sidebar.radio(
    "Select Action",
    [
        "Upload & Process CSV",
        "Clean Databases",
        "Service Health",
    ]
)


# =============================================================================
# CSV Format Info
# =============================================================================

# Required columns (minimal - embedding_text is auto-generated)
REQUIRED_COLUMNS = ["asin", "title"]

# Optional columns for original mode
ORIGINAL_OPTIONAL_COLUMNS = [
    "brand", "price", "list_price", "stars", "reviews_count",
    "category_level1", "category_level2", "category_level3",
    "is_best_seller", "is_amazon_choice", "prime_eligible",
    "product_description", "features", "product_url", "img_url",
    "bought_in_last_month", "availability",
]

# Additional columns for enrich mode (GenAI fields)
ENRICH_OPTIONAL_COLUMNS = ORIGINAL_OPTIONAL_COLUMNS + [
    "short_title", "product_type", "product_type_keywords",
    "genAI_summary", "genAI_primary_function", "genAI_best_for",
    "genAI_use_cases", "genAI_target_audience", "genAI_key_capabilities",
    "genAI_unique_selling_points", "genAI_value_score",
    "genAI_detailed_description", "genAI_sentiment_score",
]


# =============================================================================
# Upload & Process CSV Section
# =============================================================================

def render_upload_section():
    """Render CSV upload and processing section."""
    st.header("📁 Upload & Process Cleaned CSV")

    # Info box
    st.info("""
    **This pipeline accepts pre-cleaned CSV files** (output from the clean stage).

    The pipeline will:
    1. Read cleaned products from CSV
    2. Auto-generate `embedding_text` if not present (from title, brand, category, genAI fields)
    3. Generate vector embeddings using the selected model
    4. Load to PostgreSQL, Qdrant, and Elasticsearch
    """)

    st.divider()

    # Pipeline Mode Selection
    st.subheader("1. Pipeline Mode & Strategy")

    col1, col2 = st.columns(2)

    with col1:
        mode = st.selectbox(
            "Pipeline Mode",
            options=["original", "enrich"],
            format_func=lambda x: {
                "original": "Original - Basic product info (parent nodes only)",
                "enrich": "Enrich - With GenAI fields (parent + child nodes)",
            }.get(x, x),
            help="""
            **Original Mode**: Creates parent nodes with basic product info (asin, title, price, etc.)

            **Enrich Mode**: Creates parent nodes with GenAI fields + child nodes for detailed content
            """,
        )

    with col2:
        # Auto-select indexing strategy based on mode, but allow override
        default_strategy = "parent_only" if mode == "original" else "enrich_existing"

        indexing_strategy = st.selectbox(
            "Indexing Strategy",
            options=["parent_only", "enrich_existing", "full_replace"],
            index=["parent_only", "enrich_existing", "full_replace"].index(default_strategy),
            format_func=lambda x: {
                "parent_only": "Parent Only - Insert parent nodes (original mode)",
                "enrich_existing": "Enrich Existing - Update parents + add children",
                "full_replace": "Full Replace - Delete all and re-insert",
            }.get(x, x),
            help="""
            **Parent Only**: Insert parent nodes only (for original mode)

            **Enrich Existing**: Update existing parent nodes + insert child nodes (for enrich mode)

            **Full Replace**: Delete all existing data and re-insert (clean slate)
            """,
        )

    # Show mode-specific info
    if mode == "original":
        st.caption("📝 Original mode: Only basic product fields will be stored. GenAI fields will be ignored.")
    else:
        st.caption("📝 Enrich mode: GenAI fields will be stored. Child nodes will be created for detailed content.")

    st.divider()

    # CSV Format Requirements
    st.subheader("2. CSV Format Requirements")

    with st.expander("📋 Required & Optional Columns", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Required Columns:**")
            st.code("asin, title")
            st.markdown("""
            - `asin`: Product ASIN (unique identifier)
            - `title`: Product title

            Note: `embedding_text` is auto-generated if not present
            """)

        with col2:
            st.markdown("**Optional Columns:**")
            if mode == "original":
                st.code(", ".join(ORIGINAL_OPTIONAL_COLUMNS[:6]) + ", ...")
            else:
                st.code(", ".join(ENRICH_OPTIONAL_COLUMNS[:6]) + ", ...")
            st.markdown("""
            - `brand`, `price`, `stars`: Basic product info
            - `genAI_*`: GenAI enrichment fields (enrich mode)
            - Other product attributes...
            """)

    st.divider()

    # File Upload
    st.subheader("3. Upload Cleaned CSV File")

    uploaded_file = st.file_uploader(
        "Choose a pre-cleaned CSV file",
        type=["csv"],
        help="Upload a CSV file that was output from the clean stage",
    )

    if uploaded_file is not None:
        # Show file info
        st.success(f"File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        # Preview data
        try:
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer

            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)

            st.markdown(f"**Total rows:** {len(df)}")
            st.markdown(f"**Columns:** {len(df.columns)}")

            # Validate columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

            # Check optional columns based on mode
            optional_cols = ENRICH_OPTIONAL_COLUMNS if mode == "enrich" else ORIGINAL_OPTIONAL_COLUMNS
            found_optional = [col for col in optional_cols if col in df.columns]

            # Check for GenAI columns
            genai_cols = [col for col in df.columns if col.lower().startswith("genai")]

            if missing_cols:
                st.error(f"**Missing required columns:** {', '.join(missing_cols)}")
                st.markdown("""
                Your CSV must have these columns:
                - `asin` - Product identifier
                - `title` - Product title

                Please ensure your CSV is the output from the **clean stage**.
                """)
                return
            else:
                st.success("All required columns present!")
                if found_optional:
                    st.info(f"**Optional columns found:** {', '.join(found_optional[:5])}{'...' if len(found_optional) > 5 else ''}")

                if mode == "enrich" and genai_cols:
                    st.info(f"**GenAI columns found:** {len(genai_cols)} columns ({', '.join(genai_cols[:3])}...)")
                elif mode == "enrich" and not genai_cols:
                    st.warning("**No GenAI columns found.** Consider using 'original' mode for this CSV.")

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return

        st.divider()

        # Processing Options
        st.subheader("4. Processing Options")

        col1, col2 = st.columns(2)

        with col1:
            limit = st.number_input(
                "Limit (rows to process)",
                min_value=0,
                max_value=len(df),
                value=0,
                help="0 = process all rows",
            )
            if limit == 0:
                limit = None

        with col2:
            batch_size = st.select_slider(
                "Batch Size",
                options=[10, 25, 50, 100, 200, 500],
                value=100,
                help="Number of products per embedding batch",
            )

        # Fetch available embedding models from Ollama
        embedding_models_response = api.get_embedding_models()
        embedding_models_available = False
        embedding_model_options = []
        default_model = None
        embedding_error_message = None

        if embedding_models_response:
            embedding_models_available = embedding_models_response.get("available", False)
            if embedding_models_available:
                models = embedding_models_response.get("models", [])
                embedding_model_options = [m.get("name") for m in models]
                default_model = embedding_models_response.get("default_model")
            else:
                embedding_error_message = embedding_models_response.get("message", "No embedding models available")
        else:
            embedding_error_message = "Failed to fetch embedding models from server"

        if embedding_models_available and embedding_model_options:
            # Show dropdown with available models
            default_index = 0
            if default_model and default_model in embedding_model_options:
                default_index = embedding_model_options.index(default_model)

            embedding_model = st.selectbox(
                "Embedding Model",
                options=embedding_model_options,
                index=default_index,
                help="Select an embedding model from Ollama server",
            )
            st.caption(f"Found {len(embedding_model_options)} embedding model(s) on Ollama server")
        else:
            # Show error and disabled dropdown
            st.selectbox(
                "Embedding Model",
                options=["No models available"],
                disabled=True,
                help="No embedding models found on Ollama server",
            )
            st.error(f"**No embedding models available:** {embedding_error_message}")
            st.markdown("""
            To use the pipeline, you need to pull an embedding model on the Ollama server:
            ```bash
            ollama pull bge-large
            # or
            ollama pull bge-large
            ```
            """)
            embedding_model = None

        st.divider()

        # Run Pipeline
        st.subheader("5. Run Pipeline")

        # Check if there's a running job
        if st.session_state.pipeline_job_id:
            job_status = api.get_job_status(st.session_state.pipeline_job_id)

            if job_status:
                status_val = job_status.get("status", "unknown")
                progress = job_status.get("progress", 0)
                stage = job_status.get("stage", "")
                message = job_status.get("message", "")

                # Show progress
                st.info(f"**Pipeline Running** - Job ID: `{st.session_state.pipeline_job_id}`")

                progress_bar = st.progress(progress / 100)
                stage_display = stage.replace("_", " ").title() if stage else "Processing"
                st.caption(f"Stage: {stage_display} - {message}")

                if status_val == "completed":
                    st.session_state.pipeline_job_id = None
                    st.session_state.pipeline_job_status = job_status
                    st.success("Pipeline completed successfully!")
                    st.rerun()

                elif status_val == "failed":
                    st.session_state.pipeline_job_id = None
                    st.session_state.pipeline_job_status = job_status
                    st.error(f"Pipeline failed: {job_status.get('error', 'Unknown error')}")
                    st.rerun()

                else:
                    # Still running - show refresh button
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("🔄 Refresh Status"):
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel (Clear Status)"):
                            st.session_state.pipeline_job_id = None
                            st.rerun()

                    # Auto-refresh hint
                    st.caption("Click 'Refresh Status' to update progress, or navigate away and come back later.")
                    return  # Don't show the rest of the form while job is running

            else:
                # Job not found, clear state
                st.session_state.pipeline_job_id = None
                st.warning("Previous job not found. It may have completed or been cleared.")

        # Helper function to get count from stats (handles both array and int formats)
        def get_count(value):
            if isinstance(value, list):
                return len(value)
            elif isinstance(value, int):
                return value
            return 0

        # Show last completed job results if available
        if st.session_state.pipeline_job_status:
            last_status = st.session_state.pipeline_job_status
            if last_status.get("status") == "completed":
                with st.expander("📊 Last Pipeline Results", expanded=True):
                    stats = last_status.get("stats", {})
                    duration = last_status.get("duration_seconds", 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mode", stats.get("mode", "N/A"))
                    with col2:
                        st.metric("Loaded from CSV", get_count(stats.get("loaded_from_csv", 0)))
                    with col3:
                        st.metric("Embedded", get_count(stats.get("embedded", 0)))
                    with col4:
                        st.metric("Duration", f"{duration:.1f}s" if duration else "N/A")

                    st.markdown("**Load Results:**")
                    load_col1, load_col2, load_col3 = st.columns(3)
                    with load_col1:
                        st.metric("PostgreSQL", get_count(stats.get("loaded_postgres", 0)))
                    with load_col2:
                        st.metric("Qdrant", get_count(stats.get("loaded_qdrant", 0)))
                    with load_col3:
                        st.metric("Elasticsearch", get_count(stats.get("loaded_elasticsearch", 0)))

                    if stats.get("errors"):
                        st.warning(f"Errors: {', '.join(stats['errors'])}")

                    if st.button("Clear Results"):
                        st.session_state.pipeline_job_status = None
                        st.rerun()

            st.divider()

        col1, col2 = st.columns([1, 3])

        # Disable run button if no embedding models available
        can_run = embedding_models_available and embedding_model is not None

        with col1:
            run_button = st.button(
                "🚀 Run Pipeline",
                type="primary",
                use_container_width=True,
                disabled=not can_run,
            )

        with col2:
            st.markdown(f"""
            **Summary:**
            - Mode: `{mode}`
            - Strategy: `{indexing_strategy}`
            - Rows: `{limit if limit else f'All ({len(df)})'}`
            - Batch size: `{batch_size}`
            - Embedding model: `{embedding_model or 'N/A'}`
            """)

        if not can_run:
            st.warning("Cannot run pipeline: No embedding model available. Please pull an embedding model on the Ollama server first.")

        if run_button and can_run:
            # First upload the file
            with st.spinner("Uploading file..."):
                upload_result = api.upload_file(uploaded_file)

            if upload_result and upload_result.get("success"):
                csv_path = upload_result.get("path")
                st.info(f"File uploaded to: `{csv_path}`")

                # Submit async pipeline job
                with st.spinner("Submitting pipeline job..."):
                    result = api.run_pipeline_async(
                        csv_path=csv_path,
                        limit=limit,
                        batch_size=batch_size,
                        embedding_model=embedding_model,
                        mode=mode,
                        indexing_strategy=indexing_strategy,
                    )

                if result and result.get("job_id"):
                    st.session_state.pipeline_job_id = result.get("job_id")
                    st.session_state.pipeline_job_status = None
                    st.success(f"Pipeline job submitted! Job ID: `{result.get('job_id')}`")
                    st.info("The pipeline is now running in the background. You can navigate away and come back to check progress.")
                    st.rerun()
                else:
                    st.error(f"Failed to submit pipeline job: {result}")
            else:
                st.error("Failed to upload file")


# =============================================================================
# Clean Databases Section
# =============================================================================

def render_clean_section():
    """Render database cleaning section."""
    st.header("🧹 Clean Product Databases")

    st.info("""
    **Note:** This only cleans **product data**. Configuration data (LLM providers, models,
    search strategies, etc.) is preserved.
    """)

    st.warning(
        "**Warning:** This will permanently delete all product data from the selected databases. "
        "This action cannot be undone."
    )

    # Target selection
    st.subheader("Select Targets")

    col1, col2, col3 = st.columns(3)

    with col1:
        clean_postgres = st.checkbox("PostgreSQL", value=True)
    with col2:
        clean_qdrant = st.checkbox("Qdrant (Vector)", value=True)
    with col3:
        clean_elasticsearch = st.checkbox("Elasticsearch", value=True)

    targets = []
    if clean_postgres:
        targets.append("postgres")
    if clean_qdrant:
        targets.append("qdrant")
    if clean_elasticsearch:
        targets.append("elasticsearch")

    st.divider()

    # Options
    st.subheader("Options")

    col1, col2 = st.columns(2)

    with col1:
        recreate_qdrant = st.checkbox(
            "Recreate Qdrant collection",
            value=True,
            help="Create a new empty collection after deletion",
        )
        if recreate_qdrant:
            vector_size = st.number_input(
                "Vector size",
                min_value=128,
                max_value=4096,
                value=1024,
                help="Dimension size for vectors",
            )
        else:
            vector_size = 1024

    with col2:
        recreate_elasticsearch = st.checkbox(
            "Recreate Elasticsearch index",
            value=True,
            help="Create a new empty index after deletion",
        )

    st.divider()

    # Confirmation
    st.subheader("Confirm")

    confirm = st.checkbox(
        "I understand this will permanently delete all product data",
        value=False,
    )

    if st.button("🗑️ Clean Databases", type="primary", disabled=not confirm or not targets):
        with st.spinner("Cleaning databases..."):
            result = api.clean_databases(
                targets=targets,
                recreate_qdrant=recreate_qdrant,
                vector_size=vector_size,
                recreate_elasticsearch=recreate_elasticsearch,
            )

        if result:
            if result.get("success"):
                st.success("Databases cleaned successfully!")

                # Show results
                results = result.get("results", {})

                for target, info in results.items():
                    if info.get("success"):
                        deleted = info.get('deleted_count', info.get('deleted_points', info.get('deleted_docs', 0)))
                        st.info(f"**{target.upper()}:** {info.get('action', 'cleaned')} - Deleted: {deleted}")
                    else:
                        st.error(f"**{target.upper()}:** Failed - {info.get('error', 'Unknown error')}")

                st.metric("Duration", f"{result.get('duration_seconds', 0):.2f}s")
            else:
                st.error(f"Clean failed: {result.get('message', 'Unknown error')}")


# =============================================================================
# Service Health Section
# =============================================================================

def render_health_section():
    """Render service health section."""
    st.header("🏥 Service Health")

    if st.button("🔄 Refresh"):
        st.rerun()

    health = api.health_check()

    if health:
        # Overall status
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success(f"Overall Status: **{status.upper()}**")
        elif status == "degraded":
            st.warning(f"Overall Status: **{status.upper()}**")
        else:
            st.error(f"Overall Status: **{status.upper()}**")

        st.markdown(f"**Service:** {health.get('service', 'N/A')}")
        st.markdown(f"**Version:** {health.get('version', 'N/A')}")
        st.markdown(f"**Timestamp:** {health.get('timestamp', 'N/A')}")

        st.divider()

        # Dependencies
        st.subheader("Dependencies")

        deps = health.get("dependencies", {})

        for name, info in deps.items():
            status = info.get("status", "unknown")
            details = info.get("details", {})

            with st.expander(f"{'✅' if status == 'healthy' else '❌'} {name.upper()}", expanded=status != "healthy"):
                st.markdown(f"**Status:** {status}")

                if details:
                    for key, value in details.items():
                        if key != "status":
                            st.markdown(f"**{key}:** {value}")
    else:
        st.error("Failed to fetch health status")


# =============================================================================
# Main Router
# =============================================================================

if action == "Upload & Process CSV":
    render_upload_section()
elif action == "Clean Databases":
    render_clean_section()
elif action == "Service Health":
    render_health_section()
