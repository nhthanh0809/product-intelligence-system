# Product Intelligence System

A multi-agent AI system for intelligent product discovery, comparison, analysis, and recommendations. Built on Amazon product data with a 3-tier storage architecture (Qdrant, Elasticsearch, PostgreSQL).

Dataset link: https://www.kaggle.com/datasets/asaniczka/amazon-canada-products-2023-2-1m-products

## Quick Links

| Document | Description |
|----------|-------------|
| [Proposal](docs/PROPOSAL.md) | User scenarios, requirements, objectives |
| [Design](docs/DESIGN.md) | Architecture and tech stack by microservice |
| [Database Schemas](docs/DATABASE_SCHEMAS.md) | Vector, ES, SQL schemas with purposes |
| [Architecture Flow](docs/ARCHITECTURE.md) | End-to-end query flow diagrams |

---

## Introduction

The Product Intelligence System enables natural language queries for product search, comparison, analysis, and recommendations. It leverages:

- **Multi-Agent Architecture**: 9 specialized agents (Intent, Search, Compare, Analyze, Price, Trend, Recommend, Attribute, Synthesis)
- **3-Tier Storage**: Qdrant (semantic search), Elasticsearch (keyword search), PostgreSQL (source of truth)
- **Hybrid Search**: Combines semantic and keyword search with MRR 0.9126 performance
- **30 User Scenarios**: Across 6 categories (Discovery, Comparison, Analysis, Price, Trend, Recommendation)

### Key Capabilities

| Category | Examples |
|----------|----------|
| **Discovery** | "Find wireless headphones under $100" |
| **Comparison** | "Compare Sony WH-1000XM5 vs Bose QC45" |
| **Analysis** | "What do people say about the battery life?" |
| **Price** | "Is $299 a good price for AirPods Pro?" |
| **Trends** | "Trending products in smart home" |
| **Recommendations** | "Accessories for Canon EOS R6" |

---

## Documentation

### 1. Proposal Document

**File:** [PROPOSAL.md](docs/PROPOSAL.md)

Contains:
- Executive summary and project overview
- 30 user scenarios across 6 categories
- Scenario frequency × depth matrix
- Functional and non-functional requirements
- Success criteria and project scope

### 2. Design Document

**File:** [DESIGN.md](docs/DESIGN.md)

Contains:
- High-level architecture diagrams
- Multi-agent architecture with agent definitions
- Microservice designs (multi-agents, data-pipeline, vector-store, frontend)
- Technology stack by component
- Data flow diagrams
- Key design decisions and rationale

### 3. Database Schemas

**File:** [DATABASE_SCHEMAS.md](docs/DATABASE_SCHEMAS.md)

Contains:
- Qdrant collection schemas (parent + 5 child node types)
- Elasticsearch index mappings with analyzers
- PostgreSQL table schemas (products, brands, categories, reviews, price_history)
- Storage decision matrix
- Query-to-storage routing rules

### 4. Architecture Flow

**File:** [ARCHITECTURE.md](docs/ARCHITECTURE.md)

Contains:
- End-to-end query processing flow (10 stages)
- Agent routing and execution
- Database access patterns
- Search strategy selection

---

## User Manual

### Download the dataset

1. Download dataset from https://www.kaggle.com/datasets/asaniczka/amazon-canada-products-2023-2-1m-products
2. Unzip the csv data into ./data/archive/amz_ca_total_products_data_processed.csv

### Setting Up the Environment

#### Prerequisites

- Docker 27+ with Docker Compose 2.31+
- NVIDIA GPU with CUDA support (for Ollama)
- 32GB+ RAM recommended
- 100GB+ disk space

#### 1. Copy Configure

```bash

# Copy environment template
cp env.example .env

# Edit configuration as needed
nano .env
```

#### 2. Start Infrastructure Services

```bash
# Start the system
docker compose up --build

# Pull and run required Ollama models
docker exec pis-ollama ollama run bge-large:latest
docker exec pis-ollama ollama run llama3.2:3b
docker exec pis-ollama ollama run qllama/bge-reranker-v2-m3:latest
docker exec pis-ollama ollama run llama3.1:8b ### if need config Agents to use different LLM
```

**Required Ollama Models:**

| Model | Size | Purpose |
|-------|------|---------|
| `bge-large:latest` | 759 MB | Embedding model |
| `llama3.1:8b` | 5.2 GB | LLM for complex tasks |
| `llama3.2:3b` | 2.5 GB | LLM for general tasks |
| `qllama/bge-reranker-v2-m3:latest` | 619 MB | Reranker model |

#### 3. Verify Services

```bash
# Check Qdrant
curl http://localhost:6333/dashboard

# Check Elasticsearch
curl http://localhost:9200

# Check Ollama
curl http://localhost:11434/api/tags

# Check PostgreSQL
docker compose exec postgres psql -U postgres -c "\l"
```

### Running the Data Pipeline

#### I. Run with pre-processed CSV files with 10K product data

##### Step 1: import CSV file in original mode and "parent_only" indexing strategy

1. Go to http://localhost:8501/Data_Pipeline
2. Clean DB (if need) by go to Clean Databases in Pipeline Actions then click Clean to clean product data (SQL, Qdrant, ES)
3. Go to Upload & Process CSV, choose "Original" in Pipeline Mode, choose "parent_only" in Indexing strategy
4. Upload original CSV file ./data/cleaned/mvp_10000_original_cleaned.csv
5. Click Run pipeline. The system will run embedding input data then import product data into parent nodes in Qdrant, SQL and ES.
6. Waiting to finish the processing.

##### Step 2: import CSV file in enrich mode and "enrich_existing" indexing strategy
After import original CSV with parent_only indexing strategy
1. Go to http://localhost:8501/Data_Pipeline
2. Go to Upload & Process CSV, choose "enrich" in Pipeline Mode, choose "enrich_existing" in Indexing strategy 
3. Click Run pipeline. The system will run embedding input data then import enriched product data into child nodes in Qdrant, SQL and ES.


#### II. Run data ingestion pipeline from scratch

This part will show you how to run data ingestion from downloaded 2.1M+ Amazon products CSV file

##### Data ingestion pipeline from scratch overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL MODE (basic product info):                                         │
│  01_extract → 03_clean → [upload to frontend]                               │
│                                                                              │
│  ENRICH MODE (full enrichment with LLM):                                    │
│  01_extract → 02a_download → 02b_markdown → 02c_llm → 03_clean → [upload]   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Mode | Steps | Use Case |
|------|-------|----------|
| `original` | 01 → 03 | Basic product records, no LLM required |
| `enrich` | 01 → 02a → 02b → 02c → 03 | Full enrichment with GenAI fields |

#### 1. Environment Setup for Data Ingestion Scripts

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install required packages
pip install pandas click structlog tqdm httpx pyyaml pydantic pydantic-settings \
            playwright beautifulsoup4 lxml markdownify numpy

# Install Playwright browser (required for HTML scraping)
playwright install --with-deps chromium
```

#### 2. Configure Pipeline

Edit `scripts/pipeline_config.yaml`:

```yaml
global:
  product_count: 50        # Number of products to process
  mode: enrich             # 'original' or 'enrich'
  source_csv: archive/amz_ca_total_products_data_processed.csv

scripts:
  02c_extract_with_llm:
    model: qwen3:8b
    ollama_url: http://localhost:11434
```

#### 3. Run Pipeline Scripts
```bash
cd scripts/data-ingestion
```

##### Option 1: Run with Original mode (skip scraping)
1. Edit pipeline_config.yaml: set mode: original
2. Run below scripts
```bash
python3 01_extract_mvp.py
python3 03_clean_data.py
```

##### Option 2: Run with Enrich mode (full pipeline)
1. Edit pipeline_config.yaml: set mode: enrich
2. Run below scripts
```bash
python3 01_extract_mvp.py           # Extract from source CSV
python3 02a_download_html.py        # Download HTML pages
python3 02b_html_to_markdown.py     # Convert to Markdown
python3 02c_extract_with_llm.py     # Extract with LLM
python3 03_clean_data.py            # Clean and normalize
```
#### 4. Script Reference

| Script | Description | Output |
|--------|-------------|--------|
| `01_extract_mvp.py` | Extract products from source CSV | `data/raw/mvp_{count}_products.csv` |
| `02a_download_html.py` | Download HTML from product URLs | `data/scraped/html/*.html` |
| `02b_html_to_markdown.py` | Convert HTML to Markdown | `data/scraped/markdown/*.md` |
| `02c_extract_with_llm.py` | Extract with LLM + GenAI enrichment | `data/scraped/mvp_{count}_{mode}_extracted.csv` |
| `03_clean_data.py` | Clean and normalize data | `data/cleaned/mvp_{count}_{mode}_cleaned.csv` |

#### 5. Batch Processing (Large Datasets)

```bash
# Download HTML in batches
python3 02a_download_html.py --start 1 --end 1000
python3 02a_download_html.py --start 1001 --end 2000

# Extract with LLM in batches
python3 02c_extract_with_llm.py --start 1 --end 500
python3 02c_extract_with_llm.py --start 501 --end 1000
```

#### 6. Upload to Frontend

Back to Part I. Run with pre-processed CSV files 


#### Data Directory Structure

```
data/
├── archive/                              # Source data
│   └── amz_ca_total_products_data_processed.csv
├── raw/                                  # 01_extract output
│   └── mvp_{count}_products.csv
├── scraped/                              # 02a, 02b, 02c output
│   ├── html/*.html
│   ├── markdown/*.md
│   └── mvp_{count}_{mode}_extracted.csv
├── cleaned/                              # 03_clean output (UPLOAD THIS)
│   └── mvp_{count}_{mode}_cleaned.csv
└── metrics/                              # Pipeline metrics
    └── *.json
```

### Data Pipeline Service API

The Data Pipeline Service provides REST API endpoints for programmatic data processing.

#### Start the Service

```bash
docker compose up -d data-pipeline
```

#### API Endpoints

Base URL: `http://localhost:8005`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embedding-models` | GET | Get available embedding models from Ollama |
| `/validate` | POST | Validate CSV file for pipeline processing |
| `/upload` | POST | Upload a CSV file for processing |
| `/run` | POST | Run pipeline synchronously (Embed → Load) |
| `/run/async` | POST | Submit pipeline job to run in background |
| `/run/status/{job_id}` | GET | Get status of background pipeline job |
| `/run/jobs` | GET | List all pipeline jobs (recent 20) |
| `/clean` | POST | Clean/reset product data from databases |
| `/health` | GET | Deep health check with all dependencies |
| `/health/live` | GET | Simple liveness check |
| `/health/ready` | GET | Readiness check for critical dependencies |

#### Quick API Examples

```bash
# Get available embedding models
curl http://localhost:8005/embedding-models

# Validate CSV file before processing
curl -X POST http://localhost:8005/validate \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/app/data/cleaned_products.csv", "mode": "cleaned"}'

# Upload CSV file
curl -X POST http://localhost:8005/upload \
  -F "file=@products.csv"

# Run pipeline (synchronous) - for small datasets
curl -X POST http://localhost:8005/run \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/app/data/cleaned_products.csv",
    "limit": 1000,
    "batch_size": 100,
    "embedding_model": "bge-large",
    "mode": "original",
    "indexing_strategy": "parent_only"
  }'

# Run pipeline (async) - for large datasets
curl -X POST http://localhost:8005/run/async \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/app/data/cleaned_products.csv",
    "limit": 10000,
    "batch_size": 100,
    "mode": "enrich",
    "indexing_strategy": "enrich_existing"
  }'

# Check job status
curl http://localhost:8005/run/status/{job_id}

# Clean databases (reset product data, preserve config)
curl -X POST http://localhost:8005/clean \
  -H "Content-Type: application/json" \
  -d '{
    "targets": ["postgres", "qdrant", "elasticsearch"],
    "recreate_qdrant": true,
    "vector_size": 1024,
    "recreate_elasticsearch": true
  }'

# Health check
curl http://localhost:8005/health
```

#### Pipeline Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `original` | Basic product info only | Parent nodes, basic search |
| `enrich` | Full enrichment with GenAI fields | Parent + child nodes with AI summaries |

#### Indexing Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `parent_only` | Only parent product nodes | Basic search (original mode) |
| `enrich_existing` | Update parents, add children | Incremental enrichment |
| `full_replace` | Delete and reload all data | Complete data refresh |

#### Configuration Options

| Model Type | Options | Default |
|------------|---------|---------|
| **Embedding** | `bge-large` (1024d) | `bge-large` |
| **LLM** | `llama3.2:3b`, `qwen2.5:7b`, `mistral:7b` | `llama3.2:3b` |

#### Clean Database Notes

The `/clean` endpoint clears **only product-related data**:
- **Cleaned**: products, brands, categories, reviews, price_history, product_trends, etc.
- **Preserved**: config_settings, llm_providers, llm_models, search_strategies, agent_model_configs, reranker_configs

#### API Documentation

- Swagger UI: http://localhost:8005/docs
- ReDoc: http://localhost:8005/redoc

### Running the Multi-Agent Service

#### 1. Start Multi-Agent Service

```bash
docker compose up -d multi-agents
```

#### 2. Test API Endpoints

```bash
# Health check
curl http://localhost:8001/health

# Simple chat
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find wireless headphones under $100"}'

# Enhanced chat (v2)
curl -X POST http://localhost:8001/chat/v2 \
  -H "Content-Type: application/json" \
  -d '{"message": "Compare Sony WH-1000XM5 vs Bose QC45"}'

# Product search
curl "http://localhost:8001/search?q=wireless+headphones&limit=10"
```

## Project Structure

```
Product_intelligence_system/
├── README.md                     # This file
├── docs/
│   ├── PROPOSAL.md               # User scenarios and requirements
│   ├── DESIGN.md                 # Architecture and tech stack
│   ├── DATABASE_SCHEMAS.md       # Database schemas
│   └── ARCHITECTURE.md           # End-to-end flow diagrams
├── services/
│   ├── multi-agents/             # Multi-agent service
│   ├── data-pipeline/            # Data pipeline service
│   ├── vector-store/             # Vector store facade
│   ├── frontend/                 # Streamlit UI
│   └── ollama-service/           # Ollama connection pool
├── docker-compose.yml            # Service orchestration
└── scripts/                      # Utility scripts
```

---

## Author
Thanh, Nguyen Huy (nhthanh0809 - nhthanh0809@gmail.com)