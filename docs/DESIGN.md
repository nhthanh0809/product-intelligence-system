# Product Intelligence System - Design Document

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION MICROSERVICES ARCHITECTURE                     │
│                         (Millions of Products Scale)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                          │
│   │    Users     │                                                          │
│   │  (Browser)   │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                    Frontend Service :8501                        │      │
│   │  ┌─────────────────────────────────────────────────────────────┐ │      │
│   │  │ Streamlit Chatbot UI (Chat, Settings, Data pipeline)        │ │      │
│   │  └─────────────────────────────────────────────────────────────┘ │      │
│   └──────────────────────────┬───────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                    Multi-Agent Service :8001                      │     │
│   │  ┌─────────────────────────────────────────────────────────────┐ │      │
│   │  │ Agent Architecture                                          │ │      │
│   │  │   ├── IntentAgent (query classification)                    │ │      │
│   │  │   ├── SupervisorAgent (execution planning)                  │ │      │
│   │  │   ├── SearchAgent (hybrid search MRR 0.9126)                │ │      │
│   │  │   ├── CompareAgent (product comparison)                     │ │      │
│   │  │   ├── AnalysisAgent (review mining, sentiment)              │ │      │
│   │  │   ├── PriceAgent (price intelligence)                       │ │      │
│   │  │   ├── TrendAgent (market analysis)                          │ │      │
│   │  │   ├── RecommendAgent (recommendations)                      │ │      │
│   │  │   └── SynthesisAgent (response generation)                  │ │      │
│   │  └─────────────────────────────────────────────────────────────┘ │      │
│   └──────────────────────────┬───────────────────────────────────────┘      │
│                              │                                               │
│          ┌───────────────────┼───────────────────┐                          │
│          │                   │                   │                          │
│          ▼                   ▼                   ▼                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │ Vector-Store │    │    Redis     │    │  PostgreSQL  │                  │
│   │    :8002     │    │    :6379     │    │    :5432     │                  │
│   │              │    │  (Cache +    │    │  (Source of  │                  │
│   │  Qdrant +    │    │   Queue)     │    │    Truth)    │                  │
│   │  ES Facade   │    │              │    │              │                  │
│   └──────┬───────┘    └──────────────┘    └──────────────┘                  │
│          │                                                                   │
│   ┌──────┴──────────────────────────────────────────────┐                   │
│   │                                                      │                   │
│   ▼                                                      ▼                   │
│ ┌──────────────┐                              ┌──────────────┐              │
│ │    Qdrant    │                              │Elasticsearch │              │
│ │    :6333     │                              │    :9200     │              │
│ │              │                              │              │              │
│ │ 2.1M vectors │                              │ 2.1M docs    │              │
│ │ + 10.5M child│                              │ Full-text    │              │
│ │   nodes      │                              │ + Aggregates │              │
│ └──────────────┘                              └──────────────┘              │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                    Data Pipeline Service :8005                    │      │
│   │  ┌─────────────────────────────────────────────────────────────┐ │      │
│   │  │ Pipeline Orchestrator                                       │ │      │
│   │  │   ├── Batch Processor (10K products/batch)                  │ │      │
│   │  │   ├── Embedding Generator (connection pool)                 │ │      │
│   │  │   ├── Multi-Store Loader (parallel writes)                  │ │      │
│   │  │   └── Progress Tracker (Redis-backed)                       │ │      │
│   │  └─────────────────────────────────────────────────────────────┘ │      │
│   └──────────────────────────┬───────────────────────────────────────┘      │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────┐    ┌──────────────┐                                      │
│   │Ollama-Service│    │    Ollama    │                                      │
│   │    :8010     │───▶│   :11434     │                                      │
│   │  (Pool: 10)  │    │  (GPU: all)  │                                      │
│   └──────────────┘    └──────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Multi-Agent Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Multi-Agent Architecture                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         INTENT AGENT                                  │  │
│  │  • 14 intent types (6 product + 6 general + 2 complex)               │  │
│  │  • Entity extraction (products, brands, categories, constraints)      │  │
│  │  • Compound query detection                                           │  │
│  │  • Rule-based + LLM fallback classification                          │  │
│  └─────────────────────────────┬────────────────────────────────────────┘  │
│                                │                                            │
│            ┌───────────────────┴───────────────────┐                       │
│            │                                       │                        │
│            ▼                                       ▼                        │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │   GENERAL AGENT     │               │  SUPERVISOR AGENT   │             │
│  │   (Non-product)     │               │  (Product queries)  │             │
│  │                     │               │                     │             │
│  │  • GREETING         │               │  • Execution plan   │             │
│  │  • FAREWELL         │               │  • Agent selection  │             │
│  │  • HELP             │               │  • Dependencies     │             │
│  │  • SMALL_TALK       │               │  • Parallel exec    │             │
│  │  • OFF_TOPIC        │               └──────────┬──────────┘             │
│  └─────────────────────┘                          │                        │
│                                    ┌──────────────┴──────────────┐         │
│                                    ▼              ▼              ▼         │
│                            ┌─────────────┐ ┌─────────────┐ ┌───────────┐   │
│                            │   Search    │ │   Compare   │ │ Recommend │   │
│                            │   Agent     │ │   Agent     │ │   Agent   │   │
│                            └─────────────┘ └─────────────┘ └───────────┘   │
│                            ┌─────────────┐ ┌─────────────┐ ┌───────────┐   │
│                            │  Attribute  │ │  Analysis   │ │   Trend   │   │
│                            │   Agent     │ │   Agent     │ │   Agent   │   │
│                            └─────────────┘ └─────────────┘ └───────────┘   │
│                            ┌─────────────┐                                  │
│                            │    Price    │                                  │
│                            │    Agent    │                                  │
│                            └─────────────┘                                  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        SYNTHESIS AGENT                                │  │
│  │  • Natural language response generation                               │  │
│  │  • Format selection (TEXT, BULLET, TABLE, CARDS, COMPARISON)         │  │
│  │  • Citation and source attribution                                    │  │
│  │  • Follow-up suggestions                                              │  │
│  │  • Confidence scoring                                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│                               Response                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Microservice Designs

### 2.1 Multi-Agent Service

**Location:** `services/multi-agents/`
**Port:** 8001

#### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | FastAPI | 0.115+ | Async API with OpenAPI |
| Agent Orchestration | LangGraph | 0.2+ | Stateful agent workflows |
| LLM Integration | LangChain | 0.3+ | Tool abstractions |
| HTTP Client | httpx | 0.28+ | Async service calls |

#### Agent Definitions

| Agent | File | Scenarios | Primary Storage | Target Latency |
|-------|------|-----------|-----------------|----------------|
| IntentAgent | `intent_agent.py` | All | - | <50ms |
| GeneralAgent | `general_agent.py` | GREETING, HELP, etc. | - | <100ms |
| SupervisorAgent | `supervisor_agent.py` | Product queries | - | <50ms |
| SearchAgent | `search_agent.py` | D1-D6 | Qdrant + ES | <200ms |
| CompareAgent | `compare_agent.py` | C1-C5 | Qdrant + PostgreSQL | <500ms |
| AnalysisAgent | `analysis_agent.py` | A1-A5 | Qdrant Child (reviews) | <400ms |
| PriceAgent | `price_agent.py` | P1-P5 | Qdrant + PostgreSQL | <400ms |
| TrendAgent | `trend_agent.py` | T1-T5 | PostgreSQL | <2s |
| RecommendAgent | `recommend_agent.py` | R1-R5 | Qdrant + PostgreSQL | <400ms |
| AttributeAgent | `attribute_agent.py` | All | - | <100ms |
| SynthesisAgent | `synthesis_agent.py` | All | - | <200ms |

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Basic chat with agent routing |
| `/chat/v2` | POST | Enhanced chat with workflow orchestration |
| `/query` | POST | Direct product search |
| `/query/v2` | POST | Workflow-based query with auto-selection |
| `/search` | GET | Product search with filters |
| `/compare` | POST | Compare specific products |
| `/recommend` | POST | Get recommendations |

#### Directory Structure

```
services/multi-agents/
├── src/
│   ├── agents/
│   │   ├── __init__.py           # Agent exports
│   │   ├── base.py               # BaseAgent class
│   │   ├── intent_agent.py       # Intent classification
│   │   ├── general_agent.py      # General conversation
│   │   ├── supervisor_agent.py   # Execution planning
│   │   ├── synthesis_agent.py    # Response generation
│   │   ├── attribute_agent.py    # Attribute extraction
│   │   ├── search_agent.py       # Product search
│   │   ├── compare_agent.py      # Product comparison
│   │   ├── recommend_agent.py    # Recommendations
│   │   ├── analysis_agent.py     # Review analysis
│   │   ├── price_agent.py        # Price intelligence
│   │   └── trend_agent.py        # Market trends
│   ├── models/
│   │   ├── intent.py             # Intent and entity models
│   │   └── execution.py          # Execution plan models
│   ├── workflows/
│   │   ├── simple_workflow.py    # Single-intent workflows
│   │   ├── compound_workflow.py  # Multi-step workflows
│   │   └── conversation_workflow.py  # Multi-turn context
│   ├── tools/
│   │   ├── search_tools.py       # Core search implementations
│   │   ├── langchain_tools.py    # LangChain wrappers
│   │   └── postgres_tools.py     # PostgreSQL operations
│   └── main.py                   # FastAPI application
└── tests/
    ├── test_intent.py            # 81 intent tests
    ├── test_general_agent.py     # 43 general agent tests
    ├── test_supervisor.py        # 54 supervisor tests
    ├── test_workflows.py         # 30 workflow tests
    └── test_scenarios.py         # 52 scenario tests
```

### 2.2 Data Pipeline Service

**Location:** `services/data-pipeline/`
**Port:** 8005

#### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | FastAPI | 0.115+ | Pipeline API |
| Data Processing | Polars | 1.18+ | High-performance DataFrames |
| Serialization | PyArrow | 18.1+ | Parquet support |
| PostgreSQL | asyncpg | 0.30+ | Async database driver |
| Qdrant | qdrant-client | 1.12+ | Vector operations |
| Elasticsearch | elasticsearch[async] | 8.17+ | Keyword indexing |

#### Pipeline Modes

| Mode | Stages | Data Richness | Processing Time |
|------|--------|---------------|-----------------|
| **original** | extract → clean → embed → load | Parent nodes only | ~2 hours for 2.1M |
| **enrich** | extract → download → html_to_md → llm_extract → clean → embed → load | Parent + 5 child nodes | ~24 hours for 2.1M |

#### Pipeline Stages

| Script | Input | Output | Mode |
|--------|-------|--------|------|
| `01_extract_mvp.py` | Source CSV | `raw/mvp_{count}_products.csv` | Both |
| `02a_download_html.py` | Product URLs | `scraped/html/*.html` | Enrich only |
| `02b_html_to_markdown.py` | HTML files | `scraped/markdown/*.md` | Enrich only |
| `02c_extract_with_llm.py` | Markdown + CSV | `scraped/mvp_{count}_{mode}_extracted.csv` | Enrich only |
| `03_clean_data.py` | Extracted data | `cleaned/mvp_{count}_{mode}_cleaned.csv` | Both |
| `04_generate_embeddings.py` | Cleaned data | `embedded/mvp_{count}_{mode}_embedded.parquet` | Both |
| `05_load_stores.py` | Embedded data | PostgreSQL, Qdrant, Elasticsearch | Both |

#### Qdrant Indexing Strategies

| Strategy | Use Case | Data Structure |
|----------|----------|----------------|
| `parent_only` | Original mode (basic product info) | Parent nodes only |
| `enrich_existing` | Enrich mode (incremental enrichment) | Update parents + add 5 children |
| `full_replace` | Complete data refresh | Delete and recreate |

### 2.3 Vector-Store Service

**Location:** `services/vector-store/`
**Port:** 8002

#### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | FastAPI | 0.115+ | Search API |
| Vector DB | Qdrant | 1.12+ | Semantic search |
| Search Engine | Elasticsearch | 8.17+ | Keyword search |

#### Search Strategies

| Strategy | Method | Performance | Best For |
|----------|--------|-------------|----------|
| KeywordPriorityHybrid | Qdrant + ES + RRF | MRR 0.9126 | General search (default) |
| Keyword | ES multi-match | R@1 87.7% | Brand+model queries |
| Semantic | Qdrant vector | MRR 0.65 | Generic/conceptual queries |
| Section | Qdrant child nodes | Best targeted | Reviews, specs, features |

#### Query Type Detection

| Query Type | Keyword Weight | Semantic Weight | Example |
|------------|----------------|-----------------|---------|
| brand_model | 0.75 | 0.25 | "Sony WH-1000XM5" |
| model_number | 0.70 | 0.30 | "WH-1000XM5" |
| short_title | 0.65 | 0.35 | "wireless headphones" |
| generic | 0.40 | 0.60 | "good laptop for travel" |

### 2.4 Frontend Service

**Location:** `services/frontend/`
**Port:** 8501

#### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | Streamlit | 1.40+ | Chat UI |
| Chat Components | streamlit-chat | 0.1.1+ | Message display |
| Charts | Plotly | 5.24+ | Trend visualization |

#### UI Features

| Feature | Description | Scenario Support |
|---------|-------------|------------------|
| Chat Interface | Natural language product queries | All scenarios |
| Product Cards | Display search results with key info | D1-D6 |
| Comparison View | Side-by-side product comparison | C1-C5 |
| Review Analysis | Sentiment visualization, pros/cons | A1-A5 |
| Price Insights | Price evaluation, deal badges | P1-P5 |
| Trend Charts | Category/brand trend visualization | T1-T5 |
| Recommendations | Similar products carousel | R1-R5 |

### 2.5 Ollama Service

**Location:** `services/ollama-service/`
**Port:** 8010

#### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| LLM Server | Ollama | 0.5+ | Model serving |
| Embedding Model | BGE-Large-EN-v1.5 | - | 1024-dim embeddings |
| Reranker | BGE-Reranker-v2-M3 | - | Cross-encoder reranking |
| LLM | Llama 3.3 70B / Qwen 2.5 | - | Product enrichment |

---

## 3. Data Flow

### 3.1 Pipeline Data Flow

```
Source CSV (2.1M products)
        │
        ▼
┌─────────────┐
│   Extract   │──▶ raw/mvp_{count}_products.csv
└─────────────┘
        │
        │ (enrich mode only)
        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Download   │────▶│ HTML to MD  │────▶│ LLM Extract │
│    HTML     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
        │
        ▼
┌─────────────┐
│    Clean    │──▶ cleaned/mvp_{count}_{mode}_cleaned.csv
│  (chunks)   │
└─────────────┘
        │
        ▼
┌─────────────┐
│    Embed    │──▶ embedded/mvp_{count}_{mode}_embedded.parquet
│  (Ollama)   │
└─────────────┘
        │
   ┌────┴────┬────────────┐
   ▼         ▼            ▼
PostgreSQL  Qdrant   Elasticsearch
(source)   (vectors)  (keywords)
```

### 3.2 Query Data Flow

```
User Query: "Compare Sony WH-1000XM5 vs Bose QC45"
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ IntentAgent                                                     │
│   1. Classify: COMPARE                                          │
│   2. Extract entities: [Sony WH-1000XM5, Bose QC45]            │
│   3. Confidence: 0.95                                           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ SupervisorAgent                                                 │
│   1. Create execution plan                                      │
│   2. Steps: [search_products, compare_products, synthesize]     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ SearchAgent (parallel)                                          │
│   ├── keyword_search("Sony WH-1000XM5") ──▶ Elasticsearch      │
│   └── keyword_search("Bose QC45") ──▶ Elasticsearch            │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ CompareAgent                                                    │
│   ├── section_search("specs", ASIN1) ──▶ Qdrant (child)        │
│   ├── section_search("specs", ASIN2) ──▶ Qdrant (child)        │
│   └── generate_comparison() ──▶ Structured output              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ SynthesisAgent                                                  │
│   ├── Format: COMPARISON                                        │
│   ├── Generate natural language response                        │
│   └── Add follow-up suggestions                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
Comparison Response
```

---

## 4. Technology Stack Summary

### 4.1 Core Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.12+ | Latest features, performance |
| Web Framework | FastAPI 0.115+ | Async support, OpenAPI |
| Frontend | Streamlit 1.40+ | Rapid UI development |
| Agent Framework | LangGraph 0.2+ | Stateful agent orchestration |

### 4.2 Databases

| Database | Technology | Purpose |
|----------|------------|---------|
| Vector DB | Qdrant 1.12+ | Semantic search, filtering |
| Search Engine | Elasticsearch 8.17+ | Keyword search, aggregations |
| Relational DB | PostgreSQL 17+ | Source of truth, analytics |
| Cache | Redis 7.4+ | Query caching, session state |

### 4.3 ML/AI

| Component | Technology | Purpose |
|-----------|------------|---------|
| Model Server | Ollama 0.5+ | Local LLM/embedding serving |
| Embedding Model | BGE-Large-EN-v1.5 | 1024-dim embeddings |
| Reranker | BGE-Reranker-v2-M3 | Cross-encoder reranking |
| LLM | Llama 3.3 70B | Product enrichment |

### 4.4 Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >=0.115.0 | Web framework |
| `uvicorn[standard]` | >=0.32.0 | ASGI server |
| `pydantic` | >=2.10.0 | Data validation |
| `langgraph` | >=0.2.60 | Agent orchestration |
| `langchain` | >=0.3.14 | LLM abstractions |
| `qdrant-client` | >=1.12.0 | Qdrant SDK |
| `elasticsearch[async]` | >=8.17.0 | Async ES client |
| `asyncpg` | >=0.30.0 | Async PostgreSQL |
| `redis` | >=5.2.0 | Async Redis client |
| `httpx` | >=0.28.0 | Async HTTP client |
| `polars` | >=1.18.0 | High-performance DataFrames |
| `structlog` | >=24.4.0 | Structured logging |

---

## 5. Key Design Decisions

### 5.1 Storage Strategy: Qdrant = Speed, PostgreSQL = Depth

**Decision:** 80%+ of queries should be answerable by Qdrant alone.

**Rationale:**
- Qdrant provides sub-100ms vector search with filtering
- PostgreSQL handles complex aggregations and time-series data
- Parent nodes contain display-ready data for immediate response
- Child nodes enable deep-dive queries (reviews, specs, features)

### 5.2 Hybrid Search with Keyword Priority

**Decision:** Default to KeywordPriorityHybrid strategy (MRR 0.9126).

**Rationale:**
- Experiments show keyword-priority outperforms balanced RRF
- Brand+model queries benefit from higher keyword weight (0.75)
- Generic queries still leverage semantic search (0.60 weight)

### 5.3 Parent-Child Node Structure in Qdrant

**Decision:** Each product has 1 parent + 5 child nodes (description, features, specs, reviews, use_cases).

**Rationale:**
- Parent answers: "What is this? What's it for? Who is it for?"
- Children answer: "Tell me more about reviews/specs/features"
- Enables section-specific search with inherited filters

### 5.4 Two Pipeline Modes

**Decision:** Support both `original` (lightweight) and `enrich` (full) modes.

**Rationale:**
- Original mode enables rapid prototyping without LLM overhead
- Enrich mode provides production-quality GenAI fields
- Same codebase, configuration-driven behavior

---

## 6. References

- Proposal Document: [PROPOSAL.md](./PROPOSAL.md)
- Database Schemas: [DATABASE_SCHEMAS.md](./DATABASE_SCHEMAS.md)
- Architecture Flow: [architecture.md](ARCHITECTURE.md)