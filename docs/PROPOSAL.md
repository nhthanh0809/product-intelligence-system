# Product Intelligence System - Proposal Document

## Executive Summary

The Product Intelligence System is a multi-agent AI system designed to handle intelligent product discovery, comparison, analysis, and recommendations for millions of Amazon products. The system leverages a 3-tier storage architecture (PostgreSQL, Qdrant, Elasticsearch) with LLM-powered agents to provide sub-second search responses across 30 real-world user scenarios.

**Key Objectives:**
- Process and index 2.1M+ Amazon products with rich metadata
- Support sub-second search latency at scale (<100ms for basic, <500ms for complex queries)
- Answer 80%+ of queries directly from Qdrant vector store
- Provide intelligent, context-aware responses through specialized agents

---

## 1. Project Overview

### 1.1 Problem Statement

E-commerce users face challenges when:
- Searching for products with natural language queries
- Comparing multiple products across different attributes
- Understanding product reviews and sentiment
- Identifying deals and price trends
- Getting personalized recommendations

Current solutions lack:
- Semantic understanding of product queries
- Intelligent routing to specialized analysis
- Unified view across product data sources
- Real-time, conversational product assistance

### 1.2 Solution

A multi-agent Product Intelligence System that:
1. **Understands** user intent through natural language processing
2. **Routes** queries to specialized agents (Search, Compare, Analyze, Price, Trend, Recommend)
3. **Retrieves** relevant data from optimized storage (Qdrant for speed, PostgreSQL for depth)
4. **Synthesizes** intelligent responses with citations and follow-up suggestions

---

## 2. User Scenarios

### 2.1 Scenario Categories

| Category | Code | Description | Agent | Complexity | Frequency |
|----------|------|-------------|-------|------------|-----------|
| **Discovery** | D1-D6 | Finding products matching criteria | SearchAgent | Low-Medium | HIGH (>60%) |
| **Comparison** | C1-C5 | Evaluating multiple products | CompareAgent | Medium-High | HIGH |
| **Analysis** | A1-A5 | Deep-dive into product details | AnalysisAgent | Medium | MEDIUM |
| **Price Intelligence** | P1-P5 | Understanding pricing and deals | PriceAgent | Low-Medium | MEDIUM |
| **Trend Analysis** | T1-T5 | Market and category insights | TrendAgent | High | LOW |
| **Recommendation** | R1-R5 | Personalized suggestions | RecommendAgent | Medium-High | MEDIUM |

### 2.2 Discovery Scenarios (D1-D6)

| ID | Scenario | Example Query | Expected Response | Target Latency |
|----|----------|---------------|-------------------|----------------|
| D1 | Basic Product Search | "Find wireless headphones under $100" | List of relevant products with key details | <100ms |
| D2 | Feature-Based Search | "Laptop with 16GB RAM for programming" | Filtered list emphasizing matching specs | <200ms |
| D3 | Use-Case Based Search | "Good camera for beginner photographer" | Curated recommendations with explanations | <200ms |
| D4 | Gift Finding | "Gift for 10-year-old who likes science" | Age-appropriate, interest-matched suggestions | <300ms |
| D5 | Brand-Specific Search | "Show me all Sony noise-cancelling headphones" | Complete brand catalog with filtering | <100ms |
| D6 | Highly-Rated Products | "Best rated coffee makers" | Ranked list based on ratings and reviews | <100ms |

### 2.3 Comparison Scenarios (C1-C5)

| ID | Scenario | Example Query | Expected Response | Target Latency |
|----|----------|---------------|-------------------|----------------|
| C1 | Direct Product Comparison | "Compare Sony WH-1000XM5 vs Bose QC45" | Detailed comparison table with pros/cons | <500ms |
| C2 | Category Comparison | "What's the difference between OLED and QLED TVs?" | Educational comparison with examples | <500ms |
| C3 | Value Comparison | "Best value laptop under $1000" | Value analysis with scoring methodology | <800ms |
| C4 | Brand Comparison | "Samsung vs LG for refrigerators" | Brand-level analysis with representatives | <500ms |
| C5 | Feature Trade-off Analysis | "Battery life vs display quality in tablets" | Trade-off analysis with product examples | <500ms |

### 2.4 Analysis Scenarios (A1-A5)

| ID | Scenario | Example Query | Expected Response | Target Latency |
|----|----------|---------------|-------------------|----------------|
| A1 | Review Summary | "What do people say about iPhone 15 battery?" | Aggregated sentiment with representative quotes | <400ms |
| A2 | Pros and Cons Analysis | "Pros and cons of this product" | Structured pros/cons list from reviews | <300ms |
| A3 | Common Issues | "Common complaints about Dyson vacuums" | Categorized issues with frequency | <400ms |
| A4 | Feature Deep-Dive | "Camera system details on Pixel 8" | Comprehensive feature breakdown | <400ms |
| A5 | Quality Assessment | "Is this product durable long-term?" | Durability insights from extended-use reviews | <800ms |

### 2.5 Price Intelligence Scenarios (P1-P5)

| ID | Scenario | Example Query | Expected Response | Target Latency |
|----|----------|---------------|-------------------|----------------|
| P1 | Price Evaluation | "Is $299 a good price for AirPods Pro?" | Price context with historical comparison | <200ms |
| P2 | Deal Finding | "Best deals on gaming monitors" | Products with significant discounts | <200ms |
| P3 | Budget Optimization | "$500 budget for home office setup" | Optimized product bundle recommendations | <400ms |
| P4 | Price Range Analysis | "Typical price range for mechanical keyboards" | Price distribution with tier explanations | <500ms |
| P5 | Alternative Finding | "Cheaper alternative to this $200 blender" | Comparable products with price savings | <400ms |

### 2.6 Trend Analysis Scenarios (T1-T5)

| ID | Scenario | Example Query | Expected Response | Target Latency |
|----|----------|---------------|-------------------|----------------|
| T1 | Category Trends | "Trending products in smart home" | Trending products with momentum indicators | <2s |
| T2 | Feature Trends | "What features are standard in laptops now?" | Feature adoption trends over time | <1s |
| T3 | Brand Performance | "Which brands are gaining popularity in headphones?" | Brand market share and trajectory | <500ms |
| T4 | Market Overview | "Give me an overview of the tablet market" | Market summary with key players and segments | <2s |
| T5 | Emerging Categories | "New product categories emerging in electronics" | Novel category identification with examples | <2s |

### 2.7 Recommendation Scenarios (R1-R5)

| ID | Scenario | Example Query | Expected Response | Target Latency |
|----|----------|---------------|-------------------|----------------|
| R1 | Accessory Recommendations | "Accessories for Canon EOS R6" | Essential and optional accessories | <400ms |
| R2 | Alternative Recommendations | "Similar products with better reviews" | Alternatives with better ratings | <300ms |
| R3 | Upgrade Recommendations | "5-year-old laptop, what should I upgrade to?" | Modern alternatives with improvements | <400ms |
| R4 | Bundle Recommendations | "Frequently bought together with this" | Common bundles with savings analysis | <300ms |
| R5 | Personalized Recommendations | "Products for photography enthusiast" | Interest-based curated suggestions | <400ms |

---

## 3. Scenario Frequency × Depth Matrix

```
                        QUESTION DEPTH
                 ┌─────────────────────────────────────────────────────────┐
                 │   SHALLOW            MEDIUM              DEEP           │
                 │   (Display only)     (Some lookup)       (Aggregation)  │
    ┌────────────┼─────────────────────────────────────────────────────────┤
    │  HIGH      │  D1,D5,D6            D2: Feature Search  C1: Compare    │
    │  (>60%)    │  Basic/Brand Search  "laptop 16GB RAM"   "X vs Y"       │
    │            │  Target: <100ms      Target: <200ms      Target: <500ms │
F   ├────────────┼─────────────────────────────────────────────────────────┤
R   │  MEDIUM    │  P1,P2               A1,A2               C2,C3          │
E   │  (20-40%)  │  Price Check/Deals   Review Summary      Category       │
Q   │            │  Target: <200ms      Target: <400ms      Target: <800ms │
    ├────────────┼─────────────────────────────────────────────────────────┤
    │  LOW       │  P4: Price Range     A5: Quality         T1-T5          │
    │  (<20%)    │  Target: <500ms      Target: <800ms      Target: <2s    │
    └────────────┴─────────────────────────────────────────────────────────┘
```

---

## 4. Requirements

### 4.1 Functional Requirements

#### FR-1: Query Understanding
- FR-1.1: Classify user queries into 14 intent types (6 product + 6 general + 2 complex)
- FR-1.2: Extract entities (products, brands, categories, constraints) from queries
- FR-1.3: Detect compound queries with multiple intents
- FR-1.4: Resolve context references in multi-turn conversations

#### FR-2: Product Search
- FR-2.1: Support semantic search using vector embeddings
- FR-2.2: Support keyword search for exact matches (brand+model)
- FR-2.3: Support hybrid search combining semantic and keyword
- FR-2.4: Support filtered search with price, rating, category constraints
- FR-2.5: Achieve MRR 0.9126+ using KeywordPriorityHybrid strategy

#### FR-3: Product Comparison
- FR-3.1: Generate side-by-side comparison tables
- FR-3.2: Highlight feature differences and winners per attribute
- FR-3.3: Provide pros/cons from reviews for each product
- FR-3.4: Generate recommendation with reasoning

#### FR-4: Review Analysis
- FR-4.1: Extract sentiment scores and labels
- FR-4.2: Identify common praises and complaints
- FR-4.3: Assess durability and long-term quality
- FR-4.4: Support section-specific search (reviews, specs, features)

#### FR-5: Price Intelligence
- FR-5.1: Evaluate price competitiveness using percentiles
- FR-5.2: Find deals with significant discounts
- FR-5.3: Provide price history and trends
- FR-5.4: Suggest budget-friendly alternatives

#### FR-6: Trend Analysis
- FR-6.1: Identify trending products in categories
- FR-6.2: Track brand market share and growth
- FR-6.3: Analyze feature adoption trends
- FR-6.4: Provide market overviews

#### FR-7: Recommendations
- FR-7.1: Find similar products using vector similarity
- FR-7.2: Suggest compatible accessories
- FR-7.3: Identify upgrade paths
- FR-7.4: Generate personalized suggestions

#### FR-8: Response Generation
- FR-8.1: Generate natural language responses
- FR-8.2: Support multiple output formats (text, bullet, table, cards, comparison)
- FR-8.3: Include citations and source attribution
- FR-8.4: Provide follow-up suggestions

### 4.2 Non-Functional Requirements

#### NFR-1: Performance
- NFR-1.1: Basic search latency <100ms (p50), <200ms (p99)
- NFR-1.2: Complex queries <500ms (p50), <2s (p99)
- NFR-1.3: Indexing throughput: 10K products/minute
- NFR-1.4: Support 100+ concurrent users

#### NFR-2: Scalability
- NFR-2.1: Handle 2.1M+ products
- NFR-2.2: Support 12.6M+ vector points (parents + children)
- NFR-2.3: Query cache hit rate >60%

#### NFR-3: Reliability
- NFR-3.1: Service availability >99.5%
- NFR-3.2: Graceful degradation when services unavailable
- NFR-3.3: Circuit breakers for external service calls

#### NFR-4: Data Quality
- NFR-4.1: 80%+ of queries answerable by Qdrant alone
- NFR-4.2: Data completeness >95% for required fields
- NFR-4.3: Embedding coherence >0.8 for related items

### 4.3 Constraints

- **Data Source**: Amazon Canada Products 2023 dataset (2.1M products)
- **Infrastructure**: Docker Compose with GPU support for Ollama
- **LLM**: Local Ollama deployment (no external API dependencies)
- **Deployment**: Single-node deployment (no Kubernetes)

---

## 5. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search MRR | >0.9126 | Hybrid search evaluation |
| Recall@10 | >80% | Retrieval evaluation |
| Classification Accuracy | >90% | Intent detection tests |
| Qdrant-Only Completion | >80% | Queries without PostgreSQL |
| Response Latency (p50) | <100ms basic, <500ms complex | Performance benchmarks |
| User Scenario Pass Rate | >90% | End-to-end scenario tests |

---

## 6. Project Scope

### 6.1 In Scope

- Multi-agent architecture with 7 specialist agents
- 3-tier storage (Qdrant, Elasticsearch, PostgreSQL)
- Data pipeline for product ingestion and enrichment
- Streamlit-based chatbot UI
- 30 user scenarios across 6 categories
- Two pipeline modes: original (lightweight) and enrich (full)

### 6.2 Out of Scope

- Real-time price tracking (batch processing only)
- User authentication/authorization
- Production scaling optimizations (horizontal scaling)
- External API integrations beyond Ollama
- Mobile applications

---

## 7. References

- Design Document: [DESIGN.md](./DESIGN.md)
- Database Schemas: [DATABASE_SCHEMAS.md](./DATABASE_SCHEMAS.md)
- Architecture Flow: [architecture.md](ARCHITECTURE.md)