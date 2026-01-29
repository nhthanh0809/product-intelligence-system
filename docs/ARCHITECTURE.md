# Multi-Agent System Architecture

## End-to-End Flow Chart

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT QUERY                                            │
│                     "Find Sony headphones under $200 with good reviews"                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 1: INTENT AGENT                                       │
│                              src/agents/intent_agent.py                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ Query Classification (14 intent types)                                           │    │
│  │                                                                                   │    │
│  │ PRODUCT INTENTS:        GENERAL INTENTS:        COMPLEX INTENTS:                │    │
│  │ • SEARCH                • GREETING              • COMPOUND                       │    │
│  │ • COMPARE               • FAREWELL              • AMBIGUOUS                      │    │
│  │ • ANALYZE               • HELP                                                   │    │
│  │ • PRICE_CHECK           • SMALL_TALK                                            │    │
│  │ • TREND                 • OFF_TOPIC                                             │    │
│  │ • RECOMMEND             • CLARIFICATION                                         │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ Entity Extraction                                                                │    │
│  │                                                                                   │    │
│  │ • products: []                    • constraints: {price_max: 200}               │    │
│  │ • brands: ["Sony"]                • use_cases: ["good reviews"]                 │    │
│  │ • categories: ["headphones"]      • references_previous: false                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  OUTPUT: IntentAnalysis {                                                                │
│    primary_intent: SEARCH,                                                               │
│    entities: {...},                                                                      │
│    complexity: SIMPLE,                                                                   │
│    confidence: 0.92                                                                      │
│  }                                                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                      ┌───────────────────┼───────────────────┐
                      │                   │                   │
                      ▼                   ▼                   ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│   GENERAL INTENT PATH       │ │   PRODUCT INTENT PATH       │ │   COMPLEX INTENT PATH       │
│   (GREETING, HELP, etc.)    │ │   (SEARCH, COMPARE, etc.)   │ │   (COMPOUND, AMBIGUOUS)     │
├─────────────────────────────┤ ├─────────────────────────────┤ ├─────────────────────────────┤
│                             │ │                             │ │                             │
│  ┌───────────────────────┐  │ │      Continue to            │ │  COMPOUND:                  │
│  │    GENERAL AGENT      │  │ │   WORKFLOW SELECTION        │ │  → CompoundWorkflow         │
│  │ general_agent.py      │  │ │          ↓                  │ │  → Decompose into steps     │
│  │                       │  │ │                             │ │  → Execute sequentially     │
│  │ Returns canned        │  │ │                             │ │                             │
│  │ responses for         │  │ │                             │ │  AMBIGUOUS:                 │
│  │ non-product queries   │  │ │                             │ │  → Ask clarification        │
│  └───────────────────────┘  │ │                             │ │  → Or use best guess        │
│            │                │ │                             │ │            │                │
│            ▼                │ │                             │ │            ▼                │
│   [Skip to SYNTHESIS]       │ │                             │ │   [Route to appropriate     │
│                             │ │                             │ │    workflow]                │
└─────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 2: WORKFLOW SELECTION                                 │
│                              src/workflows/                                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────────────┐   │
│  │  SimpleWorkflow   │  │ CompoundWorkflow  │  │    ConversationWorkflow           │   │
│  │  (single intent)  │  │ (multi-step)      │  │    (multi-turn context)           │   │
│  │                   │  │                   │  │                                   │   │
│  │ "find headphones" │  │ "find, compare,   │  │ Turn 1: "find headphones"         │   │
│  │                   │  │  then recommend"  │  │ Turn 2: "compare first two"       │   │
│  └─────────┬─────────┘  └─────────┬─────────┘  └─────────────────┬─────────────────┘   │
│            │                      │                              │                      │
│            └──────────────────────┼──────────────────────────────┘                      │
│                                   ▼                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 3: SUPERVISOR AGENT                                   │
│                              src/agents/supervisor_agent.py                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  Creates ExecutionPlan based on intent:                                                  │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ ExecutionPlan {                                                                  │    │
│  │   steps: [                                                                       │    │
│  │     { agent: "search", description: "Find Sony headphones", depends_on: [] },   │    │
│  │   ],                                                                             │    │
│  │   parallel_groups: [["search"]],                                                │    │
│  │   estimated_duration_ms: 500                                                     │    │
│  │ }                                                                                │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  AGENT ROUTING BY INTENT:                                                                │
│  ┌──────────────┬─────────────────────────────────────────────────────────────────┐    │
│  │ Intent       │ Agents Invoked                                                   │    │
│  ├──────────────┼─────────────────────────────────────────────────────────────────┤    │
│  │ SEARCH       │ SearchAgent                                                      │    │
│  │ COMPARE      │ SearchAgent → CompareAgent                                       │    │
│  │ ANALYZE      │ SearchAgent → AnalysisAgent                                      │    │
│  │ PRICE_CHECK  │ SearchAgent → PriceAgent                                         │    │
│  │ TREND        │ TrendAgent                                                       │    │
│  │ RECOMMEND    │ SearchAgent → RecommendAgent                                     │    │
│  └──────────────┴─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 4: SEARCH AGENT                                       │
│                              src/agents/search_agent.py                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4.1: QUERY ANALYSIS (QueryAnalyzer)                                        │    │
│  │ src/search/query_analyzer.py                                                    │    │
│  │                                                                                  │    │
│  │ NOTE: This is a SEARCH-SPECIFIC analyzer, different from Intent Agent.          │    │
│  │ Intent Agent detects WHAT user wants (search, compare, etc.)                    │    │
│  │ QueryAnalyzer detects HOW to search (keyword vs semantic weights)               │    │
│  │                                                                                  │    │
│  │ Input: "Find Sony headphones under $200 with good reviews"                      │    │
│  │                                                                                  │    │
│  │ Detects:                                                                         │    │
│  │ • query_type: BRAND_MODEL (has brand "Sony")                                    │    │
│  │ • has_brand: true                                                               │    │
│  │ • detected_brand: "Sony"                                                        │    │
│  │ • model_numbers: []                                                             │    │
│  │ • clean_query: "sony headphones"                                                │    │
│  │ • keyword_weight: 0.75                                                          │    │
│  │ • semantic_weight: 0.25                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4.2: FILTER EXTRACTION (FilterExtractor)                                   │    │
│  │                                                                                  │    │
│  │ Extracts from natural language:                                                 │    │
│  │ • brand: "Sony"         (pattern: known brand list)                             │    │
│  │ • price_max: 200        (pattern: "under $200")                                 │    │
│  │ • min_rating: 4.0       (pattern: "good reviews" → highly rated)               │    │
│  │ • category: "headphones"                                                        │    │
│  │                                                                                  │    │
│  │ SearchFilters {                                                                  │    │
│  │   brand: "Sony",                                                                │    │
│  │   category: "headphones",                                                       │    │
│  │   min_price: null,                                                              │    │
│  │   max_price: 200,                                                               │    │
│  │   min_rating: 4.0                                                               │    │
│  │ }                                                                                │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4.3: STRATEGY SELECTION                                                    │    │
│  │ src/search/registry.py                                                          │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ QUERY TYPE → STRATEGY MAPPING                                           │     │    │
│  │ │                                                                          │     │    │
│  │ │ ┌────────────────┬────────────────┬────────────────┬─────────────────┐ │     │    │
│  │ │ │ Query Type     │ Strategy       │ Performance    │ Weights         │ │     │    │
│  │ │ ├────────────────┼────────────────┼────────────────┼─────────────────┤ │     │    │
│  │ │ │ MODEL_NUMBER   │ Keyword        │ R@1 87.7%      │ KW:0.85 SEM:0.15│ │     │    │
│  │ │ │ BRAND_MODEL ◄──│ Keyword/Hybrid │ R@1 87.7%      │ KW:0.80 SEM:0.20│ │     │    │
│  │ │ │ SHORT_TITLE    │ Hybrid         │ MRR 0.9126     │ KW:0.65 SEM:0.35│ │     │    │
│  │ │ │ GENERIC        │ Semantic       │ MRR 0.65       │ KW:0.35 SEM:0.65│ │     │    │
│  │ │ │ SECTION        │ Section        │ Targeted       │ KW:0.30 SEM:0.70│ │     │    │
│  │ │ └────────────────┴────────────────┴────────────────┴─────────────────┘ │     │    │
│  │ │                                                                          │     │    │
│  │ │ Selected: KEYWORD strategy (brand detected → keyword priority)          │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 5: SEARCH EXECUTION                                   │
│                              src/search/clients.py + strategies/                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ╔═══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                         SEARCH STRATEGY EXECUTION                                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ OPTION A: KEYWORD SEARCH (Elasticsearch)                                        │    │
│  │ src/search/strategies/keyword/basic.py                                          │    │
│  │ Best for: Brand+Model, Model numbers                                            │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ ES QUERY STRUCTURE                                                       │     │    │
│  │ │                                                                          │     │    │
│  │ │ {                                                                        │     │    │
│  │ │   "bool": {                                                              │     │    │
│  │ │     "must": [{                                                           │     │    │
│  │ │       "multi_match": {                                                   │     │    │
│  │ │         "query": "sony headphones",                                     │     │    │
│  │ │         "fields": [                                                      │     │    │
│  │ │           "title^10.0",           ◄── BOOST WEIGHTS                     │     │    │
│  │ │           "title.autocomplete^5.0",                                     │     │    │
│  │ │           "short_title^8.0",                                            │     │    │
│  │ │           "brand^5.0",                                                  │     │    │
│  │ │           "product_type^4.0",                                           │     │    │
│  │ │           "genAI_summary^2.0",                                          │     │    │
│  │ │           "chunk_description^1.0",                                      │     │    │
│  │ │           "chunk_features^1.0"                                          │     │    │
│  │ │         ],                                                               │     │    │
│  │ │         "type": "best_fields",                                          │     │    │
│  │ │         "fuzziness": "AUTO"                                             │     │    │
│  │ │       }                                                                  │     │    │
│  │ │     }],                                                                  │     │    │
│  │ │     "filter": [                    ◄── FILTERS APPLIED                  │     │    │
│  │ │       {"term": {"brand.keyword": "Sony"}},                              │     │    │
│  │ │       {"range": {"price": {"lte": 200}}},                               │     │    │
│  │ │       {"range": {"stars": {"gte": 4.0}}}                                │     │    │
│  │ │     ]                                                                    │     │    │
│  │ │   }                                                                      │     │    │
│  │ │ }                                                                        │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ ELASTICSEARCH SCHEMA FIELDS ACCESSED                                    │     │    │
│  │ │                                                                          │     │    │
│  │ │ SEARCH FIELDS:              FILTER FIELDS:          RETURN FIELDS:      │     │    │
│  │ │ • title (text)              • brand.keyword         • asin              │     │    │
│  │ │ • title.autocomplete        • category_name.keyword • title             │     │    │
│  │ │ • short_title (text)        • price (float)         • price             │     │    │
│  │ │ • brand (text)              • stars (float)         • stars             │     │    │
│  │ │ • product_type (keyword)                            • brand             │     │    │
│  │ │ • genAI_summary (text)                              • category_name     │     │    │
│  │ │ • chunk_description (text)                          • category_level1   │     │    │
│  │ │ • chunk_features (text)                             • img_url           │     │    │
│  │ │                                                     • genAI_summary     │     │    │
│  │ │                                                     • genAI_best_for    │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ OPTION B: SEMANTIC SEARCH (Qdrant Vector DB)                                    │    │
│  │ src/search/strategies/semantic/basic.py                                         │    │
│  │ Best for: Generic/conceptual queries                                            │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ STEP 1: Generate Embedding                                               │     │    │
│  │ │                                                                          │     │    │
│  │ │ POST /embed/single → Ollama Service                                     │     │    │
│  │ │ {                                                                        │     │    │
│  │ │   "text": "sony headphones",                                            │     │    │
│  │ │   "model": "bge-large"                                                  │     │    │
│  │ │ }                                                                        │     │    │
│  │ │ Returns: [0.023, -0.156, 0.089, ...] (1024-dim vector)                 │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ STEP 2: Vector Similarity Search                                        │     │    │
│  │ │                                                                          │     │    │
│  │ │ Qdrant query_points() with:                                             │     │    │
│  │ │ • collection: "products"                                                │     │    │
│  │ │ • query: [embedding_vector]                                             │     │    │
│  │ │ • distance: COSINE                                                      │     │    │
│  │ │ • limit: 50 (fetch_multiplier=5 × limit=10)                            │     │    │
│  │ │ • score_threshold: 0.5                                                  │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ QDRANT FILTER STRUCTURE                                                 │     │    │
│  │ │                                                                          │     │    │
│  │ │ Filter {                                                                 │     │    │
│  │ │   must: [                                                                │     │    │
│  │ │     FieldCondition(key="brand", match=MatchValue("Sony")),             │     │    │
│  │ │     FieldCondition(key="price", range=Range(lte=200)),                 │     │    │
│  │ │     FieldCondition(key="stars", range=Range(gte=4.0))                  │     │    │
│  │ │   ]                                                                      │     │    │
│  │ │ }                                                                        │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ QDRANT PAYLOAD FIELDS ACCESSED                                          │     │    │
│  │ │                                                                          │     │    │
│  │ │ INDEXED FILTERS:            PAYLOAD RETURN FIELDS:                      │     │    │
│  │ │ • brand (keyword)           • asin / parent_asin                        │     │    │
│  │ │ • category_level1 (keyword) • title                                     │     │    │
│  │ │ • category_level2 (keyword) • price                                     │     │    │
│  │ │ • price (float)             • stars                                     │     │    │
│  │ │ • stars (float)             • brand                                     │     │    │
│  │ │ • reviews_count (integer)   • category_name / category_level1          │     │    │
│  │ │ • bought_in_last_month (int)• img_url / imgUrl                          │     │    │
│  │ │ • is_best_seller (bool)     • genAI_summary                             │     │    │
│  │ │ • is_amazon_choice (bool)   • genAI_best_for                            │     │    │
│  │ │ • prime_eligible (bool)     • node_type (parent/child)                  │     │    │
│  │ │ • node_type (keyword)       • section (if child node)                   │     │    │
│  │ │ • section (keyword)                                                     │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ OPTION C: HYBRID SEARCH (ES + Qdrant combined)                                  │    │
│  │ src/search/strategies/hybrid/adaptive.py                                        │    │
│  │ Best for: General search (MRR 0.9126)                                           │    │
│  │                                                                                  │    │
│  │ ┌───────────────────────────────────────────────────────────────────────┐       │    │
│  │ │                     PARALLEL EXECUTION                                 │       │    │
│  │ │                                                                        │       │    │
│  │ │   ┌─────────────────┐              ┌─────────────────┐                │       │    │
│  │ │   │ Keyword Search  │              │ Semantic Search │                │       │    │
│  │ │   │ (Elasticsearch) │              │ (Qdrant)        │                │       │    │
│  │ │   │                 │   asyncio    │                 │                │       │    │
│  │ │   │ weight: 0.80    │◄──gather()──►│ weight: 0.20    │                │       │    │
│  │ │   │                 │              │                 │                │       │    │
│  │ │   └────────┬────────┘              └────────┬────────┘                │       │    │
│  │ │            │                                │                          │       │    │
│  │ │            └────────────┬───────────────────┘                          │       │    │
│  │ │                         ▼                                              │       │    │
│  │ │            ┌─────────────────────────┐                                │       │    │
│  │ │            │   RRF FUSION            │                                │       │    │
│  │ │            │   (Reciprocal Rank      │                                │       │    │
│  │ │            │    Fusion, k=40)        │                                │       │    │
│  │ │            │                         │                                │       │    │
│  │ │            │ score = Σ weight/(k+rank)                                │       │    │
│  │ │            └────────────┬────────────┘                                │       │    │
│  │ │                         ▼                                              │       │    │
│  │ │            ┌─────────────────────────┐                                │       │    │
│  │ │            │   ADAPTIVE BOOSTS       │                                │       │    │
│  │ │            │   • Model match: ×1.6   │                                │       │    │
│  │ │            │   • Brand match: ×1.3   │                                │       │    │
│  │ │            │   • Both sources: ×1.15 │                                │       │    │
│  │ │            └─────────────────────────┘                                │       │    │
│  │ └───────────────────────────────────────────────────────────────────────┘       │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ OPTION D: POSTGRESQL DIRECT QUERIES                                             │    │
│  │ src/tools/postgres_tools.py                                                     │    │
│  │ Used by: TrendAgent, PriceAgent, AnalysisAgent                                  │    │
│  │                                                                                  │    │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐     │    │
│  │ │ QUERIES & FIELDS ACCESSED                                               │     │    │
│  │ │                                                                          │     │    │
│  │ │ get_product(asin):                                                      │     │    │
│  │ │   SELECT * FROM products WHERE asin = $1                                │     │    │
│  │ │   → All 70+ columns including genAI_* fields                            │     │    │
│  │ │                                                                          │     │    │
│  │ │ search_products(filters):                                               │     │    │
│  │ │   SELECT asin, title, brand, price, stars, reviews_count,              │     │    │
│  │ │          category_level1, is_best_seller, img_url                       │     │    │
│  │ │   FROM products                                                         │     │    │
│  │ │   WHERE category_level1 = $1 AND brand = $2                            │     │    │
│  │ │         AND price >= $3 AND price <= $4 AND stars >= $5                │     │    │
│  │ │   ORDER BY reviews_count DESC, stars DESC                              │     │    │
│  │ │                                                                          │     │    │
│  │ │ get_trending_products(category, days):                                  │     │    │
│  │ │   SELECT p.asin, p.title, p.brand, p.price, p.stars,                   │     │    │
│  │ │          p.reviews_count, p.bought_in_last_month,                       │     │    │
│  │ │          pt.trend_score, pt.review_velocity                             │     │    │
│  │ │   FROM products p LEFT JOIN product_trends pt ON p.asin = pt.asin     │     │    │
│  │ │   ORDER BY bought_in_last_month DESC, reviews_count DESC               │     │    │
│  │ │                                                                          │     │    │
│  │ │ get_price_history(asin, days):                                          │     │    │
│  │ │   SELECT price, original_price, discount_percentage, recorded_at       │     │    │
│  │ │   FROM price_history WHERE asin = $1                                   │     │    │
│  │ │                                                                          │     │    │
│  │ │ get_product_reviews(asin):                                              │     │    │
│  │ │   SELECT title, body, rating, author_name, verified_purchase,          │     │    │
│  │ │          review_date, helpful_votes, sentiment_label                    │     │    │
│  │ │   FROM reviews WHERE asin = $1                                         │     │    │
│  │ └─────────────────────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 6: OPTIONAL RERANKING                                 │
│                              src/search/clients.py:rerank_results()                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ Reranker (if enabled in config)                                                 │    │
│  │                                                                                  │    │
│  │ Model: qllama/bge-reranker-v2-m3                                                │    │
│  │                                                                                  │    │
│  │ Document Text Built From:                                                        │    │
│  │ • title                                                                          │    │
│  │ • brand ("Brand: {brand}")                                                      │    │
│  │ • genAI_summary[:200]                                                           │    │
│  │                                                                                  │    │
│  │ POST /rerank → Ollama Service                                                   │    │
│  │ {                                                                                │    │
│  │   "query": "sony headphones under $200",                                        │    │
│  │   "documents": ["Sony WH-1000XM5 | Brand: Sony | Premium noise...", ...],      │    │
│  │   "model": "qllama/bge-reranker-v2-m3"                                          │    │
│  │ }                                                                                │    │
│  │                                                                                  │    │
│  │ Returns scores [0.92, 0.87, 0.73, ...] → Re-sort results                       │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 7: RESULT FORMATTING                                  │
│                              src/agents/search_agent.py:_format_results()                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  SearchResult → FormattedResult                                                          │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ {                                                                                │    │
│  │   "asin": "B09XS7JWHH",                                                         │    │
│  │   "title": "Sony WH-1000XM5 Wireless Headphones",                               │    │
│  │   "brand": "Sony",                                                              │    │
│  │   "price": 348.00,                                                              │    │
│  │   "stars": 4.6,                                                                 │    │
│  │   "score": 0.89,                                                                │    │
│  │   "img_url": "https://m.media-amazon.com/images/I/...",                         │    │
│  │   "summary": "Premium wireless headphones with industry-leading...",           │    │
│  │   "best_for": "Audiophiles, frequent travelers, remote workers"                │    │
│  │ }                                                                                │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  Summary Generation:                                                                     │
│  "Found 8 products priced $149.00-$398.00 (avg $267.50) with average rating 4.4★        │
│   from Sony."                                                                            │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 8: SPECIALIST AGENTS (if needed)                      │
│                              Based on intent routing                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  CompareAgent   │  │  AnalysisAgent  │  │   PriceAgent    │  │   TrendAgent    │    │
│  │                 │  │                 │  │                 │  │                 │    │
│  │ Creates:        │  │ Analyzes:       │  │ Queries:        │  │ Queries:        │    │
│  │ • Comparison    │  │ • Reviews       │  │ • price_history │  │ • product_trends│    │
│  │   matrix        │  │ • Sentiment     │  │ • PG function:  │  │ • bought_in_    │    │
│  │ • Winner per    │  │ • Aspects       │  │   get_category_ │  │   last_month    │    │
│  │   attribute     │  │ • Pros/cons     │  │   price_stats() │  │ • review_       │    │
│  │ • Overall       │  │                 │  │                 │  │   velocity      │    │
│  │   recommendation│  │                 │  │                 │  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐                                               │
│  │ RecommendAgent  │  │ AttributeAgent  │                                               │
│  │                 │  │                 │                                               │
│  │ Generates:      │  │ Extracts:       │                                               │
│  │ • Top pick      │  │ • Direct attrs  │                                               │
│  │ • Alternatives  │  │   (price, stars)│                                               │
│  │ • Reasoning     │  │ • Parsed attrs  │                                               │
│  │ • Scores        │  │   (battery_hrs) │                                               │
│  │                 │  │ • Derived attrs │                                               │
│  │                 │  │   (value_score) │                                               │
│  └─────────────────┘  └─────────────────┘                                               │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 9: SYNTHESIS AGENT                                    │
│                              src/agents/synthesis_agent.py                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ INPUT                                                                            │    │
│  │                                                                                  │    │
│  │ SynthesisInput {                                                                 │    │
│  │   query: "Find Sony headphones under $200 with good reviews",                   │    │
│  │   intent: SEARCH,                                                               │    │
│  │   products: [...8 products...],                                                 │    │
│  │   format_hint: CARDS (auto-selected based on intent)                            │    │
│  │ }                                                                                │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ OUTPUT FORMAT SELECTION                                                         │    │
│  │                                                                                  │    │
│  │ ┌──────────────────┬────────────────────────────────────────────────────────┐  │    │
│  │ │ Intent           │ Format                                                  │  │    │
│  │ ├──────────────────┼────────────────────────────────────────────────────────┤  │    │
│  │ │ SEARCH           │ CARDS (product cards with image, price, rating)        │  │    │
│  │ │ COMPARE          │ COMPARISON (side-by-side table)                        │  │    │
│  │ │ ANALYZE          │ BULLET (key insights as bullet points)                 │  │    │
│  │ │ PRICE_CHECK      │ TABLE (price history table)                            │  │    │
│  │ │ TREND            │ BULLET (trending items list)                           │  │    │
│  │ │ RECOMMEND        │ TEXT + CARDS (recommendation with products)            │  │    │
│  │ └──────────────────┴────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │ RESPONSE GENERATION                                                             │    │
│  │                                                                                  │    │
│  │ SynthesisResult {                                                                │    │
│  │   response_text: "I found 8 Sony headphones under $200 with excellent          │    │
│  │                   reviews. The Sony WH-1000XM4 stands out as the top           │    │
│  │                   pick with 4.6★ rating and industry-leading noise             │    │
│  │                   cancellation...",                                             │    │
│  │   format: CARDS,                                                                │    │
│  │   products: [...],                                                              │    │
│  │   suggestions: [                                                                │    │
│  │     "Compare top 3 options",                                                    │    │
│  │     "Show price history for WH-1000XM4",                                       │    │
│  │     "Find wireless earbuds instead"                                            │    │
│  │   ],                                                                            │    │
│  │   confidence: 0.87                                                              │    │
│  │ }                                                                                │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 10: RESPONSE TO USER                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                  │    │
│  │  I found 8 Sony headphones under $200 with excellent reviews.                   │    │
│  │                                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │ 🎧 Sony WH-1000XM4                                    $348.00  ⭐ 4.6    │   │    │
│  │  │    Premium noise-cancelling wireless headphones                          │   │    │
│  │  │    Best for: Audiophiles, frequent travelers                             │   │    │
│  │  ├──────────────────────────────────────────────────────────────────────────┤   │    │
│  │  │ 🎧 Sony WH-CH720N                                     $148.00  ⭐ 4.4    │   │    │
│  │  │    Lightweight wireless headphones with ANC                              │   │    │
│  │  │    Best for: Casual listeners, commuters                                 │   │    │
│  │  ├──────────────────────────────────────────────────────────────────────────┤   │    │
│  │  │ 🎧 Sony MDR-7506                                      $89.99   ⭐ 4.7    │   │    │
│  │  │    Professional studio monitor headphones                                │   │    │
│  │  │    Best for: Studio monitoring, mixing                                   │   │    │
│  │  └──────────────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                                  │    │
│  │  💡 Suggestions:                                                                │    │
│  │  • Compare top 3 options                                                        │    │
│  │  • Show price history for WH-1000XM4                                           │    │
│  │  • Find wireless earbuds instead                                               │    │
│  │                                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Access Summary Table

| Stage | Database | Access Method | Fields Used |
|-------|----------|---------------|-------------|
| **Search - Keyword** | Elasticsearch | `keyword_search()` | Search: `title`, `title.autocomplete`, `short_title`, `brand`, `product_type`, `genAI_summary`, `chunk_description`, `chunk_features` |
| | | | Filter: `brand.keyword`, `category_name.keyword`, `price`, `stars` |
| | | | Return: `asin`, `title`, `price`, `stars`, `brand`, `category_name`, `category_level1`, `img_url`, `genAI_summary`, `genAI_best_for` |
| **Search - Semantic** | Qdrant | `semantic_search()` | Filter: `brand`, `category_name`, `price`, `stars` |
| | | | Return (payload): `asin`, `parent_asin`, `title`, `price`, `stars`, `brand`, `category_name`, `category_level1`, `img_url`, `genAI_summary`, `genAI_best_for`, `node_type` |
| **Search - Hybrid** | ES + Qdrant | Both above | Combined fields from both |
| **Trend** | PostgreSQL | `get_trending_products()` | `asin`, `title`, `brand`, `price`, `stars`, `reviews_count`, `bought_in_last_month`, `trend_score`, `review_velocity` |
| **Price** | PostgreSQL | `get_price_history()` | `price`, `original_price`, `discount_percentage`, `recorded_at` |
| **Analysis** | PostgreSQL | `get_product_reviews()` | `title`, `body`, `rating`, `author_name`, `verified_purchase`, `review_date`, `helpful_votes`, `sentiment_label` |
| **Product Details** | PostgreSQL | `get_product()` | All 70+ columns including all `genAI_*` fields |

---

## Query Type to Search Strategy Mapping

| Query Type | Example | Strategy | Keyword Weight | Semantic Weight |
|------------|---------|----------|----------------|-----------------|
| MODEL_NUMBER | "WH-1000XM5" | Keyword | 0.85 | 0.15 |
| BRAND_MODEL | "Sony WH-1000XM5" | Keyword/Hybrid | 0.80 | 0.20 |
| SHORT_TITLE | "wireless headphones" | Hybrid | 0.65 | 0.35 |
| GENERIC | "good laptop for travel" | Semantic | 0.35 | 0.65 |
| SECTION | "battery life reviews" | Section | 0.30 | 0.70 |

---

## Intent to Agent Routing

| Intent | Primary Agent | Supporting Agents | Database Access |
|--------|---------------|-------------------|-----------------|
| SEARCH | SearchAgent | - | ES, Qdrant |
| COMPARE | SearchAgent | CompareAgent, AttributeAgent | ES, Qdrant, PG |
| ANALYZE | SearchAgent | AnalysisAgent | ES, Qdrant, PG (reviews) |
| PRICE_CHECK | SearchAgent | PriceAgent | ES, Qdrant, PG (price_history) |
| TREND | TrendAgent | - | PG (product_trends) |
| RECOMMEND | SearchAgent | RecommendAgent | ES, Qdrant |
| GREETING | GeneralAgent | - | None |
| HELP | GeneralAgent | - | None |
| COMPOUND | Multiple | Based on decomposition | Varies |

---

## Detailed Flow: Product Spec Query

**Example Query:** "What are the specs of Sony WH-1000XM5?"

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    QUERY: "What are the specs of Sony WH-1000XM5?"                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INTENT AGENT                                                                    │
│                                                                                          │
│ Detected:                                                                                │
│ • primary_intent: ANALYZE (detected "specs" pattern)                                     │
│ • entities.products: ["Sony WH-1000XM5"]                                                │
│ • entities.brands: ["Sony"]                                                             │
│ • entities.attributes: ["spec"]                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SUPERVISOR AGENT                                                                │
│                                                                                          │
│ ExecutionPlan:                                                                           │
│   steps: [                                                                               │
│     { agent: "search", description: "Find Sony WH-1000XM5" },                           │
│     { agent: "attribute", description: "Extract specs", depends_on: ["search"] },       │
│     { agent: "analysis", description: "Analyze specs", depends_on: ["attribute"] }      │
│   ]                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SEARCH AGENT                                                                    │
│                                                                                          │
│ QueryAnalyzer detects: BRAND_MODEL → Keyword search priority                            │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ ELASTICSEARCH QUERY                                                              │     │
│ │ Search fields: title^10, short_title^8, brand^5                                 │     │
│ │ Filter: brand.keyword = "Sony"                                                  │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ Returns product with full data from ES index                                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: ATTRIBUTE AGENT                                                                 │
│ src/agents/attribute_agent.py                                                           │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ SPECIFICATION EXTRACTION                                                         │     │
│ │                                                                                   │     │
│ │ Data Sources (combined into spec_text):                                          │     │
│ │ • product.title                                                                  │     │
│ │ • product.description                                                            │     │
│ │ • product.specifications (JSONB from PostgreSQL)                                 │     │
│ │ • product.features (JSONB from PostgreSQL)                                       │     │
│ │ • product.bullet_points                                                          │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ EXTRACTED ATTRIBUTES (using regex patterns)                                      │     │
│ │                                                                                   │     │
│ │ Basic:                         Parsed Specs:                                     │     │
│ │ • price: $348.00              • battery_life: 30 hours (normalized)             │     │
│ │ • rating: 4.6 stars           • weight: 250g → 0.25 kg (normalized)             │     │
│ │ • review_count: 12,543        • dimensions: parsed if available                  │     │
│ │ • brand: Sony                 • storage: N/A for headphones                      │     │
│ │                                                                                   │     │
│ │ Derived Metrics:                                                                 │     │
│ │ • value_score: (rating / (price/10)) * 10 = 13.2                                │     │
│ │ • popularity_score: rating * log10(reviews) = 18.8                              │     │
│ │ • price_per_rating: $75.65/star                                                 │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ Output: AttributeExtractionOutput with ProductAttributes                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: POSTGRESQL ENRICHMENT (if needed for full specs)                               │
│ src/tools/postgres_tools.py                                                             │
│                                                                                          │
│ get_product(asin) → SELECT * FROM products WHERE asin = $1                              │
│                                                                                          │
│ Returns ALL 70+ columns including:                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ • specifications (JSONB) - Full technical specs                                  │     │
│ │ • features (JSONB) - Feature list                                               │     │
│ │ • technical_details (JSONB) - Detailed technical info                           │     │
│ │ • genAI_specs_summary - AI-generated specs summary                              │     │
│ │ • genAI_specs_comparison_ready - Normalized for comparison                      │     │
│ │ • genAI_specs_limitations - Known limitations                                   │     │
│ │ • genAI_technology_explained - Technology breakdown                             │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: SYNTHESIS AGENT                                                                 │
│                                                                                          │
│ Format: BULLET (specs work best as bullet points)                                       │
│                                                                                          │
│ Response:                                                                                │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ **Sony WH-1000XM5 Specifications**                                               │     │
│ │                                                                                   │     │
│ │ **Audio**                                                                        │     │
│ │ • Driver: 30mm dome type                                                         │     │
│ │ • Frequency Response: 4Hz-40,000Hz                                              │     │
│ │ • Noise Cancellation: Industry-leading ANC                                      │     │
│ │                                                                                   │     │
│ │ **Battery**                                                                      │     │
│ │ • Battery Life: 30 hours (ANC on)                                               │     │
│ │ • Quick Charge: 3 min charge = 3 hours playback                                 │     │
│ │                                                                                   │     │
│ │ **Physical**                                                                     │     │
│ │ • Weight: 250g                                                                   │     │
│ │ • Connectivity: Bluetooth 5.2, 3.5mm jack                                       │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary for Spec Query

| Stage | Database | Fields Accessed |
|-------|----------|-----------------|
| Search | Elasticsearch | `title`, `brand`, `asin`, basic fields |
| Attribute | ES/Qdrant payload | `title`, `description`, `specifications`, `features` |
| Enrichment | PostgreSQL | Full `specifications` (JSONB), `genAI_specs_*` fields |

---

## Detailed Flow: Compare Products Specs

**Example Query:** "Compare specs of Sony WH-1000XM5 vs Bose QC45"

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                QUERY: "Compare specs of Sony WH-1000XM5 vs Bose QC45"                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INTENT AGENT                                                                    │
│                                                                                          │
│ Detected:                                                                                │
│ • primary_intent: COMPARE (detected "vs" and "compare" patterns)                        │
│ • entities.products: ["Sony WH-1000XM5", "Bose QC45"]                                   │
│ • entities.brands: ["Sony", "Bose"]                                                     │
│ • entities.attributes: ["specs"]                                                        │
│ • complexity: SIMPLE (single intent with multiple products)                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SUPERVISOR AGENT                                                                │
│                                                                                          │
│ ExecutionPlan:                                                                           │
│   steps: [                                                                               │
│     { agent: "search", description: "Find both products" },                             │
│     { agent: "attribute", description: "Extract specs", depends_on: ["search"] },       │
│     { agent: "compare", description: "Compare specs", depends_on: ["attribute"] }       │
│   ]                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SEARCH AGENT (executed for each product)                                        │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ Search 1: "Sony WH-1000XM5"                                                      │     │
│ │ • QueryType: MODEL_NUMBER (detected WH-1000XM5)                                 │     │
│ │ • Strategy: KEYWORD (0.85 keyword, 0.15 semantic)                               │     │
│ │ • ES Query: title^10, brand^5, filter: brand.keyword="Sony"                     │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ Search 2: "Bose QC45"                                                            │     │
│ │ • QueryType: BRAND_MODEL                                                        │     │
│ │ • Strategy: KEYWORD (0.80 keyword, 0.20 semantic)                               │     │
│ │ • ES Query: title^10, brand^5, filter: brand.keyword="Bose"                     │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ Returns: 2 products with full payload data                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: ATTRIBUTE AGENT                                                                 │
│ src/agents/attribute_agent.py                                                           │
│                                                                                          │
│ For EACH product, extracts:                                                              │
│                                                                                          │
│ ┌──────────────────────────────────┬──────────────────────────────────┐                 │
│ │ Sony WH-1000XM5                  │ Bose QC45                        │                 │
│ ├──────────────────────────────────┼──────────────────────────────────┤                 │
│ │ price: $348.00                   │ price: $279.00                   │                 │
│ │ rating: 4.6 stars                │ rating: 4.5 stars                │                 │
│ │ review_count: 12,543             │ review_count: 8,234              │                 │
│ │ battery_life: 30 hours           │ battery_life: 24 hours           │                 │
│ │ weight: 250g (0.25 kg)           │ weight: 238g (0.238 kg)          │                 │
│ │ value_score: 13.2                │ value_score: 16.1                │                 │
│ │ popularity_score: 18.8           │ popularity_score: 17.6           │                 │
│ └──────────────────────────────────┴──────────────────────────────────┘                 │
│                                                                                          │
│ Output: AttributeExtractionOutput { products: [ProductAttributes × 2] }                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: COMPARE AGENT                                                                   │
│ src/agents/compare_agent.py                                                             │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ STEP 5.1: BUILD COMPARISON CONTEXT                                               │     │
│ │                                                                                   │     │
│ │ For each product, collects from payload:                                         │     │
│ │ • title, brand, price, stars                                                     │     │
│ │ • genAI_best_for (from ES/Qdrant)                                               │     │
│ │ • genAI_pros, genAI_cons (from ES/Qdrant)                                       │     │
│ │ • genAI_key_capabilities (from ES/Qdrant)                                       │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ STEP 5.2: BUILD COMPARISON MATRIX                                                │     │
│ │                                                                                   │     │
│ │ Attributes compared (with weights):                                              │     │
│ │ ┌────────────────────┬────────┬───────────────┬───────────────────────────────┐ │     │
│ │ │ Attribute          │ Weight │ Lower=Better? │ Purpose                       │ │     │
│ │ ├────────────────────┼────────┼───────────────┼───────────────────────────────┤ │     │
│ │ │ price              │ 0.25   │ Yes           │ Cost comparison               │ │     │
│ │ │ rating             │ 0.25   │ No            │ Quality indicator             │ │     │
│ │ │ value_score        │ 0.20   │ No            │ Bang for buck                 │ │     │
│ │ │ review_count       │ 0.15   │ No            │ Popularity/trust              │ │     │
│ │ │ popularity_score   │ 0.15   │ No            │ Market position               │ │     │
│ │ └────────────────────┴────────┴───────────────┴───────────────────────────────┘ │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ STEP 5.3: NORMALIZE & SCORE                                                      │     │
│ │                                                                                   │     │
│ │ For each attribute:                                                              │     │
│ │ 1. Get values: Sony=$348, Bose=$279                                             │     │
│ │ 2. Normalize to 0-1: Sony=0.0, Bose=1.0 (lower is better for price)            │     │
│ │ 3. Identify winner: Bose (for price)                                            │     │
│ │                                                                                   │     │
│ │ Calculate overall scores:                                                        │     │
│ │ • Sony: (0.0×0.25) + (1.0×0.25) + (0.0×0.20) + (1.0×0.15) + (1.0×0.15) = 55.0  │     │
│ │ • Bose: (1.0×0.25) + (0.0×0.25) + (1.0×0.20) + (0.0×0.15) + (0.0×0.15) = 45.0  │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ STEP 5.4: LLM COMPARISON (Ollama)                                                │     │
│ │                                                                                   │     │
│ │ POST /generate with comparison prompt                                            │     │
│ │                                                                                   │     │
│ │ Extracts from LLM response:                                                      │     │
│ │ • KEY DIFFERENCES (parsed from bullet list)                                      │     │
│ │ • WINNER + WINNER REASON                                                         │     │
│ │ • BEST VALUE + VALUE REASON                                                      │     │
│ │ • RECOMMENDATION                                                                 │     │
│ │ • SUMMARY                                                                        │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│ Output: ComparisonResult                                                                │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ {                                                                                │     │
│ │   comparison_matrix: { product_ids, attributes, overall_scores, winner_id },    │     │
│ │   attribute_winners: { price: "Bose", rating: "Sony", battery: "Sony", ... },   │     │
│ │   winner: ProductComparison(Sony WH-1000XM5),                                   │     │
│ │   winner_reason: "Superior ANC and battery life justify premium",               │     │
│ │   best_value: ProductComparison(Bose QC45),                                     │     │
│ │   key_differences: [                                                             │     │
│ │     "Sony has 6 hours more battery (30 vs 24)",                                 │     │
│ │     "Bose is $69 cheaper",                                                      │     │
│ │     "Sony has more reviews (12,543 vs 8,234)"                                   │     │
│ │   ],                                                                             │     │
│ │   recommendation: "Sony for frequent travelers, Bose for budget-conscious"      │     │
│ │ }                                                                                │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: SYNTHESIS AGENT                                                                 │
│                                                                                          │
│ Format: COMPARISON (side-by-side table)                                                 │
│                                                                                          │
│ Response:                                                                                │
│ ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│ │ ## Sony WH-1000XM5 vs Bose QC45                                                  │     │
│ │                                                                                   │     │
│ │ | Spec           | Sony WH-1000XM5    | Bose QC45         | Winner      |        │     │
│ │ |----------------|--------------------| ------------------|-------------|        │     │
│ │ | Price          | $348               | $279              | 🏆 Bose     |        │     │
│ │ | Rating         | 4.6★               | 4.5★              | 🏆 Sony     |        │     │
│ │ | Battery        | 30 hours           | 24 hours          | 🏆 Sony     |        │     │
│ │ | Weight         | 250g               | 238g              | 🏆 Bose     |        │     │
│ │ | Reviews        | 12,543             | 8,234             | 🏆 Sony     |        │     │
│ │                                                                                   │     │
│ │ **Winner:** Sony WH-1000XM5 (Score: 55/100)                                      │     │
│ │ **Best Value:** Bose QC45                                                        │     │
│ │                                                                                   │     │
│ │ **Recommendation:**                                                              │     │
│ │ Choose Sony for best-in-class ANC and longer battery.                           │     │
│ │ Choose Bose for comfort and value.                                              │     │
│ └─────────────────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary for Comparison Query

| Stage | Database | Fields Accessed |
|-------|----------|-----------------|
| Search | Elasticsearch | `title`, `brand`, `price`, `stars`, `genAI_*` fields |
| Attribute | ES payload | `price`, `stars`, `reviews_count`, calculated `value_score` |
| Compare | ES/Qdrant | `genAI_pros`, `genAI_cons`, `genAI_best_for`, `genAI_key_capabilities` |
| Compare (LLM) | None | Uses extracted data, generates via Ollama |

### Comparison Scoring Algorithm

```
Overall Score = Σ (normalized_value × weight)

Where:
- normalized_value = (value - min) / (max - min)  [0 to 1]
- For "lower is better" attributes: normalized = 1 - normalized

Weights:
- price: 25%
- rating: 25%
- value_score: 20%
- review_count: 15%
- popularity_score: 15%
```

### Key Differences Generation

The `generate_key_differences()` method automatically creates difference statements:

```python
# For price attribute
if attr == "price":
    diff_pct = ((max_val - min_val) / min_val) * 100
    → "Bose is $69 (20%) cheaper than Sony"

# For rating attribute
if attr == "rating":
    → "Sony has higher rating (4.6) vs Bose (4.5)"

# For review_count
if attr == "review_count":
    → "Sony has more reviews (12,543) vs Bose (8,234)"
```
