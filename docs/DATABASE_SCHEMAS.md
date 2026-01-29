# Product Intelligence System - Database Schemas

## Overview

The Product Intelligence System uses a 3-tier storage architecture optimized for different query patterns:

| Storage | Technology | Purpose | Query Target |
|---------|------------|---------|--------------|
| **Vector Store** | Qdrant | Semantic search, filtering, parent-child hierarchy | 80%+ of queries |
| **Search Engine** | Elasticsearch | Keyword search, autocomplete, aggregations | Hybrid search support |
| **Relational DB** | PostgreSQL | Source of truth, analytics, time-series | Complex aggregations |

**Design Principle:** Qdrant = Speed (<300ms), PostgreSQL = Depth (500ms-2s)

---

## 1. Qdrant Vector Store

### 1.1 Collection: `products`

#### Configuration

```python
collection_config = {
    "vectors": {
        "size": 1024,  # BGE-Large-EN-v1.5
        "distance": "Cosine",
        "on_disk": True,
    },
    "optimizers_config": {
        "memmap_threshold": 20000,
        "indexing_threshold": 10000,
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000,
    },
}
```

### 1.2 Parent Node Schema

**Purpose:** Answer basic questions (What is this? What's it for? Who is it for?)

```yaml
qdrant_parent_payload:
  # ═══════════════════════════════════════════════════════════════
  # IDENTITY (What is this?)
  # ═══════════════════════════════════════════════════════════════
  asin: "B09XS7JWHH"                          # Primary key
  title: "Sony WH-1000XM5 Wireless..."        # Original title
  short_title: "Sony WH-1000XM5 Headphones"   # GenAI: Cleaned version
  brand: "Sony"                               # Filter + Display
  product_type: "Wireless Headphones"         # GenAI: Clean product type
  product_type_keywords: ["headphones", "wireless", "bluetooth", "noise-cancelling"]

  # ═══════════════════════════════════════════════════════════════
  # CATEGORY (Filtering)
  # ═══════════════════════════════════════════════════════════════
  category_level1: "Electronics"              # Indexed filter
  category_level2: "Headphones"               # Indexed filter
  category_level3: "Over-Ear"                 # Indexed filter

  # ═══════════════════════════════════════════════════════════════
  # DISPLAY FIELDS (Search results)
  # ═══════════════════════════════════════════════════════════════
  price: 349.99                               # Filter + Display
  list_price: 399.99                          # Display (show discount)
  stars: 4.7                                  # Filter + Display
  reviews_count: 12543                        # Display + Sort
  img_url: "https://..."                      # Display
  is_best_seller: true                        # Filter + Badge
  availability: "in_stock"                    # Filter + Display

  # ═══════════════════════════════════════════════════════════════
  # GENAI QUICK ANSWERS (Answer basic questions without child lookup)
  # ═══════════════════════════════════════════════════════════════
  genAI_summary: "Premium wireless headphones with industry-leading noise cancellation..."
  genAI_primary_function: "Noise-cancelling headphones for immersive audio"
  genAI_best_for: "Best for: travelers, remote workers, audiophiles"
  genAI_use_cases: ["music", "calls", "travel", "commuting", "focus work"]
  genAI_target_audience: ["frequent travelers", "remote workers", "audiophiles"]
  genAI_key_capabilities: ["30h battery", "Industry-leading ANC", "Multipoint", "Hi-Res Audio"]
  genAI_unique_selling_points: "Best-in-class noise cancellation with all-day comfort"

  # ═══════════════════════════════════════════════════════════════
  # COMPUTED SCORES (Pre-calculated for sorting/ranking)
  # ═══════════════════════════════════════════════════════════════
  popularity_score: 95                        # For trending/popular sorts
  trending_rank: 3                            # Within category
  price_percentile: 85                        # For "is this expensive?" questions
  genAI_value_score: 8                        # 1-10 value rating

  # ═══════════════════════════════════════════════════════════════
  # METADATA
  # ═══════════════════════════════════════════════════════════════
  node_type: "parent"
  indexed_at: "2026-01-22T..."
```

#### Parent Node Indexed Fields

```yaml
qdrant_parent_indexes:
  - asin (keyword)
  - brand (keyword)
  - product_type (keyword)
  - category_level1 (keyword)
  - category_level2 (keyword)
  - category_level3 (keyword)
  - price (float)
  - stars (float)
  - reviews_count (integer)
  - is_best_seller (bool)
  - availability (keyword)
  - popularity_score (integer)
  - trending_rank (integer)
  - node_type (keyword)
```

### 1.3 Child Node Schemas

#### Child 1: `description` - Product Details

**Purpose:** Answer "Tell me more about this product"

```yaml
child_description:
  node_type: "child"
  parent_asin: "B09XS7JWHH"
  section: "description"
  content_preview: "First 200 chars..."

  # GenAI-enriched fields
  genAI_detailed_description: "2-3 paragraph detailed description"
  genAI_how_it_works: "Explanation of how the product works"
  genAI_whats_included: ["Headphones", "USB-C cable", "Audio cable", "Carrying case"]
  genAI_materials: "Premium protein leather earcups, memory foam padding"

  # Inherited filters
  category_level1: "Electronics"
  brand: "Sony"
  price: 349.99
  stars: 4.7
```

| Question | Field |
|----------|-------|
| "Tell me more about this product" | genAI_detailed_description |
| "How does noise cancellation work?" | genAI_how_it_works |
| "What comes in the box?" | genAI_whats_included |
| "What material is it made of?" | genAI_materials |

#### Child 2: `features` - Capabilities & Functions

**Purpose:** Answer "What makes this special?"

```yaml
child_features:
  node_type: "child"
  parent_asin: "B09XS7JWHH"
  section: "features"
  content_preview: "Industry-leading noise cancellation..."

  genAI_features_detailed: [
    {
      "feature": "Active Noise Cancellation",
      "description": "8 microphones and 2 processors analyze ambient sound",
      "benefit": "Blocks up to 99% of outside noise"
    },
    {
      "feature": "30-Hour Battery",
      "description": "Long-lasting with quick charge support",
      "benefit": "3 min charge = 3 hours playback"
    }
  ]
  genAI_standout_features: ["Best-in-class ANC", "Multipoint connection", "Wear detection"]
  genAI_technology_explained: "Uses dual noise sensor technology with HD Noise Cancelling Processor QN1"
  genAI_feature_comparison: "Better ANC than Bose QC45, longer battery than AirPods Max"

  # Inherited filters
  category_level1: "Electronics"
  brand: "Sony"
  price: 349.99
  stars: 4.7
```

| Question | Field |
|----------|-------|
| "How good is the noise cancellation?" | genAI_features_detailed |
| "What makes these headphones special?" | genAI_standout_features |
| "What technology does it use?" | genAI_technology_explained |
| "How does this compare to Bose?" | genAI_feature_comparison |

#### Child 3: `specs` - Technical Specifications

**Purpose:** Answer "What's the battery life? Does it support X?"

```yaml
child_specs:
  node_type: "child"
  parent_asin: "B09XS7JWHH"
  section: "specs"
  content_preview: "Driver: 30mm, Bluetooth 5.2, 30h battery..."

  specs_searchable: "30mm driver Bluetooth 5.2 LDAC AAC SBC 30 hour battery 250g USB-C"

  specs_key: {
    "Battery Life": "30 hours",
    "Bluetooth": "5.2",
    "Codecs": ["LDAC", "AAC", "SBC"],
    "Weight": "250g",
    "Driver": "30mm",
    "ANC": "Yes",
    "Water Resistance": "No"
  }

  genAI_specs_summary: "Premium specs with Hi-Res Audio and LDAC support"
  genAI_specs_highlights: ["Hi-Res Audio certified", "LDAC codec", "30-hour battery"]
  genAI_specs_limitations: ["No water resistance", "No aptX support"]

  # Inherited filters
  category_level1: "Electronics"
  brand: "Sony"
  price: 349.99
  stars: 4.7
```

| Question | Field |
|----------|-------|
| "What's the battery life?" | specs_key["Battery Life"] |
| "Does it support LDAC?" | specs_key["Codecs"] |
| "How heavy is it?" | specs_key["Weight"] |
| "Any limitations?" | genAI_specs_limitations |

#### Child 4: `reviews` - User Feedback & Sentiment

**Purpose:** Answer "What do users say? Is it durable?"

```yaml
child_reviews:
  node_type: "child"
  parent_asin: "B09XS7JWHH"
  section: "reviews"
  content_preview: "Users praise exceptional noise cancellation..."

  genAI_sentiment_score: 0.82          # -1 to 1
  genAI_sentiment_label: "Very Positive"

  genAI_common_praises: [
    {"theme": "Noise Cancellation", "mentions": 3420, "sample": "Best ANC I've ever used"},
    {"theme": "Comfort", "mentions": 2890, "sample": "Can wear all day"},
    {"theme": "Sound Quality", "mentions": 2650, "sample": "Rich, balanced audio"},
    {"theme": "Battery Life", "mentions": 1890, "sample": "Lasts my whole work week"}
  ]

  genAI_common_complaints: [
    {"theme": "Price", "mentions": 1200, "sample": "Expensive but worth it"},
    {"theme": "Touch Controls", "mentions": 450, "sample": "Sometimes triggers accidentally"}
  ]

  genAI_durability_feedback: "Users report lasting 2-3 years with daily use"
  genAI_value_perception: "Most users feel price is justified by quality"
  genAI_top_helpful_review: "After 6 months of daily use..."

  # Inherited filters
  category_level1: "Electronics"
  brand: "Sony"
  price: 349.99
  stars: 4.7
```

| Question | Field |
|----------|-------|
| "What do people think of this?" | genAI_sentiment_label |
| "What do users like about it?" | genAI_common_praises |
| "What are the complaints?" | genAI_common_complaints |
| "Is it durable?" | genAI_durability_feedback |
| "Is it worth the price?" | genAI_value_perception |

#### Child 5: `use_cases` - Scenarios & Applications

**Purpose:** Answer "Is this good for flights? Who should buy?"

```yaml
child_use_cases:
  node_type: "child"
  parent_asin: "B09XS7JWHH"
  section: "use_cases"
  content_preview: "Perfect for flights, office, commuting..."

  genAI_use_case_scenarios: [
    {
      "scenario": "Long-haul flights",
      "why_good": "Blocks 99% of engine noise, 30h battery",
      "user_quote": "Game changer for my 14-hour flights"
    },
    {
      "scenario": "Work from home",
      "why_good": "Great for video calls, comfortable all-day",
      "user_quote": "Perfect for my 8-hour remote work days"
    }
  ]

  genAI_ideal_user_profiles: [
    {"profile": "Frequent Traveler", "match_score": 95},
    {"profile": "Remote Worker", "match_score": 90},
    {"profile": "Audiophile", "match_score": 85}
  ]

  genAI_not_recommended_for: [
    {"profile": "Gym/Sports", "reason": "Not sweat resistant"},
    {"profile": "Budget Buyers", "reason": "Premium price point"}
  ]

  genAI_problems_solved: [
    {"problem": "Can't focus in noisy environment", "solution": "Industry-leading ANC"},
    {"problem": "Headphones die mid-flight", "solution": "30-hour battery"}
  ]

  # Inherited filters
  category_level1: "Electronics"
  brand: "Sony"
  price: 349.99
  stars: 4.7
```

| Question | Field |
|----------|-------|
| "Is this good for flights?" | genAI_use_case_scenarios |
| "Best headphones for remote work?" | genAI_ideal_user_profiles |
| "Can I use these for gym?" | genAI_not_recommended_for |
| "What problems does this solve?" | genAI_problems_solved |

#### Child Node Indexed Fields

```yaml
qdrant_child_indexes:
  - node_type (keyword)
  - parent_asin (keyword)
  - section (keyword)
  - category_level1 (keyword)
  - brand (keyword)
  - price (float)
  - stars (float)
  - genAI_sentiment_score (float)
```

---

## 2. Elasticsearch Index

### 2.1 Index: `products`

#### Settings

```json
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "autocomplete_analyzer": {
          "type": "custom",
          "tokenizer": "autocomplete_tokenizer",
          "filter": ["lowercase"]
        },
        "product_search_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "porter_stem", "product_synonym"]
        }
      },
      "tokenizer": {
        "autocomplete_tokenizer": {
          "type": "edge_ngram",
          "min_gram": 2,
          "max_gram": 20,
          "token_chars": ["letter", "digit"]
        }
      },
      "filter": {
        "product_synonym": {
          "type": "synonym",
          "synonyms": [
            "headphones, earphones, earbuds",
            "laptop, notebook, computer",
            "phone, smartphone, mobile"
          ]
        }
      }
    }
  }
}
```

#### Mappings

```json
{
  "mappings": {
    "properties": {
      "asin": { "type": "keyword" },
      "title": {
        "type": "text",
        "analyzer": "product_search_analyzer",
        "fields": {
          "keyword": { "type": "keyword" },
          "autocomplete": {
            "type": "text",
            "analyzer": "autocomplete_analyzer",
            "search_analyzer": "standard"
          }
        }
      },
      "short_title": {
        "type": "text",
        "analyzer": "product_search_analyzer"
      },
      "brand": {
        "type": "text",
        "fields": { "keyword": { "type": "keyword" } }
      },
      "product_type": {
        "type": "text",
        "fields": { "keyword": { "type": "keyword" } }
      },
      "product_type_keywords": { "type": "keyword" },
      "description": {
        "type": "text",
        "analyzer": "product_search_analyzer"
      },
      "features_text": {
        "type": "text",
        "analyzer": "product_search_analyzer"
      },
      "specs_text": {
        "type": "text",
        "analyzer": "product_search_analyzer"
      },
      "genAI_summary": { "type": "text" },
      "genAI_best_for": { "type": "text" },
      "genAI_use_cases": { "type": "keyword" },
      "category_level1": { "type": "keyword" },
      "category_level2": { "type": "keyword" },
      "category_level3": { "type": "keyword" },
      "price": { "type": "float" },
      "list_price": { "type": "float" },
      "stars": { "type": "float" },
      "reviews_count": { "type": "integer" },
      "is_best_seller": { "type": "boolean" },
      "availability": { "type": "keyword" },
      "indexed_at": { "type": "date" }
    }
  }
}
```

### 2.2 Field Boosting Configuration

```python
ELASTICSEARCH_FIELD_BOOSTS = {
    "title": 4.0,
    "title.autocomplete": 2.5,
    "short_title": 3.0,
    "brand": 2.5,
    "product_type": 2.0,
    "product_type_keywords": 1.8,
    "genAI_summary": 1.5,
    "genAI_best_for": 1.5,
    "description": 1.0,
    "features_text": 1.2,
    "specs_text": 0.8,
}
```

### 2.3 Search Types Supported

| Search Type | ES Query Type | Fields Used |
|-------------|---------------|-------------|
| Autocomplete | prefix + edge_ngram | title.autocomplete |
| Exact Match | term query | asin, brand.keyword |
| Full-text | multi_match | title, description, features |
| Faceted | aggregations | category, brand, price ranges |
| Fuzzy | match + fuzziness | title, product_type |

---

## 3. PostgreSQL Database

### 3.1 Products Table (Source of Truth)

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) UNIQUE NOT NULL,

    -- Basic Info (from CSV)
    title TEXT NOT NULL,
    brand VARCHAR(255),
    price DECIMAL(10,2),
    list_price DECIMAL(10,2),
    stars DECIMAL(2,1),
    reviews_count INTEGER DEFAULT 0,
    bought_in_last_month INTEGER DEFAULT 0,
    is_best_seller BOOLEAN DEFAULT FALSE,

    -- Category
    category_name TEXT,
    category_level1 VARCHAR(255),
    category_level2 VARCHAR(255),
    category_level3 VARCHAR(255),

    -- URLs
    product_url TEXT,
    img_url TEXT,

    -- Scraped Content (from productURL)
    product_description TEXT,
    about_this_item JSONB,
    technical_details JSONB,
    specifications JSONB,
    additional_info JSONB,
    top_reviews JSONB,
    review_summary_raw JSONB,
    frequently_bought JSONB,
    availability VARCHAR(100),

    -- GenAI Enriched (full versions)
    genAI_summary TEXT,
    genAI_short_title VARCHAR(255),
    genAI_product_type VARCHAR(255),
    genAI_product_type_keywords JSONB,
    genAI_primary_function TEXT,
    genAI_detailed_description TEXT,
    genAI_how_it_works TEXT,
    genAI_whats_included JSONB,
    genAI_materials TEXT,

    genAI_features_detailed JSONB,
    genAI_standout_features JSONB,
    genAI_technology_explained TEXT,
    genAI_feature_comparison TEXT,

    genAI_pros JSONB,
    genAI_cons JSONB,
    genAI_sentiment_score DECIMAL(3,2),
    genAI_sentiment_label VARCHAR(50),
    genAI_common_praises JSONB,
    genAI_common_complaints JSONB,
    genAI_durability_feedback TEXT,
    genAI_value_perception TEXT,

    genAI_use_cases JSONB,
    genAI_best_for TEXT,
    genAI_target_audience JSONB,
    genAI_use_case_scenarios JSONB,
    genAI_ideal_user_profiles JSONB,
    genAI_not_recommended_for JSONB,
    genAI_problems_solved JSONB,

    genAI_key_capabilities JSONB,
    genAI_unique_selling_points TEXT,
    genAI_value_score INTEGER,
    genAI_quality_score INTEGER,

    -- Computed Metrics
    price_percentile DECIMAL(5,2),
    popularity_score INTEGER,
    trending_rank INTEGER,

    -- Metadata
    scraped_at TIMESTAMP WITH TIME ZONE,
    genAI_enriched_at TIMESTAMP WITH TIME ZONE,
    embedding_model VARCHAR(100),
    embedded_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_products_asin ON products(asin);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_category ON products(category_level1, category_level2);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_products_stars ON products(stars);
CREATE INDEX idx_products_bestseller ON products(is_best_seller);
CREATE INDEX idx_products_updated ON products(updated_at);
```

### 3.2 Brands Table (Aggregated Statistics)

```sql
CREATE TABLE brands (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    slug VARCHAR(255),
    product_count INTEGER DEFAULT 0,
    avg_rating DECIMAL(2,1),
    avg_price DECIMAL(10,2),
    price_min DECIMAL(10,2),
    price_max DECIMAL(10,2),
    total_reviews INTEGER DEFAULT 0,
    market_share DECIMAL(5,2),
    popularity_rank INTEGER,
    growth_rate DECIMAL(5,2),
    is_emerging BOOLEAN DEFAULT FALSE,
    categories JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_brands_name ON brands(name);
CREATE INDEX idx_brands_popularity ON brands(popularity_rank);
```

### 3.3 Categories Table (Aggregated Statistics)

```sql
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    level1 VARCHAR(255) NOT NULL,
    level2 VARCHAR(255),
    level3 VARCHAR(255),
    full_path TEXT,
    product_count INTEGER DEFAULT 0,
    avg_price DECIMAL(10,2),
    price_min DECIMAL(10,2),
    price_max DECIMAL(10,2),
    avg_rating DECIMAL(2,1),
    trending_features JSONB,
    declining_features JSONB,
    top_brands JSONB,
    emerging_brands JSONB,
    budget_threshold DECIMAL(10,2),
    premium_threshold DECIMAL(10,2),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(level1, level2, level3)
);

CREATE INDEX idx_categories_level1 ON categories(level1);
CREATE INDEX idx_categories_full_path ON categories(full_path);
```

### 3.4 Reviews Table

```sql
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL REFERENCES products(asin),
    review_id VARCHAR(50) UNIQUE,

    -- Review content
    title VARCHAR(500),
    body TEXT,
    rating DECIMAL(2, 1),

    -- Author
    author_name VARCHAR(255),
    author_id VARCHAR(50),
    verified_purchase BOOLEAN DEFAULT FALSE,

    -- Metadata
    review_date DATE,
    helpful_votes INTEGER DEFAULT 0,

    -- Analysis (computed)
    sentiment_score DECIMAL(3, 2),
    sentiment_label VARCHAR(20),
    aspects JSONB,  -- Extracted aspects with sentiments

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_reviews_asin ON reviews(asin);
CREATE INDEX idx_reviews_rating ON reviews(rating);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label);
```

### 3.5 Price History Table (Time-series)

```sql
CREATE TABLE price_history (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    list_price DECIMAL(10,2),
    discount_percentage DECIMAL(5,2),
    is_deal BOOLEAN DEFAULT FALSE,
    deal_type VARCHAR(50),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_price_history_asin_date ON price_history(asin, recorded_at DESC);
```

### 3.6 Product Relationships Table

```sql
CREATE TABLE product_relationships (
    id SERIAL PRIMARY KEY,
    source_asin VARCHAR(20) NOT NULL,
    target_asin VARCHAR(20) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,  -- 'accessory', 'alternative', 'upgrade', 'bundle'
    confidence DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_asin, target_asin, relationship_type)
);

CREATE INDEX idx_relationships_source ON product_relationships(source_asin, relationship_type);
```

### 3.7 Product Trends Table

```sql
CREATE TABLE product_trends (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL REFERENCES products(asin),
    date DATE NOT NULL,

    -- Trend metrics
    review_velocity INTEGER,  -- Reviews in last 7 days
    rating_change DECIMAL(3, 2),  -- Change from 30 days ago
    price_change DECIMAL(10, 2),
    rank_in_category INTEGER,

    -- Computed trend score
    trend_score DECIMAL(5, 2),

    UNIQUE (asin, date)
);

CREATE INDEX idx_trends_date ON product_trends(date);
CREATE INDEX idx_trends_score ON product_trends(trend_score DESC);
```

### 3.8 Product Features Table (for trend analysis)

```sql
CREATE TABLE product_features (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    feature_value TEXT,
    feature_category VARCHAR(100)
);

CREATE INDEX idx_features_name ON product_features(feature_name);
CREATE INDEX idx_features_category ON product_features(feature_category);
```

### 3.9 Useful Views

#### Product Summary View

```sql
CREATE VIEW product_summary AS
SELECT
    p.asin,
    p.title,
    p.brand,
    p.price,
    p.stars,
    p.reviews_count,
    p.category_level1,
    p.category_level2,
    p.is_best_seller,
    b.avg_rating as brand_avg_rating,
    COALESCE(
        (SELECT trend_score FROM product_trends
         WHERE asin = p.asin ORDER BY date DESC LIMIT 1),
        0
    ) as trend_score,
    COALESCE(
        (SELECT price FROM price_history
         WHERE asin = p.asin ORDER BY recorded_at DESC LIMIT 1),
        p.price
    ) as latest_price
FROM products p
LEFT JOIN brands b ON p.brand = b.name;
```

#### Category Overview View

```sql
CREATE VIEW category_overview AS
SELECT
    c.id,
    c.level1,
    c.level2,
    c.full_path,
    c.product_count,
    c.avg_price,
    c.avg_rating,
    c.top_brands,
    c.trending_features
FROM categories c;
```

---

## 4. Storage Decision Matrix

### When to Use Each Storage

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STORAGE DECISION CRITERIA                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Store in QDRANT if:                    Store in POSTGRESQL if:                 │
│  ══════════════════                     ════════════════════════                │
│                                                                                 │
│  ✓ Needs semantic/vector search         ✓ Needs exact SQL queries              │
│  ✓ HIGH frequency access (>60%)         ✓ LOW frequency access (<20%)          │
│  ✓ Needed for search result display     ✓ Needed for detailed view only        │
│  ✓ Used for filtering during search     ✓ Aggregation/GROUP BY queries         │
│  ✓ Must respond in <100-300ms           ✓ Can tolerate 500ms-2s                 │
│  ✓ Relatively static data               ✓ Time-series/historical data          │
│  ✓ Per-product data                     ✓ Cross-product relationships          │
│                                         ✓ Source of truth (complete record)    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Query-to-Storage Routing

| Query Pattern | Primary Storage | Secondary Storage | Target Latency |
|---------------|-----------------|-------------------|----------------|
| "What is [product]?" | Qdrant Parent | - | <100ms |
| "Find [product type]" | Qdrant Parent | ES (keyword) | <100ms |
| "[product] under $X" | Qdrant Parent | - | <100ms |
| "Tell me more about [product]" | Qdrant Child (description) | Parent | <300ms |
| "What do users say about [product]?" | Qdrant Child (reviews) | - | <300ms |
| "Compare [product A] vs [product B]" | Qdrant Child (specs) | PostgreSQL | <500ms |
| "Trending products in [category]" | PostgreSQL | - | <800ms |
| "Price history of [product]" | PostgreSQL | - | <400ms |
| "Brand statistics for [brand]" | PostgreSQL | - | <400ms |

---

## 5. References

- Proposal Document: [PROPOSAL.md](./PROPOSAL.md)
- Design Document: [DESIGN.md](./DESIGN.md)
- Architecture Flow: [architecture.md](ARCHITECTURE.md)
- DB Design Proposal: [db-design-proposal.md](./db-design-proposal.md)
