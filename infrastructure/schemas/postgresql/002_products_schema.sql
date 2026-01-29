-- =============================================================================
-- PostgreSQL Schema for Product Intelligence System
-- =============================================================================
-- This schema supports the 7-agent architecture and 6 scenario categories:
-- - Discovery, Comparison, Analysis, Price Intelligence, Trend Analysis, Recommendation
-- =============================================================================

-- Enable ltree extension for hierarchical category paths
CREATE EXTENSION IF NOT EXISTS ltree;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Products: Main product data (source of truth)
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    brand VARCHAR(255),

    -- Identity (GenAI cleaned)
    short_title VARCHAR(500),
    product_type VARCHAR(255),
    product_type_keywords JSONB,

    -- Pricing
    price DECIMAL(10, 2),
    list_price DECIMAL(10, 2),
    original_price DECIMAL(10, 2),  -- Keep for backward compatibility
    currency VARCHAR(3) DEFAULT 'USD',

    -- Ratings
    stars DECIMAL(2, 1),
    reviews_count INTEGER DEFAULT 0,
    bought_in_last_month INTEGER DEFAULT 0,

    -- Categorization
    category_name TEXT,
    category_level1 VARCHAR(255),
    category_level2 VARCHAR(255),
    category_level3 VARCHAR(255),
    category_id INTEGER,

    -- Flags
    is_best_seller BOOLEAN DEFAULT FALSE,
    is_amazon_choice BOOLEAN DEFAULT FALSE,
    prime_eligible BOOLEAN DEFAULT FALSE,
    availability VARCHAR(100),

    -- Content (original)
    product_description TEXT,
    features JSONB,
    specifications JSONB,

    -- Scraped Content (from productURL)
    about_this_item JSONB,
    technical_details JSONB,
    additional_info JSONB,
    top_reviews JSONB,
    review_summary_raw JSONB,
    frequently_bought JSONB,

    -- URLs
    product_url TEXT,
    img_url TEXT,

    -- GenAI Parent Fields (quick answers)
    genAI_summary TEXT,
    genAI_primary_function TEXT,
    genAI_best_for TEXT,
    genAI_use_cases JSONB,
    genAI_target_audience JSONB,
    genAI_key_capabilities JSONB,
    genAI_unique_selling_points TEXT,
    genAI_value_score INTEGER,

    -- GenAI Description Fields
    genAI_detailed_description TEXT,
    genAI_how_it_works TEXT,
    genAI_whats_included JSONB,
    genAI_materials TEXT,

    -- GenAI Features Fields
    genAI_features_detailed JSONB,
    genAI_standout_features JSONB,
    genAI_technology_explained TEXT,
    genAI_feature_comparison TEXT,

    -- GenAI Specs Fields
    genAI_specs_summary TEXT,
    genAI_specs_comparison_ready JSONB,
    genAI_specs_limitations JSONB,

    -- GenAI Review Analysis Fields
    genAI_sentiment_score DECIMAL(3, 2),
    genAI_sentiment_label VARCHAR(50),
    genAI_common_praises JSONB,
    genAI_common_complaints JSONB,
    genAI_durability_feedback TEXT,
    genAI_value_for_money_feedback TEXT,

    -- GenAI Use Cases Fields
    genAI_use_case_scenarios JSONB,
    genAI_ideal_user_profiles JSONB,
    genAI_not_recommended_for JSONB,
    genAI_problems_solved JSONB,

    -- GenAI Pros/Cons
    genAI_pros JSONB,
    genAI_cons JSONB,

    -- Computed Metrics
    price_percentile DECIMAL(5, 2),
    popularity_score INTEGER,
    trending_rank INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_scraped_at TIMESTAMP,
    genAI_enriched_at TIMESTAMP,
    embedding_model VARCHAR(100),
    embedded_at TIMESTAMP
);

-- Indexes for products table
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_level1, category_level2);
CREATE INDEX IF NOT EXISTS idx_products_price ON products(price);
CREATE INDEX IF NOT EXISTS idx_products_stars ON products(stars);
CREATE INDEX IF NOT EXISTS idx_products_bestseller ON products(is_best_seller);
CREATE INDEX IF NOT EXISTS idx_products_updated ON products(updated_at);

-- Categories: Hierarchical category tree
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    parent_id INTEGER REFERENCES categories(id),
    level INTEGER NOT NULL DEFAULT 1,
    path LTREE,
    product_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_categories_path ON categories USING GIST (path);

-- Add foreign key from products to categories after categories table exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_products_category'
    ) THEN
        ALTER TABLE products
            ADD CONSTRAINT fk_products_category
            FOREIGN KEY (category_id) REFERENCES categories(id)
            ON DELETE SET NULL;
    END IF;
END $$;

-- Brands: Brand statistics and metadata
CREATE TABLE IF NOT EXISTS brands (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    logo_url TEXT,

    -- Aggregated stats (updated periodically)
    product_count INTEGER DEFAULT 0,
    avg_rating DECIMAL(3, 2),
    avg_price DECIMAL(10, 2),
    review_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_brands_name ON brands(name);

-- Reviews: Product review data for analysis
CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL,
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
    aspects JSONB,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT fk_reviews_product
        FOREIGN KEY (asin) REFERENCES products(asin)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_asin ON reviews(asin);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);

-- =============================================================================
-- ANALYTICS TABLES (for Trend and Price agents)
-- =============================================================================

-- Price History: Track price changes over time
CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    original_price DECIMAL(10, 2),
    discount_percentage DECIMAL(5, 2),
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_price_history_product
        FOREIGN KEY (asin) REFERENCES products(asin)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_price_history_asin_date ON price_history(asin, recorded_at DESC);

-- Product Trends: Trend metrics for products
CREATE TABLE IF NOT EXISTS product_trends (
    id SERIAL PRIMARY KEY,
    asin VARCHAR(20) NOT NULL,
    date DATE NOT NULL,

    -- Trend metrics
    review_velocity INTEGER,
    rating_change DECIMAL(3, 2),
    price_change DECIMAL(10, 2),
    rank_in_category INTEGER,

    -- Computed trend score
    trend_score DECIMAL(5, 2),

    UNIQUE (asin, date),

    CONSTRAINT fk_product_trends_product
        FOREIGN KEY (asin) REFERENCES products(asin)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trends_date ON product_trends(date);
CREATE INDEX IF NOT EXISTS idx_trends_score ON product_trends(trend_score DESC);

-- Category Stats: Daily category statistics
CREATE TABLE IF NOT EXISTS category_stats (
    id SERIAL PRIMARY KEY,
    category_id INTEGER NOT NULL,
    date DATE NOT NULL,

    -- Statistics
    product_count INTEGER,
    avg_price DECIMAL(10, 2),
    min_price DECIMAL(10, 2),
    max_price DECIMAL(10, 2),
    avg_rating DECIMAL(3, 2),
    total_reviews INTEGER,

    -- Top performers
    top_products JSONB,
    trending_products JSONB,

    UNIQUE (category_id, date),

    CONSTRAINT fk_category_stats_category
        FOREIGN KEY (category_id) REFERENCES categories(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_category_stats_date ON category_stats(date);

-- =============================================================================
-- LOGGING/TRACKING TABLES (for evaluation and analytics)
-- =============================================================================

-- Comparison Logs: Track product comparisons
CREATE TABLE IF NOT EXISTS comparison_logs (
    id SERIAL PRIMARY KEY,
    session_id UUID,

    -- Products compared
    product_asins VARCHAR(20)[] NOT NULL,
    comparison_type VARCHAR(50),

    -- User interaction
    winner_asin VARCHAR(20),
    user_feedback VARCHAR(50),

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comparison_products ON comparison_logs USING GIN (product_asins);

-- Search Logs: Track search queries for trend analysis
CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    session_id UUID,

    query_text TEXT NOT NULL,
    search_type VARCHAR(50),
    filters JSONB,

    -- Results
    result_count INTEGER,
    clicked_asins VARCHAR(20)[],

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_logs_date ON search_logs(created_at);

-- =============================================================================
-- VIEWS (for common queries)
-- =============================================================================

-- Product Summary: Combined view with latest price and trend
CREATE OR REPLACE VIEW product_summary AS
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

-- Category Overview: Category statistics view
CREATE OR REPLACE VIEW category_overview AS
SELECT
    c.id,
    c.name,
    c.level,
    c.path::text as category_path,
    COUNT(DISTINCT p.asin) as product_count,
    AVG(p.price) as avg_price,
    AVG(p.stars) as avg_rating,
    SUM(p.reviews_count) as total_reviews,
    COUNT(CASE WHEN p.is_best_seller THEN 1 END) as bestseller_count
FROM categories c
LEFT JOIN products p ON p.category_id = c.id
GROUP BY c.id, c.name, c.level, c.path;

-- Brand Performance: Brand statistics view
CREATE OR REPLACE VIEW brand_performance AS
SELECT
    b.name as brand_name,
    b.product_count,
    b.avg_rating,
    b.avg_price,
    b.review_count,
    COUNT(DISTINCT p.category_level1) as category_count,
    COUNT(CASE WHEN p.is_best_seller THEN 1 END) as bestseller_count
FROM brands b
LEFT JOIN products p ON p.brand = b.name
GROUP BY b.name, b.product_count, b.avg_rating, b.avg_price, b.review_count;

-- =============================================================================
-- FUNCTIONS (for agent tools)
-- =============================================================================

-- Function to get trending products in a category
CREATE OR REPLACE FUNCTION get_trending_products(
    p_category VARCHAR DEFAULT NULL,
    p_limit INTEGER DEFAULT 10,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    asin VARCHAR,
    title TEXT,
    brand VARCHAR,
    trend_score DECIMAL,
    review_velocity INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.asin,
        p.title,
        p.brand,
        pt.trend_score,
        pt.review_velocity
    FROM products p
    JOIN product_trends pt ON p.asin = pt.asin
    WHERE pt.date >= CURRENT_DATE - p_days
      AND (p_category IS NULL OR p.category_level1 = p_category)
    ORDER BY pt.trend_score DESC
    LIMIT p_limit;
END;
$$;

-- Function to get category price statistics
CREATE OR REPLACE FUNCTION get_category_price_stats(p_category VARCHAR)
RETURNS TABLE (
    min_price DECIMAL,
    max_price DECIMAL,
    avg_price DECIMAL,
    median_price DECIMAL,
    price_tiers JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(price) as avg_price,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
        jsonb_build_object(
            'budget', COUNT(*) FILTER (WHERE price < 50),
            'mid_range', COUNT(*) FILTER (WHERE price >= 50 AND price < 200),
            'premium', COUNT(*) FILTER (WHERE price >= 200 AND price < 500),
            'luxury', COUNT(*) FILTER (WHERE price >= 500)
        ) as price_tiers
    FROM products
    WHERE category_level1 = p_category AND price IS NOT NULL;
END;
$$;
