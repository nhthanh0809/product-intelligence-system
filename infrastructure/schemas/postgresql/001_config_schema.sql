-- Configuration Management Schema
-- Creates tables for managing LLM providers, models, search strategies, and system config
--
-- To apply this schema manually:
--   docker exec -i pis-postgres psql -U pis_user -d product_intelligence < scripts/init_database.sql
--
-- To enable auto-initialization on container restart:
--   sudo cp scripts/init_database.sql infrastructure/schemas/postgresql/001_config_schema.sql

-- =============================================================================
-- Configuration Categories
-- =============================================================================

CREATE TABLE IF NOT EXISTS config_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    icon VARCHAR(50),
    display_order INTEGER DEFAULT 0,
    is_visible BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_config_categories_display_order ON config_categories(display_order);

-- =============================================================================
-- Configuration Settings
-- =============================================================================

CREATE TABLE IF NOT EXISTS config_settings (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES config_categories(id) ON DELETE SET NULL,
    key VARCHAR(200) NOT NULL UNIQUE,
    value JSONB,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    default_value JSONB,
    label VARCHAR(200),
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    is_readonly BOOLEAN DEFAULT FALSE,
    validation_rules JSONB,
    display_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_config_settings_category ON config_settings(category_id);
CREATE INDEX IF NOT EXISTS idx_config_settings_key ON config_settings(key);

-- =============================================================================
-- LLM Providers
-- =============================================================================

CREATE TABLE IF NOT EXISTS llm_providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200),
    provider_type VARCHAR(50) NOT NULL,
    base_url VARCHAR(500),
    api_key_encrypted TEXT,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    settings JSONB DEFAULT '{}',
    health_check_url VARCHAR(500),
    last_health_check TIMESTAMP WITH TIME ZONE,
    health_status VARCHAR(20) DEFAULT 'unknown',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_providers_enabled ON llm_providers(is_enabled);
CREATE INDEX IF NOT EXISTS idx_llm_providers_type ON llm_providers(provider_type);

-- =============================================================================
-- LLM Models
-- =============================================================================

CREATE TABLE IF NOT EXISTS llm_models (
    id SERIAL PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES llm_providers(id) ON DELETE CASCADE,
    model_name VARCHAR(200) NOT NULL,
    display_name VARCHAR(200),
    model_type VARCHAR(50) NOT NULL,
    description TEXT,
    capabilities JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default_for_type BOOLEAN DEFAULT FALSE,
    settings JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(provider_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON llm_models(provider_id);
CREATE INDEX IF NOT EXISTS idx_llm_models_type ON llm_models(model_type);
CREATE INDEX IF NOT EXISTS idx_llm_models_enabled ON llm_models(is_enabled);

-- =============================================================================
-- Agent Model Configurations
-- =============================================================================

CREATE TABLE IF NOT EXISTS agent_model_configs (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    model_id INTEGER REFERENCES llm_models(id) ON DELETE SET NULL,
    purpose VARCHAR(50) DEFAULT 'primary',
    is_enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 0,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(agent_name, purpose)
);

CREATE INDEX IF NOT EXISTS idx_agent_model_configs_agent ON agent_model_configs(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_model_configs_model ON agent_model_configs(model_id);

-- =============================================================================
-- Search Strategies
-- =============================================================================

CREATE TABLE IF NOT EXISTS search_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200),
    strategy_type VARCHAR(50) NOT NULL,
    description TEXT,
    implementation_class VARCHAR(200) NOT NULL,
    settings JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_strategies_type ON search_strategies(strategy_type);
CREATE INDEX IF NOT EXISTS idx_search_strategies_enabled ON search_strategies(is_enabled);

-- =============================================================================
-- Query Strategy Mapping
-- =============================================================================

CREATE TABLE IF NOT EXISTS query_strategy_mapping (
    id SERIAL PRIMARY KEY,
    query_type VARCHAR(100) NOT NULL,
    strategy_id INTEGER NOT NULL REFERENCES search_strategies(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 0,
    is_enabled BOOLEAN DEFAULT TRUE,
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(query_type, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_query_strategy_mapping_type ON query_strategy_mapping(query_type);
CREATE INDEX IF NOT EXISTS idx_query_strategy_mapping_strategy ON query_strategy_mapping(strategy_id);

-- =============================================================================
-- Reranker Configurations
-- =============================================================================

CREATE TABLE IF NOT EXISTS reranker_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200),
    model_id INTEGER REFERENCES llm_models(id) ON DELETE SET NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    settings JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reranker_configs_model ON reranker_configs(model_id);
CREATE INDEX IF NOT EXISTS idx_reranker_configs_enabled ON reranker_configs(is_enabled);

-- =============================================================================
-- Configuration Audit Log
-- =============================================================================

CREATE TABLE IF NOT EXISTS config_audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id INTEGER NOT NULL,
    action VARCHAR(20) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    changed_fields TEXT[],
    changed_by VARCHAR(200),
    ip_address VARCHAR(45),
    user_agent TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_config_audit_log_table ON config_audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_config_audit_log_record ON config_audit_log(record_id);
CREATE INDEX IF NOT EXISTS idx_config_audit_log_changed_at ON config_audit_log(changed_at);

-- =============================================================================
-- Conversation Sessions
-- =============================================================================

CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours'
);

CREATE INDEX IF NOT EXISTS idx_conversation_sessions_active ON conversation_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_expires ON conversation_sessions(expires_at);

-- =============================================================================
-- Conversation Messages
-- =============================================================================

CREATE TABLE IF NOT EXISTS conversation_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    intent VARCHAR(100),
    entities JSONB DEFAULT '{}',
    products JSONB DEFAULT '[]',
    resolved_references JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_created ON conversation_messages(created_at);

-- =============================================================================
-- Trigger function for updating updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to all tables with updated_at
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN
        SELECT table_name
        FROM information_schema.columns
        WHERE column_name = 'updated_at'
        AND table_schema = 'public'
    LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS update_%I_updated_at ON %I;
            CREATE TRIGGER update_%I_updated_at
                BEFORE UPDATE ON %I
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        ', tbl, tbl, tbl, tbl);
    END LOOP;
END;
$$;

-- =============================================================================
-- Default Data: Configuration Categories
-- =============================================================================

INSERT INTO config_categories (name, description, icon, display_order, is_visible)
VALUES
    ('llm', 'LLM Provider and Model Configuration', 'brain', 1, TRUE),
    ('search', 'Search Strategy Configuration', 'search', 2, TRUE),
    ('system', 'System Settings', 'settings', 3, TRUE),
    ('agents', 'Agent Configuration', 'bot', 4, TRUE)
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Default Data: Ollama Provider
-- =============================================================================

INSERT INTO llm_providers (name, display_name, provider_type, base_url, is_enabled, is_default, settings, health_check_url)
VALUES (
    'ollama-local',
    'Local Ollama',
    'ollama',
    'http://ollama:11434',
    TRUE,
    TRUE,
    '{"timeout": 120, "keep_alive": -1}',
    'http://ollama:11434'
)
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Default Data: Default Models (will be added after provider exists)
-- =============================================================================

INSERT INTO llm_models (provider_id, model_name, display_name, model_type, description, capabilities, is_enabled, is_default_for_type, settings)
SELECT
    p.id,
    'llama3.2:3b',
    'Llama 3.2 3B',
    'chat',
    'Fast and efficient chat model',
    '{"max_context": 8192, "max_tokens": 4096, "supports_json": true, "supports_streaming": true}',
    TRUE,
    TRUE,
    '{"temperature": 0.7, "top_p": 0.9}'
FROM llm_providers p
WHERE p.name = 'ollama-local'
AND NOT EXISTS (SELECT 1 FROM llm_models WHERE model_name = 'llama3.2:3b' AND provider_id = p.id);

INSERT INTO llm_models (provider_id, model_name, display_name, model_type, description, capabilities, is_enabled, is_default_for_type, settings)
SELECT
    p.id,
    'bge-large',
    'BGE Large',
    'embedding',
    'High-quality embedding model (1024 dimensions)',
    '{"dimensions": 1024, "max_tokens": 512}',
    TRUE,
    TRUE,
    '{}'
FROM llm_providers p
WHERE p.name = 'ollama-local'
AND NOT EXISTS (SELECT 1 FROM llm_models WHERE model_name = 'bge-large' AND provider_id = p.id);

-- Reranker model
INSERT INTO llm_models (provider_id, model_name, display_name, model_type, description, capabilities, is_enabled, is_default_for_type, settings)
SELECT
    p.id,
    'qllama/bge-reranker-v2-m3',
    'BGE Reranker v2 M3',
    'reranker',
    'High-quality reranker model for search result optimization',
    '{"max_tokens": 512, "supports_batch": true}',
    TRUE,
    TRUE,
    '{}'
FROM llm_providers p
WHERE p.name = 'ollama-local'
AND NOT EXISTS (SELECT 1 FROM llm_models WHERE model_name = 'qllama/bge-reranker-v2-m3' AND provider_id = p.id);

-- =============================================================================
-- Default Data: Reranker Configurations
-- =============================================================================

INSERT INTO reranker_configs (name, display_name, model_id, is_enabled, is_default, settings)
SELECT
    'bge-reranker-v2-m3',
    'BGE Reranker v2 M3',
    m.id,
    TRUE,
    TRUE,
    '{"top_k": 10, "score_threshold": 0.5}'
FROM llm_models m
WHERE m.model_name = 'qllama/bge-reranker-v2-m3'
AND NOT EXISTS (SELECT 1 FROM reranker_configs WHERE name = 'bge-reranker-v2-m3');

-- =============================================================================
-- Default Data: Search Strategies
-- =============================================================================

INSERT INTO search_strategies (name, display_name, strategy_type, description, implementation_class, is_enabled, is_default, settings)
VALUES
    ('keyword_search', 'Keyword Search', 'keyword', 'Traditional keyword-based search', 'src.search.strategies.keyword.basic.BasicKeywordStrategy', TRUE, FALSE, '{"boost_title": 2.0, "boost_description": 1.0}'),
    ('semantic_search', 'Semantic Search', 'semantic', 'Vector similarity search using embeddings', 'src.search.strategies.semantic.basic.BasicSemanticStrategy', TRUE, TRUE, '{"top_k": 10, "score_threshold": 0.5}'),
    ('hybrid_search', 'Hybrid Search', 'hybrid', 'Combined keyword and semantic search', 'src.search.strategies.hybrid.keyword_priority.KeywordPriorityHybridStrategy', TRUE, FALSE, '{"keyword_weight": 0.3, "semantic_weight": 0.7}')
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Default Data: Query Strategy Mappings
-- =============================================================================

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'DEFAULT', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'SEARCH', id, 1, TRUE FROM search_strategies WHERE name = 'hybrid_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'COMPARE', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'SHORT_TITLE', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'PRODUCT_TYPE', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'CATEGORY', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'BRAND', id, 1, TRUE FROM search_strategies WHERE name = 'keyword_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'RECOMMEND', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'ANALYZE', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'PRICE_CHECK', id, 1, TRUE FROM search_strategies WHERE name = 'hybrid_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'TREND', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_search'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

-- =============================================================================
-- Default Data: Agent Model Configurations
-- =============================================================================

-- Intent Agent - uses chat model for intent classification
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'IntentAgent', id, 'primary', TRUE, 1, '{"temperature": 0.3, "max_tokens": 500}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Search Agent - uses embedding model for semantic search
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'SearchAgent', id, 'embedding', TRUE, 1, '{}'
FROM llm_models WHERE model_name = 'bge-large' AND model_type = 'embedding'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Search Agent - uses chat model for query expansion
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'SearchAgent', id, 'primary', TRUE, 1, '{"temperature": 0.5, "max_tokens": 1000}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Compare Agent
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'CompareAgent', id, 'primary', TRUE, 1, '{"temperature": 0.3, "max_tokens": 2000}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Analysis Agent
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'AnalysisAgent', id, 'primary', TRUE, 1, '{"temperature": 0.4, "max_tokens": 2000}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Price Agent
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'PriceAgent', id, 'primary', TRUE, 1, '{"temperature": 0.2, "max_tokens": 1500}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Trend Agent
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'TrendAgent', id, 'primary', TRUE, 1, '{"temperature": 0.4, "max_tokens": 2000}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Recommend Agent
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'RecommendAgent', id, 'primary', TRUE, 1, '{"temperature": 0.5, "max_tokens": 1500}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Attribute Agent
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'AttributeAgent', id, 'primary', TRUE, 1, '{"temperature": 0.2, "max_tokens": 1000}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- Synthesis Agent - generates final responses
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, priority, settings)
SELECT 'SynthesisAgent', id, 'primary', TRUE, 1, '{"temperature": 0.6, "max_tokens": 3000}'
FROM llm_models WHERE model_name = 'llama3.2:3b' AND model_type = 'chat'
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- =============================================================================
-- Default Data: System Configuration Settings
-- =============================================================================

-- Reranker settings (search category)
INSERT INTO config_settings (category_id, key, value, value_type, default_value, label, description, is_sensitive, is_readonly, display_order)
SELECT
    c.id,
    'reranker.enabled',
    'true'::jsonb,
    'bool',
    'true'::jsonb,
    'Enable Reranker',
    'Enable reranking of search results using the configured reranker model',
    FALSE,
    FALSE,
    1
FROM config_categories c WHERE c.name = 'search'
ON CONFLICT (key) DO NOTHING;

INSERT INTO config_settings (category_id, key, value, value_type, default_value, label, description, is_sensitive, is_readonly, display_order)
SELECT
    c.id,
    'reranker.top_k',
    '10'::jsonb,
    'int',
    '10'::jsonb,
    'Reranker Top K',
    'Number of top results to rerank',
    FALSE,
    FALSE,
    2
FROM config_categories c WHERE c.name = 'search'
ON CONFLICT (key) DO NOTHING;

INSERT INTO config_settings (category_id, key, value, value_type, default_value, label, description, is_sensitive, is_readonly, display_order)
SELECT
    c.id,
    'reranker.score_threshold',
    '0.5'::jsonb,
    'float',
    '0.5'::jsonb,
    'Reranker Score Threshold',
    'Minimum score threshold for reranked results',
    FALSE,
    FALSE,
    3
FROM config_categories c WHERE c.name = 'search'
ON CONFLICT (key) DO NOTHING;

-- Search settings
INSERT INTO config_settings (category_id, key, value, value_type, default_value, label, description, is_sensitive, is_readonly, display_order)
SELECT
    c.id,
    'search.default_limit',
    '10'::jsonb,
    'int',
    '10'::jsonb,
    'Default Search Limit',
    'Default number of search results to return',
    FALSE,
    FALSE,
    10
FROM config_categories c WHERE c.name = 'search'
ON CONFLICT (key) DO NOTHING;

INSERT INTO config_settings (category_id, key, value, value_type, default_value, label, description, is_sensitive, is_readonly, display_order)
SELECT
    c.id,
    'search.semantic_score_threshold',
    '0.5'::jsonb,
    'float',
    '0.5'::jsonb,
    'Semantic Score Threshold',
    'Minimum similarity score for semantic search results',
    FALSE,
    FALSE,
    11
FROM config_categories c WHERE c.name = 'search'
ON CONFLICT (key) DO NOTHING;