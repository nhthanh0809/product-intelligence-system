-- ============================================================================
-- Migration: Configuration Management System
-- Version: 001
-- Description: Create tables for system configuration, LLM providers,
--              search strategies, and enhanced conversation management
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Configuration Categories
-- ============================================================================
CREATE TABLE IF NOT EXISTS config_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    icon VARCHAR(50),  -- For UI display
    display_order INT DEFAULT 0,
    is_visible BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE config_categories IS 'Categories for grouping configuration settings';

-- ============================================================================
-- Configuration Settings
-- ============================================================================
CREATE TABLE IF NOT EXISTS config_settings (
    id SERIAL PRIMARY KEY,
    category_id INT REFERENCES config_categories(id) ON DELETE SET NULL,
    key VARCHAR(200) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    value_type VARCHAR(50) NOT NULL,  -- string, int, float, bool, json, list, secret
    default_value JSONB,
    label VARCHAR(200),  -- Human-readable label for UI
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,  -- If true, value should be encrypted/masked
    is_readonly BOOLEAN DEFAULT FALSE,  -- If true, cannot be changed via UI
    validation_rules JSONB,  -- {min, max, pattern, options, required}
    display_order INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE config_settings IS 'Key-value configuration settings with validation';

CREATE INDEX IF NOT EXISTS idx_config_settings_category ON config_settings(category_id);
CREATE INDEX IF NOT EXISTS idx_config_settings_key ON config_settings(key);

-- ============================================================================
-- LLM Providers
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,  -- ollama_local, ollama_remote, openai
    display_name VARCHAR(200),
    provider_type VARCHAR(50) NOT NULL,  -- ollama, openai, anthropic
    base_url VARCHAR(500),
    api_key_encrypted TEXT,  -- Encrypted API key for security
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    settings JSONB DEFAULT '{}'::jsonb,  -- timeout, max_retries, etc.
    health_check_url VARCHAR(500),  -- URL for health checks
    last_health_check TIMESTAMP WITH TIME ZONE,
    health_status VARCHAR(50) DEFAULT 'unknown',  -- healthy, unhealthy, unknown
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE llm_providers IS 'LLM provider configurations (Ollama, OpenAI, etc.)';

-- ============================================================================
-- LLM Models
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_models (
    id SERIAL PRIMARY KEY,
    provider_id INT REFERENCES llm_providers(id) ON DELETE CASCADE,
    model_name VARCHAR(200) NOT NULL,  -- llama3, gpt-4, etc.
    display_name VARCHAR(200),
    model_type VARCHAR(50) NOT NULL,  -- chat, embedding, reranker
    description TEXT,
    capabilities JSONB DEFAULT '{}'::jsonb,  -- max_tokens, supports_json, supports_streaming
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default_for_type BOOLEAN DEFAULT FALSE,  -- Default model for this type
    settings JSONB DEFAULT '{}'::jsonb,  -- Default temperature, max_tokens, etc.
    performance_metrics JSONB DEFAULT '{}'::jsonb,  -- latency_ms, tokens_per_sec
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(provider_id, model_name)
);

COMMENT ON TABLE llm_models IS 'Available models per LLM provider';

CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON llm_models(provider_id);
CREATE INDEX IF NOT EXISTS idx_llm_models_type ON llm_models(model_type);

-- ============================================================================
-- Agent Model Configurations
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_model_configs (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,  -- intent, synthesis, general, compare, recommend
    model_id INT REFERENCES llm_models(id) ON DELETE SET NULL,
    purpose VARCHAR(100) NOT NULL DEFAULT 'primary',  -- primary, fallback, embedding, reranker
    is_enabled BOOLEAN DEFAULT TRUE,
    priority INT DEFAULT 0,  -- For fallback ordering
    settings JSONB DEFAULT '{}'::jsonb,  -- temperature, max_tokens, timeout, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(agent_name, purpose)
);

COMMENT ON TABLE agent_model_configs IS 'Maps agents to their configured LLM models';

CREATE INDEX IF NOT EXISTS idx_agent_model_configs_agent ON agent_model_configs(agent_name);

-- ============================================================================
-- Search Strategies
-- ============================================================================
CREATE TABLE IF NOT EXISTS search_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,  -- keyword_brand_model_priority, etc.
    display_name VARCHAR(200),
    strategy_type VARCHAR(50) NOT NULL,  -- keyword, semantic, hybrid, section
    description TEXT,
    implementation_class VARCHAR(200) NOT NULL,  -- Python class path
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,  -- boost factors, weights, etc.
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    performance_metrics JSONB DEFAULT '{}'::jsonb,  -- MRR, R@1, latency_ms
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE search_strategies IS 'Available search strategies with their configurations';

CREATE INDEX IF NOT EXISTS idx_search_strategies_type ON search_strategies(strategy_type);

-- ============================================================================
-- Query Type to Strategy Mapping
-- ============================================================================
CREATE TABLE IF NOT EXISTS query_strategy_mapping (
    id SERIAL PRIMARY KEY,
    query_type VARCHAR(100) NOT NULL,  -- MODEL_NUMBER, BRAND_MODEL, GENERIC, etc.
    strategy_id INT REFERENCES search_strategies(id) ON DELETE CASCADE,
    priority INT DEFAULT 0,  -- Lower = higher priority
    is_enabled BOOLEAN DEFAULT TRUE,
    conditions JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(query_type, strategy_id)
);

COMMENT ON TABLE query_strategy_mapping IS 'Maps query types to search strategies';

CREATE INDEX IF NOT EXISTS idx_query_strategy_mapping_type ON query_strategy_mapping(query_type);

-- ============================================================================
-- Reranker Configurations
-- ============================================================================
CREATE TABLE IF NOT EXISTS reranker_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200),
    model_id INT REFERENCES llm_models(id) ON DELETE SET NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    settings JSONB DEFAULT '{}'::jsonb,  -- top_k, threshold, batch_size, etc.
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE reranker_configs IS 'Reranker model configurations';

-- ============================================================================
-- Configuration Audit Log
-- ============================================================================
CREATE TABLE IF NOT EXISTS config_audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id INT NOT NULL,
    action VARCHAR(20) NOT NULL,  -- INSERT, UPDATE, DELETE
    old_value JSONB,
    new_value JSONB,
    changed_fields TEXT[],
    changed_by VARCHAR(200),
    ip_address VARCHAR(50),
    user_agent TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE config_audit_log IS 'Audit trail for all configuration changes';

CREATE INDEX IF NOT EXISTS idx_config_audit_log_table ON config_audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_config_audit_log_record ON config_audit_log(table_name, record_id);
CREATE INDEX IF NOT EXISTS idx_config_audit_log_time ON config_audit_log(changed_at);

-- ============================================================================
-- Conversation Sessions (Enhanced)
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours',
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    message_count INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

COMMENT ON TABLE conversation_sessions IS 'Multi-turn conversation sessions with context';

CREATE INDEX IF NOT EXISTS idx_conversation_sessions_updated ON conversation_sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_active ON conversation_sessions(is_active, last_activity);

-- ============================================================================
-- Conversation Messages
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversation_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    intent VARCHAR(50),
    entities JSONB DEFAULT '{}'::jsonb,
    products JSONB DEFAULT '[]'::jsonb,
    resolved_references JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE conversation_messages IS 'Individual messages in conversations';

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_time ON conversation_messages(session_id, created_at);

-- ============================================================================
-- Insert Default Configuration Categories
-- ============================================================================
INSERT INTO config_categories (name, description, icon, display_order) VALUES
    ('general', 'General system settings', 'settings', 1),
    ('llm', 'LLM provider and model settings', 'brain', 2),
    ('agents', 'Agent-specific configurations', 'robot', 3),
    ('search', 'Search strategy settings', 'search', 4),
    ('reranker', 'Reranker model settings', 'sort', 5),
    ('conversation', 'Conversation and context settings', 'chat', 6),
    ('performance', 'Performance and caching settings', 'speed', 7)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- Insert Default Configuration Settings
-- ============================================================================
INSERT INTO config_settings (category_id, key, value, value_type, default_value, label, description, validation_rules, display_order) VALUES
    -- General settings
    ((SELECT id FROM config_categories WHERE name = 'general'),
     'system.log_level', '"INFO"', 'string', '"INFO"',
     'Log Level', 'System logging level',
     '{"options": ["DEBUG", "INFO", "WARNING", "ERROR"]}', 1),

    ((SELECT id FROM config_categories WHERE name = 'general'),
     'system.request_timeout', '120', 'int', '120',
     'Request Timeout (seconds)', 'Maximum time for API requests',
     '{"min": 10, "max": 600}', 2),

    -- LLM settings
    ((SELECT id FROM config_categories WHERE name = 'llm'),
     'llm.default_temperature', '0.7', 'float', '0.7',
     'Default Temperature', 'Default temperature for LLM generation',
     '{"min": 0, "max": 2}', 1),

    ((SELECT id FROM config_categories WHERE name = 'llm'),
     'llm.default_max_tokens', '1000', 'int', '1000',
     'Default Max Tokens', 'Default maximum tokens for LLM generation',
     '{"min": 100, "max": 4096}', 2),

    -- Agent settings
    ((SELECT id FROM config_categories WHERE name = 'agents'),
     'agent.intent.use_llm', 'true', 'bool', 'true',
     'Intent Agent: Use LLM', 'Enable LLM for intent classification', NULL, 1),

    ((SELECT id FROM config_categories WHERE name = 'agents'),
     'agent.synthesis.use_llm', 'true', 'bool', 'true',
     'Synthesis Agent: Use LLM', 'Enable LLM for response generation', NULL, 2),

    ((SELECT id FROM config_categories WHERE name = 'agents'),
     'agent.general.use_llm', 'true', 'bool', 'true',
     'General Agent: Use LLM', 'Enable LLM for general conversation', NULL, 3),

    -- Search settings
    ((SELECT id FROM config_categories WHERE name = 'search'),
     'search.default_limit', '10', 'int', '10',
     'Default Search Limit', 'Default number of results to return',
     '{"min": 1, "max": 100}', 1),

    ((SELECT id FROM config_categories WHERE name = 'search'),
     'search.hybrid.keyword_weight', '0.65', 'float', '0.65',
     'Hybrid: Keyword Weight', 'Weight for keyword search in hybrid mode',
     '{"min": 0, "max": 1}', 2),

    ((SELECT id FROM config_categories WHERE name = 'search'),
     'search.hybrid.semantic_weight', '0.35', 'float', '0.35',
     'Hybrid: Semantic Weight', 'Weight for semantic search in hybrid mode',
     '{"min": 0, "max": 1}', 3),

    -- Reranker settings
    ((SELECT id FROM config_categories WHERE name = 'reranker'),
     'reranker.enabled', 'false', 'bool', 'false',
     'Enable Reranker', 'Enable reranking of search results', NULL, 1),

    ((SELECT id FROM config_categories WHERE name = 'reranker'),
     'reranker.top_k', '10', 'int', '10',
     'Reranker Top K', 'Number of results to rerank',
     '{"min": 1, "max": 50}', 2),

    -- Conversation settings
    ((SELECT id FROM config_categories WHERE name = 'conversation'),
     'conversation.session_ttl_hours', '24', 'int', '24',
     'Session TTL (hours)', 'How long to keep conversation sessions',
     '{"min": 1, "max": 168}', 1),

    ((SELECT id FROM config_categories WHERE name = 'conversation'),
     'conversation.max_context_products', '10', 'int', '10',
     'Max Context Products', 'Maximum products to keep in context',
     '{"min": 1, "max": 50}', 2),

    ((SELECT id FROM config_categories WHERE name = 'conversation'),
     'conversation.enable_pronoun_resolution', 'true', 'bool', 'true',
     'Enable Pronoun Resolution', 'Resolve pronouns like "it", "them" from context', NULL, 3),

    -- Performance settings
    ((SELECT id FROM config_categories WHERE name = 'performance'),
     'cache.search_ttl_seconds', '300', 'int', '300',
     'Search Cache TTL', 'Cache duration for search results',
     '{"min": 0, "max": 3600}', 1),

    ((SELECT id FROM config_categories WHERE name = 'performance'),
     'cache.config_ttl_seconds', '60', 'int', '60',
     'Config Cache TTL', 'Cache duration for configuration settings',
     '{"min": 0, "max": 300}', 2)
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- Insert Default LLM Providers
-- ============================================================================
INSERT INTO llm_providers (name, display_name, provider_type, base_url, is_enabled, is_default, settings) VALUES
    ('ollama_local', 'Ollama (Local)', 'ollama', 'http://ollama:11434', TRUE, TRUE,
     '{"timeout": 120, "max_retries": 3}'),
    ('ollama_remote', 'Ollama (Remote)', 'ollama', NULL, FALSE, FALSE,
     '{"timeout": 120, "max_retries": 3}'),
    ('openai', 'OpenAI', 'openai', 'https://api.openai.com/v1', FALSE, FALSE,
     '{"timeout": 60, "max_retries": 3}')
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- Insert Default LLM Models
-- ============================================================================
INSERT INTO llm_models (provider_id, model_name, display_name, model_type, is_enabled, is_default_for_type, settings, capabilities) VALUES
    ((SELECT id FROM llm_providers WHERE name = 'ollama_local'),
     'llama3.2', 'Llama 3.2', 'chat', TRUE, TRUE,
     '{"temperature": 0.7, "max_tokens": 2048}',
     '{"max_context": 8192, "supports_json": true, "supports_streaming": true}'),

    ((SELECT id FROM llm_providers WHERE name = 'ollama_local'),
     'bge-large', 'Nomic Embed Text', 'embedding', TRUE, TRUE,
     '{}', '{"dimensions": 768}'),

    ((SELECT id FROM llm_providers WHERE name = 'ollama_local'),
     'bge-reranker-base', 'BGE Reranker Base', 'reranker', TRUE, TRUE,
     '{"top_k": 10}', '{"max_documents": 100}'),

    ((SELECT id FROM llm_providers WHERE name = 'openai'),
     'gpt-4o-mini', 'GPT-4o Mini', 'chat', FALSE, FALSE,
     '{"temperature": 0.7, "max_tokens": 4096}',
     '{"max_context": 128000, "supports_json": true, "supports_streaming": true}'),

    ((SELECT id FROM llm_providers WHERE name = 'openai'),
     'gpt-4o', 'GPT-4o', 'chat', FALSE, FALSE,
     '{"temperature": 0.7, "max_tokens": 4096}',
     '{"max_context": 128000, "supports_json": true, "supports_streaming": true}'),

    ((SELECT id FROM llm_providers WHERE name = 'openai'),
     'text-embedding-3-small', 'Text Embedding 3 Small', 'embedding', FALSE, FALSE,
     '{}', '{"dimensions": 1536}')
ON CONFLICT (provider_id, model_name) DO NOTHING;

-- ============================================================================
-- Insert Default Agent Model Configs
-- ============================================================================
INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'intent', id, 'primary', TRUE, '{"temperature": 0.1, "max_tokens": 500}'
FROM llm_models WHERE model_name = 'llama3.2' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'synthesis', id, 'primary', TRUE, '{"temperature": 0.7, "max_tokens": 1000}'
FROM llm_models WHERE model_name = 'llama3.2' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'general', id, 'primary', TRUE, '{"temperature": 0.8, "max_tokens": 500}'
FROM llm_models WHERE model_name = 'llama3.2' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'compare', id, 'primary', TRUE, '{"temperature": 0.5, "max_tokens": 1500}'
FROM llm_models WHERE model_name = 'llama3.2' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'recommend', id, 'primary', TRUE, '{"temperature": 0.6, "max_tokens": 800}'
FROM llm_models WHERE model_name = 'llama3.2' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'embedding', id, 'primary', TRUE, '{}'
FROM llm_models WHERE model_name = 'bge-large' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

INSERT INTO agent_model_configs (agent_name, model_id, purpose, is_enabled, settings)
SELECT 'reranker', id, 'primary', TRUE, '{"top_k": 10}'
FROM llm_models WHERE model_name = 'bge-reranker-base' LIMIT 1
ON CONFLICT (agent_name, purpose) DO NOTHING;

-- ============================================================================
-- Insert Default Search Strategies
-- ============================================================================
INSERT INTO search_strategies (name, display_name, strategy_type, description, implementation_class, settings, is_enabled, is_default, performance_metrics) VALUES
    ('keyword_basic', 'Basic Keyword Search', 'keyword',
     'Standard Elasticsearch keyword search',
     'src.search.strategies.keyword.BasicKeywordStrategy',
     '{"title_boost": 5.0, "brand_boost": 3.0}',
     TRUE, FALSE, '{"r_at_1": 0.75}'),

    ('keyword_brand_model_priority', 'Brand+Model Priority', 'keyword',
     'Optimized for brand and model number queries (R@1 87.7%)',
     'src.search.strategies.keyword.BrandModelPriorityStrategy',
     '{"title_boost": 10.0, "brand_boost": 5.0, "model_boost": 20.0}',
     TRUE, TRUE, '{"r_at_1": 0.877}'),

    ('keyword_fuzzy_exact', 'Fuzzy + Exact Combined', 'keyword',
     'Combines exact matching with fuzzy fallback',
     'src.search.strategies.keyword.FuzzyExactCombinedStrategy',
     '{"fuzziness": "AUTO", "exact_boost": 2.0}',
     TRUE, FALSE, '{}'),

    ('semantic_basic', 'Basic Semantic Search', 'semantic',
     'Standard vector similarity search',
     'src.search.strategies.semantic.BasicSemanticStrategy',
     '{}', TRUE, FALSE, '{"mrr": 0.55}'),

    ('semantic_format_aware', 'Format-Aware Semantic', 'semantic',
     'Matches query format to document embedding format',
     'src.search.strategies.semantic.FormatAwareSemanticStrategy',
     '{"format_template": "Product: {title}\nBrand: {brand}\nCategory: {category}"}',
     TRUE, TRUE, '{"mrr": 0.65, "similarity": 0.99}'),

    ('semantic_multi_query', 'Multi-Query Semantic', 'semantic',
     'Query expansion with RRF fusion',
     'src.search.strategies.semantic.MultiQuerySemanticStrategy',
     '{"num_variants": 3, "rrf_k": 60}',
     TRUE, FALSE, '{}'),

    ('hybrid_basic', 'Basic Hybrid Search', 'hybrid',
     'Standard RRF fusion of keyword and semantic',
     'src.search.strategies.hybrid.BasicHybridStrategy',
     '{"keyword_weight": 0.5, "semantic_weight": 0.5, "rrf_k": 60}',
     TRUE, FALSE, '{"mrr": 0.80}'),

    ('hybrid_keyword_priority', 'Keyword Priority Hybrid', 'hybrid',
     'Best overall hybrid strategy (MRR 0.9126)',
     'src.search.strategies.hybrid.KeywordPriorityHybridStrategy',
     '{"keyword_weight": 0.65, "semantic_weight": 0.35, "rrf_k": 60}',
     TRUE, TRUE, '{"mrr": 0.9126}'),

    ('hybrid_adaptive', 'Adaptive Hybrid', 'hybrid',
     'Adjusts weights based on query characteristics',
     'src.search.strategies.hybrid.AdaptiveHybridStrategy',
     '{"short_query_kw_weight": 0.75, "long_query_kw_weight": 0.45}',
     TRUE, FALSE, '{}'),

    ('section_basic', 'Basic Section Search', 'section',
     'Filter by section and search',
     'src.search.strategies.section.BasicSectionStrategy',
     '{}', TRUE, TRUE, '{}')
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- Insert Default Query-Strategy Mappings
-- ============================================================================
INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'MODEL_NUMBER', id, 0, TRUE FROM search_strategies WHERE name = 'keyword_brand_model_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'MODEL_NUMBER', id, 1, TRUE FROM search_strategies WHERE name = 'hybrid_keyword_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'BRAND_MODEL', id, 0, TRUE FROM search_strategies WHERE name = 'keyword_brand_model_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'BRAND_MODEL', id, 1, TRUE FROM search_strategies WHERE name = 'hybrid_keyword_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'GENERIC', id, 0, TRUE FROM search_strategies WHERE name = 'hybrid_keyword_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'GENERIC', id, 1, TRUE FROM search_strategies WHERE name = 'semantic_format_aware'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'SHORT_TITLE', id, 0, TRUE FROM search_strategies WHERE name = 'hybrid_keyword_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

INSERT INTO query_strategy_mapping (query_type, strategy_id, priority, is_enabled)
SELECT 'DEFAULT', id, 0, TRUE FROM search_strategies WHERE name = 'hybrid_keyword_priority'
ON CONFLICT (query_type, strategy_id) DO NOTHING;

-- ============================================================================
-- Insert Default Reranker Config
-- ============================================================================
INSERT INTO reranker_configs (name, display_name, model_id, is_enabled, is_default, settings)
SELECT 'bge_reranker', 'BGE Reranker', id, FALSE, TRUE, '{"top_k": 10, "batch_size": 32, "threshold": 0.0}'
FROM llm_models WHERE model_name = 'bge-reranker-base' LIMIT 1
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- Create Update Timestamp Trigger
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers for updated_at
DROP TRIGGER IF EXISTS update_config_categories_updated_at ON config_categories;
CREATE TRIGGER update_config_categories_updated_at
    BEFORE UPDATE ON config_categories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_config_settings_updated_at ON config_settings;
CREATE TRIGGER update_config_settings_updated_at
    BEFORE UPDATE ON config_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_llm_providers_updated_at ON llm_providers;
CREATE TRIGGER update_llm_providers_updated_at
    BEFORE UPDATE ON llm_providers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_llm_models_updated_at ON llm_models;
CREATE TRIGGER update_llm_models_updated_at
    BEFORE UPDATE ON llm_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_agent_model_configs_updated_at ON agent_model_configs;
CREATE TRIGGER update_agent_model_configs_updated_at
    BEFORE UPDATE ON agent_model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_search_strategies_updated_at ON search_strategies;
CREATE TRIGGER update_search_strategies_updated_at
    BEFORE UPDATE ON search_strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_reranker_configs_updated_at ON reranker_configs;
CREATE TRIGGER update_reranker_configs_updated_at
    BEFORE UPDATE ON reranker_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_conversation_sessions_updated_at ON conversation_sessions;
CREATE TRIGGER update_conversation_sessions_updated_at
    BEFORE UPDATE ON conversation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Create Audit Trigger Function
-- ============================================================================
CREATE OR REPLACE FUNCTION audit_config_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO config_audit_log (table_name, record_id, action, old_value, new_value)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', row_to_json(OLD), NULL);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO config_audit_log (table_name, record_id, action, old_value, new_value)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO config_audit_log (table_name, record_id, action, old_value, new_value)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', NULL, row_to_json(NEW));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers
DROP TRIGGER IF EXISTS audit_config_settings ON config_settings;
CREATE TRIGGER audit_config_settings
    AFTER INSERT OR UPDATE OR DELETE ON config_settings
    FOR EACH ROW EXECUTE FUNCTION audit_config_changes();

DROP TRIGGER IF EXISTS audit_llm_providers ON llm_providers;
CREATE TRIGGER audit_llm_providers
    AFTER INSERT OR UPDATE OR DELETE ON llm_providers
    FOR EACH ROW EXECUTE FUNCTION audit_config_changes();

DROP TRIGGER IF EXISTS audit_llm_models ON llm_models;
CREATE TRIGGER audit_llm_models
    AFTER INSERT OR UPDATE OR DELETE ON llm_models
    FOR EACH ROW EXECUTE FUNCTION audit_config_changes();

DROP TRIGGER IF EXISTS audit_search_strategies ON search_strategies;
CREATE TRIGGER audit_search_strategies
    AFTER INSERT OR UPDATE OR DELETE ON search_strategies
    FOR EACH ROW EXECUTE FUNCTION audit_config_changes();

DROP TRIGGER IF EXISTS audit_agent_model_configs ON agent_model_configs;
CREATE TRIGGER audit_agent_model_configs
    AFTER INSERT OR UPDATE OR DELETE ON agent_model_configs
    FOR EACH ROW EXECUTE FUNCTION audit_config_changes();

DROP TRIGGER IF EXISTS audit_reranker_configs ON reranker_configs;
CREATE TRIGGER audit_reranker_configs
    AFTER INSERT OR UPDATE OR DELETE ON reranker_configs
    FOR EACH ROW EXECUTE FUNCTION audit_config_changes();
