"""Settings page for configuration management."""

import httpx
import streamlit as st
from config import get_settings

# Import LLM settings components
from components.settings.llm_settings import (
    LLMSettingsAPI,
    render_llm_providers_section,
    render_llm_models_section,
    render_agent_configs_section,
    render_connection_test_section,
)

settings = get_settings()

st.set_page_config(
    page_title="Settings - Product Intelligence System",
    page_icon="⚙️",
    layout="wide",
)


# =============================================================================
# API Client
# =============================================================================

class ConfigAPIClient:
    """Client for configuration API."""

    def __init__(self, base_url: str):
        self.base_url = f"{base_url}/api/config"
        self.timeout = settings.request_timeout

    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request to config API."""
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = getattr(client, method)(url, **kwargs)
                response.raise_for_status()
                # Handle 204 No Content (e.g., DELETE responses)
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

    def patch(self, endpoint: str, **kwargs):
        return self._make_request("patch", endpoint, **kwargs)

    def delete(self, endpoint: str):
        return self._make_request("delete", endpoint)


api = ConfigAPIClient(settings.multi_agent_url)


# =============================================================================
# Page Header
# =============================================================================

st.title("⚙️ System Settings")
st.markdown("Configure LLM providers, models, search strategies, and more.")

# Check API availability
with st.spinner("Checking configuration service..."):
    health = api.get("/categories")
    if health is None:
        st.warning(
            "Configuration service is not available. "
            "The database may not be initialized. "
            "Run the migration script first."
        )
        st.stop()


# =============================================================================
# Sidebar Navigation
# =============================================================================

st.sidebar.title("Settings Sections")
section = st.sidebar.radio(
    "Navigate to",
    [
        "LLM Providers",
        "LLM Models",
        "Agent Configurations",
        "Connection Test",
        "Search Strategies",
        "Query Mappings",
        "Reranker Settings",
        "General Settings",
    ]
)

# Create LLM Settings API wrapper
llm_api = LLMSettingsAPI(api)


# =============================================================================
# Search Strategies Section
# =============================================================================

def render_search_strategies():
    """Render search strategies management section."""
    st.header("🔍 Search Strategies")
    st.markdown("Configure search strategies for different query types.")

    data = api.get("/search/strategies?enabled_only=false")
    if not data:
        return

    strategies = data.get("strategies", [])

    # Add new strategy
    with st.expander("➕ Add New Strategy", expanded=False):
        with st.form("add_strategy"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name*", placeholder="hybrid-search")
                display_name = st.text_input("Display Name", placeholder="Hybrid Search")
                strategy_type = st.selectbox("Type*", ["keyword", "semantic", "hybrid", "section"])
            with col2:
                impl_class = st.text_input(
                    "Implementation Class*",
                    placeholder="src.tools.search_tools.HybridSearch"
                )
                description = st.text_area("Description")
                is_default = st.checkbox("Set as Default")

            if st.form_submit_button("Add Strategy"):
                payload = {
                    "name": name,
                    "display_name": display_name or None,
                    "strategy_type": strategy_type,
                    "implementation_class": impl_class,
                    "description": description or None,
                    "is_default": is_default,
                }
                result = api.post("/search/strategies", json=payload)
                if result:
                    st.success(f"Strategy '{name}' created!")
                    st.rerun()

    if not strategies:
        st.info("No strategies configured. Add one above.")
        return

    for strategy in strategies:
        with st.expander(
            f"{'✅' if strategy.get('is_enabled') else '❌'} "
            f"{strategy.get('display_name') or strategy.get('name')} "
            f"({strategy.get('strategy_type')})"
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Name:** {strategy.get('name')}")
                st.markdown(f"**Type:** {strategy.get('strategy_type')}")
                st.markdown(f"**Class:** `{strategy.get('implementation_class')}`")
                if strategy.get("description"):
                    st.markdown(f"**Description:** {strategy.get('description')}")

                if strategy.get("performance_metrics"):
                    metrics = strategy["performance_metrics"]
                    st.markdown("**Performance:**")
                    for key, value in metrics.items():
                        st.markdown(f"- {key}: {value}")

            with col2:
                enabled = st.checkbox(
                    "Enabled",
                    value=strategy.get("is_enabled", True),
                    key=f"strat_enabled_{strategy['id']}"
                )
                if enabled != strategy.get("is_enabled"):
                    api.patch(f"/search/strategies/{strategy['id']}", json={"is_enabled": enabled})
                    st.rerun()

                if st.button("🗑️ Delete", key=f"del_strat_{strategy['id']}"):
                    api.delete(f"/search/strategies/{strategy['id']}")
                    st.rerun()


# =============================================================================
# Query Mappings Section
# =============================================================================

def render_query_mappings():
    """Render query-strategy mapping section."""
    st.header("🗺️ Query-Strategy Mappings")
    st.markdown("Configure which strategies to use for different query types.")

    mappings_data = api.get("/search/mappings")
    strategies_data = api.get("/search/strategies?enabled_only=false")

    if not mappings_data or not strategies_data:
        return

    mappings = mappings_data.get("mappings", [])
    strategies = {s["id"]: s["name"] for s in strategies_data.get("strategies", [])}

    # Group by query type
    query_types = {}
    for mapping in mappings:
        qt = mapping.get("query_type")
        if qt not in query_types:
            query_types[qt] = []
        query_types[qt].append(mapping)

    # Add new mapping
    with st.expander("➕ Add New Mapping", expanded=False):
        with st.form("add_mapping"):
            col1, col2 = st.columns(2)
            with col1:
                query_type = st.text_input("Query Type*", placeholder="SEARCH")
                strategy_id = st.selectbox(
                    "Strategy*",
                    options=list(strategies.keys()),
                    format_func=lambda x: strategies.get(x, str(x))
                )
            with col2:
                priority = st.number_input("Priority", min_value=0, max_value=100, value=0)

            if st.form_submit_button("Add Mapping"):
                payload = {
                    "query_type": query_type,
                    "strategy_id": strategy_id,
                    "priority": priority,
                }
                result = api.post("/search/mappings", json=payload)
                if result:
                    st.success(f"Mapping for '{query_type}' created!")
                    st.rerun()

    if not query_types:
        st.info("No mappings configured. Add one above.")
        return

    for query_type, type_mappings in sorted(query_types.items()):
        with st.expander(f"📌 {query_type}", expanded=True):
            for mapping in sorted(type_mappings, key=lambda x: x.get("priority", 0)):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    strategy = mapping.get("strategy", {})
                    st.markdown(f"**Strategy:** {strategy.get('name', 'Unknown')}")
                    st.markdown(f"**Type:** {strategy.get('strategy_type', 'Unknown')}")

                with col2:
                    st.markdown(f"**Priority:** {mapping.get('priority', 0)}")
                    st.markdown(f"**Status:** {'Enabled' if mapping.get('is_enabled') else 'Disabled'}")

                with col3:
                    if st.button("🗑️", key=f"del_map_{mapping['id']}"):
                        api.delete(f"/search/mappings/{mapping['id']}")
                        st.rerun()


# =============================================================================
# Reranker Settings Section
# =============================================================================

def render_reranker_settings():
    """Render reranker configuration section."""
    st.header("🎯 Reranker Settings")
    st.markdown("Configure reranking models for improving search results.")

    data = api.get("/rerankers?enabled_only=false")
    if not data:
        return

    configs = data.get("configs", [])

    # Global reranker toggle
    st.subheader("Global Settings")
    # Note: This would need a general settings endpoint
    st.info("Reranker can be enabled/disabled via general settings.")

    if not configs:
        st.info("No reranker configurations found.")
        return

    for config in configs:
        with st.expander(
            f"{'✅' if config.get('is_enabled') else '❌'} "
            f"{config.get('display_name') or config.get('name')} "
            f"({'Default' if config.get('is_default') else 'Alternative'})"
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Name:** {config.get('name')}")
                model = config.get("model")
                if model:
                    st.markdown(f"**Model:** {model.get('model_name')}")
                else:
                    st.markdown("**Model:** Not configured")

                if config.get("settings"):
                    st.json(config["settings"])

            with col2:
                enabled = st.checkbox(
                    "Enabled",
                    value=config.get("is_enabled", True),
                    key=f"reranker_enabled_{config['id']}"
                )
                if enabled != config.get("is_enabled"):
                    api.patch(f"/rerankers/{config['id']}", json={"is_enabled": enabled})
                    st.rerun()

                default = st.checkbox(
                    "Default",
                    value=config.get("is_default", False),
                    key=f"reranker_default_{config['id']}"
                )
                if default != config.get("is_default"):
                    api.patch(f"/rerankers/{config['id']}", json={"is_default": default})
                    st.rerun()


# =============================================================================
# General Settings Section
# =============================================================================

def render_general_settings():
    """Render general configuration settings section."""
    st.header("⚙️ General Settings")
    st.markdown("Manage system-wide configuration values.")

    # Fetch categories
    categories_data = api.get("/categories")
    if not categories_data:
        return

    categories = categories_data.get("categories", [])

    # Fetch all settings grouped by category
    by_category_data = api.get("/by-category")
    if not by_category_data:
        return

    for category in categories:
        cat_name = category.get("name")
        cat_settings = by_category_data.get(cat_name, [])

        if not cat_settings:
            continue

        with st.expander(f"📁 {cat_name}", expanded=False):
            if category.get("description"):
                st.caption(category["description"])

            for setting in cat_settings:
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown(f"**{setting.get('label') or setting.get('key')}**")
                    if setting.get("description"):
                        st.caption(setting["description"])

                with col2:
                    value_type = setting.get("value_type")
                    current_value = setting.get("value")
                    key = setting.get("key")

                    # Render appropriate input based on type
                    if value_type == "bool":
                        new_value = st.checkbox(
                            "Value",
                            value=bool(current_value),
                            key=f"setting_{key}",
                            label_visibility="hidden"
                        )
                    elif value_type == "int":
                        new_value = st.number_input(
                            "Value",
                            value=int(current_value or 0),
                            key=f"setting_{key}",
                            label_visibility="hidden"
                        )
                    elif value_type == "float":
                        new_value = st.number_input(
                            "Value",
                            value=float(current_value or 0.0),
                            key=f"setting_{key}",
                            label_visibility="hidden"
                        )
                    else:
                        new_value = st.text_input(
                            "Value",
                            value=str(current_value or ""),
                            key=f"setting_{key}",
                            label_visibility="hidden"
                        )

                    # Update if changed
                    if new_value != current_value:
                        if st.button("Save", key=f"save_{key}"):
                            api.patch(f"/settings/{key}", json={"value": new_value})
                            st.success(f"Updated {key}")
                            st.rerun()

    # Uncategorized settings
    uncategorized = by_category_data.get("uncategorized", [])
    if uncategorized:
        with st.expander("📁 Other Settings", expanded=False):
            for setting in uncategorized:
                st.markdown(f"**{setting.get('key')}:** {setting.get('value')}")


# =============================================================================
# Main Router
# =============================================================================

if section == "LLM Providers":
    render_llm_providers_section(llm_api)
elif section == "LLM Models":
    render_llm_models_section(llm_api)
elif section == "Agent Configurations":
    render_agent_configs_section(llm_api)
elif section == "Connection Test":
    render_connection_test_section(llm_api)
elif section == "Search Strategies":
    render_search_strategies()
elif section == "Query Mappings":
    render_query_mappings()
elif section == "Reranker Settings":
    render_reranker_settings()
elif section == "General Settings":
    render_general_settings()
