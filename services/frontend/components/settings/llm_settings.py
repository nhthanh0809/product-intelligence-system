"""LLM Settings components for provider, model, and agent configuration.

This module provides Streamlit components for managing:
- LLM Providers (Ollama, OpenAI, etc.)
- LLM Models per provider
- Agent-to-model mappings
- Connection testing
"""

import streamlit as st
from typing import Any


class LLMSettingsAPI:
    """API client wrapper for LLM settings."""

    def __init__(self, api_client):
        """Initialize with config API client.

        Args:
            api_client: ConfigAPIClient instance from Settings page
        """
        self.api = api_client

    # =========================================================================
    # Provider Operations
    # =========================================================================

    def get_providers(self) -> list[dict]:
        """Get all LLM providers."""
        data = self.api.get("/llm/providers?enabled_only=false")
        return data.get("providers", []) if data else []

    def create_provider(self, payload: dict) -> dict | None:
        """Create a new provider."""
        return self.api.post("/llm/providers", json=payload)

    def update_provider(self, provider_id: int, payload: dict) -> dict | None:
        """Update a provider."""
        return self.api.patch(f"/llm/providers/{provider_id}", json=payload)

    def delete_provider(self, provider_id: int) -> bool:
        """Delete a provider."""
        return self.api.delete(f"/llm/providers/{provider_id}") is not None

    def health_check_provider(self, provider_id: int) -> dict | None:
        """Check provider health."""
        return self.api.post(f"/llm/providers/{provider_id}/health-check")

    def test_connection(
        self,
        provider_id: int,
        model_id: int | None = None,
        test_prompt: str = "Say hello in one sentence.",
    ) -> dict | None:
        """Test provider connection by generating a response."""
        params = {"test_prompt": test_prompt}
        if model_id:
            params["model_id"] = model_id
        return self.api.post(
            f"/llm/providers/{provider_id}/test-connection",
            params=params,
        )

    # =========================================================================
    # Model Operations
    # =========================================================================

    def get_models(self, provider_id: int | None = None) -> list[dict]:
        """Get LLM models, optionally filtered by provider."""
        endpoint = "/llm/models?enabled_only=false"
        if provider_id:
            endpoint += f"&provider_id={provider_id}"
        data = self.api.get(endpoint)
        return data.get("models", []) if data else []

    def create_model(self, payload: dict) -> dict | None:
        """Create a new model."""
        return self.api.post("/llm/models", json=payload)

    def update_model(self, model_id: int, payload: dict) -> dict | None:
        """Update a model."""
        return self.api.patch(f"/llm/models/{model_id}", json=payload)

    def delete_model(self, model_id: int) -> bool:
        """Delete a model."""
        return self.api.delete(f"/llm/models/{model_id}") is not None

    # =========================================================================
    # Agent Config Operations
    # =========================================================================

    def get_agent_configs(self) -> list[dict]:
        """Get all agent model configurations."""
        data = self.api.get("/agents")
        return data.get("configs", []) if data else []

    def update_agent_config(self, config_id: int, payload: dict) -> dict | None:
        """Update an agent configuration."""
        return self.api.patch(f"/agents/{config_id}", json=payload)

    def create_agent_config(self, payload: dict) -> dict | None:
        """Create an agent configuration."""
        return self.api.post("/agents", json=payload)


def render_llm_providers_section(api: LLMSettingsAPI):
    """Render the LLM Providers management section.

    Features:
    - List all providers with status indicators
    - Add new providers
    - Edit existing providers
    - Delete providers
    - Health check
    """
    st.header("LLM Providers")
    st.markdown("Manage language model providers (Ollama, OpenAI, etc.)")

    providers = api.get_providers()

    # Add New Provider section
    with st.expander("Add New Provider", expanded=False):
        _render_provider_form(api, None)

    if not providers:
        st.info("No providers configured. Add your first provider above.")
        return

    # List existing providers
    for provider in providers:
        _render_provider_card(api, provider)


def _render_provider_form(api: LLMSettingsAPI, provider: dict | None):
    """Render add/edit form for a provider.

    Args:
        api: LLM Settings API client
        provider: Existing provider data for edit, or None for new
    """
    is_edit = provider is not None
    form_key = f"edit_provider_{provider['id']}" if is_edit else "add_provider"

    with st.form(form_key):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Name*",
                value=provider.get("name", "") if is_edit else "",
                placeholder="ollama-local",
                help="Unique identifier for this provider",
            )
            display_name = st.text_input(
                "Display Name",
                value=provider.get("display_name", "") if is_edit else "",
                placeholder="Local Ollama Server",
            )
            provider_type = st.selectbox(
                "Type*",
                ["ollama", "openai", "anthropic"],
                index=(
                    ["ollama", "openai", "anthropic"].index(provider.get("provider_type", "ollama"))
                    if is_edit else 0
                ),
            )

        with col2:
            base_url = st.text_input(
                "Base URL",
                value=provider.get("base_url", "") if is_edit else "",
                placeholder="http://localhost:11434",
                help="API endpoint URL",
            )
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-..." if not is_edit else "(unchanged)",
                help="Leave empty to keep existing key" if is_edit else "API key if required",
            )
            is_default = st.checkbox(
                "Set as Default Provider",
                value=provider.get("is_default", False) if is_edit else False,
            )
            is_enabled = st.checkbox(
                "Enabled",
                value=provider.get("is_enabled", True) if is_edit else True,
            )

        # Settings JSON
        with st.expander("Advanced Settings (JSON)"):
            settings_json = st.text_area(
                "Settings",
                value=str(provider.get("settings", {})) if is_edit else "{}",
                help="Additional settings as JSON (e.g., timeout, max_retries)",
            )

        submit_label = "Update Provider" if is_edit else "Add Provider"
        if st.form_submit_button(submit_label, type="primary"):
            if not name:
                st.error("Name is required")
                return

            payload = {
                "name": name,
                "display_name": display_name or None,
                "provider_type": provider_type,
                "base_url": base_url or None,
                "is_default": is_default,
                "is_enabled": is_enabled,
            }

            # Only include API key if provided (for edit, empty means keep existing)
            if api_key:
                payload["api_key"] = api_key

            # Parse settings JSON
            try:
                import json
                if settings_json and settings_json.strip() != "{}":
                    payload["settings"] = json.loads(settings_json.replace("'", '"'))
            except json.JSONDecodeError:
                st.error("Invalid JSON in settings")
                return

            if is_edit:
                result = api.update_provider(provider["id"], payload)
                if result:
                    st.success(f"Provider '{name}' updated!")
                    st.rerun()
            else:
                result = api.create_provider(payload)
                if result:
                    st.success(f"Provider '{name}' created!")
                    st.rerun()


def _render_provider_card(api: LLMSettingsAPI, provider: dict):
    """Render a provider card with info and actions."""
    status_icon = "✅" if provider.get("is_enabled") else "❌"
    default_badge = " (Default)" if provider.get("is_default") else ""
    health = provider.get("health_status", "unknown")
    health_icon = "🟢" if health == "healthy" else "🔴" if health == "unhealthy" else "⚪"

    with st.expander(
        f"{status_icon} {provider.get('display_name') or provider.get('name')}"
        f"{default_badge} {health_icon}"
    ):
        # Provider info
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"**Name:** `{provider.get('name')}`")
            st.markdown(f"**Type:** {provider.get('provider_type')}")
            st.markdown(f"**Base URL:** {provider.get('base_url') or 'Not set'}")

        with col2:
            st.markdown(f"**Status:** {'Enabled' if provider.get('is_enabled') else 'Disabled'}")
            st.markdown(f"**Health:** {health}")
            if provider.get("last_health_check"):
                st.markdown(f"**Last Check:** {provider.get('last_health_check')}")

        with col3:
            # Action buttons
            if st.button("🔄 Health Check", key=f"health_{provider['id']}"):
                with st.spinner("Checking..."):
                    result = api.health_check_provider(provider["id"])
                    if result:
                        st.info(f"Status: {result.get('health_status')}")

            if st.button("🧪 Test Connection", key=f"test_{provider['id']}"):
                st.session_state[f"show_test_{provider['id']}"] = True

            if st.button("🗑️ Delete", key=f"del_{provider['id']}"):
                if api.delete_provider(provider["id"]):
                    st.success("Provider deleted")
                    st.rerun()

        # Edit form (collapsible)
        with st.expander("Edit Provider"):
            _render_provider_form(api, provider)

        # Connection test section
        if st.session_state.get(f"show_test_{provider['id']}", False):
            st.divider()
            _render_inline_connection_test(api, provider)


def _render_inline_connection_test(api: LLMSettingsAPI, provider: dict):
    """Render inline connection test for a provider."""
    st.subheader("Connection Test")

    # Get models for this provider
    models = api.get_models(provider["id"])
    chat_models = [m for m in models if m.get("model_type") == "chat"]

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox(
            "Model",
            options=[m["id"] for m in chat_models] if chat_models else [None],
            format_func=lambda x: (
                next((m.get("display_name") or m["model_name"] for m in chat_models if m["id"] == x), "Default")
                if x else "Use default"
            ),
            key=f"test_model_{provider['id']}",
        )

    with col2:
        test_prompt = st.text_input(
            "Test Prompt",
            value="Say hello in one sentence.",
            key=f"test_prompt_{provider['id']}",
        )

    if st.button("Run Test", key=f"run_test_{provider['id']}", type="primary"):
        with st.spinner("Testing connection..."):
            result = api.test_connection(
                provider["id"],
                model_id=selected_model,
                test_prompt=test_prompt,
            )

            if result:
                if result.get("success"):
                    st.success(f"Connection successful! ({result.get('latency_ms', 0):.0f}ms)")
                    st.markdown(f"**Model:** `{result.get('model')}`")
                    st.markdown("**Response:**")
                    st.info(result.get("response", ""))
                    if result.get("tokens"):
                        tokens = result["tokens"]
                        st.caption(
                            f"Tokens: {tokens.get('prompt', 0)} prompt + "
                            f"{tokens.get('completion', 0)} completion = "
                            f"{tokens.get('total', 0)} total"
                        )
                else:
                    st.error(f"Connection failed: {result.get('error')}")
                    st.caption(f"Latency: {result.get('latency_ms', 0):.0f}ms")

    if st.button("Close", key=f"close_test_{provider['id']}"):
        st.session_state[f"show_test_{provider['id']}"] = False
        st.rerun()


def render_llm_models_section(api: LLMSettingsAPI):
    """Render the LLM Models management section.

    Features:
    - List models grouped by provider
    - Filter by type (chat, embedding, reranker)
    - Add/Edit/Delete models
    """
    st.header("LLM Models")
    st.markdown("Manage language models for different tasks.")

    providers = api.get_providers()
    provider_map = {p["id"]: p["name"] for p in providers}

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox(
            "Filter by Type",
            ["All", "chat", "embedding", "reranker"],
        )
    with col2:
        filter_provider = st.selectbox(
            "Filter by Provider",
            ["All"] + list(provider_map.values()),
        )

    # Add new model
    with st.expander("Add New Model", expanded=False):
        _render_model_form(api, None, providers)

    # Get and filter models
    models = api.get_models()

    if filter_type != "All":
        models = [m for m in models if m.get("model_type") == filter_type]
    if filter_provider != "All":
        provider_id = next(
            (pid for pid, name in provider_map.items() if name == filter_provider),
            None
        )
        if provider_id:
            models = [m for m in models if m.get("provider_id") == provider_id]

    if not models:
        st.info("No models found. Add a model above.")
        return

    # List models
    for model in models:
        _render_model_card(api, model, provider_map, providers)


def _render_model_form(api: LLMSettingsAPI, model: dict | None, providers: list[dict]):
    """Render add/edit form for a model."""
    is_edit = model is not None
    form_key = f"edit_model_{model['id']}" if is_edit else "add_model"

    provider_map = {p["id"]: p["name"] for p in providers}

    with st.form(form_key):
        col1, col2 = st.columns(2)

        with col1:
            provider_ids = list(provider_map.keys())
            current_provider_idx = (
                provider_ids.index(model["provider_id"])
                if is_edit and model.get("provider_id") in provider_ids
                else 0
            )
            provider_id = st.selectbox(
                "Provider*",
                options=provider_ids,
                format_func=lambda x: provider_map.get(x, str(x)),
                index=current_provider_idx,
            )

            model_name = st.text_input(
                "Model Name*",
                value=model.get("model_name", "") if is_edit else "",
                placeholder="llama3.1:8b",
                help="The actual model identifier used by the provider",
            )

            display_name = st.text_input(
                "Display Name",
                value=model.get("display_name", "") if is_edit else "",
                placeholder="Llama 3.1 8B",
            )

        with col2:
            model_types = ["chat", "embedding", "reranker"]
            current_type_idx = (
                model_types.index(model.get("model_type", "chat"))
                if is_edit else 0
            )
            model_type = st.selectbox(
                "Type*",
                model_types,
                index=current_type_idx,
            )

            description = st.text_area(
                "Description",
                value=model.get("description", "") if is_edit else "",
                placeholder="Model description...",
            )

            is_default = st.checkbox(
                "Default for this type",
                value=model.get("is_default_for_type", False) if is_edit else False,
            )

            is_enabled = st.checkbox(
                "Enabled",
                value=model.get("is_enabled", True) if is_edit else True,
            )

        # Capabilities JSON
        with st.expander("Model Capabilities"):
            col_a, col_b = st.columns(2)
            with col_a:
                max_context = st.number_input(
                    "Max Context Tokens",
                    value=(
                        model.get("capabilities", {}).get("max_context", 8192)
                        if is_edit else 8192
                    ),
                    min_value=512,
                    max_value=1000000,
                )
            with col_b:
                max_tokens = st.number_input(
                    "Max Output Tokens",
                    value=(
                        model.get("capabilities", {}).get("max_tokens", 4096)
                        if is_edit else 4096
                    ),
                    min_value=128,
                    max_value=100000,
                )

        submit_label = "Update Model" if is_edit else "Add Model"
        if st.form_submit_button(submit_label, type="primary"):
            if not model_name:
                st.error("Model name is required")
                return

            payload = {
                "provider_id": provider_id,
                "model_name": model_name,
                "display_name": display_name or None,
                "model_type": model_type,
                "description": description or None,
                "is_default_for_type": is_default,
                "is_enabled": is_enabled,
                "capabilities": {
                    "max_context": max_context,
                    "max_tokens": max_tokens,
                },
            }

            if is_edit:
                result = api.update_model(model["id"], payload)
                if result:
                    st.success(f"Model '{model_name}' updated!")
                    st.rerun()
            else:
                result = api.create_model(payload)
                if result:
                    st.success(f"Model '{model_name}' added!")
                    st.rerun()


def _render_model_card(
    api: LLMSettingsAPI,
    model: dict,
    provider_map: dict[int, str],
    providers: list[dict],
):
    """Render a model card with info and actions."""
    status_icon = "✅" if model.get("is_enabled") else "❌"
    provider_name = provider_map.get(model.get("provider_id"), "Unknown")
    default_badge = " ⭐" if model.get("is_default_for_type") else ""

    with st.expander(
        f"{status_icon} {model.get('display_name') or model.get('model_name')} "
        f"({model.get('model_type')}) - {provider_name}{default_badge}"
    ):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Model Name:** `{model.get('model_name')}`")
            st.markdown(f"**Provider:** {provider_name}")
            st.markdown(f"**Type:** {model.get('model_type')}")

            if model.get("description"):
                st.markdown(f"**Description:** {model.get('description')}")

            if model.get("capabilities"):
                caps = model["capabilities"]
                st.markdown(
                    f"**Capabilities:** {caps.get('max_context', 'N/A')} context, "
                    f"{caps.get('max_tokens', 'N/A')} output"
                )

        with col2:
            # Toggle enabled
            enabled = st.checkbox(
                "Enabled",
                value=model.get("is_enabled", True),
                key=f"model_enabled_{model['id']}",
            )
            if enabled != model.get("is_enabled"):
                api.update_model(model["id"], {"is_enabled": enabled})
                st.rerun()

            if st.button("🗑️ Delete", key=f"del_model_{model['id']}"):
                if api.delete_model(model["id"]):
                    st.success("Model deleted")
                    st.rerun()

        # Edit form
        with st.expander("Edit Model"):
            _render_model_form(api, model, providers)


def render_agent_configs_section(api: LLMSettingsAPI):
    """Render the Agent Model Configurations section.

    Features:
    - View agent-to-model mappings
    - Change model assignments
    - Enable/disable agent LLM usage
    """
    st.header("Agent Model Configurations")
    st.markdown("Configure which models each agent uses for different purposes.")

    configs = api.get_agent_configs()
    models = api.get_models()

    # Build model lookup - use display_name if available, otherwise model_name
    model_options = {
        m["id"]: f"{m.get('display_name') or m['model_name']} ({m['model_type']})"
        for m in models
    }

    if not configs:
        st.info("No agent configurations found.")

        # Add new agent config
        with st.expander("Add Agent Configuration"):
            _render_add_agent_config(api, models)
        return

    # Group by agent name
    agents: dict[str, list[dict]] = {}
    for config in configs:
        agent_name = config.get("agent_name")
        if agent_name not in agents:
            agents[agent_name] = []
        agents[agent_name].append(config)

    # Add new agent config
    with st.expander("Add Agent Configuration", expanded=False):
        _render_add_agent_config(api, models)

    # Display by agent
    for agent_name, agent_configs in sorted(agents.items()):
        with st.expander(f"🤖 {agent_name.replace('_', ' ').title()}", expanded=True):
            for config in agent_configs:
                _render_agent_config_row(api, config, model_options, models)


def _render_add_agent_config(api: LLMSettingsAPI, models: list[dict]):
    """Render form to add a new agent configuration."""
    with st.form("add_agent_config"):
        col1, col2 = st.columns(2)

        with col1:
            agent_name = st.text_input(
                "Agent Name*",
                placeholder="intent_agent",
                help="The internal agent name (e.g., intent_agent, synthesis_agent)",
            )
            purpose = st.selectbox(
                "Purpose*",
                ["primary", "fallback", "embedding"],
            )

        with col2:
            model_id = st.selectbox(
                "Model*",
                options=[m["id"] for m in models],
                format_func=lambda x: next(
                    (f"{m.get('display_name') or m['model_name']} ({m['model_type']})" for m in models if m["id"] == x),
                    str(x)
                ),
            )
            is_enabled = st.checkbox("Enabled", value=True)

        if st.form_submit_button("Add Configuration", type="primary"):
            if not agent_name:
                st.error("Agent name is required")
                return

            payload = {
                "agent_name": agent_name,
                "model_id": model_id,
                "purpose": purpose,
                "is_enabled": is_enabled,
            }

            result = api.create_agent_config(payload)
            if result:
                st.success(f"Configuration for '{agent_name}' added!")
                st.rerun()


def _render_agent_config_row(
    api: LLMSettingsAPI,
    config: dict,
    model_options: dict[int, str],
    models: list[dict],
):
    """Render a single agent configuration row."""
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**Purpose:** {config.get('purpose')}")
        current_model = config.get("model")
        if current_model:
            st.markdown(f"**Current Model:** `{current_model.get('model_name')}`")
        else:
            st.markdown("**Current Model:** Not set")

    with col2:
        # Model selection dropdown
        current_model_id = config.get("model_id")
        new_model_id = st.selectbox(
            "Change Model",
            options=[None] + list(model_options.keys()),
            format_func=lambda x: "-- Keep Current --" if x is None else model_options.get(x, str(x)),
            key=f"agent_model_{config['id']}",
            index=0,
        )

        if new_model_id and new_model_id != current_model_id:
            if st.button("Save", key=f"save_agent_{config['id']}"):
                result = api.update_agent_config(config["id"], {"model_id": new_model_id})
                if result:
                    st.success("Updated!")
                    st.rerun()

    with col3:
        enabled = st.checkbox(
            "Enabled",
            value=config.get("is_enabled", True),
            key=f"agent_enabled_{config['id']}",
        )
        if enabled != config.get("is_enabled"):
            api.update_agent_config(config["id"], {"is_enabled": enabled})
            st.rerun()


def render_connection_test_section(api: LLMSettingsAPI):
    """Render a dedicated connection test section.

    This provides a comprehensive test interface for verifying
    LLM provider connectivity.
    """
    st.header("Connection Test")
    st.markdown("Test LLM provider connections by generating a response.")

    providers = api.get_providers()
    enabled_providers = [p for p in providers if p.get("is_enabled")]

    if not enabled_providers:
        st.warning("No enabled providers available. Enable a provider first.")
        return

    # Provider selection
    col1, col2 = st.columns(2)

    with col1:
        selected_provider = st.selectbox(
            "Select Provider",
            options=[p["id"] for p in enabled_providers],
            format_func=lambda x: next(
                (p.get("display_name") or p.get("name") for p in enabled_providers if p["id"] == x),
                str(x)
            ),
        )

    # Get models for selected provider
    models = api.get_models(selected_provider)
    chat_models = [m for m in models if m.get("model_type") == "chat" and m.get("is_enabled")]

    with col2:
        if chat_models:
            selected_model = st.selectbox(
                "Select Model",
                options=[None] + [m["id"] for m in chat_models],
                format_func=lambda x: "Use Provider Default" if x is None else next(
                    (m.get("display_name") or m["model_name"] for m in chat_models if m["id"] == x), str(x)
                ),
            )
        else:
            st.warning("No chat models configured for this provider")
            selected_model = None

    # Test prompt
    test_prompt = st.text_area(
        "Test Prompt",
        value="Say hello and tell me one interesting fact about AI in two sentences.",
        height=100,
    )

    # Run test
    if st.button("Run Connection Test", type="primary", use_container_width=True):
        with st.spinner("Testing connection..."):
            result = api.test_connection(
                selected_provider,
                model_id=selected_model,
                test_prompt=test_prompt,
            )

            if result:
                st.divider()

                if result.get("success"):
                    st.success("Connection Successful!")

                    # Results in columns
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
                    with col_b:
                        tokens = result.get("tokens", {})
                        st.metric("Tokens Used", tokens.get("total", 0))

                    st.markdown("**Provider:** " + result.get("provider_name", ""))
                    st.markdown("**Model:** `" + result.get("model", "") + "`")

                    st.markdown("**Response:**")
                    st.info(result.get("response", ""))

                    # Token breakdown
                    if tokens:
                        st.caption(
                            f"Token breakdown: {tokens.get('prompt', 0)} prompt + "
                            f"{tokens.get('completion', 0)} completion"
                        )

                else:
                    st.error("Connection Failed!")
                    st.markdown("**Error:**")
                    st.error(result.get("error", "Unknown error"))

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
                    with col_b:
                        if result.get("retryable"):
                            st.warning("This error may be retryable")
            else:
                st.error("Failed to reach the API. Check if the service is running.")


def render_llm_settings_tabs(api_client):
    """Render all LLM settings in a tabbed interface.

    Args:
        api_client: ConfigAPIClient instance
    """
    api = LLMSettingsAPI(api_client)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔌 Providers",
        "🧠 Models",
        "🤖 Agent Mappings",
        "🧪 Connection Test",
    ])

    with tab1:
        render_llm_providers_section(api)

    with tab2:
        render_llm_models_section(api)

    with tab3:
        render_agent_configs_section(api)

    with tab4:
        render_connection_test_section(api)
