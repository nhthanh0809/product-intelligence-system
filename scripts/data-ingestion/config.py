#!/usr/bin/env python3
"""
Configuration module for data ingestion pipeline.

This module reads from pipeline_config.yaml and provides helper functions
for scripts to access configuration values with path resolution.

Usage:
    import config as cfg

    mode = cfg.get_mode()           # 'original' or 'enrich'
    count = cfg.get_count()         # global product_count
    path = cfg.get_path("01_extract_mvp", "input", "default.csv")
    setting = cfg.get_script("02a_download_html", "concurrency", 4)
"""

import os
from pathlib import Path
from typing import Any

import yaml

# Find the config file relative to this module
_CONFIG_DIR = Path(__file__).parent.parent  # scripts/
_CONFIG_FILE = _CONFIG_DIR / "pipeline_config.yaml"

# Cache for loaded config
_config_cache: dict | None = None


def _load_config() -> dict:
    """Load and cache the pipeline configuration."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {_CONFIG_FILE}\n"
            f"Please create pipeline_config.yaml in {_CONFIG_DIR}"
        )

    with open(_CONFIG_FILE, "r") as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def _get_data_dir() -> Path:
    """Get the data directory from config or auto-detect."""
    config = _load_config()
    global_config = config.get("global", {})

    # Check if explicitly set in config
    data_dir = global_config.get("data_dir")
    if data_dir:
        return Path(data_dir)

    # Auto-detect: look for common data directory locations
    project_root = _CONFIG_DIR.parent  # Product_intelligence_system/
    possible_dirs = [
        project_root / "data",
        project_root / "data-pipeline" / "data",
        Path.cwd() / "data",
    ]

    for d in possible_dirs:
        if d.exists():
            return d

    # Default to project_root/data (will be created if needed)
    return project_root / "data"


def _substitute_placeholders(value: str, count: int | None = None, mode: str | None = None) -> str:
    """Substitute {count} and {mode} placeholders in a string."""
    if not isinstance(value, str):
        return value

    if count is None:
        count = get_count()
    if mode is None:
        mode = get_mode()

    return value.replace("{count}", str(count)).replace("{mode}", mode)


def get_mode(script_name: str = None) -> str:
    """
    Get the pipeline mode ('original' or 'enrich').

    Args:
        script_name: Optional, ignored (for compatibility with older scripts)
    """
    config = _load_config()
    return config.get("global", {}).get("mode", "original")


def get_count() -> int:
    """Get the global product count."""
    config = _load_config()
    return config.get("global", {}).get("product_count", 50)


def get_indexing_strategy(script_name: str = None) -> str:
    """
    Get the indexing strategy ('parent_only', 'enrich_existing', 'full_replace').

    Args:
        script_name: Optional script name to check for override
    """
    config = _load_config()

    # Check for script-specific override first
    if script_name:
        script_config = config.get("scripts", {}).get(script_name, {})
        strategy = script_config.get("indexing_strategy")
        if strategy:
            return strategy

    return config.get("global", {}).get("indexing_strategy", "parent_only")


def get_source_csv() -> str:
    """Get the source CSV file path."""
    config = _load_config()
    source_csv = config.get("global", {}).get("source_csv", "archive/amz_ca_total_products_data_processed.csv")
    data_dir = _get_data_dir()
    return str(data_dir / source_csv)


def get_path(script_name: str, key: str = None, default: str | None = None) -> Path:
    """
    Get a path from script configuration with placeholder substitution.

    Args:
        script_name: Name of the script (e.g., "01_extract_mvp")
        key: Configuration key (e.g., "input", "output", "metrics")
        default: Default value if key not found

    Returns:
        Resolved Path with {count} and {mode} substituted

    Example:
        input_path = cfg.get_path("01_extract_mvp", "input", "raw/mvp_products.csv")
    """
    config = _load_config()
    scripts_config = config.get("scripts", {})
    script_config = scripts_config.get(script_name, {})

    # Get the value
    value = script_config.get(key, default) if key else default

    if value is None:
        raise ValueError(f"No path configured for {script_name}.{key} and no default provided")

    # Substitute placeholders
    value = _substitute_placeholders(str(value))

    # Resolve relative to data directory
    path = Path(value)
    if not path.is_absolute():
        data_dir = _get_data_dir()
        path = data_dir / path

    return path


def get_script(script_name: str, key: str = None, default: Any = None) -> Any:
    """
    Get a script-specific configuration value.

    Args:
        script_name: Name of the script (e.g., "01_extract_mvp")
        key: Configuration key (e.g., "count", "concurrency")
        default: Default value if key not found

    Returns:
        Configuration value (with placeholder substitution for strings)

    Example:
        concurrency = cfg.get_script("02a_download_html", "concurrency", 4)
        columns = cfg.get_script("01_extract_mvp", "extract_columns", [])
    """
    config = _load_config()
    scripts_config = config.get("scripts", {})
    script_config = scripts_config.get(script_name, {})

    # If no key provided, return the entire script config
    if key is None:
        return script_config

    value = script_config.get(key, default)

    # Handle null values from YAML
    if value is None and default is not None:
        value = default

    # Substitute placeholders in string values
    if isinstance(value, str):
        value = _substitute_placeholders(value)

    return value


def get_global(key: str, default: Any = None) -> Any:
    """
    Get a global configuration value.

    Args:
        key: Configuration key (e.g., "product_count", "mode")
        default: Default value if key not found

    Returns:
        Configuration value
    """
    config = _load_config()
    return config.get("global", {}).get(key, default)


def get_genai_config(section: str = None) -> dict:
    """
    Get GenAI enrichment configuration from 02c_extract_with_llm.

    Args:
        section: Optional specific section (e.g., "basic_fields", "parent_fields")

    Returns:
        GenAI configuration dict or specific section
    """
    config = _load_config()
    genai_config = config.get("scripts", {}).get("02c_extract_with_llm", {}).get("genai_enrichment", {})

    if section:
        return genai_config.get(section, {})
    return genai_config


def is_genai_enabled() -> bool:
    """Check if GenAI enrichment is enabled."""
    return get_genai_config().get("enabled", False)


def get_enabled_genai_fields(section: str) -> list[str]:
    """
    Get list of enabled GenAI fields for a section.

    Args:
        section: Section name (e.g., "basic_fields", "parent_fields", "child_description_fields")

    Returns:
        List of field names that are enabled (True)
    """
    section_config = get_genai_config(section)
    return [field for field, enabled in section_config.items() if enabled]


def get_qdrant_payload_fields(script_name: str = "05_load_stores") -> dict:
    """
    Get Qdrant payload field configuration for a script.

    Args:
        script_name: Script name (default: "05_load_stores")

    Returns:
        Dict with 'parent_fields_original', 'parent_fields_enrich', 'child_fields', 'child_sections'
    """
    config = _load_config()
    script_config = config.get("scripts", {}).get(script_name, {})

    return {
        "parent_fields_original": script_config.get("qdrant_parent_fields_original", []),
        "parent_fields_enrich": script_config.get("qdrant_parent_fields_enrich", []),
        "child_fields": script_config.get("qdrant_child_fields", []),
        "child_sections": script_config.get("child_sections", []),
        "content_preview_max_length": script_config.get("content_preview_max_length", 200),
    }


def reload_config() -> None:
    """Force reload of configuration file (useful for testing)."""
    global _config_cache
    _config_cache = None
    _load_config()


# For debugging
if __name__ == "__main__":
    print("Configuration Module Test")
    print("=" * 60)
    print(f"Config file: {_CONFIG_FILE}")
    print(f"Data dir: {_get_data_dir()}")
    print(f"Mode: {get_mode()}")
    print(f"Count: {get_count()}")
    print(f"Indexing strategy: {get_indexing_strategy()}")
    print(f"Source CSV: {get_source_csv()}")
    print()
    print("Script paths:")
    print(f"  01_extract_mvp.input: {get_path('01_extract_mvp', 'input')}")
    print(f"  01_extract_mvp.output: {get_path('01_extract_mvp', 'output')}")
    print(f"  02a_download_html.output_dir: {get_path('02a_download_html', 'output_dir')}")
    print(f"  03_clean_data.output: {get_path('03_clean_data', 'output')}")
    print()
    print("Script settings:")
    print(f"  02a_download_html.concurrency: {get_script('02a_download_html', 'concurrency', 4)}")
    print(f"  02c_extract_with_llm.model: {get_script('02c_extract_with_llm', 'model')}")
    print()
    print("GenAI config:")
    print(f"  enabled: {is_genai_enabled()}")
    print(f"  basic_fields: {get_enabled_genai_fields('basic_fields')}")
    print(f"  parent_fields: {get_enabled_genai_fields('parent_fields')}")
