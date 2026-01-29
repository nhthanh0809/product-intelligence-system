"""LangGraph DAG definitions for multi-agent workflows.

Provides:
- Typed state definitions
- Sub-workflows for specialist agents
- Workflow composition utilities
"""

from src.dag.state import (
    AgentState,
    SearchState,
    ComparisonState,
    AnalysisState,
    PriceState,
    TrendState,
    RecommendState,
)
from src.dag.comparison_graph import create_comparison_workflow
from src.dag.analysis_graph import create_analysis_workflow

__all__ = [
    # State definitions
    "AgentState",
    "SearchState",
    "ComparisonState",
    "AnalysisState",
    "PriceState",
    "TrendState",
    "RecommendState",
    # Workflows
    "create_comparison_workflow",
    "create_analysis_workflow",
]
