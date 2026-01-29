#!/usr/bin/env python3
"""
Run All Search Flow Experiments

Master script to run all search type experiments and produce comprehensive report.

Usage:
    python scripts/research-search-flows/run_all_experiments.py --eval-data data/eval/datasets/level3_retrieval_evaluation.json
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from base import (
    SearchConfig,
    DatabaseClients,
    load_evaluation_data,
    run_experiment,
    print_experiment_results,
    ExperimentResult,
    get_search_flows_config,
)
import config as cfg

# Import all strategies
from semantic_experiments import (
    BasicSemanticStrategy,
    ParentOnlySemanticStrategy,
    ParentChildAggregationStrategy,
    MultiQuerySemanticStrategy,
    PreprocessedSemanticStrategy,
    OverfetchRerankStrategy,
    CombinedSemanticStrategy,
    AggressiveMultiQueryStrategy,
    HybridBoostSemanticStrategy,
    # Format-aware strategies (CRITICAL FIX)
    FormatAwareSemanticStrategy,
    FormatAwareMultiQueryStrategy,
    FormatAwareHybridStrategy,
    # Optimized strategy
    OptimizedSemanticStrategy,
    # Reranker strategies
    SemanticRerankerStrategy,
    ParentOnlyRerankerStrategy,
    CombinedRerankerStrategy,
)

from keyword_experiments import (
    BasicKeywordStrategy,
    HighTitleBoostStrategy,
    PhraseMatchStrategy,
    FuzzyExactCombinedStrategy,
    CrossFieldsStrategy,
    AllFieldsSmartBoostStrategy,
    CombinedKeywordStrategy,
    ModelNumberAwareStrategy,
    BrandModelPriorityStrategy,  # BEST for brand+model queries
    OptimizedKeywordStrategy,
)

from hybrid_experiments import (
    BasicRRFStrategy,
    RRFK20Strategy,
    RRFK100Strategy,
    WeightedFusionStrategy,
    SemanticFirstStrategy,
    AdaptiveFusionStrategy,
    CombinedHybridStrategy,
    KeywordPriorityHybridStrategy,  # BEST PERFORMER - MRR 0.9126
    OptimizedHybridStrategy,
    # Reranker strategies
    HybridRerankerStrategy,
    WeightedFusionRerankerStrategy,
)

from section_experiments import (
    BasicSectionStrategy,
    SectionWithFallbackStrategy,
    CrossSectionBoostStrategy,
    SectionAwareQueryStrategy,
    HierarchicalSectionStrategy,
    CombinedSectionStrategy,
    MultiSignalSectionStrategy,
    TitleFocusedSectionStrategy,
    OptimizedSectionStrategy,  # BEST for section queries
    # Reranker strategies
    SectionRerankerStrategy,
    MultiSignalRerankerStrategy,
)

logger = structlog.get_logger()


def generate_report(results: list[ExperimentResult], output_path: Path):
    """Generate comprehensive JSON report."""
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_experiments": len(results),
        "summary": {},
        "by_search_type": {},
        "recommendations": {},
    }

    # Group by search type
    by_type = {}
    for r in results:
        if r.search_type not in by_type:
            by_type[r.search_type] = []
        by_type[r.search_type].append(r)

    # Find best strategy per search type
    for search_type, type_results in by_type.items():
        def composite_score(r: ExperimentResult) -> float:
            m = r.metrics
            return (
                m.recall_at_10 * 0.25 +
                m.precision_at_10 * 0.15 +
                m.mrr * 0.25 +
                m.ndcg_at_10 * 0.20 +
                m.hit_rate * 0.15
            )

        sorted_results = sorted(type_results, key=composite_score, reverse=True)
        best = sorted_results[0]

        report["by_search_type"][search_type] = {
            "strategies_tested": len(type_results),
            "best_strategy": best.strategy_name,
            "best_description": best.strategy_description,
            "best_metrics": {
                "recall_at_5": best.metrics.recall_at_5,
                "recall_at_10": best.metrics.recall_at_10,
                "precision_at_5": best.metrics.precision_at_5,
                "precision_at_10": best.metrics.precision_at_10,
                "mrr": best.metrics.mrr,
                "ndcg_at_10": best.metrics.ndcg_at_10,
                "hit_rate": best.metrics.hit_rate,
                "latency_p50_ms": best.metrics.latency_p50_ms,
                "latency_p95_ms": best.metrics.latency_p95_ms,
                "composite_score": composite_score(best),
            },
            "all_strategies": [
                {
                    "name": r.strategy_name,
                    "description": r.strategy_description,
                    "recall_at_10": r.metrics.recall_at_10,
                    "mrr": r.metrics.mrr,
                    "ndcg_at_10": r.metrics.ndcg_at_10,
                    "hit_rate": r.metrics.hit_rate,
                    "composite_score": composite_score(r),
                }
                for r in sorted_results
            ],
        }

        # Generate recommendation
        report["recommendations"][search_type] = {
            "use_strategy": best.strategy_name,
            "reason": best.strategy_description,
            "expected_improvement": f"{(composite_score(best) - composite_score(sorted_results[-1])) * 100:.1f}% over baseline",
        }

    # Overall summary
    all_scores = [
        report["by_search_type"][st]["best_metrics"]["composite_score"]
        for st in report["by_search_type"]
    ]
    report["summary"] = {
        "average_best_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "search_types_evaluated": list(by_type.keys()),
        "total_strategies_tested": len(results),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def get_config_defaults() -> dict:
    """Get CLI defaults from pipeline_config.yaml."""
    config = get_search_flows_config()
    data_dir = cfg.get_data_dir()
    product_count = cfg.get_count()
    pipeline_mode = cfg.get_mode()

    # Resolve eval_data path with {count} and {mode} placeholders
    eval_data = config.get("eval_data", "eval/datasets/level3_retrieval_{count}_{mode}.json")
    eval_data = eval_data.replace("{count}", str(product_count)).replace("{mode}", pipeline_mode)
    eval_data_path = data_dir / eval_data if not Path(eval_data).is_absolute() else Path(eval_data)

    # Resolve output_report path with {count} and {mode} placeholders
    output_report = config.get("output_report", "eval/search_experiments_{count}_{mode}_report.json")
    output_report = output_report.replace("{count}", str(product_count)).replace("{mode}", pipeline_mode)
    output_path = data_dir / output_report if not Path(output_report).is_absolute() else Path(output_report)

    return {
        "eval_data": str(eval_data_path),
        "output": str(output_path),
        "qdrant_host": config.get("qdrant_host", "localhost"),
        "qdrant_port": config.get("qdrant_port", 6333),
        "es_host": config.get("elasticsearch_host", "localhost"),
        "es_port": config.get("elasticsearch_port", 9200),
        "ollama_url": config.get("ollama_url", "http://localhost:8010"),
        "search_types": config.get("search_types", "all"),
        "verbose": config.get("verbose", False),
        "product_count": product_count,
        "pipeline_mode": pipeline_mode,
    }


def print_recommendations(report: dict):
    """Print implementation recommendations."""
    print("\n" + "=" * 80)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 80)

    for search_type, rec in report["recommendations"].items():
        print(f"\n{search_type.upper()} SEARCH:")
        print(f"  Strategy: {rec['use_strategy']}")
        print(f"  Reason: {rec['reason']}")
        print(f"  Expected: {rec['expected_improvement']}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
1. Review the detailed results in the report JSON file
2. For each search type, examine the winning strategy implementation
3. Update the vector-store service code:
   - semantic.py: Apply combined strategy techniques
   - keyword.py: Apply best field boosting and query structure
   - hybrid.py: Apply optimal fusion parameters
4. Re-run Level 3 evaluation to verify improvements
""")


# Get defaults from config at module load time
_config_defaults = get_config_defaults()


@click.command()
@click.option(
    "--eval-data",
    type=click.Path(exists=True),
    default=None,
    help=f"Evaluation dataset JSON file (default: from config)",
)
@click.option("--qdrant-host", default=None, help="Qdrant host (default: from config)")
@click.option("--qdrant-port", default=None, type=int, help="Qdrant port (default: from config)")
@click.option("--es-host", default=None, help="Elasticsearch host (default: from config)")
@click.option("--es-port", default=None, type=int, help="Elasticsearch port (default: from config)")
@click.option("--ollama-url", default=None, help="Ollama service URL (default: from config)")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output report JSON file (default: from config)",
)
@click.option("--verbose", is_flag=True, default=None, help="Verbose output (default: from config)")
@click.option(
    "--search-types",
    default=None,
    help="Comma-separated search types to test (semantic,keyword,hybrid,section) or 'all' (default: from config)",
)
@click.option(
    "--mode",
    type=click.Choice(["original", "enrich", "auto"]),
    default="auto",
    help="Pipeline mode: 'original' (no genAI fields), 'enrich' (with genAI fields), 'auto' (from config)",
)
def main(
    eval_data: str | None,
    qdrant_host: str | None,
    qdrant_port: int | None,
    es_host: str | None,
    es_port: int | None,
    ollama_url: str | None,
    output: str | None,
    verbose: bool | None,
    search_types: str | None,
    mode: str,
):
    # Apply config defaults for unset options
    eval_data = eval_data or _config_defaults["eval_data"]
    qdrant_host = qdrant_host or _config_defaults["qdrant_host"]
    qdrant_port = qdrant_port or _config_defaults["qdrant_port"]
    es_host = es_host or _config_defaults["es_host"]
    es_port = es_port or _config_defaults["es_port"]
    ollama_url = ollama_url or _config_defaults["ollama_url"]
    output = output or _config_defaults["output"]
    verbose = verbose if verbose is not None else _config_defaults["verbose"]
    search_types = search_types or _config_defaults["search_types"]

    # Determine pipeline mode
    from base import get_pipeline_mode
    if mode == "auto":
        pipeline_mode = get_pipeline_mode()
    else:
        pipeline_mode = mode

    print("=" * 80)
    print("COMPREHENSIVE SEARCH FLOW EXPERIMENTS")
    print("=" * 80)
    print(f"Pipeline Mode: {pipeline_mode.upper()}")
    if pipeline_mode == "original":
        print("  Fields: Core fields only (no genAI_* fields)")
    else:
        print("  Fields: Core + genAI fields")
    print(f"Evaluation data: {eval_data}")
    print(f"Output report: {output}")
    print()

    # Configuration
    config = SearchConfig(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        es_host=es_host,
        es_port=es_port,
        ollama_url=ollama_url,
        pipeline_mode=pipeline_mode,
    )

    # Load evaluation data
    eval_path = Path(eval_data)
    all_queries = load_evaluation_data(eval_path)
    print(f"Total evaluation queries: {len(all_queries)}")

    # Determine which search types to test
    if search_types == "all":
        types_to_test = ["semantic", "keyword", "hybrid", "section"]
    else:
        types_to_test = [t.strip() for t in search_types.split(",")]

    print(f"Search types to test: {types_to_test}")

    # Initialize clients
    clients = DatabaseClients(config)

    # Define all strategies by search type
    all_strategies = {
        "semantic": [
            BasicSemanticStrategy(clients, config),
            ParentOnlySemanticStrategy(clients, config),
            ParentChildAggregationStrategy(clients, config),
            MultiQuerySemanticStrategy(clients, config),
            PreprocessedSemanticStrategy(clients, config),
            OverfetchRerankStrategy(clients, config),
            CombinedSemanticStrategy(clients, config),
            AggressiveMultiQueryStrategy(clients, config),
            HybridBoostSemanticStrategy(clients, config),
            # Format-aware strategies (CRITICAL FIX for embedding mismatch)
            FormatAwareSemanticStrategy(clients, config),
            FormatAwareMultiQueryStrategy(clients, config),
            FormatAwareHybridStrategy(clients, config),
            # Optimized strategy (BEST for semantic-only)
            OptimizedSemanticStrategy(clients, config),
            # Reranker strategies
            SemanticRerankerStrategy(clients, config),
            ParentOnlyRerankerStrategy(clients, config),
            CombinedRerankerStrategy(clients, config),
        ],
        "keyword": [
            BasicKeywordStrategy(clients, config),
            HighTitleBoostStrategy(clients, config),
            PhraseMatchStrategy(clients, config),
            FuzzyExactCombinedStrategy(clients, config),
            CrossFieldsStrategy(clients, config),
            AllFieldsSmartBoostStrategy(clients, config),
            CombinedKeywordStrategy(clients, config),
            ModelNumberAwareStrategy(clients, config),
            BrandModelPriorityStrategy(clients, config),  # BEST for brand+model queries
            OptimizedKeywordStrategy(clients, config),
        ],
        "hybrid": [
            BasicRRFStrategy(clients, config),
            RRFK20Strategy(clients, config),
            RRFK100Strategy(clients, config),
            WeightedFusionStrategy(clients, config),
            SemanticFirstStrategy(clients, config),
            AdaptiveFusionStrategy(clients, config),
            CombinedHybridStrategy(clients, config),
            KeywordPriorityHybridStrategy(clients, config),  # BEST PERFORMER - MRR 0.9126
            OptimizedHybridStrategy(clients, config),
            # Reranker strategies
            HybridRerankerStrategy(clients, config),
            WeightedFusionRerankerStrategy(clients, config),
        ],
        "section": [
            BasicSectionStrategy(clients, config),
            SectionWithFallbackStrategy(clients, config),
            CrossSectionBoostStrategy(clients, config),
            SectionAwareQueryStrategy(clients, config),
            HierarchicalSectionStrategy(clients, config),
            CombinedSectionStrategy(clients, config),
            MultiSignalSectionStrategy(clients, config),
            TitleFocusedSectionStrategy(clients, config),
            OptimizedSectionStrategy(clients, config),  # BEST for section queries
            # Reranker strategies
            SectionRerankerStrategy(clients, config),
            MultiSignalRerankerStrategy(clients, config),
        ],
    }

    async def run_all():
        all_results = []

        for search_type in types_to_test:
            if search_type not in all_strategies:
                print(f"Unknown search type: {search_type}")
                continue

            # Get queries for this search type
            type_queries = [q for q in all_queries if q.search_type == search_type]

            # Add section filter for section queries
            if search_type == "section":
                for q in type_queries:
                    if q.target_section:
                        q.filters = q.filters or {}
                        q.filters["section"] = q.target_section

            print(f"\n{'='*60}")
            print(f"Testing {search_type.upper()} search ({len(type_queries)} queries)")
            print(f"{'='*60}")

            if not type_queries:
                print(f"  No queries for {search_type}")
                continue

            strategies = all_strategies[search_type]

            for strategy in strategies:
                print(f"  Running: {strategy.name}...", end=" ", flush=True)
                try:
                    result = await run_experiment(
                        strategy,
                        type_queries,
                        search_type=search_type,
                        verbose=verbose,
                    )
                    all_results.append(result)
                    print(f"R@10={result.metrics.recall_at_10:.3f}, MRR={result.metrics.mrr:.3f}")
                except Exception as e:
                    print(f"FAILED: {e}")

        await clients.close()
        return all_results

    # Run experiments
    results = asyncio.run(run_all())

    if not results:
        print("No results generated!")
        return

    # Print comparison
    print_experiment_results(results)

    # Generate report
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, output_path)

    # Print recommendations
    print_recommendations(report)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
