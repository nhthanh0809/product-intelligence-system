"""Enumerations for the data pipeline."""

from enum import Enum


class PipelineMode(str, Enum):
    """Pipeline execution mode."""

    ORIGINAL = "original"
    """Extract -> Clean -> Embed -> Load (parent nodes only)."""

    ENRICH = "enrich"
    """Original + Download -> HTML to MD -> LLM Extract (parent + 5 child nodes)."""


class IndexingStrategy(str, Enum):
    """Qdrant indexing strategy."""

    PARENT_ONLY = "parent_only"
    """Load only parent nodes (for original mode)."""

    ENRICH_EXISTING = "enrich_existing"
    """Update existing parents, add child nodes."""

    ADD_CHILD_NODE = "add_child_node"
    """Fresh load with parent + all children."""

    FULL_REPLACE = "full_replace"
    """Delete existing and load fresh (destructive)."""


class JobStatus(str, Enum):
    """Pipeline job status."""

    PENDING = "pending"
    """Job is queued but not started."""

    RUNNING = "running"
    """Job is currently executing."""

    COMPLETED = "completed"
    """Job finished successfully."""

    FAILED = "failed"
    """Job failed with error."""

    CANCELLED = "cancelled"
    """Job was cancelled by user."""

    PAUSED = "paused"
    """Job is paused (can be resumed)."""


class StageStatus(str, Enum):
    """Individual stage status within a job."""

    PENDING = "pending"
    """Stage has not started yet."""

    RUNNING = "running"
    """Stage is currently executing."""

    COMPLETED = "completed"
    """Stage finished successfully."""

    FAILED = "failed"
    """Stage failed with error."""

    SKIPPED = "skipped"
    """Stage was skipped (not needed for this mode)."""


class NodeType(str, Enum):
    """Vector node type for Qdrant."""

    PARENT = "parent"
    """Parent product node with summary embedding."""

    CHILD = "child"
    """Child node with section-specific embedding."""


class SectionType(str, Enum):
    """Product section types for child nodes."""

    DESCRIPTION = "description"
    """Product description section."""

    FEATURES = "features"
    """Product features section."""

    SPECS = "specs"
    """Technical specifications section."""

    REVIEWS = "reviews"
    """Review summary section."""

    USE_CASES = "use_cases"
    """Use cases and recommendations section."""
