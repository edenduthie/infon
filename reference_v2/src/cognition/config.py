"""CognitionConfig: unified configuration for local and cloud backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CognitionConfig:
    """Configuration for a Cognition instance.

    Minimal setup (local, schema only — no training needed):
        config = CognitionConfig(schema_path="data/schema.json")

    With saved config directory:
        config = CognitionConfig.local("models/my-domain")

    Cloud setup:
        config = CognitionConfig.aws(
            model_dir="models/my-domain",
            table="cognition-prod",
            bucket="cognition-prod-data",
        )
    """

    # Backend: "local" or "aws"
    backend: str = "local"

    # Model (splade-tiny bundled with the package — no download needed)
    model_dir: str | None = None
    model_name: str = ""  # empty = use bundled splade-tiny
    n_anchors: int = 160
    max_length: int = 256
    device: str | None = None

    # Schema
    schema_path: str | None = None

    # Local store
    db_path: str = "cognition.db"

    # AWS store
    aws_table: str = "cognition"
    aws_bucket: str = ""
    aws_region: str = "us-east-1"
    aws_lambda_function: str = "cognition-worker"

    # Extraction (splade-tiny scores are in ~[0, 2.5])
    activation_threshold: float = 0.3
    top_k_per_role: int = 3
    min_confidence: float = 0.05
    batch_size: int = 32

    # Importance weights
    w_activation: float = 0.3
    w_coherence: float = 0.25
    w_specificity: float = 0.2
    w_novelty: float = 0.15
    w_reinforcement: float = 0.1
    decay_rate: float = 0.01
    prune_threshold: float = 0.05

    # Consolidation
    consolidation_interval: int = 100  # run every N new infons

    # Query
    default_top_k: int = 50

    # Diagnostics
    # Opt-in flag for the per-infon mass logger used by Epic 01 Stage B
    # (collapse-localization diagnostic) and Epic 02 (ablation analysis).
    # When True, ``HypergraphReasoner.reason()`` is permitted to emit a
    # structured per-infon mass record on the returned ``ReasoningResult``.
    # Default is False; production callers see no change. The logger is a
    # permanent feature (per design.md Open Question), not debug-only.
    # See: openspec/changes/epic-01-stabilize-theta/spec.md
    #      Requirement: Per-Infon Mass Logging
    log_per_infon_masses: bool = False

    # Fusion cap
    # Maximum number of decisive per-infon masses fused via
    # ``combine_multiple`` inside ``HypergraphReasoner.reason()``. Default
    # 3, reduced from the previous hardcoded 5 (logic.py:1085 literal
    # ``[:5]``) per the audit's §Path 2.4 finding that with 5 confident
    # agreeing supports Dempster's rule mathematically drives m(Θ) → 0.
    # Capping at 3 (or lower) is the cheapest mechanical lever to preserve
    # some Θ. Setting ``decisive_top_k=1`` is equivalent to fusion
    # ``rule="top1"`` for any rule.
    # Trade-off: smaller top_k preserves more m(Θ) but reduces polarity
    # certainty (focal mass on S or R shrinks).
    # See: openspec/changes/epic-01-stabilize-theta/spec.md
    #      Requirement: Configurable Fusion Cap
    #      docs/publication/reproduction_audit.md §Path 2.4
    decisive_top_k: int = 3

    @classmethod
    def local(cls, model_dir: str, db_path: str = "cognition.db",
              schema_path: str | None = None, **kw) -> CognitionConfig:
        return cls(backend="local", model_dir=model_dir,
                   db_path=db_path, schema_path=schema_path, **kw)

    @classmethod
    def aws(cls, model_dir: str, table: str, bucket: str,
            region: str = "us-east-1", **kw) -> CognitionConfig:
        return cls(backend="aws", model_dir=model_dir,
                   aws_table=table, aws_bucket=bucket,
                   aws_region=region, **kw)
