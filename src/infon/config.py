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
            table="infon-prod",
            bucket="infon-prod-data",
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
    db_path: str = "infon.db"

    # AWS store
    aws_table: str = "infon"
    aws_bucket: str = ""
    aws_region: str = "us-east-1"
    aws_lambda_function: str = "infon-worker"

    # Extraction (splade-tiny scores are in ~[0, 2.5])
    activation_threshold: float = 0.3
    top_k_per_role: int = 3
    min_confidence: float = 0.05
    batch_size: int = 32
    coreference: bool = True   # resolve pronouns → actor names during extract

    # Quality filters on extracted triples. The extractor computes a
    # joint geometric-mean score across the three roles; triples below
    # `quality_threshold` are dropped entirely, and we keep at most
    # `max_triples_per_sentence` triples per sentence (the top-K by
    # score). Default settings remove most noise while preserving
    # canonical SVO triples on clean text.
    quality_threshold: float = 0.05
    max_triples_per_sentence: int = 3

    # Trained SentenceEmbedder for GNN node features + head-driven
    # triple extraction. When True, the Cognition instance will look
    # for (or lazily train) an embedder against the current schema.
    use_trained_embedder: bool = False
    embedder_model_dir: str | None = None   # default: alongside db_path
    embedder_n_synth: int = 1500             # synthetic corpus size
    embedder_epochs: int = 20
    embedder_trunk_dim: int = 256

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

    # Deterministic reproducibility. When set, threads a single seed
    # through torch, numpy, Python random, and any per-instance
    # generators. Identical to sklearn's random_state: set it once
    # at the top and every downstream .fit() / synthetic-corpus
    # generation / shuffle becomes bit-identical.
    random_state: int | None = None

    @classmethod
    def local(cls, model_dir: str, db_path: str = "infon.db",
              schema_path: str | None = None, **kw) -> CognitionConfig:
        return cls(backend="local", model_dir=model_dir,
                   db_path=db_path, schema_path=schema_path, **kw)

    @classmethod
    def aws(cls, model_dir: str, table: str, bucket: str,
            region: str = "us-east-1", **kw) -> CognitionConfig:
        return cls(backend="aws", model_dir=model_dir,
                   aws_table=table, aws_bucket=bucket,
                   aws_region=region, **kw)
