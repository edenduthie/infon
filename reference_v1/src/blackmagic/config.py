"""BlackMagicConfig: unified configuration for the package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlackMagicConfig:
    """Configuration for a BlackMagic instance.

    Minimal setup:
        config = BlackMagicConfig(schema_path="schemas/automotive.json")

    With custom persistence:
        config = BlackMagicConfig(
            schema_path="schemas/automotive.json",
            db_path="mykb.db",
            activation_threshold=0.4,
        )
    """

    # ── Schema ──────────────────────────────────────────────────────────
    schema_path: str | None = None

    # ── Encoder (splade-tiny, bundled with the package) ────────────────
    # An empty string means use the bundled model under src/blackmagic/model/
    model_name: str = ""
    max_length: int = 256
    device: str | None = None

    # ── Store (SQLite only, no cloud) ──────────────────────────────────
    db_path: str = "blackmagic.db"

    # ── Extraction thresholds (splade-tiny scores are in ~[0, 2.5]) ────
    activation_threshold: float = 0.3
    top_k_per_role: int = 3
    min_confidence: float = 0.05
    batch_size: int = 32

    # ── Importance composition weights ─────────────────────────────────
    w_activation: float = 0.3
    w_coherence: float = 0.25
    w_specificity: float = 0.2
    w_novelty: float = 0.15
    w_reinforcement: float = 0.1
    decay_rate: float = 0.01
    prune_threshold: float = 0.05

    # ── Consolidation (NEXT edges + constraints aggregation) ───────────
    consolidation_interval: int = 100  # run every N new infons; 0 disables auto

    # ── Retrieval defaults ─────────────────────────────────────────────
    default_top_k: int = 50
    persona: str = "analyst"           # default persona for valence scoring

    # ── Imagination (GA) ───────────────────────────────────────────────
    imagine_population: int = 50
    imagine_generations: int = 10
    imagine_mutation_rate: float = 0.7
    imagine_elitism: float = 0.2
    imagine_cost_weights: dict = field(
        default_factory=lambda: {"grammar": 1.0, "logic": 1.0, "health": 1.0}
    )
