"""TDD RED test for the aggregator swap ablation — Epic 02.

Spec requirement: ``Config.aggregator: Literal["typed_ikl", "uniform_mean"]``

When ``aggregator="uniform_mean"`` is selected, message passing uses a single
``W_self`` matrix and uniform-mean over neighbour embeddings instead of the
typed-IKL kernel. This ablation isolates H1 (typed-IKL vs uniform aggregation
at high hop counts).

RED phase: ``CognitionConfig`` does NOT yet have an ``aggregator`` field, so
the test that instantiates ``CognitionConfig(aggregator="uniform_mean")`` will
raise ``TypeError: __init__() got an unexpected keyword argument 'aggregator'``.

The "green section" at the bottom is protected behind the RED assertion guard
and documents the contract that must hold once the field is implemented:
  - Two reasoners (typed_ikl vs uniform_mean) produce different layer norms
    after fitting.
  - Specifically, ``sum(p.norm() for p in reasoner.parameters())`` differs
    between the two configurations.

See: openspec/changes/epic-02-synthetic-stress/tasks.md A.3
"""

from __future__ import annotations

import os
import tempfile

import pytest

from cognition.config import CognitionConfig
from experiments.ev_corpus import DOCUMENTS, setup_cognition


# ── Helper ────────────────────────────────────────────────────────────────────


def _build_fitted_reasoner(tmpdir: str, db_name: str, aggregator: str):
    """Build and fit a HypergraphReasoner with the given aggregator config.

    Parameters
    ----------
    tmpdir:
        Temporary directory for the SQLite store.
    db_name:
        Filename for the SQLite database (allows independent stores per call).
    aggregator:
        Value of ``CognitionConfig.aggregator`` to pass at construction.

    Returns
    -------
    reasoner : HypergraphReasoner
        A fitted reasoner ready for parameter-norm inspection.
    """
    from cognition.logic import HypergraphReasoner

    db_path = os.path.join(tmpdir, db_name)
    cog = setup_cognition(db_path)
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()

    # RED: CognitionConfig does not accept ``aggregator`` yet — this raises
    # TypeError. Once the field is added, the reasoner should honour it.
    cfg = CognitionConfig(aggregator=aggregator)  # <-- RED failure point

    reasoner = HypergraphReasoner(
        cog.store,
        cog.encoder,
        cog.schema,
        hidden_dim=64,
        n_layers=2,
        config=cfg,
    )
    graph = reasoner.builder.build(feature_dim=64)
    reasoner.fit(graph=graph, epochs=5, lr=1e-3, sheaf_weight=1.0, seed=42)
    cog.close()
    return reasoner


# ── RED test 1: CognitionConfig accepts the aggregator field ─────────────────


def test_cognition_config_aggregator_field_exists():
    """CognitionConfig must accept aggregator='uniform_mean' without TypeError.

    This test is RED because the dataclass has no ``aggregator`` field.
    """
    # Will raise TypeError: __init__() got an unexpected keyword argument
    # 'aggregator' until the field is added.
    cfg = CognitionConfig(aggregator="uniform_mean")
    assert cfg.aggregator == "uniform_mean"


def test_cognition_config_aggregator_default_is_typed_ikl():
    """Default aggregator must be 'typed_ikl' to preserve backward compat.

    RED: field does not exist yet, so attribute access will raise AttributeError.
    """
    cfg = CognitionConfig()
    assert cfg.aggregator == "typed_ikl"


# ── RED test 2: reasoner runs without crashing under uniform_mean ─────────────


def test_reasoner_runs_with_uniform_mean_aggregator():
    """HypergraphReasoner.fit() must complete without error when
    config.aggregator='uniform_mean'.

    RED: CognitionConfig raises TypeError before HypergraphReasoner is reached.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _build_fitted_reasoner(tmpdir, "uniform_mean.db", aggregator="uniform_mean")


# ── RED test 3: the two aggregators produce different weight norms ────────────


def test_aggregators_produce_different_weight_norms():
    """typed_ikl and uniform_mean aggregators must produce different total
    parameter L2 norms after fitting on the same corpus.

    This verifies that the two code paths actually differ — if both modes
    used the same tensors the norms would be identical, meaning the
    uniform_mean branch was never wired up.

    RED: CognitionConfig raises TypeError before the comparison can happen.

    GREEN contract (once implemented):
        norm_typed_ikl != norm_uniform_mean
        i.e. abs(norm_typed_ikl - norm_uniform_mean) > 1e-3
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reasoner_typed = _build_fitted_reasoner(
            tmpdir, "typed_ikl.db", aggregator="typed_ikl"
        )
        reasoner_uniform = _build_fitted_reasoner(
            tmpdir, "uniform_mean.db", aggregator="uniform_mean"
        )

    norm_typed = sum(p.norm().item() for p in reasoner_typed.parameters())
    norm_uniform = sum(p.norm().item() for p in reasoner_uniform.parameters())

    assert abs(norm_typed - norm_uniform) > 1e-3, (
        f"typed_ikl and uniform_mean produced identical parameter norms "
        f"({norm_typed:.6f} vs {norm_uniform:.6f}); "
        "the uniform_mean branch is likely not wired up"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  Aggregator swap ablation — RED tests")
    print("=" * 60)
    print()
    print("Expected: all three tests FAIL with TypeError because")
    print("CognitionConfig has no 'aggregator' field yet.")
    print()

    import traceback

    for fn in (
        test_cognition_config_aggregator_field_exists,
        test_cognition_config_aggregator_default_is_typed_ikl,
        test_reasoner_runs_with_uniform_mean_aggregator,
        test_aggregators_produce_different_weight_norms,
    ):
        try:
            fn()
            print(f"  PASS (unexpected): {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL (expected):   {fn.__name__}")
            print(f"    {type(exc).__name__}: {exc}")
    print()
    print("=" * 60)
