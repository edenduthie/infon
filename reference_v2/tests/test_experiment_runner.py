"""Red-phase TDD tests for the experiment runner + sweep harness.

This file is intentionally written BEFORE A.6b implements the real
``experiments.run.run`` and ``experiments.sweep.run_sweep`` entry points.
All four tests below MUST fail today; the failure modes are spelled out
in each test's docstring so A.6b knows what to make pass.

Spec:
- openspec/changes/epic-01-stabilize-theta/spec.md
  Requirements: Configuration Sweeps, Deterministic Reproduction,
                Canonical Configuration
- openspec/changes/epic-01-stabilize-theta/tasks.md A.6 (red) → A.6b (green)
- A.1 stub: reference_v2/experiments/run.py raises NotImplementedError

YAML schema authored here (this is the contract A.7 must ship — see also
tests/fixtures/runner_baseline.yaml):

    name: str                       — config identifier; the runner writes
                                      JSON to <output_dir>/<name>__seed=<seed>.json
    version: str                    — semver-style metadata
    rationale: str                  — one-line free text
    seeds: list[int]                — pinned seed list; absent ⇒ ConfigError
    coherence_weight: float         — sheaf regularizer weight
    fusion_rule: str                — {dempster, yager, murphy, top1}
    decisive_top_k: int             — fusion cap (default 3, post-A.5b)
    activation_threshold: float     — CognitionConfig.activation_threshold
    log_per_infon_masses: bool      — A.3b diagnostic log toggle

Sweep YAML schema (consumed by experiments/sweep.py — does not exist yet,
test 4 fails on ImportError until A.6b lands the module):

    name: str
    sweep:
      coherence_weight: [<floats>]
      fusion_rule: [<strs>]
      decisive_top_k: [<ints>]
      seed: [<ints>]                — note: singular `seed:` inside `sweep:`,
                                      because each cell pins one seed.

JSON report schema (locked by tests 1 + 3, consumed by Stage B/C/D):

    {
      "config": <loaded YAML dict>,
      "seed": <int>,
      "loss_trace": [<float>, ...],   — length ~30 for default epochs
      "queries": {
        "<query_name>": {
          "verdict": <str>,
          "mass":  [m_S, m_R, m_U, m_Theta]
        },
        ...
      }
    }

Both the Toyota and Honda diagnostic queries MUST be present in `queries`.

No mocks; real YAML I/O via tmp_path, real (or absent) module imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Path to the committed baseline fixture ────────────────────────────────
FIXTURE_DIR = Path(__file__).parent / "fixtures"
BASELINE_FIXTURE = FIXTURE_DIR / "runner_baseline.yaml"


def _write_yaml(path: Path, body: str) -> Path:
    """Write a YAML literal to ``path`` and return ``path``."""
    path.write_text(body)
    return path


# ──────────────────────────────────────────────────────────────────────────
# Test 1 — happy-path JSON report
# ──────────────────────────────────────────────────────────────────────────


def test_run_writes_json_report(tmp_path):
    """``run(config_path, output_dir)`` writes a JSON report with the
    documented schema.

    Phase A.6 (RED): expected failure mode is
        NotImplementedError: Implemented in A.6b
    raised by the A.1 stub at experiments/run.py:32.

    Phase A.6b (GREEN): the runner loads the YAML, fits a reasoner with the
    pinned seed, runs the diagnostic queries (Toyota + Honda + Tesla + CATL),
    and dumps the report. Per epic spec, both Toyota and Honda must be
    present and each query must carry a 4-element mass vector.
    """
    from experiments import run as runner_mod

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner_mod.run(config_path=str(BASELINE_FIXTURE), output_dir=str(output_dir))

    # File at <output_dir>/<name>__seed=<seed>.json
    expected = output_dir / "test_baseline__seed=42.json"
    assert expected.exists(), (
        f"runner did not write {expected}; output_dir contained: "
        f"{sorted(p.name for p in output_dir.iterdir())}"
    )

    report = json.loads(expected.read_text())

    # Top-level keys
    assert "config" in report, "report missing 'config' key"
    assert "seed" in report, "report missing 'seed' key"
    assert "loss_trace" in report, "report missing 'loss_trace' key"
    assert "queries" in report, "report missing 'queries' key"

    # Config round-trips the loaded YAML
    assert report["config"]["name"] == "test_baseline"
    assert report["config"]["fusion_rule"] == "dempster"
    assert report["config"]["decisive_top_k"] == 3

    # Seed round-trips the pinned value
    assert report["seed"] == 42

    # Loss trace is a list of floats with length ~30 (default epochs).
    # We tolerate a wide band because A.6b may legitimately tune epochs;
    # the bit-equal contract is locked separately by test 3.
    assert isinstance(report["loss_trace"], list)
    assert len(report["loss_trace"]) >= 5, (
        f"loss_trace too short ({len(report['loss_trace'])}); "
        "expected ~30 for default epochs"
    )
    for i, lv in enumerate(report["loss_trace"]):
        assert isinstance(lv, (int, float)), (
            f"loss_trace[{i}]={lv!r} is not numeric"
        )

    # Queries: at minimum Toyota and Honda must be present (audit's two
    # diagnostic probes); each query carries verdict + 4-element mass.
    queries = report["queries"]
    assert isinstance(queries, dict)
    diagnostic_names = set(queries.keys())
    assert any("toyota" in n.lower() for n in diagnostic_names), (
        f"no Toyota query in report; got {sorted(diagnostic_names)}"
    )
    assert any("honda" in n.lower() for n in diagnostic_names), (
        f"no Honda query in report; got {sorted(diagnostic_names)}"
    )

    for qname, qrec in queries.items():
        assert "verdict" in qrec, f"query {qname!r} missing 'verdict'"
        assert "mass" in qrec, f"query {qname!r} missing 'mass'"
        mass = qrec["mass"]
        assert isinstance(mass, list) and len(mass) == 4, (
            f"query {qname!r} mass is not a 4-vector: {mass!r}"
        )
        for j, mv in enumerate(mass):
            assert isinstance(mv, (int, float)), (
                f"query {qname!r} mass[{j}]={mv!r} is not numeric"
            )
        # DS-axiom invariant: m(S)+m(R)+m(U)+m(Θ) = 1 within float tol
        total = sum(mass)
        assert abs(total - 1.0) < 1e-3, (
            f"query {qname!r} mass does not sum to 1 (sum={total}, mass={mass})"
        )


# ──────────────────────────────────────────────────────────────────────────
# Test 2 — runner refuses to start without a seed
# ──────────────────────────────────────────────────────────────────────────


def test_run_refuses_missing_seed(tmp_path):
    """A YAML config lacking both ``seed:`` and ``seeds:`` SHALL raise
    a configuration error.

    Spec: openspec/changes/epic-01-stabilize-theta/spec.md
          Requirement: Deterministic Reproduction —
          "The experiment runner ... SHALL refuse to start a sweep
           without an explicit `seed` (or `seeds:` list) in the
           configuration; an unseeded run is a configuration error."

    Phase A.6 (RED): NotImplementedError today (the stub fires before
    any validation). The wrong error class is the red-phase failure mode.

    Phase A.6b (GREEN): A.6b SHALL define a ``ConfigError`` (or use
    ``ValueError``) and validate the seed field BEFORE any work.
    """
    from experiments import run as runner_mod

    bad_yaml = _write_yaml(
        tmp_path / "no_seed.yaml",
        # Identical to baseline EXCEPT no seed/seeds field.
        "\n".join(
            [
                "name: test_no_seed",
                'version: "0.0.0"',
                'rationale: "fixture: missing-seed -> ConfigError"',
                "coherence_weight: 0.2",
                "fusion_rule: dempster",
                "decisive_top_k: 3",
                "activation_threshold: 0.2",
                "log_per_infon_masses: true",
                "",
            ]
        ),
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Accept either a custom ConfigError (preferred) or ValueError
    # (acceptable fallback). NotImplementedError, KeyError, etc. are
    # rejected — the runner must validate before doing any work.
    expected_excs: tuple[type[BaseException], ...] = (ValueError,)
    try:
        from experiments.run import ConfigError  # type: ignore[attr-defined]

        expected_excs = (ConfigError, ValueError)
    except ImportError:
        # ConfigError not yet defined in A.6 stub; ValueError-only is
        # the contract A.6b must honour.
        pass

    with pytest.raises(expected_excs):
        runner_mod.run(config_path=str(bad_yaml), output_dir=str(output_dir))


# ──────────────────────────────────────────────────────────────────────────
# Test 3 — determinism contract (byte-identical reports)
# ──────────────────────────────────────────────────────────────────────────


def test_run_is_deterministic(tmp_path):
    """Two invocations of ``run()`` with identical config + seed produce
    bit-equal loss traces and bit-equal query masses.

    Spec: openspec/changes/epic-01-stabilize-theta/spec.md
          Requirement: Deterministic Reproduction (locked at the
          runner boundary, not just at fit()).

    Phase A.6 (RED): NotImplementedError raised at the first run() call.

    Phase A.6b (GREEN): the runner uses the seeded fit() path
    (already landed in A.2b) and writes JSON deterministically.

    We compare structurally rather than via raw file bytes — JSON dict
    ordering, key insertion order, and float repr can differ across
    serializers without violating the determinism contract; loss trace
    bit-equality and per-query mass bit-equality are what Stage B/C
    actually rely on.
    """
    from experiments import run as runner_mod

    out_a = tmp_path / "a"
    out_a.mkdir()
    out_b = tmp_path / "b"
    out_b.mkdir()

    runner_mod.run(config_path=str(BASELINE_FIXTURE), output_dir=str(out_a))
    runner_mod.run(config_path=str(BASELINE_FIXTURE), output_dir=str(out_b))

    report_a_path = out_a / "test_baseline__seed=42.json"
    report_b_path = out_b / "test_baseline__seed=42.json"
    assert report_a_path.exists() and report_b_path.exists(), (
        "runner did not produce both reports for the determinism check"
    )

    report_a = json.loads(report_a_path.read_text())
    report_b = json.loads(report_b_path.read_text())

    # Loss trace: bit-identical (lists of floats; equality is IEEE-754).
    assert report_a["loss_trace"] == report_b["loss_trace"], (
        "loss_trace diverged across two seeded runs:\n"
        f"  a = {report_a['loss_trace']}\n"
        f"  b = {report_b['loss_trace']}"
    )

    # Query masses: bit-identical per query.
    assert set(report_a["queries"].keys()) == set(report_b["queries"].keys()), (
        f"query name sets diverged: "
        f"a={sorted(report_a['queries'])} vs b={sorted(report_b['queries'])}"
    )
    for qname in report_a["queries"]:
        rec_a = report_a["queries"][qname]
        rec_b = report_b["queries"][qname]
        assert rec_a["mass"] == rec_b["mass"], (
            f"query {qname!r} mass diverged across two seeded runs:\n"
            f"  a = {rec_a['mass']}\n"
            f"  b = {rec_b['mass']}"
        )
        assert rec_a["verdict"] == rec_b["verdict"], (
            f"query {qname!r} verdict diverged: "
            f"{rec_a['verdict']!r} vs {rec_b['verdict']!r}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Test 4 — sweep harness produces per-cell JSONs + aggregate
# ──────────────────────────────────────────────────────────────────────────


def test_sweep_produces_per_cell_jsons(tmp_path):
    """A 2x2x2x2 sweep produces 16 per-cell JSONs and an aggregate.json
    with 8 rows (one per (coherence, fusion_rule, top_k) triple, summarized
    over 2 seeds).

    Spec: openspec/changes/epic-01-stabilize-theta/spec.md
          Requirement: Configuration Sweeps —
          "exactly N per-cell JSONs and one aggregate.json exist;
           aggregate.json has (N / n_seeds) rows."

    Phase A.6 (RED): expected failure mode is
        ImportError: cannot import name 'sweep' from 'experiments'
    because reference_v2/experiments/sweep.py does not exist yet.

    Phase A.6b (GREEN): A.6b lands experiments/sweep.py with a
    ``run_sweep(config_path, output_dir)`` entry point that iterates the
    full Cartesian product, invokes ``run.run`` per cell, and writes
    aggregate.json.
    """
    # NOTE: this import is the load-bearing red-phase failure for test 4.
    from experiments import sweep as sweep_mod  # noqa: F401  — ImportError today

    sweep_yaml = _write_yaml(
        tmp_path / "tiny_sweep.yaml",
        "\n".join(
            [
                "name: test_sweep",
                'version: "0.0.0"',
                'rationale: "tiny 2x2x2x2 = 16-cell fixture for A.6 red test"',
                "activation_threshold: 0.2",
                "log_per_infon_masses: true",
                "sweep:",
                "  coherence_weight: [0.0, 0.2]",
                "  fusion_rule: [dempster, yager]",
                "  decisive_top_k: [1, 3]",
                "  seed: [42, 0]",
                "",
            ]
        ),
    )

    output_dir = tmp_path / "sweep_out"
    output_dir.mkdir()

    sweep_mod.run_sweep(
        config_path=str(sweep_yaml),
        output_dir=str(output_dir),
    )

    # Exactly 16 per-cell JSONs.
    per_cell = sorted(p for p in output_dir.glob("*.json") if p.name != "aggregate.json")
    assert len(per_cell) == 16, (
        f"expected 16 per-cell JSONs (2x2x2x2), got {len(per_cell)}: "
        f"{[p.name for p in per_cell]}"
    )

    # aggregate.json with 8 rows (16 cells / 2 seeds).
    agg_path = output_dir / "aggregate.json"
    assert agg_path.exists(), "sweep did not write aggregate.json"

    agg = json.loads(agg_path.read_text())
    # Aggregate is either a list-of-rows or a dict with a 'rows' key —
    # accept both shapes; A.6b chooses one and the schema is locked here.
    if isinstance(agg, list):
        rows = agg
    elif isinstance(agg, dict) and "rows" in agg:
        rows = agg["rows"]
    else:
        raise AssertionError(
            f"aggregate.json must be a list or {{rows: [...]}}, got: "
            f"{type(agg).__name__} with keys {list(agg) if isinstance(agg, dict) else 'n/a'}"
        )

    assert len(rows) == 8, (
        f"expected 8 aggregate rows (one per (coherence, fusion_rule, top_k) "
        f"triple, mean/std over 2 seeds); got {len(rows)}"
    )


if __name__ == "__main__":
    # Allow direct invocation; pytest still runs all four via -v.
    test_run_writes_json_report(Path("/tmp/t1"))
    test_run_refuses_missing_seed(Path("/tmp/t2"))
    test_run_is_deterministic(Path("/tmp/t3"))
    test_sweep_produces_per_cell_jsons(Path("/tmp/t4"))
