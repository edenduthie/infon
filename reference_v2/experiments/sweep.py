"""Sweep harness: Cartesian-product YAML configs through the runner.

This is the green-phase implementation for Stage A.6b (infon-6o3.11). It
iterates a YAML ``sweep:`` block over the four primary axes (coherence
weight, fusion rule, decisive top-k, seed), invokes ``run.run`` once per
cell, and writes ``aggregate.json`` summarising mean/std for each
diagnostic query's m(Θ) and a polarity-correctness flag per row.

Stage B's 480-cell collapse sweep (6 coherences × 4 rules × 4 caps × 5
seeds) is the load-bearing consumer; this module is the entry point for
that experiment.

YAML schema
-----------
::

    name: str
    version: str           # optional, free
    rationale: str         # optional, free
    activation_threshold: float
    log_per_infon_masses: bool
    sweep:
      coherence_weight: [<float>, ...]
      fusion_rule:       [<str>,   ...]
      decisive_top_k:    [<int>,   ...]
      seed:              [<int>,   ...]   # singular `seed:` here

The cell config is the top-level config with each axis value overlaid as
a top-level field; the singular ``seed`` is converted to a single-seed
``seeds: [<int>]`` for ``run.run``.

Aggregate schema
----------------
::

    [
      {
        "coherence_weight": <float>,
        "fusion_rule": <str>,
        "decisive_top_k": <int>,
        "n_seeds": <int>,
        "queries": {
          "<query_name>": {
            "mean_mass": [m_S, m_R, m_U, m_Theta],
            "std_mass":  [s_S, s_R, s_U, s_Theta],
            "polarity_correct": <bool>   # all seeds matched expected verdict
          },
          ...
        }
      },
      ...
    ]

The list-of-rows shape is the canonical one; the test fixture accepts
either ``[rows...]`` or ``{"rows": [rows...]}`` (per A.6's contract),
so future revisions can wrap in a dict without breaking consumers.

References
----------
- openspec/changes/epic-01-stabilize-theta/spec.md
  Requirement: Configuration Sweeps
- tests/test_experiment_runner.py::test_sweep_produces_per_cell_jsons
- A.6 task notes (infon-6o3.10): aggregate schema accepts list or dict.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .run import ConfigError, run as run_one


__all__ = ["run_sweep", "SWEEP_AXES"]


# Order matters: it determines the per-cell config name suffix layout
# (deterministic across platforms because we never iterate dict keys
# from the YAML).
SWEEP_AXES: tuple[str, ...] = (
    "coherence_weight",
    "fusion_rule",
    "decisive_top_k",
    "seed",
)


# Stage A acceptance gate (epic infon-6o3): Toyota and Honda → SUPPORTS
# on the EV corpus. Tesla expanded production and CATL produces batteries
# in the EV corpus → both also SUPPORTS. The polarity-correctness flag
# in aggregate.json is computed against this map.
EXPECTED_VERDICTS: dict[str, str] = {
    "toyota": "SUPPORTS",
    "honda": "SUPPORTS",
    "tesla": "SUPPORTS",
    "catl": "SUPPORTS",
}


def run_sweep(
    config_path: str | Path, output_dir: str | Path,
) -> Path:
    """Run a Cartesian sweep over a YAML ``sweep:`` block.

    Parameters
    ----------
    config_path : str | Path
        Path to a sweep YAML (see module docstring schema).
    output_dir : str | Path
        Directory to write per-cell JSONs and ``aggregate.json``.

    Returns
    -------
    Path
        Path to ``aggregate.json``.

    Raises
    ------
    ConfigError
        If the YAML is missing the ``sweep:`` block or any required axis.
    """
    sweep_cfg = _load_sweep(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_block = sweep_cfg["sweep"]
    axis_values = [sweep_block[axis] for axis in SWEEP_AXES]

    # Per-cell run. Each cell carries a single seed (seed lives inside
    # the sweep block, so each sweep cell pins exactly one seed).
    for cell in itertools.product(*axis_values):
        cell_dict = dict(zip(SWEEP_AXES, cell))
        cell_yaml_path = _materialise_cell_yaml(
            sweep_cfg, cell_dict, output_dir,
        )
        run_one(config_path=str(cell_yaml_path), output_dir=str(output_dir))
        # The cell YAML is a build artefact, not a deliverable: remove
        # so output_dir contains only per-cell JSONs + aggregate.json.
        cell_yaml_path.unlink()

    aggregate_path = _write_aggregate(sweep_cfg, output_dir)
    return aggregate_path


# ── YAML loading ─────────────────────────────────────────────────────


def _load_sweep(config_path: str | Path) -> dict[str, Any]:
    """Load + validate a sweep YAML.

    Returns the loaded mapping with the ``sweep:`` block normalised
    (every required axis present and a non-empty list).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"sweep config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"sweep config must be a YAML mapping; got "
            f"{type(raw).__name__} in {config_path}"
        )
    if "name" not in raw or not isinstance(raw["name"], str):
        raise ConfigError(
            f"sweep config missing required `name: str`; got "
            f"{raw.get('name')!r}"
        )
    if "sweep" not in raw or not isinstance(raw["sweep"], dict):
        raise ConfigError(
            f"sweep config missing the `sweep:` mapping; "
            f"got {raw.get('sweep')!r}"
        )

    sweep_block = raw["sweep"]
    for axis in SWEEP_AXES:
        if axis not in sweep_block:
            raise ConfigError(
                f"sweep block missing required axis `{axis}`"
            )
        values = sweep_block[axis]
        if not isinstance(values, list) or not values:
            raise ConfigError(
                f"sweep axis `{axis}` must be a non-empty list; "
                f"got {values!r}"
            )

    return raw


def _materialise_cell_yaml(
    sweep_cfg: dict[str, Any],
    cell: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write a per-cell YAML config to ``output_dir`` and return its path.

    The cell config is the sweep config with the four axis values
    overlaid as top-level fields and the ``sweep:`` block removed. The
    ``name`` becomes ``<sweep_name>__<axis_summary>`` so per-cell JSONs
    written by ``run.run`` have a unique, sortable filename.
    """
    cell_cfg: dict[str, Any] = {
        k: v for k, v in sweep_cfg.items() if k != "sweep"
    }
    # Per-cell name: prefix + axis suffix. Includes seed so the per-cell
    # JSON written by run.run (``<name>__seed=<seed>.json``) is unique
    # across the sweep without collision.
    base_name = sweep_cfg["name"]
    cell_cfg["name"] = (
        f"{base_name}"
        f"__cw={cell['coherence_weight']}"
        f"__fr={cell['fusion_rule']}"
        f"__tk={cell['decisive_top_k']}"
    )
    cell_cfg["coherence_weight"] = cell["coherence_weight"]
    cell_cfg["fusion_rule"] = cell["fusion_rule"]
    cell_cfg["decisive_top_k"] = cell["decisive_top_k"]
    # Sweep cells pin one seed each; convert to the runner's canonical
    # ``seeds: [<int>]`` shape.
    cell_cfg["seeds"] = [cell["seed"]]
    cell_cfg.pop("seed", None)

    cell_yaml_path = output_dir / f"_cell_{cell_cfg['name']}__seed={cell['seed']}.yaml"
    cell_yaml_path.write_text(yaml.safe_dump(cell_cfg), encoding="utf-8")
    return cell_yaml_path


# ── Aggregation ──────────────────────────────────────────────────────


def _write_aggregate(
    sweep_cfg: dict[str, Any], output_dir: Path,
) -> Path:
    """Read every per-cell JSON in ``output_dir`` and write aggregate.json.

    One aggregate row per (coherence_weight, fusion_rule, decisive_top_k)
    triple, summarising mean/std across the seeds in that cell.
    """
    sweep_block = sweep_cfg["sweep"]
    coherences = sweep_block["coherence_weight"]
    rules = sweep_block["fusion_rule"]
    top_ks = sweep_block["decisive_top_k"]
    seeds = sweep_block["seed"]

    # Index every per-cell report by its (cw, fr, tk) triple → list of
    # JSON dicts (one per seed).
    by_triple: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for report_path in sorted(output_dir.glob("*.json")):
        if report_path.name == "aggregate.json":
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cfg = report["config"]
        triple = (
            cfg["coherence_weight"],
            cfg["fusion_rule"],
            cfg["decisive_top_k"],
        )
        by_triple.setdefault(triple, []).append(report)

    rows: list[dict[str, Any]] = []
    # Iterate axes in declared order so aggregate.json row order is
    # deterministic across platforms (avoids relying on dict iteration
    # of by_triple, which would key on float repr).
    for cw in coherences:
        for fr in rules:
            for tk in top_ks:
                triple = (cw, fr, tk)
                cell_reports = by_triple.get(triple, [])
                if not cell_reports:
                    # Defensive: a missing triple is a sweep bug; surface
                    # rather than silently emit a half-filled row.
                    raise ConfigError(
                        f"sweep produced no reports for triple {triple!r}; "
                        f"expected {len(seeds)} seed(s)"
                    )
                rows.append(
                    _aggregate_one_triple(
                        coherence_weight=cw,
                        fusion_rule=fr,
                        decisive_top_k=tk,
                        cell_reports=cell_reports,
                    )
                )

    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return aggregate_path


def _aggregate_one_triple(
    *,
    coherence_weight: float,
    fusion_rule: str,
    decisive_top_k: int,
    cell_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean/std for each query's mass + polarity-correctness flag."""
    # All cell reports share the same query set (asserted invariant of
    # the runner). Use the first report as the canonical key list.
    query_names = list(cell_reports[0]["queries"].keys())

    queries_summary: dict[str, dict[str, Any]] = {}
    for qname in query_names:
        per_seed_masses = [
            rep["queries"][qname]["mass"] for rep in cell_reports
        ]
        per_seed_verdicts = [
            rep["queries"][qname]["verdict"] for rep in cell_reports
        ]
        mean_mass = [
            statistics.fmean(component)
            for component in zip(*per_seed_masses)
        ]
        std_mass = [
            _std(component) for component in zip(*per_seed_masses)
        ]
        expected = EXPECTED_VERDICTS.get(qname)
        polarity_correct = (
            expected is not None
            and all(v == expected for v in per_seed_verdicts)
        )
        queries_summary[qname] = {
            "mean_mass": mean_mass,
            "std_mass": std_mass,
            "polarity_correct": polarity_correct,
        }

    return {
        "coherence_weight": coherence_weight,
        "fusion_rule": fusion_rule,
        "decisive_top_k": decisive_top_k,
        "n_seeds": len(cell_reports),
        "queries": queries_summary,
    }


def _std(values: tuple[float, ...] | list[float]) -> float:
    """Population std for n>=2; 0.0 for n<=1 (single-seed cells).

    ``statistics.stdev`` requires n>=2; for the single-seed corner the
    std is mathematically undefined but reported as 0.0 so downstream
    consumers (Stage C ranking, B.4 plot) don't have to special-case
    None. The ``n_seeds`` field on the row signals this case.
    """
    values = list(values)
    if len(values) <= 1:
        return 0.0
    # ``pstdev`` is bit-stable; sample std would also work here, but the
    # sweep treats seeds as the population of replicates, not a sample.
    return float(statistics.pstdev(values))
