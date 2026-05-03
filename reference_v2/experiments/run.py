"""Experiment runner: load YAML config, fit a seeded reasoner, write JSON.

This is the green-phase implementation for Stage A.6b (infon-6o3.11). It
replaces the A.1 stub (``raise NotImplementedError``) with a real entry
point that the Stage B sweep harness (``experiments.sweep``) and the
Stage B/C/D analyses depend on.

Contracts (locked by ``tests/test_experiment_runner.py`` + Epic 01 spec.md):

YAML config schema
------------------
Required:
  name: str                       — config identifier; the runner writes
                                    JSON to ``<output_dir>/<name>__seed=<seed>.json``.
  coherence_weight: float         — sheaf coherence regulariser weight
                                    (passed to ``HypergraphReasoner.fit``
                                    as ``sheaf_weight``).
  fusion_rule: str                — one of {"dempster","yager","murphy","top1"}.
  decisive_top_k: int             — fusion cap (>=1).

Required (one of):
  seeds: list[int]                — pinned seeds; the runner iterates them
                                    and writes one JSON per seed.
  seed: int                       — single-seed shorthand; equivalent to
                                    ``seeds: [seed]``.

Optional:
  version: str                    — semver-style metadata.
  rationale: str                  — free-text annotation.
  activation_threshold: float     — passed to ``CognitionConfig`` (default 0.2).
  log_per_infon_masses: bool      — toggles A.3b's diagnostic log
                                    (default True for forward compat with
                                    Stage B.2; the records are emitted
                                    unconditionally today, see A.3b notes).
  hidden_dim: int                 — reasoner hidden dim (default 64).
  n_layers: int                   — message passing layers (default 2).
  fit_epochs: int                 — fit() epochs (default 30).
  fit_lr: float                   — fit() lr (default 1e-3).
  patience: int                   — fit() early-stopping patience (default 8).
  grad_clip: float                — fit() gradient clip (default 1.0).

JSON report schema
------------------
::

    {
      "config": <loaded YAML dict>,
      "seed": <int>,
      "loss_trace": [<float>, ...],
      "queries": {
        "<query_name>": {
          "verdict": <str>,
          "mass": [m_S, m_R, m_U, m_Theta]
        },
        ...
      }
    }

Output filename: ``<output_dir>/<name>__seed=<seed>.json`` (one JSON per
seed; ``run`` returns the path of the LAST file written so the sweep
harness can chain through).

Determinism
-----------
``HypergraphReasoner.fit(seed=...)`` (A.2b) pins random/numpy/torch RNG
state at the top of the call and re-initialises submodule parameters
under the seeded torch RNG. The runner only needs to (a) construct a
fresh reasoner per seed, (b) pass the seed correctly, and (c) avoid any
new sources of non-determinism (no ``set()`` iteration over masses, no
unsorted dict-driven branching). Two invocations with the same config +
seed produce bit-equal loss traces and bit-equal query masses.

References
----------
- openspec/changes/epic-01-stabilize-theta/spec.md
  Requirements: Configuration Sweeps, Canonical Configuration,
                Deterministic Reproduction, Alternative Fusion Rules
- tests/test_experiment_runner.py
- A.6 task notes (infon-6o3.10): YAML/JSON schema lock.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


__all__ = ["ConfigError", "run", "load_config", "REQUIRED_FIELDS"]


class ConfigError(ValueError):
    """Raised when an experiment YAML has missing or invalid fields.

    Subclasses ``ValueError`` so that callers using the pre-A.6b
    ``pytest.raises(ValueError)`` contract continue to work; tests/code
    that want the precise type can catch ``ConfigError`` directly.
    """


# Required top-level fields (excluding seed/seeds, validated separately
# because the spec accepts EITHER ``seed: int`` or ``seeds: list[int]``).
REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "coherence_weight",
    "fusion_rule",
    "decisive_top_k",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a runner YAML config.

    Returns
    -------
    dict
        The loaded YAML document with ``seeds`` normalised to a
        ``list[int]`` (single-seed shorthand expanded; original
        ``seed:`` field removed in favour of the canonical ``seeds:``).

    Raises
    ------
    ConfigError
        If the file is missing, is not a YAML mapping, lacks a required
        field, or has neither ``seed`` nor ``seeds``.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"config must be a YAML mapping; got {type(raw).__name__} "
            f"in {config_path}"
        )

    # Determinism gate: spec.md Requirement: Deterministic Reproduction —
    # "the runner SHALL refuse to start a sweep without an explicit seed
    # (or seeds: list) in the configuration".
    seeds = _normalise_seeds(raw)

    missing = [k for k in REQUIRED_FIELDS if k not in raw]
    if missing:
        raise ConfigError(
            f"missing required field(s) {missing!r} in config {config_path}"
        )

    # Type checks for the required fields (cheap; surface mistakes early).
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ConfigError(
            f"`name` must be a non-empty string; got {raw['name']!r}"
        )
    if not isinstance(raw["coherence_weight"], (int, float)):
        raise ConfigError(
            f"`coherence_weight` must be numeric; "
            f"got {type(raw['coherence_weight']).__name__}"
        )
    if not isinstance(raw["fusion_rule"], str):
        raise ConfigError(
            f"`fusion_rule` must be a string; "
            f"got {type(raw['fusion_rule']).__name__}"
        )
    if raw["fusion_rule"] not in {"dempster", "yager", "murphy", "top1"}:
        raise ConfigError(
            f"`fusion_rule` must be one of "
            f"{{'dempster', 'yager', 'murphy', 'top1'}}; "
            f"got {raw['fusion_rule']!r}"
        )
    if not isinstance(raw["decisive_top_k"], int) or raw["decisive_top_k"] < 1:
        raise ConfigError(
            f"`decisive_top_k` must be int >= 1; got {raw['decisive_top_k']!r}"
        )

    # Normalise: store the canonical seeds list and drop the singular form.
    raw["seeds"] = seeds
    raw.pop("seed", None)

    return raw


def _normalise_seeds(raw: dict[str, Any]) -> list[int]:
    """Return ``raw['seeds']`` as a non-empty ``list[int]``.

    Accepts either ``seeds: list[int]`` (preferred) or ``seed: int``
    (single-seed shorthand). Raises ``ConfigError`` otherwise.
    """
    if "seeds" in raw:
        seeds = raw["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError(
                f"`seeds` must be a non-empty list of ints; got {seeds!r}"
            )
        for s in seeds:
            if not isinstance(s, int):
                raise ConfigError(
                    f"`seeds` must contain ints; found {s!r} "
                    f"({type(s).__name__})"
                )
        return list(seeds)

    if "seed" in raw:
        s = raw["seed"]
        if not isinstance(s, int):
            raise ConfigError(
                f"`seed` must be an int; got {s!r} ({type(s).__name__})"
            )
        return [s]

    raise ConfigError(
        "seed or seeds field required (Deterministic Reproduction "
        "spec: the runner refuses to start an unseeded run)"
    )


def run(config_path: str | Path, output_dir: str | Path) -> Path:
    """Run a single experiment from a YAML config.

    For each seed in the config's ``seeds`` list, build a fresh
    Cognition + HypergraphReasoner, fit with that seed, run the four
    diagnostic queries (Toyota, Honda, Tesla, CATL), and dump a JSON
    report at ``<output_dir>/<name>__seed=<seed>.json``.

    Parameters
    ----------
    config_path : str | Path
        Path to a YAML configuration file (see module docstring schema).
    output_dir : str | Path
        Directory in which the JSON report(s) will be written. Must exist.

    Returns
    -------
    Path
        Path of the LAST JSON report written. The sweep harness calls
        ``run`` once per cell, so returning the most recent path lets
        callers chain through reports without rederiving the filename.

    Raises
    ------
    ConfigError
        Per ``load_config``.
    """
    cfg = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_path: Path | None = None
    for seed in cfg["seeds"]:
        last_path = _run_one_seed(cfg, seed, output_dir)

    # cfg["seeds"] is guaranteed non-empty by load_config, so last_path
    # is always set. Asserting keeps mypy happy and documents the
    # invariant.
    assert last_path is not None
    return last_path


def _run_one_seed(
    cfg: dict[str, Any], seed: int, output_dir: Path,
) -> Path:
    """Fit + evaluate a single (config, seed) cell; return the report path.

    The Cognition store and schema are written to a fresh tempdir per
    cell so concurrent invocations of ``run`` (e.g. parallel sweeps)
    cannot collide on a shared SQLite file. The tempdir is cleaned up at
    the end of the cell.
    """
    # Imports are local so that ``ConfigError`` can be raised without
    # paying the (large) torch + transformers import cost on a missing
    # config — important for fast-fail validation in the sweep harness.
    from cognition.logic import HypergraphReasoner

    from .ev_corpus import DIAGNOSTIC_QUERIES, DOCUMENTS, setup_cognition

    fusion_rule = cfg["fusion_rule"]
    decisive_top_k = int(cfg["decisive_top_k"])
    coherence_weight = float(cfg["coherence_weight"])
    activation_threshold = float(cfg.get("activation_threshold", 0.2))
    log_per_infon_masses = bool(cfg.get("log_per_infon_masses", True))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    n_layers = int(cfg.get("n_layers", 2))
    fit_epochs = int(cfg.get("fit_epochs", 30))
    fit_lr = float(cfg.get("fit_lr", 1e-3))
    patience = int(cfg.get("patience", 8))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    with tempfile.TemporaryDirectory(prefix="run_") as tmpdir:
        db_path = os.path.join(tmpdir, "store.db")
        cog = setup_cognition(db_path)
        # Apply runner-level Cognition overrides. setup_cognition() ships
        # a fixed activation_threshold=0.2; baseline configs may override.
        cog.config.activation_threshold = activation_threshold
        cog.config.log_per_infon_masses = log_per_infon_masses
        cog.config.decisive_top_k = decisive_top_k
        cog.config.fusion_rule = fusion_rule

        try:
            for doc in DOCUMENTS:
                cog.ingest([doc])
            cog.consolidate()

            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=hidden_dim, n_layers=n_layers,
                log_per_infon_masses=log_per_infon_masses,
            )
            graph = reasoner.builder.build(feature_dim=hidden_dim)

            fit_stats = reasoner.fit(
                graph=graph,
                epochs=fit_epochs,
                lr=fit_lr,
                sheaf_weight=coherence_weight,
                grad_clip=grad_clip,
                patience=patience,
                seed=seed,
            )
            loss_trace = [float(loss) for loss in fit_stats.get("losses", [])]

            # Diagnostic queries are iterated in a fixed order so the
            # report's ``queries`` dict is reproducible.
            queries_report: dict[str, dict[str, Any]] = {}
            for qname, qtext in DIAGNOSTIC_QUERIES.items():
                result = reasoner.reason(
                    qtext,
                    decisive_top_k=decisive_top_k,
                    fusion_rule=fusion_rule,
                )
                m = result.mass
                queries_report[qname] = {
                    "verdict": result.verdict,
                    "mass": [
                        float(m.supports),
                        float(m.refutes),
                        float(m.uncertain),
                        float(m.theta),
                    ],
                }
        finally:
            cog.close()

    report = {
        "config": cfg,
        "seed": seed,
        "loss_trace": loss_trace,
        "queries": queries_report,
    }
    report_path = output_dir / f"{cfg['name']}__seed={seed}.json"
    # ``sort_keys=True`` makes the JSON byte-stable across Python dict
    # insertion orders so comparing two reports byte-for-byte is a
    # legitimate determinism probe (test 3 compares structurally so
    # sorting is not required for green, but it makes future diffs
    # cleaner).
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path
