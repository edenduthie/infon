"""Per-infon mass diagnostic for the Toyota/Honda Θ-collapse (Stage B.2).

This is the diagnostic that disambiguates Path 2.1 of the reproduction audit
(`docs/publication/reproduction_audit.md`): is Θ collapsing in **training** or
in **fusion**?

- If individual per-infon masses already have ``m(Θ) ≈ 0.002`` → collapse
  is in **training**; Stage E priority is regularizer redesign (E.1) and
  teacher reconstruction (E.2).
- If individual per-infon masses have ``m(Θ) ≈ 0.20–0.30`` and only the
  fused mass collapses → collapse is in **fusion**; Stage E priority is
  fusion-rule swap and top-k cap (already exposed by A.4b/A.5b).

Why fit-and-extract rather than read-from-JSON?
-----------------------------------------------
The Stage B.1 baseline JSONs (``experiments/results/baseline/baseline__seed=*.json``)
serialize only the *fused* per-query mass, not the per-infon log: the runner's
report schema is ``{config, seed, loss_trace, queries: {<name>: {verdict, mass}}}``
(see ``experiments/run.py``). The per-infon records exist on
``ReasoningResult.per_infon_masses`` (A.3b) but are not persisted.

This script therefore re-runs the same baseline configuration in-process,
extracting ``ReasoningResult.per_infon_masses`` directly. Because
``HypergraphReasoner.fit(seed=...)`` is byte-deterministic (A.2b) and the
runner's setup is replicated faithfully here, the fused query masses
produced by this script reproduce the committed baseline JSONs to within
floating-point tolerance — verified by an explicit cross-check on every
seed and query.

Inputs
------
``experiments/results/baseline/baseline__seed={42,0,1}.json`` — the
committed baseline reports. Used as the source of truth for the config
that the script replays AND as a cross-check on the fused mass.

Outputs
-------
For each ``query ∈ {toyota, honda}`` and each ``seed ∈ {42, 0, 1}``:

    experiments/results/diagnostic/<query>__seed=<seed>.json

Each output contains:

- ``per_infon_mass_distribution``: min/max/mean/median/p25/p75/count of the
  per-infon ``m(Θ)`` values for the query.
- ``bimodality``: split of per-infon masses into "decisive" (m(Θ) < 0.5)
  and "uncertain" (m(Θ) ≥ 0.5); counts and group-mean m(Θ) reported.
- ``polarity_diversity``: count of contributors whose argmax over
  ``(m(S), m(R), m(U), m(Θ))`` is each of SUPPORTS/REFUTES/UNCERTAIN/Θ.
  Surfaces A.5b's structural finding that the most-decisive (smallest-Θ)
  per-infon masses are REFUTES, while highest-relevance contributors are
  SUPPORTS.
- ``per_infon_masses``: the raw per-infon records (infon_id, mass tuple,
  relevance_score) so downstream tasks (C.4 figures, D.3 memo) can re-derive
  any statistic without re-fitting.
- ``fused_mass`` and ``cross_check_baseline_match``: an internal sanity
  probe — the fused mass produced here vs the committed baseline JSON.

Determinism
-----------
The script is a pure function of (a) the committed baseline JSONs and
(b) the corpus + reasoner code. Running it twice produces byte-equal
outputs. No randomness beyond the seeds the baseline configs already pin.

Usage
-----
From ``reference_v2/``::

    PYTHONPATH=src python3 experiments/results/diagnostic/per_infon_analysis.py

References
----------
- openspec/changes/epic-01-stabilize-theta/tasks.md (Task B.2)
- docs/publication/reproduction_audit.md §Path 2.1
- A.5b learnings (infon-6o3.9): trained reasoner's min-Θ per-infon
  masses are REFUTES; this is the diagnostic that quantifies that.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

# When invoked as ``python3 experiments/results/diagnostic/per_infon_analysis.py``
# from the ``reference_v2/`` cwd, the experiments package is reachable but
# the cognition source is not on sys.path; add it. (The runner avoids this
# by living one level up; this script is a leaf, hence the explicit add.)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Diagnostic queries we care about for B.2 (the two named in the Epic 01
# acceptance gate). The full corpus has four queries (toyota/honda/tesla/catl);
# only Toyota and Honda are load-bearing for the audit.
DIAGNOSTIC_TARGETS: tuple[str, ...] = ("toyota", "honda")

# Float tolerance for the cross-check between the script's fused mass and
# the committed baseline JSON. The runner is byte-deterministic so in
# practice the difference is exactly zero, but use a tolerance that
# allows for any future numerical-precision drift without spuriously
# tripping the consistency check.
CROSS_CHECK_TOL = 1e-9

# Bimodality split point: a per-infon mass with m(Θ) < THRESHOLD is
# "decisive" (the readout commits to one focal element), otherwise
# "uncertain" (the readout retains substantial Θ). Per task spec.
BIMODALITY_THRESHOLD = 0.5


_POLARITY_LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "UNCERTAIN", "THETA")


def _argmax_label(mass: tuple[float, float, float, float]) -> str:
    """Return the polarity label of a per-infon mass.

    Categories follow the four-vector ``(m_S, m_R, m_U, m_Θ)``; ties are
    broken by index so the result is deterministic.
    """
    return _POLARITY_LABELS[max(range(4), key=lambda i: mass[i])]


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy default).

    Used for p25/p75 in the per-infon m(Θ) distribution. Pure Python so
    the script has no numpy dependency.
    """
    if not values:
        raise ValueError("cannot compute percentile of empty list")
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _summarise_distribution(theta_values: list[float]) -> dict[str, Any]:
    """Min/max/mean/median/p25/p75/count for a list of m(Θ) values."""
    if not theta_values:
        return {
            "count": 0,
            "min": None, "max": None, "mean": None,
            "median": None, "p25": None, "p75": None,
        }
    return {
        "count": len(theta_values),
        "min": float(min(theta_values)),
        "max": float(max(theta_values)),
        "mean": float(statistics.mean(theta_values)),
        "median": float(statistics.median(theta_values)),
        "p25": _percentile(theta_values, 25.0),
        "p75": _percentile(theta_values, 75.0),
    }


def _bimodality_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Split per-infon masses into decisive (Θ<0.5) and uncertain (Θ≥0.5)."""
    decisive_thetas: list[float] = []
    uncertain_thetas: list[float] = []
    for rec in records:
        theta = rec["mass"][3]
        if theta < BIMODALITY_THRESHOLD:
            decisive_thetas.append(theta)
        else:
            uncertain_thetas.append(theta)

    def _group_summary(thetas: list[float]) -> dict[str, Any]:
        return {
            "count": len(thetas),
            "mean_theta": float(statistics.mean(thetas)) if thetas else None,
        }

    return {
        "threshold": BIMODALITY_THRESHOLD,
        "decisive": _group_summary(decisive_thetas),
        "uncertain": _group_summary(uncertain_thetas),
    }


def _polarity_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count argmax labels across per-infon masses.

    Also reports the polarity of the *single* most-decisive (smallest-Θ)
    contributor, because A.5b found that this single mass is REFUTES on
    the trained reasoner — a structural property the audit asked us to
    surface explicitly.
    """
    counts = {"SUPPORTS": 0, "REFUTES": 0, "UNCERTAIN": 0, "THETA": 0}
    for rec in records:
        counts[_argmax_label(tuple(rec["mass"]))] += 1

    if records:
        # Smallest-Θ contributor: ties broken by infon_id for determinism.
        most_decisive = min(
            records, key=lambda r: (r["mass"][3], r["infon_id"]),
        )
        most_decisive_label = _argmax_label(tuple(most_decisive["mass"]))
        most_decisive_summary = {
            "infon_id": most_decisive["infon_id"],
            "mass": list(most_decisive["mass"]),
            "polarity": most_decisive_label,
            "relevance_score": most_decisive["relevance_score"],
        }
    else:
        most_decisive_summary = None

    return {
        "argmax_counts": counts,
        "most_decisive_contributor": most_decisive_summary,
    }


def _fit_and_extract(
    config: dict[str, Any], seed: int,
) -> dict[str, Any]:
    """Re-fit the baseline config and extract per-infon masses for all queries.

    Mirrors ``experiments.run._run_one_seed`` exactly so that the fused
    mass reproduces the committed baseline JSONs bit-equal under a fixed
    seed. The only addition is that ``ReasoningResult.per_infon_masses``
    is captured (the runner discards it).

    Returns
    -------
    dict
        ``{<query_name>: {"fused_mass": [...], "verdict": str,
        "per_infon_records": [{infon_id, mass, relevance_score}, ...]}}``
    """
    # Local imports keep ConfigError-equivalent behaviour fast: if the
    # baseline JSONs are missing, the script can still print a friendly
    # error before paying the torch import cost.
    from cognition.logic import HypergraphReasoner  # noqa: WPS433
    from experiments.ev_corpus import (  # noqa: WPS433
        DIAGNOSTIC_QUERIES, DOCUMENTS, setup_cognition,
    )

    fusion_rule = config["fusion_rule"]
    decisive_top_k = int(config["decisive_top_k"])
    coherence_weight = float(config["coherence_weight"])
    activation_threshold = float(config.get("activation_threshold", 0.2))
    log_per_infon_masses = bool(config.get("log_per_infon_masses", True))
    hidden_dim = int(config.get("hidden_dim", 64))
    n_layers = int(config.get("n_layers", 2))
    fit_epochs = int(config.get("fit_epochs", 30))
    fit_lr = float(config.get("fit_lr", 1e-3))
    patience = int(config.get("patience", 8))
    grad_clip = float(config.get("grad_clip", 1.0))

    extracted: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="b2_diag_") as tmpdir:
        db_path = os.path.join(tmpdir, "store.db")
        cog = setup_cognition(db_path)
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

            reasoner.fit(
                graph=graph,
                epochs=fit_epochs,
                lr=fit_lr,
                sheaf_weight=coherence_weight,
                grad_clip=grad_clip,
                patience=patience,
                seed=seed,
            )

            for qname, qtext in DIAGNOSTIC_QUERIES.items():
                result = reasoner.reason(
                    qtext,
                    decisive_top_k=decisive_top_k,
                    fusion_rule=fusion_rule,
                )
                m = result.mass
                # Convert PerInfonMassRecord dataclasses to plain dicts
                # so the output is JSON-serialisable and stable.
                records: list[dict[str, Any]] = []
                for rec in result.per_infon_masses:
                    rec_dict = asdict(rec)
                    rec_dict["mass"] = list(rec.mass)
                    records.append(rec_dict)
                extracted[qname] = {
                    "fused_mass": [
                        float(m.supports),
                        float(m.refutes),
                        float(m.uncertain),
                        float(m.theta),
                    ],
                    "verdict": result.verdict,
                    "per_infon_records": records,
                }
        finally:
            cog.close()

    return extracted


def _cross_check_fused(
    extracted: dict[str, dict[str, Any]],
    baseline_queries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare in-process fused masses to the committed baseline JSON.

    Returns a per-query record of (max abs delta, ok flag) so the
    diagnostic JSON carries explicit evidence that this script's output
    is consistent with the audit anchor.
    """
    report: dict[str, Any] = {}
    for qname, qdata in extracted.items():
        baseline_mass = baseline_queries[qname]["mass"]
        deltas = [
            abs(a - b)
            for a, b in zip(qdata["fused_mass"], baseline_mass)
        ]
        max_delta = max(deltas) if deltas else 0.0
        report[qname] = {
            "max_abs_delta": float(max_delta),
            "tolerance": CROSS_CHECK_TOL,
            "matches": max_delta <= CROSS_CHECK_TOL,
            "baseline_mass": list(baseline_mass),
            "diagnostic_mass": list(qdata["fused_mass"]),
        }
    return report


def analyse_seed(
    seed: int, baseline_dir: Path, diagnostic_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Run the diagnostic for one seed; write per-query JSONs.

    Returns a dict ``{<query_name>: <per-query analysis>}`` for the
    aggregated cross-seed table the caller assembles.
    """
    baseline_path = baseline_dir / f"baseline__seed={seed}.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"baseline JSON not found: {baseline_path} — run B.1 first.",
        )
    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    config = baseline["config"]
    if seed not in config["seeds"]:
        # Sanity: the seed embedded in the report's config should agree.
        raise ValueError(
            f"baseline JSON seed mismatch: {baseline_path} declares "
            f"seeds={config['seeds']!r} but file is named for {seed}",
        )

    extracted = _fit_and_extract(config, seed)
    cross_check = _cross_check_fused(extracted, baseline["queries"])

    per_query_summary: dict[str, dict[str, Any]] = {}
    for qname in DIAGNOSTIC_TARGETS:
        if qname not in extracted:
            raise KeyError(
                f"diagnostic query {qname!r} missing from corpus — "
                f"check experiments/ev_corpus.py DIAGNOSTIC_QUERIES",
            )
        qdata = extracted[qname]
        records = qdata["per_infon_records"]
        theta_values = [rec["mass"][3] for rec in records]

        analysis = {
            "query_name": qname,
            "seed": seed,
            "config": {
                "name": config.get("name"),
                "coherence_weight": config["coherence_weight"],
                "fusion_rule": config["fusion_rule"],
                "decisive_top_k": config["decisive_top_k"],
                "activation_threshold": config.get(
                    "activation_threshold", 0.2,
                ),
            },
            "fused_mass": qdata["fused_mass"],
            "verdict": qdata["verdict"],
            "n_per_infon_contributors": len(records),
            "per_infon_theta_distribution": _summarise_distribution(
                theta_values,
            ),
            "bimodality": _bimodality_split(records),
            "polarity_diversity": _polarity_diversity(records),
            "cross_check_baseline_match": cross_check[qname],
            "per_infon_records": records,
        }

        out_path = diagnostic_dir / f"{qname}__seed={seed}.json"
        out_path.write_text(
            json.dumps(analysis, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        per_query_summary[qname] = analysis

    return per_query_summary


def main(
    baseline_dir: str | Path | None = None,
    diagnostic_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (42, 0, 1),
) -> dict[int, dict[str, dict[str, Any]]]:
    """Entry point: run the diagnostic for all three baseline seeds.

    Returns a nested dict ``{seed: {query: analysis}}`` so callers (or a
    REPL) can inspect the result without re-reading the JSON outputs.
    """
    if baseline_dir is None:
        baseline_dir = Path(__file__).resolve().parents[1] / "baseline"
    else:
        baseline_dir = Path(baseline_dir)
    if diagnostic_dir is None:
        diagnostic_dir = Path(__file__).resolve().parent
    else:
        diagnostic_dir = Path(diagnostic_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    aggregated: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        print(f"[per_infon_analysis] seed={seed} ...", flush=True)
        aggregated[seed] = analyse_seed(seed, baseline_dir, diagnostic_dir)

    # Print a one-line summary per (seed, query) so the script also acts
    # as a quick CLI smoke check. The committed JSONs are the source of
    # truth; this is just a TTY courtesy.
    print()
    print(
        f"{'seed':>5} {'query':>7} {'n':>3} "
        f"{'min(Θ)':>8} {'med(Θ)':>8} {'max(Θ)':>8} "
        f"{'fusedΘ':>8} {'dec':>3} {'unc':>3}"
    )
    for seed, qmap in aggregated.items():
        for qname in DIAGNOSTIC_TARGETS:
            a = qmap[qname]
            d = a["per_infon_theta_distribution"]
            b = a["bimodality"]
            print(
                f"{seed:>5} {qname:>7} {d['count']:>3} "
                f"{d['min']:>8.4f} {d['median']:>8.4f} {d['max']:>8.4f} "
                f"{a['fused_mass'][3]:>8.4f} "
                f"{b['decisive']['count']:>3} {b['uncertain']['count']:>3}"
            )

    return aggregated


if __name__ == "__main__":
    main()
