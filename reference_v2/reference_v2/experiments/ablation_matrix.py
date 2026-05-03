"""Ablation matrix runner for synthetic pilot scenarios.

CLI: python -m reference_v2.experiments.ablation_matrix \
        --config <yaml> --seeds <csv> --data <dir> --out <dir>

For each cell in the YAML ``cells`` list:
1. Load all scenarios from train.json / dev.json / test.json.
2. Build a dynamic schema covering all entity IDs in the corpus (0-99).
3. Create one Cognition instance with all scenario documents in a tempdir.
4. Ingest every scenario's sentences as a single document per scenario.
5. Build the hypergraph and (unless no_training=True) fit the reasoner.
6. Evaluate on test scenarios: run reason() per scenario or fall back to
   the raw DS teacher mass for the ``teacher_only`` cell.
7. Compute all 6 metrics and write one JSON per cell.

Entity numbering
----------------
Each synthetic sentence encodes a global entity ID in its text:
  "Entity 56: fact 56 is demonstrably false."
The schema created by make_synthetic_schema(100) maps entity56 → tokens
["entity 56", "Entity 56"], so querying with "entity 56" activates the
right anchor. The positional index of a scenario within a split (e.g.,
scenario_idx=0) is different from the global entity ID (e.g., 56).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reference_v2.experiments.metrics import (
    aurc,
    brier_3way,
    ece,
    polarity_accuracy_3way,
    selective_accuracy_at_coverage,
    spearman_rho,
)
from reference_v2.experiments.synthetic_schema import make_synthetic_schema

# Planted verdict strings → 3-class integer label (SUPPORTS=0, REFUTES=1, NEI=2)
PLANTED_TO_LABEL: dict[str, int] = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NEI": 2,
}


def _extract_entity_num(text: str) -> int | None:
    """Extract the global entity ID from a synthetic sentence.

    E.g. "Entity 56: fact 56 is demonstrably false." → 56.
    Returns None if no match.
    """
    m = re.match(r"Entity (\d+)", text)
    return int(m.group(1)) if m else None


def _get_scenario_entity_num(scenario: dict) -> int | None:
    """Return the global entity ID for a scenario (from first sentence)."""
    for sent in scenario["corpus_sentences"]:
        n = _extract_entity_num(sent["text"])
        if n is not None:
            return n
    return None


def _mass_to_3class_probs(mass_tuple: tuple[float, float, float, float]) -> np.ndarray:
    """Map 4-mass (m_S, m_R, m_U, m_Theta) to 3-class probability vector.

    The uncertain mass and theta mass are merged into the NEI class (index 2).
    The resulting vector is renormalised to sum to 1.
    """
    m_s, m_r, m_u, m_theta = mass_tuple
    probs = np.array([m_s, m_r, m_u + m_theta], dtype=float)
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return probs


def _verdict_from_mass(mass_tuple: tuple[float, float, float, float]) -> str:
    """Pignistic decision rule mirroring logic.py's reason() method."""
    m_s, m_r, m_u, m_theta = mass_tuple
    total_focal = m_s + m_r + m_u
    if total_focal > 0:
        pig_s = m_s + m_theta * (m_s / total_focal)
        pig_r = m_r + m_theta * (m_r / total_focal)
    else:
        pig_s = m_theta / 3
        pig_r = m_theta / 3

    if pig_r > 0.15 and pig_r > pig_s:
        return "REFUTES"
    elif pig_s > 0.25 and pig_s > pig_r:
        return "SUPPORTS"
    else:
        return "NOT ENOUGH INFO"


def _compute_teacher_mass(
    store, schema_types: dict, doc_id: str
) -> tuple[float, float, float, float]:
    """Compute the combined DS teacher mass for a scenario document from the store.

    Fetches all infons belonging to the scenario's document (by doc_id), then
    combines their teacher masses (polarity + alignment + distance + confidence)
    using Dempster's rule. Returns (m_S, m_R, m_U, m_Theta).

    This is used by the teacher_only cell, which skips GNN training and falls
    back to the raw DS signal. Querying by doc_id is necessary because the
    SPLADE encoder does not distinguish individual entity numbers (entity0
    through entity99 all share the "entity" token), so entity-key queries
    do not reliably retrieve the right infons for a specific scenario.
    """
    from cognition.dempster_shafer import (
        MassFunction,
        combine_multiple,
        mass_from_anchor_distance,
        mass_from_confidence,
        mass_from_polarity,
        mass_from_triple_alignment,
    )

    infons = store.query_infons(doc_id=doc_id, limit=50)
    if not infons:
        return (0.0, 0.0, 0.0, 1.0)  # full vacuous mass (no triples extracted)

    per_infon_masses: list[MassFunction] = []
    # Use the first infon's subject as the claim anchor (best available
    # approximation since the SPLADE encoder conflates entity IDs).
    claim_anchors = {inf.subject: 1.0 for inf in infons[:3]}

    for infon in infons:
        sources = [
            mass_from_polarity(infon),
            mass_from_triple_alignment(claim_anchors, infon, schema_types),
            mass_from_anchor_distance(claim_anchors, infon, schema_types),
            mass_from_confidence(infon),
        ]
        combined = combine_multiple(sources)
        per_infon_masses.append(combined)

    decisive = sorted(per_infon_masses, key=lambda m: m.theta)[:5]
    final_mass = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)

    return (
        float(final_mass.supports),
        float(final_mass.refutes),
        float(final_mass.uncertain),
        float(final_mass.theta),
    )


def _compute_metrics(
    per_scenario_results: list[dict],
) -> dict[str, float]:
    """Compute all 6 ablation metrics from per-scenario results.

    Parameters
    ----------
    per_scenario_results:
        List of dicts with keys: oracle_verdict, predicted_verdict,
        planted_thinness, mass ([m_S, m_R, m_U, m_Theta]).

    Returns
    -------
    dict with all metric keys required by the output schema.
    """
    n = len(per_scenario_results)
    if n == 0:
        return {
            "polarity_acc_mean": 0.0,
            "polarity_acc_std": 0.0,
            "polarity_acc_ci_low": 0.0,
            "polarity_acc_ci_high": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "aurc": 0.0,
            "sel_acc_at_50": 0.0,
            "sel_acc_at_70": 0.0,
            "sel_acc_at_90": 0.0,
            "spearman_thinness": 0.0,
            "spearman_thinness_ci_low": 0.0,
            "spearman_thinness_ci_high": 0.0,
        }

    # Ground truth labels (planted_verdict)
    true_labels = np.array([
        PLANTED_TO_LABEL.get(r["oracle_verdict"], 2) for r in per_scenario_results
    ])

    # Predicted labels (from predicted_verdict)
    pred_verdict_map = {
        "SUPPORTS": 0,
        "REFUTES": 1,
        "NOT ENOUGH INFO": 2,
        "NEI": 2,
    }
    pred_labels = np.array([
        pred_verdict_map.get(r["predicted_verdict"], 2) for r in per_scenario_results
    ])

    # 3-class probability arrays from 4-mass
    probs = np.array([
        _mass_to_3class_probs(tuple(r["mass"])) for r in per_scenario_results
    ])  # shape (n, 3)

    # --- polarity accuracy (single seed → CI collapses to point estimate) ---
    acc = polarity_accuracy_3way(pred_labels, true_labels)
    ci_low = float(acc)
    ci_high = float(acc)

    # --- ECE ---
    ece_val = ece(probs, true_labels, n_bins=15)

    # --- Brier ---
    brier_val = brier_3way(probs, true_labels)

    # --- AURC ---
    # Confidence = max probability over 3 classes; risk = 1 - max_prob
    max_probs = probs.max(axis=1)
    risk_vals = 1.0 - max_probs
    # Sort by ascending risk (descending confidence) for AURC computation
    sort_idx = np.argsort(risk_vals)
    sorted_risks = risk_vals[sort_idx]
    # Coverage increases from 0 to 1 as we include more samples
    coverages = np.linspace(1.0 / n, 1.0, n)
    aurc_val = aurc(sorted_risks, coverages)

    # --- Selective accuracy ---
    confidence_scores = max_probs
    sel_50 = selective_accuracy_at_coverage(confidence_scores, pred_labels, true_labels, 0.5)
    sel_70 = selective_accuracy_at_coverage(confidence_scores, pred_labels, true_labels, 0.7)
    sel_90 = selective_accuracy_at_coverage(confidence_scores, pred_labels, true_labels, 0.9)

    # --- Spearman rho: m_theta vs planted_thinness ---
    # H2 metric: high thinness (many witnesses) → low theta (high confidence)
    m_theta_vals = np.array([r["mass"][3] for r in per_scenario_results])
    thinness_vals = np.array([float(r["planted_thinness"]) for r in per_scenario_results])

    if len(np.unique(thinness_vals)) < 2 or len(np.unique(m_theta_vals)) < 2:
        # Not enough variance to compute correlation
        spearman_val = 0.0
        spearman_ci_low = 0.0
        spearman_ci_high = 0.0
    else:
        spearman_val = spearman_rho(m_theta_vals, thinness_vals)
        # Single-seed CI collapses to the point estimate
        spearman_ci_low = spearman_val
        spearman_ci_high = spearman_val

    return {
        "polarity_acc_mean": float(acc),
        "polarity_acc_std": 0.0,  # single seed
        "polarity_acc_ci_low": ci_low,
        "polarity_acc_ci_high": ci_high,
        "ece": float(ece_val),
        "brier": float(brier_val),
        "aurc": float(aurc_val),
        "sel_acc_at_50": float(sel_50),
        "sel_acc_at_70": float(sel_70),
        "sel_acc_at_90": float(sel_90),
        "spearman_thinness": float(spearman_val),
        "spearman_thinness_ci_low": float(spearman_ci_low),
        "spearman_thinness_ci_high": float(spearman_ci_high),
    }


def run_cell(
    cell_cfg: dict[str, Any],
    seed: int,
    data_dir: Path,
    out_dir: Path,
) -> Path:
    """Run one ablation cell for one seed and write a JSON result file.

    Parameters
    ----------
    cell_cfg:
        One cell dict from the YAML ``cells`` list.
    seed:
        The random seed to use for this run.
    data_dir:
        Directory containing train.json / dev.json / test.json.
    out_dir:
        Output directory for result JSON files.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    from cognition import Cognition, CognitionConfig
    from cognition.logic import HypergraphReasoner

    cell_name = cell_cfg["name"]
    no_training = bool(cell_cfg.get("no_training", False))
    n_layers = int(cell_cfg.get("n_layers", 2))
    hidden_dim = 64
    fusion_rule = cell_cfg["fusion_rule"]
    decisive_top_k = int(cell_cfg["decisive_top_k"])
    coherence_weight = float(cell_cfg["coherence_weight"])
    aggregator = cell_cfg.get("aggregator", "typed_ikl")
    readout = cell_cfg.get("readout", "ds_4mass")
    teacher_sources = list(cell_cfg.get("teacher_sources", [
        "polarity", "alignment", "distance", "confidence"
    ]))

    print(f"[{cell_name}] seed={seed} loading scenarios …")

    # Load all three splits; test scenarios are evaluated, rest only help build graph
    train_scenarios = json.loads((data_dir / "train.json").read_text())
    dev_scenarios = json.loads((data_dir / "dev.json").read_text())
    test_scenarios = json.loads((data_dir / "test.json").read_text())

    all_scenarios = train_scenarios + dev_scenarios + test_scenarios

    # Determine all entity IDs in the corpus so we can build a complete schema.
    # Entity IDs are global (0-99), not positional within the split.
    all_entity_nums: set[int] = set()
    for sc in all_scenarios:
        n = _get_scenario_entity_num(sc)
        if n is not None:
            all_entity_nums.add(n)
    # Schema must cover every ID from 0 to max so schema keys stay stable
    max_entity_num = max(all_entity_nums) if all_entity_nums else 99
    schema = make_synthetic_schema(max_entity_num + 1)

    # Build a mapping from scenario to its entity num and scenario index
    # (the global scenario_idx = position in the original all_scenarios list).
    scenario_entity_nums: list[int | None] = [
        _get_scenario_entity_num(sc) for sc in all_scenarios
    ]

    print(f"[{cell_name}] {len(all_scenarios)} total scenarios, "
          f"schema covers entity0–entity{max_entity_num}")

    with tempfile.TemporaryDirectory(prefix=f"ablation_{cell_name}_") as tmpdir:
        db_path = os.path.join(tmpdir, "store.db")
        schema_path = os.path.join(tmpdir, "schema.json")

        with open(schema_path, "w") as fh:
            json.dump(schema, fh)

        cog_config = CognitionConfig(
            schema_path=schema_path,
            db_path=db_path,
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
            aggregator=aggregator,
            readout=readout,
            teacher_sources=teacher_sources,
        )
        cog = Cognition(cog_config)

        try:
            # Ingest each scenario as one document keyed by its global entity ID.
            # Doc IDs use the global entity number so the store stays consistent.
            print(f"[{cell_name}] ingesting {len(all_scenarios)} documents …")
            for global_idx, (sc, ent_num) in enumerate(
                zip(all_scenarios, scenario_entity_nums)
            ):
                if ent_num is None:
                    # Fallback: use positional index as doc ID
                    doc_id = f"scenario_{global_idx:04d}"
                else:
                    doc_id = f"scenario_{ent_num:04d}"

                combined_text = " ".join(
                    sent["text"] for sent in sc["corpus_sentences"]
                )
                cog.ingest([{"id": doc_id, "text": combined_text}])

            cog.consolidate()

            # Build the hypergraph
            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                log_per_infon_masses=False,
                config=cog_config,
            )
            graph = reasoner.builder.build(feature_dim=hidden_dim)
            print(f"[{cell_name}] graph: {graph.n_nodes} nodes, {graph.n_edges} edges")

            if not no_training:
                print(f"[{cell_name}] fitting (seed={seed}) …")
                fit_stats = reasoner.fit(
                    graph=graph,
                    epochs=30,
                    lr=1e-3,
                    sheaf_weight=coherence_weight,
                    grad_clip=1.0,
                    patience=8,
                    seed=seed,
                    teacher_sources=teacher_sources,
                )
                print(f"[{cell_name}] fit done: "
                      f"best_loss={fit_stats.get('best_loss', '?'):.4f}")
            else:
                print(f"[{cell_name}] skipping fit (no_training=True)")

            # ── Evaluate on test scenarios ───────────────────────────────
            # test_scenarios are at the end of all_scenarios
            test_start_idx = len(train_scenarios) + len(dev_scenarios)
            per_scenario: list[dict] = []

            schema_types = {k: v.get("type", "") for k, v in schema.items()}

            for test_local_idx, sc in enumerate(test_scenarios):
                global_idx = test_start_idx + test_local_idx
                ent_num = scenario_entity_nums[global_idx]

                oracle_verdict = sc["planted_verdict"]
                planted_thinness = int(sc.get("planted_thinness", 0))

                # Determine the doc_id for this test scenario
                if ent_num is None:
                    test_doc_id = f"scenario_{global_idx:04d}"
                else:
                    test_doc_id = f"scenario_{ent_num:04d}"

                if ent_num is None:
                    # Cannot query without an entity number; emit vacuous mass
                    mass_tuple = (0.0, 0.0, 0.0, 1.0)
                    predicted_verdict = "NOT ENOUGH INFO"
                elif no_training:
                    # teacher_only cell: use raw DS mass from store.
                    # Query by doc_id (not entity anchor key) because the SPLADE
                    # encoder conflates all entity{N} anchors via the shared
                    # "entity" token — get_infons_for_anchor('entity56') would
                    # return the wrong infons. Doc-level retrieval is reliable.
                    mass_tuple = _compute_teacher_mass(
                        cog.store, schema_types, test_doc_id
                    )
                    predicted_verdict = _verdict_from_mass(mass_tuple)
                else:
                    # Standard GNN evaluation: query with entity token text
                    # The schema token is "entity {N}" (lowercase); encoding
                    # this text will activate the entity{N} anchor.
                    query_text = f"entity {ent_num}"
                    try:
                        result = reasoner.reason(
                            query_text,
                            decisive_top_k=decisive_top_k,
                            fusion_rule=fusion_rule,
                        )
                        m = result.mass
                        mass_tuple = (
                            float(m.supports),
                            float(m.refutes),
                            float(m.uncertain),
                            float(m.theta),
                        )
                        predicted_verdict = result.verdict
                    except Exception as exc:
                        print(f"[{cell_name}] reason() failed for entity {ent_num}: {exc}")
                        # Fall back to vacuous mass
                        mass_tuple = (0.0, 0.0, 0.0, 1.0)
                        predicted_verdict = "NOT ENOUGH INFO"

                per_scenario.append({
                    "scenario_idx": global_idx,
                    "oracle_verdict": oracle_verdict,
                    "predicted_verdict": predicted_verdict,
                    "planted_thinness": planted_thinness,
                    "mass": list(mass_tuple),
                })

        finally:
            cog.close()

    # Compute metrics
    metrics = _compute_metrics(per_scenario)

    result_dict = {
        "cell": cell_name,
        "seed": seed,
        "n_test": len(test_scenarios),
        "config": {
            "aggregator": aggregator,
            "readout": readout,
            "teacher_sources": teacher_sources,
            "coherence_weight": coherence_weight,
            "fusion_rule": fusion_rule,
            "decisive_top_k": decisive_top_k,
            "n_layers": n_layers,
            "no_training": no_training,
        },
        "metrics": metrics,
        "per_scenario": per_scenario,
    }

    out_path = out_dir / f"{cell_name}__seed={seed}.json"
    out_path.write_text(json.dumps(result_dict, indent=2, sort_keys=True))
    print(f"[{cell_name}] wrote {out_path}")
    return out_path


def run_matrix(
    config_path: Path,
    seeds_override: list[int] | None,
    data_dir: Path,
    out_dir: Path,
) -> list[Path]:
    """Run all cells in the ablation matrix YAML.

    Parameters
    ----------
    config_path:
        Path to the YAML file containing a top-level ``cells`` list.
    seeds_override:
        If provided, overrides the ``seeds`` field for every cell.
    data_dir:
        Directory containing train.json / dev.json / test.json.
    out_dir:
        Output directory for result JSON files.

    Returns
    -------
    list[Path]
        Paths to all written JSON files (one per cell × seed).
    """
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cells = raw.get("cells", [])
    if not cells:
        raise ValueError(f"No cells found in {config_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for cell_cfg in cells:
        seeds = seeds_override if seeds_override else cell_cfg.get("seeds", [42])
        for seed in seeds:
            path = run_cell(cell_cfg, seed, data_dir, out_dir)
            written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation matrix on synthetic pilot scenarios."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to ablation_matrix.yaml",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated seed list (overrides per-cell seeds in YAML)",
    )
    parser.add_argument(
        "--data", required=True, type=Path,
        help="Directory containing train.json / dev.json / test.json",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for per-cell result JSON files",
    )
    args = parser.parse_args()

    seeds_override: list[int] | None = None
    if args.seeds:
        seeds_override = [int(s.strip()) for s in args.seeds.split(",")]

    written = run_matrix(args.config, seeds_override, args.data, args.out)
    print(f"\nDone. Wrote {len(written)} result file(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
