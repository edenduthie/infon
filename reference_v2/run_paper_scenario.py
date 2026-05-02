"""Run the paper's diagnostic scenario with pinned seeds and emit a JSON report.

Reproduces the EV-battery scenario from tests/test_logic.py and runs the
four diagnostic queries from draft2.txt §Verdict-calibration, plus the
edge-discovery and anchor-discovery passes. Designed to give one
authoritative set of numbers per seed for paper revision.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_logic import DOCUMENTS, SCHEMA_DEFS  # noqa: E402


def pin_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def run(seed: int, fit_epochs: int = 30) -> dict:
    pin_seeds(seed)

    from cognition import Cognition, CognitionConfig
    from cognition.logic import HypergraphBuilder, HypergraphReasoner

    out: dict = {"seed": seed, "fit_epochs": fit_epochs}

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "scenario.db")
        schema_path = os.path.join(tmpdir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA_DEFS, f)

        cfg = CognitionConfig(
            schema_path=schema_path,
            db_path=db_path,
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
        )
        cog = Cognition(cfg)

        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        out["corpus"] = {
            "n_docs": len(DOCUMENTS),
            "n_infons": cog.stats()["infon_count"],
            "n_constraints": cog.stats()["constraint_count"],
            "n_anchors_active": cog.stats()["anchors"],
            "n_anchors_schema": len(SCHEMA_DEFS),
        }

        builder = HypergraphBuilder(cog.store, cog.encoder, cog.schema)
        graph = builder.build(feature_dim=64)
        out["graph"] = {
            "n_nodes": int(graph.n_nodes),
            "n_edges": int(graph.n_edges),
            "n_anchors": len(graph.anchor_map),
            "n_infons": len(graph.infon_map),
        }

        # Re-pin between graph build and reasoner construction so that
        # network init is deterministic per seed.
        pin_seeds(seed)
        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema)

        # Pre-fit message-passing norm sanity (paper claim: delta > 0.01)
        with torch.no_grad():
            h0 = graph.node_features
            h2 = reasoner.forward(graph)
        out["norms"] = {
            "input_norm": float(h0.norm()),
            "layer2_norm": float(h2.norm()),
            "delta_norm": float((h2 - h0).norm()),
        }

        # Transductive fit
        fit_stats = reasoner.fit(graph=graph, epochs=fit_epochs, verbose=False)
        losses = fit_stats.get("losses", [])
        out["training"] = {
            "epochs_run": fit_stats.get("epochs", fit_epochs),
            "loss_initial": float(losses[0]) if losses else float("nan"),
            "loss_final": float(fit_stats["final_loss"]),
            "loss_best": float(fit_stats.get("best_loss", fit_stats["final_loss"])),
            "fiedler": fit_stats.get("sheaf_fiedler"),
            "early_stopped": bool(fit_stats.get("early_stopped", False)),
            "n_targets": fit_stats.get("n_targets"),
        }

        # Diagnostic queries from the paper
        diag_queries = [
            "Did Toyota invest in battery technology?",
            "Did Honda delay its electric vehicles?",
            "Is Tesla expanding production?",
            "Does CATL produce batteries in China?",
        ]
        out["queries"] = []
        for q in diag_queries:
            res = reasoner.reason(q)
            m = res.mass
            out["queries"].append({
                "query": q,
                "verdict": res.verdict,
                "S": round(float(m.supports), 4),
                "R": round(float(m.refutes), 4),
                "U": round(float(m.uncertain), 4),
                "theta": round(float(m.theta), 4),
                "n_relevant": res.n_relevant,
            })

        # Refinement pass: counts edges by type
        ref_result = reasoner.refine(graph=graph)
        out["refinement"] = {
            "infons_updated": ref_result.infons_updated,
            "next_edges": ref_result.temporal_added,
            "causes_edges": ref_result.causal_added,
            "contradicts_edges": ref_result.contradictions_found,
            "pairs_evaluated": ref_result.pairs_checked,
        }

        # Anchor discovery: returns (schema, list[DiscoveredAnchor], stats)
        disc_schema, disc_anchors, disc_stats = reasoner.discover_anchors(graph=graph)
        anchor_names = list(graph.anchor_map.keys())
        out["discovery"] = {
            "n_input_anchors": disc_stats.get("n_anchors"),
            "n_clusters": disc_stats.get("n_clusters"),
            "silhouette": round(float(disc_stats.get("silhouette", 0.0)), 4),
            "cluster_sizes": disc_stats.get("cluster_sizes"),
            "eigenvalues": [round(float(e), 4)
                            for e in disc_stats.get("eigenvalues", [])],
            "clusters": [
                {
                    "name": da.name,
                    "type": da.inferred_type,
                    "size": da.size,
                    "members": [anchor_names[i] for i in da.centroid_indices],
                    "coherence": round(float(da.coherence), 4),
                }
                for da in disc_anchors
            ],
        }

        cog.close()
    return out


if __name__ == "__main__":
    seeds = [42, 0, 1]
    runs = [run(seed=s) for s in seeds]
    report = {"runs": runs}
    out_path = REPO_ROOT / "paper_scenario_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nSummary across {len(seeds)} seeds:")
    for r in runs:
        print(f"\n  seed={r['seed']}  loss {r['training']['loss_initial']:.4f}"
              f" → {r['training']['loss_final']:.4f}"
              f"  fiedler={r['training']['fiedler']}")
        for q in r["queries"]:
            print(f"    {q['verdict']:<8s}  S={q['S']:.3f} R={q['R']:.3f}"
                  f" U={q['U']:.3f} θ={q['theta']:.3f}  | {q['query']}")
        ref = r["refinement"]
        print(f"    edges: NEXT={ref['next_edges']}  "
              f"CAUSES={ref['causes_edges']}  "
              f"CONTRADICTS={ref['contradicts_edges']}")
        disc = r["discovery"]
        print(f"    discovery: {disc['n_clusters']} clusters,"
              f" silhouette={disc['silhouette']:.3f}")
