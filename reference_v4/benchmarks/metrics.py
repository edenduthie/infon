"""Evaluation metrics for NeurIPS paper tables.

Produces:
  - Table 1: FEVER evidence retrieval comparison
  - Table 2: HoVer multi-hop reasoning comparison
  - Table 3: Ablation (schema discovery vs fixed vs none)
  - LaTeX output for camera-ready
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ConditionMetrics:
    """Metrics for one experimental condition."""
    condition: str
    benchmark: str

    # Evidence retrieval
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0

    # Label accuracy
    label_accuracy: float = 0.0

    # FEVER-specific
    fever_score: float = 0.0

    # HoVer-specific
    path_coverage: float = 0.0
    hop_2_f1: float = 0.0
    hop_3_f1: float = 0.0
    hop_4_f1: float = 0.0

    # Efficiency
    time_s: float = 0.0
    infons_per_claim: float = 0.0
    edges_per_claim: float = 0.0

    # Representation
    evidence_unit: str = ""  # "text_snippet" | "infon" | "infon_discovered"
    retrieval_unit: str = ""  # "dense_vector" | "anchor_projection" | "anchor_discovered"


def load_results(path: str | Path) -> dict:
    """Load benchmark_results.json."""
    with open(path) as f:
        return json.load(f)


def format_latex_table_fever(results: dict) -> str:
    """Generate LaTeX table for FEVER results (Table 1 in paper)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{FEVER evidence retrieval: Infons (atomic units) vs.\ text snippets.}",
        r"\label{tab:fever}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Evidence Unit & Prec. & Rec. & F1 & FEVER Score \\",
        r"\midrule",
    ]

    condition_labels = {
        "rag": ("RAG (dense)", "Text snippet"),
        "fixed": ("Cognition (fixed)", "Infon"),
        "discovered": ("Cognition (Kan ext.)", "Infon"),
    }

    for cond, metrics in results.get("fever", {}).items():
        label, unit = condition_labels.get(cond, (cond, "?"))
        p = metrics.get("evidence_precision", 0)
        r = metrics.get("evidence_recall", 0)
        f1 = metrics.get("evidence_f1", 0)
        fs = metrics.get("fever_score", 0)
        lines.append(
            f"{label} & {unit} & {p:.3f} & {r:.3f} & {f1:.3f} & {fs:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_latex_table_hover(results: dict) -> str:
    """Generate LaTeX table for HoVer results (Table 2 in paper)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{HoVer multi-hop reasoning: hyperedge traversal vs.\ document linking.}",
        r"\label{tab:hover}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Reasoning Unit & Prec. & Rec. & F1 & Path Cov. & 2-hop F1 \\",
        r"\midrule",
    ]

    condition_labels = {
        "rag": ("RAG (dense)", "Document"),
        "fixed": ("Cognition (fixed)", "Hyperedge"),
        "discovered": ("Cognition (Kan ext.)", "Hyperedge"),
    }

    for cond, metrics in results.get("hover", {}).items():
        label, unit = condition_labels.get(cond, (cond, "?"))
        p = metrics.get("fact_precision", 0)
        r = metrics.get("fact_recall", 0)
        f1 = metrics.get("fact_f1", 0)
        pc = metrics.get("path_coverage", 0)
        hops = metrics.get("by_hops", {})
        h2 = hops.get("2_hop", {}).get("fact_f1", 0)
        lines.append(
            f"{label} & {unit} & {p:.3f} & {r:.3f} & {f1:.3f} & {pc:.3f} & {h2:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_comparison_table(results: dict) -> str:
    """Unified comparison table mapping the user's original specification.

    RAG FEVER: Gold Text Snippets → Label + Sentence IDs
    RAG HoVer: Linked Documents → Reasoning Paths
    Cognition FEVER: Infons (Atomic Units) → Infon IDs + Support Types
    Cognition HoVer: Hyperedges (Relational Paths) → Edge Chains
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Representation comparison: structured infons vs.\ unstructured retrieval across evidence granularity levels.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Dataset & System & Retrieval Unit & Evidence Form & F1 & Coverage \\",
        r"\midrule",
    ]

    fever = results.get("fever", {})
    hover = results.get("hover", {})

    # FEVER rows
    if "rag" in fever:
        m = fever["rag"]
        lines.append(
            f"FEVER & RAG & Text Snippets & Label + Sent. IDs & "
            f"{m.get('evidence_f1', 0):.3f} & --- \\\\"
        )
    if "fixed" in fever:
        m = fever["fixed"]
        lines.append(
            f"FEVER & Cognition (fixed) & Infons (Atomic) & Infon IDs + Support & "
            f"{m.get('evidence_f1', 0):.3f} & --- \\\\"
        )
    if "discovered" in fever:
        m = fever["discovered"]
        lines.append(
            f"FEVER & Cognition (Kan) & Infons (Atomic) & Infon IDs + Support & "
            f"{m.get('evidence_f1', 0):.3f} & --- \\\\"
        )

    lines.append(r"\midrule")

    # HoVer rows
    if "rag" in hover:
        m = hover["rag"]
        lines.append(
            f"HoVer & RAG & Linked Documents & Reasoning Paths & "
            f"{m.get('fact_f1', 0):.3f} & {m.get('path_coverage', 0):.3f} \\\\"
        )
    if "fixed" in hover:
        m = hover["fixed"]
        lines.append(
            f"HoVer & Cognition (fixed) & Hyperedges & Relational Paths & "
            f"{m.get('fact_f1', 0):.3f} & {m.get('path_coverage', 0):.3f} \\\\"
        )
    if "discovered" in hover:
        m = hover["discovered"]
        lines.append(
            f"HoVer & Cognition (Kan) & Hyperedges & Relational Paths & "
            f"{m.get('fact_f1', 0):.3f} & {m.get('path_coverage', 0):.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_all_tables(results_path: str | Path, output_dir: str | Path):
    """Generate all LaTeX tables from results."""
    results = load_results(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table_fever.tex": format_latex_table_fever(results),
        "table_hover.tex": format_latex_table_hover(results),
        "table_comparison.tex": format_comparison_table(results),
    }

    for name, content in tables.items():
        path = output_dir / name
        with open(path, "w") as f:
            f.write(content)
        print(f"  Written: {path}")


if __name__ == "__main__":
    import sys
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/tables"
    generate_all_tables(results_path, output_dir)
