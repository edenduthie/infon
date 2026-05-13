"""Quick validation run with synthetic data — verifies the pipeline works
before committing to the full Wikipedia download.

Creates a minimal FEVER-like and HoVer-like dataset from hand-written
examples, runs all three conditions, and prints results.

Usage:
    cd benchmarks && python run_quick.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cognition" / "src"))

SCHEMA_PATH = Path(__file__).parent / "schemas" / "wikipedia_general.json"


# ── Synthetic Wikipedia pages ──────────────────────────────────────────

WIKI_PAGES = {
    "Albert_Einstein": [
        "Albert Einstein was a German-born theoretical physicist.",
        "He developed the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921.",
        "He was born in Ulm, in the Kingdom of Württemberg in the German Empire.",
        "Einstein moved to Switzerland in 1895.",
    ],
    "Theory_of_relativity": [
        "The theory of relativity usually encompasses two interrelated theories by Albert Einstein.",
        "Special relativity was published in 1905.",
        "General relativity was published in 1915.",
        "The theory transformed theoretical physics and astronomy during the 20th century.",
    ],
    "Nobel_Prize_in_Physics": [
        "The Nobel Prize in Physics is a yearly award given by the Royal Swedish Academy of Sciences.",
        "It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895.",
        "Notable recipients include Albert Einstein, Niels Bohr, and Richard Feynman.",
    ],
    "Ulm": [
        "Ulm is a city in the German state of Baden-Württemberg.",
        "It is situated on the River Danube.",
        "Ulm is known as the birthplace of Albert Einstein.",
        "The city has a population of approximately 126,000.",
    ],
    "Swiss_Federal_Institute_of_Technology": [
        "ETH Zurich is a public research university in Zurich, Switzerland.",
        "Albert Einstein studied there from 1896 to 1900.",
        "It is consistently ranked among the top universities in the world.",
    ],
    "Marie_Curie": [
        "Marie Curie was a Polish and naturalized-French physicist and chemist.",
        "She conducted pioneering research on radioactivity.",
        "She was the first woman to win a Nobel Prize.",
        "Curie was born in Warsaw, then part of the Russian Empire.",
    ],
    "Radioactivity": [
        "Radioactivity is the spontaneous emission of radiation from atomic nuclei.",
        "It was discovered by Henri Becquerel in 1896.",
        "Marie Curie coined the term radioactivity.",
    ],
}


# ── Synthetic FEVER claims ─────────────────────────────────────────────

FEVER_CLAIMS = [
    {
        "id": 1,
        "claim": "Albert Einstein was born in Germany.",
        "label": "SUPPORTS",
        "evidence": [[[0, 0, "Albert_Einstein", 0]]],
    },
    {
        "id": 2,
        "claim": "Einstein received the Nobel Prize in Chemistry.",
        "label": "REFUTES",
        "evidence": [[[0, 0, "Albert_Einstein", 2]]],
    },
    {
        "id": 3,
        "claim": "The theory of relativity was developed by Niels Bohr.",
        "label": "REFUTES",
        "evidence": [[[0, 0, "Theory_of_relativity", 0]]],
    },
    {
        "id": 4,
        "claim": "Special relativity was published in 1905.",
        "label": "SUPPORTS",
        "evidence": [[[0, 0, "Theory_of_relativity", 1]]],
    },
    {
        "id": 5,
        "claim": "Marie Curie was born in France.",
        "label": "REFUTES",
        "evidence": [[[0, 0, "Marie_Curie", 3]]],
    },
    {
        "id": 6,
        "claim": "Marie Curie researched radioactivity.",
        "label": "SUPPORTS",
        "evidence": [[[0, 0, "Marie_Curie", 1], [0, 0, "Radioactivity", 2]]],
    },
    {
        "id": 7,
        "claim": "Ulm is located on the Rhine River.",
        "label": "REFUTES",
        "evidence": [[[0, 0, "Ulm", 1]]],
    },
    {
        "id": 8,
        "claim": "Einstein studied at ETH Zurich.",
        "label": "SUPPORTS",
        "evidence": [[[0, 0, "Swiss_Federal_Institute_of_Technology", 1]]],
    },
]


# ── Synthetic HoVer claims (multi-hop) ────────────────────────────────

HOVER_CLAIMS = [
    {
        "uid": "h1",
        "claim": "The physicist who developed relativity was born in a city on the Danube.",
        "label": "SUPPORTED",
        "supporting_facts": [
            ["Theory_of_relativity", 0],
            ["Albert_Einstein", 3],
            ["Ulm", 1],
        ],
    },
    {
        "uid": "h2",
        "claim": "The woman who coined the term radioactivity won a Nobel Prize.",
        "label": "SUPPORTED",
        "supporting_facts": [
            ["Radioactivity", 2],
            ["Marie_Curie", 2],
        ],
    },
    {
        "uid": "h3",
        "claim": "Einstein studied at a university in the country he moved to in 1895.",
        "label": "SUPPORTED",
        "supporting_facts": [
            ["Albert_Einstein", 4],
            ["Swiss_Federal_Institute_of_Technology", 0],
            ["Swiss_Federal_Institute_of_Technology", 1],
        ],
    },
    {
        "uid": "h4",
        "claim": "The Nobel Prize in Physics was won by someone born in Warsaw.",
        "label": "NOT_SUPPORTED",
        "supporting_facts": [
            ["Nobel_Prize_in_Physics", 2],
            ["Marie_Curie", 3],
        ],
    },
]


def run_fever_quick():
    """Run FEVER evaluation on synthetic data."""
    from fever.evaluate import CognitionFEVERRunner, aggregate_results

    print("=" * 60)
    print("FEVER (synthetic, 8 claims)")
    print("=" * 60)

    runner = CognitionFEVERRunner(schema_path=SCHEMA_PATH)
    t0 = time.time()
    results = runner.evaluate_batch(FEVER_CLAIMS, WIKI_PAGES, top_k=20)
    elapsed = time.time() - t0

    agg = aggregate_results(results)
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Evidence P={agg['evidence_precision']:.3f} "
          f"R={agg['evidence_recall']:.3f} F1={agg['evidence_f1']:.3f}")
    print(f"  FEVER score={agg['fever_score']:.3f}")
    print(f"  Label acc={agg['label_accuracy']:.3f}")
    print(f"  Avg infons/claim={agg['avg_infons_per_claim']:.1f}")
    print(f"  Avg retrieved sents={agg['avg_retrieved_sents']:.1f}")

    # Detail per claim
    print("\n  Per-claim detail:")
    for r in results:
        status = "OK" if r.predicted_label == r.gold_label else "XX"
        print(f"    [{status}] claim={r.claim_id} gold={r.gold_label} "
              f"pred={r.predicted_label} "
              f"P={r.evidence_precision:.2f} R={r.evidence_recall:.2f} "
              f"infons={len(r.infons)} sents={len(r.retrieved_sent_ids)}")

    return agg


def run_hover_quick():
    """Run HoVer evaluation on synthetic data."""
    from hover.evaluate import CognitionHoVerRunner, aggregate_results

    print("\n" + "=" * 60)
    print("HoVer (synthetic, 4 multi-hop claims)")
    print("=" * 60)

    runner = CognitionHoVerRunner(schema_path=SCHEMA_PATH)
    t0 = time.time()
    results = runner.evaluate_batch(HOVER_CLAIMS, WIKI_PAGES, chain_depth=10)
    elapsed = time.time() - t0

    agg = aggregate_results(results)
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Fact P={agg['fact_precision']:.3f} "
          f"R={agg['fact_recall']:.3f} F1={agg['fact_f1']:.3f}")
    print(f"  Path coverage={agg['path_coverage']:.3f}")
    print(f"  Avg infons/claim={agg['avg_infons_per_claim']:.1f}")
    print(f"  Avg edges/claim={agg['avg_edges_per_claim']:.1f}")
    print(f"  Avg paths/claim={agg['avg_paths_per_claim']:.1f}")

    if agg.get("by_hops"):
        print("\n  Per-hop breakdown:")
        for hops, info in agg["by_hops"].items():
            print(f"    {hops}: n={info['count']} F1={info['fact_f1']:.3f} "
                  f"coverage={info['path_coverage']:.3f} "
                  f"per_hop_recall={info['per_hop_recall']}")

    # Detail per claim
    print("\n  Per-claim detail:")
    for r in results:
        status = "OK" if r.predicted_label == r.gold_label else "XX"
        print(f"    [{status}] uid={r.claim_id} hops={r.num_hops} "
              f"gold={r.gold_label} pred={r.predicted_label} "
              f"P={r.fact_precision:.2f} R={r.fact_recall:.2f} "
              f"coverage={r.path_coverage:.2f} "
              f"edges={len(r.edges)} paths={len(r.paths_found)}")

    return agg


def run_rag_quick():
    """Run RAG baseline on synthetic data (requires sentence-transformers + faiss)."""
    try:
        from baselines.rag_baseline import RAGBaseline
    except ImportError as e:
        print(f"\n[skip] RAG baseline: {e}")
        print("  Install: pip install sentence-transformers faiss-cpu")
        return None

    print("\n" + "=" * 60)
    print("RAG Baseline (synthetic)")
    print("=" * 60)

    # Flatten wiki pages to sentences
    wiki_sentences = []
    for doc_id, sents in WIKI_PAGES.items():
        for sent_idx, text in enumerate(sents):
            if text.strip():
                wiki_sentences.append({
                    "doc_id": doc_id,
                    "sent_id": sent_idx,
                    "text": text,
                })

    print(f"  Indexing {len(wiki_sentences)} sentences...")
    t0 = time.time()
    baseline = RAGBaseline()
    baseline.build_index(wiki_sentences)
    print(f"  Index built in {time.time() - t0:.1f}s")

    # FEVER
    print("\n  --- FEVER ---")
    fever_results = baseline.evaluate_fever(FEVER_CLAIMS, top_k=5)
    for r in fever_results:
        top_sents = [s["text"][:50] for s in r.retrieved_sentences[:3]]
        print(f"    claim={r.claim_id}: {top_sents}")

    # HoVer
    print("\n  --- HoVer ---")
    hover_results = baseline.evaluate_hover(HOVER_CLAIMS, top_k=10)
    for r in hover_results:
        docs_hit = set(s["doc_id"] for s in r.retrieved_sentences)
        print(f"    uid={r.claim_id}: docs={docs_hit}")

    return {"fever": fever_results, "hover": hover_results}


if __name__ == "__main__":
    print("Cognition NeurIPS 2026 Benchmark — Quick Validation\n")

    # Run cognition conditions
    fever_agg = run_fever_quick()
    hover_agg = run_hover_quick()

    # Run RAG baseline (optional — needs extra deps)
    rag_results = run_rag_quick()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python download_data.py   — get FEVER + HoVer data")
    print("  2. python run_all.py --limit 200  — dev run on real data")
    print("  3. python run_all.py --limit 0    — full benchmark")
