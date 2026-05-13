"""Run all benchmark conditions and produce comparison tables.

Three conditions:
  1. RAG baseline (dense retrieval, text snippets)
  2. Cognition-fixed (hand-crafted Wikipedia-general schema)
  3. Cognition-discovered (Kan extension schema from corpus)

Two benchmarks:
  A. FEVER — evidence precision/recall/F1, FEVER score
  B. HoVer — supporting fact P/R/F1, path coverage, per-hop recall

Usage:
    python run_all.py --fever --hover --limit 200
    python run_all.py --fever --conditions rag,fixed
    python run_all.py --hover --conditions discovered --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cognition" / "src"))

DATA_DIR = Path(__file__).parent / "data"
SCHEMA_DIR = Path(__file__).parent / "schemas"
RESULTS_DIR = Path(__file__).parent / "results"


def run_fever_rag(claims, wiki_pages, wiki_sentences, top_k=5):
    """RAG baseline on FEVER."""
    from baselines.rag_baseline import RAGBaseline

    print("\n[FEVER] RAG baseline")
    print(f"  Building index over {len(wiki_sentences)} sentences...")
    t0 = time.time()

    baseline = RAGBaseline()
    baseline.build_index(wiki_sentences, batch_size=256)
    print(f"  Index built in {time.time() - t0:.1f}s")

    print(f"  Evaluating {len(claims)} claims...")
    t0 = time.time()
    results = baseline.evaluate_fever(claims, top_k=top_k)
    elapsed = time.time() - t0

    # Compute metrics
    from fever.evaluate import extract_gold_sent_ids, _compute_evidence_metrics
    metrics = {"condition": "rag", "time_s": elapsed, "top_k": top_k}

    n = len(results)
    precisions, recalls, f1s = [], [], []
    label_correct = 0
    fever_scores = 0

    for r in results:
        gold_set = extract_gold_sent_ids(r.gold_evidence)
        retrieved_set = {(s["doc_id"], s.get("sent_id", -1)) for s in r.retrieved_sentences}

        p, rec, f1 = _compute_evidence_metrics(retrieved_set, gold_set)
        precisions.append(p)
        recalls.append(rec)
        f1s.append(f1)

    metrics["evidence_precision"] = sum(precisions) / n if n else 0
    metrics["evidence_recall"] = sum(recalls) / n if n else 0
    metrics["evidence_f1"] = sum(f1s) / n if n else 0

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Evidence P={metrics['evidence_precision']:.3f} "
          f"R={metrics['evidence_recall']:.3f} F1={metrics['evidence_f1']:.3f}")

    return metrics


def run_fever_cognition(claims, wiki_pages, schema_path, condition_name, top_k=20):
    """Cognition system on FEVER."""
    from fever.evaluate import CognitionFEVERRunner, aggregate_results

    print(f"\n[FEVER] Cognition ({condition_name})")
    print(f"  Schema: {schema_path}")
    t0 = time.time()

    runner = CognitionFEVERRunner(schema_path=schema_path)
    results = runner.evaluate_batch(claims, wiki_pages, top_k=top_k)
    elapsed = time.time() - t0

    agg = aggregate_results(results)
    agg["condition"] = condition_name
    agg["time_s"] = elapsed

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Evidence P={agg['evidence_precision']:.3f} "
          f"R={agg['evidence_recall']:.3f} F1={agg['evidence_f1']:.3f}")
    print(f"  FEVER score={agg['fever_score']:.3f}")
    print(f"  Label acc={agg['label_accuracy']:.3f}")

    return agg


def run_hover_rag(claims, wiki_pages, wiki_sentences, top_k=10):
    """RAG baseline on HoVer."""
    from baselines.rag_baseline import RAGBaseline

    print("\n[HoVer] RAG baseline")
    print(f"  Building index over {len(wiki_sentences)} sentences...")
    t0 = time.time()

    baseline = RAGBaseline()
    baseline.build_index(wiki_sentences, batch_size=256)
    print(f"  Index built in {time.time() - t0:.1f}s")

    print(f"  Evaluating {len(claims)} claims...")
    t0 = time.time()
    results = baseline.evaluate_hover(claims, top_k=top_k)
    elapsed = time.time() - t0

    # Compute metrics
    from hover.evaluate import extract_gold_facts, _compute_fact_metrics
    metrics = {"condition": "rag", "time_s": elapsed, "top_k": top_k}

    n = len(results)
    precisions, recalls, f1s = [], [], []

    for r in results:
        gold_set = extract_gold_facts(r.gold_evidence)
        retrieved_set = {(s["doc_id"], s.get("sent_id", -1)) for s in r.retrieved_sentences}

        p, rec, f1 = _compute_fact_metrics(retrieved_set, gold_set)
        precisions.append(p)
        recalls.append(rec)
        f1s.append(f1)

    metrics["fact_precision"] = sum(precisions) / n if n else 0
    metrics["fact_recall"] = sum(recalls) / n if n else 0
    metrics["fact_f1"] = sum(f1s) / n if n else 0

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Fact P={metrics['fact_precision']:.3f} "
          f"R={metrics['fact_recall']:.3f} F1={metrics['fact_f1']:.3f}")

    return metrics


def run_hover_cognition(claims, wiki_pages, schema_path, condition_name, chain_depth=15):
    """Cognition system on HoVer with hyperedge traversal."""
    from hover.evaluate import CognitionHoVerRunner, aggregate_results

    print(f"\n[HoVer] Cognition ({condition_name})")
    print(f"  Schema: {schema_path}")
    t0 = time.time()

    runner = CognitionHoVerRunner(schema_path=schema_path)
    results = runner.evaluate_batch(claims, wiki_pages, chain_depth=chain_depth)
    elapsed = time.time() - t0

    agg = aggregate_results(results)
    agg["condition"] = condition_name
    agg["time_s"] = elapsed

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Fact P={agg['fact_precision']:.3f} "
          f"R={agg['fact_recall']:.3f} F1={agg['fact_f1']:.3f}")
    print(f"  Path coverage={agg['path_coverage']:.3f}")
    if agg.get("by_hops"):
        for hops, info in agg["by_hops"].items():
            print(f"    {hops}: F1={info['fact_f1']:.3f} "
                  f"coverage={info['path_coverage']:.3f}")

    return agg


def discover_schema(wiki_pages, n_anchors=50, sample_size=5000):
    """Run SchemaDiscovery (Kan extension) on a sample of Wikipedia sentences."""
    from cognition.category import SchemaDiscovery
    from cognition.encoder import SpladeEncoder

    print("\n[Schema Discovery] Running Kan extension on Wikipedia sample...")

    # Sample sentences from wiki pages
    sentences = []
    for doc_id, sents in wiki_pages.items():
        for s in sents:
            if s.strip():
                sentences.append(s)
        if len(sentences) >= sample_size:
            break
    sentences = sentences[:sample_size]

    print(f"  Corpus: {len(sentences)} sentences")
    print(f"  Target anchors: {n_anchors}")

    t0 = time.time()
    discoverer = SchemaDiscovery()
    schema, discovered = discoverer.discover(
        sentences,
        n_anchors=n_anchors,
        min_doc_freq=3,
        activation_threshold=0.2,
    )
    elapsed = time.time() - t0

    print(f"  Discovered {len(discovered)} anchors in {elapsed:.1f}s")

    # Save discovered schema
    discovered_path = SCHEMA_DIR / "wikipedia_discovered.json"
    schema.save(discovered_path)
    print(f"  Saved to {discovered_path}")

    # Report anchor type distribution
    type_counts = {}
    for da in discovered:
        type_counts[da.inferred_type] = type_counts.get(da.inferred_type, 0) + 1
    print(f"  Types: {type_counts}")
    print(f"  Top anchors: {[da.name for da in discovered[:10]]}")

    return discovered_path


def build_wiki_sentences(wiki_pages: dict[str, list[str]]) -> list[dict]:
    """Flatten wiki pages into sentence records for RAG indexing."""
    sentences = []
    for doc_id, sents in wiki_pages.items():
        for sent_idx, text in enumerate(sents):
            if text.strip():
                sentences.append({
                    "doc_id": doc_id,
                    "sent_id": sent_idx,
                    "text": text,
                })
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Run FEVER/HoVer benchmarks")
    parser.add_argument("--fever", action="store_true", help="Run FEVER benchmark")
    parser.add_argument("--hover", action="store_true", help="Run HoVer benchmark")
    parser.add_argument("--conditions", default="rag,fixed,discovered",
                        help="Comma-separated conditions to run")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max claims per benchmark (for dev)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k retrieval for cognition")
    parser.add_argument("--chain-depth", type=int, default=15,
                        help="NEXT chain depth for HoVer")
    parser.add_argument("--discover-anchors", type=int, default=50,
                        help="Number of anchors for schema discovery")
    parser.add_argument("--wiki-dir", type=str, default=None,
                        help="Path to wiki-pages directory")
    args = parser.parse_args()

    if not args.fever and not args.hover:
        args.fever = True
        args.hover = True

    conditions = set(args.conditions.split(","))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Wikipedia pages
    wiki_dir = args.wiki_dir or str(DATA_DIR / "fever" / "wiki-pages")
    print(f"Loading Wikipedia from {wiki_dir}...")
    from fever.evaluate import load_wiki_pages
    wiki_pages = load_wiki_pages(wiki_dir)
    print(f"  Loaded {len(wiki_pages)} pages")

    # Build sentence list for RAG
    wiki_sentences = build_wiki_sentences(wiki_pages) if "rag" in conditions else []

    # Schema discovery (if needed)
    discovered_schema_path = SCHEMA_DIR / "wikipedia_discovered.json"
    if "discovered" in conditions and not discovered_schema_path.exists():
        discovered_schema_path = discover_schema(
            wiki_pages, n_anchors=args.discover_anchors
        )

    fixed_schema_path = SCHEMA_DIR / "wikipedia_general.json"

    all_results = {}

    # === FEVER ===
    if args.fever:
        print("\n" + "=" * 60)
        print("FEVER BENCHMARK")
        print("=" * 60)

        fever_path = DATA_DIR / "fever" / "paper_dev.jsonl"
        if not fever_path.exists():
            print(f"  [skip] {fever_path} not found — run download_data.py first")
        else:
            from fever.evaluate import load_fever_claims
            claims = load_fever_claims(fever_path, limit=args.limit)
            print(f"  Loaded {len(claims)} claims")

            fever_results = {}

            if "rag" in conditions:
                fever_results["rag"] = run_fever_rag(
                    claims, wiki_pages, wiki_sentences, top_k=5
                )

            if "fixed" in conditions:
                fever_results["fixed"] = run_fever_cognition(
                    claims, wiki_pages, fixed_schema_path,
                    "fixed", top_k=args.top_k
                )

            if "discovered" in conditions:
                fever_results["discovered"] = run_fever_cognition(
                    claims, wiki_pages, discovered_schema_path,
                    "discovered", top_k=args.top_k
                )

            all_results["fever"] = fever_results

    # === HoVer ===
    if args.hover:
        print("\n" + "=" * 60)
        print("HoVer BENCHMARK")
        print("=" * 60)

        hover_path = DATA_DIR / "hover" / "hover_dev.json"
        if not hover_path.exists():
            print(f"  [skip] {hover_path} not found — run download_data.py first")
        else:
            from hover.evaluate import load_hover_claims
            claims = load_hover_claims(hover_path, limit=args.limit)
            print(f"  Loaded {len(claims)} claims")

            hover_results = {}

            if "rag" in conditions:
                hover_results["rag"] = run_hover_rag(
                    claims, wiki_pages, wiki_sentences, top_k=10
                )

            if "fixed" in conditions:
                hover_results["fixed"] = run_hover_cognition(
                    claims, wiki_pages, fixed_schema_path,
                    "fixed", chain_depth=args.chain_depth
                )

            if "discovered" in conditions:
                hover_results["discovered"] = run_hover_cognition(
                    claims, wiki_pages, discovered_schema_path,
                    "discovered", chain_depth=args.chain_depth
                )

            all_results["hover"] = hover_results

    # === Summary table ===
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print_summary(all_results)

    # Save results
    results_file = RESULTS_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


def print_summary(all_results):
    """Print comparison tables."""

    if "fever" in all_results:
        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│ FEVER: Evidence Retrieval (Infons vs Text Snippets)     │")
        print("├──────────────┬───────────┬────────┬──────┬──────────────┤")
        print("│ Condition    │ Precision │ Recall │  F1  │ FEVER Score  │")
        print("├──────────────┼───────────┼────────┼──────┼──────────────┤")
        for name, m in all_results["fever"].items():
            p = m.get("evidence_precision", 0)
            r = m.get("evidence_recall", 0)
            f1 = m.get("evidence_f1", 0)
            fs = m.get("fever_score", 0)
            print(f"│ {name:<12} │   {p:.3f}   │ {r:.3f}  │{f1:.3f}│    {fs:.3f}     │")
        print("└──────────────┴───────────┴────────┴──────┴──────────────┘")

    if "hover" in all_results:
        print("\n┌───────────────────────────────────────────────────────────┐")
        print("│ HoVer: Multi-hop Reasoning (Hyperedges vs Doc Links)     │")
        print("├──────────────┬───────────┬────────┬──────┬───────────────┤")
        print("│ Condition    │ Precision │ Recall │  F1  │ Path Coverage │")
        print("├──────────────┼───────────┼────────┼──────┼───────────────┤")
        for name, m in all_results["hover"].items():
            p = m.get("fact_precision", 0)
            r = m.get("fact_recall", 0)
            f1 = m.get("fact_f1", 0)
            pc = m.get("path_coverage", 0)
            print(f"│ {name:<12} │   {p:.3f}   │ {r:.3f}  │{f1:.3f}│     {pc:.3f}     │")
        print("└──────────────┴───────────┴────────┴──────┴───────────────┘")


if __name__ == "__main__":
    main()
