"""Benchmark: measure wall-clock time + memory across scales.

Runs the cognition pipeline at three scales — 1k, 5k, 25k infons —
and reports:
    - ingestion throughput (infons/sec)
    - graph build time
    - embedder train time
    - GNN fit time (transductive)
    - query latency p50 / p99

All data is synthetic so numbers are reproducible. Outputs a markdown
table to BENCHMARK.md or returns it as a string.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import statistics
import sys
import tempfile
import time

import torch

from cognition.synth import Schema, generate_corpus


# Scales to run — can be reduced for unit tests
DEFAULT_SCALES = [1_000, 5_000, 25_000]


def _peak_rss_mb() -> float:
    """Peak resident-set-size in MB. macOS reports in bytes, Linux in KB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


def _make_corpus(n_infons: int, seed: int = 0) -> tuple[list, Schema]:
    """Turn the synth generator into a document-style corpus.

    The synth generator produces one sentence per example. We lump
    them into 'documents' of 50 sentences each and return (docs, schema)
    so our ingestion code sees realistic document boundaries.
    """
    schema = Schema(
        actors=["actor_a", "actor_b", "actor_c",
                "actor_d", "actor_e"],
        relations=["invests", "partners", "produces",
                   "expands", "delays", "acquires"],
        features=["battery", "solidstate", "ev",
                  "factory", "supply"],
        markets=["japan", "china", "america"],
    )
    # With the quality filter (max_triples_per_sentence=2), each
    # extracted sentence yields ~1-2 infons. Generate slightly more
    # sentences than the target to account for filter drops.
    n_sentences = max(10, int(n_infons * 0.7))
    examples = generate_corpus(
        schema=schema, n=n_sentences, seed=seed,
    )
    sentences = [ex.sentence for ex in examples]
    # Batch into 50-sentence documents
    docs = []
    for i in range(0, len(sentences), 50):
        text = " ".join(sentences[i:i + 50])
        docs.append({"id": f"doc_{i // 50:05d}", "text": text})
    return docs, schema


def run_one_scale(target_infons: int,
                  use_trained_embedder: bool = False,
                  verbose: bool = True) -> dict:
    """Run the pipeline at one scale and return timings."""
    from cognition import Cognition, CognitionConfig
    from cognition.logic import HypergraphReasoner

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Scale: {target_infons} infons")
        print(f"─" * 60)

    docs, schema = _make_corpus(target_infons)

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = os.path.join(tmpdir, "schema.json")
        # Write the schema as a JSON dict keyed by anchor name
        schema_dict = {}
        for name in schema.actors:
            schema_dict[name] = {"type": "actor", "tokens": [name]}
        for name in schema.relations:
            schema_dict[name] = {"type": "relation", "tokens": [name]}
        for name in schema.features:
            schema_dict[name] = {"type": "feature", "tokens": [name]}
        for name in schema.markets:
            schema_dict[name] = {"type": "market", "tokens": [name]}
        with open(schema_path, "w") as f:
            json.dump(schema_dict, f)

        # ── Init cost (includes embedder training if enabled) ──
        rss0 = _peak_rss_mb()
        t0 = time.perf_counter()
        cog = Cognition(CognitionConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "bench.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
            # Quality filter: only accept above-threshold triples,
            # at most 2 per sentence, with hard role-type constraints
            quality_threshold=0.04,
            max_triples_per_sentence=2,
            use_trained_embedder=use_trained_embedder,
            embedder_model_dir=os.path.join(tmpdir, "emb_cache"),
            embedder_n_synth=400,
            embedder_epochs=8,
            embedder_trunk_dim=64,
        ))
        t_init = time.perf_counter() - t0
        if verbose:
            print(f"  init                 {t_init*1000:>8.0f} ms")

        # ── Ingestion ──
        t0 = time.perf_counter()
        total_infons = 0
        for doc in docs:
            total_infons += cog.ingest([doc])
        cog.consolidate()
        t_ingest = time.perf_counter() - t0
        ingest_rate = total_infons / t_ingest if t_ingest > 0 else 0.0
        if verbose:
            print(f"  ingest               {t_ingest*1000:>8.0f} ms  "
                  f"({total_infons} infons, "
                  f"{ingest_rate:.0f} infons/sec)")

        # ── Graph build ──
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )
        if cog.embedder is not None:
            reasoner.builder.embedder = cog.embedder

        t0 = time.perf_counter()
        graph = reasoner.builder.build(
            max_infons=total_infons, feature_dim=64,
        )
        t_build = time.perf_counter() - t0
        n_nodes = graph.n_nodes
        n_edges = graph.n_edges
        if verbose:
            print(f"  graph build          {t_build*1000:>8.0f} ms  "
                  f"({n_nodes} nodes, {n_edges} edges)")

        # ── GNN fit ──
        t0 = time.perf_counter()
        fit_stats = reasoner.fit(graph=graph, epochs=15, verbose=False)
        t_fit = time.perf_counter() - t0
        if verbose:
            print(f"  GNN fit (15 ep)      {t_fit*1000:>8.0f} ms  "
                  f"(loss {fit_stats['losses'][0]:.3f} → "
                  f"{fit_stats['final_loss']:.3f})")

        # ── Query latency ──
        queries = [
            "Did actor_a invest in battery?",
            "Did actor_b partner with actor_c?",
            "Does actor_d produce ev?",
            "Is actor_e expanding in china?",
            "Did actor_a delay ev?",
        ]
        latencies_ms = []
        for q in queries:
            for _ in range(3):  # 3 repeats per query
                t0 = time.perf_counter()
                result = reasoner.reason(q)
                _ = result.mass
                latencies_ms.append((time.perf_counter() - t0) * 1000)
        p50 = statistics.median(latencies_ms)
        p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99) - 1] \
              if len(latencies_ms) > 1 else p50
        if verbose:
            print(f"  query p50 / p99      {p50:>8.0f} / {p99:.0f} ms "
                  f"({len(latencies_ms)} queries)")

        rss_peak = _peak_rss_mb() - rss0
        if verbose:
            print(f"  peak RSS delta       {rss_peak:>8.0f} MB")

        cog.close()
        gc.collect()

        return {
            "n_infons_target": target_infons,
            "n_infons_actual": total_infons,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "init_ms": t_init * 1000,
            "ingest_ms": t_ingest * 1000,
            "ingest_rate": ingest_rate,
            "graph_build_ms": t_build * 1000,
            "gnn_fit_ms": t_fit * 1000,
            "query_p50_ms": p50,
            "query_p99_ms": p99,
            "rss_delta_mb": rss_peak,
            "use_trained_embedder": use_trained_embedder,
        }


def format_table(rows: list[dict]) -> str:
    """Pretty-print a list of benchmark rows as a markdown table."""
    headers = [
        ("Scale (target)",       "n_infons_target"),
        ("Actual infons",        "n_infons_actual"),
        ("Nodes",                "n_nodes"),
        ("Edges",                "n_edges"),
        ("Init (ms)",            "init_ms"),
        ("Ingest (ms)",          "ingest_ms"),
        ("Rate (inf/s)",         "ingest_rate"),
        ("Graph build (ms)",     "graph_build_ms"),
        ("GNN fit (ms)",         "gnn_fit_ms"),
        ("Query p50 (ms)",       "query_p50_ms"),
        ("Query p99 (ms)",       "query_p99_ms"),
        ("ΔRSS (MB)",            "rss_delta_mb"),
    ]
    md = "| " + " | ".join(h[0] for h in headers) + " |\n"
    md += "|" + "|".join("---" for _ in headers) + "|\n"
    for row in rows:
        cells = []
        for _, key in headers:
            v = row.get(key)
            if isinstance(v, float):
                cells.append(f"{v:.0f}")
            else:
                cells.append(str(v))
        md += "| " + " | ".join(cells) + " |\n"
    return md


def main(scales: list[int] | None = None,
         output_path: str | None = None,
         verbose: bool = True) -> str:
    """Run the benchmark across `scales` and write results to a
    markdown file (or return the rendered markdown)."""
    scales = scales or DEFAULT_SCALES
    rows = []
    for n in scales:
        rows.append(run_one_scale(
            target_infons=n, use_trained_embedder=False, verbose=verbose,
        ))
    md = format_table(rows)
    if verbose:
        print("\n\nmarkdown table:\n")
        print(md)
    if output_path:
        with open(output_path, "w") as f:
            f.write("# Benchmark results\n\n")
            f.write(f"Machine: CPU, Python {sys.version_info.major}."
                    f"{sys.version_info.minor}\n\n")
            f.write(md)
        if verbose:
            print(f"wrote {output_path}")
    return md


if __name__ == "__main__":
    # Default run: 1k, 5k, 25k at small synth scale
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(root, "BENCHMARK.md")
    main(scales=DEFAULT_SCALES, output_path=out_path)
