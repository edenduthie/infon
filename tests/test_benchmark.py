"""Smoke test: the benchmark harness runs at small scale."""
from __future__ import annotations

import os
import sys
import time

import pytest


def test_benchmark_small_scale_runs():
    """Run the 500-target-infon scale of the benchmark and verify the
    reported numbers are finite and within reasonable bounds."""
    # Make examples/ importable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from benchmark import run_one_scale

    t0 = time.perf_counter()
    row = run_one_scale(target_infons=500, verbose=False)
    elapsed = time.perf_counter() - t0

    print(f"\n  benchmark row: {row}")
    print(f"  wall clock: {elapsed:.1f}s")

    # Structural checks
    assert row["n_infons_actual"] > 0
    assert row["n_nodes"] > 0
    assert row["init_ms"] > 0
    assert row["ingest_ms"] > 0
    assert row["ingest_rate"] > 10, (
        f"ingest rate {row['ingest_rate']:.0f} infons/sec below 10 — "
        f"something is very wrong"
    )
    assert row["query_p50_ms"] > 0
    # Query p50 is typically ~1s at this scale with rebuild-graph-each-time;
    # set a generous upper bound
    assert row["query_p50_ms"] < 10000, (
        f"query p50 {row['query_p50_ms']:.0f} ms — suspiciously slow"
    )
    # Full run should finish in under 2 minutes
    assert elapsed < 120, f"test took {elapsed:.0f}s, expected < 120s"


def test_format_table_emits_markdown():
    """format_table produces valid markdown."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from benchmark import format_table
    rows = [
        {"n_infons_target": 500, "n_infons_actual": 100,
         "n_nodes": 120, "n_edges": 300,
         "init_ms": 1000, "ingest_ms": 500, "ingest_rate": 200,
         "graph_build_ms": 800, "gnn_fit_ms": 1500,
         "query_p50_ms": 800, "query_p99_ms": 900, "rss_delta_mb": 200},
    ]
    md = format_table(rows)
    assert "| Scale" in md
    assert "|---" in md
    # Check the row rendered
    assert "500" in md
    assert "100" in md
    print(f"\n{md}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
