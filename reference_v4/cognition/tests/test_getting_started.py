"""Smoke test: the getting_started example runs end-to-end."""
from __future__ import annotations

import os
import sys
import time

import pytest


def test_getting_started_runs_under_60s():
    """Import and run examples.getting_started.main(). Verify it
    completes cleanly and produces a well-formed result."""
    # Make the examples directory importable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from getting_started import main

    t0 = time.perf_counter()
    summary = main(verbose=False)
    elapsed = time.perf_counter() - t0

    print(f"\n  getting-started ran in {elapsed:.2f}s")
    print(f"  summary: {summary}")

    # Structural checks
    assert summary["n_infons"] > 0
    assert summary["verdict"] in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
    assert abs(summary["mass_sum"] - 1.0) < 1e-3
    assert summary["query_latency_ms"] > 0
    # Soft performance check: should run in under 60s
    assert elapsed < 60.0, f"demo took {elapsed:.2f}s, expected < 60s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
