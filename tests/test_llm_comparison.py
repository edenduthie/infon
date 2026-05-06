"""Smoke test: the LLM comparison harness runs with the mock backend."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


def test_comparison_runs_with_mock():
    """Run main() with the mock LLM backend, verify both systems
    produce verdicts on all 20 gold claims, and the markdown output
    is well-formed."""
    # Make examples/ importable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from llm_comparison import main, GOLD_CLAIMS

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "comparison.md")
        cog_results, llm_results, md = main(
            output_path=out_path, verbose=False,
        )

        # Structural checks: both systems answered every claim
        assert len(cog_results.results) == len(GOLD_CLAIMS)
        assert len(llm_results.results) == len(GOLD_CLAIMS)

        # Every verdict is well-formed
        for r in cog_results.results + llm_results.results:
            assert r.predicted in (
                "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO",
            )
            assert 0.0 <= r.theta <= 1.0

        # Accuracy is at least better than coin flip
        print(f"\n  infon: accuracy {cog_results.accuracy:.0%}, "
              f"θ on NEI {cog_results.calibration_on_nei():.2f}")
        print(f"  mock LLM:  accuracy {llm_results.accuracy:.0%}, "
              f"θ on NEI {llm_results.calibration_on_nei():.2f}")
        assert cog_results.accuracy > 0.25
        assert llm_results.accuracy > 0.25

        # Markdown output is well-formed
        assert "| Claim |" in md
        assert "## Summary" in md
        assert "θ on NEI claims" in md
        assert os.path.exists(out_path)


def test_comparison_gold_claim_shape():
    """20 gold claims, each is (str, label) with a valid label."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from llm_comparison import GOLD_CLAIMS
    assert len(GOLD_CLAIMS) == 20
    labels = set()
    for claim, label in GOLD_CLAIMS:
        assert isinstance(claim, str) and claim.endswith("?")
        assert label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
        labels.add(label)
    # Gold set should cover all three verdict classes
    assert labels == {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
