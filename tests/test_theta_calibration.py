"""θ (ignorance) calibration: does the system correctly say 'I don't
know' on claims the corpus doesn't speak to?"""
from __future__ import annotations

import os
import sys

import pytest


def test_nei_claims_have_high_theta():
    """On the 20-claim LLM comparison gold set, NEI claims should have
    mean θ > 0.5 (previously 0.00 — systematic overcommit)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from llm_comparison import main

    cog_results, _, _ = main(verbose=False)

    nei_claims = [r for r in cog_results.results
                  if r.gold == "NOT_ENOUGH_INFO"]
    assert len(nei_claims) > 0, "test needs NEI claims in gold set"
    mean_theta = sum(r.theta for r in nei_claims) / len(nei_claims)
    print(f"\n  NEI claims: {len(nei_claims)}")
    print(f"  mean θ on NEI: {mean_theta:.3f}")
    assert mean_theta > 0.5, (
        f"mean θ on NEI claims is {mean_theta:.3f}; "
        f"calibration is broken (should be > 0.5)"
    )


def test_supported_claims_have_low_theta():
    """On SUPPORTS claims, θ should be low (the system can commit)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from llm_comparison import main

    cog_results, _, _ = main(verbose=False)

    supports_claims = [r for r in cog_results.results
                       if r.gold == "SUPPORTS" and r.predicted == "SUPPORTS"]
    if not supports_claims:
        pytest.skip("no correctly-predicted SUPPORTS claims")
    mean_theta = sum(r.theta for r in supports_claims) / len(supports_claims)
    print(f"\n  correctly-SUPPORTS claims: {len(supports_claims)}")
    print(f"  mean θ on those: {mean_theta:.3f}")
    # These should have θ below ~0.3 — system is confident
    assert mean_theta < 0.3, (
        f"mean θ on confidently-SUPPORTS claims is {mean_theta:.3f}; "
        f"expected < 0.3"
    )


def test_accuracy_still_reasonable():
    """The calibration fix shouldn't tank accuracy overall. Require
    accuracy ≥ 0.6 on the 20-claim set (baseline was 0.45)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root, "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from llm_comparison import main

    cog_results, _, _ = main(verbose=False)
    print(f"\n  accuracy: {cog_results.accuracy:.0%}")
    assert cog_results.accuracy >= 0.60, (
        f"accuracy {cog_results.accuracy:.0%} too low after "
        f"calibration fix"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
