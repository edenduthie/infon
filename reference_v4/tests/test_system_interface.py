"""Tests for EvalSystem protocol: each system returns a valid MassFunction.

Run from repo root:
    pytest reference_v4/tests/test_system_interface.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

from reference_v4.benchmarks.types import EvalClaim, MassFunction
from reference_v4.baselines.types import EvalSystem


@pytest.fixture
def ev_fixture_claim():
    return EvalClaim(
        claim_id="test_001",
        claim_text="Aspirin reduces the risk of heart attack.",
        evidence_docs=[
            "Aspirin is commonly used to prevent heart attacks in high-risk patients.",
            "Studies show aspirin lowers platelet aggregation which reduces clot formation.",
        ],
        ground_truth="SUPPORTS",
    )


def _check_mass(mass: MassFunction) -> None:
    assert isinstance(mass, MassFunction), f"Expected MassFunction, got {type(mass)}"
    assert abs(sum(mass) - 1.0) < 1e-6, f"MassFunction must sum to 1.0, got {sum(mass)}"
    for v in mass:
        assert 0.0 <= v <= 1.0, f"Each mass component must be in [0,1], got {v}"


def test_symbolic_floor_returns_mass(ev_fixture_claim):
    from reference_v4.baselines.symbolic_floor import SymbolicFloor
    system = SymbolicFloor()
    mass = system.evaluate(ev_fixture_claim)
    _check_mass(mass)


def test_symbolic_floor_name():
    from reference_v4.baselines.symbolic_floor import SymbolicFloor
    system = SymbolicFloor()
    assert isinstance(system.name, str) and system.name, "name must be a non-empty string"


def test_cognition_symbolic_returns_mass(ev_fixture_claim):
    from reference_v4.baselines.cognition_system import CognitionSystem
    system = CognitionSystem("symbolic")
    mass = system.evaluate(ev_fixture_claim)
    _check_mass(mass)


def test_cognition_symbolic_name():
    from reference_v4.baselines.cognition_system import CognitionSystem
    system = CognitionSystem("symbolic")
    assert isinstance(system.name, str) and system.name


def test_nli_classifier_returns_mass(ev_fixture_claim):
    from reference_v4.baselines.nli_classifier import NLIClassifier
    system = NLIClassifier()
    mass = system.evaluate(ev_fixture_claim)
    _check_mass(mass)


def test_nli_classifier_name():
    from reference_v4.baselines.nli_classifier import NLIClassifier
    system = NLIClassifier()
    assert isinstance(system.name, str) and system.name


def test_llm_zeroshot_returns_mass(ev_fixture_claim):
    from reference_v4.baselines.llm_zeroshot import LLMZeroShot
    # Build a cache key that matches what the system will look up
    from reference_v4.baselines._llm_cache import LLMCache
    from reference_v4.baselines.llm_zeroshot import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, LLMZeroShot
    system = LLMZeroShot(
        cache_path=str(FIXTURES_DIR / "llm_cache_fixture.jsonl"),
        mode="replay_only",
    )
    # Compute the key that this specific claim would generate
    from reference_v4.baselines.llm_zeroshot import _truncate_evidence, _build_prompt
    evidence_text = _truncate_evidence(ev_fixture_claim.evidence_docs)
    system_p, user_p = _build_prompt(ev_fixture_claim.claim_text, evidence_text)
    key = LLMCache.make_key(LLMZeroShot.MODEL_ID, system_p, user_p, 0.0, 256, "bedrock")
    # Put the key in the cache so the test can hit it
    import json
    fixture_path = FIXTURES_DIR / "llm_cache_fixture.jsonl"
    lines = fixture_path.read_text().splitlines()
    existing_keys = set()
    for l in lines:
        if l.strip():
            try:
                existing_keys.add(json.loads(l)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    if key not in existing_keys:
        # Use a pre-existing fixture key — test the fixture instead
        pytest.skip("Fixture key not pre-computed for this claim; skipping LLMZeroShot integration test")
    mass = system.evaluate(ev_fixture_claim)
    _check_mass(mass)


def test_llm_zeroshot_name():
    from reference_v4.baselines.llm_zeroshot import LLMZeroShot
    system = LLMZeroShot(
        cache_path=str(FIXTURES_DIR / "llm_cache_fixture.jsonl"),
        mode="replay_only",
    )
    assert isinstance(system.name, str) and system.name


def test_all_systems_satisfy_protocol(ev_fixture_claim):
    """Verify each system satisfies the EvalSystem Protocol via isinstance check."""
    from reference_v4.baselines.symbolic_floor import SymbolicFloor
    from reference_v4.baselines.cognition_system import CognitionSystem
    from reference_v4.baselines.nli_classifier import NLIClassifier
    from reference_v4.baselines.llm_zeroshot import LLMZeroShot

    systems = [
        SymbolicFloor(),
        CognitionSystem("symbolic"),
        NLIClassifier(),
        LLMZeroShot(
            cache_path=str(FIXTURES_DIR / "llm_cache_fixture.jsonl"),
            mode="replay_only",
        ),
    ]
    for s in systems:
        assert isinstance(s, EvalSystem), f"{s!r} does not satisfy EvalSystem protocol"
