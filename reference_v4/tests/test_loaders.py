"""Tests for dataset loaders: HoVer, AVeriTeC, SciFact.

Run from repo root:
    pytest reference_v4/tests/test_loaders.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `reference_v4.benchmarks` is importable.
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_v4.benchmarks.types import EvalClaim

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hover_claims():
    from reference_v4.benchmarks.hover import load_hover
    fixture = FIXTURES_DIR / "hover_fixture.json"
    return load_hover(data_path=str(fixture))


@pytest.fixture
def averitec_claims():
    from reference_v4.benchmarks.averitec import load_averitec
    fixture = FIXTURES_DIR / "averitec_fixture.json"
    return load_averitec(data_path=str(fixture))


@pytest.fixture
def scifact_claims():
    from reference_v4.benchmarks.scifact import load_scifact
    fixture = FIXTURES_DIR / "scifact_fixture.json"
    return load_scifact(data_path=str(fixture))


# ---------------------------------------------------------------------------
# Generic loader tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loader_name", ["hover", "averitec", "scifact"])
def test_loader_returns_eval_claims(loader_name):
    """Each loader must return a non-empty list[EvalClaim] from fixture data."""
    if loader_name == "hover":
        from reference_v4.benchmarks.hover import load_hover
        claims = load_hover(data_path=str(FIXTURES_DIR / "hover_fixture.json"))
    elif loader_name == "averitec":
        from reference_v4.benchmarks.averitec import load_averitec
        claims = load_averitec(data_path=str(FIXTURES_DIR / "averitec_fixture.json"))
    else:
        from reference_v4.benchmarks.scifact import load_scifact
        claims = load_scifact(data_path=str(FIXTURES_DIR / "scifact_fixture.json"))

    assert isinstance(claims, list), "Loader must return a list"
    assert len(claims) > 0, "Loader must return at least one claim"

    for claim in claims:
        assert isinstance(claim, EvalClaim), f"Each item must be EvalClaim, got {type(claim)}"
        assert isinstance(claim.claim_id, str) and claim.claim_id, "claim_id must be non-empty string"
        assert isinstance(claim.claim_text, str) and claim.claim_text, "claim_text must be non-empty string"
        assert isinstance(claim.evidence_docs, list), "evidence_docs must be a list"
        assert all(isinstance(d, str) for d in claim.evidence_docs), "evidence_docs items must be strings"
        assert claim.ground_truth in ("SUPPORTS", "REFUTES", "NEI"), (
            f"ground_truth must be SUPPORTS/REFUTES/NEI, got {claim.ground_truth!r}"
        )
        assert isinstance(claim.metadata, dict), "metadata must be a dict"


# ---------------------------------------------------------------------------
# Dataset-specific tests
# ---------------------------------------------------------------------------

def test_hover_has_num_hops(hover_claims):
    """HoVer claims must carry num_hops in metadata with values in {2, 3, 4}."""
    for claim in hover_claims:
        assert "num_hops" in claim.metadata, f"claim {claim.claim_id} missing num_hops in metadata"
        assert claim.metadata["num_hops"] in {2, 3, 4}, (
            f"num_hops must be 2, 3, or 4; got {claim.metadata['num_hops']}"
        )


def test_hover_ground_truth_mapping(hover_claims):
    """HoVer SUPPORTED -> SUPPORTS, NOT_SUPPORTED -> NEI (no REFUTES in HoVer)."""
    for claim in hover_claims:
        assert claim.ground_truth in ("SUPPORTS", "NEI"), (
            f"HoVer should only produce SUPPORTS or NEI, got {claim.ground_truth!r}"
        )


def test_averitec_has_nei(averitec_claims):
    """AVeriTeC fixture includes a NEI label (Not Enough Evidence)."""
    ground_truths = [c.ground_truth for c in averitec_claims]
    assert "NEI" in ground_truths, "AVeriTeC claims must include at least one NEI"


def test_averitec_evidence_format(averitec_claims):
    """AVeriTeC evidence docs should be QA pairs formatted as '[Q] ... [A] ...'."""
    for claim in averitec_claims:
        for doc in claim.evidence_docs:
            assert "[Q]" in doc and "[A]" in doc, (
                f"AVeriTeC evidence doc should contain '[Q]' and '[A]', got: {doc[:80]!r}"
            )


def test_averitec_conflicting_maps_to_nei(averitec_claims):
    """AVeriTeC 'Conflicting Evidence/Cherrypicking' should map to NEI."""
    # Fixture claim_id=4 has "Conflicting Evidence/Cherrypicking"
    conflicting = [c for c in averitec_claims if "Some studies suggest" in c.claim_text]
    assert len(conflicting) == 1
    assert conflicting[0].ground_truth == "NEI"


def test_scifact_has_rationales(scifact_claims):
    """SciFact claims with evidence must have rationales (sentence indices) in metadata."""
    claims_with_evidence = [c for c in scifact_claims if c.evidence_docs]
    assert len(claims_with_evidence) > 0, "Expected some SciFact claims to have evidence"
    for claim in claims_with_evidence:
        assert claim.metadata.get("rationales"), (
            f"SciFact claim {claim.claim_id} with evidence_docs must have rationales in metadata"
        )


def test_scifact_ground_truth_mapping(scifact_claims):
    """SciFact SUPPORTS -> SUPPORTS, CONTRADICT -> REFUTES, no evidence -> NEI."""
    ground_truths = {c.ground_truth for c in scifact_claims}
    # Fixture has SUPPORTS, CONTRADICT (->REFUTES), and empty evidence (->NEI)
    assert "SUPPORTS" in ground_truths
    assert "REFUTES" in ground_truths
    assert "NEI" in ground_truths


def test_limit_parameter():
    """All loaders must respect the limit parameter."""
    from reference_v4.benchmarks.hover import load_hover
    from reference_v4.benchmarks.averitec import load_averitec
    from reference_v4.benchmarks.scifact import load_scifact

    hover_limited = load_hover(
        data_path=str(FIXTURES_DIR / "hover_fixture.json"), limit=2
    )
    assert len(hover_limited) <= 2, f"Expected at most 2 HoVer claims with limit=2, got {len(hover_limited)}"

    averitec_limited = load_averitec(
        data_path=str(FIXTURES_DIR / "averitec_fixture.json"), limit=2
    )
    assert len(averitec_limited) <= 2, f"Expected at most 2 AVeriTeC claims with limit=2, got {len(averitec_limited)}"

    scifact_limited = load_scifact(
        data_path=str(FIXTURES_DIR / "scifact_fixture.json"), limit=2
    )
    assert len(scifact_limited) <= 2, f"Expected at most 2 SciFact claims with limit=2, got {len(scifact_limited)}"
