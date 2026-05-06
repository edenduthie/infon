"""Tests for CognitionSystem: wraps InfonStore, no LLM calls.

Run from repo root:
    pytest reference_v4/tests/test_cognition_system.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_v4.benchmarks.types import EvalClaim, MassFunction


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


def test_cognition_symbolic_evaluate(ev_fixture_claim):
    from reference_v4.baselines.cognition_system import CognitionSystem
    system = CognitionSystem("symbolic")
    mass = system.evaluate(ev_fixture_claim)
    assert isinstance(mass, MassFunction)
    assert abs(sum(mass) - 1.0) < 1e-6


def test_cognition_gnn_evaluate(ev_fixture_claim):
    from reference_v4.baselines.cognition_system import CognitionSystem
    system = CognitionSystem("gnn")
    mass = system.evaluate(ev_fixture_claim)
    assert isinstance(mass, MassFunction)
    assert abs(sum(mass) - 1.0) < 1e-6


def test_cognition_does_not_call_bedrock(ev_fixture_claim):
    """CognitionSystem must never invoke boto3 or any LLM service."""
    from reference_v4.baselines.cognition_system import CognitionSystem
    with patch("boto3.client") as mock_client:
        system = CognitionSystem("gnn")
        system.evaluate(ev_fixture_claim)
        mock_client.assert_not_called()


def test_cognition_name_symbolic():
    from reference_v4.baselines.cognition_system import CognitionSystem
    system = CognitionSystem("symbolic")
    assert system.name == "cognition_symbolic"


def test_cognition_name_gnn():
    from reference_v4.baselines.cognition_system import CognitionSystem
    system = CognitionSystem("gnn")
    assert system.name == "cognition_gnn"


def test_cognition_empty_evidence():
    """CognitionSystem with no evidence returns m_theta=1.0 (pure ignorance)."""
    from reference_v4.baselines.cognition_system import CognitionSystem
    claim = EvalClaim(
        claim_id="empty_001",
        claim_text="Vaccines cause autism.",
        evidence_docs=[],
        ground_truth="NEI",
    )
    system = CognitionSystem("symbolic")
    mass = system.evaluate(claim)
    assert isinstance(mass, MassFunction)
    assert abs(sum(mass) - 1.0) < 1e-6
    # With no evidence, should return full ignorance
    assert mass.m_theta == 1.0
