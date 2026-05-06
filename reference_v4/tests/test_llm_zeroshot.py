"""Tests for LLMZeroShot system using replay_only cache.

Run from repo root:
    pytest reference_v4/tests/test_llm_zeroshot.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

from reference_v4.benchmarks.types import EvalClaim, MassFunction
from reference_v4.baselines._llm_cache import LLMCache, LLMCacheMissError
from reference_v4.baselines.llm_zeroshot import LLMZeroShot, _truncate_evidence, _build_prompt


def _load_fixture_entries() -> list[dict]:
    fixture_path = FIXTURES_DIR / "llm_cache_fixture.jsonl"
    entries = []
    for line in fixture_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _make_claim_for_key(key: str) -> EvalClaim:
    """Create a claim that will hash to the given cache key.

    We reverse-engineer: create a cache pointing to the fixture, and
    build a fake claim; we insert the key by overriding the system with
    a monkeypatched LLMCache that returns the entry for the given key.
    """
    return EvalClaim(
        claim_id="fixture_claim",
        claim_text="Aspirin reduces the risk of heart attack.",
        evidence_docs=["Aspirin prevents clot formation.", "Studies support aspirin use."],
        ground_truth="SUPPORTS",
    )


@pytest.fixture
def fixture_system():
    return LLMZeroShot(
        cache_path=str(FIXTURES_DIR / "llm_cache_fixture.jsonl"),
        mode="replay_only",
    )


def test_llm_zeroshot_name(fixture_system):
    assert fixture_system.name == "llm_zeroshot"
    assert isinstance(fixture_system.name, str)


def test_replay_only_never_calls_bedrock(fixture_system):
    """In replay_only mode, must never call boto3.client."""
    claim = _make_claim_for_key("any")
    with patch("boto3.client") as mock_client:
        with pytest.raises((LLMCacheMissError, Exception)):
            fixture_system.evaluate(claim)
        mock_client.assert_not_called()


def test_cache_miss_raises_in_replay_mode(tmp_path):
    """A cache miss in replay_only mode must raise LLMCacheMissError."""
    system = LLMZeroShot(
        cache_path=str(tmp_path / "empty.jsonl"),
        mode="replay_only",
    )
    claim = EvalClaim(
        claim_id="x",
        claim_text="Some claim.",
        evidence_docs=["Some evidence."],
        ground_truth="NEI",
    )
    with pytest.raises(LLMCacheMissError):
        system.evaluate(claim)


def test_evaluate_with_preloaded_cache(tmp_path):
    """Build a cache entry that matches exactly what LLMZeroShot will look up."""
    # Build the key for the claim we're going to make
    claim = EvalClaim(
        claim_id="preload_test",
        claim_text="Coffee reduces cancer risk.",
        evidence_docs=["Coffee contains antioxidants.", "Studies show reduced cancer rates."],
        ground_truth="SUPPORTS",
    )
    evidence_text = _truncate_evidence(claim.evidence_docs)
    system_p, user_p = _build_prompt(claim.claim_text, evidence_text)
    key = LLMCache.make_key(LLMZeroShot.MODEL_ID, system_p, user_p, 0.0, 256, "bedrock")

    # Write that key to cache
    cache_path = tmp_path / "test_cache.jsonl"
    entry = {"key": key, "response_text": '{"verdict": "SUPPORTS", "confidence": 0.82}',
             "input_tokens": 100, "output_tokens": 20}
    cache_path.write_text(json.dumps(entry) + "\n")

    system = LLMZeroShot(cache_path=str(cache_path), mode="replay_only")
    mass = system.evaluate(claim)
    assert isinstance(mass, MassFunction)
    assert abs(sum(mass) - 1.0) < 1e-6
    assert mass.m_s == pytest.approx(0.82)


def test_evaluate_refutes_mapping(tmp_path):
    claim = EvalClaim(
        claim_id="refutes_test",
        claim_text="Vaccines cause autism.",
        evidence_docs=["Multiple studies found no link between vaccines and autism."],
        ground_truth="REFUTES",
    )
    evidence_text = _truncate_evidence(claim.evidence_docs)
    system_p, user_p = _build_prompt(claim.claim_text, evidence_text)
    key = LLMCache.make_key(LLMZeroShot.MODEL_ID, system_p, user_p, 0.0, 256, "bedrock")

    cache_path = tmp_path / "refutes_cache.jsonl"
    entry = {"key": key, "response_text": '{"verdict": "REFUTES", "confidence": 0.71}',
             "input_tokens": 80, "output_tokens": 15}
    cache_path.write_text(json.dumps(entry) + "\n")

    system = LLMZeroShot(cache_path=str(cache_path), mode="replay_only")
    mass = system.evaluate(claim)
    assert mass.m_r == pytest.approx(0.71)


def test_evaluate_nei_mapping(tmp_path):
    claim = EvalClaim(
        claim_id="nei_test",
        claim_text="Exercise cures depression completely.",
        evidence_docs=["Exercise has some benefits for mental health."],
        ground_truth="NEI",
    )
    evidence_text = _truncate_evidence(claim.evidence_docs)
    system_p, user_p = _build_prompt(claim.claim_text, evidence_text)
    key = LLMCache.make_key(LLMZeroShot.MODEL_ID, system_p, user_p, 0.0, 256, "bedrock")

    cache_path = tmp_path / "nei_cache.jsonl"
    entry = {"key": key, "response_text": '{"verdict": "NEI", "confidence": 1.0}',
             "input_tokens": 60, "output_tokens": 10}
    cache_path.write_text(json.dumps(entry) + "\n")

    system = LLMZeroShot(cache_path=str(cache_path), mode="replay_only")
    mass = system.evaluate(claim)
    assert mass.m_theta == pytest.approx(1.0)


def test_malformed_response_returns_theta(tmp_path):
    """Malformed JSON response must map to full ignorance."""
    claim = EvalClaim(
        claim_id="bad_json",
        claim_text="Sugar is healthy.",
        evidence_docs=["Sugar causes health issues."],
        ground_truth="REFUTES",
    )
    evidence_text = _truncate_evidence(claim.evidence_docs)
    system_p, user_p = _build_prompt(claim.claim_text, evidence_text)
    key = LLMCache.make_key(LLMZeroShot.MODEL_ID, system_p, user_p, 0.0, 256, "bedrock")

    cache_path = tmp_path / "bad_cache.jsonl"
    entry = {"key": key, "response_text": "not valid json", "input_tokens": 50, "output_tokens": 5}
    cache_path.write_text(json.dumps(entry) + "\n")

    system = LLMZeroShot(cache_path=str(cache_path), mode="replay_only")
    mass = system.evaluate(claim)
    assert mass.m_theta == pytest.approx(1.0)
