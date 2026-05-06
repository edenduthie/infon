"""Tests for LLM prompt building.

Run from repo root:
    pytest reference_v4/tests/test_llm_prompt.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_v4.baselines.llm_zeroshot import _build_prompt, SYSTEM_PROMPT


def test_two_calls_return_identical_strings():
    system_p1, user_p1 = _build_prompt("claim text", "evidence text")
    system_p2, user_p2 = _build_prompt("claim text", "evidence text")
    assert system_p1 == system_p2
    assert user_p1 == user_p2


def test_system_prompt_contains_nei():
    assert "NEI" in SYSTEM_PROMPT


def test_system_prompt_contains_confidence_1():
    # The system prompt must mention "confidence 1.0" or "confidence: 1.0"
    assert "1.0" in SYSTEM_PROMPT


def test_user_prompt_starts_with_claim():
    _, user_p = _build_prompt("My test claim.", "Evidence here.")
    assert "My test claim." in user_p


def test_user_prompt_contains_evidence():
    _, user_p = _build_prompt("Claim.", "Evidence here.")
    assert "Evidence here." in user_p


def test_user_prompt_first_doc_numbered():
    evidence_text = "[1] First document."
    _, user_p = _build_prompt("Claim.", evidence_text)
    assert "[1]" in user_p


def test_build_prompt_returns_tuple_of_two_strings():
    result = _build_prompt("claim", "evidence")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(s, str) for s in result)
