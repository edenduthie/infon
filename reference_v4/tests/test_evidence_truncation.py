"""Tests for evidence truncation helpers.

Run from repo root:
    pytest reference_v4/tests/test_evidence_truncation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_v4.baselines.llm_zeroshot import _truncate_evidence


def _make_doc(n_words: int) -> str:
    return " ".join(["word"] * n_words)


def test_single_long_doc_truncated():
    doc = _make_doc(700)
    result = _truncate_evidence([doc])
    assert "[truncated]" in result
    words = result.split()
    # Should be roughly 600 words + "[truncated]"
    assert len(words) <= 602  # 600 words + "[truncated]" as one token


def test_single_short_doc_not_truncated():
    doc = _make_doc(100)
    result = _truncate_evidence([doc])
    assert "[truncated]" not in result
    assert "[1 document omitted]" not in result


def test_six_docs_omits_one():
    docs = [_make_doc(100) for _ in range(6)]
    result = _truncate_evidence(docs)
    assert "[1 document omitted]" in result


def test_five_docs_no_omission():
    docs = [_make_doc(50) for _ in range(5)]
    result = _truncate_evidence(docs)
    assert "omitted" not in result


def test_empty_list_returns_placeholder():
    result = _truncate_evidence([])
    assert result == "[none provided]"


def test_six_docs_contains_five_sections():
    """After limiting to 5 docs, result should have 5 numbered sections."""
    docs = [f"Document {i} content." for i in range(6)]
    result = _truncate_evidence(docs)
    # Should include [1] through [5] but omit the 6th
    for i in range(1, 6):
        assert f"[{i}]" in result


def test_word_count_within_limit():
    """With a 700-word doc, output word count should not exceed ~602."""
    doc = _make_doc(700)
    result = _truncate_evidence([doc])
    # Count words before "[truncated]"
    parts = result.split("[truncated]")
    words_before = len(parts[0].split())
    assert words_before <= 601
