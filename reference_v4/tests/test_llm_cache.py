"""Tests for LLMCache: replay-only and record modes.

Run from repo root:
    pytest reference_v4/tests/test_llm_cache.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

from reference_v4.baselines._llm_cache import LLMCache, LLMCacheMissError


@pytest.fixture
def fixture_cache():
    return FIXTURES_DIR / "llm_cache_fixture.jsonl"


def _first_valid_key(fixture_cache: Path) -> str:
    """Return the first valid (non-malformed) key from the fixture."""
    for line in fixture_cache.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            return entry["key"]
        except (json.JSONDecodeError, KeyError):
            continue
    raise RuntimeError("No valid keys found in fixture")


def test_replay_hit(fixture_cache):
    cache = LLMCache(fixture_cache, mode="replay_only")
    key = _first_valid_key(fixture_cache)
    result = cache.get(key)
    assert "response_text" in result
    assert isinstance(result["response_text"], str)


def test_replay_miss_raises(tmp_path):
    cache = LLMCache(tmp_path / "empty.jsonl", mode="replay_only")
    with pytest.raises(LLMCacheMissError) as exc_info:
        cache.get("a" * 64)
    assert len(str(exc_info.value)) >= 12  # key prefix in message


def test_replay_miss_message_contains_key_prefix(tmp_path):
    cache = LLMCache(tmp_path / "empty.jsonl", mode="replay_only")
    test_key = "abcdef1234567890" + "0" * 48
    with pytest.raises(LLMCacheMissError) as exc_info:
        cache.get(test_key)
    assert "abcdef12" in str(exc_info.value)


def test_record_mode_appends(tmp_path):
    cache_path = tmp_path / "record_test.jsonl"
    cache = LLMCache(cache_path, mode="record")
    key = LLMCache.make_key("model-x", "sys", "user", 0.0, 256, "bedrock")
    cache.put(key, '{"verdict": "SUPPORTS", "confidence": 0.9}', 100, 50)
    # Read back
    cache2 = LLMCache(cache_path, mode="replay_only")
    entry = cache2.get(key)
    assert entry["response_text"] == '{"verdict": "SUPPORTS", "confidence": 0.9}'
    assert entry["input_tokens"] == 100
    assert entry["output_tokens"] == 50


def test_record_mode_multiple_entries(tmp_path):
    cache_path = tmp_path / "multi.jsonl"
    cache = LLMCache(cache_path, mode="record")
    keys = []
    for i in range(3):
        k = LLMCache.make_key(f"model-{i}", "sys", "user", 0.0, 256, "bedrock")
        cache.put(k, f"response_{i}", 10 * i, 5 * i)
        keys.append(k)
    # Reload
    cache2 = LLMCache(cache_path, mode="replay_only")
    for i, k in enumerate(keys):
        assert cache2.get(k)["response_text"] == f"response_{i}"


def test_token_counter_accumulates(fixture_cache):
    cache = LLMCache(fixture_cache, mode="replay_only")
    # Get all valid keys
    valid_keys = []
    for line in fixture_cache.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            valid_keys.append(entry["key"])
        except (json.JSONDecodeError, KeyError):
            continue
    assert len(valid_keys) >= 2, "Need at least 2 valid keys for token counter test"
    cache.get(valid_keys[0])
    cache.get(valid_keys[1])
    assert cache.total_input_tokens > 0


def test_make_key_is_deterministic():
    k1 = LLMCache.make_key("model", "sys", "user", 0.0, 256, "bedrock")
    k2 = LLMCache.make_key("model", "sys", "user", 0.0, 256, "bedrock")
    assert k1 == k2


def test_make_key_is_sha256():
    key = LLMCache.make_key("model", "sys", "user", 0.0, 256, "bedrock")
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_make_key_differs_on_model():
    k1 = LLMCache.make_key("model-a", "sys", "user", 0.0, 256, "bedrock")
    k2 = LLMCache.make_key("model-b", "sys", "user", 0.0, 256, "bedrock")
    assert k1 != k2


def test_load_skips_malformed_lines(fixture_cache):
    """Loading a fixture with malformed JSON lines must not raise."""
    # The fixture has a malformed line — loading should succeed
    cache = LLMCache(fixture_cache, mode="replay_only")
    assert isinstance(cache.total_input_tokens, int)


def test_empty_file_loads_ok(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("")
    cache = LLMCache(empty_path, mode="replay_only")
    assert cache.total_input_tokens == 0
