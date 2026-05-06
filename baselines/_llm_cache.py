"""LLM response cache: replay pre-recorded responses or record new ones.

Modes:
  replay_only  — raise LLMCacheMissError on a cache miss (default, safe for CI)
  record       — call the real LLM, append to the JSONL file

Cache format: one JSON object per line (JSONL), keyed by sha256 of the
model + prompt + parameters. Token counts are tracked for cost accounting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LLMCacheMissError(KeyError):
    """Raised when a key is not in the cache and mode is 'replay_only'."""


class LLMCache:
    """File-backed key-value cache for LLM responses.

    Args:
        cache_path: Path to the JSONL cache file (created if absent).
        mode:       "replay_only" (default) or "record".
    """

    def __init__(self, cache_path: str | Path, mode: str = "replay_only") -> None:
        self.cache_path = Path(cache_path)
        self.mode = mode
        self.total_input_tokens: int = 0
        self._cache: dict[str, dict] = {}
        self._load()

    # ── key generation ──────────────────────────────────────────────────

    @staticmethod
    def make_key(
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        api_provider: str = "bedrock",
    ) -> str:
        """Deterministic sha256 key over all parameters that affect the response."""
        raw = f"{model_id}|{system_prompt}|{user_prompt}|{temperature}|{max_tokens}|{api_provider}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── read ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> dict:
        """Return the cached entry for ``key``.

        Returns:
            dict with keys: "response_text", "input_tokens", "output_tokens"

        Raises:
            LLMCacheMissError: if the key is not in the cache.
        """
        if key not in self._cache:
            raise LLMCacheMissError(f"Cache miss: key prefix {key[:12]}")
        entry = self._cache[key]
        self.total_input_tokens += entry.get("input_tokens", 0)
        return entry

    # ── write ────────────────────────────────────────────────────────────

    def put(
        self,
        key: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Append a new entry to the cache file and in-memory index.

        Only call in "record" mode. The method does not enforce this to
        keep the class simple; callers are responsible for the mode check.
        """
        entry: dict = {
            "key": key,
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._cache[key] = entry
        self.total_input_tokens += input_tokens

    # ── internal ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all valid entries from the JSONL file. Skips malformed lines."""
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError):
                    # Silently skip malformed or key-less lines
                    continue
