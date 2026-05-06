"""Wikipedia page text fetcher for HoVer evidence.

Batch-fetches Wikipedia page introductions and caches results locally.
Used by the HoVer loader to replace page-title-only evidence with actual
Wikipedia text.

Cache location: reference_v4/experiments/data/hover/wiki_cache.json
Cache format:   {"Page Title": "intro text ...", ...}

Fetch strategy:
  - MediaWiki action=query with prop=extracts&exintro=True (intro section)
  - 50 titles per request to stay within API limits
  - Respects the MediaWiki User-Agent policy
  - Missing pages (redirects, disambiguation) stored as empty string
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "infon-research-eval/1.0 (academic research; contact: research@infon)"
_BATCH_SIZE = 50
_CACHE_FILE = "reference_v4/experiments/data/hover/wiki_cache.json"
_REQUEST_DELAY = 0.5   # seconds between batches (Wikipedia rate limit courtesy)
_MAX_RETRIES = 4       # retry 429/5xx errors with exponential backoff


def _load_cache(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, str], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=None, separators=(",", ":"))


def _fetch_batch(titles: list[str]) -> dict[str, str]:
    """Fetch intro text for a batch of Wikipedia page titles.

    Returns dict mapping title -> intro text (empty string if not found).
    Retries on 429 (Too Many Requests) and 5xx errors with exponential backoff.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "titles": "|".join(titles),
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "redirects": True,
                    "format": "json",
                },
                headers={"User-Agent": _USER_AGENT},
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = (2 ** attempt) * 2.0  # 2s, 4s, 8s, 16s
                print(f"  Wikipedia API {resp.status_code}, retrying in {wait:.0f}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) * 2.0
            print(f"  Wikipedia API error (attempt {attempt + 1}): {e}, retrying in {wait:.0f}s ...")
            time.sleep(wait)
    else:
        print(f"  Wikipedia API failed after {_MAX_RETRIES} retries for batch of {len(titles)}")
        return {t: "" for t in titles}

    result: dict[str, str] = {}

    # Build redirect map: normalized/redirect title → canonical title
    redirects: dict[str, str] = {}
    for r in data.get("query", {}).get("redirects", []):
        redirects[r["from"]] = r["to"]
    for n in data.get("query", {}).get("normalized", []):
        redirects[n["from"]] = n["to"]

    # Map canonical titles back to original requested titles
    canonical_to_original: dict[str, str] = {}
    for orig in titles:
        # Follow redirect chain
        current = orig
        for _ in range(3):
            if current in redirects:
                current = redirects[current]
            else:
                break
        canonical_to_original[current] = orig

    pages = data.get("query", {}).get("pages", {})
    found_canonical: set[str] = set()
    for _pid, page in pages.items():
        canonical_title = page.get("title", "")
        extract = page.get("extract", "")
        found_canonical.add(canonical_title)
        # Map back to the original requested title
        orig_title = canonical_to_original.get(canonical_title, canonical_title)
        result[orig_title] = extract

    # Fill in any titles that got no result
    for title in titles:
        if title not in result:
            result[title] = ""

    return result


def fetch_wiki_texts(
    page_titles: list[str],
    cache_path: str = _CACHE_FILE,
    verbose: bool = True,
) -> dict[str, str]:
    """Fetch Wikipedia intro text for a list of page titles.

    Results are cached to disk and reused across calls. Only missing pages
    are fetched from the API.

    Args:
        page_titles: Wikipedia page titles to fetch.
        cache_path:  Path to the JSON cache file.
        verbose:     Print progress messages.

    Returns:
        dict mapping page_title -> intro text string.
    """
    cache_file = Path(cache_path)
    cache = _load_cache(cache_file)

    missing = [t for t in page_titles if t not in cache]

    if not missing:
        return {t: cache[t] for t in page_titles}

    if verbose:
        print(
            f"Fetching {len(missing)} Wikipedia pages "
            f"({len(page_titles) - len(missing)} cached) ..."
        )

    # Process in batches
    n_batches = (len(missing) + _BATCH_SIZE - 1) // _BATCH_SIZE
    for batch_idx in range(n_batches):
        batch = missing[batch_idx * _BATCH_SIZE : (batch_idx + 1) * _BATCH_SIZE]
        batch_results = _fetch_batch(batch)
        cache.update(batch_results)

        if verbose and (batch_idx + 1) % 10 == 0:
            print(
                f"  Fetched {min((batch_idx + 1) * _BATCH_SIZE, len(missing))}"
                f"/{len(missing)} pages ..."
            )

        if batch_idx < n_batches - 1:
            time.sleep(_REQUEST_DELAY)

    # Save updated cache
    _save_cache(cache, cache_file)

    return {t: cache.get(t, "") for t in page_titles}
