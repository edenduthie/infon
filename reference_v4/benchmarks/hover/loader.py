"""HoVer dataset loader returning EvalClaim objects.

HoVer real schema (hover-nlp/hover dev_release_v1.1.json):
  {"uid": "hover_dev_0", "claim": "...", "label": "SUPPORTED",
   "supporting_facts": [["WikiPage", 0], ...], "num_hops": 2}

Label mapping:
  "SUPPORTED"     -> "SUPPORTS"
  "NOT_SUPPORTED" -> "NEI"  (HoVer has no REFUTES)

Evidence strategy: fetch the Wikipedia introduction section for each
supporting_facts page via the MediaWiki API and cache locally. Page titles
alone (the previous approach) produce zero SPLADE activations and are
useless as evidence. The introduction text gives the CognitionSystem
enough context to extract infons.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from reference_v4.benchmarks.types import EvalClaim
from reference_v4.benchmarks.hover.wiki_fetcher import fetch_wiki_texts

_HOVER_DEV_URL = (
    "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/"
    "hover_dev_release_v1.1.json"
)

_LABEL_MAP: dict[str, str] = {
    "SUPPORTED": "SUPPORTS",
    "NOT_SUPPORTED": "NEI",
}


def _download_hover(dest_path: Path) -> None:
    """Download the HoVer dev file from GitHub."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HoVer dev set from {_HOVER_DEV_URL} ...")
    response = requests.get(_HOVER_DEV_URL, timeout=120)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    print(f"Saved to {dest_path}")


def load_hover(
    data_dir: str = "reference_v4/experiments/data/hover",
    split: str = "dev",
    limit: int | None = None,
    data_path: str | None = None,
    fetch_wiki: bool = True,
) -> list[EvalClaim]:
    """Load HoVer claims and return as EvalClaim objects.

    Args:
        data_dir: Directory where downloaded data is stored.
        split: Dataset split (only 'dev' supported for now).
        limit: If set, return at most this many claims.
        data_path: Direct path to a JSON file (overrides data_dir; used for fixtures).
        fetch_wiki: If True, fetch Wikipedia intro text for evidence pages
                    and cache locally. If False, use raw page titles only
                    (useful for unit tests that supply fixture data).

    Returns:
        List of EvalClaim with metadata["num_hops"] set.
    """
    if data_path is not None:
        file_path = Path(data_path)
    else:
        filename = f"hover_{split}_release_v1.1.json"
        file_path = Path(data_dir) / filename
        if not file_path.exists():
            _download_hover(file_path)

    with open(file_path) as f:
        records = json.load(f)

    if limit is not None:
        records = records[:limit]

    # Collect unique page titles across the batch for bulk Wikipedia fetch
    all_page_titles: set[str] = set()
    for rec in records:
        for sf in rec.get("supporting_facts", []):
            if sf:
                all_page_titles.add(sf[0])

    wiki_texts: dict[str, str] = {}
    if fetch_wiki and all_page_titles and data_path is None:
        # Use the data_dir-relative cache path
        cache_path = str(Path(data_dir) / "wiki_cache.json")
        wiki_texts = fetch_wiki_texts(list(all_page_titles), cache_path=cache_path)

    claims: list[EvalClaim] = []
    for rec in records:
        uid = str(rec["uid"])
        claim_text = rec["claim"]
        label_raw = rec.get("label", "NOT_SUPPORTED")
        ground_truth = _LABEL_MAP.get(label_raw, "NEI")
        num_hops = int(rec.get("num_hops", 2))

        supporting_facts = rec.get("supporting_facts", [])
        unique_pages = list({sf[0] for sf in supporting_facts if sf})

        if wiki_texts:
            # Use Wikipedia intro text as evidence; skip pages with no text
            evidence_docs = [
                wiki_texts[page]
                for page in unique_pages
                if wiki_texts.get(page)
            ]
        else:
            # Fallback: page titles only (unit tests / fixture mode)
            evidence_docs = unique_pages

        claims.append(
            EvalClaim(
                claim_id=uid,
                claim_text=claim_text,
                evidence_docs=evidence_docs,
                ground_truth=ground_truth,
                metadata={"num_hops": num_hops, "supporting_facts": supporting_facts},
            )
        )

    return claims
