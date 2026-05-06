"""HoVer dataset loader returning EvalClaim objects.

HoVer real schema (hover-nlp/hover dev_release_v1.1.json):
  {"uid": "hover_dev_0", "claim": "...", "label": "SUPPORTED",
   "supporting_facts": [["WikiPage", 0], ...], "num_hops": 2}

Label mapping:
  "SUPPORTED"     -> "SUPPORTS"
  "NOT_SUPPORTED" -> "NEI"  (HoVer has no REFUTES)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from reference_v4.benchmarks.types import EvalClaim

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
) -> list[EvalClaim]:
    """Load HoVer claims and return as EvalClaim objects.

    Args:
        data_dir: Directory where downloaded data is stored.
        split: Dataset split (only 'dev' supported for now).
        limit: If set, return at most this many claims.
        data_path: Direct path to a JSON file (overrides data_dir; used for fixtures).

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

    claims: list[EvalClaim] = []
    for rec in records:
        uid = str(rec["uid"])
        claim_text = rec["claim"]
        label_raw = rec.get("label", "NOT_SUPPORTED")
        ground_truth = _LABEL_MAP.get(label_raw, "NEI")
        num_hops = int(rec.get("num_hops", 2))

        # Supporting facts are [[page_title, sent_id], ...] — use page titles as evidence docs
        supporting_facts = rec.get("supporting_facts", [])
        evidence_docs = list({sf[0] for sf in supporting_facts if sf})

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
