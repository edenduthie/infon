"""AVeriTeC dataset loader returning EvalClaim objects.

AVeriTeC real schema (MichSchli/AVeriTeC dev.json):
  {
    "claim": "...",
    "label": "Supported",
    "questions": [
      {
        "question": "...",
        "answers": [{"answer": "...", "answer_type": "...", ...}]
      }
    ]
  }

Fixture schema (simplified):
  {
    "claim_id": 0,
    "claim": "...",
    "label": "Supported",
    "evidence": [{"question": "...", "answer": "..."}]
  }

Label mapping:
  "Supported"                          -> "SUPPORTS"
  "Refuted"                            -> "REFUTES"
  "Not Enough Evidence"                -> "NEI"
  "Conflicting Evidence/Cherrypicking" -> "NEI"

Evidence docs: each QA pair is formatted as "[Q] question [A] answer".
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from reference_v4.benchmarks.types import EvalClaim

_AVERITEC_DEV_URL = (
    "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"
)

_LABEL_MAP: dict[str, str] = {
    "Supported": "SUPPORTS",
    "Refuted": "REFUTES",
    "Not Enough Evidence": "NEI",
    "Conflicting Evidence/Cherrypicking": "NEI",
}


def _download_averitec(dest_path: Path) -> None:
    """Download the AVeriTeC dev file from GitHub."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AVeriTeC dev set from {_AVERITEC_DEV_URL} ...")
    response = requests.get(_AVERITEC_DEV_URL, timeout=120)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    print(f"Saved to {dest_path}")


def _format_flat_evidence(evidence_list: list[dict]) -> list[str]:
    """Format flat evidence list [{"question": ..., "answer": ...}] as QA strings."""
    docs = []
    for item in evidence_list:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question or answer:
            docs.append(f"[Q] {question} [A] {answer}")
    return docs


def _format_nested_questions(questions: list[dict]) -> list[str]:
    """Format real AVeriTeC questions schema as QA strings.

    Real schema: [{"question": "...", "answers": [{"answer": "..."}]}]
    Each question is expanded into one doc per answer.
    """
    docs = []
    for item in questions:
        question = item.get("question", "").strip()
        answers = item.get("answers", [])
        for ans_item in answers:
            answer = ans_item.get("answer", "").strip()
            if question or answer:
                docs.append(f"[Q] {question} [A] {answer}")
    return docs


def _extract_evidence_docs(rec: dict) -> list[str]:
    """Extract evidence docs from a record, handling both schemas."""
    if "questions" in rec:
        # Real AVeriTeC schema
        return _format_nested_questions(rec["questions"])
    elif "evidence" in rec:
        # Fixture / simplified schema
        return _format_flat_evidence(rec["evidence"])
    return []


def load_averitec(
    data_dir: str = "reference_v4/experiments/data/averitec",
    split: str = "dev",
    limit: int | None = None,
    data_path: str | None = None,
) -> list[EvalClaim]:
    """Load AVeriTeC claims and return as EvalClaim objects.

    Args:
        data_dir: Directory where downloaded data is stored.
        split: Dataset split ('dev' by default).
        limit: If set, return at most this many claims.
        data_path: Direct path to a JSON file (overrides data_dir; used for fixtures).

    Returns:
        List of EvalClaim with evidence formatted as "[Q] ... [A] ..." strings.
    """
    if data_path is not None:
        file_path = Path(data_path)
    else:
        filename = f"averitec_{split}.json"
        file_path = Path(data_dir) / filename
        if not file_path.exists():
            _download_averitec(file_path)

    with open(file_path) as f:
        records = json.load(f)

    if limit is not None:
        records = records[:limit]

    claims: list[EvalClaim] = []
    for idx, rec in enumerate(records):
        claim_id = str(rec.get("claim_id", idx))
        claim_text = rec["claim"]
        label_raw = rec.get("label", "Not Enough Evidence")
        ground_truth = _LABEL_MAP.get(label_raw, "NEI")
        evidence_docs = _extract_evidence_docs(rec)

        claims.append(
            EvalClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                evidence_docs=evidence_docs,
                ground_truth=ground_truth,
                metadata={"original_label": label_raw},
            )
        )

    return claims
