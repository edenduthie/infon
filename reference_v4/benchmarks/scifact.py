"""SciFact dataset loader returning EvalClaim objects.

SciFact real schema:
  Claims file (claims_test.jsonl):
    {
      "id": 1,
      "claim": "...",
      "cited_doc_ids": [4346465],
      "evidence": {
        "4346465": [{"sentences": [0, 1], "label": "SUPPORTS"}]
      }
    }

  Corpus file (corpus.jsonl):
    {"doc_id": 4346465, "title": "...", "abstract": ["sentence0", "sentence1", ...]}

  Fixture file (scifact_fixture.json) combines both under a single JSON object:
    {"claims": [...], "corpus": [...]}

Label mapping:
  If any evidence entry has "SUPPORTS"  -> "SUPPORTS"
  If any evidence entry has "CONTRADICT" -> "REFUTES"
  If no evidence                         -> "NEI"

Evidence docs: joined abstract sentences for cited_doc_ids.
Rationale: sentence indices from evidence entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from reference_v4.benchmarks.types import EvalClaim

_SCIFACT_S3_TARBALL_URL = (
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
)

_LABEL_PRIORITY: dict[str, int] = {
    # Both "SUPPORTS" (fixture) and "SUPPORT" (real dev/train) are handled
    "SUPPORTS": 2,
    "SUPPORT": 2,
    "CONTRADICT": 1,
}

_LABEL_MAP: dict[str, str] = {
    "SUPPORTS": "SUPPORTS",
    "SUPPORT": "SUPPORTS",
    "CONTRADICT": "REFUTES",
}


def _download_scifact(dest_dir: Path) -> None:
    """Download and extract the SciFact data tarball from S3."""
    import tarfile
    import io

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SciFact data from {_SCIFACT_S3_TARBALL_URL} ...")
    response = requests.get(_SCIFACT_S3_TARBALL_URL, timeout=300, stream=True)
    response.raise_for_status()

    # Extract the tarball in-memory, placing files directly into dest_dir
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Strip the leading 'data/' prefix from paths in the tarball
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            relative_path = Path(*parts[1:])
            dest_path = dest_dir / relative_path
            if member.isdir():
                dest_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    dest_path.write_bytes(file_obj.read())

    print(f"SciFact data extracted to {dest_dir}")


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_corpus_index(corpus_records: list[dict]) -> dict[int, dict]:
    """Build a doc_id -> record index from corpus records."""
    return {rec["doc_id"]: rec for rec in corpus_records}


def _get_ground_truth(evidence: dict) -> str:
    """Determine ground truth label from evidence dict.

    Args:
        evidence: Dict mapping doc_id (str) -> list of {"sentences": [...], "label": "..."}

    Returns:
        "SUPPORTS", "REFUTES", or "NEI".
    """
    if not evidence:
        return "NEI"

    best_label: str | None = None
    best_priority = -1

    for _doc_id, entries in evidence.items():
        for entry in entries:
            raw_label = entry.get("label", "")
            priority = _LABEL_PRIORITY.get(raw_label, 0)
            if priority > best_priority:
                best_priority = priority
                best_label = raw_label

    if best_label is None:
        return "NEI"
    return _LABEL_MAP.get(best_label, "NEI")


def _get_rationales(evidence: dict) -> list[int]:
    """Collect all sentence indices mentioned in evidence entries."""
    sentence_indices: list[int] = []
    for _doc_id, entries in evidence.items():
        for entry in entries:
            sentence_indices.extend(entry.get("sentences", []))
    return sorted(set(sentence_indices))


def _get_evidence_docs(
    cited_doc_ids: list[int],
    corpus_index: dict[int, dict],
) -> list[str]:
    """Join abstract sentences for each cited doc into evidence strings."""
    docs = []
    for doc_id in cited_doc_ids:
        record = corpus_index.get(doc_id)
        if record is None:
            continue
        abstract: list[str] = record.get("abstract", [])
        if abstract:
            docs.append(" ".join(abstract))
    return docs


def _load_from_combined_fixture(
    data: dict,
    limit: int | None,
) -> list[EvalClaim]:
    """Load from a fixture JSON that has {"claims": [...], "corpus": [...]}."""
    corpus_records: list[dict] = data.get("corpus", [])
    corpus_index = _build_corpus_index(corpus_records)
    claims_records: list[dict] = data.get("claims", [])

    if limit is not None:
        claims_records = claims_records[:limit]

    return _build_eval_claims(claims_records, corpus_index)


def _build_eval_claims(
    claims_records: list[dict],
    corpus_index: dict[int, dict],
) -> list[EvalClaim]:
    """Convert raw SciFact claim records into EvalClaim objects."""
    claims: list[EvalClaim] = []
    for rec in claims_records:
        claim_id = str(rec["id"])
        claim_text = rec["claim"]
        evidence: dict = rec.get("evidence", {})
        # Keys in evidence dict may be ints or strings depending on source
        evidence_str_keys = {str(k): v for k, v in evidence.items()}

        cited_doc_ids: list[int] = rec.get("cited_doc_ids", [])

        ground_truth = _get_ground_truth(evidence_str_keys)
        rationales = _get_rationales(evidence_str_keys)
        evidence_docs = _get_evidence_docs(cited_doc_ids, corpus_index)

        metadata: dict = {"rationales": rationales}

        claims.append(
            EvalClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                evidence_docs=evidence_docs,
                ground_truth=ground_truth,
                metadata=metadata,
            )
        )
    return claims


def load_scifact(
    data_dir: str = "reference_v4/experiments/data/scifact",
    split: str = "test",
    limit: int | None = None,
    data_path: str | None = None,
) -> list[EvalClaim]:
    """Load SciFact claims and return as EvalClaim objects.

    Args:
        data_dir: Directory where downloaded data files are stored.
        split: Dataset split ('test' by default, uses claims_test.jsonl).
        limit: If set, return at most this many claims.
        data_path: Direct path to a combined fixture JSON file (overrides data_dir).

    Returns:
        List of EvalClaim with metadata["rationales"] set.
    """
    if data_path is not None:
        # Fixture mode: single JSON file with {"claims": [...], "corpus": [...]}
        with open(data_path) as f:
            data = json.load(f)
        return _load_from_combined_fixture(data, limit)

    data_path_dir = Path(data_dir)
    claims_file = data_path_dir / f"claims_{split}.jsonl"
    corpus_file = data_path_dir / "corpus.jsonl"

    if not claims_file.exists() or not corpus_file.exists():
        _download_scifact(data_path_dir)

    claims_records = _load_jsonl(claims_file)
    corpus_records = _load_jsonl(corpus_file)

    if limit is not None:
        claims_records = claims_records[:limit]

    corpus_index = _build_corpus_index(corpus_records)
    return _build_eval_claims(claims_records, corpus_index)
