"""Persistent findings — notes and investigation artifacts that outlive a session.

What belongs here vs. not:

  belongs:
    • "Nvidia's Q1 supply chain investigation — key chain is TSMC → HBM
       → SK Hynix, later contradicted by d11 Samsung retraction"
    • "Schema note: 'partner' needs 'partnership' and 'venture' tokens
       to catch all surface forms on this corpus"
    • Any synthesis a user or agent produces on top of the store.

  does NOT belong (recompute, don't persist):
    • extraction_report output (cheap, deterministic)
    • constraint aggregates (cheap, always current)
    • trajectory sequences (cheap index-read)
    • reasoner verdicts (cheap, fully reproducible from the manifest)

Layout on disk:

  <root>/findings/<finding_id>.json
      {
        "id": "f_20260504T072030_a3b1",
        "created_at": "...",
        "title": "...",
        "body": "...",       # markdown OK
        "tags": ["..."],
        "cites": [            # optional — reasoner sources the finding is anchored to
          {"infon_id": "...", "cassette_id": "...", "sentence": "..."}
        ],
        "schema_ref": "...",  # the schema active when finding was written
        "snapshot_id": "..."  # the manifest HEAD when written
      }

Findings are flat files, not indexed — at ≤10k findings the scan cost is
in single-digit ms. Add an index if someone hits that ceiling.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Finding:
    """A persisted note tied to a store's state at a moment in time."""
    id: str
    created_at: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    cites: list[dict] = field(default_factory=list)
    schema_ref: str = ""
    snapshot_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _findings_dir(root: str) -> str:
    return os.path.join(root, "findings")


def _finding_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"f_{ts}_{uuid.uuid4().hex[:4]}"


def record(root: str, *, title: str, body: str,
           tags: list[str] | None = None,
           cites: list[dict] | None = None,
           schema_ref: str = "",
           snapshot_id: str = "") -> Finding:
    """Write a finding under <root>/findings/. Returns the Finding."""
    d = _findings_dir(root)
    os.makedirs(d, exist_ok=True)
    f = Finding(
        id=_finding_id(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        title=title,
        body=body,
        tags=list(tags or []),
        cites=list(cites or []),
        schema_ref=schema_ref,
        snapshot_id=snapshot_id,
    )
    path = os.path.join(d, f"{f.id}.json")
    with open(path, "w") as fp:
        json.dump(f.to_dict(), fp, indent=2)
    return f


def list_all(root: str, *, tag: str | None = None,
             limit: int | None = None) -> list[Finding]:
    """Enumerate findings, optionally tag-filtered, newest first."""
    d = _findings_dir(root)
    if not os.path.isdir(d):
        return []
    files = sorted(
        (f for f in os.listdir(d) if f.endswith(".json")),
        reverse=True,
    )
    out: list[Finding] = []
    for name in files:
        with open(os.path.join(d, name)) as fp:
            data = json.load(fp)
        if tag and tag not in data.get("tags", []):
            continue
        out.append(Finding(**data))
        if limit and len(out) >= limit:
            break
    return out


def get(root: str, finding_id: str) -> Finding:
    """Load one finding by id."""
    path = os.path.join(_findings_dir(root), f"{finding_id}.json")
    with open(path) as fp:
        return Finding(**json.load(fp))
