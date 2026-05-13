"""Split indexes + manifest for cassette-backed stores.

Layout (S3-compatible; works on any fsspec path):

    <root>/cassettes/<cassette_id>.inf
    <root>/index/by_triple/<cassette_id>.parquet    # (s,p,o,polarity,conf,ts, cassette_id, offset, length, infon_id)
    <root>/index/by_time/<cassette_id>.parquet
    <root>/index/by_anchor/<cassette_id>.parquet    # (anchor, role, ...)
    <root>/_manifest/<snapshot>.json                # live cassettes + index shards

Customers push new cassettes + their per-shard index parquets. A new manifest
snapshot that lists the added shards is the only "commit" — existing shards
are never rewritten, so deltas are O(new infons).

Queries return list[Hit] where each Hit pairs cassette_id with a RecordLoc,
so the reader knows which cassette to range-fetch.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field

import pyarrow as pa
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════
# PATH UTILITIES  (local + fsspec URIs transparently)
# ═══════════════════════════════════════════════════════════════════════

def _is_remote(path: str) -> bool:
    return "://" in path


def _joinpath(*parts: str) -> str:
    """Join paths that may be s3:// URIs or local. os.path.join mangles
    s3:// schemes on POSIX (strips the double slash); we normalize."""
    if not parts:
        return ""
    base = parts[0]
    if _is_remote(base):
        # Collapse any leading trailing slashes; join with "/".
        return "/".join(p.rstrip("/") for p in parts)
    return os.path.join(*parts)


def _ensure_dir(path: str) -> None:
    """mkdir -p that's a no-op for remote URIs (object stores don't need it)."""
    if _is_remote(path):
        return
    os.makedirs(path, exist_ok=True)

from .format import Footer, RecordLoc


INDEX_KINDS = ("by_triple", "by_time", "by_anchor")


@dataclass
class Hit:
    """Index hit: cassette_id + RecordLoc for range-fetch."""
    cassette_id: str
    loc: RecordLoc


def _loc_row(f: Footer, loc: RecordLoc) -> dict:
    return {
        "cassette_id": f.cassette_id,
        "infon_id": loc.infon_id,
        "offset": loc.offset,
        "length": loc.length,
    }


def _triple_table(footer: Footer) -> pa.Table:
    rows = [{**_loc_row(footer, r),
             "subject": r.subject, "predicate": r.predicate, "object": r.object,
             "polarity": r.polarity, "confidence": r.confidence,
             "timestamp": r.timestamp or ""}
            for r in footer.records]
    return pa.Table.from_pylist(rows)


def _time_table(footer: Footer) -> pa.Table:
    rows = [{**_loc_row(footer, r), "timestamp": r.timestamp or ""}
            for r in footer.records if r.timestamp]
    return pa.Table.from_pylist(rows)


def _anchor_table(footer: Footer) -> pa.Table:
    rows = []
    for r in footer.records:
        base = {**_loc_row(footer, r),
                "subject": r.subject, "predicate": r.predicate, "object": r.object,
                "confidence": r.confidence, "timestamp": r.timestamp or ""}
        rows.append({**base, "anchor": r.subject, "role": "subject"})
        rows.append({**base, "anchor": r.predicate, "role": "predicate"})
        rows.append({**base, "anchor": r.object, "role": "object"})
    return pa.Table.from_pylist(rows)


def build_indexes(footer: Footer, index_root: str) -> dict[str, str]:
    """Write the three per-cassette index parquets. Returns kind→path.

    index_root can be a local path OR an fsspec URI ("s3://bucket/idx").
    pyarrow.parquet.write_table already speaks fsspec — we just need
    path handling that doesn't mangle the scheme.
    """
    out: dict[str, str] = {}
    remote = _is_remote(index_root)
    for kind, builder in (("by_triple", _triple_table),
                          ("by_time", _time_table),
                          ("by_anchor", _anchor_table)):
        d = _joinpath(index_root, kind)
        _ensure_dir(d)
        path = _joinpath(d, f"{footer.cassette_id}.parquet")
        tbl = builder(footer)
        if tbl.num_rows > 0:
            if remote:
                # pyarrow takes fsspec URIs directly through filesystem arg.
                import fsspec
                fs, rel = fsspec.core.url_to_fs(path)
                pq.write_table(tbl, rel, filesystem=fs, compression="zstd")
            else:
                pq.write_table(tbl, path, compression="zstd")
            out[kind] = path
    return out


# ═══════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Manifest:
    """Snapshot of live cassettes + index shards. Iceberg-style pointer."""
    snapshot_id: str = ""
    created_at: str = ""
    root: str = ""
    cassettes: list[dict] = field(default_factory=list)
    indexes: dict[str, list[str]] = field(default_factory=dict)
    parent_snapshot: str | None = None

    @classmethod
    def new(cls, root: str, parent: "Manifest | None" = None) -> "Manifest":
        return cls(
            snapshot_id=time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"_{uuid.uuid4().hex[:6]}",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            root=root,
            parent_snapshot=parent.snapshot_id if parent else None,
            cassettes=list(parent.cassettes) if parent else [],
            indexes={k: list(parent.indexes.get(k, [])) for k in INDEX_KINDS}
                    if parent else {k: [] for k in INDEX_KINDS},
        )

    def add_cassette(self, footer: Footer, cassette_path: str, index_paths: dict[str, str]):
        # Store the full anchor set so the manifest-level pruner can skip
        # cassettes that can't possibly contain a target anchor. Subjects/
        # predicates/objects are small (footer already dedupes them), so
        # the cost is negligible vs. the savings at query time.
        anchors = sorted(set(footer.subjects) | set(footer.predicates) | set(footer.objects))
        self.cassettes.append({
            "cassette_id": footer.cassette_id,
            "path": cassette_path,
            "n_records": footer.n_records,
            "t_min": footer.t_min,
            "t_max": footer.t_max,
            "sha256_body": footer.sha256_body,
            "anchors": anchors,
            "subjects": footer.subjects,
            "predicates": footer.predicates,
            "objects": footer.objects,
            "index_paths": index_paths,  # per-cassette paths for pruning
        })
        for kind, p in index_paths.items():
            self.indexes.setdefault(kind, []).append(p)

    def cassette_path(self, cassette_id: str) -> str:
        for c in self.cassettes:
            if c["cassette_id"] == cassette_id:
                return c["path"]
        raise KeyError(cassette_id)

    def save(self) -> str:
        d = _joinpath(self.root, "_manifest")
        _ensure_dir(d)
        path = _joinpath(d, f"{self.snapshot_id}.json")
        head_path = _joinpath(d, "HEAD")
        payload = json.dumps(asdict(self), indent=2)

        if _is_remote(self.root):
            import fsspec
            with fsspec.open(path, "w") as f:
                f.write(payload)
            with fsspec.open(head_path, "w") as f:
                f.write(self.snapshot_id)
        else:
            with open(path, "w") as f:
                f.write(payload)
            with open(head_path, "w") as f:
                f.write(self.snapshot_id)
        return path

    @classmethod
    def load_head(cls, root: str) -> "Manifest":
        head_path = _joinpath(root, "_manifest", "HEAD")
        if _is_remote(root):
            import fsspec
            with fsspec.open(head_path, "r") as f:
                snap = f.read().strip()
        else:
            with open(head_path) as f:
                snap = f.read().strip()
        return cls.load_at(root, snap)

    @classmethod
    def load_at(cls, root: str, snapshot_id: str) -> "Manifest":
        """Load a specific snapshot. Enables time-travel / audit queries —
        answer a question using only data that was visible at snapshot S.
        """
        path = _joinpath(root, "_manifest", f"{snapshot_id}.json")
        if _is_remote(root):
            import fsspec
            with fsspec.open(path, "r") as f:
                return cls(**json.load(f))
        else:
            with open(path) as f:
                return cls(**json.load(f))

    @classmethod
    def list_snapshots(cls, root: str) -> list[str]:
        """All known snapshots, newest last (lexical sort matches creation
        order since snapshot_id starts with YYYYMMDDTHHMMSS)."""
        d = _joinpath(root, "_manifest")
        if _is_remote(root):
            import fsspec
            fs, rel = fsspec.core.url_to_fs(d)
            try:
                names = fs.ls(rel, detail=False)
            except FileNotFoundError:
                return []
            return sorted(
                os.path.basename(n)[:-5]
                for n in names if n.endswith(".json")
            )
        if not os.path.isdir(d):
            return []
        return sorted(
            f[:-5] for f in os.listdir(d) if f.endswith(".json")
        )


# ═══════════════════════════════════════════════════════════════════════
# QUERIES  (predicate pushdown → Hit list)
# ═══════════════════════════════════════════════════════════════════════

def _scan(paths: list[str], filters) -> pa.Table:
    if not paths:
        return pa.table({})
    # If any path is remote, route through fsspec filesystem. Mixed
    # local+remote lists aren't supported; we assume the manifest is
    # internally consistent (all shards under one root/scheme).
    if _is_remote(paths[0]):
        import fsspec
        fs, _ = fsspec.core.url_to_fs(paths[0])
        # Strip the scheme prefix for pyarrow's filesystem arg.
        rel_paths = [fsspec.core.url_to_fs(p)[1] for p in paths]
        return pq.ParquetDataset(rel_paths, filesystem=fs,
                                  filters=filters).read()
    return pq.ParquetDataset(paths, filters=filters).read()


# ═══════════════════════════════════════════════════════════════════════
# MANIFEST-LEVEL PRUNING
# ═══════════════════════════════════════════════════════════════════════
# Skip cassettes that can't match before opening any parquet. For 100+
# cassettes this is the main win — parquet-dataset open cost scales with
# shard count, so cutting the shard list dominates index-scan wall time.


def _prune_by_anchors(manifest: Manifest, kind: str,
                      required_anchors: set[str],
                      roles: tuple[str, ...] = ("subjects", "predicates", "objects")
                      ) -> list[str]:
    """Return index-shard paths for cassettes whose declared anchor sets
    intersect `required_anchors`. `roles` restricts which footer fields
    count (e.g. just "subjects" for a subject-only filter)."""
    out = []
    for c in manifest.cassettes:
        hit = False
        for role in roles:
            if set(c.get(role, ())) & required_anchors:
                hit = True
                break
        if not hit:
            continue
        ip = c.get("index_paths", {})
        if kind in ip:
            out.append(ip[kind])
    return out


def _prune_by_time(manifest: Manifest, kind: str,
                   t_start: str, t_end: str) -> list[str]:
    out = []
    for c in manifest.cassettes:
        t_min, t_max = c.get("t_min"), c.get("t_max")
        if t_min is None or t_max is None:
            continue
        if t_max < t_start or t_min > t_end:
            continue
        ip = c.get("index_paths", {})
        if kind in ip:
            out.append(ip[kind])
    return out


def _all_paths(manifest: Manifest, kind: str) -> list[str]:
    return list(manifest.indexes.get(kind, []))


def _tbl_to_hits(tbl: pa.Table) -> list[Hit]:
    if tbl.num_rows == 0:
        return []
    d = tbl.to_pydict()
    n = tbl.num_rows
    out: list[Hit] = []
    for i in range(n):
        out.append(Hit(
            cassette_id=d["cassette_id"][i],
            loc=RecordLoc(
                infon_id=d["infon_id"][i],
                offset=int(d["offset"][i]),
                length=int(d["length"][i]),
                subject=d.get("subject", [""] * n)[i] or "",
                predicate=d.get("predicate", [""] * n)[i] or "",
                object=d.get("object", [""] * n)[i] or "",
                polarity=int(d.get("polarity", [1] * n)[i]) if "polarity" in d else 1,
                confidence=float(d.get("confidence", [0.0] * n)[i]) if "confidence" in d else 0.0,
                timestamp=d.get("timestamp", [None] * n)[i] or None,
            )
        ))
    return out


def query_triple(manifest: Manifest, subject: str | None = None,
                 predicate: str | None = None, object: str | None = None,
                 min_confidence: float = 0.0) -> list[Hit]:
    filters = []
    if subject: filters.append(("subject", "=", subject))
    if predicate: filters.append(("predicate", "=", predicate))
    if object: filters.append(("object", "=", object))
    if min_confidence > 0: filters.append(("confidence", ">=", min_confidence))

    # Prune by whichever role is pinned; if none, fall back to all shards.
    if subject:
        paths = _prune_by_anchors(manifest, "by_triple", {subject}, ("subjects",))
    elif predicate:
        paths = _prune_by_anchors(manifest, "by_triple", {predicate}, ("predicates",))
    elif object:
        paths = _prune_by_anchors(manifest, "by_triple", {object}, ("objects",))
    else:
        paths = _all_paths(manifest, "by_triple")
    return _tbl_to_hits(_scan(paths, filters or None))


def query_time_range(manifest: Manifest, t_start: str, t_end: str) -> list[Hit]:
    paths = _prune_by_time(manifest, "by_time", t_start, t_end)
    filters = [("timestamp", ">=", t_start), ("timestamp", "<=", t_end)]
    return _tbl_to_hits(_scan(paths, filters))


def query_anchor(manifest: Manifest, anchor: str, role: str | None = None) -> list[Hit]:
    # Role-agnostic by default → check all three role sets.
    roles = ("subjects", "predicates", "objects")
    if role == "subject": roles = ("subjects",)
    elif role == "predicate": roles = ("predicates",)
    elif role == "object": roles = ("objects",)
    paths = _prune_by_anchors(manifest, "by_anchor", {anchor}, roles)
    filters = [("anchor", "=", anchor)]
    if role:
        filters.append(("role", "=", role))
    return _tbl_to_hits(_scan(paths, filters))
