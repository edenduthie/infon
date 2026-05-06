"""InfonStore — the one-verb API over cassettes.

Design goals (from demo_customer.py UX — this is what users expect):
  • One object. Create, ingest, query. Don't juggle managers.
  • Local-first. `InfonStore("./data/auto")` works immediately.
  • S3-ready. `InfonStore("s3://bucket/prefix")` — same API, no code changes.
  • Idempotent ingest. Content-addressed cassettes mean re-running is free.
  • Schema-as-tag. Each cassette records the schema it was extracted under;
    swapping ontologies is an additive op, never destructive.

Not in scope (yet):
  • Rich structural analysis (Kano, conjoint, Kan extension) — those live
    in cognition.structural and operate on in-memory infons, not the
    cassette store. The bridge is `store.read_all_infons()` → pass to
    StructuralAnalyzer.
  • Remote extraction (Lambda). The executor seam is ready; LambdaExecutor
    lands once the layer packaging is done.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import CognitionConfig
from ..encoder import Encoder
from ..extract import extract_infons
from ..atom import Infon
from ..schema import AnchorSchema

from .format import CassetteWriter, Footer
from .index import Manifest, build_indexes, _joinpath, _is_remote, _ensure_dir
from .dsl import (
    Query, trajectory_hits, next_edges, constraint,
    NextEdge, Constraint,
)
from .reader import LocalFetcher, RangeFetcher, FsspecFetcher, hydrate_locs
from .reason import reason as _reason, Verdict
from .reason_path import reason_connectivity, reason_any_target
from .executor import Executor, SyncExecutor, Result
from .diagnostic import ExtractionReport, compute_report
from . import findings as _findings
from .findings import Finding
from .migrate import (
    SchemaFunctor, MigrationReport, plan_migration, migrate_store,
)


# ═══════════════════════════════════════════════════════════════════════
# INGEST WORKER (module-level, picklable — required for ProcessExecutor)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IngestJob:
    """One doc → one cassette. Self-contained, picklable."""
    doc: dict                  # {id, text, timestamp?}
    schema_path: str           # absolute path to schema JSON
    schema_ref: str            # hash/version identifier
    out_root: str              # cassettes/ + index/ live under here
    cassette_id: str           # content-addressed; predetermined


@dataclass
class IngestBatch:
    """N docs → N cassettes, amortizes SPLADE load across jobs."""
    jobs: list[IngestJob]


def _ingest_batch(batch: IngestBatch) -> list[dict]:
    """Worker: run a batch of jobs so SPLADE loads once per worker.

    Returned payload matches the per-job schema — one summary dict per
    input job — so the driver code treats batched results the same way.

    Cache schema + encoder as worker globals so consecutive map() calls
    within the same worker don't pay the ~1s cold start twice. Do NOT
    import torch at module level: ProcessExecutor.spawn would force the
    driver to pay too.
    """
    global _WORKER_ENC, _WORKER_SCHEMA, _WORKER_SCHEMA_PATH
    try:
        _WORKER_ENC
    except NameError:
        _WORKER_ENC = None
        _WORKER_SCHEMA = None
        _WORKER_SCHEMA_PATH = None

    if not batch.jobs:
        return []

    schema_path = batch.jobs[0].schema_path
    if _WORKER_SCHEMA_PATH != schema_path:
        _WORKER_SCHEMA = AnchorSchema.from_file(schema_path)
        _WORKER_SCHEMA_PATH = schema_path
        _WORKER_ENC = None

    if _WORKER_ENC is None:
        _WORKER_ENC = Encoder(schema=_WORKER_SCHEMA)

    config = CognitionConfig(schema_path=schema_path)

    out = []
    remote = "://" in batch.jobs[0].out_root if batch.jobs else False
    for job in batch.jobs:
        infons, _edges = extract_infons(
            [job.doc], _WORKER_ENC, _WORKER_SCHEMA, config,
        )
        cass_dir = _joinpath(job.out_root, "cassettes")
        _ensure_dir(cass_dir)
        cass_path = _joinpath(cass_dir, f"{job.cassette_id}.inf")

        if remote:
            import fsspec
            with fsspec.open(cass_path, "wb") as f:
                w = CassetteWriter(f, cassette_id=job.cassette_id,
                                    schema_ref=job.schema_ref)
                for inf in infons:
                    w.add(inf)
                footer = w.close()
        else:
            with open(cass_path, "wb") as f:
                w = CassetteWriter(f, cassette_id=job.cassette_id,
                                    schema_ref=job.schema_ref)
                for inf in infons:
                    w.add(inf)
                footer = w.close()

        index_paths = build_indexes(footer,
                                     _joinpath(job.out_root, "index"))
        out.append({
            "cassette_id": job.cassette_id,
            "cassette_path": cass_path,
            "footer_json": footer.to_json().decode(),
            "index_paths": index_paths,
            "n_infons": footer.n_records,
            "doc_id": job.doc.get("id", ""),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════
# INFONSTORE
# ═══════════════════════════════════════════════════════════════════════

class InfonStore:
    """One handle, one root URI, everything else is schema + docs.

        store = InfonStore("./data/auto")
        store.set_schema("schemas/auto_v1.json")

        store.ingest(documents)                    # list of {id, text, timestamp?}
        v = store.ask(Query().where(subject="toyota", predicate="invest",
                                     object="solid_state"))
        v = store.connect("toyota", "catl")
        vs = store.any_of("toyota", {"catl", "lg", "samsung_sdi"})

        store.snapshots()                          # time-travel list
        store.at(snap_id).ask(...)                 # read-only past view

        store.set_schema("schemas/auto_v2.json")   # new ontology
        store.reingest()                           # re-extract all known docs

    Local-first: URI without a scheme → local filesystem. S3 via
    "s3://bucket/prefix"; requires s3fs installed.
    """

    def __init__(self, root: str, schema_path: str | None = None):
        self.root = root
        self.schema_path: str | None = None
        self.schema_ref: str = ""
        self._schema: AnchorSchema | None = None
        self._manifest: Manifest | None = None
        self._at_snapshot: str | None = None  # None = HEAD
        self._fetcher: RangeFetcher | None = None
        # Cached encoder for preview (keeps SPLADE in-process across
        # successive preview calls — ~100ms per doc, not 2s).
        self._preview_encoder: Encoder | None = None
        self._preview_encoder_schema_path: str | None = None
        self._ensure_root()
        if schema_path:
            self.set_schema(schema_path)

    # ── scheme-aware path plumbing ──────────────────────────────────────
    def _ensure_root(self):
        if _is_remote(self.root):
            # Object stores don't need explicit mkdir; prefixes are implicit.
            return
        Path(self.root).mkdir(parents=True, exist_ok=True)
        Path(self.root, "docs").mkdir(exist_ok=True)

    # ── schema ──────────────────────────────────────────────────────────
    def set_schema(self, path: str):
        """Point the store at a schema file. Tags future cassettes.

        Changing the schema does NOT invalidate prior cassettes — they
        remain queryable under their original schema_ref. New ingest calls
        produce cassettes tagged with the new schema."""
        self.schema_path = os.path.abspath(path)
        self._schema = AnchorSchema.from_file(self.schema_path)
        # schema_ref = 12-char content hash so cassette IDs stay stable
        # across machines for the same JSON content.
        with open(self.schema_path, "rb") as f:
            self.schema_ref = hashlib.sha256(f.read()).hexdigest()[:12]
        # Invalidate the preview encoder if it was tied to a different schema.
        if self._preview_encoder_schema_path != self.schema_path:
            self._preview_encoder = None
            self._preview_encoder_schema_path = None

    def set_schema_from_dict(self, schema: dict) -> str:
        """Write a schema dict to disk under the store root and activate it.

        Returns the path written. This is the ergonomic entry point for
        agents and REPL users who want to propose a schema without
        managing temp files themselves."""
        schemas_dir = os.path.join(
            self.root if not _is_remote(self.root) else "/tmp",
            "_schemas",
        )
        os.makedirs(schemas_dir, exist_ok=True)
        # Version-tag the filename by content hash so repeated calls
        # with distinct schemas don't collide.
        body = json.dumps(schema, sort_keys=True).encode()
        sha = hashlib.sha256(body).hexdigest()[:12]
        path = os.path.join(schemas_dir, f"schema_{sha}.json")
        with open(path, "wb") as f:
            f.write(body)
        self.set_schema(path)
        return path

    @property
    def schema(self) -> AnchorSchema:
        if self._schema is None:
            raise RuntimeError("No schema set. Call store.set_schema(path) first.")
        return self._schema

    # ── manifest (current view) ─────────────────────────────────────────
    @property
    def manifest(self) -> Manifest:
        """Lazy-loaded manifest. Refreshes whenever HEAD moves or the
        caller pinned a specific snapshot via .at()."""
        if self._manifest is None:
            if self._at_snapshot:
                self._manifest = Manifest.load_at(self.root, self._at_snapshot)
            else:
                head_file = os.path.join(self.root, "_manifest", "HEAD")
                if os.path.exists(head_file):
                    self._manifest = Manifest.load_head(self.root)
                else:
                    # No snapshots yet. Return an IN-MEMORY empty manifest;
                    # don't persist it (else we'd create a spurious empty
                    # snapshot before any ingest happens).
                    self._manifest = Manifest.new(self.root)
        return self._manifest

    def _refresh_manifest(self):
        """Called after ingest to pick up the new HEAD."""
        self._manifest = None

    def snapshots(self) -> list[str]:
        """All known snapshots, oldest first."""
        return Manifest.list_snapshots(self.root)

    def at(self, snapshot_id: str) -> "InfonStore":
        """Return a read-only view pinned to a specific snapshot.

        Returns a copy of the store with its manifest fixed. Useful for
        audit queries: `store.at(yesterday).ask(...)`."""
        view = InfonStore.__new__(InfonStore)
        view.root = self.root
        view.schema_path = self.schema_path
        view.schema_ref = self.schema_ref
        view._schema = self._schema
        view._manifest = None
        view._at_snapshot = snapshot_id
        view._fetcher = None
        return view

    # ── docs registry (content-addressed) ───────────────────────────────
    def _doc_hash(self, doc: dict) -> str:
        """Stable hash over (text, schema_ref). Used as cassette_id so
        re-ingesting the same doc under the same schema is a no-op."""
        h = hashlib.sha256()
        h.update(doc["text"].encode())
        h.update(self.schema_ref.encode())
        return h.hexdigest()[:16]

    def _registry_path(self, doc_hash: str) -> str:
        return _joinpath(self.root, "docs", f"{doc_hash}.json")

    def _is_ingested(self, doc_hash: str) -> bool:
        p = self._registry_path(doc_hash)
        if _is_remote(p):
            import fsspec
            fs, rel = fsspec.core.url_to_fs(p)
            return fs.exists(rel)
        return os.path.exists(p)

    def _record_doc(self, doc: dict, doc_hash: str, cassette_id: str):
        """Persist doc metadata so reingest() knows what we've seen."""
        payload = json.dumps({
            "doc_id": doc.get("id", ""),
            "cassette_id": cassette_id,
            "schema_ref": self.schema_ref,
            "timestamp": doc.get("timestamp"),
            "text": doc["text"],
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        })
        p = self._registry_path(doc_hash)
        if _is_remote(p):
            import fsspec
            with fsspec.open(p, "w") as f:
                f.write(payload)
        else:
            with open(p, "w") as f:
                f.write(payload)

    def known_docs(self) -> list[dict]:
        """All docs the store has ever ingested (any schema)."""
        out = []
        docs_dir = _joinpath(self.root, "docs")
        if _is_remote(docs_dir):
            import fsspec
            fs, rel = fsspec.core.url_to_fs(docs_dir)
            try:
                names = fs.ls(rel, detail=False)
            except FileNotFoundError:
                return out
            for name in sorted(names):
                if not name.endswith(".json"):
                    continue
                full = _joinpath(docs_dir, os.path.basename(name))
                with fsspec.open(full, "r") as f:
                    out.append(json.load(f))
            return out
        if not os.path.isdir(docs_dir):
            return out
        for name in sorted(os.listdir(docs_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(docs_dir, name)) as f:
                out.append(json.load(f))
        return out

    # ── ingest ──────────────────────────────────────────────────────────
    def ingest(
        self,
        documents: Iterable[dict],
        *,
        executor: Executor | None = None,
        skip_ingested: bool = True,
    ) -> dict:
        """Extract each document's infons and commit one manifest snapshot.

        Args:
          documents: iterable of {"id", "text", "timestamp"?} dicts.
          executor: SyncExecutor (default), ProcessExecutor(workers=N),
            or any Executor. Fan-out changes parallelism, not results.
          skip_ingested: if True (default), docs already seen under the
            current schema are skipped. Set False to force re-extraction.

        Returns:
          {
            "ingested": [cassette_id, ...],   # newly written
            "skipped":  [cassette_id, ...],   # already existed
            "errors":   [(doc_id, error), ...],
            "snapshot_id": "<new_snapshot>",
            "n_infons": int,
          }

        Ingest is idempotent: cassette IDs are content-hashed over
        (doc.text, schema_ref), so re-running with the same docs + schema
        is free.
        """
        if self.schema_path is None:
            raise RuntimeError(
                "No schema set. Call store.set_schema(path) before ingest."
            )

        executor = executor or SyncExecutor()
        docs = list(documents)

        # Plan: what's new vs. what's already there.
        jobs: list[IngestJob] = []
        skipped: list[str] = []
        for doc in docs:
            doc_hash = self._doc_hash(doc)
            if skip_ingested and self._is_ingested(doc_hash):
                skipped.append(doc_hash)
                continue
            jobs.append(IngestJob(
                doc=doc,
                schema_path=self.schema_path,
                schema_ref=self.schema_ref,
                out_root=self.root,
                cassette_id=doc_hash,
            ))

        if not jobs:
            return {
                "ingested": [],
                "skipped": skipped,
                "errors": [],
                "snapshot_id": self.manifest.snapshot_id,
                "n_infons": 0,
            }

        # Fan-out. Split jobs into one batch per worker so SPLADE cold-
        # start amortizes across the batch instead of paying per doc.
        n_workers = getattr(executor, "workers", 1)
        n_batches = max(1, min(n_workers, len(jobs)))
        batch_size = (len(jobs) + n_batches - 1) // n_batches
        batches = [
            IngestBatch(jobs=jobs[i:i + batch_size])
            for i in range(0, len(jobs), batch_size)
        ]
        results: list[Result] = executor.map(_ingest_batch, batches)

        # Collect footers and commit ONE manifest snapshot for the whole batch.
        parent = self.manifest if self.manifest.cassettes else None
        new_manifest = Manifest.new(self.root, parent=parent)
        if parent is not None:
            new_manifest.cassettes = list(parent.cassettes)
            for k in new_manifest.indexes:
                new_manifest.indexes[k] = list(parent.indexes.get(k, []))

        ingested: list[str] = []
        errors: list[tuple[str, str]] = []
        total_new_infons = 0
        # Flatten batch results back to individual job summaries.
        flat_summaries: list[tuple[IngestJob, dict | None, str | None]] = []
        for batch_idx, res in enumerate(results):
            batch_jobs = batches[batch_idx].jobs
            if not res.ok:
                for j in batch_jobs:
                    flat_summaries.append((j, None, res.error))
            else:
                summaries = res.value  # list[dict]
                for j, s in zip(batch_jobs, summaries):
                    flat_summaries.append((j, s, None))

        for job, summary, err in flat_summaries:
            if err is not None:
                errors.append((job.doc.get("id", ""), err))
                continue
            footer = Footer.from_json(summary["footer_json"].encode())
            new_manifest.add_cassette(
                footer, summary["cassette_path"], summary["index_paths"],
            )
            self._record_doc(job.doc, job.cassette_id, summary["cassette_id"])
            ingested.append(summary["cassette_id"])
            total_new_infons += summary["n_infons"]

        if ingested:
            new_manifest.save()
            self._refresh_manifest()
            snap_id = new_manifest.snapshot_id
        else:
            snap_id = self.manifest.snapshot_id

        # Always attach a fresh extraction report so the user sees
        # coverage issues without a second call. Skip on empty results —
        # no infons means no meaningful stats.
        report: ExtractionReport | None = None
        if total_new_infons > 0 or ingested:
            try:
                report = self.extraction_report()
            except Exception:
                # Never let a diagnostic failure abort ingest; just skip.
                report = None

        return {
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
            "snapshot_id": snap_id,
            "n_infons": total_new_infons,
            "report": report,
        }

    # ── preview (no-write extraction) ───────────────────────────────────
    def preview(self, documents: Iterable[dict],
                *, schema_path: str | None = None) -> list[Infon]:
        """Extract infons from docs WITHOUT writing cassettes or manifest.

        Use for iterative schema tuning:
          infons = store.preview([{"id":"d1","text":"..."}])
          # inspect triples → edit schema → preview again

        The encoder is cached on the store, so successive calls with
        the same schema reuse SPLADE (~100ms per doc after first call).

        Args:
          documents: list of {id, text, timestamp?} dicts.
          schema_path: optional — test a candidate schema without
            changing the store's active schema. Defaults to the active
            schema set via set_schema().

        Returns:
          list[Infon] — extracted infons across all docs. NOT persisted.
          Edges are dropped; use the full ingest pipeline for those.
        """
        sp = schema_path or self.schema_path
        if sp is None:
            raise RuntimeError(
                "No schema provided. Pass schema_path or call set_schema() first."
            )
        sp = os.path.abspath(sp)

        # Reuse cached encoder if schema didn't change.
        if self._preview_encoder is None or self._preview_encoder_schema_path != sp:
            schema = AnchorSchema.from_file(sp)
            self._preview_encoder = Encoder(schema=schema)
            self._preview_encoder_schema_path = sp
        else:
            schema = self._preview_encoder.schema if hasattr(
                self._preview_encoder, "schema") else AnchorSchema.from_file(sp)

        config = CognitionConfig(schema_path=sp)
        docs = list(documents)
        infons, _edges = extract_infons(docs, self._preview_encoder,
                                         schema, config)
        return infons

    def reingest(self, *, executor: Executor | None = None) -> dict:
        """Re-extract every known doc under the current schema.

        Use this after `set_schema()` to migrate the corpus to a new
        ontology without losing the originals (old cassettes stay;
        new ones appear tagged with the new schema_ref)."""
        docs = [{"id": d["doc_id"], "text": d["text"],
                 "timestamp": d.get("timestamp")}
                for d in self.known_docs()]
        # Force re-extraction: the new schema_ref will make cassette IDs
        # different from the old ones, so skip_ingested naturally won't
        # short-circuit. Leave it on.
        return self.ingest(docs, executor=executor)

    # ── query / reason ──────────────────────────────────────────────────
    def search(self, query: Query) -> list:
        """Raw index hits — no hydration. Cheap; use for aggregates."""
        return query.run(self.manifest)

    def ask(self, claim: Query, **kw) -> Verdict:
        """Answer a single-claim Query via the calibrated reasoner."""
        return _reason(self.manifest, claim, fetcher=self._get_fetcher(), **kw)

    def connect(self, source: str, target: str, **kw) -> Verdict:
        """Is `source` connected to `target` via a chain of relations?"""
        return reason_connectivity(self.manifest, source, target,
                                    fetcher=self._get_fetcher(), **kw)

    def any_of(self, source: str, targets: set[str], **kw) -> dict:
        """Which of `targets` is `source` connected to? One tree walk."""
        return reason_any_target(self.manifest, source, targets,
                                  fetcher=self._get_fetcher(), **kw)

    def hydrate(self, hits) -> list[Infon]:
        """Range-fetch infons for a list of Hit objects."""
        return hydrate_locs(self._get_fetcher(), self.manifest, hits)

    # ── module 6: trajectory + constraint (query-time NEXT) ─────────────
    def trajectory(self, anchor: str, *, hydrate: bool = True
                   ) -> list[Infon] | list:
        """Time-ordered sequence of infons mentioning `anchor`.

        Consecutive pairs are NEXT edges — we derive them at query time
        from the by_anchor index's timestamp column, so delta ingests
        pick up automatically with no re-indexing. Time-travel views
        (via `at()`) give you the trajectory "as of" that snapshot for
        free.

        Args:
          hydrate: True (default) returns full Infon objects. False
            returns Hits — cheaper when you only need timestamps/triples.
        """
        hits = trajectory_hits(self.manifest, anchor)
        if not hydrate:
            return hits
        return hydrate_locs(self._get_fetcher(), self.manifest, hits)

    def next_edges(self, anchor: str) -> list[NextEdge]:
        """NEXT edges for `anchor`'s trajectory. Pure index read."""
        return next_edges(self.manifest, anchor)

    def constraint(self, subject: str, predicate: str, object: str
                    ) -> Constraint:
        """Corpus-level aggregate for a specific triple — count,
        polarity balance, time span, mean confidence. Replaces the old
        ingest-time consolidation with a read-time query that's always
        consistent with HEAD."""
        return constraint(self.manifest, subject, predicate, object)

    def read_all_infons(self, limit: int | None = None) -> list[Infon]:
        """Full-scan hydrate. For in-memory analysis (StructuralAnalyzer,
        etc.) — bounded by `limit` to avoid pulling huge corpora."""
        hits = Query().run(self.manifest)
        if limit:
            hits = hits[:limit]
        return hydrate_locs(self._get_fetcher(), self.manifest, hits)

    def _get_fetcher(self) -> RangeFetcher:
        if self._fetcher is None:
            # Use fsspec for remote roots so range-gets route through s3fs
            # (or whatever backend matches the scheme). Local stays on
            # plain open() for speed.
            if _is_remote(self.root):
                self._fetcher = FsspecFetcher()
            else:
                self._fetcher = LocalFetcher()
        return self._fetcher

    # ── findings (persistent, cross-session) ────────────────────────────
    def record_finding(self, *, title: str, body: str,
                        tags: list[str] | None = None,
                        cites: list[dict] | None = None) -> Finding:
        """Persist a note or investigation artifact under <root>/findings/.

        Findings capture meta that the store can't recompute: why the user
        cared, what they concluded, schema lessons learned. Carries the
        active schema_ref and snapshot_id automatically so future readers
        know when the note was made."""
        return _findings.record(
            self.root,
            title=title, body=body, tags=tags, cites=cites,
            schema_ref=self.schema_ref,
            snapshot_id=self.manifest.snapshot_id,
        )

    def findings(self, *, tag: str | None = None,
                 limit: int | None = None) -> list[Finding]:
        """List findings, newest first, optionally tag-filtered."""
        return _findings.list_all(self.root, tag=tag, limit=limit)

    def get_finding(self, finding_id: str) -> Finding:
        return _findings.get(self.root, finding_id)

    # ── schema migration (Kan-based, no re-extraction) ─────────────────
    def plan_migration(self, functor: SchemaFunctor,
                        target_schema_path: str) -> MigrationReport:
        """Preview a schema migration without writing anything.

        Shows how many infons will be dropped (due to deletions), how
        many merges collapse, how many orphan anchors will appear in
        the target. User reads this, decides, then calls migrate()."""
        target_schema = AnchorSchema.from_file(target_schema_path)
        # Use read_all_infons for full hydration — migration needs
        # every Infon's metadata.
        infons = self.read_all_infons()
        return plan_migration(infons, functor, self.schema, target_schema)

    def migrate(self, functor: SchemaFunctor,
                target_schema_path: str,
                *, verbose: bool = False) -> MigrationReport:
        """Apply a schema functor to existing cassettes; write migrated
        copies tagged with the new schema_ref; activate the new schema.

        Old cassettes are NEVER rewritten — they remain queryable under
        the old schema by loading a prior snapshot. New cassettes join
        the manifest at HEAD."""
        report, _ = migrate_store(
            self, functor, target_schema_path, verbose=verbose,
        )
        return report

    # ── diagnostics ─────────────────────────────────────────────────────
    def extraction_report(self) -> ExtractionReport:
        """Honest feedback about what extraction captured vs. dropped.

        Reads the by_anchor + by_triple indexes and the docs registry;
        never hydrates infons. Use this right after `ingest()` to spot
        silent failures (schema tokens that didn't match, anchors with
        no signal, objects hoarded by a single feature) before running
        queries and drawing wrong conclusions.

        The same object is returned from `ingest()` as a field, so you
        rarely need to call this directly."""
        return compute_report(
            self.manifest, self.schema, self.known_docs(),
        )

    # ── introspection ───────────────────────────────────────────────────
    def stats(self) -> dict:
        m = self.manifest
        return {
            "root": self.root,
            "schema_ref": self.schema_ref,
            "snapshot_id": m.snapshot_id,
            "cassettes": len(m.cassettes),
            "infons": sum(c["n_records"] for c in m.cassettes),
            "known_docs": len(self.known_docs()),
            "snapshots": len(self.snapshots()),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"InfonStore(root={self.root!r}, cassettes={s['cassettes']}, "
                f"infons={s['infons']}, snapshot={s['snapshot_id'][:12]}...)")
