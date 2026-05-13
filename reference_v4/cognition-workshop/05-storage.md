# Module 05: The Cassette Substrate

## What You'll Learn

- Why an immutable, content-addressed cassette format beats a mutable relational store for knowledge-graph workloads
- How split Parquet indexes (by_triple / by_time / by_anchor) replace SQL queries with zero-hydration filters
- What the manifest pruner is and why it gives 7–16× query speedup at 300 cassettes
- How time-travel works for free when the storage layer is append-only
- Why the same code runs against a local filesystem or S3 with a one-character URI change

## Background

Earlier versions of this workshop taught a mutable SQLite schema with importance decay and soft-delete. That design fought every real workload: delta ingest rewrote edge tables, audit queries required keeping pruned rows around, and porting to the cloud meant a parallel DynamoDB backend that duplicated every abstraction. The cassette substrate replaces all of this with three immutable ideas:

1. **Cassettes are content-addressed byte blobs.** Each ingest batch produces one `.inf` file named by `sha256(text + schema_ref)`. Re-ingesting the same documents is a no-op because the hash matches. Deletes are not part of the model.

2. **Indexes are per-cassette Parquet shards.** When you write a cassette, you also write three small Parquet files — one each for triple, time, and anchor lookups. No global index to update, no locks, no rebuild step.

3. **The manifest is an append-only pointer chain.** Every ingest creates a new `_manifest/<snapshot>.json` that lists the cassettes visible at that moment. `HEAD` is one tiny file. Time-travel is just reading an older manifest.

The result: the same Python API runs against a local filesystem or `s3://` with no backend-specific code path. Everything routes through `fsspec`. And because nothing is ever rewritten, delta ingest on S3 is actually fast (no read-modify-write cycles to fight the object store's semantics).

## Prompt 1: Build the Cassette Format

---

```
Create the on-disk cassette format in cognition/src/cognition/cassette/format.py.

A cassette is a single binary file with this layout:

  [MAGIC 8B "INFC0001"]
  [HEADER_LEN u32 LE][HEADER JSON]              # creator, schema_ref, created_at
  [ RECORD: [LEN u32 LE][GZIP(JSON infon)] ]*   # addressable frames
  [FOOTER_LEN u32 LE][FOOTER JSON]              # records[]: offset, len, triple, conf, ts
  [FOOTER_OFFSET u64 LE][MAGIC 8B]

Two classes:

1. CassetteWriter(fp, cassette_id, schema_ref)
   - add(infon) appends one gzip frame, tracks (offset, length) in footer
   - close() writes the footer + trailer, returns the Footer
   - Each record is independently gzip-compressed so range-get reads
     work without decompressing the whole file.

2. CassetteReader(fp)
   - Bootstraps by reading the last 16 bytes (trailer) — one range GET
     tells us where the footer lives.
   - read_at(offset, length) decompresses one record.
   - iter_records() walks the footer and emits Infon objects.

The footer is JSON because it's small (O(n_records), not O(bytes)) and
human-inspectable. If the footer format changes later, the magic field
carries a version code so readers can tell.

Both read and write paths should work unchanged for fsspec-addressable
paths — so `CassetteReader(fsspec.open("s3://...", "rb").open())`
does the right thing with s3fs installed.
```

---

## Prompt 2: Split Indexes + Manifest

---

```
Create cognition/src/cognition/cassette/index.py.

When a cassette is written, we also write three per-cassette Parquet
shards under <root>/index/:

  by_triple/<cassette_id>.parquet    # (s, p, o, polarity, conf, ts,
                                     #  cassette_id, offset, length, infon_id)
  by_time/<cassette_id>.parquet      # (ts, cassette_id, offset, length, infon_id)
  by_anchor/<cassette_id>.parquet    # (anchor, role, cassette_id, offset, length)

Three views of the same data. Each query shape uses the narrowest one:

  Query().where(subject=X)            → by_triple (filter on subject)
  Query().mentioning(X)               → by_anchor (anchor=X, any role)
  Query().between(t1, t2)             → by_time (range filter)

The manifest is <root>/_manifest/<snapshot>.json, with an append-only
parent chain:

  snapshot_id → parent_snapshot_id → grandparent → ...

HEAD is a one-line text file under _manifest/ pointing at the current
snapshot. Every ingest creates a new snapshot whose cassette list is
its parent's PLUS the newly-written cassettes. Time-travel to a prior
snapshot is just reading that older JSON.

Manifest.cassettes is a list of dicts, each carrying the per-cassette
summary from the footer: subjects/predicates/objects sets, time range,
record count. This is what powers the pruner.
```

---

## Prompt 3: The Pruner

---

```
Add to cognition/src/cognition/cassette/index.py:

_prune_by_anchors(manifest, kind, anchors, roles)
_prune_by_time(manifest, kind, t_start, t_end)

Before any Parquet scan, ask: which cassettes could possibly contain
this anchor / this time range? Return only the index paths whose
cassette's summary intersects the filter. Skip the others.

On a 300-cassette store:
- Generic top-anchor queries keep ~9 shards (30% speedup).
- Mid-frequency anchors keep ~3 shards (10× speedup).
- Anchors never seen keep 0 shards (278× speedup — literally no work).

The pruner is cheap because cassette summaries are already in memory
(loaded from the manifest JSON). It runs before any remote S3 GET, so
the benefit is both speed and cost.

query_triple / query_time_range / query_anchor should all route through
the pruner. _scan(paths, filters) then opens only the surviving shards
via pyarrow.ParquetDataset — which knows how to read from s3:// too.
```

---

## Prompt 4: Range-Addressable Reader

---

```
Create cognition/src/cognition/cassette/reader.py.

RangeFetcher protocol:
  fetch(path, offset, length) -> bytes
  size(path) -> int

Two implementations:
  LocalFetcher    — open()/seek()/read() on the local FS
  FsspecFetcher  — fsspec.open() + seek/read for any scheme

The reader takes a list of Hit objects (one per matching index row) and
issues one range GET per hit via fetch(). A request counter lets probes
measure the actual S3 call volume.

hydrate_locs(fetcher, manifest, hits) → list[Infon]
  Groups hits by cassette_id, issues parallel range GETs, decompresses
  each record, returns the Infon list in input order.

For small batches (≤10 hits) LocalFetcher is ~1ms/hit; FsspecFetcher
over s3fs is ~30ms/hit (limited by S3 GET latency, not bandwidth). At
100+ hits per query the difference disappears because range GETs can
fan out in parallel.
```

---

## Why This Layout, Not Something Else

### Why per-cassette shards, not one giant index

Global indexes have to be rewritten on every insert. Per-cassette shards are written once and never touched again — the cassette is the shard's unit of content-addressing. The manifest pruner then skips shards that can't contribute, so query cost scales with filter selectivity rather than corpus size.

This is the same pattern Common Crawl uses for its CDX+WARC layout: tiny static shards plus a smart plan to skip most of them.

### Why append-only manifest, not transactional rewrites

The typical "rewrite the index on commit" pattern breaks spectacularly on S3 because object puts aren't transactional and readers may observe torn updates. Append-only dodges this entirely: the newest manifest file is complete before HEAD is updated; older files stay valid forever. Time-travel is a free side-effect.

### Why JSON in the manifest, not Parquet

The manifest is small (a few KB even at 10k cassettes), human-readable, and changes on every ingest. JSON means git can diff it, `cat` works, and there's no schema-evolution story to maintain. Parquet would be premature.

### Why 17 MB SPLADE ships bundled with the package, not in a cassette

The encoder is shared infrastructure, not per-corpus data. It lives in `cognition/model/` inside the Python package so anyone who `pip install`s gets it. Cassettes carry the extracted infons, not the extractor.

## Checkpoint

- [ ] `InfonStore("./data", schema_path="x.json").ingest([{"id":"d1","text":"..."}])` produces cassettes + indexes + manifest under `./data/`.
- [ ] Re-ingesting the same doc produces `{"ingested": [], "skipped": [doc_hash]}` (idempotency).
- [ ] `Query().where(subject="toyota").run(store.manifest)` returns hits with zero hydration.
- [ ] `store.at(prior_snapshot_id)` returns a view whose queries see only the pre-snapshot cassettes.
- [ ] Swapping the root for `s3://bucket/prefix` changes nothing in the caller code.
