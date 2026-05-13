# Module 09: Cloud Deployment

## What You'll Learn

- Why the cassette substrate needs zero backend-specific code to run on S3 — `fsspec` is the single seam
- How the `Executor` protocol separates ingestion fan-out from storage so the same `store.ingest()` call runs on a laptop, a CPU pool, or Lambda
- Why we ship a Lambda container image instead of a zip layer (torch is 1.5GB unzipped, layers cap at 250MB)
- How to go from Python to deployed Lambda function without shell scripts, Terraform, or SAM CLI
- What to monitor once the system is running in production

## Background

Earlier versions of this module taught a DynamoDB single-table design with a parallel `CloudStore` backend and Lambda zip layers. The cassette substrate made most of it unnecessary:

- **Storage is just files on S3.** Cassettes are immutable byte blobs; Parquet indexes are immutable shards; the manifest is a small JSON file. No DynamoDB, no GSIs, no LSIs — `fsspec + s3fs` is enough.

- **Fan-out is a protocol, not a service.** `Executor.map(fn, items)` has three implementations. Pick the one that matches your workload. No changes to the ingest pipeline.

- **Deployment is Python.** `cognition.cassette.lambda_container` builds the image, pushes to ECR, and creates/updates the Lambda function via `boto3`. One probe runs the whole flow dry-run → build → push → deploy with four command-line flags.

What remains: understand why a container is the right packaging choice, how the executor protocol maps onto Lambda's concurrency model, and how to observe the system once it's live.

## Prompt 1: FsspecFetcher — the range-get abstraction

---

```
Add to cognition/src/cognition/cassette/reader.py:

  class FsspecFetcher:
      \"\"\"RangeFetcher backed by fsspec for any scheme.\"\"\"
      requests: int = 0   # for probe instrumentation
      bytes_read: int = 0

      def fetch(self, path: str, offset: int, length: int) -> bytes:
          with fsspec.open(path, \"rb\") as f:
              f.seek(offset)
              buf = f.read(length)
          self.requests += 1
          self.bytes_read += len(buf)
          return buf

      def size(self, path: str) -> int:
          fs, rel = fsspec.core.url_to_fs(path)
          return fs.size(rel)

      def reset_counters(self):
          self.requests = 0
          self.bytes_read = 0

InfonStore._get_fetcher branches exactly once on `\"://\" in self.root`.
Local paths stay on LocalFetcher (plain open/seek/read, no s3fs cost);
remote paths switch to FsspecFetcher.

That's the entire S3 story at the storage layer. Everything else
(writes, indexes, manifest, migration) works unchanged because pyarrow
and fsspec both speak URIs natively.
```

---

## Prompt 2: The Executor Protocol

---

```
Build cognition/src/cognition/cassette/executor.py:

  class Executor(Protocol):
      def map(self, fn, items) -> list[Result]: ...

  @dataclass
  class Result:
      index: int
      value: Any | None = None
      error: str | None = None
      @property
      def ok(self) -> bool: return self.error is None

Three implementations:

  SyncExecutor        — in-process sequential baseline.
  ProcessExecutor     — concurrent.futures.ProcessPoolExecutor with
                        spawn context. Used via `with ProcessExecutor(4)
                        as ex: store.ingest(docs, executor=ex)`.
  LambdaExecutor      — submits each IngestBatch as boto3 invoke() with
                        InvocationType=\"RequestResponse\".
                        Lives in executor_lambda.py so boto3 stays
                        optional until you need it.

All three wrap exceptions into Result.error strings — one bad doc
never aborts a batch of 10,000.

InfonStore.ingest(docs, executor=None) batches items per worker so the
SPLADE cold start amortizes (spawning 4 workers to extract 4 docs one
at a time would be absurd). Default batch size = ceil(len(docs) /
executor.workers).
```

---

## Prompt 3: The Lambda Handler

---

```
cognition/src/cognition/cassette/handler.py is the Lambda entrypoint.
~80 lines. Nothing fancy.

  def handler(event, context):
      schema_s3     = event[\"schema_s3\"]
      schema_ref    = event[\"schema_ref\"]
      out_root      = event[\"out_root\"]        # must be s3://...
      jobs          = event[\"jobs\"]

      schema = load_schema_from_s3(schema_s3)
      encoder = cached_encoder(schema)

      summaries = []
      for job in jobs:
          infons, _ = extract_infons([job[\"doc\"]], encoder, schema, cfg)
          path = f\"{out_root}/cassettes/{job['cassette_id']}.inf\"
          with fsspec.open(path, \"wb\") as f:
              w = CassetteWriter(f, job[\"cassette_id\"], schema_ref)
              for inf in infons: w.add(inf)
              footer = w.close()
          build_indexes(footer, f\"{out_root}/index\")
          summaries.append({
              \"cassette_id\": job[\"cassette_id\"],
              \"cassette_path\": path,
              \"footer_json\": footer.to_json().decode(),
              ...
          })
      return {\"summaries\": summaries}

Schema + encoder are cached as module-level globals so warm invocations
skip the ~400ms SPLADE load. Cold starts pay it once per container
lifecycle.
```

---

## Prompt 4: Container Packaging

---

```
Build cognition/src/cognition/cassette/lambda_container.py with four
entry points, each owning one step:

  build_image(image_tag, build_dir, requirements=None,
              base_image=\"public.ecr.aws/lambda/python:3.11\",
              platform=\"linux/amd64\")
      # Writes Dockerfile + handler.py + cognition/ into build_dir.
      # Runs `docker buildx build --platform linux/amd64 --provenance=false
      #       --load -t image_tag build_dir`.
      # Returns the image tag on success.

  ensure_ecr_repo(repo_name, region) -> registry_host
      # boto3 client(\"ecr\").describe_repositories; create if missing.

  push_image(image_tag, repo_name, region, tag=\"latest\") -> image_uri
      # boto3 get_authorization_token → docker login --password-stdin
      # docker tag; docker push
      # Returns the RepoDigest pin (sha256 digest, not :latest).

  publish_function(image_uri, function_name, region, role_arn,
                    memory_mb=2048, timeout_s=300)
      # boto3 create_function with PackageType=Image.
      # On ResourceConflictException: update_function_code +
      # update_function_configuration, waiting on function_updated.

Why these tradeoffs:
- --platform linux/amd64 cross-builds from macOS without Docker Desktop.
- --provenance=false: otherwise the manifest has an attestation that
  Lambda rejects.
- --load makes the image appear in the local daemon so docker tag/push
  work afterwards.
- PyTorch 2.6+ CPU wheels are ~1.5GB unzipped. 250MB zip-layer limit
  is impossible; 10GB container limit has real headroom.
- 2GB memory, 5min timeout. Cold start ~3s (SPLADE load); warm ~50ms.

The probe experiments/cassette_lab/probe_lambda_container.py has four
modes so you can step through without committing to a push:
  --dry-run → write Dockerfile + context, no docker
  --build   → docker buildx, report image size
  --push    → + ECR login + push
  --deploy  → + Lambda publish + smoke invoke
Default is --dry-run: no flag is ever destructive.
```

---

## Operational Runbook

### Schema migration in the cloud

Migrations are a read-hydrate-rewrite pattern: read every infon under the old schema, apply the `SchemaFunctor`, write new cassettes tagged with the new `schema_ref`. On S3 this costs the same bandwidth as a full corpus read plus ~10% for the new writes.

```python
store = InfonStore("s3://acme/chips", schema_path="schema_v1.json")
functor = SchemaFunctor(
    rename={"toyota_motors": "toyota"},
    merge={"panasonic_energy": "panasonic"},
    delete={"tpu"},
)
print(store.plan_migration(functor, "schema_v2.json").summary())  # preview
store.migrate(functor, "schema_v2.json")                          # commit
```

Old cassettes remain at their existing S3 keys. Time-travel to the pre-migration snapshot is free — the manifest chain still resolves.

### Time-travel in the cloud

`Manifest.load_at(root, snap_id)` reads one JSON file from `_manifest/<snap>.json` and returns an immutable manifest. Queries against that manifest see only cassettes that existed at that snapshot. Costs the same as any other manifest read — a single S3 GET.

### What to monitor

| Metric | Where | What it tells you |
|---|---|---|
| Lambda invocations + duration | CloudWatch | Ingest throughput; cold-start rate |
| Lambda concurrent executions | CloudWatch | Hitting account concurrency cap? |
| S3 GETs on `cassettes/` | S3 Storage Lens | Query hydration volume |
| S3 GETs on `index/by_triple/` | S3 Storage Lens | Planner scan cost |
| `_manifest/HEAD` object timestamps | S3 | When last ingest committed |
| `docs/` prefix size | S3 `ls` | How much corpus has been registered |

No bespoke observability system needed. CloudWatch + S3 metrics cover it.

## Why no DynamoDB

The old design used DynamoDB for the infon index (fast key-value + GSIs for anchor lookup) and S3 for the rest. Three practical reasons the cassette substrate drops it:

1. **Parquet on S3 with pyarrow does the same job, cheaper.** Single-digit ms scans per shard, and the manifest pruner skips ~90% of shards on realistic workloads. Per-query cost is ~$0.0001 at typical scale.

2. **Schema evolution was awful on DynamoDB.** Adding a GSI to an existing table takes hours and blocks writes. Cassettes let us add a whole new shard type (`by_edge.parquet` for example) by just writing the Parquet files — no schema change, no migration window.

3. **Delta ingest was the killer.** DynamoDB puts are transactional per-item, not per-batch. Ingesting a batch of 100 docs required 100 puts plus index updates plus consistency waits. Cassette batches are one object write + three Parquet writes. On S3 that's ~200ms; on DynamoDB it was 2s.

DynamoDB is still supported via `cognition/src/cognition/store/cloud.py` for legacy integrations. New deployments should use the cassette path.

## Checkpoint

- [ ] `InfonStore("s3://bucket/prefix", schema_path=...)` creates the expected five directories under the prefix.
- [ ] `store.ingest(docs)` writes one cassette + 3 indexes + 1 manifest snapshot per batch.
- [ ] `Query().where(...).run(store.manifest)` works identically against s3:// and local.
- [ ] `ProcessExecutor(workers=4)` fan-out completes at ≥3× the SyncExecutor wall time at 100+ docs.
- [ ] `probe_lambda_container.py --dry-run` writes a sane Dockerfile without docker installed.
- [ ] `probe_lambda_container.py --deploy` on a test account produces a Lambda that round-trips one doc → cassette on S3.
