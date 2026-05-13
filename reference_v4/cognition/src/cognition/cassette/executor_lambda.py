"""LambdaExecutor — AWS Lambda fan-out that conforms to the Executor protocol.

Same `.map(fn, items) → list[Result]` as SyncExecutor and ProcessExecutor.
Swap executors and the ingest pipeline is unchanged.

Over-the-wire contract:
  • items are `IngestBatch` instances (from cognition.cassette.store).
  • Each batch is converted to the handler's event schema (JSON).
  • boto3 invokes the function concurrently via a ThreadPoolExecutor
    (Lambda invoke calls are network-bound, not CPU).
  • Response payload JSON is parsed back into the same summary dict
    shape that ProcessExecutor returns — so the driver needs no branching.

Scope limits (intentional for v1):
  • Synchronous invoke only (`InvocationType=RequestResponse`). Async
    invoke + SQS result aggregation comes later if batches exceed the
    Lambda 15-minute timeout.
  • No retry. boto3 retries transient errors at the transport layer;
    non-transient errors become Result.error.
  • Assumes the handler function already exists. Deployment is a
    separate step (see lambda_package.publish_function).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from .executor import Executor, Result


class LambdaExecutor:
    """Fan out ingest batches to an AWS Lambda function.

    Args:
      function_name: Lambda function ARN or name.
      region:        AWS region.
      max_concurrency: how many in-flight invokes at once. Lambda's
        default account-level concurrency is 1000 per region; we cap
        ours low (16) so a single batch doesn't starve other workloads
        sharing the account. Bump if you have dedicated capacity.
      schema_s3:     s3:// URI of the schema JSON (uploaded once per run
        by the driver; the handler downloads it to /tmp). Overrides the
        per-job `schema_path` from IngestJob, since Lambda can't read
        the driver's local filesystem.
    """

    def __init__(self, function_name: str, region: str,
                 schema_s3: str,
                 max_concurrency: int = 16):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "LambdaExecutor requires boto3. "
                "`pip install boto3` before use."
            )
        self.function_name = function_name
        self.region = region
        self.schema_s3 = schema_s3
        self.max_concurrency = max_concurrency
        self._client = boto3.client("lambda", region_name=region)
        # Expose `workers` for the store's batching logic. This sizes
        # the fan-out, not the in-process pool.
        self.workers = max_concurrency

    # ── Executor protocol ───────────────────────────────────────────────
    def map(self, fn: Callable, items: Iterable) -> list[Result]:
        items = list(items)
        results: list[Result] = [Result(index=i) for i in range(len(items))]

        def invoke_one(i: int, item: Any) -> tuple[int, bool, Any]:
            try:
                payload = self._item_to_event(item)
                resp = self._client.invoke(
                    FunctionName=self.function_name,
                    InvocationType="RequestResponse",
                    Payload=json.dumps(payload).encode(),
                )
                status = resp.get("StatusCode", 0)
                if status >= 300:
                    return i, False, f"Lambda status {status}"
                body = resp["Payload"].read()
                if not body:
                    return i, False, "empty Lambda response"
                parsed = json.loads(body)
                if "errorMessage" in parsed:
                    return i, False, f"Lambda error: {parsed['errorMessage']}"
                # The handler returns {"summaries": [...]} per batch;
                # callers expect the list of per-job summaries directly
                # (matches ProcessExecutor's _ingest_batch contract).
                return i, True, parsed["summaries"]
            except Exception as exc:
                return i, False, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [pool.submit(invoke_one, i, item)
                       for i, item in enumerate(items)]
            for fut in as_completed(futures):
                i, ok, payload = fut.result()
                if ok:
                    results[i].value = payload
                else:
                    results[i].error = payload
        return results

    # ── IngestBatch → Lambda event ──────────────────────────────────────
    def _item_to_event(self, item: Any) -> dict:
        """Convert an IngestBatch to the JSON shape handler.py expects.

        The handler's event carries the schema's S3 URI (the local
        schema_path on each IngestJob is irrelevant to Lambda), plus the
        list of docs + their predetermined cassette IDs."""
        batch = item  # IngestBatch
        if not batch.jobs:
            return {"schema_s3": self.schema_s3, "schema_ref": "",
                    "out_root": "", "jobs": []}
        j0 = batch.jobs[0]
        return {
            "schema_s3": self.schema_s3,
            "schema_ref": j0.schema_ref,
            "out_root": j0.out_root,  # must be s3:// for Lambda
            "jobs": [
                {"doc": j.doc, "cassette_id": j.cassette_id}
                for j in batch.jobs
            ],
        }
