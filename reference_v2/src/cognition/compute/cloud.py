"""CloudCompute: Lambda + SQS for serverless parallel execution.

Pywren-style: serialize function + args → S3, invoke Lambda, collect results.
Lambda containers have ModernBERT baked in (~600MB image), no VPC, no EFS.
"""

from __future__ import annotations

import json
import pickle
import uuid
import time
from typing import Callable, Any


def _require_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "CloudCompute requires boto3. Install with: pip install cognition[aws]"
        )


class CloudCompute:
    """AWS Lambda + S3 compute backend.

    Each work unit is:
    1. Pickle (fn, args) → S3
    2. Invoke Lambda with pointer to S3 object
    3. Lambda executes, writes result to S3
    4. Collect results

    The Lambda container image must have cognition + torch + transformers
    installed and the trained model baked in at /opt/model/.
    """

    def __init__(self, function_name: str, bucket: str,
                 region: str = "us-east-1", timeout: int = 300):
        self.function_name = function_name
        self.bucket = bucket
        self.region = region
        self.timeout = timeout
        self._lambda = None
        self._s3 = None

    @property
    def lambda_client(self):
        if self._lambda is None:
            boto3 = _require_boto3()
            self._lambda = boto3.client("lambda", region_name=self.region)
        return self._lambda

    @property
    def s3(self):
        if self._s3 is None:
            boto3 = _require_boto3()
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def map(self, fn: Callable, items: list[Any], **kwargs) -> list[Any]:
        """Fan-out: invoke one Lambda per item, collect results."""
        if not items:
            return []

        job_id = uuid.uuid4().hex[:8]
        invocations = []

        # Fan out
        for i, item in enumerate(items):
            key = f"cognition/jobs/{job_id}/{i:04d}/input.pkl"
            payload = pickle.dumps({"fn": fn, "args": (item,), "kwargs": kwargs})
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload)

            result_key = f"cognition/jobs/{job_id}/{i:04d}/output.pkl"
            self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="Event",  # async
                Payload=json.dumps({
                    "bucket": self.bucket,
                    "input_key": key,
                    "output_key": result_key,
                }),
            )
            invocations.append(result_key)

        # Collect results with polling
        results = [None] * len(invocations)
        deadline = time.time() + self.timeout
        pending = set(range(len(invocations)))

        while pending and time.time() < deadline:
            for i in list(pending):
                try:
                    resp = self.s3.get_object(
                        Bucket=self.bucket, Key=invocations[i],
                    )
                    results[i] = pickle.loads(resp["Body"].read())
                    pending.remove(i)
                except self.s3.exceptions.NoSuchKey:
                    continue
                except Exception:
                    continue

            if pending:
                time.sleep(2)

        if pending:
            raise TimeoutError(
                f"CloudCompute: {len(pending)} of {len(invocations)} "
                f"tasks did not complete within {self.timeout}s"
            )

        return results

    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a single task. Returns a CloudFuture."""
        job_id = uuid.uuid4().hex[:8]
        key = f"cognition/jobs/{job_id}/input.pkl"
        result_key = f"cognition/jobs/{job_id}/output.pkl"

        payload = pickle.dumps({"fn": fn, "args": args, "kwargs": kwargs})
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload)

        self.lambda_client.invoke(
            FunctionName=self.function_name,
            InvocationType="Event",
            Payload=json.dumps({
                "bucket": self.bucket,
                "input_key": key,
                "output_key": result_key,
            }),
        )

        return CloudFuture(self.s3, self.bucket, result_key, self.timeout)

    def shutdown(self) -> None:
        pass  # Serverless — nothing to clean up


class CloudFuture:
    """A future-like wrapper for an async Lambda result in S3."""

    def __init__(self, s3_client, bucket: str, key: str, timeout: int):
        self._s3 = s3_client
        self._bucket = bucket
        self._key = key
        self._timeout = timeout
        self._result = None
        self._done = False

    def result(self, timeout: int | None = None) -> Any:
        if self._done:
            return self._result

        deadline = time.time() + (timeout or self._timeout)
        while time.time() < deadline:
            try:
                resp = self._s3.get_object(Bucket=self._bucket, Key=self._key)
                self._result = pickle.loads(resp["Body"].read())
                self._done = True
                return self._result
            except Exception:
                time.sleep(2)

        raise TimeoutError(f"CloudFuture: result not available within timeout")

    def done(self) -> bool:
        if self._done:
            return True
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key)
            return True
        except Exception:
            return False
