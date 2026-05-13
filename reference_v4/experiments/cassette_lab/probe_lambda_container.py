"""Container-image Lambda deployment probe.

Modes (default is --dry-run — no AWS, no network, no push):

    --dry-run (default)
        Write the Dockerfile + build context. Stop.
        Does NOT require docker.

    --build
        Dry-run + actually invoke docker buildx. Reports image size.
        Requires docker.

    --push
        Everything above + ECR login + docker push.
        Requires docker + boto3 + credentials + --region + --repo.

    --deploy
        Everything above + publish Lambda function + invoke once.
        Requires --role-arn + --bucket.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))


def cmd_dry_run(args):
    from cognition.cassette.lambda_container import (
        _write_build_context, _DEFAULT_REQS, _BASE_IMAGE,
    )
    from pathlib import Path
    import shutil

    build_dir = Path(args.build_dir).resolve()
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    print("== dry-run: write build context ==")
    _write_build_context(build_dir, _DEFAULT_REQS, _BASE_IMAGE)

    print(f"\n  contents of {build_dir}:")
    for entry in sorted(build_dir.iterdir()):
        if entry.is_file():
            print(f"    {entry.name:<24} {entry.stat().st_size:>8}B")
        else:
            total = sum(f.stat().st_size
                         for f in entry.rglob("*") if f.is_file())
            print(f"    {entry.name}/  ({total / 1024 / 1024:.1f}MB)")

    print(f"\n  Dockerfile preview:")
    for line in (build_dir / "Dockerfile").read_text().splitlines():
        print(f"    {line}")


def cmd_build(args):
    from cognition.cassette.lambda_container import build_image
    print("== build image ==")
    t0 = time.perf_counter()
    tag = build_image(
        image_tag=args.image_tag,
        build_dir=args.build_dir,
    )
    wall = time.perf_counter() - t0
    print(f"\n  done: {tag}  (build {wall:.0f}s)")


def cmd_push(args):
    from cognition.cassette.lambda_container import build_image, push_image
    if not args.repo:
        raise SystemExit("--repo required for --push")
    build_image(image_tag=args.image_tag, build_dir=args.build_dir)
    uri = push_image(
        image_tag=args.image_tag,
        repo_name=args.repo,
        region=args.region,
    )
    print(f"\n  image URI: {uri}")


def cmd_deploy(args):
    import boto3
    from cognition.cassette.lambda_container import (
        build_image, push_image, publish_function,
    )
    if not (args.repo and args.role_arn and args.bucket):
        raise SystemExit("--repo, --role-arn, --bucket all required for --deploy")

    build_image(image_tag=args.image_tag, build_dir=args.build_dir)
    uri = push_image(image_tag=args.image_tag, repo_name=args.repo,
                     region=args.region)
    publish_function(
        image_uri=uri,
        function_name=args.function_name,
        region=args.region,
        role_arn=args.role_arn,
    )

    # Upload a tiny schema + invoke once to prove it runs end-to-end.
    s3 = boto3.client("s3", region_name=args.region)
    schema_body = json.dumps({
        "toyota":    {"type": "actor", "tokens": ["toyota"]},
        "panasonic": {"type": "actor", "tokens": ["panasonic"]},
        "partner":   {"type": "relation", "tokens": ["partner", "partnered"]},
        "batteries": {"type": "feature", "tokens": ["battery", "batteries"]},
    }).encode()
    schema_key = "cognition-probe/schema.json"
    s3.put_object(Bucket=args.bucket, Key=schema_key, Body=schema_body)
    schema_s3 = f"s3://{args.bucket}/{schema_key}"
    print(f"\n  → uploaded schema to {schema_s3}")

    event = {
        "schema_s3": schema_s3,
        "schema_ref": "probe",
        "out_root": f"s3://{args.bucket}/cognition-probe/data",
        "jobs": [
            {"doc": {"id": "d1",
                     "text": "Toyota partnered with Panasonic on batteries.",
                     "timestamp": "2026-01-04"},
             "cassette_id": "probe_c1"},
        ],
    }
    lam = boto3.client("lambda", region_name=args.region)
    print(f"  → invoking {args.function_name}")
    t0 = time.perf_counter()
    resp = lam.invoke(
        FunctionName=args.function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(event).encode(),
    )
    wall = (time.perf_counter() - t0) * 1000
    body = resp["Payload"].read().decode()
    print(f"\n  invoke: status={resp['StatusCode']} wall={wall:.0f}ms")
    print(f"  response (first 600 chars):")
    print(f"    {body[:600]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--deploy", action="store_true")

    ap.add_argument("--build-dir", default="./build/lambda-image")
    ap.add_argument("--image-tag", default="cognition-ingest:latest")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--repo", default="")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--role-arn", default="")
    ap.add_argument("--function-name", default="cognition-ingest")
    args = ap.parse_args()

    if args.deploy:
        cmd_deploy(args)
    elif args.push:
        cmd_push(args)
    elif args.build:
        cmd_build(args)
    else:
        cmd_dry_run(args)


if __name__ == "__main__":
    main()
