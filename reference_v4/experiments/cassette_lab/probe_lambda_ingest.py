"""Lambda deployment probe.

Three modes, controlled by flags:

    --dry-run (default)
        Build the layer zip locally. Report size. Stop.
        No AWS calls, no credentials required.

    --publish
        Build + upload the layer to AWS. Requires boto3 + credentials
        + --region + --bucket (for the layer stage). Publishes layer
        version and prints ARN.

    --deploy
        Everything above, PLUS: publish the function, upload schema
        + a test doc to S3, invoke once, print result.
        Requires --role-arn (pre-created IAM role with Lambda basic
        execution + S3 read/write on --bucket).

Default is dry-run so no flag ever runs destructively.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))


def cmd_dry_run(args):
    from cognition.cassette.lambda_package import build_layer, build_function_zip
    print("== dry-run: build layer ==")
    t0 = time.perf_counter()
    layer_zip = build_layer(output_dir=args.output_dir)
    print(f"  layer built in {time.perf_counter() - t0:.1f}s")
    print(f"  zip: {layer_zip}")
    print(f"  unzipped peek (run `unzip -l {layer_zip} | head`):")
    import zipfile
    with zipfile.ZipFile(layer_zip) as zf:
        items = zf.infolist()
    total_mb = sum(i.file_size for i in items) / 1024 / 1024
    print(f"    {len(items)} files, {total_mb:.0f}MB unzipped")
    # Show top-level directories inside the layer.
    tops: dict[str, int] = {}
    for i in items:
        parts = i.filename.split("/")
        if len(parts) >= 2:
            tops[parts[1]] = tops.get(parts[1], 0) + i.file_size
    print(f"  top packages by size:")
    for name, size in sorted(tops.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {name:<24} {size / 1024 / 1024:6.1f}MB")

    print("\n== build handler zip ==")
    handler_zip = build_function_zip(output_dir=args.output_dir)
    print(f"  zip: {handler_zip}")


def cmd_publish(args):
    from cognition.cassette.lambda_package import build_layer, publish_layer
    layer_zip = build_layer(output_dir=args.output_dir)
    resp = publish_layer(
        zip_path=layer_zip,
        layer_name=args.layer_name,
        region=args.region,
        upload_bucket=args.bucket,
    )
    print(f"\nLayer ARN: {resp['LayerVersionArn']}")


def cmd_deploy(args):
    import boto3
    import json as _json
    from cognition.cassette.lambda_package import (
        build_layer, publish_layer, build_function_zip, publish_function,
    )

    if not args.role_arn:
        raise SystemExit("--role-arn is required for --deploy")

    # 1. Layer.
    layer_zip = build_layer(output_dir=args.output_dir)
    layer_resp = publish_layer(
        zip_path=layer_zip, layer_name=args.layer_name,
        region=args.region, upload_bucket=args.bucket,
    )
    layer_arn = layer_resp["LayerVersionArn"]

    # 2. Function.
    handler_zip = build_function_zip(output_dir=args.output_dir)
    publish_function(
        zip_path=handler_zip,
        function_name=args.function_name,
        region=args.region,
        layer_arns=[layer_arn],
        role_arn=args.role_arn,
    )

    # 3. Upload schema + test invoke.
    s3 = boto3.client("s3", region_name=args.region)
    schema_body = _json.dumps({
        "toyota":    {"type": "actor", "tokens": ["toyota"]},
        "panasonic": {"type": "actor", "tokens": ["panasonic"]},
        "partner":   {"type": "relation", "tokens": ["partner", "partnered"]},
        "batteries": {"type": "feature", "tokens": ["battery", "batteries"]},
    }).encode()
    schema_key = "cognition-probe/schema.json"
    s3.put_object(Bucket=args.bucket, Key=schema_key, Body=schema_body)
    schema_s3 = f"s3://{args.bucket}/{schema_key}"
    print(f"  → uploaded schema to {schema_s3}")

    # 4. Invoke once.
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
    t0 = time.perf_counter()
    resp = lam.invoke(
        FunctionName=args.function_name,
        InvocationType="RequestResponse",
        Payload=_json.dumps(event).encode(),
    )
    wall = (time.perf_counter() - t0) * 1000
    body = resp["Payload"].read().decode()
    print(f"\n  invoke: status={resp['StatusCode']} wall={wall:.0f}ms")
    print(f"  response: {body[:600]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    p0 = ap.add_argument_group("modes")
    p0.add_argument("--dry-run", action="store_true", default=True)
    p0.add_argument("--publish", action="store_true")
    p0.add_argument("--deploy", action="store_true")

    ap.add_argument("--output-dir", default="./build")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--layer-name", default="cognition-runtime")
    ap.add_argument("--function-name", default="cognition-ingest")
    ap.add_argument("--role-arn", default="")

    args = ap.parse_args()

    if args.deploy:
        if not args.bucket:
            raise SystemExit("--bucket is required for --deploy")
        cmd_deploy(args)
    elif args.publish:
        if not args.bucket:
            raise SystemExit("--bucket is required for --publish")
        cmd_publish(args)
    else:
        cmd_dry_run(args)


if __name__ == "__main__":
    main()
