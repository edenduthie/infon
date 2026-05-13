#!/usr/bin/env python3
"""
Monitor SQS-based NuExtract-2.0-4B GPU extraction progress.

Reads SQS queue depth for work remaining and per-instance progress files from S3.

Usage:
    python -m sirius.step1_v1.monitor_gpu                # One-shot status
    python -m sirius.step1_v1.monitor_gpu --watch        # Poll every 60s
    python -m sirius.step1_v1.monitor_gpu --spot-check   # Read a sample result file
"""

import argparse
import json
import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
OUTPUT_BUCKET = "sirius-dimefiled-results"
OUTPUT_PREFIX = "nuextract-v2-events/"
PROGRESS_PREFIX = "nuextract-v2-progress/"
QUEUE_NAME = "nuextract-gpu-work"
DLQ_NAME = "nuextract-gpu-work-dlq"

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def get_queue_stats():
    """Get SQS queue and DLQ message counts."""
    stats = {"queued": 0, "in_flight": 0, "dlq": 0, "queue_exists": False}

    try:
        queue_url = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
            ],
        )["Attributes"]
        stats["queued"] = int(attrs.get("ApproximateNumberOfMessages", 0))
        stats["in_flight"] = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
        stats["queue_exists"] = True
    except Exception:
        return stats

    try:
        dlq_url = sqs.get_queue_url(QueueName=DLQ_NAME)["QueueUrl"]
        dlq_attrs = sqs.get_queue_attributes(
            QueueUrl=dlq_url,
            AttributeNames=["ApproximateNumberOfMessages"],
        )["Attributes"]
        stats["dlq"] = int(dlq_attrs.get("ApproximateNumberOfMessages", 0))
    except Exception:
        pass

    return stats


def get_worker_progress():
    """Read per-instance progress files from S3."""
    paginator = s3.get_paginator("list_objects_v2")
    workers = []

    for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=f"{PROGRESS_PREFIX}instance-"):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=OUTPUT_BUCKET, Key=obj["Key"])
                data = json.loads(resp["Body"].read())
                workers.append(data)
            except Exception:
                pass

    return workers


def count_result_files():
    """Count total result files in the output prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    total_size = 0
    for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get("Contents", []):
            count += 1
            total_size += obj.get("Size", 0)
    return count, total_size


def status():
    """Display NuExtract extraction progress."""
    queue_stats = get_queue_stats()
    workers = get_worker_progress()
    file_count, total_size = count_result_files()

    print(f"{'=' * 70}")
    print(f"NuExtract-2.0-4B GPU Extraction Progress (SQS Multi-Worker)")
    print(f"{'=' * 70}")

    # SQS queue status
    if not queue_stats["queue_exists"]:
        print("\n  SQS queue not found. Run --enqueue first.")
        print(f"{'=' * 70}")
        return False

    remaining = queue_stats["queued"] + queue_stats["in_flight"]

    print(f"\nSQS Queue: {QUEUE_NAME}")
    print(f"  Messages queued:     {queue_stats['queued']:,}")
    print(f"  Messages in-flight:  {queue_stats['in_flight']:,}")
    print(f"  Total remaining:     {remaining:,}")
    if queue_stats["dlq"] > 0:
        print(f"  DLQ (failed 3x):     {queue_stats['dlq']:,}")

    # Per-worker progress
    total_processed = 0
    total_relevant = 0
    total_high = 0
    total_medium = 0
    total_errors = 0
    total_rps = 0.0
    active_workers = 0

    if workers:
        print(f"\nWorkers ({len(workers)}):")
        for w in sorted(workers, key=lambda x: (x.get("instance_id", ""), x.get("gpu_id", 0))):
            inst = w.get("instance_id", "?")[:20]
            gpu = w.get("gpu_id", "?")
            processed = w.get("total_processed", 0)
            relevant = w.get("total_relevant", 0)
            rps = w.get("records_per_sec", 0)
            worker_status = w.get("status", "running")

            status_indicator = "DONE" if worker_status in ("complete", "shutdown") else "LIVE"
            print(f"  {inst} GPU-{gpu}: {processed:>8,} processed, {relevant:>6,} relevant, {rps:5.1f} rec/s [{status_indicator}]")

            total_processed += processed
            total_relevant += relevant
            total_high += w.get("total_high", 0)
            total_medium += w.get("total_medium", 0)
            total_errors += w.get("total_errors", 0)
            total_rps += rps
            if worker_status not in ("complete", "shutdown"):
                active_workers += 1
    else:
        print("\n  No worker progress files found yet.")

    # Estimate progress
    completed_messages = total_processed // 10 if total_processed > 0 else 0
    total_messages = remaining + completed_messages
    pct = (completed_messages / total_messages * 100) if total_messages > 0 else 0

    # Aggregate stats
    print(f"\nAggregate:")
    print(f"  Records processed: {total_processed:,}")
    if total_messages > 0:
        print(f"  Messages progress: ~{completed_messages:,} / {total_messages:,} ({pct:.1f}%)")
    print(f"  Relevant docs:     {total_relevant:,}")
    print(f"  HIGH events:       {total_high:,}")
    print(f"  MEDIUM events:     {total_medium:,}")
    print(f"  Errors:            {total_errors:,}")
    print(f"  Throughput:        {total_rps:.1f} rec/s ({total_rps * 3600:.0f}/hr)")
    print(f"  Active workers:    {active_workers}")
    print(f"  S3 result files:   {file_count:,} ({total_size / 1024 / 1024:.1f} MB)")

    # ETA
    if total_rps > 0 and remaining > 0:
        remaining_records = remaining * 10
        eta_h = remaining_records / total_rps / 3600
        print(f"  ETA:               {eta_h:.1f}h remaining")

    # Cost estimate (g5.xlarge spot ~$0.42/hr per instance)
    if active_workers > 0:
        spot_rate = 0.42 * active_workers
        print(f"  Spot cost rate:    ~${spot_rate:.2f}/hr ({active_workers} instance(s))")

    # Status determination
    is_done = remaining == 0 and total_processed > 0 and active_workers == 0
    if is_done:
        print(f"\n  STATUS: COMPLETE")
    elif total_processed > 0 or queue_stats["in_flight"] > 0:
        print(f"\n  STATUS: IN PROGRESS")
    elif queue_stats["queued"] > 0:
        print(f"\n  STATUS: QUEUED (waiting for workers)")
    else:
        print(f"\n  STATUS: NOT STARTED")

    print(f"{'=' * 70}")
    return is_done


def spot_check():
    """Read and display a sample result file."""
    paginator = s3.get_paginator("list_objects_v2")
    target_key = None
    for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=f"{OUTPUT_PREFIX}gpu_"):
        for obj in page.get("Contents", []):
            target_key = obj["Key"]
            break
        if target_key:
            break

    if not target_key:
        for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=OUTPUT_PREFIX, MaxKeys=5):
            for obj in page.get("Contents", []):
                target_key = obj["Key"]
                break
            if target_key:
                break

    if not target_key:
        print("No result files found.")
        return

    print(f"Spot-checking: s3://{OUTPUT_BUCKET}/{target_key}\n")
    resp = s3.get_object(Bucket=OUTPUT_BUCKET, Key=target_key)
    data = json.loads(resp["Body"].read())
    docs = data if isinstance(data, list) else [data]

    print(f"Documents in file: {len(docs)}")
    relevant = [d for d in docs if d.get("relevant")]
    print(f"Relevant: {len(relevant)}")
    print()

    for doc in relevant[:3]:
        print(f"  URL:   {doc.get('url', '?')[:80]}")
        print(f"  Title: {str(doc.get('title', '?'))[:80]}")
        for evt in doc.get("events", [])[:2]:
            print(f"    Domain:      {evt.get('l1_domain', '?')}")
            print(f"    Escalation:  {evt.get('escalation_classification', '?')}")
            print(f"    Initiators:  {evt.get('initiators', [])}")
            print(f"    Targets:     {evt.get('targets', [])}")
            print(f"    Edges:       {evt.get('edge_count', 0)}")
            print(f"    Quality:     {evt.get('quality', '?')}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor NuExtract GPU extraction progress")
    parser.add_argument("--watch", action="store_true", help="Poll every 60s until done")
    parser.add_argument("--spot-check", action="store_true", help="Read a sample result")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    if args.spot_check:
        spot_check()
    elif args.watch:
        while True:
            done = status()
            if done:
                print("\nExtraction complete!")
                break
            print(f"\nNext check in {args.interval}s...\n")
            time.sleep(args.interval)
    else:
        status()
