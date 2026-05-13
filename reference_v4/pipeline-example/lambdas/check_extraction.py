"""Lambda handler: Check if GPU extraction is complete.

Polls the SQS queue depth and counts S3 result files to determine whether
the extraction step has finished processing all records.
"""

import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
OUTPUT_BUCKET = os.environ.get("S3_BUCKET", "sirius-dimefiled-results")
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "nuextract-v2-events/")

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def handler(event, context):
    """Lambda entry point.

    Input:
        queue_url: SQS queue URL to check
        allow_partial: If True, consider usable even if some messages remain

    Output:
        remaining: Number of messages still in queue (available + in-flight)
        results_count: Number of result JSON files in S3
        complete: Whether extraction is considered complete
    """
    queue_url = event["queue_url"]
    allow_partial = event.get("allow_partial", False)

    # Check SQS queue depth
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )["Attributes"]

    queued = int(attrs.get("ApproximateNumberOfMessages", 0))
    in_flight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
    remaining = queued + in_flight

    # Count S3 result files
    results_count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                results_count += 1

    # Determine completion
    if remaining == 0:
        complete = True
    elif allow_partial and results_count > 0:
        # Batch job may have failed (spot interruption) but work was done
        complete = True
    else:
        complete = False

    print(f"Queue: {remaining} remaining ({queued} queued, {in_flight} in-flight), "
          f"{results_count} S3 results, complete={complete}")

    return {
        "remaining": remaining,
        "queued": queued,
        "in_flight": in_flight,
        "results_count": results_count,
        "complete": complete,
    }
