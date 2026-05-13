"""Lambda: Split large Athena CSV into chunks on S3 and create the SQS queue.

Streams the source CSV from S3 line-by-line, writes chunk CSVs back to S3,
and creates the SQS work queue + DLQ.  Chunk size is computed dynamically
so the total number of chunks stays under the Inline Map limit (40).

Input:
    {
        "csv_s3_uri": "s3://bucket/index/step1_results.csv",
        "record_count": 11094515,   # from Athena step (used to size chunks)
        "max_records": null          # optional cap for testing
    }

Output:
    {
        "queue_url": "https://sqs.../nuextract-gpu-work",
        "chunks": [
            {"csv_s3_uri": "s3://bucket/index/chunks/chunk_0000.csv", "record_count": 350000},
            ...
        ],
        "total_records": 11094515,
        "chunk_count": 32
    }
"""

import json
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
QUEUE_NAME = os.environ.get("SQS_QUEUE_NAME", "nuextract-gpu-work")
DLQ_NAME = os.environ.get("SQS_DLQ_NAME", "nuextract-gpu-work-dlq")
VISIBILITY_TIMEOUT = int(os.environ.get("SQS_VISIBILITY_TIMEOUT", "600"))
MESSAGE_RETENTION = 345600  # 4 days
DLQ_MAX_RECEIVE_COUNT = 3
MAX_CHUNKS = 35  # keep under 40-item Inline Map limit
MIN_CHUNK_SIZE = 100_000

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)


def handler(event, context):
    csv_s3_uri = event["csv_s3_uri"]
    record_count = int(event.get("record_count") or 0)
    max_records = int(event["max_records"]) if event.get("max_records") else None

    # Effective record count for chunk sizing
    effective = min(record_count, max_records) if max_records and record_count else (max_records or record_count)
    chunk_size = max(effective // MAX_CHUNKS, MIN_CHUNK_SIZE) if effective else MIN_CHUNK_SIZE

    # Parse S3 URI
    path = csv_s3_uri.replace("s3://", "")
    bucket, key = path.split("/", 1)
    key_base = key.rsplit(".", 1)[0]  # e.g. "index/step1_results"
    chunk_prefix = f"{key_base}_chunks/"

    # Create SQS queues
    queue_url = _ensure_queues()

    # Stream CSV from S3 and write chunks
    resp = s3.get_object(Bucket=bucket, Key=key)
    lines_iter = resp["Body"].iter_lines()

    # Read header
    header_bytes = next(lines_iter)
    header = header_bytes if isinstance(header_bytes, bytes) else header_bytes.encode("utf-8")

    chunks = []
    chunk_buf = [header]
    chunk_rows = 0
    total_rows = 0
    chunk_idx = 0

    for raw_line in lines_iter:
        line = raw_line if isinstance(raw_line, bytes) else raw_line.encode("utf-8")
        if not line.strip():
            continue

        chunk_buf.append(line)
        chunk_rows += 1
        total_rows += 1

        if chunk_rows >= chunk_size:
            chunk_key = f"{chunk_prefix}chunk_{chunk_idx:04d}.csv"
            _write_chunk(bucket, chunk_key, chunk_buf)
            chunks.append({
                "csv_s3_uri": f"s3://{bucket}/{chunk_key}",
                "record_count": chunk_rows,
            })
            chunk_buf = [header]
            chunk_rows = 0
            chunk_idx += 1

        if max_records and total_rows >= max_records:
            break

    # Flush remaining rows
    if chunk_rows > 0:
        chunk_key = f"{chunk_prefix}chunk_{chunk_idx:04d}.csv"
        _write_chunk(bucket, chunk_key, chunk_buf)
        chunks.append({
            "csv_s3_uri": f"s3://{bucket}/{chunk_key}",
            "record_count": chunk_rows,
        })

    print(
        f"Split {total_rows} records into {len(chunks)} chunks "
        f"(~{chunk_size} rows each) at s3://{bucket}/{chunk_prefix}"
    )

    return {
        "queue_url": queue_url,
        "chunks": chunks,
        "total_records": total_rows,
        "chunk_count": len(chunks),
    }


def _write_chunk(bucket: str, key: str, lines: list):
    """Write a list of byte lines as a single CSV object to S3."""
    body = b"\n".join(
        l if isinstance(l, bytes) else l.encode("utf-8") for l in lines
    )
    s3.put_object(Bucket=bucket, Key=key, Body=body)


def _ensure_queues() -> str:
    """Create SQS work queue + DLQ.  Return work queue URL."""
    # DLQ
    try:
        resp = sqs.create_queue(
            QueueName=DLQ_NAME,
            Attributes={"MessageRetentionPeriod": str(MESSAGE_RETENTION)},
        )
        dlq_url = resp["QueueUrl"]
    except sqs.exceptions.QueueNameExists:
        dlq_url = sqs.get_queue_url(QueueName=DLQ_NAME)["QueueUrl"]

    dlq_arn = sqs.get_queue_attributes(
        QueueUrl=dlq_url, AttributeNames=["QueueArn"]
    )["Attributes"]["QueueArn"]

    # Work queue
    redrive_policy = json.dumps({
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": str(DLQ_MAX_RECEIVE_COUNT),
    })

    try:
        resp = sqs.create_queue(
            QueueName=QUEUE_NAME,
            Attributes={
                "VisibilityTimeout": str(VISIBILITY_TIMEOUT),
                "MessageRetentionPeriod": str(MESSAGE_RETENTION),
                "RedrivePolicy": redrive_policy,
            },
        )
        return resp["QueueUrl"]
    except sqs.exceptions.QueueNameExists:
        queue_url = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
        sqs.set_queue_attributes(
            QueueUrl=queue_url,
            Attributes={
                "VisibilityTimeout": str(VISIBILITY_TIMEOUT),
                "MessageRetentionPeriod": str(MESSAGE_RETENTION),
                "RedrivePolicy": redrive_policy,
            },
        )
        return queue_url
