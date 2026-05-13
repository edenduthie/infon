"""Lambda: Dispatch a CSV chunk of WARC records to SQS.

Reads a single chunk CSV from S3 (produced by split_csv), batches records
into SQS messages, and sends them using concurrent threads.

Input:
    {
        "csv_s3_uri": "s3://bucket/index/step1_results_chunks/chunk_0003.csv",
        "queue_url": "https://sqs.us-east-1.amazonaws.com/..../nuextract-gpu-work"
    }

Output:
    {"messages_sent": 3500, "record_count": 35000}
"""

import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
RECORDS_PER_MESSAGE = int(os.environ.get("SQS_RECORDS_PER_MESSAGE", "10"))
SQS_SEND_THREADS = int(os.environ.get("SQS_SEND_THREADS", "20"))

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def handler(event, context):
    csv_s3_uri = event["csv_s3_uri"]
    queue_url = event["queue_url"]

    # Parse S3 URI
    path = csv_s3_uri.replace("s3://", "")
    bucket, key = path.split("/", 1)

    # Stream CSV from S3
    resp = s3.get_object(Bucket=bucket, Key=key)
    lines = resp["Body"].iter_lines()

    # Parse header
    header_line = next(lines).decode("utf-8")
    fieldnames = next(csv.reader(io.StringIO(header_line)))

    # Stream rows → batch into SQS messages → collect into send batches
    record_buf = []
    sqs_batches = []       # list of lists, each inner list is up to 10 message bodies
    current_batch = []
    total_records = 0

    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        parsed = next(csv.reader(io.StringIO(line)))
        row = dict(zip(fieldnames, parsed))
        record_buf.append({
            "url": row["url"],
            "warc_filename": row["warc_filename"],
            "warc_record_offset": int(row["warc_record_offset"]),
            "warc_record_length": int(row["warc_record_length"]),
            "crawl_date": row.get("crawl_date", ""),
        })
        total_records += 1

        # Pack RECORDS_PER_MESSAGE records into one SQS message
        if len(record_buf) >= RECORDS_PER_MESSAGE:
            current_batch.append(json.dumps(record_buf))
            record_buf = []

            # Pack 10 messages into one SQS SendMessageBatch call
            if len(current_batch) >= 10:
                sqs_batches.append(current_batch)
                current_batch = []

    # Flush remaining records and messages
    if record_buf:
        current_batch.append(json.dumps(record_buf))
    if current_batch:
        sqs_batches.append(current_batch)

    # Send all batches using a thread pool
    sent = 0
    failed = 0

    def _send_batch(batch):
        entries = [
            {"Id": str(j), "MessageBody": body}
            for j, body in enumerate(batch)
        ]
        resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        return len(resp.get("Successful", [])), len(resp.get("Failed", []))

    with ThreadPoolExecutor(max_workers=SQS_SEND_THREADS) as pool:
        futures = {pool.submit(_send_batch, batch): batch for batch in sqs_batches}
        for future in as_completed(futures):
            s, f = future.result()
            sent += s
            failed += f

    print(
        f"Dispatched {sent} messages ({total_records} records) "
        f"from {csv_s3_uri} — {failed} failures"
    )

    return {
        "messages_sent": sent,
        "record_count": total_records,
    }
