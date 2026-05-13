#!/usr/bin/env python3
"""
GPU-accelerated NuExtract-2.0-8B WARC extraction -- SQS-based worker.

Uses vLLM offline LLM class with xgrammar-constrained JSON output.
SQS-based worker architecture:
  - Model: NuExtract-2.0-8B via vLLM
  - Inference: llm.chat() with batched messages + guided_json
  - Single-pass extraction: template includes is_relevant enum (no separate classification)
  - Visibility timeout: 600s (up from 300s)
  - Non-English articles auto-translated to English via AWS Translate
  - Full untruncated source text stored in output; inference input capped at 40K chars

Architecture:
  - N GPU worker processes (one per GPU on this instance)
  - Each worker independently: polls SQS -> prefetches WARCs -> vLLM inference -> writes S3
  - vLLM continuous batching processes multiple records simultaneously
  - xgrammar constrains output to valid JSON matching DIMEFILED schema
  - SIGTERM handler for spot interruption graceful shutdown

Usage (on EC2, invoked by user-data):
    python gpu_extract.py --sqs-queue-url https://sqs.us-east-1.amazonaws.com/... --num-gpus 1

Local testing:
    python gpu_extract.py --local-test --local-records '[{"url":"...","warc_filename":"...","warc_record_offset":0,"warc_record_length":1000}]'
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION = os.environ.get("AWS_REGION", "us-east-1")
OUTPUT_BUCKET = "sirius-dimefiled-results"
OUTPUT_PREFIX = "nuextract-v2-events/"
PROGRESS_PREFIX = "nuextract-v2-progress/"

PREFETCH_WORKERS = 16          # Threads for WARC fetching per GPU worker
SQS_WAIT_SECONDS = 20          # Long-poll wait time
SQS_VISIBILITY_TIMEOUT = 600   # 10 min per batch (LLM generation slower)
EMPTY_POLLS_BEFORE_EXIT = 3    # Exit after N consecutive empty polls

# vLLM model configuration
MODEL_NAME = "numind/NuExtract-2.0-8B"
MAX_MODEL_LEN = 32768  # Qwen2.5 arch supports 131K; 32K gives headroom for 16K output
GPU_MEMORY_UTILIZATION = 0.90

# Inference chunking: long texts are split into overlapping windows so the model
# sees the entire article.  Results from overlapping chunks are deduplicated.
INFERENCE_CHUNK_SIZE = 40000   # chars per chunk (~10-14K tokens)
INFERENCE_CHUNK_OVERLAP = 4000 # overlap between consecutive chunks

# Graceful shutdown flag
_SHUTDOWN = threading.Event()


def signal_handler(signum, frame):
    """Handle SIGTERM (spot interruption 2-min warning) -- stop polling, finish current batch."""
    print(f"\n[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
    _SHUTDOWN.set()


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ---------------------------------------------------------------------------
# Instance identity
# ---------------------------------------------------------------------------

_IMDS_HOST = "169.254.169.254"


def _imds_request(path: str, method: str = "GET", headers: dict | None = None) -> str:
    """Low-level EC2 IMDS fetch via http.client (link-local, HTTP only)."""
    import http.client
    conn = http.client.HTTPConnection(_IMDS_HOST, timeout=2)
    conn.request(method, path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    return body


def get_instance_id():
    """Get EC2 instance ID from metadata service, or generate a fallback."""
    try:
        token = _imds_request(
            "/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        return _imds_request(
            "/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
    except Exception:
        import socket
        return f"local-{socket.gethostname()}-{os.getpid()}"


# ---------------------------------------------------------------------------
# Prefetch: fetch WARC records in parallel threads
# ---------------------------------------------------------------------------

def prefetch_record(record):
    """Fetch a single WARC record, return (text, title, record) or error tuple."""
    from warc_fetch import fetch_and_normalize
    text, title, error = fetch_and_normalize(record)
    if error:
        return (None, None, record, error)
    return (text, title, record)


def prefetch_batch(records):
    """Prefetch a batch of WARC records in parallel. Returns list of results."""
    results = []
    with ThreadPoolExecutor(max_workers=min(PREFETCH_WORKERS, len(records))) as pool:
        futures = {pool.submit(prefetch_record, r): r for r in records}
        for future in as_completed(futures):
            results.append(future.result())
    return results


# ---------------------------------------------------------------------------
# vLLM model loading
# ---------------------------------------------------------------------------

def load_model(gpu_id):
    """Load model via vLLM offline LLM class on specified GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
    )
    return llm


def build_sampling_params():
    """Build vLLM SamplingParams with structured JSON output constraint."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from nuextract_template import XGRAMMAR_SCHEMA

    return SamplingParams(
        temperature=0,
        max_tokens=16384,
        structured_outputs=StructuredOutputsParams(json=XGRAMMAR_SCHEMA),
    )


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def _split_text_chunks(text):
    """Split text into overlapping chunks for inference.

    Short texts (≤ INFERENCE_CHUNK_SIZE) return a single chunk.
    Longer texts are split on sentence boundaries within the overlap zone
    so chunks don't cut mid-sentence.

    Returns:
        list of text chunks
    """
    if len(text) <= INFERENCE_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + INFERENCE_CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break on a sentence boundary within the last 1000 chars of the chunk
        boundary_zone = text[end - 1000:end]
        # Look for sentence-ending punctuation followed by space
        best_break = -1
        for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
            pos = boundary_zone.rfind(sep)
            if pos > best_break:
                best_break = pos

        if best_break >= 0:
            end = (end - 1000) + best_break + 2  # include the punctuation + space

        chunks.append(text[start:end])
        start = end - INFERENCE_CHUNK_OVERLAP

    return chunks


def _merge_chunk_results(chunk_results):
    """Merge parsed NuExtract outputs from multiple overlapping chunks.

    Deduplicates events whose action text overlaps (from the chunk overlap zone).
    Returns a single merged parsed dict, or None if all chunks failed.
    """
    valid = [r for r in chunk_results if r is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    # Use the first chunk's top-level fields as the base
    merged = dict(valid[0])
    base_events = merged.get("events", [])
    if not isinstance(base_events, list):
        base_events = [base_events] if base_events else []

    # Collect action texts already seen for dedup
    seen_actions = set()
    for evt in base_events:
        for action in (evt.get("actions", []) if isinstance(evt, dict) else []):
            action_text = action.get("text", "") if isinstance(action, dict) else str(action)
            if action_text:
                seen_actions.add(action_text[:200])

    # Merge events from subsequent chunks, skipping duplicates
    for chunk_parsed in valid[1:]:
        chunk_events = chunk_parsed.get("events", [])
        if not isinstance(chunk_events, list):
            chunk_events = [chunk_events] if chunk_events else []

        for evt in chunk_events:
            if not isinstance(evt, dict):
                continue
            # Check if this event's primary action is already seen
            actions = evt.get("actions", [])
            action_texts = []
            for a in (actions if isinstance(actions, list) else []):
                at = a.get("text", "") if isinstance(a, dict) else str(a)
                if at:
                    action_texts.append(at[:200])

            if action_texts and any(at in seen_actions for at in action_texts):
                continue  # Duplicate from overlap zone

            base_events.append(evt)
            seen_actions.update(action_texts)

    merged["events"] = base_events
    return merged


def run_batch_inference(llm, sampling_params, texts_and_records):
    """Run batched vLLM inference on a list of (text, title, record) tuples.

    Long texts are split into overlapping chunks so the model sees the full
    article.  Results from overlapping chunks are merged and deduplicated.
    The full untruncated text is returned as source_text.

    Args:
        llm: vLLM LLM instance
        sampling_params: SamplingParams with guided_json constraint
        texts_and_records: list of (text, title, record) tuples

    Returns:
        list of (parsed_json_or_None, record, title, source_text, error_or_None)
    """
    from nuextract_template import build_messages, NUEXTRACT_TEMPLATE

    # Build messages batch — split long texts into overlapping chunks
    messages_batch = []
    # Track which doc each message belongs to: (doc_index, chunk_index)
    message_map = []

    for doc_idx, (text, title, record) in enumerate(texts_and_records):
        chunks = _split_text_chunks(text)
        for chunk_idx, chunk in enumerate(chunks):
            messages_batch.append(build_messages(chunk))
            message_map.append((doc_idx, chunk_idx, len(chunks)))

    # Run batched inference -- vLLM continuous batching handles parallelism
    try:
        outputs = llm.chat(
            messages_batch,
            sampling_params,
            chat_template_kwargs={"template": NUEXTRACT_TEMPLATE},
        )
    except Exception as e:
        print(f"[INFERENCE ERROR] Batch inference failed: {e}")
        return [(None, r, t, "", str(e)) for _, t, r in texts_and_records]

    # Parse outputs and group by document
    doc_chunk_results = {}  # doc_idx -> list of (chunk_idx, parsed_or_None)
    doc_errors = {}         # doc_idx -> last error string

    for msg_idx, output in enumerate(outputs):
        doc_idx, chunk_idx, num_chunks = message_map[msg_idx]
        if doc_idx not in doc_chunk_results:
            doc_chunk_results[doc_idx] = [None] * num_chunks

        try:
            raw_text = output.outputs[0].text
            parsed = json.loads(raw_text)
            doc_chunk_results[doc_idx][chunk_idx] = parsed
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            doc_errors[doc_idx] = f"parse_error:{e}"

    # Merge chunks per document
    results = []
    for doc_idx, (text, title, record) in enumerate(texts_and_records):
        chunk_results = doc_chunk_results.get(doc_idx, [])

        if all(r is None for r in chunk_results):
            error = doc_errors.get(doc_idx, "parse_error:all_chunks_failed")
            results.append((None, record, title, text, error))
            continue

        merged = _merge_chunk_results(chunk_results)
        results.append((merged, record, title, text, None))

    return results


# ---------------------------------------------------------------------------
# S3 batch writing
# ---------------------------------------------------------------------------

def _write_batch(s3, batch_buffer, batch_label):
    """Write a batch of results to S3."""
    s3_key = f"{OUTPUT_PREFIX}{batch_label}.json"
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=s3_key,
        Body=json.dumps(batch_buffer, ensure_ascii=False),
        ContentType="application/json",
    )
    print(f"[S3] Wrote s3://{OUTPUT_BUCKET}/{s3_key} ({len(batch_buffer)} docs)")


def _write_progress(s3, instance_id, gpu_id, stats):
    """Write per-instance progress to S3."""
    key = f"{PROGRESS_PREFIX}instance-{instance_id}-gpu{gpu_id}.json"
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=key,
        Body=json.dumps(stats, ensure_ascii=False),
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# SQS Worker: one per GPU, fully independent
# ---------------------------------------------------------------------------

def sqs_worker(gpu_id, sqs_queue_url, instance_id, num_gpus):
    """Poll SQS -> prefetch WARCs -> vLLM batch inference -> write S3 -> delete messages."""
    from postprocess import postprocess_nuextract_output, split_to_per_action_events

    print(f"[GPU-{gpu_id}] Loading {MODEL_NAME} on cuda:{gpu_id}...")
    llm = load_model(gpu_id)
    sampling_params = build_sampling_params()
    print(f"[GPU-{gpu_id}] Model loaded, ready for inference")

    sqs = boto3.client("sqs", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    batch_num = 0
    total_processed = 0
    total_relevant = 0
    total_high = 0
    total_medium = 0
    total_errors = 0
    error_types = {}  # error_type -> count
    result_buffer = []
    empty_polls = 0
    start_time = time.time()

    while not _SHUTDOWN.is_set():
        # Poll SQS for up to 10 messages (long poll)
        try:
            resp = sqs.receive_message(
                QueueUrl=sqs_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=SQS_WAIT_SECONDS,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
            )
        except Exception as e:
            print(f"[GPU-{gpu_id}] SQS poll error: {e}")
            time.sleep(5)
            continue

        messages = resp.get("Messages", [])
        if not messages:
            empty_polls += 1
            print(f"[GPU-{gpu_id}] Empty poll ({empty_polls}/{EMPTY_POLLS_BEFORE_EXIT})")
            if empty_polls >= EMPTY_POLLS_BEFORE_EXIT:
                print(f"[GPU-{gpu_id}] Queue empty, exiting")
                break
            continue

        empty_polls = 0

        # Parse all records from messages (each message contains up to 10 records)
        records = []
        for msg in messages:
            try:
                records.extend(json.loads(msg["Body"]))
            except Exception as e:
                print(f"[GPU-{gpu_id}] Bad message body: {e}")

        # Prefetch all WARCs in parallel
        prefetched = prefetch_batch(records)

        # Separate successful fetches from errors
        valid_items = []
        for item in prefetched:
            if _SHUTDOWN.is_set():
                break
            if len(item) == 4:
                # Prefetch error: (None, None, record, error_msg)
                total_errors += 1
                total_processed += 1
                etype = item[3].split(":")[0] if isinstance(item[3], str) else "unknown"
                error_types[etype] = error_types.get(etype, 0) + 1
            else:
                valid_items.append(item)  # (text, title, record)

        # Run batched vLLM inference on all valid items
        if valid_items and not _SHUTDOWN.is_set():
            inference_results = run_batch_inference(llm, sampling_params, valid_items)

            for parsed, record, title, source_text, error in inference_results:
                total_processed += 1

                if error:
                    total_errors += 1
                    etype = error.split(":")[0] if isinstance(error, str) else "unknown"
                    error_types[etype] = error_types.get(etype, 0) + 1
                    continue

                # Postprocess into v2 format
                url = record.get("url", "")
                crawl_date = record.get("crawl_date", "")
                event = postprocess_nuextract_output(
                    parsed, url=url, crawl_date=crawl_date, title=title or "",
                )

                if event is None:
                    # Not relevant
                    continue

                # Split aggregated event into per-action events
                per_action_events = split_to_per_action_events(event)

                # Filter to HIGH/MEDIUM quality events
                quality_events = [e for e in per_action_events
                                  if e.get("quality") in ("HIGH", "MEDIUM")]
                if not quality_events:
                    continue

                total_relevant += len(quality_events)
                total_high += sum(1 for e in quality_events if e.get("quality") == "HIGH")
                total_medium += sum(1 for e in quality_events if e.get("quality") == "MEDIUM")

                # Wrap in doc-level structure matching pipeline schema
                doc_result = {
                    "url": url,
                    "crawl_date": crawl_date,
                    "title": title,
                    "source_text": source_text or "",
                    "relevant": True,
                    "events": quality_events,
                    "quality_metrics": {
                        "total_events": len(quality_events),
                        "high_quality_events": sum(1 for e in quality_events if e.get("quality") == "HIGH"),
                        "medium_quality_events": sum(1 for e in quality_events if e.get("quality") == "MEDIUM"),
                        "quality_ratio": sum(1 for e in quality_events if e.get("quality") == "HIGH") / max(len(quality_events), 1),
                    },
                }
                result_buffer.append(doc_result)

        # Write results to S3 after every batch (no minimum threshold)
        if result_buffer:
            batch_label = f"gpu_{instance_id}_{gpu_id}_{batch_num:06d}"
            _write_batch(s3, result_buffer, batch_label)
            batch_num += 1
            result_buffer = []

        # Delete all messages from SQS (they've been processed)
        try:
            entries = [
                {"Id": str(i), "ReceiptHandle": m["ReceiptHandle"]}
                for i, m in enumerate(messages)
            ]
            sqs.delete_message_batch(QueueUrl=sqs_queue_url, Entries=entries)
        except Exception as e:
            print(f"[GPU-{gpu_id}] SQS delete error: {e}")

        # Periodic logging + progress
        if total_processed % 100 < len(records):
            elapsed = time.time() - start_time
            rps = total_processed / max(elapsed, 1)
            print(
                f"[GPU-{gpu_id}] {total_processed:,} processed | "
                f"relevant={total_relevant:,} HIGH={total_high:,} MED={total_medium:,} "
                f"err={total_errors:,} | {rps:.1f} rec/s"
            )
            gc.collect()

            _write_progress(s3, instance_id, gpu_id, {
                "instance_id": instance_id,
                "gpu_id": gpu_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "total_processed": total_processed,
                "total_relevant": total_relevant,
                "total_high": total_high,
                "total_medium": total_medium,
                "total_errors": total_errors,
                "error_types": error_types,
                "batches_written": batch_num,
                "elapsed_seconds": elapsed,
                "records_per_sec": rps,
            })

    # Flush remaining results
    if result_buffer:
        batch_label = f"gpu_{instance_id}_{gpu_id}_{batch_num:06d}"
        _write_batch(s3, result_buffer, batch_label)
        batch_num += 1

    # Final progress
    elapsed = time.time() - start_time
    rps = total_processed / max(elapsed, 1)
    _write_progress(s3, instance_id, gpu_id, {
        "instance_id": instance_id,
        "gpu_id": gpu_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_processed": total_processed,
        "total_relevant": total_relevant,
        "total_high": total_high,
        "total_medium": total_medium,
        "total_errors": total_errors,
        "error_types": error_types,
        "batches_written": batch_num,
        "elapsed_seconds": elapsed,
        "records_per_sec": rps,
        "status": "shutdown" if _SHUTDOWN.is_set() else "complete",
    })

    print(
        f"[GPU-{gpu_id}] Finished -- {total_processed:,} processed, "
        f"{total_relevant:,} relevant, {batch_num} batches, "
        f"{rps:.1f} rec/s over {elapsed/3600:.2f}h"
    )


# ---------------------------------------------------------------------------
# Local test mode
# ---------------------------------------------------------------------------

def local_test(records_json):
    """Run a local test on a small batch of records without SQS."""
    from postprocess import postprocess_nuextract_output

    records = json.loads(records_json) if isinstance(records_json, str) else records_json
    print(f"[LOCAL TEST] Processing {len(records)} records...")

    # Load model on GPU 0
    llm = load_model(0)
    sampling_params = build_sampling_params()

    # Prefetch WARCs
    print("[LOCAL TEST] Fetching WARC records...")
    prefetched = prefetch_batch(records)

    valid_items = []
    for item in prefetched:
        if len(item) == 4:
            _, _, record, error = item
            print(f"  [FETCH ERROR] {record.get('url', '?')[:60]}: {error}")
        else:
            text, title, record = item
            print(f"  [FETCHED] {record.get('url', '?')[:60]} len={len(text)} title={str(title)[:40]}")
            valid_items.append(item)

    if not valid_items:
        print("[LOCAL TEST] No records fetched successfully")
        return

    # Run inference
    print(f"\n[LOCAL TEST] Running vLLM inference on {len(valid_items)} records...")
    inference_results = run_batch_inference(llm, sampling_params, valid_items)

    # Postprocess and display
    for parsed, record, title, source_text, error in inference_results:
        url = record.get("url", "?")
        if error:
            print(f"\n  [ERROR] {url[:60]}: {error}")
            continue

        event = postprocess_nuextract_output(
            parsed,
            url=url,
            crawl_date=record.get("crawl_date", ""),
            title=title or "",
        )

        if event is None:
            print(f"\n  [NOT RELEVANT] {url[:60]}")
            continue

        print(f"\n  [RESULT] {url[:60]}")
        print(f"    Domain:      {event.get('l1_domain')}")
        print(f"    Subdomain:   {event.get('l2_subdomain')}")
        print(f"    Escalation:  {event.get('escalation_classification')}")
        print(f"    Scope:       {event.get('scope')}")
        print(f"    Claim:       {event.get('claim_type')}")
        print(f"    Initiators:  {event.get('initiators')}")
        print(f"    Targets:     {event.get('targets')}")
        print(f"    Actions:     {len(event.get('actions_flat', []))}")
        print(f"    Locations:   {event.get('locations_flat')}")
        print(f"    Edges:       {event.get('edge_count')}")
        print(f"    Patterns:    {event.get('influence_patterns', [])}")
        print(f"    Contestation:{event.get('contestation_dynamics', '')}")
        print(f"    Quality:     {event.get('quality')}")

    print("\n[LOCAL TEST] Done")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU NuExtract-2.0-8B extraction (SQS mode)")
    parser.add_argument("--sqs-queue-url", default=None, help="SQS queue URL to poll")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--local-test", action="store_true", help="Run local test mode")
    parser.add_argument("--local-records", default=None, help="JSON string of records for local test")
    args = parser.parse_args()

    if args.local_test:
        if not args.local_records:
            print("[ERROR] --local-records required for --local-test mode")
            sys.exit(1)
        local_test(args.local_records)
        return

    sqs_queue_url = args.sqs_queue_url or os.environ.get("SQS_QUEUE_URL")
    if not sqs_queue_url:
        print("[ERROR] --sqs-queue-url or SQS_QUEUE_URL env var required")
        sys.exit(1)
    args.sqs_queue_url = sqs_queue_url

    import torch
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    instance_id = get_instance_id()

    print("=" * 70)
    print("NuExtract-2.0-8B GPU Extraction (SQS Mode)")
    print("=" * 70)
    print(f"  Instance:    {instance_id}")
    print(f"  GPUs:        {args.num_gpus}")
    print(f"  SQS queue:   {args.sqs_queue_url}")
    print(f"  Output:      s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Max seq len: {MAX_MODEL_LEN}")
    print(f"  GPU mem:     {GPU_MEMORY_UTILIZATION}")

    import torch as _torch
    print(f"  CUDA avail:  {_torch.cuda.is_available()}")
    if _torch.cuda.is_available():
        print(f"  GPU count:   {_torch.cuda.device_count()}")
        for i in range(_torch.cuda.device_count()):
            print(f"  GPU-{i}:       {_torch.cuda.get_device_name(i)}")
    print("=" * 70)

    # Validate GPU count
    available_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    if args.num_gpus > available_gpus:
        print(f"[ERROR] Requested {args.num_gpus} GPUs but only {available_gpus} available")
        sys.exit(1)

    # Launch one worker process per GPU
    workers = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=sqs_worker,
            args=(gpu_id, args.sqs_queue_url, instance_id, args.num_gpus),
        )
        p.start()
        workers.append(p)
        print(f"[MAIN] Started SQS worker GPU-{gpu_id} (pid={p.pid})")

    # Wait for all workers to finish
    for p in workers:
        p.join()
        print(f"[MAIN] Worker pid={p.pid} exited with code {p.exitcode}")

    print()
    print("=" * 70)
    print("All GPU workers finished -- instance ready for self-termination")
    print("=" * 70)


if __name__ == "__main__":
    main()
