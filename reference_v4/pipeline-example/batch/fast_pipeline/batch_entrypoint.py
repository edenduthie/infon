#!/usr/bin/env python3
"""AWS Batch entrypoint for the fast DuckDB pipeline.

Downloads NuExtract JSON results from S3, runs the optimized local_pipeline.py
(Stages 0-7), and uploads the resulting DuckDB database + XLSX workbook back to S3.

Environment Variables:
    NUEXTRACT_S3_BUCKET  — S3 bucket containing NuExtract results
    NUEXTRACT_S3_PREFIX  — S3 key prefix for JSON files (default: nuextract-v2-events/)
    OUTPUT_S3_PREFIX     — S3 key prefix for pipeline outputs (default: pipeline-output/)
    AWS_REGION           — AWS region (default: us-east-1)
"""

import os
import sys
import tempfile
import time

import boto3

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("NUEXTRACT_S3_BUCKET", "sirius-dimefiled-results")
S3_PREFIX = os.environ.get("NUEXTRACT_S3_PREFIX", "nuextract-v2-events/")
OUTPUT_PREFIX = os.environ.get("OUTPUT_S3_PREFIX", "pipeline-output/")
REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=REGION)


def download_json_from_s3(local_dir: str) -> int:
    """Download all JSON files from S3 prefix to local directory.

    Returns the number of files downloaded.
    """
    print(f"Downloading JSON from s3://{S3_BUCKET}/{S3_PREFIX} ...")
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)
            s3.download_file(S3_BUCKET, key, local_path)
            count += 1
            if count % 1000 == 0:
                print(f"  Downloaded {count:,} files ...", flush=True)
    print(f"  Total: {count:,} JSON files downloaded.")
    return count


def upload_outputs(db_path: str, xlsx_path: str):
    """Upload DuckDB database and XLSX workbook to S3."""
    print(f"\nUploading outputs to s3://{S3_BUCKET}/{OUTPUT_PREFIX} ...")

    if os.path.exists(db_path):
        key = f"{OUTPUT_PREFIX}pipeline_output.duckdb"
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"  Uploading {key} ({size_mb:.1f} MB)")
        s3.upload_file(db_path, S3_BUCKET, key)

    if os.path.exists(xlsx_path):
        key = f"{OUTPUT_PREFIX}PipelineOutput.xlsx"
        size_mb = os.path.getsize(xlsx_path) / (1024 * 1024)
        print(f"  Uploading {key} ({size_mb:.1f} MB)")
        s3.upload_file(xlsx_path, S3_BUCKET, key)

    print("  Upload complete.")


def main():
    t_start = time.time()
    print("=" * 70)
    print("SIRIUS Fast Pipeline — AWS Batch Entrypoint")
    print("=" * 70)
    print(f"  Bucket:  {S3_BUCKET}")
    print(f"  Prefix:  {S3_PREFIX}")
    print(f"  Output:  {OUTPUT_PREFIX}")
    print(f"  Region:  {REGION}")

    # Create temp directories for input and output
    work_dir = tempfile.mkdtemp(prefix="sirius_pipeline_")
    raw_dir = os.path.join(work_dir, "raw")
    os.makedirs(raw_dir)
    db_path = os.path.join(work_dir, "pipeline_output.duckdb")
    xlsx_path = os.path.join(work_dir, "PipelineOutput.xlsx")

    # Step 1: Download JSON from S3
    t0 = time.time()
    n_files = download_json_from_s3(raw_dir)
    print(f"  Download took {time.time() - t0:.1f}s")

    if n_files == 0:
        print("ERROR: No JSON files found in S3. Aborting.")
        sys.exit(1)

    # Step 2: Run the pipeline
    # Import local_pipeline (packaged alongside this file in the container)
    import duckdb
    import local_pipeline

    print(f"\nStarting pipeline with {n_files:,} JSON files ...")
    conn = duckdb.connect(db_path)

    local_pipeline.stage0_ingest(raw_dir, conn)
    alias_index = local_pipeline.stage1_entities(conn)
    alias_index = local_pipeline.stage1b_refine_entities(conn, alias_index)
    local_pipeline.stage2_relations(conn, alias_index)
    local_pipeline.stage3_temporal(conn)
    local_pipeline.stage4_dedup_scoring(conn)
    local_pipeline.stage4b_validate_dedup(conn)
    local_pipeline.stage5_regimes(conn)
    local_pipeline.stage5b_validate_regimes(conn)
    local_pipeline.stage5c_validate_crises(conn)
    local_pipeline.stage6_situations(conn)
    local_pipeline.stage6_validate_situations(conn)
    local_pipeline.stage6b_cross_links(conn)
    local_pipeline.stage8_topology(conn)
    local_pipeline.stage8b_sheaf_cohomology(conn)
    local_pipeline.stage8c_persistence_tree(conn)
    local_pipeline.stage9_hmss_tree(conn)
    local_pipeline.stage7_xlsx_export(conn, output_path=xlsx_path)

    # Final stats
    print("\nTable row counts:")
    _TABLE_COUNT_QUERIES = {
        "articles":          "SELECT COUNT(*) FROM articles",
        "events":            "SELECT COUNT(*) FROM events",
        "actions":           "SELECT COUNT(*) FROM actions",
        "locations":         "SELECT COUNT(*) FROM locations",
        "entities":          "SELECT COUNT(*) FROM entities",
        "entity_hierarchy":  "SELECT COUNT(*) FROM entity_hierarchy",
        "entity_merge_map":  "SELECT COUNT(*) FROM entity_merge_map",
        "relations":         "SELECT COUNT(*) FROM relations",
        "relation_event_map":"SELECT COUNT(*) FROM relation_event_map",
        "deduped_events":    "SELECT COUNT(*) FROM deduped_events",
        "mdus_scores":       "SELECT COUNT(*) FROM mdus_scores",
        "regimes":           "SELECT COUNT(*) FROM regimes",
        "crises":            "SELECT COUNT(*) FROM crises",
        "situations":        "SELECT COUNT(*) FROM situations",
        "cross_pair_links":  "SELECT COUNT(*) FROM cross_pair_links",
        "theaters":          "SELECT COUNT(*) FROM theaters",
        "topology_summary":  "SELECT COUNT(*) FROM topology_summary",
        "persistent_features":"SELECT COUNT(*) FROM persistent_features",
        "commutation_scores":"SELECT COUNT(*) FROM commutation_scores",
        "topological_cycles":"SELECT COUNT(*) FROM topological_cycles",
        "sheaf_cohomology":  "SELECT COUNT(*) FROM sheaf_cohomology",
        "sheaf_vertex_scores":"SELECT COUNT(*) FROM sheaf_vertex_scores",
        "sheaf_obstructions":"SELECT COUNT(*) FROM sheaf_obstructions",
        "natural_transformations":"SELECT COUNT(*) FROM natural_transformations",
        "merge_tree_clusters":"SELECT COUNT(*) FROM merge_tree_clusters",
        "merge_tree_summary":"SELECT COUNT(*) FROM merge_tree_summary",
        "hmss_tree":         "SELECT COUNT(*) FROM hmss_tree",
        "hmss_event_assignment":"SELECT COUNT(*) FROM hmss_event_assignment",
        "hmss_summary":      "SELECT COUNT(*) FROM hmss_summary",
    }
    for table, query in _TABLE_COUNT_QUERIES.items():
        try:
            n = conn.execute(query).fetchone()[0]
            print(f"  {table:<25} {n:>10,}")
        except Exception:
            print(f"  {table:<25} MISSING")

    conn.close()

    # Step 3: Upload results to S3
    upload_outputs(db_path, xlsx_path)

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"BATCH JOB COMPLETE: {t_total:.1f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
