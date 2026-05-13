"""Batch job: Strategic Analysis — counterfactual, implicit, and Faustian inference.

For each cluster with sufficient signal density, this step:
  1. Aggregates decision_context, implicit_signals, and trade_off_signals
     across all constituent raw_events
  2. Calls Bedrock LLM to triangulate unstated objectives, construct
     counterfactual decision branches, and identify Faustian trades
  3. Writes results to the strategic_analysis LanceDB table

The LLM does NOT invent signals — it reasons over the per-article signals
that NuExtract already extracted. Its job is cross-event triangulation:
signals that appear individually ambiguous may become clear in aggregate.

Usage (as Batch job):
    python handler.py

Environment:
    LANCEDB_URI         — LanceDB S3 URI
    BEDROCK_MODEL_ID    — Bedrock model for strategic reasoning
    AWS_REGION          — AWS region for Bedrock
    MIN_SIGNALS         — Minimum non-empty signal fields to process a cluster (default: 3)
"""

import json
import os
import sys
import uuid
from collections import Counter, defaultdict

import boto3
import lancedb

# Add parent path for shared.db
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingest_normalize"))
from shared import db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANCEDB_URI = os.environ.get("LANCEDB_URI", "s3://sirius-dimefiled-results/hypergraph/")
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MIN_SIGNALS = int(os.environ.get("MIN_SIGNALS", "3"))
S3_BUCKET = os.environ.get("S3_BUCKET", "sirius-dimefiled-results")
S3_RESULTS_PREFIX = os.environ.get("S3_RESULTS_PREFIX", "strategic-analysis/")

# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------

def _merge_list_dicts(dicts: list[dict]) -> dict:
    """Merge a list of dicts-of-lists, deduplicating values."""
    merged: dict[str, list] = defaultdict(list)
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if isinstance(v, list):
                for item in v:
                    if item and isinstance(item, str) and item.strip():
                        merged[k].append(item.strip())
            elif isinstance(v, str) and v.strip():
                merged[k].append(v.strip())
    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in merged.items()}


def _signal_density(agg: dict) -> int:
    """Count non-empty fields across an aggregated signal dict."""
    count = 0
    for v in agg.values():
        if isinstance(v, list) and v:
            count += 1
        elif isinstance(v, str) and v.strip():
            count += 1
    return count


def aggregate_cluster_signals(
    cluster_id: str,
    cluster_event_ids: set[str],
    events_df,
) -> dict | None:
    """Aggregate counterfactual/implicit/trade-off signals for a cluster.

    Returns None if the cluster has insufficient signal density.
    """
    decision_dicts = []
    implicit_dicts = []
    trade_off_dicts = []

    for _, row in events_df.iterrows():
        if row["event_id"] not in cluster_event_ids:
            continue

        for col, target in [
            ("decision_context", decision_dicts),
            ("implicit_signals", implicit_dicts),
            ("trade_off_signals", trade_off_dicts),
        ]:
            raw = row.get(col, "{}")
            if not raw or (isinstance(raw, str) and not raw.strip()):
                continue
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                target.append(parsed)

    dc_agg = _merge_list_dicts(decision_dicts)
    imp_agg = _merge_list_dicts(implicit_dicts)
    tos_agg = _merge_list_dicts(trade_off_dicts)

    total_density = _signal_density(dc_agg) + _signal_density(imp_agg) + _signal_density(tos_agg)
    if total_density < MIN_SIGNALS:
        return None

    return {
        "decision_context": dc_agg,
        "implicit_signals": imp_agg,
        "trade_off_signals": tos_agg,
        "signal_density": total_density,
    }


# ---------------------------------------------------------------------------
# LLM strategic analysis
# ---------------------------------------------------------------------------

STRATEGIC_ANALYSIS_SYSTEM = """\
You are a strategic analyst specializing in gray zone geopolitical operations.

You are given aggregated signals from multiple news articles about events in a \
spatio-temporal cluster. These signals were extracted by NuExtract from individual articles. \
Your job is to TRIANGULATE across them — signals that seem ambiguous in isolation \
may reveal clear strategic patterns when viewed together.

You must reason ONLY from the provided signals. Do not invent signals not present \
in the input. Your analysis should be evidence-based, citing which signals support \
each inference.

Use the provide_analysis tool to return your structured analysis."""

STRATEGIC_ANALYSIS_TOOL = {
    "name": "provide_analysis",
    "description": "Return structured strategic analysis for a cluster of events.",
    "input_schema": {
        "type": "object",
        "properties": {
            "unstated_objectives": {
                "type": "array",
                "description": "Inferred unstated strategic objectives, triangulated from hints across events",
                "items": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string", "description": "The inferred unstated objective"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Which specific signals support this inference",
                        },
                    },
                    "required": ["objective", "confidence", "evidence"],
                },
            },
            "counterfactual_branches": {
                "type": "array",
                "description": "Alternative decision paths not taken, inferred from constraints and alternatives mentioned",
                "items": {
                    "type": "object",
                    "properties": {
                        "decision_point": {"type": "string", "description": "What choice was made"},
                        "alternative_action": {"type": "string", "description": "What could have been done instead"},
                        "projected_outcome": {"type": "string", "description": "Likely consequence of the alternative"},
                        "why_not_taken": {"type": "string", "description": "Inferred reason this path was rejected"},
                    },
                    "required": ["decision_point", "alternative_action", "projected_outcome"],
                },
            },
            "faustian_trades": {
                "type": "array",
                "description": "Short-term gains exchanged for long-term costs or risks",
                "items": {
                    "type": "object",
                    "properties": {
                        "short_term_gain": {"type": "string"},
                        "long_term_cost": {"type": "string"},
                        "actors_affected": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reversibility": {
                            "type": "string",
                            "enum": ["permanent", "durable", "fragile", "temporary"],
                        },
                    },
                    "required": ["short_term_gain", "long_term_cost"],
                },
            },
            "strategic_narrative": {
                "type": "string",
                "description": "2-4 sentence synthesis of the strategic picture: what is actually happening beneath the surface events",
            },
            "decision_tree": {
                "type": "object",
                "description": "Structured decision tree with root action and branches",
                "properties": {
                    "root_action": {"type": "string"},
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition": {"type": "string"},
                                "outcome": {"type": "string"},
                                "probability": {"type": "string", "enum": ["likely", "possible", "unlikely"]},
                            },
                            "required": ["condition", "outcome"],
                        },
                    },
                },
            },
            "confidence": {
                "type": "number",
                "description": "Overall confidence 0.0-1.0 based on signal density and consistency",
            },
        },
        "required": [
            "unstated_objectives", "counterfactual_branches",
            "faustian_trades", "strategic_narrative", "confidence",
        ],
    },
}


def analyze_cluster(
    cluster_id: str,
    cluster_meta: dict,
    signals: dict,
    bedrock_client,
) -> dict:
    """Call Bedrock to produce strategic analysis for one cluster."""

    user_msg = json.dumps({
        "cluster_id": cluster_id,
        "location": cluster_meta.get("location", ""),
        "timeframe": f"{cluster_meta.get('timeframe_start', '')} to {cluster_meta.get('timeframe_end', '')}",
        "l1_domain": cluster_meta.get("l1_domain", ""),
        "escalation": cluster_meta.get("escalation", ""),
        "signal_density": signals["signal_density"],
        "decision_context": signals["decision_context"],
        "implicit_signals": signals["implicit_signals"],
        "trade_off_signals": signals["trade_off_signals"],
    }, indent=2)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "system": STRATEGIC_ANALYSIS_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
        "tools": [STRATEGIC_ANALYSIS_TOOL],
        "tool_choice": {"type": "tool", "name": "provide_analysis"},
    })

    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())

    # Extract tool use result
    for block in result.get("content", []):
        if block.get("type") == "tool_use":
            return block.get("input", {})

    return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run():
    """Run strategic analysis across all clusters."""
    print("=" * 60)
    print("Strategic Analysis: Counterfactual / Implicit / Faustian")
    print("=" * 60)

    conn = lancedb.connect(LANCEDB_URI)

    # Load clusters
    clusters_df = conn.open_table("clusters").to_pandas()
    print(f"  Loaded {len(clusters_df)} clusters")

    # Load raw_events (only the columns we need)
    raw_table = conn.open_table("raw_events")
    raw_ds = raw_table.to_lance()
    events_df = raw_ds.scanner(columns=[
        "event_id", "url",
        "decision_context", "implicit_signals", "trade_off_signals",
    ]).to_table().to_pandas()
    print(f"  Loaded {len(events_df)} raw events")

    # Load relations to map cluster → event_ids
    rel_table = conn.open_table("relations")
    rel_ds = rel_table.to_lance()
    rels_df = rel_ds.scanner(columns=[
        "source_event_id", "cluster_id",
    ]).to_table().to_pandas()

    # Build cluster → event_id set
    cluster_events: dict[str, set[str]] = defaultdict(set)
    for _, rel in rels_df.iterrows():
        cid = rel.get("cluster_id", "")
        eid = rel.get("source_event_id", "")
        if cid and eid:
            cluster_events[cid].add(eid)

    print(f"  Mapped {len(cluster_events)} clusters to event IDs")

    # Process each cluster
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    results = []
    skipped = 0
    errors = 0

    for _, cluster in clusters_df.iterrows():
        cluster_id = cluster["cluster_id"]
        depth = int(cluster.get("depth", 0))

        # Only process Knox root clusters (depth=0) to avoid redundant analysis
        if depth != 0:
            continue

        event_ids = cluster_events.get(cluster_id, set())
        if not event_ids:
            skipped += 1
            continue

        # Aggregate signals
        signals = aggregate_cluster_signals(cluster_id, event_ids, events_df)
        if signals is None:
            skipped += 1
            continue

        cluster_meta = {
            "location": str(cluster.get("location", "")),
            "timeframe_start": str(cluster.get("timeframe_start", "")),
            "timeframe_end": str(cluster.get("timeframe_end", "")),
            "l1_domain": str(cluster.get("l1_domain", "")),
            "escalation": str(cluster.get("escalation", "")),
        }

        print(f"  Analyzing cluster {cluster_id[:12]}... "
              f"(density={signals['signal_density']}, events={len(event_ids)})")

        try:
            analysis = analyze_cluster(cluster_id, cluster_meta, signals, bedrock)

            row = {
                "analysis_id": str(uuid.uuid4()),
                "cluster_id": cluster_id,
                "decision_context_agg": json.dumps(signals["decision_context"]),
                "implicit_signals_agg": json.dumps(signals["implicit_signals"]),
                "trade_off_signals_agg": json.dumps(signals["trade_off_signals"]),
                "unstated_objectives": json.dumps(analysis.get("unstated_objectives", [])),
                "counterfactual_branches": json.dumps(analysis.get("counterfactual_branches", [])),
                "faustian_trades": json.dumps(analysis.get("faustian_trades", [])),
                "strategic_narrative": analysis.get("strategic_narrative", ""),
                "decision_tree": json.dumps(analysis.get("decision_tree", {})),
                "confidence": float(analysis.get("confidence", 0.0)),
                "model_id": BEDROCK_MODEL_ID,
            }
            results.append(row)
            print(f"    -> {len(analysis.get('unstated_objectives', []))} objectives, "
                  f"{len(analysis.get('counterfactual_branches', []))} branches, "
                  f"{len(analysis.get('faustian_trades', []))} trades")

        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1
            continue

    # Write results to LanceDB
    if results:
        db.get_or_create_table(conn, "strategic_analysis", data=results)
        print(f"\n  Wrote {len(results)} strategic analyses to LanceDB")
    else:
        print("\n  No analyses produced")

    # Also write results as JSON to S3 for downstream consumption
    if results:
        s3 = boto3.client("s3")
        s3_key = f"{S3_RESULTS_PREFIX}strategic_analysis.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(results, indent=2),
            ContentType="application/json",
        )
        print(f"  Uploaded to s3://{S3_BUCKET}/{s3_key}")

    print(f"\n  Strategic Analysis complete:")
    print(f"    Analyzed: {len(results)} clusters")
    print(f"    Skipped:  {skipped} (insufficient signals)")
    print(f"    Errors:   {errors}")
    print("=" * 60)

    return {
        "analyzed": len(results),
        "skipped": skipped,
        "errors": errors,
    }


if __name__ == "__main__":
    run()
