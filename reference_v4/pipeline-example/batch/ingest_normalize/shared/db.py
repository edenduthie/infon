"""LanceDB connection helper and table schema definitions.

All pipeline steps read/write LanceDB tables on S3:
  s3://sirius-dimefiled-results/hypergraph/

The NuExtract extraction step writes results as JSON batch files to:
  s3://sirius-dimefiled-results/nuextract-v2-events/

Tables:
  raw_events  — ingested from NuExtract v2 extraction output (flat schema)
  entities    — canonical entity registry (built by LLM agent)
  relations   — directed edges with canonical entity IDs
  clusters    — Knox spatio-temporal clusters
  hyperedges  — denoised directed hyperedges (final graph)
"""

import os

import lancedb
import pyarrow as pa

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

LANCEDB_URI = os.environ.get(
    "LANCEDB_URI", "s3://sirius-dimefiled-results/hypergraph/"
)

# S3 bucket/prefix for NuExtract extraction results (JSON batch files)
NUEXTRACT_S3_BUCKET = os.environ.get("NUEXTRACT_S3_BUCKET", "sirius-dimefiled-results")
NUEXTRACT_S3_PREFIX = os.environ.get("NUEXTRACT_S3_PREFIX", "nuextract-v2-events/")


def connect() -> lancedb.DBConnection:
    """Return a LanceDB connection (S3-backed by default)."""
    return lancedb.connect(LANCEDB_URI)


# ---------------------------------------------------------------------------
# Schemas (PyArrow)
# ---------------------------------------------------------------------------

RAW_EVENTS_SCHEMA = pa.schema([
    pa.field("event_id", pa.utf8()),
    pa.field("url", pa.utf8()),
    pa.field("crawl_date", pa.utf8()),
    pa.field("title", pa.utf8()),
    pa.field("text", pa.utf8()),
    # Classifications (from NuExtract v2)
    pa.field("l1_domain", pa.utf8()),
    pa.field("l2_subdomain", pa.utf8()),
    pa.field("escalation", pa.utf8()),
    pa.field("scope", pa.utf8()),
    pa.field("claim_type", pa.utf8()),
    pa.field("dominant_action_type", pa.utf8()),
    # Directed hyperedge structure: initiators → action → targets
    pa.field("initiators", pa.list_(pa.utf8())),
    pa.field("targets", pa.list_(pa.utf8())),
    # All actors (flat + typed)
    pa.field("actors_flat", pa.list_(pa.utf8())),
    pa.field("actors_typed", pa.utf8()),    # JSON: {nation_states:[], military_orgs:[], ...}
    # Actions
    pa.field("actions_flat", pa.list_(pa.utf8())),
    pa.field("actions_typed", pa.utf8()),   # JSON: [{type, text}, ...]
    # Locations
    pa.field("locations_flat", pa.list_(pa.utf8())),
    # Temporal
    pa.field("dates", pa.utf8()),           # JSON: [{raw, iso}, ...]
    # Directed edges from NuExtract relation extraction
    pa.field("edges", pa.utf8()),           # JSON: [{source, target, relation_type}, ...]
    pa.field("edge_count", pa.int32()),
    # Quality & metadata
    pa.field("quality", pa.utf8()),
    pa.field("assets", pa.utf8()),          # JSON: [vessel names, ...]
    # Type system fields (from SYSTEM_SPECIFICATION.md Section 3)
    pa.field("state_indicators", pa.utf8()),     # JSON: {actor_postures, regional_tension, state_descriptions}
    pa.field("cost_profile", pa.utf8()),         # JSON: {resource_types, max_cost_magnitude, max_risk_level, reversibility}
    pa.field("asymmetry_profile", pa.utf8()),    # JSON: {surprise, ambiguity, cross_domain_linkages, constancy}
    pa.field("situation_framing", pa.utf8()),    # JSON: list of situation framings
    pa.field("actor_roles", pa.utf8()),          # JSON: {initiator_roles, target_roles}
    pa.field("influence_patterns", pa.utf8()),   # JSON: list of influence pattern tags
    pa.field("contestation_dynamics", pa.utf8()),
    pa.field("narrative_framing", pa.utf8()),    # JSON: list of narrative framings
    pa.field("situation_signals", pa.utf8()),    # JSON: {saving_face_indicators, justice_penalty_indicators, historical_references}
    # Counterfactual / Implicit / Trade-off analysis signals
    pa.field("decision_context", pa.utf8()),      # JSON: {alternatives_mentioned, constraints_cited, preconditions, ...}
    pa.field("implicit_signals", pa.utf8()),      # JSON: {stated_justifications, unstated_objective_hints, denials, ...}
    pa.field("trade_off_signals", pa.utf8()),     # JSON: {benefits_claimed, costs_acknowledged, costs_omitted_hints, ...}
])

ENTITIES_SCHEMA = pa.schema([
    pa.field("entity_id", pa.utf8()),
    pa.field("canonical_name", pa.utf8()),
    pa.field("short_name", pa.utf8()),
    pa.field("entity_type", pa.utf8()),
    pa.field("aliases", pa.list_(pa.utf8())),
    pa.field("parent_entity_id", pa.utf8()),
    pa.field("metadata", pa.utf8()),       # JSON-encoded dict
    pa.field("vector", pa.list_(pa.float32(), 256)),
])

RELATIONS_SCHEMA = pa.schema([
    pa.field("relation_id", pa.utf8()),
    pa.field("source_entity_id", pa.utf8()),
    pa.field("target_entity_id", pa.utf8()),
    pa.field("relation_type", pa.utf8()),
    pa.field("action_text", pa.utf8()),
    pa.field("action_type", pa.utf8()),
    pa.field("location", pa.utf8()),
    pa.field("timestamp_iso", pa.utf8()),
    pa.field("source_event_id", pa.utf8()),
    pa.field("source_url", pa.utf8()),
    pa.field("cluster_id", pa.utf8()),
    pa.field("cross_event_type", pa.utf8()),
])

CLUSTERS_SCHEMA = pa.schema([
    pa.field("cluster_id", pa.utf8()),
    pa.field("parent_cluster_id", pa.utf8()),    # "" for Knox root clusters
    pa.field("depth", pa.int32()),               # 0=Knox, 1=location, 2=hourly
    pa.field("event_name", pa.utf8()),
    pa.field("timeframe_start", pa.utf8()),
    pa.field("timeframe_end", pa.utf8()),
    pa.field("location", pa.utf8()),
    pa.field("actor_perspectives", pa.utf8()),   # JSON-encoded dict
    pa.field("article_urls", pa.list_(pa.utf8())),
    pa.field("l1_domain", pa.utf8()),
    pa.field("l2_subdomain", pa.utf8()),
    pa.field("escalation", pa.utf8()),
    pa.field("relation_ids", pa.list_(pa.utf8())),
    pa.field("metadata", pa.utf8()),             # JSON-encoded dict
])

HYPEREDGES_SCHEMA = pa.schema([
    pa.field("hyperedge_id", pa.utf8()),
    pa.field("cluster_id", pa.utf8()),
    pa.field("initiators", pa.list_(pa.utf8())),
    pa.field("action_summary", pa.utf8()),
    pa.field("action_type", pa.utf8()),
    pa.field("targets", pa.list_(pa.utf8())),
    pa.field("location", pa.utf8()),
    pa.field("timestamp_iso", pa.utf8()),
    pa.field("l1_domain", pa.utf8()),
    pa.field("escalation", pa.utf8()),
    pa.field("confidence", pa.float32()),
    pa.field("supporting_relations", pa.list_(pa.utf8())),
    pa.field("vector", pa.list_(pa.float32(), 256)),
    # Type system fields (populated by denoising step, SYSTEM_SPECIFICATION.md Section 4)
    pa.field("state_before", pa.utf8()),
    pa.field("state_after", pa.utf8()),
    pa.field("actor_posture", pa.utf8()),
    pa.field("cost_magnitude", pa.utf8()),
    pa.field("risk_level", pa.utf8()),
    pa.field("resource_types", pa.utf8()),         # JSON: list of resource type strings
    pa.field("reversibility", pa.utf8()),
    pa.field("asymmetry_qualities", pa.utf8()),    # JSON: {surprise, ambiguity, cross_domain, constancy}
    pa.field("situation_framing", pa.utf8()),
    pa.field("initiator_role", pa.utf8()),
    pa.field("target_role", pa.utf8()),
])

STRATEGIC_ANALYSIS_SCHEMA = pa.schema([
    pa.field("analysis_id", pa.utf8()),
    pa.field("cluster_id", pa.utf8()),
    # Aggregated signals from constituent events
    pa.field("decision_context_agg", pa.utf8()),      # JSON: merged decision contexts across events
    pa.field("implicit_signals_agg", pa.utf8()),       # JSON: merged implicit signals across events
    pa.field("trade_off_signals_agg", pa.utf8()),      # JSON: merged trade-off signals across events
    # LLM-synthesized strategic analysis
    pa.field("unstated_objectives", pa.utf8()),        # JSON: [{objective, confidence, evidence}]
    pa.field("counterfactual_branches", pa.utf8()),    # JSON: [{condition, alternative_action, projected_outcome}]
    pa.field("faustian_trades", pa.utf8()),            # JSON: [{short_term_gain, long_term_cost, actors_affected}]
    pa.field("strategic_narrative", pa.utf8()),        # LLM-generated synthesis
    pa.field("decision_tree", pa.utf8()),              # JSON: structured decision tree
    pa.field("confidence", pa.float32()),
    pa.field("model_id", pa.utf8()),
])

# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

TABLE_SCHEMAS = {
    "raw_events": RAW_EVENTS_SCHEMA,
    "entities": ENTITIES_SCHEMA,
    "relations": RELATIONS_SCHEMA,
    "clusters": CLUSTERS_SCHEMA,
    "hyperedges": HYPEREDGES_SCHEMA,
    "strategic_analysis": STRATEGIC_ANALYSIS_SCHEMA,
}


def get_or_create_table(
    db: lancedb.DBConnection,
    name: str,
    data=None,
) -> lancedb.table.Table:
    """Open an existing table or create it from schema/data.

    If *data* is provided it is used to create/overwrite the table.
    Otherwise an empty table is created from the predefined schema.
    """
    if data is not None:
        return db.create_table(name, data=data, mode="overwrite")
    try:
        return db.open_table(name)
    except Exception:
        schema = TABLE_SCHEMAS[name]
        return db.create_table(name, schema=schema)


def table_len(table: lancedb.table.Table) -> int:
    """Return the number of rows in a LanceDB table."""
    return table.count_rows()
