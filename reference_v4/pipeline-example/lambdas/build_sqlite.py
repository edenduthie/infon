"""Lambda handler: Build SQLite databases from LanceDB and upload to S3.

Converts the LanceDB hyperedges/entities/clusters into the star-topology
SQLite databases consumed by the Strands analysis agent:
  - grayzone_hypergraph.db  (nodes/edges for the agent)
  - entity_knowledge.db     (alias lookups)
"""

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
s3_client = boto3.client("s3", region_name=REGION)

# Domain and escalation mappings (mirrors build_sqlite.py in Sprint2)
DOMAIN_TO_CATEGORY = {
    "diplomatic": "Diplomatic",
    "economic": "Economic",
    "financial": "Financial",
    "law_enforcement": "Law Enforcement",
    "military": "Military",
    "intelligence": "Intelligence",
    "informational": "Information",
    "information": "Information",
    "development": "Development",
    "environmental": "Environmental",
    "maritime dispute": "Military",
    "legal": "Law Enforcement",
}

ESCALATION_MAP = {
    "escalation": "high",
    "de-escalation": "low",
    "cooperation-building": "low",
    "cooperation-maintaining": "low",
    "cooperation": "low",
    "neutral": "medium",
    "ambiguous": "medium",
}

HYPERGRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    name TEXT,
    category TEXT,
    action_type TEXT,
    escalation_level TEXT,
    timestamp TEXT,
    actor_posture TEXT,
    cost_magnitude TEXT,
    risk_level TEXT,
    situation_framing TEXT,
    reversibility TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category);
CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON nodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_nodes_posture ON nodes(actor_posture);
CREATE INDEX IF NOT EXISTS idx_nodes_cost ON nodes(cost_magnitude);
CREATE INDEX IF NOT EXISTS idx_nodes_risk ON nodes(risk_level);
CREATE INDEX IF NOT EXISTS idx_nodes_framing ON nodes(situation_framing);
CREATE INDEX IF NOT EXISTS idx_nodes_reversibility ON nodes(reversibility);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    role TEXT NOT NULL,
    metadata TEXT,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);
CREATE INDEX IF NOT EXISTS idx_edges_role ON edges(role);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
"""

MAX_NEXT_GAP_DAYS = 60


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")[:50]


def _doc_id(url: str) -> str:
    return f"doc_{hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]}"


def handler(event, context):
    """Lambda entry point.

    Input:
        lancedb_uri: LanceDB S3 URI
        s3_bucket: Output S3 bucket
        s3_key_prefix: S3 key prefix for SQLite files (default: sqlite/)

    Output:
        hypergraph_db_uri: S3 URI of grayzone_hypergraph.db
        entity_kb_uri: S3 URI of entity_knowledge.db
        stats: Build statistics
    """
    import lancedb
    import pandas as pd

    lancedb_uri = event.get("lancedb_uri", os.environ.get(
        "LANCEDB_URI", "s3://sirius-dimefiled-results/hypergraph/"
    ))
    s3_bucket = event.get("s3_bucket", os.environ.get(
        "S3_BUCKET", "sirius-dimefiled-results"
    ))
    s3_key_prefix = event.get("s3_key_prefix", "sqlite/")

    conn = lancedb.connect(lancedb_uri)

    entities_df = conn.open_table("entities").to_pandas()
    hyperedges_df = conn.open_table("hyperedges").to_pandas()
    clusters_df = conn.open_table("clusters").to_pandas()

    # Strategic analysis (may not exist on older pipelines)
    try:
        strategic_df = conn.open_table("strategic_analysis").to_pandas()
        strategic_lookup = {
            row["cluster_id"]: row for _, row in strategic_df.iterrows()
        }
    except Exception:
        strategic_df = pd.DataFrame()
        strategic_lookup = {}
    # Backward compat: ensure hierarchy columns exist
    if "depth" not in clusters_df.columns:
        clusters_df["depth"] = 0
    if "parent_cluster_id" not in clusters_df.columns:
        clusters_df["parent_cluster_id"] = ""

    entity_lookup = {row["entity_id"]: row for _, row in entities_df.iterrows()}
    cluster_lookup = {row["cluster_id"]: row for _, row in clusters_df.iterrows()}

    # Build hypergraph SQLite
    hg_path = os.path.join(tempfile.gettempdir(), "grayzone_hypergraph.db")
    if os.path.exists(hg_path):
        os.remove(hg_path)

    hg_conn = sqlite3.connect(hg_path)
    hg_conn.executescript(HYPERGRAPH_SCHEMA)
    hg_conn.execute("PRAGMA journal_mode=WAL")
    stats = defaultdict(int)

    # Actor nodes
    actor_ids_seen = set()
    for eid, row in entity_lookup.items():
        if str(row.get("entity_type", "")).lower() == "location":
            continue
        name = row.get("short_name") or row.get("canonical_name") or ""
        metadata = {}
        parent_id = row.get("parent_entity_id", "")
        if parent_id and parent_id in entity_lookup:
            parent_row = entity_lookup[parent_id]
            if str(parent_row.get("entity_type", "")).lower() == "nation_state":
                metadata["country"] = parent_row.get("short_name") or ""
        metadata["parent_class"] = str(row.get("entity_type", ""))
        hg_conn.execute(
            "INSERT OR IGNORE INTO nodes (id, node_type, name, metadata) VALUES (?, ?, ?, ?)",
            (eid, "Actor", name, json.dumps(metadata) if metadata else None),
        )
        actor_ids_seen.add(eid)
        stats["actors"] += 1

    # Location nodes from entities
    location_ids_seen = set()
    for eid, row in entity_lookup.items():
        if str(row.get("entity_type", "")).lower() != "location":
            continue
        name = row.get("canonical_name") or row.get("short_name") or ""
        loc_id = f"loc_{_slugify(name)}"
        if loc_id not in location_ids_seen:
            hg_conn.execute(
                "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                (loc_id, "Location", name),
            )
            location_ids_seen.add(loc_id)
            stats["locations"] += 1

    # Location hierarchy PART_OF edges from entity parent_entity_id chains
    # Built dynamically by Phase 3b (LLM location hierarchy resolution)
    loc_entity_to_node: dict[str, str] = {}  # entity_id → loc_node_id
    for eid, row in entity_lookup.items():
        if str(row.get("entity_type", "")).lower() != "location":
            continue
        name = row.get("canonical_name") or row.get("short_name") or ""
        loc_entity_to_node[eid] = f"loc_{_slugify(name)}"

    for eid, row in entity_lookup.items():
        if str(row.get("entity_type", "")).lower() != "location":
            continue
        parent_eid = str(row.get("parent_entity_id", "") or "").strip()
        if not parent_eid or parent_eid not in entity_lookup:
            continue
        parent_row = entity_lookup[parent_eid]
        # Parent could be a location or a nation_state (country)
        parent_type = str(parent_row.get("entity_type", "")).lower()
        child_name = row.get("canonical_name") or row.get("short_name") or ""
        parent_name = parent_row.get("canonical_name") or parent_row.get("short_name") or ""
        child_id = f"loc_{_slugify(child_name)}"
        if parent_type == "location":
            parent_id = f"loc_{_slugify(parent_name)}"
        elif parent_type == "nation_state":
            # Nation as geographic container
            parent_id = f"loc_{_slugify(parent_name)}"
            if parent_id not in location_ids_seen:
                hg_conn.execute(
                    "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                    (parent_id, "Location", parent_name),
                )
                location_ids_seen.add(parent_id)
        else:
            continue

        if child_id not in location_ids_seen:
            hg_conn.execute(
                "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                (child_id, "Location", child_name),
            )
            location_ids_seen.add(child_id)
        if parent_id not in location_ids_seen:
            hg_conn.execute(
                "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                (parent_id, "Location", parent_name),
            )
            location_ids_seen.add(parent_id)
        hg_conn.execute(
            "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
            (child_id, parent_id, "PART_OF"),
        )
        stats["location_hierarchy_edges"] += 1

    # Event nodes + edges
    event_locations = {}
    for _, he in hyperedges_df.iterrows():
        evt_id = f"evt_{he['hyperedge_id']}"
        domain_raw = str(he.get("l1_domain", "")).lower()
        category = DOMAIN_TO_CATEGORY.get(domain_raw)
        if category is None:
            print(f"  Warning: unmapped domain '{domain_raw}' for {evt_id}")
            category = domain_raw
        esc_raw = str(he.get("escalation", "")).lower()
        escalation_level = ESCALATION_MAP.get(esc_raw)
        if escalation_level is None:
            print(f"  Warning: unmapped escalation '{esc_raw}' for {evt_id}")
            escalation_level = esc_raw

        # Type system fields (may be absent on older data)
        posture = str(he.get("actor_posture", "") or "")
        cost_mag = str(he.get("cost_magnitude", "") or "")
        risk = str(he.get("risk_level", "") or "")
        framing = str(he.get("situation_framing", "") or "")
        reversibility = str(he.get("reversibility", "") or "")

        # Rich metadata including type system signals
        meta = {
            "cluster_id": he.get("cluster_id", ""),
            "l1_domain": he.get("l1_domain", ""),
            "confidence": float(he.get("confidence", 0)),
            "state_before": str(he.get("state_before", "") or ""),
            "state_after": str(he.get("state_after", "") or ""),
            "actor_posture": posture,
            "cost_magnitude": cost_mag,
            "risk_level": risk,
            "resource_types": json.loads(he["resource_types"]) if isinstance(he.get("resource_types"), str) and he["resource_types"].strip() else [],
            "reversibility": reversibility,
            "situation_framing": framing,
            "initiator_role": str(he.get("initiator_role", "") or ""),
            "target_role": str(he.get("target_role", "") or ""),
        }
        # Parse asymmetry_qualities (stored as JSON string in LanceDB)
        aq_raw = he.get("asymmetry_qualities", "")
        if isinstance(aq_raw, str) and aq_raw:
            try:
                meta["asymmetry"] = json.loads(aq_raw)
            except (json.JSONDecodeError, TypeError):
                meta["asymmetry"] = {}
        elif isinstance(aq_raw, dict):
            meta["asymmetry"] = aq_raw

        hg_conn.execute(
            """INSERT OR IGNORE INTO nodes
               (id, node_type, name, category, action_type,
                escalation_level, timestamp,
                actor_posture, cost_magnitude, risk_level,
                situation_framing, reversibility, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (evt_id, "Event", str(he.get("action_summary", ""))[:200],
             category, str(he.get("action_type", "")),
             escalation_level, str(he.get("timestamp_iso", "")),
             posture or None, cost_mag or None, risk or None,
             framing or None, reversibility or None,
             json.dumps(meta)),
        )
        stats["events"] += 1

        # AGENT edges
        initiators = he.get("initiators")
        if initiators is not None:
            for init_id in initiators:
                if init_id and init_id in actor_ids_seen:
                    hg_conn.execute(
                        "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
                        (init_id, evt_id, "AGENT"),
                    )
                    stats["agent_edges"] += 1

        # PATIENT edges
        targets = he.get("targets")
        if targets is not None:
            for tgt_id in targets:
                if tgt_id and tgt_id in actor_ids_seen:
                    hg_conn.execute(
                        "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
                        (evt_id, tgt_id, "PATIENT"),
                    )
                    stats["patient_edges"] += 1

        # LOCATED_AT
        location_name = str(he.get("location", "")).strip()
        if location_name:
            loc_id = f"loc_{_slugify(location_name)}"
            if loc_id not in location_ids_seen:
                hg_conn.execute(
                    "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                    (loc_id, "Location", location_name),
                )
                location_ids_seen.add(loc_id)
            hg_conn.execute(
                "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
                (evt_id, loc_id, "LOCATED_AT"),
            )
            stats["located_at_edges"] += 1
            event_locations[evt_id] = loc_id

    # Document nodes + CITES edges
    cluster_to_evts = defaultdict(list)
    for _, he in hyperedges_df.iterrows():
        cid = he.get("cluster_id", "")
        if cid:
            cluster_to_evts[cid].append(f"evt_{he['hyperedge_id']}")

    doc_ids_seen = set()
    for cid, evt_ids in cluster_to_evts.items():
        if cid not in cluster_lookup:
            continue
        urls = cluster_lookup[cid].get("article_urls")
        if urls is None:
            continue
        for url in urls:
            url = str(url).strip()
            if not url:
                continue
            did = _doc_id(url)
            if did not in doc_ids_seen:
                hg_conn.execute(
                    "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                    (did, "Document", url),
                )
                doc_ids_seen.add(did)
                stats["documents"] += 1
            for eid in evt_ids:
                hg_conn.execute(
                    "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
                    (did, eid, "CITES"),
                )
                stats["cites_edges"] += 1

    # Cluster hierarchy nodes + PART_OF edges
    # depth=0 Knox roots, depth=1 location sub-clusters, depth=2 hourly bins
    cluster_ids_seen = set()
    for _, cl in clusters_df.iterrows():
        cid = cl.get("cluster_id", "")
        if not cid:
            continue
        depth = int(cl.get("depth", 0))
        parent_cid = str(cl.get("parent_cluster_id", "") or "")
        location = str(cl.get("location", "") or "")
        tf_start = str(cl.get("timeframe_start", "") or "")
        tf_end = str(cl.get("timeframe_end", "") or "")
        depth_label = {0: "knox", 1: "topic"}.get(depth, "")
        cl_meta = {
            "depth": depth,
            "depth_label": depth_label,
            "parent_cluster_id": parent_cid,
            "timeframe_start": tf_start,
            "timeframe_end": tf_end,
        }
        cl_name = f"{depth_label}:{location or 'unknown'}" if depth > 0 else location
        hg_conn.execute(
            "INSERT OR IGNORE INTO nodes (id, node_type, name, category, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"cl_{cid}", "Cluster", cl_name[:200], depth_label, tf_start,
             json.dumps(cl_meta)),
        )
        cluster_ids_seen.add(cid)
        stats["cluster_nodes"] += 1

        # PART_OF edge: sub-cluster → parent cluster
        if parent_cid:
            hg_conn.execute(
                "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
                (f"cl_{cid}", f"cl_{parent_cid}", "PART_OF",
                 json.dumps({"child_depth": depth})),
            )
            stats["part_of_edges"] += 1

    # MEMBER_OF edges: events → their finest cluster
    # Map events to all clusters, then pick the deepest
    evt_to_deepest: dict[str, tuple[str, int]] = {}
    for _, cl in clusters_df.iterrows():
        cid = cl.get("cluster_id", "")
        depth = int(cl.get("depth", 0))
        rel_ids = cl.get("relation_ids")
        if rel_ids is None or not cid:
            continue
        # Map cluster's events (via cluster_to_evts built earlier)
        for evt_id in cluster_to_evts.get(cid, []):
            existing = evt_to_deepest.get(evt_id)
            if existing is None or depth > existing[1]:
                evt_to_deepest[evt_id] = (cid, depth)

    for evt_id, (cid, depth) in evt_to_deepest.items():
        hg_conn.execute(
            "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
            (evt_id, f"cl_{cid}", "MEMBER_OF"),
        )
        stats["member_of_edges"] += 1

    # NEXT edges (temporal stitching)
    loc_events = defaultdict(list)
    for _, he in hyperedges_df.iterrows():
        evt_id = f"evt_{he['hyperedge_id']}"
        ts_str = str(he.get("timestamp_iso", "")).strip()
        loc_name = str(he.get("location", "")).strip()
        if not ts_str or not loc_name:
            continue
        loc_id = f"loc_{_slugify(loc_name)}"
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        loc_events[loc_id].append((evt_id, ts))

    for loc_id, events in loc_events.items():
        events.sort(key=lambda x: x[1])
        for i in range(len(events) - 1):
            curr_id, curr_ts = events[i]
            next_id, next_ts = events[i + 1]
            delta_days = (next_ts - curr_ts).days
            if 0 <= delta_days <= MAX_NEXT_GAP_DAYS:
                hg_conn.execute(
                    "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
                    (curr_id, next_id, "NEXT", json.dumps({"delta_days": delta_days})),
                )
                stats["next_edges"] += 1

    # NEXT_ACTOR edges (actor-based temporal stitching)
    # Groups events by initiator actor and creates temporal chains
    # across locations, complementing the location-based NEXT edges.
    actor_events = defaultdict(list)
    for _, he in hyperedges_df.iterrows():
        evt_id = f"evt_{he['hyperedge_id']}"
        ts_str = str(he.get("timestamp_iso", "")).strip()
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        initiators = he.get("initiators")
        if initiators is not None:
            for init_id in initiators:
                if init_id and init_id in actor_ids_seen:
                    actor_events[init_id].append((evt_id, ts))

    for actor_id, events in actor_events.items():
        events.sort(key=lambda x: x[1])
        for i in range(len(events) - 1):
            curr_id, curr_ts = events[i]
            next_id, next_ts = events[i + 1]
            delta_days = (next_ts - curr_ts).days
            if 0 <= delta_days <= MAX_NEXT_GAP_DAYS:
                hg_conn.execute(
                    "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
                    (curr_id, next_id, "NEXT_ACTOR",
                     json.dumps({"delta_days": delta_days, "actor_id": actor_id})),
                )
                stats["next_actor_edges"] += 1

    # Strategy nodes + HAS_STRATEGY edges (Cluster → Strategy)
    # From strategic analysis step: counterfactual branches, Faustian trades
    for cid, sa in strategic_lookup.items():
        strategy_id = f"strat_{sa.get('analysis_id', '')[:12]}"
        narrative = str(sa.get("strategic_narrative", "") or "")
        if not narrative:
            continue

        # Parse JSON fields
        sa_meta = {}
        for field in ("unstated_objectives", "counterfactual_branches",
                       "faustian_trades", "decision_tree"):
            raw = sa.get(field, "")
            if isinstance(raw, str) and raw.strip():
                try:
                    sa_meta[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    sa_meta[field] = raw
            elif not isinstance(raw, str):
                sa_meta[field] = raw

        sa_meta["confidence"] = float(sa.get("confidence", 0))
        sa_meta["model_id"] = str(sa.get("model_id", ""))

        hg_conn.execute(
            "INSERT OR IGNORE INTO nodes (id, node_type, name, metadata) "
            "VALUES (?, ?, ?, ?)",
            (strategy_id, "Strategy", narrative[:200], json.dumps(sa_meta, default=str)),
        )
        stats["strategies"] += 1

        # Link cluster → strategy
        cl_node_id = f"cl_{cid}"
        hg_conn.execute(
            "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
            (cl_node_id, strategy_id, "HAS_STRATEGY",
             json.dumps({"confidence": sa_meta["confidence"]})),
        )
        stats["has_strategy_edges"] += 1

    # FRAMED_AS edges (Event → Situation framing node)
    # Create lightweight Situation nodes for each distinct framing archetype
    framing_ids_seen = set()
    for _, he in hyperedges_df.iterrows():
        framing = str(he.get("situation_framing", "") or "").strip()
        if not framing:
            continue
        evt_id = f"evt_{he['hyperedge_id']}"
        sit_id = f"sit_{_slugify(framing)}"
        if sit_id not in framing_ids_seen:
            hg_conn.execute(
                "INSERT OR IGNORE INTO nodes (id, node_type, name) VALUES (?, ?, ?)",
                (sit_id, "Situation", framing),
            )
            framing_ids_seen.add(sit_id)
            stats["situations"] += 1
        hg_conn.execute(
            "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
            (evt_id, sit_id, "FRAMED_AS"),
        )
        stats["framed_as_edges"] += 1

    # CHANGES_STATE edges (Event → State nodes for state_after)
    state_ids_seen = set()
    for _, he in hyperedges_df.iterrows():
        state_after = str(he.get("state_after", "") or "").strip()
        if not state_after:
            continue
        evt_id = f"evt_{he['hyperedge_id']}"
        state_id = f"state_{hashlib.md5(state_after.encode(), usedforsecurity=False).hexdigest()[:12]}"
        if state_id not in state_ids_seen:
            posture = str(he.get("actor_posture", "") or "")
            hg_conn.execute(
                "INSERT OR IGNORE INTO nodes (id, node_type, name, actor_posture) "
                "VALUES (?, ?, ?, ?)",
                (state_id, "State", state_after[:200], posture or None),
            )
            state_ids_seen.add(state_id)
            stats["states"] += 1
        hg_conn.execute(
            "INSERT INTO edges (source_id, target_id, role) VALUES (?, ?, ?)",
            (evt_id, state_id, "CHANGES_STATE"),
        )
        stats["changes_state_edges"] += 1

    hg_conn.commit()
    node_count = hg_conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = hg_conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    hg_conn.close()

    # Entity knowledge DB
    ek_path = os.path.join(tempfile.gettempdir(), "entity_knowledge.db")
    if os.path.exists(ek_path):
        os.remove(ek_path)

    ek_conn = sqlite3.connect(ek_path)
    ek_conn.execute(
        "CREATE TABLE IF NOT EXISTS entities "
        "(canonical_id TEXT PRIMARY KEY, canonical_name TEXT, country TEXT, parent_class TEXT)"
    )
    ek_conn.execute(
        "CREATE TABLE IF NOT EXISTS aliases "
        "(alias TEXT, canonical_id TEXT)"
    )
    ek_conn.execute("CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias)")

    for eid, row in entity_lookup.items():
        if str(row.get("entity_type", "")).lower() == "location":
            continue
        canonical_name = row.get("canonical_name") or row.get("short_name") or ""
        country = None
        parent_id = row.get("parent_entity_id", "")
        if parent_id and parent_id in entity_lookup:
            parent_row = entity_lookup[parent_id]
            if str(parent_row.get("entity_type", "")).lower() == "nation_state":
                country = parent_row.get("short_name") or ""
        ek_conn.execute(
            "INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)",
            (eid, canonical_name, country, str(row.get("entity_type", ""))),
        )
        aliases = row.get("aliases")
        if aliases is not None:
            for alias in aliases:
                alias = str(alias).strip()
                if alias:
                    ek_conn.execute("INSERT INTO aliases VALUES (?, ?)", (alias.lower(), eid))

    ek_conn.commit()
    ek_conn.close()

    # Upload to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    hg_s3_key = f"{s3_key_prefix}grayzone_hypergraph_{timestamp}.db"
    ek_s3_key = f"{s3_key_prefix}entity_knowledge_{timestamp}.db"

    # Also upload as "latest" for stable references
    hg_latest_key = f"{s3_key_prefix}grayzone_hypergraph.db"
    ek_latest_key = f"{s3_key_prefix}entity_knowledge.db"

    for local, keys in [
        (hg_path, [hg_s3_key, hg_latest_key]),
        (ek_path, [ek_s3_key, ek_latest_key]),
    ]:
        for key in keys:
            s3_client.upload_file(local, s3_bucket, key)

    stats["total_nodes"] = node_count
    stats["total_edges"] = edge_count

    print(f"Built SQLite: {node_count} nodes, {edge_count} edges")

    return {
        "hypergraph_db_uri": f"s3://{s3_bucket}/{hg_latest_key}",
        "entity_kb_uri": f"s3://{s3_bucket}/{ek_latest_key}",
        "stats": dict(stats),
    }
