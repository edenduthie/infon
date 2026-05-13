"""Lambda handler: Mine sequential patterns and construct V2 type tables.

Reads the star-topology SQLite database produced by BuildSQLite and runs
the sequential mining pipeline:

  1. Extract trajectories (location-based NEXT + actor-based NEXT_ACTOR)
  2. PrefixSpan closed pattern mining -> SkeletalPlay candidates (Type 9)
  3. Causal sequence construction with typed links (Type 5)
  4. Temporal motif discovery (retaliation, cross-domain coordination)
  5. Discriminative analysis (escalation / de-escalation predictors)
  6. VF2 subgraph matching -> Play instances (Type 10)
  7. Write V2 type tables to SQLite
  8. Export V2 tables as Parquet -> S3

Dependencies: prefixspan, networkx, pyarrow, pandas
"""

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
s3_client = boto3.client("s3", region_name=REGION)

# ---------------------------------------------------------------------------
# Code mappings (align with DIMEFILED L1 taxonomy)
# ---------------------------------------------------------------------------

DOMAIN_CODES: Dict[str, str] = {
    "Military": "M",
    "Diplomatic": "D",
    "Economic": "E",
    "Information": "I",
    "Financial": "F",
    "Intelligence": "IN",
    "Law Enforcement": "L",
    "Environmental": "EN",
    "Development": "DV",
}

ESC_CODES: Dict[str, str] = {
    "high": "hi",
    "medium": "med",
    "low": "lo",
}

ESC_NUMERIC: Dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

# ---------------------------------------------------------------------------
# MDUS (Multi-Dimensional Utility Scoring) constants
# ---------------------------------------------------------------------------

# Escalation classification -> numeric value (per MDUS spec)
ESCALATION_VALUES: Dict[str, int] = {
    "escalation": 2,
    "ambiguous": 1,
    "neutral": 0,
    "cooperation-building": -1,
    "cooperation-maintaining": -1,
    "de-escalation": -2,
}

# Contestation dynamics that force de-escalatory values
DE_ESCALATORY_CONTESTATIONS: Set[str] = {
    "reducing-contestation",
    "building-cooperation",
    "maintaining-cooperation",
    "deepening-partnership",
    "strengthening-alliance",
}

# Influence pattern -> strategic leverage weight
INFLUENCE_WEIGHTS: Dict[str, float] = {
    "fait-accompli": 1.0,
    "power-transition": 0.9,
    "escalation-spiral": 0.85,
    "tit-for-tat": 0.8,
    "provocation-response": 0.8,
    "threshold-testing": 0.75,
    "demonstration": 0.7,
    "deterrence": 0.6,
    "containment": 0.6,
    "signaling": 0.5,
    "consolidation": 0.5,
    "normalization": 0.3,
}

# Actor type -> multiplier
ACTOR_TYPE_MULTIPLIERS: Dict[str, float] = {
    "military": 1.2,
    "state": 1.0,
    "government": 0.8,
    "international-org": 0.8,
    "ngo": 0.6,
    "media": 0.6,
    "corporate": 0.7,
    "other": 0.5,
}

# MDUS component weights
MDUS_W_ESCALATION = 0.4
MDUS_W_CROSSDOMAIN = 0.3
MDUS_W_LEVERAGE = 0.3

# Mining parameters (from empirical investigation)
MIN_SUPPORT = 5
MIN_CHAIN_LENGTH = 2
MAX_PATTERN_LENGTH = 8
PREFIXSPAN_CLOSED = True
VF2_PREFIX_LENGTH = 3
COORDINATION_WINDOW_DAYS = 7

# V2 schema SQL for the mining output tables
V2_MINING_SCHEMA = """
-- Type #9: Skeletal plays - mined sequential templates
CREATE TABLE IF NOT EXISTS skeletal_plays (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pattern_codes JSON NOT NULL,
    domain_sequence JSON NOT NULL,
    escalation_sequence JSON,
    support INTEGER NOT NULL,
    location_count INTEGER,
    locations JSON,
    outcome_distribution JSON,
    discriminative_score REAL,
    discriminative_direction TEXT,
    source TEXT DEFAULT 'prefixspan_closed',
    metadata JSON
);

-- Type #5: Causal sequences - ordered chains with typed links
CREATE TABLE IF NOT EXISTS causal_sequences (
    id TEXT PRIMARY KEY,
    action_ids JSON NOT NULL,
    causal_edges JSON,
    trajectory_code JSON,
    outcome TEXT,
    location_id TEXT,
    chain_type TEXT DEFAULT 'location',
    actor_id TEXT,
    metadata JSON
);

-- Type #10: Plays - instances of skeletal plays matched in data
CREATE TABLE IF NOT EXISTS plays (
    id TEXT PRIMARY KEY,
    skeletal_play_id TEXT REFERENCES skeletal_plays(id),
    causal_sequence_id TEXT REFERENCES causal_sequences(id),
    matched_codes JSON,
    match_score REAL,
    match_length INTEGER,
    location TEXT,
    timespan_start TEXT,
    timespan_end TEXT,
    metadata JSON
);

-- Temporal motifs: retaliation patterns
CREATE TABLE IF NOT EXISTS retaliation_motifs (
    id TEXT PRIMARY KEY,
    event_a_id TEXT NOT NULL,
    event_b_id TEXT NOT NULL,
    retaliating_actor_id TEXT,
    original_actor_id TEXT,
    domain_transition TEXT,
    escalation_behavior TEXT,
    delta_days INTEGER,
    metadata JSON
);

-- Temporal motifs: cross-domain coordination windows
CREATE TABLE IF NOT EXISTS coordination_windows (
    id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    domains JSON NOT NULL,
    event_ids JSON NOT NULL,
    window_start TEXT,
    window_end TEXT,
    span_days INTEGER,
    metadata JSON
);

-- Discriminative patterns: escalation / de-escalation predictors
CREATE TABLE IF NOT EXISTS discriminative_patterns (
    id TEXT PRIMARY KEY,
    pattern_codes JSON NOT NULL,
    log_odds REAL NOT NULL,
    direction TEXT NOT NULL,
    support_escalating INTEGER,
    support_deescalating INTEGER,
    metadata JSON
);

-- MDUS utility scores per pattern/play
CREATE TABLE IF NOT EXISTS utility_scores (
    id TEXT PRIMARY KEY,
    pattern_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    mdus_median REAL,
    mdus_min REAL,
    mdus_q25 REAL,
    mdus_q75 REAL,
    mdus_max REAL,
    mdus_std REAL,
    escalation_magnitude REAL,
    cross_domain_spread REAL,
    strategic_leverage REAL,
    actor_type_multiplier REAL,
    metadata JSON
);

-- Mining run metadata
CREATE TABLE IF NOT EXISTS mining_runs (
    id TEXT PRIMARY KEY,
    run_timestamp TEXT NOT NULL,
    parameters JSON,
    stats JSON
);

-- V2 indexes
CREATE INDEX IF NOT EXISTS idx_skeletal_plays_support ON skeletal_plays(support);
CREATE INDEX IF NOT EXISTS idx_causal_sequences_outcome ON causal_sequences(outcome);
CREATE INDEX IF NOT EXISTS idx_causal_sequences_location ON causal_sequences(location_id);
CREATE INDEX IF NOT EXISTS idx_causal_sequences_chain_type ON causal_sequences(chain_type);
CREATE INDEX IF NOT EXISTS idx_plays_skeletal ON plays(skeletal_play_id);
CREATE INDEX IF NOT EXISTS idx_retaliation_actor ON retaliation_motifs(retaliating_actor_id);
CREATE INDEX IF NOT EXISTS idx_coordination_actor ON coordination_windows(actor_id);
CREATE INDEX IF NOT EXISTS idx_discriminative_direction ON discriminative_patterns(direction);
CREATE INDEX IF NOT EXISTS idx_utility_pattern ON utility_scores(pattern_id);
CREATE INDEX IF NOT EXISTS idx_utility_type ON utility_scores(pattern_type);
CREATE INDEX IF NOT EXISTS idx_utility_median ON utility_scores(mdus_median DESC);
"""


# ---------------------------------------------------------------------------
# Trajectory extraction from SQLite
# ---------------------------------------------------------------------------

def _load_graph_data(db_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load nodes, edge adjacency, and edge lists from SQLite.

    Returns:
        nodes: {node_id: {node_type, name, category, escalation_level, action_type, timestamp, metadata, ...}}
        next_edges: {source_id: [(target_id, metadata_dict), ...]}
        all_edges: {role: [(source_id, target_id, metadata_dict), ...]}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    nodes = {}
    for row in conn.execute("SELECT * FROM nodes"):
        d = dict(row)
        meta_str = d.pop("metadata", None)
        if meta_str:
            try:
                d["_meta"] = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError):
                d["_meta"] = {}
        else:
            d["_meta"] = {}
        nodes[d["id"]] = d

    # Build edge lists by role
    all_edges: Dict[str, List[Tuple[str, str, dict]]] = defaultdict(list)
    for row in conn.execute("SELECT source_id, target_id, role, metadata FROM edges"):
        meta = {}
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        all_edges[row["role"]].append((row["source_id"], row["target_id"], meta))

    conn.close()
    return nodes, all_edges


def _extract_chains(
    edges: List[Tuple[str, str, dict]],
) -> List[List[str]]:
    """Extract maximal chains from directed edge list.

    Assumes edges form linear chains (no branching).
    Returns list of chains, each a list of node IDs.
    """
    next_out: Dict[str, str] = {}
    next_in: Set[str] = set()

    for src, tgt, _ in edges:
        next_out[src] = tgt
        next_in.add(tgt)

    # Chain starts: nodes with outgoing but no incoming
    chain_starts = [n for n in next_out if n not in next_in]

    chains: List[List[str]] = []
    visited: Set[str] = set()

    for start in chain_starts:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        current = start
        while current in next_out:
            nxt = next_out[current]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(chain) >= MIN_CHAIN_LENGTH:
            chains.append(chain)

    return chains


def _extract_actor_chains(
    edges: List[Tuple[str, str, dict]],
) -> List[Tuple[str, List[str]]]:
    """Extract actor-based chains from NEXT_ACTOR edges.

    Groups edges by actor_id (from metadata) and extracts chains per actor.
    Returns list of (actor_id, chain) tuples.
    """
    # Group edges by actor
    actor_edges: Dict[str, List[Tuple[str, str, dict]]] = defaultdict(list)
    for src, tgt, meta in edges:
        actor_id = meta.get("actor_id", "")
        if actor_id:
            actor_edges[actor_id].append((src, tgt, meta))

    result = []
    for actor_id, a_edges in actor_edges.items():
        chains = _extract_chains(a_edges)
        for chain in chains:
            result.append((actor_id, chain))

    return result


def _encode_event(node: dict) -> str:
    """Encode event as Domain:Escalation code (e.g. 'D_hi', 'M_lo').

    Raises KeyError if domain or escalation is not in the mapping.
    """
    cat = node.get("category", "")
    esc = node.get("escalation_level", "")
    domain = DOMAIN_CODES.get(cat)
    if domain is None:
        raise KeyError(f"Unmapped domain category: '{cat}' for node {node.get('id', '?')}")
    level = ESC_CODES.get(esc)
    if level is None:
        raise KeyError(f"Unmapped escalation level: '{esc}' for node {node.get('id', '?')}")
    return f"{domain}_{level}"


def _classify_outcome(nodes: Dict, chain: List[str]) -> str:
    """Classify a chain's overall escalation direction."""
    levels = []
    for eid in chain:
        node = nodes.get(eid, {})
        esc = node.get("escalation_level", "")
        numeric = ESC_NUMERIC.get(esc)
        if numeric is None:
            raise KeyError(f"Unmapped escalation level: '{esc}' for node {eid}")
        levels.append(numeric)

    if len(levels) < 2:
        return "stable"

    deltas = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]

    if all(d == 0 for d in deltas):
        return "stable"

    has_up = any(d > 0 for d in deltas)
    has_down = any(d < 0 for d in deltas)

    if has_up and has_down:
        return "mixed"
    if has_up:
        return "escalating"
    if has_down:
        return "de-escalating"
    return "mixed"


def _get_chain_location(nodes: Dict, all_edges: Dict, chain: List[str]) -> str:
    """Get the location for a chain from LOCATED_AT edges."""
    # Build a quick lookup of event → location
    located_at = {}
    for src, tgt, _ in all_edges.get("LOCATED_AT", []):
        located_at[src] = tgt

    first_evt = chain[0]
    loc_id = located_at.get(first_evt, "")
    if loc_id and loc_id in nodes:
        return nodes[loc_id].get("name", loc_id)
    return ""


def _build_trajectories(
    nodes: Dict, all_edges: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Build location-based and actor-based trajectory dicts.

    Returns:
        (location_trajectories, actor_trajectories)
    """
    # Location-based from NEXT edges
    next_edges = all_edges.get("NEXT", [])
    loc_chains = _extract_chains(next_edges)

    loc_trajectories = []
    for chain in loc_chains:
        codes = [_encode_event(nodes.get(eid, {})) for eid in chain]
        outcome = _classify_outcome(nodes, chain)
        location = _get_chain_location(nodes, all_edges, chain)

        # Get location ID for foreign key
        located_at = {}
        for src, tgt, _ in all_edges.get("LOCATED_AT", []):
            located_at[src] = tgt
        loc_id = located_at.get(chain[0], "")

        loc_trajectories.append({
            "id": f"traj_{chain[0]}",
            "chain": chain,
            "codes": codes,
            "outcome": outcome,
            "location": location,
            "location_id": loc_id,
            "chain_type": "location",
        })

    # Actor-based from NEXT_ACTOR edges
    next_actor_edges = all_edges.get("NEXT_ACTOR", [])
    actor_chains = _extract_actor_chains(next_actor_edges)

    actor_trajectories = []
    for actor_id, chain in actor_chains:
        codes = [_encode_event(nodes.get(eid, {})) for eid in chain]
        outcome = _classify_outcome(nodes, chain)

        actor_trajectories.append({
            "id": f"atraj_{actor_id}_{chain[0]}",
            "chain": chain,
            "codes": codes,
            "outcome": outcome,
            "actor_id": actor_id,
            "actor_name": nodes.get(actor_id, {}).get("name", ""),
            "chain_type": "actor",
        })

    return loc_trajectories, actor_trajectories


# ---------------------------------------------------------------------------
# PrefixSpan closed pattern mining
# ---------------------------------------------------------------------------

def _mine_closed_patterns(
    trajectories: List[Dict],
    min_support: int = MIN_SUPPORT,
    max_length: int = MAX_PATTERN_LENGTH,
) -> List[Dict]:
    """Run PrefixSpan closed sequential pattern mining.

    Returns list of pattern dicts with support, locations, outcome distribution.
    """
    from prefixspan import PrefixSpan

    # Convert trajectory codes to integer sequences for PrefixSpan
    all_codes = sorted({c for t in trajectories for c in t["codes"]})
    code_to_int = {c: i for i, c in enumerate(all_codes)}
    int_to_code = {i: c for c, i in code_to_int.items()}

    sequences = []
    for traj in trajectories:
        seq = [code_to_int[c] for c in traj["codes"]]
        sequences.append(seq)

    if not sequences:
        return []

    ps = PrefixSpan(sequences)
    ps.minlen = 3
    ps.maxlen = max_length

    closed_patterns = ps.frequent(min_support, closed=True)

    # Build trajectory lookup for outcome distribution
    traj_lookup = {i: t for i, t in enumerate(trajectories)}

    results = []
    for support, pattern_ints in closed_patterns:
        pattern_codes = [int_to_code[i] for i in pattern_ints]

        # Extract domain sequence (strip escalation)
        domain_seq = [c.split("_")[0] for c in pattern_codes]
        esc_seq = [c.split("_")[1] if "_" in c else "unk" for c in pattern_codes]

        # Find which trajectories contain this pattern (as contiguous subsequence)
        containing_trajs = []
        for idx, seq in enumerate(sequences):
            if _contains_subsequence(seq, pattern_ints):
                containing_trajs.append(idx)

        # Outcome distribution
        outcome_dist = Counter()
        locations = set()
        for idx in containing_trajs:
            traj = traj_lookup.get(idx)
            if traj:
                outcome_dist[traj["outcome"]] += 1
                loc = traj.get("location", "")
                if loc:
                    locations.add(loc)

        # Generate name from pattern
        name = " -> ".join(domain_seq)
        if len(set(esc_seq)) == 1:
            name += f" [{esc_seq[0]}]"

        pat_id = f"sp_{hashlib.md5('_'.join(pattern_codes).encode(), usedforsecurity=False).hexdigest()[:12]}"

        results.append({
            "id": pat_id,
            "name": name,
            "pattern_codes": pattern_codes,
            "domain_sequence": domain_seq,
            "escalation_sequence": esc_seq,
            "support": support,
            "location_count": len(locations),
            "locations": sorted(locations)[:20],
            "outcome_distribution": dict(outcome_dist),
        })

    # Sort by support descending
    results.sort(key=lambda x: x["support"], reverse=True)
    return results


def _contains_subsequence(seq: List[int], subseq: List[int]) -> bool:
    """Check if seq contains subseq as a contiguous subsequence."""
    if len(subseq) > len(seq):
        return False
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i + len(subseq)] == subseq:
            return True
    return False


# ---------------------------------------------------------------------------
# Causal sequence construction
# ---------------------------------------------------------------------------

def _infer_causal_type(prev_node: dict, curr_node: dict) -> str:
    """Infer causal relationship between consecutive events.

    Returns: cause | enable | prevent | respond | temporal
    """
    prev_esc = ESC_NUMERIC[prev_node["escalation_level"]]
    curr_esc = ESC_NUMERIC[curr_node["escalation_level"]]
    prev_type = prev_node.get("action_type", "")
    curr_type = curr_node.get("action_type", "")

    if prev_type == "cooperative" and curr_esc < prev_esc:
        return "enable"
    if prev_type == "confrontational" and curr_esc > prev_esc:
        return "cause"
    if prev_type == "cooperative" and curr_esc > prev_esc:
        return "prevent"
    if curr_type in ("confrontational", "cooperative"):
        return "respond"
    return "temporal"


def _build_causal_sequences(
    nodes: Dict, trajectories: List[Dict],
) -> List[Dict]:
    """Build causal sequences with typed links from trajectories."""
    sequences = []

    for traj in trajectories:
        chain = traj["chain"]
        if len(chain) < 2:
            continue

        causal_edges = []
        for i in range(1, len(chain)):
            prev = nodes.get(chain[i - 1], {})
            curr = nodes.get(chain[i], {})
            causal_type = _infer_causal_type(prev, curr)
            prev_esc = ESC_NUMERIC[prev["escalation_level"]]
            curr_esc = ESC_NUMERIC[curr["escalation_level"]]
            causal_edges.append({
                "source_idx": i - 1,
                "target_idx": i,
                "relation": causal_type,
                "escalation_delta": curr_esc - prev_esc,
            })

        seq_id = traj["id"].replace("traj_", "cseq_").replace("atraj_", "cseq_a_")

        seq = {
            "id": seq_id,
            "action_ids": chain,
            "causal_edges": causal_edges,
            "trajectory_code": traj["codes"],
            "outcome": traj["outcome"],
            "chain_type": traj.get("chain_type", "location"),
        }

        if traj.get("location_id"):
            seq["location_id"] = traj["location_id"]
        if traj.get("actor_id"):
            seq["actor_id"] = traj["actor_id"]

        sequences.append(seq)

    return sequences


# ---------------------------------------------------------------------------
# Temporal motif discovery
# ---------------------------------------------------------------------------

def _detect_retaliations(
    nodes: Dict, all_edges: Dict,
) -> List[Dict]:
    """Detect retaliation patterns: patient of event N becomes agent of event N+1.

    A retaliation occurs when an actor targeted in one event initiates
    the next event in a NEXT chain.
    """
    # Build actor role lookups from AGENT and PATIENT edges
    agent_of: Dict[str, Set[str]] = defaultdict(set)  # event -> set of agent actor IDs
    patient_of: Dict[str, Set[str]] = defaultdict(set)  # event -> set of patient actor IDs

    for src, tgt, _ in all_edges.get("AGENT", []):
        # AGENT edge: actor_id -> event_id
        agent_of[tgt].add(src)
    for src, tgt, _ in all_edges.get("PATIENT", []):
        # PATIENT edge: event_id -> actor_id
        patient_of[src].add(tgt)

    # Check each NEXT edge for retaliation
    retaliations = []
    for src, tgt, meta in all_edges.get("NEXT", []):
        patients_a = patient_of.get(src, set())
        agents_b = agent_of.get(tgt, set())

        # Retaliation: someone who was a patient in A is an agent in B
        retaliating = patients_a & agents_b
        if retaliating:
            for actor_id in retaliating:
                # Find original agent of event A
                original_agents = agent_of.get(src, set())

                node_a = nodes.get(src, {})
                node_b = nodes.get(tgt, {})
                esc_a = node_a.get("escalation_level", "")
                esc_b = node_b.get("escalation_level", "")
                cat_a = node_a.get("category", "")
                cat_b = node_b.get("category", "")

                if ESC_NUMERIC[esc_b] > ESC_NUMERIC[esc_a]:
                    esc_behavior = "escalates"
                elif ESC_NUMERIC[esc_b] < ESC_NUMERIC[esc_a]:
                    esc_behavior = "de-escalates"
                else:
                    esc_behavior = "matches"

                ret_id = f"ret_{hashlib.md5(f'{src}_{tgt}_{actor_id}'.encode(), usedforsecurity=False).hexdigest()[:12]}"

                retaliations.append({
                    "id": ret_id,
                    "event_a_id": src,
                    "event_b_id": tgt,
                    "retaliating_actor_id": actor_id,
                    "original_actor_id": sorted(original_agents)[0] if original_agents else None,
                    "domain_transition": f"{cat_a} -> {cat_b}",
                    "escalation_behavior": esc_behavior,
                    "delta_days": meta.get("delta_days", 0),
                })

    return retaliations


def _detect_coordination_windows(
    nodes: Dict, all_edges: Dict,
    window_days: int = COORDINATION_WINDOW_DAYS,
) -> List[Dict]:
    """Detect cross-domain coordination: actors operating in 2+ domains within a window.

    Finds actors who initiate events across multiple DIME-FIL domains
    within a sliding window of `window_days`.
    """
    # Build actor -> list of (event_id, timestamp, domain)
    actor_events: Dict[str, List[Tuple[str, datetime, str]]] = defaultdict(list)

    for src, tgt, _ in all_edges.get("AGENT", []):
        # src = actor_id, tgt = event_id
        node = nodes.get(tgt, {})
        ts_str = node.get("timestamp", "")
        cat = node.get("category", "")
        if not ts_str or not cat:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        actor_events[src].append((tgt, ts, cat))

    windows = []
    for actor_id, events in actor_events.items():
        if len(events) < 2:
            continue
        events.sort(key=lambda x: x[1])

        # Sliding window
        for i in range(len(events)):
            window_evts = [events[i]]
            domains = {events[i][2]}
            for j in range(i + 1, len(events)):
                delta = (events[j][1] - events[i][1]).days
                if delta > window_days:
                    break
                window_evts.append(events[j])
                domains.add(events[j][2])

            if len(domains) >= 2:
                evt_ids = [e[0] for e in window_evts]
                win_start = window_evts[0][1]
                win_end = window_evts[-1][1]
                span = (win_end - win_start).days

                win_id = f"coord_{hashlib.md5(f'{actor_id}_{win_start.isoformat()}'.encode(), usedforsecurity=False).hexdigest()[:12]}"

                windows.append({
                    "id": win_id,
                    "actor_id": actor_id,
                    "actor_name": nodes.get(actor_id, {}).get("name", ""),
                    "domains": sorted(domains),
                    "event_ids": evt_ids,
                    "window_start": win_start.isoformat(),
                    "window_end": win_end.isoformat(),
                    "span_days": span,
                })

    # Deduplicate: keep the widest window per actor per start date
    seen = {}
    for w in windows:
        key = (w["actor_id"], w["window_start"])
        if key not in seen or len(w["event_ids"]) > len(seen[key]["event_ids"]):
            seen[key] = w

    return sorted(seen.values(), key=lambda x: len(x["domains"]), reverse=True)


# ---------------------------------------------------------------------------
# Discriminative analysis
# ---------------------------------------------------------------------------

def _compute_discriminative_patterns(
    trajectories: List[Dict],
    n: int = 3,
    min_support: int = 2,
) -> List[Dict]:
    """Compute log-odds patterns that discriminate escalation vs de-escalation."""
    esc_trajs = [t for t in trajectories if t["outcome"] == "escalating"]
    deesc_trajs = [t for t in trajectories if t["outcome"] == "de-escalating"]
    total_esc = len(esc_trajs)
    total_deesc = len(deesc_trajs)

    if total_esc == 0 or total_deesc == 0:
        return []

    esc_counts: Dict[Tuple[str, ...], int] = Counter()
    deesc_counts: Dict[Tuple[str, ...], int] = Counter()

    for traj in esc_trajs:
        codes = traj["codes"]
        seen: set = set()
        for i in range(len(codes) - n + 1):
            gram = tuple(codes[i:i + n])
            if gram not in seen:
                esc_counts[gram] += 1
                seen.add(gram)

    for traj in deesc_trajs:
        codes = traj["codes"]
        seen: set = set()
        for i in range(len(codes) - n + 1):
            gram = tuple(codes[i:i + n])
            if gram not in seen:
                deesc_counts[gram] += 1
                seen.add(gram)

    all_grams = set(esc_counts) | set(deesc_counts)
    results = []

    for gram in all_grams:
        s_esc = esc_counts.get(gram, 0)
        s_deesc = deesc_counts.get(gram, 0)
        total = s_esc + s_deesc

        if total < min_support:
            continue

        p_esc = (s_esc + 1) / (total_esc + 2)
        p_deesc = (s_deesc + 1) / (total_deesc + 2)

        odds_esc = p_esc / (1 - p_esc) if p_esc < 1 else 10.0
        odds_deesc = p_deesc / (1 - p_deesc) if p_deesc < 1 else 10.0
        log_odds = math.log(odds_esc / odds_deesc) if odds_deesc > 0 else 0.0

        direction = "escalation" if log_odds > 0 else "de-escalation"
        pat_id = f"disc_{hashlib.md5('_'.join(gram).encode(), usedforsecurity=False).hexdigest()[:12]}"

        results.append({
            "id": pat_id,
            "pattern_codes": list(gram),
            "log_odds": round(log_odds, 4),
            "direction": direction,
            "support_escalating": s_esc,
            "support_deescalating": s_deesc,
        })

    results.sort(key=lambda x: abs(x["log_odds"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# MDUS (Multi-Dimensional Utility Scoring)
# ---------------------------------------------------------------------------

def _get_event_escalation_value(node: dict) -> int:
    """Get escalation value for an event, respecting contestation overrides."""
    meta = node.get("_meta", {})
    esc_class = meta.get("escalation_classification", "")
    contestation = meta.get("contestation_dynamics", "")

    # Contestation override: force de-escalatory values
    if contestation in DE_ESCALATORY_CONTESTATIONS:
        return -1

    return ESCALATION_VALUES.get(esc_class, 0)


def _get_event_domains(node: dict) -> Set[str]:
    """Get the set of DIME-FIL domains affected by an event."""
    domains = set()
    cat = node.get("category", "")
    code = DOMAIN_CODES.get(cat)
    if code:
        domains.add(code)
    # Check cross-domain linkage from metadata
    meta = node.get("_meta", {})
    cross_domain = meta.get("cross_domain_linkage", "")
    if cross_domain and isinstance(cross_domain, str):
        for token in cross_domain.replace(",", " ").split():
            token = token.strip().upper()
            if token in {"D", "I", "M", "E", "F", "IN", "L", "EN", "DV"}:
                domains.add(token)
    return domains


def _get_event_strategic_leverage(node: dict) -> float:
    """Compute strategic leverage from influence patterns."""
    meta = node.get("_meta", {})
    patterns = meta.get("influence_patterns", [])
    if not patterns or not isinstance(patterns, list):
        return 0.0
    weights = [INFLUENCE_WEIGHTS.get(p, 0.0) for p in patterns if isinstance(p, str)]
    return max(weights) if weights else 0.0


def _get_actor_type_multiplier(node: dict) -> float:
    """Get actor type multiplier from event metadata."""
    meta = node.get("_meta", {})
    # Check organisation types from actors
    org_types = meta.get("organisation_types", [])
    if not org_types or not isinstance(org_types, list):
        # Fallback: infer from category
        cat = node.get("category", "")
        if cat == "Military":
            return ACTOR_TYPE_MULTIPLIERS["military"]
        return ACTOR_TYPE_MULTIPLIERS.get("state", 1.0)

    # Use the highest multiplier across actor types
    multipliers = [ACTOR_TYPE_MULTIPLIERS.get(ot, 0.5) for ot in org_types]
    return max(multipliers) if multipliers else 1.0


def _compute_event_utility(node: dict) -> float:
    """Compute MDUS utility for a single event.

    utility = actor_type_multiplier * (
        0.4 * escalation_magnitude +
        0.3 * cross_domain_spread +
        0.3 * strategic_leverage
    )
    """
    # Escalation magnitude: abs(escalation_value) normalised to 0-1
    esc_val = _get_event_escalation_value(node)
    escalation_magnitude = abs(esc_val) / 2.0  # max is 2, normalise to 0-1

    # Cross-domain spread: number of domains normalised to 0-1
    domains = _get_event_domains(node)
    cross_domain_spread = min(len(domains) / 9.0, 1.0)  # 9 possible domains

    # Strategic leverage: max influence pattern weight (already 0-1)
    strategic_leverage = _get_event_strategic_leverage(node)

    # Actor type multiplier
    multiplier = _get_actor_type_multiplier(node)

    utility = multiplier * (
        MDUS_W_ESCALATION * escalation_magnitude
        + MDUS_W_CROSSDOMAIN * cross_domain_spread
        + MDUS_W_LEVERAGE * strategic_leverage
    )

    return round(utility, 4)


def _compute_pattern_utility(
    nodes: Dict,
    trajectories: List[Dict],
    skeletal_plays: List[Dict],
    play_instances: List[Dict],
) -> List[Dict]:
    """Compute MDUS utility scores for skeletal plays and play instances.

    For each skeletal play, collects all event utilities from matching
    trajectories and computes distribution stats (min, Q25, median, Q75, max, std).
    Also scores individual play instances.
    """
    import statistics

    # Pre-compute utility for all event nodes
    event_utilities: Dict[str, float] = {}
    for nid, node in nodes.items():
        if node.get("node_type") == "Event":
            event_utilities[nid] = _compute_event_utility(node)

    # Build play_id -> skeletal_play_id mapping
    sp_to_event_utilities: Dict[str, List[float]] = defaultdict(list)

    # Collect utilities from trajectories matching each skeletal play
    for traj in trajectories:
        chain = traj.get("chain", [])
        for eid in chain:
            u = event_utilities.get(eid, 0.0)
            # Every trajectory contributes to the global pool
            sp_to_event_utilities["__global__"].append(u)

    # Collect per-play-instance utilities
    for pi in play_instances:
        sp_id = pi.get("skeletal_play_id", "")
        event_ids = pi.get("event_ids", [])
        for eid in event_ids:
            u = event_utilities.get(eid, 0.0)
            sp_to_event_utilities[sp_id].append(u)

    utility_scores = []

    # Score each skeletal play
    for sp in skeletal_plays:
        sp_id = sp["id"]
        utils = sp_to_event_utilities.get(sp_id, [])
        if not utils:
            # Fallback: use global distribution scaled by pattern length
            continue

        utils_sorted = sorted(utils)
        n = len(utils_sorted)
        median_u = statistics.median(utils_sorted) if n else 0.0
        q25 = utils_sorted[n // 4] if n >= 4 else utils_sorted[0] if n else 0.0
        q75 = utils_sorted[3 * n // 4] if n >= 4 else utils_sorted[-1] if n else 0.0
        std_u = statistics.stdev(utils_sorted) if n >= 2 else 0.0

        # Also compute avg component values for the pattern
        avg_esc = 0.0
        avg_cds = 0.0
        avg_lev = 0.0
        avg_mult = 0.0
        event_count = 0
        for pi in play_instances:
            if pi.get("skeletal_play_id") != sp_id:
                continue
            for eid in pi.get("event_ids", []):
                nd = nodes.get(eid, {})
                if nd.get("node_type") != "Event":
                    continue
                avg_esc += abs(_get_event_escalation_value(nd)) / 2.0
                avg_cds += min(len(_get_event_domains(nd)) / 9.0, 1.0)
                avg_lev += _get_event_strategic_leverage(nd)
                avg_mult += _get_actor_type_multiplier(nd)
                event_count += 1

        if event_count > 0:
            avg_esc /= event_count
            avg_cds /= event_count
            avg_lev /= event_count
            avg_mult /= event_count

        uid = f"util_{sp_id}"
        utility_scores.append({
            "id": uid,
            "pattern_id": sp_id,
            "pattern_type": "skeletal_play",
            "mdus_median": round(median_u, 4),
            "mdus_min": round(utils_sorted[0], 4) if utils_sorted else 0.0,
            "mdus_q25": round(q25, 4),
            "mdus_q75": round(q75, 4),
            "mdus_max": round(utils_sorted[-1], 4) if utils_sorted else 0.0,
            "mdus_std": round(std_u, 4),
            "escalation_magnitude": round(avg_esc, 4),
            "cross_domain_spread": round(avg_cds, 4),
            "strategic_leverage": round(avg_lev, 4),
            "actor_type_multiplier": round(avg_mult, 4),
        })

        # Also attach median utility to the skeletal play for ranking
        sp["mdus_median"] = round(median_u, 4)

    # Score individual play instances
    for pi in play_instances:
        event_ids = pi.get("event_ids", [])
        utils = [event_utilities.get(eid, 0.0) for eid in event_ids]
        if not utils:
            continue
        utils_sorted = sorted(utils)
        n = len(utils_sorted)
        median_u = statistics.median(utils_sorted)
        q25 = utils_sorted[n // 4] if n >= 4 else utils_sorted[0]
        q75 = utils_sorted[3 * n // 4] if n >= 4 else utils_sorted[-1]
        std_u = statistics.stdev(utils_sorted) if n >= 2 else 0.0

        uid = f"util_{pi['id']}"
        utility_scores.append({
            "id": uid,
            "pattern_id": pi["id"],
            "pattern_type": "play_instance",
            "mdus_median": round(median_u, 4),
            "mdus_min": round(utils_sorted[0], 4),
            "mdus_q25": round(q25, 4),
            "mdus_q75": round(q75, 4),
            "mdus_max": round(utils_sorted[-1], 4),
            "mdus_std": round(std_u, 4),
            "escalation_magnitude": 0.0,
            "cross_domain_spread": 0.0,
            "strategic_leverage": 0.0,
            "actor_type_multiplier": 0.0,
        })

        # Attach to play instance for downstream use
        pi["mdus_median"] = round(median_u, 4)

    return utility_scores


# ---------------------------------------------------------------------------
# VF2 subgraph matching
# ---------------------------------------------------------------------------

def _match_plays(
    trajectories: List[Dict],
    skeletal_plays: List[Dict],
    prefix_length: int = VF2_PREFIX_LENGTH,
) -> List[Dict]:
    """Match trajectories against mined skeletal plays using VF2.

    Uses NetworkX DiGraphMatcher for subgraph isomorphism at the specified
    prefix length. Returns Play instances where a trajectory matches a
    skeletal play prefix.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import DiGraphMatcher

    plays = []

    for sp in skeletal_plays:
        sp_codes = sp["pattern_codes"]
        # Use prefix of the skeletal play
        prefix_codes = sp_codes[:prefix_length]
        if len(prefix_codes) < 2:
            continue

        # Build template graph
        template = nx.DiGraph()
        for i, code in enumerate(prefix_codes):
            domain = code.split("_")[0]
            template.add_node(i, domain=domain, code=code)
            if i > 0:
                template.add_edge(i - 1, i)

        for traj in trajectories:
            codes = traj["codes"]
            if len(codes) < len(prefix_codes):
                continue

            # Slide window over trajectory
            for start in range(len(codes) - len(prefix_codes) + 1):
                window = codes[start:start + len(prefix_codes)]

                # Build candidate graph
                candidate = nx.DiGraph()
                for i, code in enumerate(window):
                    domain = code.split("_")[0]
                    candidate.add_node(i, domain=domain, code=code)
                    if i > 0:
                        candidate.add_edge(i - 1, i)

                # Domain-only matching
                def node_match(n1, n2):
                    return n1["domain"] == n2["domain"]

                matcher = DiGraphMatcher(candidate, template, node_match=node_match)
                if matcher.is_isomorphic():
                    # Compute match score based on escalation similarity
                    match_score = _compute_match_score(window, prefix_codes)
                    chain = traj["chain"]

                    # Get timestamps for timespan
                    matched_events = chain[start:start + len(prefix_codes)]

                    play_id = f"play_{hashlib.md5(f'{sp["id"]}_{traj["id"]}_{start}'.encode(), usedforsecurity=False).hexdigest()[:12]}"
                    cseq_id = traj["id"].replace("traj_", "cseq_").replace("atraj_", "cseq_a_")

                    plays.append({
                        "id": play_id,
                        "skeletal_play_id": sp["id"],
                        "causal_sequence_id": cseq_id,
                        "matched_codes": window,
                        "match_score": match_score,
                        "match_length": len(prefix_codes),
                        "location": traj.get("location", ""),
                        "event_ids": matched_events,
                    })
                    break  # One match per trajectory per play

    return plays


def _compute_match_score(actual_codes: List[str], template_codes: List[str]) -> float:
    """Compute match score (0-1) based on exact code match percentage."""
    if not template_codes:
        return 0.0
    matches = sum(1 for a, t in zip(actual_codes, template_codes) if a == t)
    return round(matches / len(template_codes), 3)


# ---------------------------------------------------------------------------
# Add discriminative scores to skeletal plays
# ---------------------------------------------------------------------------

def _enrich_skeletal_plays(
    skeletal_plays: List[Dict],
    discriminative_patterns: List[Dict],
) -> List[Dict]:
    """Add discriminative scores to skeletal plays by matching patterns."""
    disc_lookup = {}
    for dp in discriminative_patterns:
        key = tuple(dp["pattern_codes"])
        disc_lookup[key] = dp

    for sp in skeletal_plays:
        codes = tuple(sp["pattern_codes"])
        # Check all sub-patterns of length 3
        best_score = 0.0
        best_direction = None
        for i in range(len(codes) - 2):
            sub = codes[i:i + 3]
            dp = disc_lookup.get(sub)
            if dp and abs(dp["log_odds"]) > abs(best_score):
                best_score = dp["log_odds"]
                best_direction = dp["direction"]

        sp["discriminative_score"] = round(best_score, 4) if best_direction else None
        sp["discriminative_direction"] = best_direction

    return skeletal_plays


# ---------------------------------------------------------------------------
# Write V2 tables to SQLite
# ---------------------------------------------------------------------------

def _write_v2_tables(
    db_path: str,
    skeletal_plays: List[Dict],
    causal_sequences: List[Dict],
    plays: List[Dict],
    retaliations: List[Dict],
    coordination_windows: List[Dict],
    discriminative_patterns: List[Dict],
    utility_scores: List[Dict],
    run_stats: Dict,
):
    """Write all V2 mining tables to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.executescript(V2_MINING_SCHEMA)

    # Skeletal plays
    for sp in skeletal_plays:
        conn.execute(
            """INSERT OR REPLACE INTO skeletal_plays
               (id, name, pattern_codes, domain_sequence, escalation_sequence,
                support, location_count, locations, outcome_distribution,
                discriminative_score, discriminative_direction, source, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sp["id"], sp["name"],
             json.dumps(sp["pattern_codes"]),
             json.dumps(sp["domain_sequence"]),
             json.dumps(sp.get("escalation_sequence")),
             sp["support"], sp.get("location_count", 0),
             json.dumps(sp.get("locations", [])),
             json.dumps(sp.get("outcome_distribution", {})),
             sp.get("discriminative_score"),
             sp.get("discriminative_direction"),
             sp.get("source", "prefixspan_closed"),
             json.dumps(sp.get("metadata", {}))),
        )

    # Causal sequences
    for cs in causal_sequences:
        conn.execute(
            """INSERT OR REPLACE INTO causal_sequences
               (id, action_ids, causal_edges, trajectory_code, outcome,
                location_id, chain_type, actor_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cs["id"],
             json.dumps(cs["action_ids"]),
             json.dumps(cs["causal_edges"]),
             json.dumps(cs["trajectory_code"]),
             cs["outcome"],
             cs.get("location_id"),
             cs.get("chain_type", "location"),
             cs.get("actor_id"),
             json.dumps(cs.get("metadata", {}))),
        )

    # Plays
    for p in plays:
        conn.execute(
            """INSERT OR REPLACE INTO plays
               (id, skeletal_play_id, causal_sequence_id, matched_codes,
                match_score, match_length, location, timespan_start,
                timespan_end, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["id"], p["skeletal_play_id"], p["causal_sequence_id"],
             json.dumps(p["matched_codes"]),
             p["match_score"], p["match_length"],
             p.get("location"),
             p.get("timespan_start"),
             p.get("timespan_end"),
             json.dumps({"event_ids": p.get("event_ids", [])})),
        )

    # Retaliation motifs
    for r in retaliations:
        conn.execute(
            """INSERT OR REPLACE INTO retaliation_motifs
               (id, event_a_id, event_b_id, retaliating_actor_id,
                original_actor_id, domain_transition, escalation_behavior,
                delta_days, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["id"], r["event_a_id"], r["event_b_id"],
             r["retaliating_actor_id"], r.get("original_actor_id"),
             r["domain_transition"], r["escalation_behavior"],
             r.get("delta_days", 0),
             json.dumps(r.get("metadata", {}))),
        )

    # Coordination windows
    for cw in coordination_windows:
        conn.execute(
            """INSERT OR REPLACE INTO coordination_windows
               (id, actor_id, domains, event_ids, window_start,
                window_end, span_days, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cw["id"], cw["actor_id"],
             json.dumps(cw["domains"]),
             json.dumps(cw["event_ids"]),
             cw["window_start"], cw["window_end"],
             cw.get("span_days", 0),
             json.dumps({"actor_name": cw.get("actor_name", "")})),
        )

    # Discriminative patterns
    for dp in discriminative_patterns:
        conn.execute(
            """INSERT OR REPLACE INTO discriminative_patterns
               (id, pattern_codes, log_odds, direction,
                support_escalating, support_deescalating, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (dp["id"],
             json.dumps(dp["pattern_codes"]),
             dp["log_odds"], dp["direction"],
             dp["support_escalating"], dp["support_deescalating"],
             json.dumps(dp.get("metadata", {}))),
        )

    # MDUS utility scores
    for us in utility_scores:
        conn.execute(
            """INSERT OR REPLACE INTO utility_scores
               (id, pattern_id, pattern_type, mdus_median, mdus_min,
                mdus_q25, mdus_q75, mdus_max, mdus_std,
                escalation_magnitude, cross_domain_spread,
                strategic_leverage, actor_type_multiplier, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (us["id"], us["pattern_id"], us["pattern_type"],
             us["mdus_median"], us["mdus_min"],
             us["mdus_q25"], us["mdus_q75"], us["mdus_max"], us["mdus_std"],
             us["escalation_magnitude"], us["cross_domain_spread"],
             us["strategic_leverage"], us["actor_type_multiplier"],
             json.dumps(us.get("metadata", {}))),
        )

    # Mining run metadata
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conn.execute(
        "INSERT INTO mining_runs (id, run_timestamp, parameters, stats) VALUES (?, ?, ?, ?)",
        (run_id, datetime.now().isoformat(),
         json.dumps({
             "min_support": MIN_SUPPORT,
             "max_pattern_length": MAX_PATTERN_LENGTH,
             "vf2_prefix_length": VF2_PREFIX_LENGTH,
             "coordination_window_days": COORDINATION_WINDOW_DAYS,
         }),
         json.dumps(run_stats)),
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------

def _export_parquet(
    db_path: str,
    s3_bucket: str,
    s3_key_prefix: str,
) -> Dict[str, str]:
    """Export V2 tables as Parquet files to S3.

    Returns dict of table_name -> S3 URI.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    conn = sqlite3.connect(db_path)

    _V2_QUERIES = {
        "skeletal_plays":          "SELECT * FROM skeletal_plays",
        "causal_sequences":        "SELECT * FROM causal_sequences",
        "plays":                   "SELECT * FROM plays",
        "retaliation_motifs":      "SELECT * FROM retaliation_motifs",
        "coordination_windows":    "SELECT * FROM coordination_windows",
        "discriminative_patterns": "SELECT * FROM discriminative_patterns",
        "utility_scores":          "SELECT * FROM utility_scores",
        "mining_runs":             "SELECT * FROM mining_runs",
    }

    exported = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for table, query in _V2_QUERIES.items():
        try:
            df = pd.read_sql_query(query, conn)
        except Exception:
            continue

        if df.empty:
            continue

        local_path = os.path.join(tempfile.gettempdir(), f"{table}.parquet")
        df.to_parquet(local_path, engine="pyarrow", index=False)

        s3_key = f"{s3_key_prefix}v2_types/{table}_{timestamp}.parquet"
        s3_latest_key = f"{s3_key_prefix}v2_types/{table}.parquet"

        s3_client.upload_file(local_path, s3_bucket, s3_key)
        s3_client.upload_file(local_path, s3_bucket, s3_latest_key)

        exported[table] = f"s3://{s3_bucket}/{s3_latest_key}"

        # Clean up local file
        os.remove(local_path)

    conn.close()
    return exported


# ---------------------------------------------------------------------------
# Hypergraph enrichment: add mined types as nodes + edges
# ---------------------------------------------------------------------------

def _enrich_hypergraph(
    db_path: str,
    skeletal_plays: List[Dict],
    causal_sequences: List[Dict],
    play_instances: List[Dict],
):
    """Add mined types as first-class nodes + edges in the hypergraph.

    New node types:
        - SkeletalPlay: Template patterns mined via PrefixSpan (Type #9)
        - CausalSequence: Ordered chains with causal links (Type #5)
        - Play: Matched instances of skeletal plays (Type #10)

    New edge roles:
        - INSTANCE_OF: Play → SkeletalPlay (play is instance of template)
        - CONTAINS: CausalSequence → Event (sequence contains this event)
        - MATCHED_IN: Play → CausalSequence (play matched in this sequence)
        - HAS_TEMPLATE_STEP: SkeletalPlay → domain code string (ordered)
    """
    conn = sqlite3.connect(db_path)

    # Insert SkeletalPlay nodes
    for sp in skeletal_plays:
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, name, category, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (sp["id"], "SkeletalPlay", sp["name"],
             sp.get("source", "prefixspan_closed"),
             json.dumps({
                 "pattern_codes": sp["pattern_codes"],
                 "domain_sequence": sp["domain_sequence"],
                 "support": sp["support"],
                 "location_count": sp.get("location_count", 0),
                 "outcome_distribution": sp.get("outcome_distribution", {}),
                 "discriminative_score": sp.get("discriminative_score"),
                 "discriminative_direction": sp.get("discriminative_direction"),
             })),
        )

    # Insert CausalSequence nodes
    for cs in causal_sequences:
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, name, category, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (cs["id"], "CausalSequence",
             " → ".join(cs["trajectory_code"][:5]) + ("..." if len(cs["trajectory_code"]) > 5 else ""),
             cs.get("chain_type", "location"),
             json.dumps({
                 "outcome": cs["outcome"],
                 "length": len(cs["action_ids"]),
                 "chain_type": cs.get("chain_type", "location"),
             })),
        )

        # CONTAINS edges: CausalSequence → each Event
        for evt_id in cs["action_ids"]:
            conn.execute(
                "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
                (cs["id"], evt_id, "CONTAINS",
                 json.dumps({"sequence_id": cs["id"]})),
            )

    # Insert Play nodes
    for p in play_instances:
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, name, category, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (p["id"], "Play",
             f"Match of {p['skeletal_play_id']} (score={p['match_score']:.2f})",
             p.get("location", ""),
             json.dumps({
                 "match_score": p["match_score"],
                 "match_length": p["match_length"],
                 "matched_codes": p["matched_codes"],
                 "event_ids": p.get("event_ids", []),
             })),
        )

        # INSTANCE_OF edge: Play → SkeletalPlay
        conn.execute(
            "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
            (p["id"], p["skeletal_play_id"], "INSTANCE_OF",
             json.dumps({"match_score": p["match_score"]})),
        )

        # MATCHED_IN edge: Play → CausalSequence
        conn.execute(
            "INSERT INTO edges (source_id, target_id, role, metadata) VALUES (?, ?, ?, ?)",
            (p["id"], p["causal_sequence_id"], "MATCHED_IN",
             json.dumps({"match_length": p["match_length"]})),
        )

    conn.commit()
    enrichment = {
        "skeletal_play_nodes": len(skeletal_plays),
        "causal_sequence_nodes": len(causal_sequences),
        "play_nodes": len(play_instances),
        "contains_edges": sum(len(cs["action_ids"]) for cs in causal_sequences),
        "instance_of_edges": len(play_instances),
        "matched_in_edges": len(play_instances),
    }
    conn.close()
    return enrichment


# ---------------------------------------------------------------------------
# Main entry point (Batch job or Lambda)
# ---------------------------------------------------------------------------

def run(
    hypergraph_db_uri: str = "",
    s3_bucket: str = "",
    s3_key_prefix: str = "sqlite/",
) -> Dict[str, Any]:
    """Core mining pipeline. Called by both Batch and Lambda entry points.

    Args:
        hypergraph_db_uri: S3 URI of grayzone_hypergraph.db
        s3_bucket: Output S3 bucket
        s3_key_prefix: S3 key prefix for outputs

    Returns:
        Dict with enhanced_db_uri, parquet_uris, stats.
    """
    if not s3_bucket:
        s3_bucket = os.environ.get("S3_BUCKET", "sirius-dimefiled-results")

    # Parse S3 URI and download the SQLite database
    if hypergraph_db_uri.startswith("s3://"):
        parts = hypergraph_db_uri.replace("s3://", "").split("/", 1)
        src_bucket = parts[0]
        src_key = parts[1]
    else:
        src_bucket = s3_bucket
        src_key = f"{s3_key_prefix}grayzone_hypergraph.db"

    db_path = os.path.join(tempfile.gettempdir(), "grayzone_hypergraph.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    s3_client.download_file(src_bucket, src_key, db_path)

    print(f"Downloaded SQLite from s3://{src_bucket}/{src_key}")

    # Load graph data
    nodes, all_edges = _load_graph_data(db_path)

    event_count = sum(1 for n in nodes.values() if n.get("node_type") == "Event")
    next_count = len(all_edges.get("NEXT", []))
    next_actor_count = len(all_edges.get("NEXT_ACTOR", []))
    print(f"Loaded: {event_count} events, {next_count} NEXT edges, {next_actor_count} NEXT_ACTOR edges")

    # Step 1: Extract trajectories
    loc_trajs, actor_trajs = _build_trajectories(nodes, all_edges)
    all_trajs = loc_trajs + actor_trajs
    print(f"Trajectories: {len(loc_trajs)} location-based, {len(actor_trajs)} actor-based")

    # Step 2: PrefixSpan closed pattern mining (on location trajectories)
    skeletal_plays = _mine_closed_patterns(loc_trajs, min_support=MIN_SUPPORT)
    print(f"Mined {len(skeletal_plays)} closed skeletal play patterns (min_support={MIN_SUPPORT})")

    # Also mine actor trajectories separately
    actor_plays = _mine_closed_patterns(actor_trajs, min_support=MIN_SUPPORT)
    for ap in actor_plays:
        ap["source"] = "prefixspan_closed_actor"
        ap["id"] = ap["id"].replace("sp_", "spa_")
    print(f"Mined {len(actor_plays)} actor-based patterns")

    # Step 3: Build causal sequences
    causal_sequences = _build_causal_sequences(nodes, all_trajs)
    print(f"Built {len(causal_sequences)} causal sequences")

    # Step 4: Temporal motif discovery
    retaliations = _detect_retaliations(nodes, all_edges)
    print(f"Detected {len(retaliations)} retaliation motifs")

    coordination_windows = _detect_coordination_windows(nodes, all_edges)
    print(f"Detected {len(coordination_windows)} coordination windows")

    # Step 5: Discriminative analysis
    disc_patterns = _compute_discriminative_patterns(loc_trajs, n=3, min_support=2)
    print(f"Computed {len(disc_patterns)} discriminative patterns")

    # Enrich skeletal plays with discriminative scores
    all_skeletal = skeletal_plays + actor_plays
    all_skeletal = _enrich_skeletal_plays(all_skeletal, disc_patterns)

    # Step 6: VF2 matching against top skeletal plays
    top_plays = [sp for sp in skeletal_plays if sp["support"] >= MIN_SUPPORT][:50]
    play_instances = _match_plays(loc_trajs, top_plays, prefix_length=VF2_PREFIX_LENGTH)
    print(f"Matched {len(play_instances)} play instances via VF2")

    # Step 6b: MDUS utility scoring
    utility_scores = _compute_pattern_utility(
        nodes, all_trajs, all_skeletal, play_instances,
    )
    print(f"Computed {len(utility_scores)} MDUS utility scores")

    # Step 7: Write V2 type tables to SQLite
    stats = {
        "events": event_count,
        "next_edges": next_count,
        "next_actor_edges": next_actor_count,
        "location_trajectories": len(loc_trajs),
        "actor_trajectories": len(actor_trajs),
        "skeletal_plays_location": len(skeletal_plays),
        "skeletal_plays_actor": len(actor_plays),
        "causal_sequences": len(causal_sequences),
        "retaliation_motifs": len(retaliations),
        "coordination_windows": len(coordination_windows),
        "discriminative_patterns": len(disc_patterns),
        "play_instances": len(play_instances),
        "utility_scores": len(utility_scores),
    }

    _write_v2_tables(
        db_path, all_skeletal, causal_sequences, play_instances,
        retaliations, coordination_windows, disc_patterns, utility_scores, stats,
    )
    print("Wrote V2 tables to SQLite")

    # Step 8: Enrich hypergraph with new node types + edges
    enrichment = _enrich_hypergraph(
        db_path, all_skeletal, causal_sequences, play_instances,
    )
    stats.update(enrichment)
    print(f"Enriched hypergraph: {enrichment['skeletal_play_nodes']} SkeletalPlay nodes, "
          f"{enrichment['causal_sequence_nodes']} CausalSequence nodes, "
          f"{enrichment['play_nodes']} Play nodes, "
          f"{enrichment['contains_edges']} CONTAINS edges")

    # Step 9: Export V2 tables as Parquet to S3
    parquet_uris = _export_parquet(db_path, s3_bucket, s3_key_prefix)
    print(f"Exported {len(parquet_uris)} Parquet files to S3")

    # Re-upload enhanced SQLite to S3
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    enhanced_key = f"{s3_key_prefix}grayzone_hypergraph_{ts}.db"
    enhanced_latest_key = f"{s3_key_prefix}grayzone_hypergraph.db"

    s3_client.upload_file(db_path, s3_bucket, enhanced_key)
    s3_client.upload_file(db_path, s3_bucket, enhanced_latest_key)
    print(f"Uploaded enhanced SQLite to s3://{s3_bucket}/{enhanced_latest_key}")

    # Write results JSON for Step Functions to pick up
    result = {
        "enhanced_db_uri": f"s3://{s3_bucket}/{enhanced_latest_key}",
        "parquet_uris": parquet_uris,
        "stats": stats,
    }
    result_key = f"{s3_key_prefix}mine_patterns_result.json"
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=result_key,
        Body=json.dumps(result, indent=2),
        ContentType="application/json",
    )
    print(f"Wrote result JSON to s3://{s3_bucket}/{result_key}")

    return result


def handler(event, context):
    """Lambda entry point (kept for backward compatibility)."""
    return run(
        hypergraph_db_uri=event.get("hypergraph_db_uri", ""),
        s3_bucket=event.get("s3_bucket", ""),
        s3_key_prefix=event.get("s3_key_prefix", "sqlite/"),
    )


# ---------------------------------------------------------------------------
# Batch job entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Entry point when run as a Batch job container.

    Reads configuration from environment variables:
        HYPERGRAPH_DB_URI: S3 URI of grayzone_hypergraph.db
        S3_BUCKET: Output S3 bucket
        S3_KEY_PREFIX: S3 key prefix (default: sqlite/)
    """
    import sys

    db_uri = os.environ.get("HYPERGRAPH_DB_URI", "")
    bucket = os.environ.get("S3_BUCKET", "sirius-dimefiled-results")
    prefix = os.environ.get("S3_KEY_PREFIX", "sqlite/")

    if not db_uri:
        db_uri = f"s3://{bucket}/{prefix}grayzone_hypergraph.db"

    print(f"=== MinePatterns Batch Job ===")
    print(f"DB URI: {db_uri}")
    print(f"S3 Bucket: {bucket}")
    print(f"S3 Prefix: {prefix}")
    print(f"{'=' * 40}")

    try:
        result = run(
            hypergraph_db_uri=db_uri,
            s3_bucket=bucket,
            s3_key_prefix=prefix,
        )
        print(f"\n=== COMPLETED ===")
        print(json.dumps(result.get("stats", {}), indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"\n=== FAILED ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)
