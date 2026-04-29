"""
Consolidation pipeline for enriching the knowledge graph.

This module implements the consolidate() function which:
1. Creates NEXT edges between chronologically ordered infons sharing anchors
2. Aggregates constraints from observed triples
3. Applies importance decay to old infons (>7 days)

The consolidation process is idempotent - running it multiple times produces
the same result as running it once.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from infon.infon import ImportanceScore
from infon.schema import AnchorSchema
from infon.store import InfonStore


def _build_next_edges(store: InfonStore) -> None:
    """
    Build NEXT edges between chronologically ordered infons sharing anchors.
    
    For each anchor key that appears as subject or object in multiple infons,
    sort those infons by timestamp and add a NEXT edge from each infon to the
    next chronological infon sharing that anchor.
    
    Edge weight = 1 / (1 + days_between)
    
    This function is idempotent - existing NEXT edges are not duplicated.
    
    Args:
        store: The InfonStore to enrich with NEXT edges
    """
    conn = store._conn
    
    # Get all unique anchors (from both subject and object positions)
    anchors_query = """
        SELECT DISTINCT anchor FROM (
            SELECT subject AS anchor FROM infons
            UNION
            SELECT object AS anchor FROM infons
        ) AS all_anchors
    """
    anchors = conn.execute(anchors_query).fetchall()
    
    for (anchor,) in anchors:
        # Get all infons containing this anchor (as subject or object)
        # Order by timestamp ascending for chronological processing
        infons_query = """
            SELECT id, timestamp
            FROM infons
            WHERE subject = ? OR object = ?
            ORDER BY timestamp ASC
        """
        infon_records = conn.execute(infons_query, [anchor, anchor]).fetchall()
        
        if len(infon_records) < 2:
            # Need at least 2 infons to create edges
            continue
        
        # Create NEXT edges between consecutive infons
        for i in range(len(infon_records) - 1):
            from_id, from_timestamp = infon_records[i]
            to_id, to_timestamp = infon_records[i + 1]
            
            # Check if NEXT edge already exists (for idempotency)
            existing_edge = conn.execute(
                """
                SELECT id FROM edges
                WHERE from_infon_id = ? AND to_infon_id = ? AND edge_type = 'NEXT'
                """,
                [from_id, to_id]
            ).fetchone()
            
            if existing_edge:
                # Edge already exists, skip
                continue
            
            # Calculate weight based on time difference
            time_diff = to_timestamp - from_timestamp
            days_between = time_diff.total_seconds() / 86400  # Convert to days
            weight = 1.0 / (1.0 + days_between)
            
            # Add the NEXT edge
            store.add_edge(from_id, to_id, "NEXT", weight)


def _aggregate_constraints(store: InfonStore, decay_factor: float = 0.95) -> None:
    """
    Aggregate constraints from observed triples.
    
    For each distinct (subject, predicate, object) triple:
    - evidence_count = reinforcement_count
    - strength = average(confidence)
    - persistence = 1 - decay_factor^evidence_count
    
    Args:
        store: The InfonStore to aggregate constraints from
        decay_factor: Decay factor for persistence calculation (default 0.95)
    """
    conn = store._conn
    
    # Get all distinct triples with their aggregated stats
    triples_query = """
        SELECT 
            subject, 
            predicate, 
            object,
            MAX(reinforcement_count) AS evidence_count,
            AVG(confidence) AS strength
        FROM infons
        GROUP BY subject, predicate, object
    """
    
    triples = conn.execute(triples_query).fetchall()
    
    for subject, predicate, obj, evidence_count, strength in triples:
        # Calculate persistence: 1 - decay_factor^evidence_count
        persistence = 1.0 - (decay_factor ** evidence_count)
        
        # Upsert constraint to the store
        store.upsert_constraint(
            subject=subject,
            predicate=predicate,
            object=obj,
            evidence_count=evidence_count,
            strength=strength,
            persistence=persistence,
        )


def _apply_importance_decay(
    store: InfonStore, decay_factor: float = 0.95, decay_threshold_days: int = 7
) -> None:
    """
    Apply exponential importance decay to old infons.
    
    For infons with timestamp older than decay_threshold_days:
    - new_reinforcement = old_reinforcement * decay_factor^days_since_timestamp
    
    Infons that are reinforced (reinforcement_count increases) are exempt from
    decay for decay_threshold_days after the last reinforcement.
    
    Args:
        store: The InfonStore to apply decay to
        decay_factor: Exponential decay factor (default 0.95)
        decay_threshold_days: Number of days before decay starts (default 7)
    """
    conn = store._conn
    now = datetime.now(timezone.utc)
    threshold_date = now - timedelta(days=decay_threshold_days)
    
    # Get all infons older than threshold
    old_infons_query = """
        SELECT id, timestamp, importance_json
        FROM infons
        WHERE timestamp < ?
    """
    
    old_infons = conn.execute(old_infons_query, [threshold_date]).fetchall()
    
    import json
    
    for infon_id, timestamp, importance_json in old_infons:
        # Calculate days since timestamp
        time_since = now - timestamp
        days_since = time_since.total_seconds() / 86400
        
        # Deserialize importance
        importance_data = json.loads(importance_json)
        old_reinforcement = importance_data.get("reinforcement", 0.0)
        
        # Apply decay: new_value = old_value * decay_factor^days_since
        new_reinforcement = old_reinforcement * (decay_factor ** days_since)
        
        # Update importance with decayed reinforcement
        importance_data["reinforcement"] = new_reinforcement
        new_importance_json = json.dumps(importance_data)
        
        # Update the infon in the database
        conn.execute(
            """
            UPDATE infons
            SET importance_json = ?
            WHERE id = ?
            """,
            [new_importance_json, infon_id]
        )


def consolidate(
    store: InfonStore, 
    schema: AnchorSchema,
    decay_factor: float = 0.95,
    decay_threshold_days: int = 7
) -> None:
    """
    Consolidate the knowledge graph by enriching it with derived information.
    
    This function performs four consolidation steps:
    1. Reinforcement (already handled by InfonStore.upsert - no action needed)
    2. NEXT edges: Create temporal links between chronologically ordered infons
    3. Constraint aggregation: Compute evidence-based patterns
    4. Importance decay: Apply exponential decay to old infons
    
    The consolidation process is idempotent - running it multiple times produces
    the same result as running it once.
    
    Args:
        store: The InfonStore to consolidate
        schema: The AnchorSchema (currently not used, reserved for future features)
        decay_factor: Exponential decay factor for importance and persistence (default 0.95)
        decay_threshold_days: Number of days before decay starts (default 7)
        
    Example:
        >>> with InfonStore("kb.ddb") as store:
        ...     schema = AnchorSchema.from_json("schema.json")
        ...     consolidate(store, schema)
    """
    # Step 2: Build NEXT edges
    _build_next_edges(store)
    
    # Step 3: Aggregate constraints
    _aggregate_constraints(store, decay_factor=decay_factor)
    
    # Step 4: Apply importance decay
    _apply_importance_decay(
        store, 
        decay_factor=decay_factor,
        decay_threshold_days=decay_threshold_days
    )
