"""Consolidation: aggregate, sequence, reinforce, decay.

Handles the Atomic Architecture lifecycle:
1. Aggregate: group infons by (S, P, O) into constraints
2. Sequence: build NEXT edges — temporal chains per shared anchor
3. Reinforce: when a duplicate triple arrives, boost importance
4. Decay: reduce importance of stale infons over time
5. Prune: soft-delete infons below threshold
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from .infon import Constraint, Edge, Infon


def aggregate_constraints(infons: list[Infon]) -> list[Constraint]:
    """Group infons by (S, P, O) triple into constraints.

    Each constraint tracks evidence count, doc count, mean confidence,
    temporal persistence (distinct month windows), and a composite score.
    """
    groups: dict[tuple, list[Infon]] = defaultdict(list)
    for inf in infons:
        groups[inf.triple_key()].append(inf)

    constraints = []
    for (subj, pred, obj), group in groups.items():
        doc_ids = set()
        time_windows = set()
        confidences = []

        for inf in group:
            doc_ids.add(inf.doc_id)
            confidences.append(inf.confidence)
            if inf.timestamp:
                time_windows.add(inf.timestamp[:7])  # monthly bucket

        evidence = len(group)
        doc_count = len(doc_ids)
        strength = sum(confidences) / len(confidences) if confidences else 0.0
        persistence = len(time_windows)

        # Composite score
        score = _constraint_score(evidence, doc_count, strength, persistence)

        constraints.append(Constraint(
            subject=subj,
            predicate=pred,
            object=obj,
            evidence=evidence,
            doc_count=doc_count,
            strength=strength,
            persistence=persistence,
            score=score,
            infon_ids=[inf.infon_id for inf in group],
        ))

    constraints.sort(key=lambda c: -c.score)
    return constraints


def _constraint_score(evidence: int, doc_count: int,
                      strength: float, persistence: int) -> float:
    """Compute constraint composite score."""
    # Normalize evidence and persistence with log scaling
    n_ev = math.log1p(evidence) / math.log1p(100)  # caps at ~4.6
    n_dc = math.log1p(doc_count) / math.log1p(50)
    n_pe = math.log1p(persistence) / math.log1p(24)  # 24 months
    return 0.25 * n_ev + 0.25 * n_dc + 0.25 * strength + 0.25 * n_pe


def build_next_edges(infons: list[Infon]) -> list[Edge]:
    """Build NEXT edges: temporal chains per shared anchor.

    For each anchor, sort all infons mentioning that anchor by timestamp,
    then link consecutive infons with NEXT edges carrying the shared
    anchor name, role, and gap in days.
    """
    # Index: anchor → [(infon, role, timestamp)]
    anchor_chains: dict[str, list[tuple[Infon, str, str]]] = defaultdict(list)

    for inf in infons:
        if not inf.timestamp:
            continue
        anchor_chains[inf.subject].append((inf, "subject", inf.timestamp))
        anchor_chains[inf.predicate].append((inf, "predicate", inf.timestamp))
        anchor_chains[inf.object].append((inf, "object", inf.timestamp))

    edges = []
    for anchor, entries in anchor_chains.items():
        # Sort by timestamp
        entries.sort(key=lambda x: x[2])

        for i in range(len(entries) - 1):
            inf_a, role_a, ts_a = entries[i]
            inf_b, role_b, ts_b = entries[i + 1]

            if inf_a.infon_id == inf_b.infon_id:
                continue

            gap_days = _days_between(ts_a, ts_b)

            edges.append(Edge(
                source=inf_a.infon_id,
                target=inf_b.infon_id,
                edge_type="NEXT",
                weight=1.0,
                metadata={
                    "anchor": anchor,
                    "anchor_role": role_a,
                    "gap_days": gap_days,
                },
            ))

    return edges


def _days_between(ts_a: str, ts_b: str) -> int:
    """Compute days between two ISO timestamps (best-effort)."""
    try:
        # Handle various formats: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, YYYY-MM
        da = _parse_date(ts_a)
        db = _parse_date(ts_b)
        return abs((db - da).days)
    except (ValueError, TypeError):
        return 0


def _parse_date(ts: str) -> datetime:
    """Parse an ISO-ish timestamp to datetime."""
    ts = ts.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(ts[:len(fmt.replace("%", "0"))], fmt)
        except ValueError:
            continue
    # Last resort: just the year
    return datetime.strptime(ts[:4], "%Y")


def reinforce(existing: Infon, new_infon: Infon, config) -> Infon:
    """Reinforce an existing infon when a duplicate triple arrives.

    Boosts importance via reinforcement_count and updates last_reinforced.
    Returns the updated existing infon (mutated in place).
    """
    existing.reinforcement_count += 1
    existing.last_reinforced = new_infon.timestamp or datetime.now(tz=None).isoformat()

    # Recompute importance with reinforcement bonus
    existing.importance = (
        config.w_activation * existing.activation
        + config.w_coherence * existing.coherence
        + config.w_specificity * existing.specificity
        + config.w_novelty * existing.novelty
        + config.w_reinforcement * math.log1p(existing.reinforcement_count)
    )

    # Average in the new confidence
    n = existing.reinforcement_count + 1
    existing.confidence = (
        existing.confidence * (n - 1) + new_infon.confidence
    ) / n

    return existing


def apply_decay(infons: list[Infon], reference_date: str | None = None) -> list[Infon]:
    """Apply temporal importance decay to infons.

    importance -= decay_rate * days_since_last_reinforced

    Returns the list with importance values updated.
    """
    if reference_date:
        ref = _parse_date(reference_date)
    else:
        ref = datetime.now(tz=None)

    for inf in infons:
        ts = inf.last_reinforced or inf.timestamp
        if not ts:
            continue
        try:
            last = _parse_date(ts)
            days = max(0, (ref - last).days)
            decay = inf.decay_rate * days
            inf.importance = max(0.0, inf.importance - decay)
        except (ValueError, TypeError):
            pass

    return infons


def consolidate(
    new_infons: list[Infon],
    store,
    config,
) -> tuple[list[Constraint], list[Edge]]:
    """Full consolidation pass: reinforce, aggregate, sequence.

    1. Check each new infon against existing (S, P, O) in the store.
       If duplicate, reinforce. If new, insert.
    2. Build constraints from all infons.
    3. Build NEXT edges from all timestamped infons.

    Returns (constraints, next_edges).
    """
    # Phase 1: Reinforce or insert
    for inf in new_infons:
        existing = store.query_infons(
            subject=inf.subject,
            predicate=inf.predicate,
            object=inf.object,
            limit=1,
        )
        if existing:
            reinforced = reinforce(existing[0], inf, config)
            store.put_infon(reinforced)
            # Set novelty low for reinforced infons
            inf.novelty = 0.1
        else:
            store.put_infon(inf)

    # Phase 2: Aggregate constraints
    all_infons = store.query_infons(limit=10000)
    constraints = aggregate_constraints(all_infons)
    for c in constraints:
        store.put_constraint(c)

    # Phase 3: Build NEXT edges
    next_edges = build_next_edges(all_infons)
    store.put_edges(next_edges)

    return constraints, next_edges
