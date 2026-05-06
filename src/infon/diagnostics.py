"""Data diagnostics — is this corpus well-posed for the reasoner?

Six cheap checks that run against a populated InfonEngine store and
flag degenerate cases *before* any SSL / reasoning training. Motivated
by the planner-fails-on-hub-anchor result we hit at scale: the
diagnostics would have told us `battery` was a hub at a glance.

Usage
-----
    from infon.diagnostics import analyze_corpus
    report = analyze_corpus(cog)
    print(report.summary())
    if report.warnings:
        for w in report.warnings:
            print("  !", w)

Every number is computed from a single pass over the store — no model
is invoked. The goal is to give the user actionable signal about their
data *before* they spend compute on it.

The six probes
--------------
1. `H(O | S, P)`  — conditional entropy of the object anchor given
   (subject, predicate). The primary claim-verification signal. Near 0
   = deterministic / memorized; near log|O| = no signal to learn.
2. Hub concentration — max anchor frequency / number of infons. > 0.5
   means one anchor dominates and will confuse contrastive scoring.
3. Triple coverage — |observed triples| / |actors × relations × objects|.
   Low = sparse; near-saturated = no room for held-out NEI claims.
4. Temporal-edge density — #NEXT edges / #infons. Low = Temporal-JEPA
   has nothing to train on.
5. Relational asymmetry — fraction of relations where (s, r, o) and
   (o, r, s) distributions differ. Drives whether sheaf's P_fwd ≠ P_bwd
   has any asymmetry to exploit.
6. Role-marginal entropy — how balanced the subject / predicate /
   object distributions are. Imbalance = fewer effective training
   examples.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class DataReport:
    """Diagnostic summary of an InfonEngine corpus."""
    n_infons: int = 0
    n_unique_triples: int = 0
    n_actors: int = 0
    n_relations: int = 0
    n_objects: int = 0           # features + markets

    cond_entropy_H_O_given_SP: float = 0.0
    cond_entropy_normalized: float = 0.0      # / log|O|, ∈ [0, 1]

    hub_concentration: float = 0.0    # max anchor freq / n_infons
    hub_anchor: str = ""

    triple_coverage: float = 0.0      # ∈ [0, 1]

    temporal_edge_density: float = 0.0   # # NEXT / n_infons
    causes_edge_count: int = 0

    relational_asymmetry: float = 0.0    # fraction of rels with (s,r,o) ≠ (o,r,s)

    subject_entropy_norm: float = 0.0    # ∈ [0, 1]
    predicate_entropy_norm: float = 0.0
    object_entropy_norm: float = 0.0

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Infon corpus diagnostics",
            "─" * 40,
            f"  infons:            {self.n_infons}",
            f"  unique triples:    {self.n_unique_triples}",
            f"  anchors:           {self.n_actors} actors, "
            f"{self.n_relations} relations, {self.n_objects} objects",
            "",
            f"  H(O | S, P):       {self.cond_entropy_H_O_given_SP:.3f} nats"
            f"   ({self.cond_entropy_normalized:.0%} of max)",
            f"  hub concentration: {self.hub_concentration:.0%}"
            + (f"  (dominated by '{self.hub_anchor}')" if self.hub_anchor else ""),
            f"  triple coverage:   {self.triple_coverage:.1%}",
            f"  NEXT-edge density: {self.temporal_edge_density:.2f} per infon"
            f"    (CAUSES edges: {self.causes_edge_count})",
            f"  rel asymmetry:     {self.relational_asymmetry:.0%} "
            f"of relations are asymmetric",
            "",
            f"  role-marginal entropy (normalized, higher = more balanced):",
            f"    subject:   {self.subject_entropy_norm:.2f}",
            f"    predicate: {self.predicate_entropy_norm:.2f}",
            f"    object:    {self.object_entropy_norm:.2f}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "warnings"} | {
            "warnings": list(self.warnings),
        }


def _entropy(counts) -> float:
    """Shannon entropy in nats over a count dict / Counter."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def _conditional_entropy(triples: list[tuple[str, str, str]]) -> tuple[float, float]:
    """H(O | S, P) over observed triples.

    Returns (H, H_normalized_by_log_|O|). Normalized in [0, 1].
    """
    if not triples:
        return 0.0, 0.0
    # P(O | S, P) — collect per-(s, p) distributions over o
    sp_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    sp_totals: Counter = Counter()
    for s, p, o in triples:
        sp_counts[(s, p)][o] += 1
        sp_totals[(s, p)] += 1

    total = sum(sp_totals.values())
    objects = {o for (_, _, o) in triples}
    log_o = math.log(len(objects)) if len(objects) > 1 else 1.0

    h = 0.0
    for sp, cnt in sp_counts.items():
        weight = sp_totals[sp] / total
        h += weight * _entropy(cnt)
    return h, h / log_o


def _relational_asymmetry(triples: list[tuple[str, str, str]]) -> float:
    """Fraction of relations whose (s, r, o) distribution differs from
    (o, r, s) — measured as the fraction of (s, r, o) triples whose
    mirrored form is absent from the corpus."""
    triple_set = set(triples)
    if not triple_set:
        return 0.0
    n_asym = 0
    for s, r, o in triple_set:
        if (o, r, s) not in triple_set:
            n_asym += 1
    return n_asym / len(triple_set)


def analyze_corpus(cog,
                   max_infons: int = 5000,
                   hub_threshold: float = 0.4,
                   low_entropy_threshold: float = 0.1,
                   coverage_ceiling: float = 0.9,
                   temporal_density_floor: float = 0.2,
                   ) -> DataReport:
    """Run six cheap diagnostic probes against the cog's current store.

    Produces a DataReport with numbers + optional warnings when a
    measurement crosses a known-problematic threshold.

    Thresholds are conservative defaults — users with big specialized
    corpora should override them to match their regime.
    """
    rep = DataReport()

    # Pull infons (cheap — just read from store)
    infons = cog.store.query_infons(limit=max_infons)
    rep.n_infons = len(infons)
    if rep.n_infons == 0:
        rep.warnings.append("empty store — ingest some documents first")
        return rep

    triples = [(i.subject, i.predicate, i.object) for i in infons]
    unique = set(triples)
    rep.n_unique_triples = len(unique)

    # Anchor-type breakdown
    schema_types = cog.schema.types
    rep.n_actors = sum(1 for t in schema_types.values() if t == "actor")
    rep.n_relations = sum(1 for t in schema_types.values() if t == "relation")
    rep.n_objects = sum(1 for t in schema_types.values()
                        if t in ("feature", "market"))

    # 1. H(O | S, P)
    h, h_norm = _conditional_entropy(triples)
    rep.cond_entropy_H_O_given_SP = h
    rep.cond_entropy_normalized = h_norm

    if h_norm < low_entropy_threshold:
        rep.warnings.append(
            f"H(O|S,P) / log|O| = {h_norm:.2f} — very low conditional "
            f"entropy. Corpus likely deterministic; the reasoner will "
            f"memorize rather than generalize."
        )
    if h_norm > 0.95:
        rep.warnings.append(
            f"H(O|S,P) / log|O| = {h_norm:.2f} — near-maximum entropy; "
            f"no signal for the reasoner to learn. Consider tighter "
            f"activation_threshold or more distinctive schema anchors."
        )

    # 2. Hub concentration
    anchor_counts: Counter = Counter()
    for s, p, o in triples:
        anchor_counts[s] += 1
        anchor_counts[p] += 1
        anchor_counts[o] += 1
    total_roles = 3 * rep.n_infons
    if total_roles > 0:
        top_anchor, top_count = anchor_counts.most_common(1)[0]
        rep.hub_concentration = top_count / rep.n_infons
        rep.hub_anchor = top_anchor
        if rep.hub_concentration > hub_threshold:
            rep.warnings.append(
                f"'{top_anchor}' appears in {rep.hub_concentration:.0%} "
                f"of infons — this anchor is a hub. Expect weak contrastive "
                f"signal and poor latent-planner performance."
            )

    # 3. Triple coverage
    actors = {a for a, t in schema_types.items() if t == "actor"}
    rels = {a for a, t in schema_types.items() if t == "relation"}
    objs = {a for a, t in schema_types.items() if t in ("feature", "market")}
    max_triples = max(len(actors) * len(rels) * len(objs), 1)
    rep.triple_coverage = len(unique) / max_triples
    if rep.triple_coverage > coverage_ceiling:
        rep.warnings.append(
            f"triple coverage = {rep.triple_coverage:.1%}: the corpus "
            f"saturates the S×R×O product. Self-generated NEI queries "
            f"will be hard to construct because every role-swap lands "
            f"on something real."
        )

    # 4. Temporal-edge density
    try:
        edges = cog.store.get_edges(limit=max_infons * 5)
        next_edges = [e for e in edges if e.edge_type == "NEXT"]
        causes_edges = [e for e in edges if e.edge_type == "CAUSES"]
        rep.temporal_edge_density = len(next_edges) / max(rep.n_infons, 1)
        rep.causes_edge_count = len(causes_edges)
        if rep.temporal_edge_density < temporal_density_floor:
            rep.warnings.append(
                f"NEXT-edge density = {rep.temporal_edge_density:.2f}: "
                f"too few temporal edges for Temporal-JEPA to learn "
                f"transitions. Ingest documents with timestamps + shared "
                f"actors, or call cog.consolidate()."
            )
    except Exception:
        # Some stores may not expose get_edges uniformly; skip silently
        pass

    # 5. Relational asymmetry
    rep.relational_asymmetry = _relational_asymmetry(triples)
    # No warning: a low value just means sheaf's P_fwd/P_bwd asymmetry
    # is less useful. High value = sheaf will pay off.

    # 6. Role-marginal entropy (normalized by log of unique values used)
    def _norm_entropy(items):
        counter = Counter(items)
        if len(counter) <= 1:
            return 0.0
        return _entropy(counter) / math.log(len(counter))

    rep.subject_entropy_norm = _norm_entropy([s for s, _, _ in triples])
    rep.predicate_entropy_norm = _norm_entropy([p for _, p, _ in triples])
    rep.object_entropy_norm = _norm_entropy([o for _, _, o in triples])

    # Flag severe imbalance
    if min(rep.subject_entropy_norm,
           rep.predicate_entropy_norm,
           rep.object_entropy_norm) < 0.3:
        rep.warnings.append(
            "one or more role distributions is heavily skewed "
            "(entropy < 0.3). The reasoner will see few distinct examples "
            "on the minority roles."
        )

    return rep


# Convenience: tests want this importable too
__all__ = ["DataReport", "analyze_corpus"]
