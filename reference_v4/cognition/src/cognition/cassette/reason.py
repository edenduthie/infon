"""Reasoner adapter — DSL → hydrated evidence → calibrated verdict.

The adapter is deliberately small:
  1. A `Query` produces index hits (pruned).
  2. Range-get hydrates the matched infons.
  3. Each infon contributes a DS mass function against the claim.
  4. Dempster combines masses across the top-K most decisive.
  5. Return verdict + mass + the source infons that drove it.

Key design choice: the *claim* is expressed as a `Query`, not free text.
This avoids entity resolution (the DSL already names the anchors) and
makes the reasoner deterministic — same query, same verdict, every time.

Free-text claims can still use this by building the Query externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..infon import Infon
from ..dempster_shafer import (
    MassFunction, combine_multiple,
    mass_from_polarity, mass_from_confidence,
)
from .dsl import Query
from .index import Manifest
from .reader import RangeFetcher, LocalFetcher, hydrate_locs


# ═══════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Verdict:
    """Reasoning result — mirrors HypergraphReasoner's output shape."""
    label: str                           # SUPPORTS / REFUTES / NOT_ENOUGH_INFO
    mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    claim: Query | None = None
    sources: list[Infon] = field(default_factory=list)
    n_candidates: int = 0
    n_hydrated: int = 0
    range_gets: int = 0

    def __repr__(self) -> str:
        m = self.mass
        return (f"Verdict({self.label}  S={m.supports:.2f}  R={m.refutes:.2f}  "
                f"θ={m.theta:.2f}  n={self.n_hydrated})")


# ═══════════════════════════════════════════════════════════════════════
# MASS FROM EVIDENCE  (simpler than structural.verify_claim because the
# DSL has already committed to a specific claim triple — no claim-anchor
# activation to muddy things)
# ═══════════════════════════════════════════════════════════════════════

def _claim_triple(claim: Query) -> tuple[str | None, str | None, str | None, int]:
    """Extract the triple and expected polarity (default=affirmed=1)."""
    return (claim.subject, claim.predicate, claim.object,
            1 if claim.polarity is None else claim.polarity)


def _evidence_mass(infon: Infon, claim: Query) -> MassFunction:
    """Per-infon DS mass against the specific claim the Query expresses.

    Six cases, all with calibrated θ:
      a) exact triple match, polarity agrees → strong SUPPORTS
      b) exact triple match, polarity disagrees → strong REFUTES
      c) S+P match, different O → REFUTES (semantic contradiction)
      d) S only match → weak SUPPORTS or REFUTES by polarity
      e) partial overlap (P or O only) → θ-heavy
      f) no overlap → pure θ
    """
    cs, cp, co, cpol = _claim_triple(claim)

    s_match = cs is not None and infon.subject == cs
    p_match = cp is not None and infon.predicate == cp
    o_match = co is not None and infon.object == co

    # Multiplicative confidence prior (prevents low-conf extractions from
    # driving high-magnitude masses). Mirrors mass_from_confidence's role.
    c = max(0.0, min(infon.confidence, 1.0))
    w = 0.20 + 0.55 * c   # at c=0 → 0.20, at c=1 → 0.75

    # Case a/b: exact triple match
    if s_match and p_match and o_match:
        if infon.polarity == cpol:
            return MassFunction(supports=w * 0.9, theta=1.0 - w * 0.9)
        else:
            return MassFunction(refutes=w * 0.9, theta=1.0 - w * 0.9)

    # Case c: same S+P, different O — same fact asserted about a different thing
    if s_match and p_match and co is not None and not o_match:
        # Whether the evidence is polarity-flipped or not, saying
        # "X does P to Y" when the claim is "X does P to Z" argues against Z.
        return MassFunction(refutes=w * 0.55, theta=1.0 - w * 0.55)

    # Case d: subject only.
    # Downweight hard: an S-only match is barely evidence. If the claim
    # pins a predicate the corpus has never linked to this subject, we
    # should NOT tip SUPPORTS. The Dempster combiner amplifies any
    # positive S across infons, so even 0.15 per row compounds.
    if s_match and not p_match:
        if infon.polarity == cpol:
            return MassFunction(supports=w * 0.05, theta=1.0 - w * 0.05)
        else:
            return MassFunction(refutes=w * 0.03, theta=1.0 - w * 0.03)

    # Case e: just P or O match — weak but directional
    if p_match or o_match:
        if infon.polarity == cpol:
            return MassFunction(supports=w * 0.10, theta=1.0 - w * 0.10)
        else:
            return MassFunction(refutes=w * 0.08, theta=1.0 - w * 0.08)

    # Case f: nothing overlaps → no information
    return MassFunction(theta=1.0)


# ═══════════════════════════════════════════════════════════════════════
# REASONER
# ═══════════════════════════════════════════════════════════════════════

def reason(
    manifest: Manifest,
    claim: Query,
    *,
    fetcher: RangeFetcher | None = None,
    evidence_filter: Query | None = None,
    max_evidence: int = 20,
    supports_threshold: float = 0.25,
    refutes_threshold: float = 0.15,
) -> Verdict:
    """Answer whether `claim` is supported by cassette-stored evidence.

    Args:
      claim          — the claim expressed as a (partially) pinned Query.
                       Must name at least the subject, so we have something
                       to retrieve on.
      evidence_filter — optional Query used to gather candidate evidence.
                       Defaults to `Query().where(subject=claim.subject)` so
                       we hydrate every infon mentioning the claim's subject.
                       Pass a wider query for cross-entity reasoning.
      max_evidence   — cap on hydrated infons; we already rank by decisiveness
                       before combining, so this is mostly a cost bound.

    Returns a Verdict with mass, label, and the source infons that drove it.
    """
    if not claim.subject and not claim.predicate and not claim.object:
        return Verdict(label="NOT_ENOUGH_INFO", claim=claim)

    if fetcher is None:
        fetcher = LocalFetcher()
    before_gets = getattr(fetcher, "requests", 0)

    # 1. Build evidence query. Default: everything that mentions the claim's
    #    subject (broadest directional signal); claim's polarity is NOT
    #    propagated — we want both agreeing and refuting evidence.
    if evidence_filter is None:
        evidence_filter = Query()
        if claim.subject:
            evidence_filter = evidence_filter.where(subject=claim.subject)
        elif claim.object:
            evidence_filter = evidence_filter.where(object=claim.object)
        else:
            evidence_filter = evidence_filter.where(predicate=claim.predicate)

    hits = evidence_filter.run(manifest)
    n_candidates = len(hits)
    if not hits:
        v = Verdict(label="NOT_ENOUGH_INFO", claim=claim)
        v.n_candidates = 0
        return v

    # 2. Hydrate (cap by max_evidence — prefer higher-confidence rows).
    hits.sort(key=lambda h: -h.loc.confidence)
    hits = hits[:max_evidence]
    infons = hydrate_locs(fetcher, manifest, hits)

    # 3. Per-infon masses.
    masses = [_evidence_mass(inf, claim) for inf in infons]

    # 4. Combine top-5 most decisive (lowest-θ) — matches verify_claim.
    decisive = sorted(zip(masses, infons), key=lambda x: x[0].theta)[:5]
    if not decisive:
        claim_mass = MassFunction(theta=1.0)
        sources: list[Infon] = []
    else:
        claim_mass = combine_multiple([m for m, _ in decisive])
        sources = [inf for _, inf in decisive if masses[infons.index(inf)].theta < 0.95]

    # 5. Verdict from belief thresholds.
    if claim_mass.supports >= supports_threshold and \
       claim_mass.supports > claim_mass.refutes:
        label = "SUPPORTS"
    elif claim_mass.refutes >= refutes_threshold and \
         claim_mass.refutes > claim_mass.supports:
        label = "REFUTES"
    else:
        label = "NOT_ENOUGH_INFO"

    return Verdict(
        label=label,
        mass=claim_mass,
        claim=claim,
        sources=sources,
        n_candidates=n_candidates,
        n_hydrated=len(infons),
        range_gets=getattr(fetcher, "requests", 0) - before_gets,
    )


# ═══════════════════════════════════════════════════════════════════════
# BATCH HELPER  (reason over a list of claims; reuses the fetcher)
# ═══════════════════════════════════════════════════════════════════════

def reason_many(manifest: Manifest, claims: Iterable[Query],
                *, fetcher: RangeFetcher | None = None, **kw) -> list[Verdict]:
    fetcher = fetcher or LocalFetcher()
    return [reason(manifest, c, fetcher=fetcher, **kw) for c in claims]
