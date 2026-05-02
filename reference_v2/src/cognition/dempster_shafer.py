"""Dempster-Shafer belief functions for infon-based fact verification.

The frame of discernment is Θ = {SUPPORTS, REFUTES, UNCERTAIN}.

Each evidence source produces a mass function m: 2^Θ → [0,1] representing
how much belief it assigns to each hypothesis. Dempster's rule of combination
fuses independent sources while redistributing conflicting mass.

Evidence sources:
  1. Lexical polarity — explicit negation cues in the sentence
  2. Triple alignment — claim (S,P,O) vs evidence (S',P',O') overlap
  3. Anchor distance — semantic proximity of claim/evidence anchors
  4. Confidence mass — model confidence as evidence strength

This replaces the naive "if polarity==0 → REFUTES" heuristic with a
principled belief calculus that can detect semantic contradictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .infon import Infon


# ═══════════════════════════════════════════════════════════════════════
# MASS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

# Frame of discernment
SUPPORTS = frozenset({"SUPPORTS"})
REFUTES = frozenset({"REFUTES"})
UNCERTAIN = frozenset({"UNCERTAIN"})
THETA = frozenset({"SUPPORTS", "REFUTES", "UNCERTAIN"})  # full ignorance

# All focal elements we use
FOCAL_ELEMENTS = [SUPPORTS, REFUTES, UNCERTAIN, THETA]


@dataclass
class MassFunction:
    """A basic probability assignment (BPA) on the frame {S, R, U}.

    Stores mass for each focal element. Masses must sum to 1.
    THETA represents total ignorance (mass distributed to all hypotheses).
    """
    supports: float = 0.0
    refutes: float = 0.0
    uncertain: float = 0.0
    theta: float = 1.0  # ignorance — default: complete ignorance

    def __post_init__(self):
        # Normalize to sum to 1
        total = self.supports + self.refutes + self.uncertain + self.theta
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.supports /= total
            self.refutes /= total
            self.uncertain /= total
            self.theta /= total

    def belief(self, hypothesis: frozenset) -> float:
        """Belief: sum of masses of all subsets of hypothesis."""
        bel = 0.0
        if SUPPORTS.issubset(hypothesis):
            bel += self.supports
        if REFUTES.issubset(hypothesis):
            bel += self.refutes
        if UNCERTAIN.issubset(hypothesis):
            bel += self.uncertain
        # THETA mass only contributes if hypothesis == THETA
        if hypothesis == THETA:
            bel += self.theta
        return bel

    def plausibility(self, hypothesis: frozenset) -> float:
        """Plausibility: 1 - Belief(complement)."""
        complement = THETA - hypothesis
        return 1.0 - self.belief(complement)

    def invert(self) -> MassFunction:
        """Swap supports↔refutes: view from the contrary perspective."""
        return MassFunction(
            supports=self.refutes,
            refutes=self.supports,
            uncertain=self.uncertain,
            theta=self.theta,
        )

    def to_dict(self) -> dict:
        return {
            "supports": self.supports,
            "refutes": self.refutes,
            "uncertain": self.uncertain,
            "theta": self.theta,
        }


def combine_dempster(m1: MassFunction, m2: MassFunction) -> MassFunction:
    """Dempster's rule of combination for two independent mass functions.

    Combines evidence while normalizing away conflict (mass assigned to ∅).
    """
    # Compute all pairwise products and accumulate by intersection
    combined = {"supports": 0.0, "refutes": 0.0, "uncertain": 0.0, "theta": 0.0}
    conflict = 0.0

    # Mass pairs: (focal_element_1, mass_1) × (focal_element_2, mass_2)
    m1_masses = [
        (SUPPORTS, m1.supports),
        (REFUTES, m1.refutes),
        (UNCERTAIN, m1.uncertain),
        (THETA, m1.theta),
    ]
    m2_masses = [
        (SUPPORTS, m2.supports),
        (REFUTES, m2.refutes),
        (UNCERTAIN, m2.uncertain),
        (THETA, m2.theta),
    ]

    for A, a_mass in m1_masses:
        for B, b_mass in m2_masses:
            product = a_mass * b_mass
            if product == 0:
                continue

            intersection = A & B
            if not intersection:
                # Empty intersection → conflict
                conflict += product
            elif intersection == SUPPORTS:
                combined["supports"] += product
            elif intersection == REFUTES:
                combined["refutes"] += product
            elif intersection == UNCERTAIN:
                combined["uncertain"] += product
            elif intersection == THETA:
                combined["theta"] += product
            else:
                # Partial overlap — distribute to the intersection
                # E.g., THETA ∩ SUPPORTS = SUPPORTS
                if SUPPORTS.issubset(intersection) and len(intersection) < len(THETA):
                    if REFUTES.issubset(intersection):
                        # {S, R} — split between them or assign to theta
                        combined["theta"] += product
                    else:
                        combined["supports"] += product
                elif REFUTES.issubset(intersection):
                    combined["refutes"] += product
                else:
                    combined["theta"] += product

    # Normalize by (1 - conflict)
    normalization = 1.0 - conflict
    if normalization <= 0:
        # Total conflict — return vacuous
        return MassFunction(theta=1.0)

    return MassFunction(
        supports=combined["supports"] / normalization,
        refutes=combined["refutes"] / normalization,
        uncertain=combined["uncertain"] / normalization,
        theta=combined["theta"] / normalization,
    )


def combine_multiple(masses: list[MassFunction]) -> MassFunction:
    """Combine multiple mass functions via iterative Dempster combination."""
    if not masses:
        return MassFunction(theta=1.0)
    result = masses[0]
    for m in masses[1:]:
        result = combine_dempster(result, m)
    return result


# ═══════════════════════════════════════════════════════════════════════
# EVIDENCE SOURCES
# ═══════════════════════════════════════════════════════════════════════

def mass_from_polarity(infon: Infon) -> MassFunction:
    """Evidence source 1: lexical polarity detection.

    If the sentence contains explicit negation → mass toward REFUTES.
    If affirmed → mass toward SUPPORTS.
    Remaining mass → ignorance (theta).
    """
    strength = min(infon.confidence, 0.8)  # cap to leave room for ignorance

    if infon.polarity == 0:
        # Explicit negation detected
        return MassFunction(
            refutes=strength * 0.7,
            uncertain=strength * 0.1,
            theta=1.0 - strength * 0.8,
        )
    else:
        # Affirmed
        return MassFunction(
            supports=strength * 0.5,
            theta=1.0 - strength * 0.5,
        )


def mass_from_triple_alignment(
    claim_anchors: dict[str, float],
    infon: Infon,
    schema_types: dict[str, str],
) -> MassFunction:
    """Evidence source 2: triple alignment between claim and evidence.

    Compare the claim's activated anchors against the infon's (S, P, O).
    - Same S+P+O, affirmed → strong SUPPORTS
    - Same S+P+O, negated → strong REFUTES (negated alignment)
    - Same S+P, different O → strong REFUTES (semantic contradiction)
    - Same S, different P → weak evidence
    - No overlap → ignorance

    This is the key source for detecting FEVER REFUTES claims where
    the evidence mentions the same entity/relation but a different object.
    """
    if not claim_anchors:
        return MassFunction(theta=1.0)

    # Check if the infon's anchors appear in the claim's activations
    s_in_claim = claim_anchors.get(infon.subject, 0.0)
    p_in_claim = claim_anchors.get(infon.predicate, 0.0)
    o_in_claim = claim_anchors.get(infon.object, 0.0)

    # Subject + Predicate match → evidence is about the same fact
    sp_match = min(s_in_claim, p_in_claim)

    if sp_match > 0.1:
        if o_in_claim > 0.1:
            # Full triple alignment: S+P+O all in claim
            alignment = (s_in_claim * p_in_claim * o_in_claim) ** (1/3)
            if infon.polarity == 1:
                # Affirmed + full match → SUPPORTS
                return MassFunction(
                    supports=min(alignment * 0.8, 0.7),
                    theta=max(1.0 - alignment * 0.8, 0.3),
                )
            else:
                # Negated + full match → REFUTES (evidence denies what claim asserts)
                return MassFunction(
                    refutes=min(alignment * 0.8, 0.7),
                    theta=max(1.0 - alignment * 0.8, 0.3),
                )
        else:
            # S+P match but object differs → REFUTES
            # The evidence talks about the same subject doing the same thing
            # but to/with a different object than the claim states
            mismatch_strength = sp_match * (1.0 - o_in_claim)
            return MassFunction(
                refutes=min(mismatch_strength * 0.6, 0.5),
                supports=mismatch_strength * 0.05,
                theta=max(1.0 - mismatch_strength * 0.65, 0.3),
            )
    elif s_in_claim > 0.1:
        # Subject only — weak directional evidence
        if infon.polarity == 1:
            return MassFunction(
                supports=s_in_claim * 0.15,
                theta=1.0 - s_in_claim * 0.15,
            )
        else:
            return MassFunction(
                refutes=s_in_claim * 0.15,
                theta=1.0 - s_in_claim * 0.15,
            )
    else:
        # No meaningful alignment → ignorance
        return MassFunction(theta=1.0)


def mass_from_anchor_distance(
    claim_anchors: dict[str, float],
    infon: Infon,
    schema_types: dict[str, str],
) -> MassFunction:
    """Evidence source 3: anchor type distance.

    When the claim and evidence share subject+predicate but have different
    specific anchors in the object role, this signals a potential contradiction
    (e.g., claim="born in France" vs evidence="born in Poland").

    IMPORTANT: This source only fires when there IS subject/predicate overlap
    between claim and evidence. If the evidence is about a different entity
    entirely, this source returns ignorance.
    """
    if not claim_anchors:
        return MassFunction(theta=1.0)

    # First check: does the infon's subject AND predicate overlap with claim?
    s_in_claim = claim_anchors.get(infon.subject, 0.0)
    p_in_claim = claim_anchors.get(infon.predicate, 0.0)

    # Only fire if this infon is actually about the same fact as the claim
    if s_in_claim < 0.1 or p_in_claim < 0.1:
        return MassFunction(theta=1.0)

    # Get type of infon's object
    infon_obj_type = schema_types.get(infon.object, "feature")

    # Find claim anchors of the same type as the infon's object
    competing_objects = [
        (name, score) for name, score in claim_anchors.items()
        if schema_types.get(name, "feature") == infon_obj_type
        and name != infon.object
        and score > 0.1
    ]

    if not competing_objects:
        return MassFunction(theta=1.0)

    # If claim activates a DIFFERENT anchor of the same type as the infon's
    # object, and that different anchor is NOT the infon's object → possible refutation
    max_competing = max(score for _, score in competing_objects)
    o_in_claim = claim_anchors.get(infon.object, 0.0)

    if max_competing > o_in_claim + 0.1:
        # Claim expects a different object than what the evidence provides
        delta = max_competing - o_in_claim
        return MassFunction(
            refutes=min(delta * 0.4, 0.35),
            theta=max(1.0 - delta * 0.4, 0.65),
        )
    elif o_in_claim > max_competing:
        # Evidence matches what the claim expects
        return MassFunction(
            supports=min(o_in_claim * 0.2, 0.2),
            theta=max(1.0 - o_in_claim * 0.2, 0.8),
        )

    return MassFunction(theta=1.0)


def mass_from_confidence(infon: Infon) -> MassFunction:
    """Evidence source 4: model confidence as credibility weight.

    Confidence modulates how much we trust the other sources — it doesn't
    provide directional evidence itself. Low confidence → nearly all mass
    goes to theta (ignorance), weakening whatever the other sources say.

    We implement this by assigning mass only to theta and a small amount
    to UNCERTAIN. This acts as a "discount" on the other evidence when
    combined via Dempster's rule.
    """
    c = infon.confidence
    if c < 0.05:
        # Very low confidence: don't let this infon contribute much
        return MassFunction(uncertain=0.3, theta=0.7)

    # Higher confidence = less mass on uncertain = less discounting
    uncertain_mass = max(0.05, 0.3 * (1 - c))
    return MassFunction(
        uncertain=uncertain_mass,
        theta=1.0 - uncertain_mass,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLAIM-LEVEL VERDICT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VerificationVerdict:
    """The combined Dempster-Shafer verdict for a claim."""
    label: str  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    belief_supports: float = 0.0
    belief_refutes: float = 0.0
    belief_uncertain: float = 0.0
    plausibility_supports: float = 0.0
    plausibility_refutes: float = 0.0
    conflict: float = 0.0  # total conflict across evidence sources
    n_evidence: int = 0
    per_infon_masses: list = field(default_factory=list)


def verify_claim(
    infons: list[Infon],
    claim_anchors: dict[str, float],
    schema_types: dict[str, str],
    supports_threshold: float = 0.25,
    refutes_threshold: float = 0.15,
    contrary: bool = False,
) -> VerificationVerdict:
    """Verify a claim using Dempster-Shafer combination of infon evidence.

    For each infon, computes 4 independent mass functions and combines them.
    Then combines across infons to get the claim-level verdict.

    Args:
        infons: retrieved infons (evidence)
        claim_anchors: {anchor_name: activation_score} from encoding the claim
        schema_types: {anchor_name: type} from the schema
        supports_threshold: minimum belief(SUPPORTS) to predict SUPPORTS
        refutes_threshold: minimum belief(REFUTES) to predict REFUTES
        contrary: if True, invert the frame — evidence that refuted the
                  original claim now supports the contrary view
    """
    if not infons:
        return VerificationVerdict(label="NOT ENOUGH INFO")

    # Compute per-infon combined mass
    per_infon_masses = []
    for infon in infons[:10]:  # top-10 infons by importance
        # Relevance gate: how much does this infon overlap with the claim?
        s_overlap = claim_anchors.get(infon.subject, 0.0)
        p_overlap = claim_anchors.get(infon.predicate, 0.0)
        relevance = max(s_overlap, p_overlap)

        if relevance < 0.05:
            # Infon is about something unrelated to the claim — skip it
            per_infon_masses.append(MassFunction(theta=1.0))
            continue

        sources = [
            mass_from_polarity(infon),
            mass_from_triple_alignment(claim_anchors, infon, schema_types),
            mass_from_anchor_distance(claim_anchors, infon, schema_types),
            mass_from_confidence(infon),
        ]
        combined = combine_multiple(sources)
        if contrary:
            combined = combined.invert()
        per_infon_masses.append(combined)

    # Combine across infons — each infon is an independent evidence source
    # But be careful: too many weak sources can overwhelm via repeated combination.
    # Use only the top-k most decisive (lowest theta) infons.
    decisive = sorted(per_infon_masses, key=lambda m: m.theta)[:5]
    claim_mass = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)

    # Extract beliefs
    bel_s = claim_mass.supports + claim_mass.theta * (
        claim_mass.supports / max(claim_mass.supports + claim_mass.refutes + claim_mass.uncertain, 1e-10)
    ) if claim_mass.supports > 0 else 0.0

    bel_r = claim_mass.refutes + claim_mass.theta * (
        claim_mass.refutes / max(claim_mass.supports + claim_mass.refutes + claim_mass.uncertain, 1e-10)
    ) if claim_mass.refutes > 0 else 0.0

    # Pignistic probability (decision-making transform of DS)
    total_focal = claim_mass.supports + claim_mass.refutes + claim_mass.uncertain
    if total_focal > 0:
        # Distribute theta proportionally (pignistic transform)
        pig_s = claim_mass.supports + claim_mass.theta * (claim_mass.supports / total_focal)
        pig_r = claim_mass.refutes + claim_mass.theta * (claim_mass.refutes / total_focal)
        pig_u = claim_mass.uncertain + claim_mass.theta * (claim_mass.uncertain / total_focal)
    else:
        pig_s = claim_mass.theta / 3
        pig_r = claim_mass.theta / 3
        pig_u = claim_mass.theta / 3

    # Decision
    if pig_r > refutes_threshold and pig_r > pig_s:
        label = "REFUTES"
    elif pig_s > supports_threshold and pig_s > pig_r:
        label = "SUPPORTS"
    else:
        label = "NOT ENOUGH INFO"

    return VerificationVerdict(
        label=label,
        belief_supports=pig_s,
        belief_refutes=pig_r,
        belief_uncertain=pig_u,
        plausibility_supports=claim_mass.plausibility(SUPPORTS),
        plausibility_refutes=claim_mass.plausibility(REFUTES),
        conflict=0.0,
        n_evidence=len(per_infon_masses),
        per_infon_masses=[m.to_dict() for m in per_infon_masses],
    )
