"""Tests for the Dempster-Shafer belief function module.

Verifies:
1. Mass function normalization and combination
2. Evidence source correctness (polarity, alignment, distance, confidence)
3. Claim-level verdict for SUPPORTS, REFUTES, NEI cases
4. Edge cases: no evidence, conflicting evidence, irrelevant evidence
"""

import pytest
from cognition.dempster_shafer import (
    MassFunction, combine_dempster, combine_multiple,
    mass_from_polarity, mass_from_triple_alignment,
    mass_from_anchor_distance, mass_from_confidence,
    verify_claim, VerificationVerdict,
    SUPPORTS, REFUTES, UNCERTAIN, THETA,
)
from cognition.infon import Infon


# ── Mass function basics ───────────────────────────────────────────────

class TestMassFunction:
    def test_normalization(self):
        m = MassFunction(supports=0.3, refutes=0.2, uncertain=0.1, theta=0.4)
        total = m.supports + m.refutes + m.uncertain + m.theta
        assert abs(total - 1.0) < 1e-6

    def test_unnormalized_input(self):
        m = MassFunction(supports=3.0, refutes=2.0, uncertain=1.0, theta=4.0)
        total = m.supports + m.refutes + m.uncertain + m.theta
        assert abs(total - 1.0) < 1e-6

    def test_vacuous(self):
        m = MassFunction(theta=1.0)
        assert m.supports == 0.0
        assert m.refutes == 0.0
        assert m.theta == 1.0

    def test_belief(self):
        m = MassFunction(supports=0.6, theta=0.4)
        assert m.belief(SUPPORTS) == 0.6
        assert m.belief(THETA) == 1.0  # all mass is subset of THETA

    def test_plausibility(self):
        m = MassFunction(supports=0.6, refutes=0.1, theta=0.3)
        # Pl(SUPPORTS) = 1 - Bel(complement of SUPPORTS)
        # complement = {REFUTES, UNCERTAIN}
        assert m.plausibility(SUPPORTS) == pytest.approx(0.9)


# ── Dempster combination ───────────────────────────────────────────────

class TestCombination:
    def test_vacuous_combination(self):
        """Combining with vacuous mass should not change anything."""
        m1 = MassFunction(supports=0.6, theta=0.4)
        m2 = MassFunction(theta=1.0)
        result = combine_dempster(m1, m2)
        assert result.supports == pytest.approx(0.6, abs=0.01)
        assert result.theta == pytest.approx(0.4, abs=0.01)

    def test_agreeing_combination(self):
        """Two sources agreeing on SUPPORTS should increase belief."""
        m1 = MassFunction(supports=0.5, theta=0.5)
        m2 = MassFunction(supports=0.4, theta=0.6)
        result = combine_dempster(m1, m2)
        assert result.supports > 0.6  # more than either alone

    def test_conflicting_combination(self):
        """Conflicting sources should reduce both beliefs via normalization."""
        m1 = MassFunction(supports=0.8, theta=0.2)
        m2 = MassFunction(refutes=0.8, theta=0.2)
        result = combine_dempster(m1, m2)
        # Neither should dominate completely
        assert result.supports > 0
        assert result.refutes > 0

    def test_multiple_combination(self):
        masses = [
            MassFunction(supports=0.3, theta=0.7),
            MassFunction(supports=0.4, theta=0.6),
            MassFunction(supports=0.2, theta=0.8),
        ]
        result = combine_multiple(masses)
        assert result.supports > 0.5  # accumulated support


# ── Evidence sources ───────────────────────────────────────────────────

class TestEvidenceSources:
    def test_polarity_affirmed(self):
        inf = Infon(polarity=1, confidence=0.7)
        m = mass_from_polarity(inf)
        assert m.supports > 0
        assert m.refutes == 0

    def test_polarity_negated(self):
        inf = Infon(polarity=0, confidence=0.7)
        m = mass_from_polarity(inf)
        assert m.refutes > 0
        assert m.supports == 0

    def test_alignment_full_match(self):
        inf = Infon(subject='X', predicate='P', object='Y', polarity=1)
        m = mass_from_triple_alignment(
            {'X': 0.8, 'P': 0.7, 'Y': 0.6}, inf, {}
        )
        assert m.supports > 0.3

    def test_alignment_object_mismatch(self):
        inf = Infon(subject='X', predicate='P', object='Y', polarity=1)
        # Claim has X and P but NOT Y
        m = mass_from_triple_alignment(
            {'X': 0.8, 'P': 0.7, 'Z': 0.6}, inf, {}
        )
        assert m.refutes > 0.2

    def test_alignment_no_overlap(self):
        inf = Infon(subject='A', predicate='B', object='C', polarity=1)
        m = mass_from_triple_alignment(
            {'X': 0.8, 'Y': 0.7}, inf, {}
        )
        assert m.theta > 0.9  # ignorance

    def test_distance_requires_sp_overlap(self):
        """Anchor distance should NOT fire without subject+predicate overlap."""
        inf = Infon(subject='A', predicate='B', object='C', polarity=1)
        m = mass_from_anchor_distance(
            {'X': 0.9, 'Y': 0.8, 'Z': 0.7}, inf,
            {'A': 'actor', 'C': 'feature', 'Z': 'feature'}
        )
        assert m.theta == 1.0  # no fire


# ── Claim-level verdict ────────────────────────────────────────────────

class TestVerdict:
    def test_supports(self):
        infon = Infon(subject='E', predicate='born', object='G',
                      polarity=1, confidence=0.7)
        v = verify_claim(
            [infon],
            claim_anchors={'E': 0.8, 'born': 0.9, 'G': 0.7},
            schema_types={'E': 'actor', 'born': 'relation', 'G': 'feature'},
        )
        assert v.label == "SUPPORTS"

    def test_refutes_object_mismatch(self):
        infon = Infon(subject='C', predicate='born', object='P',
                      polarity=1, confidence=0.8)
        v = verify_claim(
            [infon],
            claim_anchors={'C': 0.9, 'born': 0.8, 'F': 0.7},
            schema_types={'C': 'actor', 'born': 'relation', 'F': 'feature', 'P': 'feature'},
        )
        assert v.label == "REFUTES"

    def test_refutes_negation(self):
        infon = Infon(subject='E', predicate='won', object='chem',
                      polarity=0, confidence=0.6)
        v = verify_claim(
            [infon],
            claim_anchors={'E': 0.8, 'won': 0.7, 'chem': 0.6},
            schema_types={'E': 'actor', 'won': 'relation', 'chem': 'feature'},
        )
        assert v.label == "REFUTES"

    def test_nei_no_evidence(self):
        v = verify_claim([], {}, {})
        assert v.label == "NOT ENOUGH INFO"

    def test_nei_irrelevant(self):
        infon = Infon(subject='A', predicate='B', object='C',
                      polarity=1, confidence=0.3)
        v = verify_claim(
            [infon],
            claim_anchors={'X': 0.9, 'Y': 0.8, 'Z': 0.7},
            schema_types={'A': 'actor', 'B': 'relation', 'C': 'feature'},
        )
        assert v.label == "NOT ENOUGH INFO"

    def test_accumulation(self):
        """Multiple supporting infons should strengthen SUPPORTS."""
        infons = [
            Infon(subject='E', predicate='born', object='G', polarity=1, confidence=0.4),
            Infon(subject='E', predicate='born', object='G', polarity=1, confidence=0.3),
            Infon(subject='E', predicate='born', object='G', polarity=1, confidence=0.35),
        ]
        v = verify_claim(
            infons,
            claim_anchors={'E': 0.8, 'born': 0.9, 'G': 0.7},
            schema_types={'E': 'actor', 'born': 'relation', 'G': 'feature'},
        )
        assert v.label == "SUPPORTS"
        assert v.belief_supports > 0.8
