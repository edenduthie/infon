"""Tests for contrary-view query-time lens.

Verifies that contrary=True inverts the evidential frame across all
three layers: MassFunction.invert(), DS verify_claim(), query ranking,
and MCTS traversal.
"""

import pytest
from pathlib import Path

from cognition.dempster_shafer import (
    MassFunction, verify_claim, VerificationVerdict,
)
from cognition.infon import Infon
from cognition import Cognition, CognitionConfig
from cognition.graph_mcts import GraphMCTS


SCHEMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "automotive_schema.json")


# ── MassFunction.invert() ─────────────────────────────────────────────

class TestMassFunctionInvert:
    def test_swap_supports_refutes(self):
        m = MassFunction(supports=0.6, refutes=0.1, uncertain=0.05, theta=0.25)
        inv = m.invert()
        assert inv.supports == pytest.approx(0.1)
        assert inv.refutes == pytest.approx(0.6)
        assert inv.uncertain == pytest.approx(0.05)
        assert inv.theta == pytest.approx(0.25)

    def test_double_invert_is_identity(self):
        m = MassFunction(supports=0.4, refutes=0.3, uncertain=0.1, theta=0.2)
        assert m.invert().invert().supports == pytest.approx(m.supports)
        assert m.invert().invert().refutes == pytest.approx(m.refutes)

    def test_invert_vacuous(self):
        m = MassFunction(theta=1.0)
        inv = m.invert()
        assert inv.theta == pytest.approx(1.0)
        assert inv.supports == 0.0
        assert inv.refutes == 0.0

    def test_invert_normalized(self):
        m = MassFunction(supports=0.5, refutes=0.2, theta=0.3)
        inv = m.invert()
        total = inv.supports + inv.refutes + inv.uncertain + inv.theta
        assert abs(total - 1.0) < 1e-6


# ── verify_claim(contrary=True) ──────────────────────────────────────

class TestVerifyClaimContrary:
    def test_supports_becomes_refutes(self):
        """An infon that SUPPORTS the original claim should REFUTE the contrary."""
        infon = Infon(
            subject='E', predicate='born', object='G',
            polarity=1, confidence=0.7,
        )
        normal = verify_claim(
            [infon],
            claim_anchors={'E': 0.8, 'born': 0.9, 'G': 0.7},
            schema_types={'E': 'actor', 'born': 'relation', 'G': 'feature'},
        )
        contrary = verify_claim(
            [infon],
            claim_anchors={'E': 0.8, 'born': 0.9, 'G': 0.7},
            schema_types={'E': 'actor', 'born': 'relation', 'G': 'feature'},
            contrary=True,
        )
        assert normal.label == "SUPPORTS"
        assert contrary.label == "REFUTES"

    def test_refutes_becomes_supports(self):
        """An infon that REFUTES the original claim should SUPPORT the contrary."""
        infon = Infon(
            subject='E', predicate='won', object='chem',
            polarity=0, confidence=0.6,
        )
        normal = verify_claim(
            [infon],
            claim_anchors={'E': 0.8, 'won': 0.7, 'chem': 0.6},
            schema_types={'E': 'actor', 'won': 'relation', 'chem': 'feature'},
        )
        contrary = verify_claim(
            [infon],
            claim_anchors={'E': 0.8, 'won': 0.7, 'chem': 0.6},
            schema_types={'E': 'actor', 'won': 'relation', 'chem': 'feature'},
            contrary=True,
        )
        assert normal.label == "REFUTES"
        assert contrary.label == "SUPPORTS"

    def test_nei_stays_nei(self):
        """Not-enough-info should remain unchanged under inversion."""
        v = verify_claim([], {}, {}, contrary=True)
        assert v.label == "NOT ENOUGH INFO"

    def test_belief_values_swap(self):
        """Pignistic supports/refutes should swap between normal and contrary."""
        infon = Infon(
            subject='X', predicate='P', object='Y',
            polarity=1, confidence=0.8,
        )
        normal = verify_claim(
            [infon],
            claim_anchors={'X': 0.9, 'P': 0.8, 'Y': 0.7},
            schema_types={'X': 'actor', 'P': 'relation', 'Y': 'feature'},
        )
        contrary = verify_claim(
            [infon],
            claim_anchors={'X': 0.9, 'P': 0.8, 'Y': 0.7},
            schema_types={'X': 'actor', 'P': 'relation', 'Y': 'feature'},
            contrary=True,
        )
        assert normal.belief_supports > normal.belief_refutes
        assert contrary.belief_refutes > contrary.belief_supports


# ── Query ranking with contrary=True ─────────────────────────────────

@pytest.fixture
def cog_with_mixed_polarity():
    """Build a graph with both affirmed and negated infons."""
    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.02,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=999,
    )
    cog = Cognition(config)
    docs = [
        {"text": "Toyota invested heavily in solid-state battery technology.",
         "id": "d1", "timestamp": "2024-01-01"},
        {"text": "Toyota has not invested in hydrogen fuel cells.",
         "id": "d2", "timestamp": "2024-02-01"},
        {"text": "Toyota did not expand into the Chinese EV market.",
         "id": "d3", "timestamp": "2024-03-01"},
        {"text": "Toyota launched a new electric vehicle platform in Europe.",
         "id": "d4", "timestamp": "2024-04-01"},
    ]
    cog.ingest(docs, consolidate_now=True)
    yield cog
    cog.close()


class TestQueryContrary:
    def test_contrary_flips_valence_sign(self, cog_with_mixed_polarity):
        normal = cog_with_mixed_polarity.query("What is Toyota investing in?")
        contrary = cog_with_mixed_polarity.query(
            "What is Toyota investing in?", contrary=True,
        )
        shared_ids = set(normal.valence) & set(contrary.valence)
        for iid in shared_ids:
            if normal.valence[iid] != 0:
                assert normal.valence[iid] == pytest.approx(
                    -contrary.valence[iid],
                )

    def test_contrary_boosts_negated_infons(self, cog_with_mixed_polarity):
        contrary = cog_with_mixed_polarity.query(
            "What is Toyota investing in?", contrary=True,
        )
        if len(contrary.infons) >= 2:
            negated = [i for i in contrary.infons if i.polarity == 0]
            affirmed = [i for i in contrary.infons if i.polarity == 1]
            if negated and affirmed:
                best_neg_idx = min(contrary.infons.index(n) for n in negated)
                best_aff_idx = min(contrary.infons.index(a) for a in affirmed)
                assert best_neg_idx < best_aff_idx

    def test_contrary_same_infon_set(self, cog_with_mixed_polarity):
        """Contrary mode retrieves the same infons, just re-ranked."""
        normal = cog_with_mixed_polarity.query("What is Toyota investing in?")
        contrary = cog_with_mixed_polarity.query(
            "What is Toyota investing in?", contrary=True,
        )
        normal_ids = {i.infon_id for i in normal.infons}
        contrary_ids = {i.infon_id for i in contrary.infons}
        assert normal_ids == contrary_ids


# ── MCTS contrary mode ───────────────────────────────────────────────

class TestMCTSContrary:
    def test_contrary_inverts_verdict(self, cog_with_mixed_polarity):
        cog = cog_with_mixed_polarity
        normal_mcts = GraphMCTS(
            store=cog.store, encoder=cog.encoder, schema=cog.schema,
            max_iterations=3,
        )
        contrary_mcts = GraphMCTS(
            store=cog.store, encoder=cog.encoder, schema=cog.schema,
            max_iterations=3, contrary=True,
        )
        normal_result = normal_mcts.search("Did Toyota invest in batteries?")
        contrary_result = contrary_mcts.search("Did Toyota invest in batteries?")

        nm = normal_result.combined_mass
        cm = contrary_result.combined_mass
        assert nm.supports + nm.refutes + nm.uncertain + nm.theta == pytest.approx(1.0, abs=1e-5)
        assert cm.supports + cm.refutes + cm.uncertain + cm.theta == pytest.approx(1.0, abs=1e-5)

        if normal_result.verdict == "SUPPORTS":
            assert cm.refutes >= cm.supports or contrary_result.verdict != "SUPPORTS"

    def test_contrary_mass_normalized(self, cog_with_mixed_polarity):
        cog = cog_with_mixed_polarity
        mcts = GraphMCTS(
            store=cog.store, encoder=cog.encoder, schema=cog.schema,
            max_iterations=3, contrary=True,
        )
        result = mcts.search("Toyota battery investment")
        m = result.combined_mass
        total = m.supports + m.refutes + m.uncertain + m.theta
        assert abs(total - 1.0) < 1e-5
