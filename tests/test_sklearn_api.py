"""sklearn-style additions: random_state, Estimator contract,
diagnostics, model-selection helpers."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "invests":  {"type": "relation", "tokens": ["invest", "invests"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners"]},
    "produces": {"type": "relation", "tokens": ["produce", "produces"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    "factory":  {"type": "feature",  "tokens": ["factory", "plant"]},
    "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
}


def _build_cog(tmpdir, *, seed=42):
    from infon import Cognition, CognitionConfig
    sp = os.path.join(tmpdir, "schema.json")
    with open(sp, "w") as f:
        json.dump(SCHEMA, f)
    return Cognition(CognitionConfig(
        schema_path=sp,
        db_path=os.path.join(tmpdir, "cog.db"),
        quality_threshold=0.04,
        max_triples_per_sentence=2,
        random_state=seed,
    ))


DOCS = [
    {"id": "d1", "text": "Toyota invests in battery technology in Japan."},
    {"id": "d2", "text": "Tesla invests in battery factory."},
    {"id": "d3", "text": "Honda partners on battery."},
    {"id": "d4", "text": "Toyota produces batteries."},
]


# ═════════════════════════════════════════════════════════════════════
# random_state
# ═════════════════════════════════════════════════════════════════════

def test_random_state_makes_runs_identical():
    """Two runs with the same random_state give bit-identical verdicts
    and masses."""
    results = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            cog = _build_cog(tmp, seed=42)
            for d in DOCS:
                cog.ingest([d])
            cog.consolidate()
            r = cog.reasoner(hidden_dim=16, n_layers=2)
            out = r.reason("Did Toyota invest in batteries?")
            results.append((
                out.verdict,
                round(out.mass.supports, 6),
                round(out.mass.theta, 6),
            ))
            cog.close()
    assert results[0] == results[1]


def test_random_state_none_does_not_break():
    with tempfile.TemporaryDirectory() as tmp:
        from infon import Cognition, CognitionConfig
        sp = os.path.join(tmp, "schema.json")
        with open(sp, "w") as f:
            json.dump(SCHEMA, f)
        cog = Cognition(CognitionConfig(schema_path=sp,
                                        db_path=os.path.join(tmp, "cog.db")))
        assert cog.random_state is None
        cog.close()


# ═════════════════════════════════════════════════════════════════════
# Estimator contract
# ═════════════════════════════════════════════════════════════════════

def test_reasoner_is_estimator():
    from infon.estimator import Estimator
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        r = cog.reasoner(hidden_dim=16, n_layers=2)
        assert isinstance(r, Estimator)
        cog.close()


def test_predict_and_score():
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        r = cog.reasoner(hidden_dim=16, n_layers=2)

        preds = r.predict([
            "Did Toyota invest in batteries?",
            "Did Honda merge with Tesla?",
        ])
        assert len(preds) == 2
        for p in preds:
            assert p in ("SUPPORTS", "REFUTES",
                         "NOT ENOUGH INFO", "CONTRADICTED")

        s = r.score([
            {"q": "Did Toyota invest in batteries?", "gold": "SUPPORTS"},
            {"q": "Did Honda merge with Tesla?",
             "gold": "NOT ENOUGH INFO"},
        ])
        assert 0.0 <= s <= 1.0
        cog.close()


def test_predict_mass_returns_dict():
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        r = cog.reasoner(hidden_dim=16, n_layers=2)
        out = r.predict_mass(["Did Toyota invest in batteries?"])
        assert len(out) == 1
        m = out[0]
        assert set(m.keys()) == {
            "verdict", "supports", "refutes", "uncertain", "theta",
        }
        # Mass values sum to ~1
        total = m["supports"] + m["refutes"] + m["uncertain"] + m["theta"]
        assert abs(total - 1.0) < 1e-3
        cog.close()


# ═════════════════════════════════════════════════════════════════════
# Diagnostics
# ═════════════════════════════════════════════════════════════════════

def test_analyze_corpus_basic():
    from infon.diagnostics import analyze_corpus
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        rep = analyze_corpus(cog)
        assert rep.n_infons > 0
        assert rep.n_actors == 3
        assert rep.n_relations == 3
        assert rep.n_objects == 3   # battery + factory + japan
        # Entropy in [0, log|O|]
        assert 0.0 <= rep.cond_entropy_normalized <= 1.0
        # Hub concentration in [0, 1]
        assert 0.0 <= rep.hub_concentration <= 1.0
        cog.close()


def test_analyze_corpus_flags_empty_store():
    from infon.diagnostics import analyze_corpus
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        rep = analyze_corpus(cog)
        assert rep.n_infons == 0
        assert any("empty store" in w for w in rep.warnings)
        cog.close()


# ═════════════════════════════════════════════════════════════════════
# Model selection
# ═════════════════════════════════════════════════════════════════════

def test_held_out_infon_split_is_deterministic():
    from infon.model_selection import held_out_infon_split
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()

        a1, h1 = held_out_infon_split(cog, frac=0.3, seed=7)
        a2, h2 = held_out_infon_split(cog, frac=0.3, seed=7)
        assert a1 == a2 and h1 == h2
        assert set(a1).isdisjoint(h1)
        cog.close()


def test_auto_select_ssl_returns_result():
    from infon.model_selection import auto_select_ssl
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        results = auto_select_ssl(cog, modes=["laplacian"],
                                  heldout_frac=0.3, epochs=5)
        assert len(results) == 1
        top = results[0]
        assert 0.0 <= top.score <= 1.0
        assert top.config_overrides["ssl_mode"] == "laplacian"
        cog.close()


def test_sweep_sorts_by_score():
    from infon.model_selection import sweep
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        results = sweep(cog,
                        grid={"hidden_dim": [16, 32]},
                        heldout_frac=0.3, epochs=5)
        assert len(results) == 2
        # Must be sorted descending
        assert results[0].score >= results[1].score
        cog.close()


# ═════════════════════════════════════════════════════════════════════
# AutoML additions: k-fold CV, ensemble, time budget
# ═════════════════════════════════════════════════════════════════════

def test_kfold_infon_split_disjoint_and_covering():
    from infon.model_selection import kfold_infon_split
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        folds = kfold_infon_split(cog, k=3, seed=0)
        assert len(folds) == 3
        heldouts = [set(h) for _, h in folds]
        # Heldouts are pairwise disjoint
        for i in range(len(heldouts)):
            for j in range(i + 1, len(heldouts)):
                assert heldouts[i].isdisjoint(heldouts[j])
        # Union covers all infons
        all_heldout = set().union(*heldouts)
        all_infons = {i.infon_id for i in cog.store.query_infons(limit=100)}
        assert all_heldout == all_infons
        cog.close()


def test_cross_val_score_returns_mean_std():
    from infon.model_selection import cross_val_score
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        out = cross_val_score(cog, config_overrides={"hidden_dim": 16},
                              k=3, epochs=4)
        assert set(out.keys()) == {
            "mean_score", "std_score", "per_fold", "n_folds",
        }
        assert out["n_folds"] >= 1
        for s in out["per_fold"]:
            assert 0.0 <= s <= 1.0
        assert out["std_score"] >= 0.0
        cog.close()


def test_time_budget_stops_sweep_early():
    """A tiny budget should mean we run fewer than all combos."""
    from infon.model_selection import sweep
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        # 8 combos — too many for a 1-second budget on CPU
        results = sweep(
            cog,
            grid={"hidden_dim": [8, 16, 32, 64],
                  "n_layers":   [1, 2]},
            heldout_frac=0.3, epochs=8,
            time_budget_sec=0.01,   # effectively "one combo max"
        )
        # The first combo is allowed to finish, so at least 1 but not all 8
        assert 0 < len(results) < 8
        cog.close()


def test_ensemble_top_k_runs_and_returns_verdicts():
    from infon.model_selection import ensemble_top_k
    with tempfile.TemporaryDirectory() as tmp:
        cog = _build_cog(tmp)
        cog.ingest(DOCS)
        cog.consolidate()
        out = ensemble_top_k(
            cog,
            grid={"hidden_dim": [16, 32]},
            k=2, heldout_frac=0.3, epochs=5,
        )
        assert "ensemble_verdicts" in out
        assert "ensemble_score" in out
        assert len(out["members"]) <= 2
        for v in out["ensemble_verdicts"]:
            assert v in ("SUPPORTS", "REFUTES",
                         "NOT ENOUGH INFO", "CONTRADICTED")
        assert 0.0 <= out["ensemble_score"] <= 1.0
        cog.close()
