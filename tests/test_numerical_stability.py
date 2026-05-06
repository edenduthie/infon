"""Numerical-stability tests.

Concrete measurements of how the system behaves under perturbation:

    S1. Input-perturbation Lipschitz check — tiny input noise produces
        bounded output deltas.
    S2. Readout continuity — small embedding changes → small mass changes.
    S3. Repeated forward pass stability — drift under many forward calls.
    S4. Conditioning of the random projection matrix (JL distortion).
    S5. Sheaf coherence score is bounded [0, 1] for arbitrary infon triples.
    S6. Training loss monotonicity in expectation on a fixed graph.
    S7. Dempster commutativity under cumulative float error.
"""

from __future__ import annotations

import math
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from infon import Cognition, CognitionConfig
from infon.logic import HypergraphReasoner


# ── Helpers ──────────────────────────────────────────────────────────

SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "invests":  {"type": "relation", "tokens": ["invest", "invests"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners"]},
    "produces": {"type": "relation", "tokens": ["produce", "produces"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    "ev":       {"type": "feature",  "tokens": ["ev", "electric vehicle"]},
    "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
}

DOCS = [
    {"id": "d1", "text": "Toyota invests in battery technology. "
                          "Toyota partners with Japanese suppliers."},
    {"id": "d2", "text": "Honda produces electric vehicles in Japan."},
    {"id": "d3", "text": "Tesla produces batteries at its Gigafactory."},
]


def _build():
    tmpdir = tempfile.mkdtemp()
    sp = os.path.join(tmpdir, "schema.json")
    with open(sp, "w") as f:
        json.dump(SCHEMA, f)
    cog = Cognition(CognitionConfig(
        schema_path=sp,
        db_path=os.path.join(tmpdir, "test.db"),
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
    ))
    for d in DOCS:
        cog.ingest([d])
    cog.consolidate()
    return cog, tmpdir


# ── S1. Input-perturbation Lipschitz check ─────────────────────────────

def test_lipschitz_input_perturbation():
    """||f(x + ε) - f(x)|| / ||ε|| remains bounded across scales of ε.

    A Lipschitz-stable network has output perturbation proportional to
    input perturbation. We measure the empirical ratio at ε scales
    1e-4, 1e-3, 1e-2, 1e-1 and check that the ratio grows at most
    linearly with ε.
    """
    cog, _ = _build()
    reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                  hidden_dim=64, n_layers=2)
    reasoner.fit(graph=None, epochs=20, verbose=False)
    reasoner.eval()
    graph = reasoner.builder.build(feature_dim=64)

    with torch.no_grad():
        base = reasoner.forward(graph)

    torch.manual_seed(42)
    ratios = {}
    for scale in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        perturbed = graph.node_features + scale * torch.randn_like(
            graph.node_features
        )
        modified = type(graph)(
            node_ids=graph.node_ids, node_types=graph.node_types,
            node_features=perturbed,
            edge_index=graph.edge_index, edge_types=graph.edge_types,
            edge_weights=graph.edge_weights,
            anchor_type_groups=graph.anchor_type_groups,
            infon_indices=graph.infon_indices,
            infon_map=graph.infon_map, anchor_map=graph.anchor_map,
            situation_features=graph.situation_features,
        )
        with torch.no_grad():
            out = reasoner.forward(modified)
        input_norm = (perturbed - graph.node_features).norm().item()
        output_norm = (out - base).norm().item()
        ratio = output_norm / max(input_norm, 1e-12)
        ratios[scale] = ratio

    print("\n  Lipschitz ratios ||Δout|| / ||Δin||:")
    for s, r in ratios.items():
        print(f"    ε = {s:.0e}: ratio = {r:.4f}")

    # The empirical Lipschitz constant should not blow up with ε.
    # Stability failure would manifest as ratio growing super-linearly.
    max_ratio = max(ratios.values())
    min_ratio = min(ratios.values())
    # Allow up to 10× variation across 5 orders of magnitude of ε —
    # this is generous but catches exponential blow-up.
    assert max_ratio / max(min_ratio, 1e-6) < 10.0, (
        f"Lipschitz ratio varied {max_ratio/min_ratio:.1f}× across ε scales; "
        f"possible instability"
    )
    assert max_ratio < 50.0, (
        f"empirical Lipschitz constant {max_ratio:.2f} is large"
    )
    cog.close()
    print("  PASS: Lipschitz input perturbation")


# ── S2. Readout continuity ────────────────────────────────────────────

def test_readout_continuity():
    """Small changes in the hidden embedding produce small changes in
    the DS mass. This is a consequence of softmax + linear being
    Lipschitz, but worth checking numerically."""
    cog, _ = _build()
    reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                  hidden_dim=64, n_layers=2)
    reasoner.fit(graph=None, epochs=15, verbose=False)
    reasoner.eval()

    torch.manual_seed(0)
    h = torch.randn(64)
    eps_scales = [1e-6, 1e-4, 1e-2]

    print("\n  Readout deltas vs embedding perturbation:")
    for eps in eps_scales:
        noise = eps * torch.randn(64)
        with torch.no_grad():
            m0 = reasoner.mass_readout(h.unsqueeze(0))
            m1 = reasoner.mass_readout((h + noise).unsqueeze(0))
        mass_delta = (m1 - m0).abs().max().item()
        print(f"    ε = {eps:.0e}: max|Δm| = {mass_delta:.6f}")
        # Output perturbation should be no larger than a small constant ×
        # input perturbation (softmax has Lipschitz constant ≤ 1)
        assert mass_delta <= 2.0 * eps + 1e-6, (
            f"readout amplified ε={eps} to Δm={mass_delta}"
        )
    cog.close()
    print("  PASS: readout continuity")


# ── S3. Repeated forward-pass stability ───────────────────────────────

def test_repeated_forward_no_drift():
    """Calling forward(graph) 50 times in eval mode produces zero drift."""
    cog, _ = _build()
    reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                  hidden_dim=64, n_layers=2)
    reasoner.fit(graph=None, epochs=10, verbose=False)
    reasoner.eval()
    graph = reasoner.builder.build(feature_dim=64)

    with torch.no_grad():
        h0 = reasoner.forward(graph).clone()
    max_drift = 0.0
    for _ in range(50):
        with torch.no_grad():
            h = reasoner.forward(graph)
        drift = (h - h0).abs().max().item()
        max_drift = max(max_drift, drift)
    print(f"\n  max drift over 50 forward passes: {max_drift:.2e}")
    assert max_drift == 0.0, f"non-deterministic forward: drift {max_drift}"
    cog.close()
    print("  PASS: no drift under repeated forward")


# ── S4. Random-projection conditioning (JL distortion) ────────────────

def test_random_projection_preserves_distances():
    """For the 30522→64 seeded Gaussian projection used by the encoder,
    pairwise distances among a set of random sparse vectors are
    preserved within the Johnson–Lindenstrauss distortion bound
    (here we check a generous empirical bound rather than the formal
    (1 ± ε)-factor one, which would require a much larger d)."""
    rng = np.random.RandomState(42)
    vocab = 30522
    dim = 64

    # Simulate 10 sparse vectors: 50 non-zeros each
    n = 10
    X = np.zeros((n, vocab), dtype=np.float32)
    for i in range(n):
        idx = rng.choice(vocab, 50, replace=False)
        X[i, idx] = rng.randn(50)

    # Same seeded projection the builder uses
    proj_rng = np.random.RandomState(42)
    proj = proj_rng.randn(vocab, dim).astype(np.float32) / np.sqrt(vocab)
    Xp = X @ proj

    # Pairwise distances
    def pairwise(A):
        d = np.sqrt(((A[:, None] - A[None, :]) ** 2).sum(-1))
        return d[np.triu_indices(n, k=1)]

    d_hi = pairwise(X)
    d_lo = pairwise(Xp)

    ratios = d_lo / np.maximum(d_hi, 1e-12)
    # The code normalizes by sqrt(vocab), so the absolute ratio is
    # sqrt(d/vocab) ≈ 0.046 for d=64 vocab=30522. The JL claim is
    # about *relative* distance preservation — pairs that are close in
    # the original space should remain close in the projected space.
    # We normalize by the mean to get a scale-invariant distortion.
    normalized = ratios / ratios.mean()
    print(f"\n  JL distortion (normalized ratio):")
    print(f"    mean = {normalized.mean():.4f}  (1.0 by construction)")
    print(f"    std  = {normalized.std():.4f}")
    print(f"    min  = {normalized.min():.4f}")
    print(f"    max  = {normalized.max():.4f}")
    print(f"    absolute-scale factor = {ratios.mean():.4f}  "
          f"(≈ sqrt(d/vocab) = {math.sqrt(dim/vocab):.4f})")

    # Standard JL: (1 - ε) ≤ ||Px - Py|| / ||x - y|| · sqrt(vocab/d) ≤ (1 + ε)
    # with high probability. With d=64 we get ε ≈ sqrt(8 log(n²/δ)/d) ≈ 0.5.
    # Our n=10 → 45 pairs; allow 20% standard deviation as a lenient bound.
    assert normalized.std() < 0.3, (
        f"JL distortion stdev {normalized.std():.3f} exceeds 0.3"
    )
    # Tightest pair shouldn't shrink below half the mean, loosest shouldn't
    # balloon past 1.5× — together equivalent to ε ~ 0.5.
    assert 0.5 < normalized.min(), f"min normalized ratio {normalized.min()}"
    assert normalized.max() < 1.5, f"max normalized ratio {normalized.max()}"
    print("  PASS: random-projection conditioning")


# ── S5. Sheaf coherence is in [0, 1] for any triple ──────────────────

def test_sheaf_coherence_bounded():
    """The coherence score for any triple, under any anchor
    distribution, is mapped into [0, 1] by the NPMI → affine remap."""
    from infon.category import SheafCoherence
    from infon.atom import Infon

    names = ["a", "b", "c", "d", "e"]
    sheaf = SheafCoherence(names)
    # Feed it a random activation matrix; never produces NaN
    np.random.seed(0)
    for _ in range(20):
        mat = np.random.rand(50, 5).astype(np.float32)
        sheaf.observe(mat, threshold=0.3)
    sheaf.fit()

    for _ in range(50):
        # Random (S, P, O) — may or may not all be in the schema
        triple = list(np.random.choice(names + ["missing1", "missing2"], 3))
        inf = Infon(
            infon_id="test", subject=triple[0], predicate=triple[1],
            object=triple[2], sentence="", doc_id="d",
        )
        score = sheaf.score_infon(inf)
        assert 0.0 <= score <= 1.0, f"coherence out of [0,1]: {score}"
        assert not math.isnan(score), "coherence is NaN"
    print("  PASS: sheaf coherence bounded")


# ── S6. Training loss monotonicity in expectation ─────────────────────

def test_training_loss_monotone_in_expectation():
    """Over multiple training runs with different seeds, the loss at
    epoch 20 is (almost always) lower than at epoch 1. We allow 1
    failure in 5 to tolerate occasional bad seeds."""
    cog, _ = _build()
    losses_first = []
    losses_last = []
    for seed in range(5):
        torch.manual_seed(seed)
        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        stats = reasoner.fit(graph=None, epochs=25, verbose=False)
        losses_first.append(stats["losses"][0])
        losses_last.append(stats["final_loss"])

    improved = sum(1 for a, b in zip(losses_first, losses_last) if b < a)
    print(f"\n  Training-loss monotonicity over 5 seeds:")
    for i, (a, b) in enumerate(zip(losses_first, losses_last)):
        marker = "↓" if b < a else "↑"
        print(f"    seed={i}: {a:.4f} → {b:.4f}   {marker}")
    print(f"    improved: {improved} / 5")
    assert improved >= 4, f"only {improved}/5 seeds improved"
    cog.close()
    print("  PASS: training loss monotone in expectation")


# ── S7. Dempster commutativity under cumulative float error ──────────

def test_dempster_massive_chain_stability():
    """Combining 50 random mass functions into a single one in both
    orders (forward and reversed) yields nearly identical results."""
    from infon.dempster_shafer import MassFunction, combine_multiple

    rng = np.random.RandomState(123)
    masses = []
    for _ in range(50):
        p = rng.rand(4).astype(np.float32)
        p /= p.sum()
        masses.append(MassFunction(
            supports=float(p[0]), refutes=float(p[1]),
            uncertain=float(p[2]), theta=float(p[3]),
        ))

    forward = combine_multiple(masses)
    backward = combine_multiple(list(reversed(masses)))

    diff = max(
        abs(forward.supports - backward.supports),
        abs(forward.refutes - backward.refutes),
        abs(forward.uncertain - backward.uncertain),
        abs(forward.theta - backward.theta),
    )
    print(f"\n  Dempster 50-chain forward vs reversed:")
    print(f"    forward:  S={forward.supports:.4f} R={forward.refutes:.4f} "
          f"U={forward.uncertain:.4f} θ={forward.theta:.4f}")
    print(f"    reversed: S={backward.supports:.4f} R={backward.refutes:.4f} "
          f"U={backward.uncertain:.4f} θ={backward.theta:.4f}")
    print(f"    max |Δ| = {diff:.6e}")
    # Float error accumulates but shouldn't explode
    assert diff < 1e-3, f"50-chain order sensitivity = {diff}"
    print("  PASS: dempster 50-chain stability")


if __name__ == "__main__":
    print("=" * 60)
    print("  Numerical stability tests")
    print("=" * 60)
    test_lipschitz_input_perturbation()
    test_readout_continuity()
    test_repeated_forward_no_drift()
    test_random_projection_preserves_distances()
    test_sheaf_coherence_bounded()
    test_training_loss_monotone_in_expectation()
    test_dempster_massive_chain_stability()
    print("\n  ALL PASSED")
