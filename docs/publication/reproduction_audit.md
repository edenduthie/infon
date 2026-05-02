# Reproduction audit of `draft2.txt`

**Date:** 2026-05-01
**Code reviewed:** `reference_v2/` (cognition package v0.1.0)
**Method:** seed-pinned re-runs of the paper's diagnostic scenario, three seeds (42, 0, 1)
**Reproducer:** `reference_v2/run_paper_scenario.py`
**Raw output:** `reference_v2/paper_scenario_report.json`

## TL;DR

The released code runs cleanly (14/14 tests pass, 29 s on a single CPU) and reproduces the paper's *qualitative architecture* — 5 documents, 49 infons, 19 active anchors, two-layer typed message passing, four-source DS teacher, coherence regularization, IF–THEN edge discovery, spectral anchor discovery. But the **quantitative results in `draft2.txt` §Verdict-calibration are not reproducible** from the released code under any pinned seed. In particular, the trained network produces Θ ≈ 0.002 on every diagnostic query — not Θ ≈ 0.30 as the paper reports. This sign-reverses the central epistemic claim of the paper.

Of the two hypotheses advanced:

- **H1 (compositional logic structure)** is *partially intact* — the architecture and operator interfaces are real and run; what is unsupported is the behavioral claim that the trained system's outputs reflect compositional reasoning, since all four diagnostic queries return near-identical SUPPORTS verdicts.
- **H2 (Θ tracks evidential thinness)** is *contradicted by reproduction* — the 4-mass readout collapses to certainty after training; the coherence regularizer at its default weight (0.2) does not prevent the collapse.

The paper should not be published as-is. Both code investigation and draft-author conversation are required before re-submission.

---

## Reproducibility table

Three pinned seeds (`torch.manual_seed`, `np.random.seed`, `random.seed`). Numbers are stable across seeds to the third decimal — randomness is not the explanation.

| Quantity | seed 42 | seed 0 | seed 1 | Paper claim | Match |
|---|---|---|---|---|---|
| Documents / infons / anchors | 5 / 49 / 19 | same | same | 5 / 49 / 22 (schema) | corpus ✅ |
| Loss initial → final at 30 epochs | 0.867 → 0.070 | 0.861 → 0.072 | 0.865 → 0.067 | 0.79 → **0.007** | ❌ ~10× off |
| Fiedler value | 0.227 | 0.227 | 0.227 | **0.11** | ❌ ~2× off |
| Layer-2 norm Δ | 66.18 | 66.30 | 66.20 | "> 0.01" | ✅ (by 4 orders of magnitude) |
| **Toyota query** m(S) / m(R) / m(Θ) | 0.987 / 0.011 / **0.002** | 0.982 / 0.015 / **0.002** | 0.986 / 0.011 / **0.002** | 0.62 / 0.05 / **0.30** | ❌ |
| **Honda EV-delay** m(S) / m(R) / m(Θ) | 0.979 / 0.017 / **0.002** | 0.977 / 0.019 / **0.002** | 0.980 / 0.016 / **0.002** | 0.04 / **0.58** / 0.35 | ❌ wrong polarity |
| Tesla query m(S) / m(R) / m(Θ) | 0.999 / 0.000 / 0.001 | 0.998 / 0.000 / 0.001 | 0.999 / 0.000 / 0.001 | not detailed | — |
| CATL query m(S) / m(R) / m(Θ) | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | not detailed | — |
| Edges (NEXT / CAUSES / CONTRADICTS) | 80 / 124 / 8 | 80 / 124 / 8 | 80 / 124 / 8 | 7 / 3 / 1 | ❌ ~20× more |
| Silhouette (k=10) | 0.816 | 0.809 | 0.653 | 0.41 (k=3) | ❌ different shape |

The full per-seed records are in `paper_scenario_report.json`.

## Hypothesis-by-hypothesis assessment

### H1 — "the forward pass is, literally, the evaluation of a small compositional logic"

**Status: partially intact.**

What survives:

- The per-relation aggregators exist and are wired correctly: `IKLAnd` (min-pool conjunction), `IKLOr` (max-pool disjunction), `IKLIf` (asymmetric attention), `IKLIst` (situation gating), `IKLForall` / `IKLExists`, plus `IKLNot` and `IKLThat`.
- `TypedMessagePassingLayer` dispatches on edge type as the paper describes.
- Two layers produce non-trivial feature change (Δ-norm = 66.18 at seed 42; paper's "above 0.01" floor is satisfied by four orders of magnitude — though the magnitude itself suggests features may be exploding, which is its own concern worth checking).
- The compound-query interface evaluates nested expressions without crashing and returns masses summing to one within numerical tolerance.
- Compound queries like `(exists actor)` and `(forall actor)` return moderate Θ (0.23–0.33 in the test output), so the *readout itself* preserves Θ on abstract logical compositions.

What is undermined:

- The paper's evidential weight for H1 comes from diagnostic queries giving *different* verdicts on *different* corpus content. The released code returns SUPPORTS at S ≥ 0.98 for all four queries, including the Honda query that the paper highlights as REFUTES. The trained system is not behaviorally distinguishing claims along the dimensions H1 predicts.
- "Traces of that pass remain humanly inspectable" is true for the architecture, less true for the verdicts: when every query collapses to maximum-support, the trace is not informative.

### H2 — "residual mass on Θ tracks evidential thinness rather than collapse into low-confidence noise"

**Status: contradicted by reproduction.**

This hypothesis is the paper's most distinctive scientific claim and the entire justification for the 4-mass readout over softmax. The paper supports it with two diagnostic queries showing Θ ≈ 0.30 on thin evidence. Both queries, regenerated from the released code with three pinned seeds, return Θ ≈ 0.002.

The trained network is *overconfident on every query*. The 4-mass readout operates as a 3-way softmax with a vestigial fourth dimension. The coherence regularizer at its default weight (`0.2 · L_coherence`) does not prevent the collapse.

Two clarifying observations from the run:

1. **The Honda verdict in the paper is independently suspect.** The paper reports m(R) = 0.58 (a REFUTES verdict) on "Did Honda delay its electric vehicles?" — but doc3 in the corpus contains the literal sentence "Honda delays its electric vehicle production timeline." The actual code returns SUPPORTS at 0.979, which is the *correct* polarity. The paper is not reporting a more-cautious version of the truth — it is reporting an incorrect polarity and presenting it as evidence of well-calibrated reasoning.

2. **The Θ collapse appears to live in evidence fusion, not the readout.** The compound-query interface (`(exists actor)`, `(forall actor)`, etc.) still returns moderate Θ. It is only the concrete reasoner queries — which go through `combine_multiple` over the top-k relevant per-infon masses — that collapse to certainty. Dempster-fusing five high-confidence SUPPORTS masses drives Θ toward zero by construction. This localizes the failure mechanically.

## Why the paper's numbers diverge from the code

The paper's numbers are mutually consistent — loss 0.007, Θ ≈ 0.30, edges 7/3/1, silhouette 0.41, Tesla isolated as its own cluster. They look like the output of *some* configuration, just not the one in the released code. Three possibilities, in rough order of likelihood:

1. **Earlier configuration drift.** The paper was drafted against an earlier branch (different teacher weighting, different fusion rule, different combine-multiple top-k cutoff, different `n_anchors` for discovery, different activation threshold for refinement) and not refreshed after the code stabilized.
2. **Numbers from the teacher rather than the trained network.** The DS teacher fuses four noisy mass functions and could plausibly produce Θ ≈ 0.30 *before* training. After training, KL divergence drives the network toward certainty regardless. If the paper's `Verdict-calibration` numbers were measured pre-training (or the network was reading out the teacher mass directly), the discrepancy is explained.
3. **A hand-edited or projected number.** Less likely, but worth ruling out: the Honda REFUTES verdict in particular is *factually inconsistent with the corpus*, which suggests it may not have been measured at all.

## Discrepancies summary

| Severity | Item | Notes |
|---|---|---|
| Critical | Θ collapse on diagnostic queries | Paper's flagship result; sign-reversal not noise |
| Critical | Honda verdict polarity | Paper says REFUTES, corpus supports SUPPORTS, code returns SUPPORTS |
| Major | Edge counts (paper 11, code 212) | ~20× difference; threshold/configuration question |
| Major | Tesla cluster shape | Paper says isolated, code groups Tesla with all other actors |
| Moderate | Loss at 30 epochs (0.007 vs 0.07) | ~10× drift |
| Moderate | Fiedler value (0.11 vs 0.227) | ~2× drift |
| Minor | Schema anchor count (22 vs 19) | Paper counts schema entries; code counts active anchors |
| Minor | Runtime (< 25 s vs 29 s) | Within run-to-run variance |
| Cosmetic | "Δ-norm > 0.01" claim | True but technically far weaker than what actually happens (Δ = 66) |

## Next steps

Both code investigation (path 2) and draft-author conversation (path 3) are required.

### Path 2 — diagnose and fix the Θ collapse

The collapse is mechanically locatable. Working hypothesis: the failure is in `HypergraphReasoner.reason()`'s use of `combine_multiple` over the top-k relevant per-infon masses, around `logic.py:967–973`. Dempster-combining several high-confidence focal masses drives Θ toward zero by the structure of the rule itself.

Investigative tasks:

1. **Confirm the collapse location.** Print per-infon masses for the Toyota and Honda queries before fusion. If individual infon masses already have Θ ≈ 0.002, the collapse is in training, not fusion. If individual masses have Θ ≈ 0.20–0.30 and only the fused result has Θ ≈ 0.002, the collapse is in fusion, and that is the place to fix.
2. **Tune the coherence regularizer weight.** The default `0.2 · L_coherence` is too weak to push back against KL-to-teacher. Sweep `coherence_weight ∈ {0.5, 1.0, 2.0, 5.0}` and check whether trained per-infon Θ rises to the paper's 0.30 range.
3. **Try a softer fusion rule.** Replace `combine_multiple` with a cautious or weighted variant (Yager's rule, Murphy's averaging, or a simple top-1) and re-check Θ on the diagnostic queries.
4. **Cap the number of fused masses.** The current code fuses up to 5 decisive masses; with 5 confident agreeing masses the result is necessarily near-certain. Try `top-k ∈ {1, 2, 3}`.
5. **Re-run the diagnostic queries after each intervention** and update the JSON report.

The acceptance criterion is that the Toyota and Honda queries produce **correct polarity with Θ in the 0.20–0.40 range** without breaking the unrelated tests. If that can be achieved, the paper's claim becomes defensible.

### Path 3 — investigate the discrepancy with the draft author

The paper's reported numbers (loss 0.007, Θ ≈ 0.30, edges 7/3/1, silhouette 0.41, Tesla isolated) are mutually consistent with some configuration. Recovering that configuration is the cleanest way to either (a) restore the paper's claims under a documented setting, or (b) confirm that no such setting was ever run.

Questions to put to the author:

1. Which branch / commit / tag were the §Verdict-calibration numbers measured on?
2. What was the value of `coherence_weight` at that time? Of `top_k_per_role` and `activation_threshold`?
3. What was the `combine_multiple` cutoff (`decisive[:5]` in current code) at that time?
4. Was the Honda REFUTES verdict measured, or was it expected?
5. For the edge-discovery 7/3/1 result, what were the m(SUPPORTS) > X and m(REFUTES) > X thresholds in `refine()`? The current code uses 0.6 and produces 212 edges — this implies either a much higher threshold or fewer eligible pairs at the time of the paper.
6. For anchor discovery, what value of `n_anchors` was passed? The current default produces 10 clusters; the paper reports a 3-cluster shape that requires `n_anchors ≈ 3`.

Even if path 2 succeeds, path 3 is needed to know whether the paper's *original* numbers were measured or projected — that determines whether the manuscript needs a methods correction (configuration change) or a substantive correction (different findings).

### Pre-submission checklist (after both paths complete)

- [ ] §Verdict-calibration numbers regenerated from a single, named, seeded config
- [ ] All four diagnostic queries shown, not two
- [ ] Honda REFUTES claim either restored mechanically or removed from the paper
- [ ] Edge counts (7/3/1 or revised) reproduced from the same config
- [ ] Anchor discovery silhouette and cluster shape regenerated
- [ ] Fiedler and loss numbers updated to match the run
- [ ] `torch.manual_seed(42)` (or equivalent) added to `HypergraphReasoner.fit()` and the seed documented in §Methods
- [ ] Ablation table added (uniform aggregator, 3-way softmax readout, teacher-only) per the prior review
- [ ] Abstract softened: "well-calibrated residual uncertainty" replaced with whatever the new measurements support, or backed by a Brier / reliability metric

## Reproducer

```bash
cd /home/ubuntu/infon/reference_v2
.venv/bin/python run_paper_scenario.py
# writes paper_scenario_report.json
```

Test suite:

```bash
.venv/bin/pytest tests/test_logic.py -v -s
```

## Files referenced

- `docs/publication/draft2.txt` — manuscript under review
- `reference_v2/src/cognition/logic.py` — `HypergraphReasoner`, `IKL*` operators, `TypedMessagePassingLayer`
  - `fit()` at line 739 — transductive training
  - `reason()` at line 898 — diagnostic-query path; suspected collapse site at lines 967–973
  - `refine()` at line 1123 — edge discovery
  - `discover_anchors()` at line 1306 — Kan-extension clustering
- `reference_v2/src/cognition/dempster_shafer.py` — `combine_multiple`, `MassFunction`
- `reference_v2/src/cognition/heads.py` — `MassReadout`
- `reference_v2/tests/test_logic.py` — synthetic EV scenario, 14 tests
- `reference_v2/run_paper_scenario.py` — seeded reproducer (this audit)
- `reference_v2/paper_scenario_report.json` — raw output
