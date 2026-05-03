# Epic 01 — Stabilize Θ: fix the readout/fusion collapse

> Phase 1 of 4 of the publication-readiness program. **Highest single risk.** If this epic does not recover non-trivial Θ, hypothesis H2 has to be retracted and Epics 2–4 are rescoped to an H1-only story.

## Why

`docs/publication/reproduction_audit.md` shows that the released `reference_v2/` code returns `m(Θ) ≈ 0.002` on every diagnostic query under three pinned seeds, contradicting the paper's central claim of `m(Θ) ≈ 0.30`. This sign-reverses the H2 hypothesis ("residual mass on Θ tracks evidential thinness") on which the paper's distinctiveness depends.

The audit (§Path 2) localizes the failure to evidence fusion: per-infon masses are plausibly non-degenerate, but `combine_multiple` over the top-k decisive masses (`logic.py:967–973`) Dempster-fuses several near-certain SUPPORTS focal masses, which mathematically drives Θ → 0 by construction. The architectural compositional logic (H1) is intact; only the readout/fusion pathway is broken.

Without a fix, no amount of additional data, ablations, or baselines can support H2 — the system simply does not produce the quantity H2 is about. This epic must complete, with a passing acceptance criterion, before Epic 2 begins.

## What Changes

- Adds a deterministic experiment runner under `reference_v2/experiments/` that pins all RNGs (`torch`, `numpy`, `random`, CUDA where applicable) and writes results to a structured JSON report.
- Adds a per-infon mass logger so the collapse can be located in *training* vs *fusion* (audit §Path 2.1).
- Adds named, version-pinned configurations (`reference_v2/experiments/configs/*.yaml`) with sweep support over: `coherence_weight ∈ {0, 0.2, 0.5, 1.0, 2.0, 5.0}`; `fusion_rule ∈ {dempster, yager, murphy_average, top1}`; `decisive_top_k ∈ {1, 2, 3, 5}`; `activation_threshold` and `top_k_per_role` per audit §Path 3 questions.
- Implements three alternative fusion rules in `reference_v2/src/cognition/dempster_shafer.py`: Yager's rule (assigns conflict mass to Θ), Murphy averaging, and a `top1` cautious fusion that selects the single most-decisive focal mass.
- Adds seed pinning to `HypergraphReasoner.fit()` (audit §Pre-submission 7).
- Produces a phase-1 technical memo at `docs/publication/phase1_collapse_fix.md` documenting the diagnostic, the chosen canonical configuration, and the remaining trade-offs.

## Phased Scope

This epic itself follows the four-stage rhythm requested for every epic:

- **Stage A — Code**: instrumentation + sweep harness + alternative fusion rules.
- **Stage B — Run**: single-seed diagnostic on the existing 49-infon scenario; multi-seed sweep over the parameter grid.
- **Stage C — Review and iterate**: identify configurations that recover acceptance, debug surprises, iterate; if no configuration recovers Θ, escalate to scope contingency (H2 retraction).
- **Stage D — Finalize**: pin canonical configuration; lock the seeded reproducer; write phase-1 memo; tag commit `v0.2.0-phase1`.

## Acceptance Criterion (gates Epic 2)

For both the **Toyota** ("Did Toyota invest in battery technology?") and **Honda** ("Did Honda delay its electric vehicles?") diagnostic queries on the existing 5-doc / 49-infon scenario, under five pinned seeds {42, 0, 1, 7, 13}, the trained system shall produce:

1. Verdict polarity matching corpus ground truth (Toyota → SUPPORTS, Honda → SUPPORTS — note the audit's correction that the corpus does in fact support the Honda claim).
2. `m(Θ)` in the range `[0.20, 0.40]` on at least one *named, committed* configuration.
3. Stability: across seeds, `std(m(Θ)) ≤ 0.05` and verdict polarity unchanged.
4. The canonical configuration shall not break the existing `tests/test_logic.py` suite (14 tests must still pass).

If no configuration meets criteria 1–4, the epic is closed with a `decision-record.md` recommending H2 retraction; Epics 2–4 are rescoped accordingly.

## Impact

- Modifies `reference_v2/src/cognition/{dempster_shafer.py, logic.py}` — additions only; existing entry points preserved.
- Adds `reference_v2/experiments/` (new package).
- Adds `docs/publication/phase1_collapse_fix.md`.
- No public API removed. No dependency added (sweep configs are plain YAML; existing PyYAML transitive dep is sufficient).
- Approximate effort: 3–5 working days of engineering + 1 day of writing.

## Learnings

- B.4 (sweep summary figures): Toyota and Honda diverge on `yager + k=5` — Honda shows the conflict-mass pathology (Θ≈0.83, polarity REFUTES) while Toyota holds Θ in the acceptance band with polarity correct, so "Yager-k=5 is broken" is query-specific, not universal.
- C.1 (acceptance gate): 10 / 96 sweep cells satisfy criteria 1–3 (polarity, Θ ∈ [0.20, 0.40], std ≤ 0.05) on both Toyota and Honda — Stage C unblocked, proceeding to C.2 ranking; Stage E not entered. All passing cells have `coherence_weight ≥ 1.0`; `top1` contributes 5 passes, Dempster/Murphy 2 each, Yager 1 (all at cw=5.0); `cw=0` and `tk=1` produce zero passes.
- C.2 (winner pick): `top1, decisive_top_k=2, coherence_weight=1.0` chosen as canonical — wins on stability (lowest std on Toyota 0.0082 / Honda 0.0092), loss convergence (mean final loss 0.30 vs 1.09 for cw=5.0 cells), and simplicity (cautious fusion at the lowest passing cw); all four diagnostic queries SUPPORTS-correct on every seed.
- C.3 (regression validated): canonical config `top1/k=2/cw=1.0` does not break the existing 33-test regression suite (14 `test_logic.py` + 19 Stage A) — added 8 parametrized assertions in `tests/test_canonical_config.py` that re-fit on 5 seeds and verify polarity SUPPORTS on all four queries + Θ in `[0.20, 0.40]` + std(Θ) ≤ 0.05; full pytest run is 41-green; no prior test required adjustment.
- C.4 (memo figures): three deterministic byte-identical PNGs in `experiments/results/canonical_v0_2_figures/` — pre-fusion per-infon Θ histogram (Toyota mean 0.235, Honda mean 0.226) sits inside `[0.20, 0.40]` while the post-fusion baseline collapses to ≈0.002 and the canonical fix recovers to 0.2296 / 0.2284, confirming the collapse is purely fusion-side; surprise: at fixed `(seed, cw=1.0, tk=2)` the loss trace is identical across all four fusion rules because `fusion_rule` only enters `reason()`, not `fit()`, so Figure 3's four lines overlap by construction (filed downstream as a follow-up — fusion never participates in training-time gradient signal).
- D.1 (canonical YAML committed): `experiments/configs/canonical_v0_2.yaml` pinned with `top1 / decisive_top_k=2 / coherence_weight=1.0 / activation_threshold=0.2 / seeds=[42,0,1,7,13]` and an in-file immutability comment per `spec.md` (any future tuning lands in `canonical_v0_3.yaml`); `load_config` accepts it and a seed=42 smoke run reproduces Toyota/Honda/Tesla/CATL all SUPPORTS with Toyota Θ=0.2296 / Honda Θ=0.2284 / Tesla Θ=0.2685 / CATL Θ=0.2275 (bit-identical to the cell at `sweep_collapse__cw=1.0__fr=top1__tk=2__seed=42.json`).
- D.3 (phase-1 memo authored): `docs/publication/phase1_collapse_fix.md` synthesises the audit-→-fix arc across all 8 prior Learnings into 8 sections (audit finding, fusion-side localisation, sweep grid + linked Toyota/Honda surface plots, canonical config + alternatives, reproduction commands, acceptance table at Toyota Θ=0.2330±0.0082 / Honda Θ=0.2247±0.0092 / Tesla Θ=0.2750±0.0140 / CATL Θ=0.2319±0.0080, "Stage E not entered" iteration history, six Epic-02 follow-ups); linked from new `reference_v2/README.md`; every numeric traces to a committed JSON; 41-test regression unaffected.

