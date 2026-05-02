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
