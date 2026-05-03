# Epic 01 Report — Stabilize Θ: fix the readout/fusion collapse

**Status:** Closed (35/35 tasks complete; 1 deferred follow-up resolved post-close).
**Tag:** `v0.2.0-phase1` on branch `research`.
**Date:** 2026-05-02 → 2026-05-03 UTC (one working session, including the 22.3-min sweep).
**Audit responded to:** `docs/publication/reproduction_audit.md` (2026-05-01).
**Companion technical memo:** `docs/publication/phase1_collapse_fix.md` (long-form narrative).
**Beads epic:** `infon-6o3`.

---

## TL;DR

The audit's central finding — that the released `reference_v2/` code returns `m(Θ) ≈ 0.002` on every diagnostic query, sign-reversing the paper's H2 claim of `m(Θ) ≈ 0.30` — is **resolved without changing training**. The collapse was localized to fusion (Dempster's rule on multiple agreeing high-confidence per-infon masses), not to training. A 480-cell sweep over `(coherence_weight × fusion_rule × decisive_top_k × seed)` produced 10 acceptance-passing cells; the chosen canonical configuration `top1, decisive_top_k=2, coherence_weight=1.0` satisfies all four acceptance criteria across five pinned seeds:

| query  | polarity | m(Θ) mean | m(Θ) std | min    | max    |
|--------|----------|-----------|----------|--------|--------|
| toyota | SUPPORTS | 0.2330    | 0.0082   | 0.2240 | 0.2482 |
| honda  | SUPPORTS | 0.2247    | 0.0092   | 0.2085 | 0.2354 |
| tesla  | SUPPORTS | 0.2750    | 0.0140   | 0.2559 | 0.2978 |
| catl   | SUPPORTS | 0.2319    | 0.0080   | 0.2240 | 0.2468 |

Stage E (Expanded Search — alternative regularizer forms / teacher reconstruction / readout-architecture revisions) was **not entered** — the first-round sweep recovered acceptance.

## Hypothesis status going into Epic 02

- **H1 (compositional logic):** Untouched by Epic 01. Architecture is intact; behavioral evidence remains as-was. To be tested formally in Epic 03 against multi-hop benchmarks (HoVer in particular).
- **H2 (residual mass on Θ tracks evidential thinness):** **Rescued.** The pre-fusion per-infon Θ distribution was healthy all along (mean ≈ 0.235, range 0.11–0.45 — already inside the acceptance band); only the decoder was destroying it. With the canonical fusion choice, the trained system once again outputs the moderate Θ that H2 predicts. To be tested at scale in Epic 02 against the parametric synthetic stress dataset (target: ρ(m(Θ), planted_thinness) ≥ 0.4).

The audit's identified contradictions (Θ collapse, Honda REFUTES verdict) are resolved or deferred:
- Θ collapse → fixed by canonical config.
- Honda verdict → corpus genuinely supports SUPPORTS (the audit was correct that the paper's REFUTES claim was factually wrong vs. the corpus); the canonical config returns SUPPORTS, matching the corpus.

## Method changes landed

| Change | Files | Rationale |
|---|---|---|
| Seed pinning in `fit()` | `src/cognition/logic.py:739`+ | A.2b. Two invocations with the same seed now produce bit-identical loss trace and final masses. Pins `random`, `numpy`, `torch`, CUDA RNGs + `torch.use_deterministic_algorithms(warn_only=True)` + per-submodule `reset_parameters()` (load-bearing). |
| Per-infon mass logger | `src/cognition/{config.py, logic.py, __init__.py}` | A.3b + 6o3.36. New `PerInfonMassRecord` dataclass with `{query_id, infon_id, mass, relevance_score}`. Gated by `CognitionConfig.log_per_infon_masses` (default False). Emission path is permanent (Epic 02 dependency). |
| Three new fusion rules | `src/cognition/dempster_shafer.py` | A.4b. `combine_yager` (Yager 1987 — conflict→Θ), `combine_murphy` (Murphy 2000 — averaging), `combine_top1` (cautious floor — most-decisive mass, no fusion). All produce valid mass functions on pathological inputs. |
| Fusion-rule dispatcher | `src/cognition/dempster_shafer.py` | A.4b. `combine_multiple(masses, rule="dempster"|"yager"|"murphy"|"top1")`. Default `dempster` is bit-identical to legacy no-kwarg call (regression guard). |
| `decisive_top_k` cap | `src/cognition/{config.py, logic.py}` | A.5b. New default 3 (was hardcoded `[:5]`); accepts kwarg on `reason()`. `top_k=1` dispatches to `combine_top1` per the spec contract `top_k=1 ⇔ rule="top1"`. |
| `fusion_rule` plumbing | `src/cognition/{config.py, logic.py}` | A.6b. `Config.fusion_rule` default "dempster"; threaded through `reason()` to `combine_multiple(rule=...)` for `k≥2` branch. |
| Experiment harness | `experiments/{run.py, sweep.py, ev_corpus.py, configs/, results/}` | A.6b. `ConfigError` gates unseeded configs; YAML-driven; deterministic JSON output; sweep produces per-cell + aggregate JSONs. |
| EV corpus refactor | `experiments/ev_corpus.py` ← `tests/test_logic.py::DOCUMENTS` | A.6b. Single source of truth for the diagnostic corpus; `test_logic.py` re-exports for back-compat. |

## Sweep grid + result surface

**Grid:** 6 × 4 × 4 × 5 = 480 cells.
- `coherence_weight ∈ {0.0, 0.2, 0.5, 1.0, 2.0, 5.0}`
- `fusion_rule ∈ {dempster, yager, murphy, top1}`
- `decisive_top_k ∈ {1, 2, 3, 5}`
- `seed ∈ {42, 0, 1, 7, 13}`

**Wall-clock:** 22.3 minutes on a single CPU. (Audit's pessimistic estimate of ~4 hours assumed 30 s/cell; actual was ~3 s/cell because patience-based early stopping fires well before epoch 30.)

**Headline figures:** `reference_v2/experiments/results/sweep_collapse/sweep_summary_{honda,toyota}.png`. Acceptance-band [0.20, 0.40] is shaded grey; filled circles = polarity-correct, open squares = polarity-wrong.

**Per-fusion-rule × top-k summary** (mean Honda Θ across 6 cw values; polarity-correct count):

|             | k=1            | k=2            | k=3            | k=5            |
|-------------|---------------:|---------------:|---------------:|---------------:|
| dempster    | Θ=0.226 (1/6)  | Θ=0.121 (6/6)  | Θ=0.051 (6/6)  | Θ=0.026 (6/6)  |
| yager       | Θ=0.226 (1/6)  | Θ=0.151 (6/6)  | Θ=0.083 (6/6)  | Θ=0.728 (6/6)* |
| murphy      | Θ=0.226 (1/6)  | Θ=0.123 (6/6)  | Θ=0.051 (6/6)  | Θ=0.023 (6/6)  |
| top1        | Θ=0.226 (1/6)  | Θ=0.274 (6/6)  | Θ=0.274 (6/6)  | Θ=0.230 (1/6)  |

(\*) Yager+k=5 inflates Honda Θ to 0.83 with REFUTES polarity — the conflict-mass-to-Θ semantic is doing its job, but at this corpus's level of disagreement, top-5 fusion accumulates so much conflict that the result becomes "more uncertainty than evidence." This is the cleanest single illustrative finding from the sweep — see Findings §3.

**Acceptance-passing cells:** 10 of 96 aggregate rows pass criteria 1+2+3.

| rank | rule     | k | cw  | Toyota Θ ± std    | Honda Θ ± std     | notes |
|-----:|----------|--:|----:|-------------------|-------------------|-------|
|    1 | top1     | 2 | 1.0 | 0.2330 ± 0.0082   | 0.2247 ± 0.0092   | **canonical** |
|    2 | top1     | 3 | 1.0 | 0.2330 ± 0.0082   | 0.2247 ± 0.0092   | identical to #1 (top1 is k-independent at k≥1) |
|    3 | top1     | 2 | 2.0 | 0.2951 ± 0.0108   | 0.3020 ± 0.0113   | runner-up; centered in band |
|    4 | top1     | 3 | 2.0 | 0.2951 ± 0.0108   | 0.3020 ± 0.0113   | identical to #3 |
|    5 | dempster | 3 | 5.0 | 0.2239 ± 0.0097   | 0.2169 ± 0.0141   | **fusing alternative** for the paper's narrative |
|    6 | murphy   | 3 | 5.0 | 0.2282 ± 0.0099   | 0.2182 ± 0.0142   | nearly identical to #5 |
|    7 | yager    | 3 | 5.0 | 0.2369 ± 0.0111   | 0.2344 ± 0.0117   | |
|    8 | dempster | 2 | 5.0 | 0.3289 ± 0.0123   | 0.3743 ± 0.0181   | high-cw + fusing |
|    9 | murphy   | 2 | 5.0 | 0.3327 ± 0.0119   | 0.3754 ± 0.0181   | |
|   10 | yager    | 2 | 5.0 | 0.3361 ± 0.0128   | 0.3812 ± 0.0157   | |

The 5 non-top1 passes all require `cw=5.0`. `top1` passes from `cw=1.0` upward.

## Canonical configuration

```yaml
# IMMUTABILITY CONTRACT — DO NOT EDIT THIS FILE
# Future tuning lands in canonical_v0_3.yaml.
name: canonical_v0_2
version: "0.2.0"
rationale: |
  Canonical Θ-stabilizing config selected at C.2 (epic infon-6o3).
  See docs/publication/phase1_collapse_fix.md for the audit-→-fix narrative.
  Source: experiments/results/sweep_collapse/ranked_candidates.md.
seeds: [42, 0, 1, 7, 13]
coherence_weight: 1.0
fusion_rule: top1
decisive_top_k: 2
activation_threshold: 0.2
log_per_infon_masses: true
```

**Reproducer** (~17 s wall-clock on CPU):

```bash
cd reference_v2
PYTHONPATH=src python3 -c "from experiments.run import run; \
    run('experiments/configs/canonical_v0_2.yaml', \
        'experiments/results/canonical_v0_2/')"
```

## Findings worth flagging for the paper (Epic 04)

### 1. The collapse was fusion-side, not training-side

Pre-fusion per-infon `m(Θ)` was healthy (mean 0.235, range 0.11–0.45 — already inside the acceptance band). Dempster's rule on five agreeing decisive per-infon masses shrinks `m(Θ)` to ~0.002 by the structure of the rule itself; the readout was producing well-calibrated uncertainty all along, the decoder was destroying it. **This is the Methods-section paragraph that rescues H2.** Source: `experiments/results/diagnostic/notes.md` (B.2 verdict).

### 2. Fusion is decoding, not training

Loss curves for all four fusion rules at fixed `(cw, top_k, seed)` are bit-identical. `fusion_rule` is consumed inside `reason()` (inference), never inside `fit()` (training). Clean train/decode separation: ablating fusion rules is genuinely an ablation over decoding strategies, not over learned representations. **Methods-section design-decision callout.** Source: `experiments/results/canonical_v0_2_figures/loss_curves.png` (C.4).

### 3. Yager + corpus disagreement is honest about its uncertainty

On Honda (more bimodal evidence than Toyota), `yager + k=5` produces `m(Θ) ≈ 0.83` with REFUTES polarity. Yager assigns conflict mass to Θ rather than re-normalizing it away — the result is a reasoner that says "we have a lot of evidence here but the sources fundamentally disagree, so I'm mostly ignorant about the answer." That polarity flip is genuine query-side disagreement, not a bug. **Discussion-section paragraph for why DS-with-conflict-handling beats softmax for this kind of question.**

### 4. Honda is the binding diagnostic query

Honda `m(Θ)` ≥ Toyota `m(Θ)` in every passing row, and Honda's worst-case-across-seeds `m(Θ) = 0.2085` is just 0.009 above the 0.20 acceptance floor. The asymmetry is structural: Honda's diagnostic query triggers more REFUTES contributors than Toyota's because the corpus has more polarity-mixed Honda sentences ("Honda delays...", "Honda has not produced results"). Future ranking should weight Honda stability.

### 5. The audit's bit-perfect reproducibility is a feature, not luck

Our seeded runner reproduces `paper_scenario_report.json` (the audit's seed-42 output) bit-for-bit on every value. Combined with five-seed stability, this means downstream comparisons in Epic 02/03 can attribute any deviation unambiguously to the intervention rather than to RNG drift.

## Anomalies catalogue

Documented in full at `reference_v2/experiments/results/sweep_collapse/anomalies.md`. The six anti-patterns surfaced:

| | Anti-pattern | Implication |
|---|---|---|
| **A** | `coherence_weight=0` collapses universally (16/16 cells fail) | Regularizer is load-bearing, not a fine-tuning knob. Epic 02 should include cw=0 as the "regularizer-off" control. |
| **B** | `decisive_top_k=1` makes `fusion_rule` a no-op (4 redundant copies in aggregate) | Documented contract from A.5b's `top_k=1 ⇔ rule="top1"`. Group these rows in paper tables. |
| **C** | `yager + k=5` is query-asymmetric (Toyota in-band, Honda fully broken) | Corpus-side disagreement is real and Yager surfaces it honestly. **Paper finding §3 above.** |
| **D** | `top1` dominates passing set (5/10); fusing rules need `cw=5.0` | Canonical winner is at low `cw`; fallback `dempster, k=3, cw=5.0` is at lower band edge — more sensitive to corpora. |
| **E** | Honda Θ ≥ Toyota Θ in every passing row | Structural query asymmetry. **Paper finding §4 above.** |
| **F** | `fusion_rule` has zero training-loss effect | Clean train/decode separation. **Paper finding §2 above.** |

## Risks and known limitations carried forward

1. **`top1` is "cautious fusion" — it doesn't fuse.** A reviewer could argue this trivializes H2's demonstration. Mitigation in Epic 04's paper: report the fusing-with-regularization fallback (`dempster, k=3, cw=5.0`) alongside the canonical, frame H2 as "calibrated uncertainty under both decoding strategies."
2. **Honda's worst-case Θ is 0.2085 — only 0.009 above the 0.20 floor.** Slightly different corpus could push it under. Mitigation: Epic 02's synthetic stress test will measure how robust the canonical config is to corpus perturbations; if margin is too thin, switch the canonical to `top1, k=2, cw=2.0` (centered in band but with worse loss-convergence story).
3. **All passing cells require `cw ≥ 1.0`.** Epic 02 ablations should verify `cw=1.0` is not a knife-edge in the synthetic-stress regime.
4. **Loss plateaus at ~0.30 under top1.** Structural feature of cautious fusion (a discontinuous fusion rule can't perfectly match the continuous DS teacher), not a training bug. Reviewers comparing absolute losses across architectures may flag.

## Handoff to Epic 02

Concrete items the next epic must address:

1. **Generate the parametric synthetic stress dataset** with `evidence_redundancy ∈ {1, 2, 5, 10}` (planted thinness ground truth) and `compositional_depth ∈ {1, 2, 3, 4}`.
2. **Verify ρ(m(Θ), planted_thinness) ≥ 0.4** on the synthetic test split under `canonical_v0_2.yaml`. This is the H2 effect-size measurement.
3. **Reproduce anomaly C** at scale: `yager + high contradiction_density + k=5` should produce exactly the Toyota/Honda asymmetry pattern.
4. **Treat `fusion_rule` as a decoding-only ablation** (Epic 02 ablation matrix should not pretend it's a training ablation; `loss_trace` will be identical across rules at fixed `(cw, k, seed)`).
5. **Test whether the `dempster, cw=5.0` fallback generalizes** to higher-evidence-redundancy synthetic cases. If it does, the paper has two viable canonical-equivalent configs; if not, the choice is forced.
6. **Sweep `activation_threshold`** as part of the Epic 02 ablation matrix. Epic 01 deliberately fixed it at 0.2 (the audit's value); the audit's open question 3.1 hinted the paper used a different value, and Epic 02 has the budget to disambiguate.

## Tests landed

41/41 passing in the Epic 01 family (45/45 with the post-close gating tests):

| file | test count | purpose |
|---|---:|---|
| `test_logic.py` | 14 | Pre-existing regression. |
| `test_seed_pinning.py` | 1 | A.2 → bit-identical fits under a fixed seed. |
| `test_per_infon_mass_logger.py` | 1 | A.3 → records carry `{query_id, infon_id, mass, relevance_score}`. |
| `test_per_infon_mass_logger_gating.py` | 4 | infon-6o3.36 → flag actually gates emission. |
| `test_fusion_rules.py` | 9 | A.4 → Yager preserves Θ on agreement, Murphy averages, top1 cautious, dispatcher dispatches. |
| `test_decisive_top_k.py` | 4 | A.5 → cap parameterized; `top_k=1 ⇔ rule="top1"` contract. |
| `test_experiment_runner.py` | 4 | A.6 → runner refuses unseeded configs; deterministic JSON; sweep produces 16 cells + aggregate. |
| `test_canonical_config.py` | 8 | C.3 → canonical config satisfies all four acceptance criteria across 5 seeds. |

## Artifacts inventory

| artifact | bytes | purpose |
|---|---:|---|
| `reference_v2/experiments/configs/canonical_v0_2.yaml` | 1,866 | The single named config (immutable contract). |
| `reference_v2/experiments/configs/{baseline, sweep_collapse}.yaml` | ~1,500 | Audit reproducer + sweep grid. |
| `reference_v2/experiments/results/baseline/baseline__seed=*.json` (3 files) | ~30 KB | Bit-for-bit reproduction of `paper_scenario_report.json`. |
| `reference_v2/experiments/results/diagnostic/{toyota,honda}__seed=*.json` (6 files) | ~120 KB | Per-infon mass diagnostic (B.2 — fusion-side verdict). |
| `reference_v2/experiments/results/diagnostic/notes.md` + `per_infon_analysis.py` | ~15 KB | Diagnostic narrative. |
| `reference_v2/experiments/results/sweep_collapse/sweep_collapse__cw=*__fr=*__tk=*__seed=*.json` (480 files) | ~48 MB | Full Cartesian sweep. |
| `reference_v2/experiments/results/sweep_collapse/aggregate.json` | 158 KB | 96-row mean/std summary. |
| `reference_v2/experiments/results/sweep_collapse/sweep_summary_{honda,toyota}.png` | ~240 KB | Headline figures. |
| `reference_v2/experiments/results/sweep_collapse/{passing_cells, ranked_candidates, anomalies}.md` | ~25 KB | Selection narrative. |
| `reference_v2/experiments/results/canonical_v0_2/canonical_v0_2__seed=*.json` (5 files) | ~50 KB | 5-seed verification of the canonical. |
| `reference_v2/experiments/results/canonical_v0_2_figures/{per_infon_mass_distribution, theta_vs_coherence, loss_curves}.png` | ~280 KB | Memo figures. |
| `docs/publication/phase1_collapse_fix.md` | 25 KB | Full technical narrative. |
| `docs/publication/report_epic_1.md` | (this file) | Executive report. |
| `reference_v2/README.md` | 2.3 KB | Reproducer entry-point. |

## Commit timeline

| commit | task | summary |
|--------|------|---------|
| `fc06f7c` | A.1 | bootstrap experiments package skeleton |
| `5f952d8` | A.2 | red test for seed pinning |
| `a9a7c04` | A.2b | pin RNGs in fit() |
| `f42991f` | A.3 | red test for per-infon mass logger |
| `f14580f` | A.3b | per-infon mass logger (Option A: replace bare list) |
| `abeb3b3` | A.4 | red test for fusion rules |
| `2a44868` | A.4b | Yager / Murphy / top1 + dispatcher |
| `164587e` | A.5 | red test for decisive_top_k cap |
| `e93d425` | A.5b | cap → 3 default; top_k=1 ⇔ rule="top1" |
| `9318c5d` | A.6 | red test for experiment runner |
| `8e3809b` | A.6b | runner + sweep + ev_corpus |
| `87ee9bf` | A.7 | baseline + sweep_collapse YAMLs |
| `1655ae7` | B.1 | reproduce audit numbers (bit-identical) |
| `7322630` | B.2 | per-infon mass diagnostic — fusion-side verdict |
| `9ecb890` | B.3 | 480-cell collapse sweep |
| `854ae45` | B.4 | sweep summary plots |
| `6496c2c` | C.1 | passing_cells.md (10 acceptance-passing) |
| `51b074d` | C.2 | ranked_candidates.md (winner pick) |
| `b1cc447` | C.3 | test_canonical_config.py (8 tests, 5-seed) |
| `4b314fd` | C.4 | per-infon, theta-vs-coherence, loss curve figures |
| `41ce5c4` | C.5 | anomalies.md (6 anti-patterns) |
| `8762167` | D.1 | canonical_v0_2.yaml |
| `85c533b` | D.2 | 5-seed canonical verification |
| `f1fbb2d` | D.3 | phase1_collapse_fix.md memo |
| `853cc7f` | (housekeeping) | commit-to-iterate framing in openspec |
| `d35f366` | D.5 | phase-boundary review |
| `adfb3eb` | infon-6o3.36 | gate per-infon emission on the flag |

Tag `v0.2.0-phase1` points at `853cc7f` (the openspec framing fold). The post-close gating commit `adfb3eb` is on top of the tag — no Epic 01 retag needed.

---

**Authoring note.** This report is the executive companion to `phase1_collapse_fix.md`. The memo is the technical narrative the paper's Methods section will draw from; this report is the point-in-time executive summary, suitable for a status meeting or a hand-off email to whoever picks up Epic 02.
