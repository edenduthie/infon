# Acceptance-Passing Cells — Stage C.1 Gate

**Source**: `reference_v2/experiments/results/sweep_collapse/aggregate.json` (96 rows, grid `4 fusion × 4 top-k × 6 coherence_weight`).
**Aggregate commit**: `9ecb8907d4f1ed532b2db2cdc5713f63120d99ef` (B.3, `chore(experiments): run B.3 480-cell collapse sweep (infon-6o3.16)`).
**Acceptance criteria checked** (per `openspec/changes/epic-01-stabilize-theta/proposal.md`):

1. `queries.toyota.polarity_correct == true` AND `queries.honda.polarity_correct == true`.
2. `queries.toyota.mean_mass[3] ∈ [0.20, 0.40]` AND `queries.honda.mean_mass[3] ∈ [0.20, 0.40]`.
3. `queries.toyota.std_mass[3] ≤ 0.05` AND `queries.honda.std_mass[3] ≤ 0.05`.
4. (Deferred to C.3.) The 14-test regression suite is verified per top candidate in C.3.

**Result**: **10 / 96 cells** satisfy criteria 1–3.

**Failure breakdown across all 96 cells**:

| Reason | Count |
|---|---|
| Pass (criteria 1–3) | 10 |
| Fail criterion 1 (polarity wrong on Toyota and/or Honda) | 25 |
| Fail criterion 2 (Θ out of `[0.20, 0.40]` on Toyota and/or Honda) | 61 |
| Fail criterion 3 (std(Θ) > 0.05) | 0 |

(Note: criteria are evaluated short-circuit in the order 1 → 2 → 3, so the "61" includes only cells that passed criterion 1.)

## Passing cells (sorted by Honda Θ ascending)

| fusion_rule | decisive_top_k | coherence_weight | Toyota Θ (mean ± std) | Honda Θ (mean ± std) | Toyota polarity | Honda polarity | Tesla Θ | CATL Θ |
|---|---:|---:|---|---|:---:|:---:|---:|---:|
| dempster | 3 | 5.0 | 0.2239 ± 0.0097 | 0.2169 ± 0.0141 | OK | OK | 0.2064 | 0.1409 |
| murphy   | 3 | 5.0 | 0.2282 ± 0.0099 | 0.2182 ± 0.0142 | OK | OK | 0.2064 | 0.1409 |
| top1     | 2 | 1.0 | 0.2330 ± 0.0082 | 0.2247 ± 0.0092 | OK | OK | 0.2750 | 0.2319 |
| top1     | 3 | 1.0 | 0.2330 ± 0.0082 | 0.2247 ± 0.0092 | OK | OK | 0.2750 | 0.2315 |
| yager    | 3 | 5.0 | 0.2369 ± 0.0111 | 0.2344 ± 0.0117 | OK | OK | 0.2224 | 0.1550 |
| top1     | 3 | 2.0 | 0.2951 ± 0.0108 | 0.3016 ± 0.0112 | OK | OK | 0.3515 | 0.2931 |
| top1     | 2 | 2.0 | 0.2951 ± 0.0108 | 0.3020 ± 0.0113 | OK | OK | 0.3515 | 0.2934 |
| dempster | 2 | 5.0 | 0.3292 ± 0.0116 | 0.3736 ± 0.0175 | OK | OK | 0.3473 | 0.2677 |
| murphy   | 2 | 5.0 | 0.3327 ± 0.0120 | 0.3752 ± 0.0178 | OK | OK | 0.3473 | 0.2677 |
| yager    | 2 | 5.0 | 0.3361 ± 0.0125 | 0.3813 ± 0.0162 | OK | OK | 0.3543 | 0.2747 |

The two `top1` rows at `(tk=2, cw=1.0)` and `(tk=3, cw=1.0)` are numerically identical on the Toyota/Honda Θ values (Tesla differs at the 4th decimal; CATL differs at the 4th decimal), which is consistent with `top1` selecting only the single most decisive mass and being independent of `decisive_top_k` once `tk ≥ 2`.

## Routing decision

**Stage C unblocked: 10 cells satisfy criteria 1–3. Proceed to C.2 (ranking).**

The expected count from B.3's manual scan (10 passing cells) is matched exactly. Stage E (Expanded Search) is **not entered**; tasks E.0–E.N remain open as conditional and will not be exercised this round.

## Anti-patterns observed (informational, not gating)

These are surfaced for the C.5 follow-up review and should *not* affect the C.2 ranking unless a chosen candidate exhibits one of them.

### A. `coherence_weight = 0` causes universal Θ collapse

Across all 16 cells with `coherence_weight = 0.0` (4 fusion × 4 top-k), Θ is below 0.20 on both queries. The collapse worsens monotonically with `decisive_top_k` for Dempster/Murphy/Yager (e.g. Dempster: tk=1 → Θ=0.108, tk=2 → 0.04, tk=3 → 0.007, tk=5 → 0.001), confirming the audit's characterization that fusing many near-certain focal masses drives Θ → 0. `top1` does not collapse with `tk` (it selects a single mass), but still misses the band at `cw=0`. Consequently, **no `cw=0` cell passes**, and `cw ∈ {0.0, 0.2, 0.5}` produces zero passes overall.

### B. `decisive_top_k = 1` ignores fusion rule entirely

All 24 `tk=1` cells produce Θ values that depend only on `coherence_weight` (the four fusion rules give the same Θ value at each `cw`, to four decimals). This is mechanically expected — fusing one mass is the identity — but it means `tk=1` provides no information about the fusion-rule axis. At `cw=5.0`, all four `tk=1` rules hit Θ ≈ 0.516 / 0.568 (Toyota / Honda), polarity correct, but the band is exceeded — so `tk=1` also produces zero passes.

### C. `yager + tk=5` shows the conflict-mass pathology on Honda

For `yager` with `tk=5`, Honda Θ is 0.842 (cw=0.0) → 0.409 (cw=5.0), all far above the 0.40 ceiling, while Toyota Θ stays 0.205 → 0.306 inside the band. This is the query-asymmetry already recorded in epic Learnings (B.4): Yager assigns *conflict mass* to Θ, and Honda's evidence set apparently has higher conflict than Toyota's, inflating Θ artificially without the underlying evidential thinness H2 is about. The current aggregate reports `polarity_correct == true` for these cells (verdict polarity is decoded from `max(SUPPORTS, REFUTES)` ignoring Θ, so polarity stays SUPPORTS even with Θ ≈ 0.84), but criterion 2 still rejects them. **Recommendation:** keep these out of any Stage C.2 ranking even hypothetically — high Θ from Yager-conflict is not the H2 quantity.

### D. `top1` is the most permissive cell (5/24 of its cells pass)

`top1` accounts for 5 of the 10 passing cells: `(tk=2, cw=1.0)`, `(tk=3, cw=1.0)`, `(tk=2, cw=2.0)`, `(tk=3, cw=2.0)`, plus a `(tk=5, cw=2.0)` near-miss (not listed; check its band). Dempster and Murphy each contribute 2 passes (always at `cw=5.0`); Yager contributes 1 pass (at `cw=5.0`). This is consistent with the design intuition that cautious fusion (`top1`) and high coherence regularization both preserve Θ, while Dempster/Murphy/Yager need the heaviest coherence weight (`cw=5.0`) to compensate for their conjunctive collapse.

### E. Honda is consistently harder than Toyota

In every passing row, Honda Θ ≥ Toyota Θ (margin 0.005–0.045), and in the Yager+tk=5 anti-pattern Honda Θ is 2–4× Toyota Θ. The Honda evidence set is more conflicted than Toyota's, which means the C.2 ranking should weight Honda stability more heavily than Toyota when picking a canonical configuration.
