# Anomalies and follow-ups — B.3 collapse sweep

> Compiled at C.5 (infon-6o3.22). Cross-references C.1's passing_cells.md, C.2's ranked_candidates.md, C.4's figure findings.

The 480-cell sweep at `experiments/results/sweep_collapse/aggregate.json` produced 10 acceptance-passing cells (full table in `passing_cells.md`) and the canonical winner `top1, decisive_top_k=2, coherence_weight=1.0` (rationale in `ranked_candidates.md`). This document records anomalies that, while not blocking the canonical config, are worth tracking for Epic 02 / Epic 03 / Epic 04 follow-up.

## Anti-patterns observed

### A. coherence_weight=0 collapses universally

All 16 cells with `cw=0.0` fail acceptance regardless of fusion_rule or decisive_top_k. m(Θ) trends to ~0.002 across the board. Interpretation: coherence regularization is **load-bearing** for Θ preservation in this corpus, not just a fine-tuning knob. The unregularized network's KL-to-teacher loss drives the readout to certainty regardless of fusion choice.

**Implication for Epic 02:** the synthetic stress dataset's ablation matrix should include `cw=0` as a "regularizer-off" control; it should produce the worst Θ scores by a wide margin.

### B. decisive_top_k=1 makes fusion_rule a no-op

All 24 `tk=1` cells produce identical Θ values per (cw, seed) regardless of which fusion rule was nominally selected. Reason: A.5b's implementation dispatches `decisive_top_k=1` to `combine_top1` directly (the spec contract `top_k=1 ⇔ rule="top1"`), bypassing the kwarg's `rule=` value. The aggregate's `fusion_rule` field is preserved for completeness but doesn't drive behaviour at `tk=1`. Documented in A.5b's commit `e93d425`.

**Implication:** when reporting sweep results in the paper, group `tk=1` rows together as "cautious-fusion baseline" (4 redundant copies); only Yager/Murphy/top1 nuances appear at `tk≥2`.

### C. yager + tk=5 produces correct-polarity Θ on Toyota but broken polarity on Honda

| cw | Toyota Θ | Toyota polarity | Honda Θ | Honda polarity |
|---:|---:|:---:|---:|:---:|
| 0.0 | 0.205 | SUPPORTS | 0.842 | REFUTES |
| 0.2 | 0.215 | SUPPORTS | 0.834 | REFUTES |
| 0.5 | 0.232 | SUPPORTS | 0.821 | REFUTES |
| 1.0 | 0.255 | SUPPORTS | 0.787 | REFUTES |
| 2.0 | 0.295 | SUPPORTS | 0.674 | REFUTES |
| 5.0 | 0.306 | SUPPORTS | 0.409 | REFUTES |

Honda's per-infon mass distribution is more bimodal than Toyota's (per B.2's diagnostic), with stronger REFUTES contributors. Yager assigns conflict mass to Θ rather than re-normalizing; with tk=5 fused masses the conflict accumulates and overwhelms agreement. Polarity flips because the "uncertainty" the Yager rule encodes is genuine query-side disagreement.

**Implication for the paper (Epic 04):** this is a clean visual narrative — Yager works as advertised when conflict is real, but it can't paper over corpus-side disagreement to produce a "wrong-but-confident-with-residual-Θ" output. Worth a paragraph in the Discussion.

**Implication for Epic 02:** the synthetic stress generator's `contradiction_density` knob should reproduce this pattern at scale — Yager + high contradiction → Θ inflation + polarity confusion.

### D. top1 dominates the passing set; high cw is required for fusing rules

Passing distribution: 5× top1 (cw ∈ {1.0, 1.0, 2.0, 2.0, also seen by tk fold}) + 2× dempster + 2× murphy + 1× yager. **All non-top1 passing cells require cw=5.0.**

**Implication:** the chosen winner (`top1, k=2, cw=1.0`) is the only passing config at low coherence_weight. The fusing-with-regularization fallback (`dempster, k=3, cw=5.0`) is at the lower band edge (Honda Θ=0.217±0.014) and is more sensitive to corpus changes.

### E. Honda is consistently harder than Toyota

Honda Θ ≥ Toyota Θ in every passing row. The asymmetry is structural: Honda's diagnostic query ("Did Honda delay its electric vehicles?") triggers more REFUTES contributors than Toyota's ("Did Toyota invest in battery technology?") — Honda's negative-sentiment supporting sentences ("Honda delays...", "Honda has not produced results") activate the REFUTES head strongly while still being polarity-correct via majority SUPPORTS evidence.

**Implication for ranking (C.2):** Honda Θ should be the binding constraint in candidate selection. Already reflected in the ranking — sort by Honda Θ ascending puts borderline-passing cells at the top.

### F. fusion_rule has zero training-loss effect (paper-worthy)

C.4 Figure 3 attempted to show distinct loss curves per fusion rule; all 4 traces are bit-identical at fixed (cw, tk, seed). Reason: `fusion_rule` is consumed only inside `reason()` (inference path), never inside `fit()` (training path). The training KL is computed against the teacher mass directly; the fusion rule shapes only the per-query decoding.

**Architecturally**, this is a clean train/decode separation — fusion is a post-training inference choice. **For the paper**, it means the "fusion rule sweep" is genuinely an ablation over decoding strategies, not over learned representations. Worth noting in Methods.

**Possible Epic 02 followup**: should the training loss include a fusion-aware term (e.g. KL between fused mass and target), so different rules produce distinct learned representations? Not a defect — a design choice with a clean rationale either way.

## Follow-up beads tickets

- `infon-6o3.36` (already filed by orchestrator after A.3b): thread `log_per_infon_masses` flag from `CognitionConfig` to `HypergraphReasoner` constructor so emission can be properly gated. Priority 2.

No new beads tickets are required from C.5 — anomalies A–F are either documented design properties (A, B, F), expected pathologies of the fusion rule under stress (C), structural query asymmetries inherent to the corpus (E), or selection criteria already absorbed into C.2's ranking (D, E).

## Items to call out in the phase-1 memo (D.3)

1. Anomaly C — "Yager preserves Θ but cannot paper over corpus disagreement" — paragraph in the Methods / Discussion sections.
2. Anomaly F — "fusion is decoding, not training" — Methods section design-decision callout.
3. Anomaly E — "Honda is the binding query" — Results table organization.
