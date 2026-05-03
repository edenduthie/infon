# Phase 2 Findings — Stage C Analysis

**Date**: 2026-05-03

## Finding-C1.a: H1 — All Trained Cells at Chance Accuracy

**Hypothesis**: typed_ikl (canonical) should outperform uniform_mean at hop_count=2.

**Result**: Both cells achieve identical polarity accuracy of **0.332** across all 5 seeds
(std = 0.0, CI = [0.332, 0.332]).

| Cell               | polarity_acc_mean | CI low | CI high |
|--------------------|-------------------|--------|---------|
| canonical          | 0.332             | 0.332  | 0.332   |
| uniform_aggregator | 0.332             | 0.332  | 0.332   |
| coherence_off      | 0.332             | 0.332  | 0.332   |
| single_layer       | 0.332             | 0.332  | 0.332   |
| softmax_readout    | 0.332             | 0.332  | 0.332   |
| top1_fusion        | 0.332             | 0.332  | 0.332   |
| dirichlet_edl_readout | 0.332          | 0.332  | 0.332   |
| teacher_only       | **0.664**         | 0.664  | 0.664   |

**Root cause**: Every trained cell collapses to predicting "SUPPORTS" for all 1000 test
scenarios. This is confirmed by inspecting per-scenario predicted_verdict: exactly one
unique predicted verdict per trained cell. The test set contains 332 SUPPORTS / 332 NEI /
336 REFUTES, so predicting SUPPORTS always yields 332/1000 = 0.332, equal to the class
prior — pure chance performance.

The constant mass output (single unique m_S and m_Theta per seed across all 1000 scenarios)
confirms that the GNN learned no discriminative signal.

**H1 null result**: No aggregator differentiation is observable because the training signal
is absent; H1 cannot be evaluated from these results.


## Finding-C1.b: H2 — Spearman rho(m_Theta, planted_thinness)

**Hypothesis**: m_Theta decreases monotonically with planted_thinness (more evidence sources
→ lower uncertainty).

**Result**: spearman_rho = **0.0** for ALL trained cells (canonical through top1_fusion).

| Cell               | spearman_thinness | CI low  | CI high |
|--------------------|-------------------|---------|---------|
| canonical          | 0.0               | 0.0     | 0.0     |
| coherence_off      | 0.0               | 0.0     | 0.0     |
| dirichlet_edl_readout | 0.0            | 0.0     | 0.0     |
| single_layer       | 0.0               | 0.0     | 0.0     |
| softmax_readout    | 0.0               | 0.0     | 0.0     |
| top1_fusion        | 0.0               | 0.0     | 0.0     |
| uniform_aggregator | 0.0               | 0.0     | 0.0     |
| teacher_only       | **-0.739**        | -0.739  | -0.739  |

**Root cause**: For all trained cells, m_Theta is a single constant value shared by every
scenario in a given seed run. With only one unique value, the Dempster-Shafer variance check
(`len(np.unique(m_theta_vals)) < 2`) short-circuits and returns 0.0.

**teacher_only interpretation**: The -0.739 correlation is a confound, not an H2 signal. It
arises because NEI and REFUTES scenarios always have planted_thinness=0 and m_Theta=1.0
(vacuous mass — SPLADE finds no matching infons), while SUPPORTS scenarios have
planted_thinness > 0 and m_Theta ≈ 0.034 (teacher signal present). This measures the
SUPPORTS vs non-SUPPORTS distinction, not the within-SUPPORTS thinness gradient. Within
SUPPORTS, m_Theta is essentially constant across thinness=2,4,10,20.

**H2 weak — rho = 0.0** for all trained cells due to constant mass output.


## Finding-C2: Stratified Outliers

Canonical cell (seed=42), per-verdict accuracy:

| Oracle verdict | Accuracy | n    |
|----------------|----------|------|
| SUPPORTS       | 1.0000   | 332  |
| NEI            | 0.0000   | 332  |
| REFUTES        | 0.0000   | 336  |

**Finding**: NEI and REFUTES strata both show 0.0 accuracy (well below the 0.5 threshold).
This holds for all 8 trained cells across all 5 seeds. The pattern is deterministic: all
trained cells predict SUPPORTS for every scenario.

**Root cause** (same as C1.a): The SPLADE encoder maps all synthetic entity tokens
("Entity 0" through "Entity 99") to the same "entity" token embedding. As a result:
- All SUPPORTS scenarios present identical SPLADE-encoded infons to the GNN
- All REFUTES scenarios return vacuous mass (SPLADE cannot match "demonstrably false" to schema)
- All NEI scenarios return vacuous mass (no supporting triples found)
- The GNN receives the same teacher signal for every training example and collapses to the
  SUPPORTS prior

No beads ticket needed — this is a known limitation of the template/SPLADE mismatch and the
recommended fix is in synthetic_v2 (see Section on Recommended Fixes).


## Finding-C3: Train vs Test Thinness Correlation

**Method**: Using canonical__seed=42.json, grouped SUPPORTS scenarios by planted_thinness
and computed mean m_S and mean m_Theta per group.

| planted_thinness | n  | mean m_S | mean m_Theta |
|------------------|----|----------|--------------|
| 2                | 83 | 0.7027   | 0.2853       |
| 4                | 83 | 0.7027   | 0.2853       |
| 10               | 83 | 0.7027   | 0.2853       |
| 20               | 83 | 0.7027   | 0.2853       |

**Finding-C3.a**: No thinness signal in trained model. Mean m_Theta and mean m_S are
**identical** across all four thinness levels (2, 4, 10, 20) — the model outputs the same
mass regardless of planted evidence redundancy. Expected H2 pattern (m_Theta decreasing with
thinness) is entirely absent.

**Root cause**: SPLADE conflates entity IDs at encoding time, so the graph has identical
structure and identical edge features for evidence sets of size 2 and 20 — the GNN cannot
distinguish them.


## Finding-C4: Bootstrap CI Cross-Check

**Method**: Loaded per-seed polarity_acc for canonical and uniform_aggregator (5 seeds each:
0, 1, 7, 13, 42). Computed pairwise differences and applied `paired_bootstrap_ci`.

| Seed  | canonical | uniform_aggregator | difference |
|-------|-----------|--------------------|------------|
| 0     | 0.332     | 0.332              | 0.0        |
| 1     | 0.332     | 0.332              | 0.0        |
| 7     | 0.332     | 0.332              | 0.0        |
| 13    | 0.332     | 0.332              | 0.0        |
| 42    | 0.332     | 0.332              | 0.0        |

| n_seeds | CI low | CI high | mean_difference |
|---------|--------|---------|-----------------|
| 3       | 0.0    | 0.0     | 0.0             |
| 5       | 0.0    | 0.0     | 0.0             |

**Finding**: Both 3-seed and 5-seed bootstrap CIs collapse to the point [0.0, 0.0] —
perfectly overlapping because all differences are exactly zero. The CIs are consistent with
each other but indicate no detectable difference between typed_ikl and uniform_mean
aggregators on this corpus.


## Finding-C5: H2 Transfer from Epic 01 Canonical Config

**From aggregate.json** for canonical cell:
- `spearman_thinness`: 0.0 (mean across 5 seeds)
- `spearman_thinness_ci_low`: 0.0
- `spearman_thinness_ci_high`: 0.0

**Finding-C5.a**: H2 completely absent (rho = 0.0) for canonical config on synthetic_v1.
The thinness signal cannot transfer from training to evaluation because:
1. The SPLADE encoder produces identical encodings for "Entity N is related to Entity M"
   regardless of N and M, so evidence of thinness=2 and thinness=20 looks identical
2. No gradient signal exists to distinguish SUPPORTS, REFUTES, or NEI scenarios
3. The GNN collapses to a constant mass function (the SUPPORTS prior)

**Recommended fix for synthetic_v2**: Use distinguishable entity names (e.g., proper nouns
from a name list) so SPLADE can differentiate entities. This will allow the teacher mass to
vary across scenarios and provide genuine training signal.


## Finding-C6: H1 Untestable — Single hop_count=2

**Design limitation**: All 1000 test scenarios in synthetic_v1 have `planted_hop_count=2`.
There is no variation in compositional depth across the test set.

**Finding-C6.a**: H1 is untestable on synthetic_v1. With a single hop_count value, the
h1_accuracy_by_hop.json table has only one row per aggregator (hop_count=2), and
both typed_ikl and uniform_mean show identical accuracy (0.332). No curve over hop counts
can be plotted. The note in h1_accuracy_by_hop.json itself states: "single hop_count=2
(full range pending)".

**Paper reframing**: The H1 evaluation (typed_ikl beats uniform_mean at higher compositional
depth) requires:
1. Varied planted_hop_count (e.g., 1, 2, 3, 4) in synthetic_v2, OR
2. Real-world claims from AVeriTeC (Epic 03) where hop count varies naturally

The synthetic_v1 results should be presented as a corpus validation failure rather than an
H1 null result, since the hypothesis was never tested.


## Root Cause Analysis

The core limitation is the SPLADE encoder cannot distinguish synthetic template entities
because all "Entity N" tokens map to the same BERT embedding. This produces a cascade of
failures:

1. **Training signal collapse**: Every scenario provides identical SPLADE-encoded inputs to
   the GNN regardless of entity identity, thinness, or verdict. The GNN loss is the same for
   all training examples → the model learns nothing beyond the class prior.

2. **Constant mass output**: At inference time, every scenario produces an identical 4-mass
   vector (specific to each random seed). All 1000 scenarios receive the same prediction.
   For canonical (seed=42): (0.703, 0.004, 0.008, 0.285) for every scenario.

3. **REFUTES scenarios are vacuous**: The REFUTES template uses "demonstrably false"
   language that SPLADE cannot match to any schema type. REFUTES scenarios return
   `(0.0, 0.0, 0.0, 1.0)` (full vacuous mass) for teacher_only — and since the GNN also
   can't learn the difference, trained cells return the SUPPORTS prior for REFUTES too.

4. **teacher_only partial success**: The teacher_only cell achieves 0.664 accuracy because
   the raw DS combination correctly identifies SUPPORTS (SPLADE does retrieve infons with
   polarity signal) and NEI (vacuous mass → NEI prediction via pignistic rule). However, it
   fails on REFUTES (also returns vacuous → NEI, not REFUTES). This 0.664 is the ceiling for
   any model on synthetic_v1 without corpus fixes.

5. **No aggregator differentiation**: Since all GNN outputs are identical, typed_ikl and
   uniform_mean produce the same mass for every scenario. H1 cannot be evaluated.


## Recommended Fixes for synthetic_v2

1. **Distinguishable entity names**: Replace "Entity N" with proper nouns (e.g., person
   names, city names from a curated list). SPLADE will then produce distinct encodings
   per entity, enabling the GNN to learn claim-specific patterns.

2. **REFUTES corpus fix**: REFUTES templates should include statements that SPLADE can
   index (e.g., "Entity X did NOT achieve Y" rather than "demonstrably false"). The DS
   polarity signal from explicit negation should produce m_R > m_S.

3. **Varied hop_count**: Introduce planted_hop_count ∈ {1, 2, 3, 4} to enable H1 evaluation
   on synthetic data before moving to AVeriTeC.

4. **Evidence set size variation**: Ensure each thinness level maps to measurably different
   infon counts in the vector store so the GNN can detect evidence redundancy.

5. **Smoke test before full run**: Add an assertion that `len(set(pred_verdicts)) > 1`
   after training to detect collapsed models early.
