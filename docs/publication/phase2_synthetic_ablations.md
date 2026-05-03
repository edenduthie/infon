# Phase 2 Technical Memo: Synthetic Stress Dataset and Full Ablation Matrix

**Date**: 2026-05-03
**Status**: Internal research report — draft for review
**Epic**: Epic 02 — Synthetic stress dataset + full ablation matrix

---

## 1. Generator Design and Oracle Correctness

### Template-Based Generation (No LLM)

The `synthetic_v1` corpus is generated entirely by a deterministic template engine in
`reference_v2/reference_v2/synthetic/generator.py`. No language model or LLM is involved at
any stage. Sentence text is constructed by Python string interpolation from a small set of
fixed templates; the only randomness is the verdict assignment (RNG-sampled from class
proportions) and the noise integers embedded in distractor sentences.

This design was deliberate: LLM-generated text would introduce uncontrolled lexical variation
that could mask or confound failures at the encoder level. With fixed templates, every failure
observed is attributable to the pipeline, not to text quality.

### Three Axes

The generator exposes three experimental axes:

| Axis | Parameter | Scenario attribute | Hypothesis |
|------|-----------|-------------------|------------|
| H1 | `compositional_depth` | `planted_hop_count` | Typed aggregator outperforms uniform at higher hop count |
| H2 | `evidence_redundancy` | `planted_thinness` | m(Theta) decreases as more supporting sentences are present |
| Confound | `contradiction_density` | verdict == REFUTES fraction | Mode collapse under adversarial evidence |

The `nei_fraction` parameter controls the NEI class proportion independently of `contradiction_density`.

### Scenario Dataclass

Each scenario is an immutable `Scenario` dataclass:

```
corpus_sentences: tuple[Sentence, ...]   # all sentences (supporting + distractor)
planted_verdict:  str                    # "SUPPORTS" | "REFUTES" | "NEI"
planted_thinness: int                    # count(s for s in corpus_sentences if s.supports_query)
planted_hop_count: int                   # equals compositional_depth parameter
```

`Sentence` carries `text: str` and `supports_query: bool`.

### Oracle Correctness Invariant

`planted_thinness` is derived directly from the corpus, not from a separate bookkeeping
variable:

```python
planted_thinness = sum(1 for s in sentences if s.supports_query)
```

This guarantees that `planted_thinness` always equals the number of sentences flagged
`supports_query=True`, by construction. Manual spot-check B.1 confirmed 10/10 oracle labels
correct on sampled scenarios.

### Determinism Guarantee

`Generator.__init__` stores the seed but does not create an RNG. `generate()` re-seeds at
the very start of every call:

```python
rng = np.random.default_rng(self._seed)
```

This means two `Generator` instances with the same seed produce byte-equal output regardless
of call order or intervening Python state. Independent re-runs with `--seed 42` produce
byte-identical JSONL files (verified via SHA-256 hashes; see Section 2).

### B.1 Spot-Check

Oracle label correctness was verified by sampling 10 scenarios and comparing
`planted_thinness` to manual counts of `supports_query=True` sentences. Result: 10/10
correct.

---

## 2. Splits and Stratification

### Split Sizes

The corpus is divided into three non-overlapping sets:

| Split | Scenarios |
|-------|-----------|
| Train | 8 000 |
| Dev   | 1 000 |
| Test  | 1 000 |

### Generation Command

```bash
PYTHONPATH=src .venv/bin/python -m reference_v2.synthetic.generate \
  --train 8000 --dev 1000 --test 1000 --seed 42 \
  --compositional-depth 2 --contradiction-density 0.33 --nei-fraction 0.33 \
  --out experiments/data/synthetic_v1/
```

Wall-clock time: ~0.192 s (~89 743 scenarios/sec). Output format: JSONL, one object per
line.

### Thinness-Curriculum Stratification

The test split is stratified across four `evidence_redundancy` strata
(`_CURRICULUM_STRATA = (1, 2, 5, 10)`) so that every stratum is represented at the
minimum required count. With `test_size = 1000` and the proportional scaling rule
`max(1, int(1000 / 400 * 100))`, the minimum per stratum is 250.

Actual test stratum counts:

| evidence_redundancy | count |
|---------------------|-------|
| 1                   | 250   |
| 2                   | 250   |
| 5                   | 250   |
| 10                  | 250   |

All four strata meet the 250 minimum (requirement: ≥ 100 at full benchmark scale).

Train and dev splits are not stratified; they receive scenarios from the leftover pool after
the guaranteed test allocation. Observed distributions:

**Train (8 000)**

| evidence_redundancy | count |
|---------------------|-------|
| 1                   | 1 996 |
| 2                   | 2 024 |
| 5                   | 1 986 |
| 10                  | 1 994 |

**Dev (1 000)**

| evidence_redundancy | count |
|---------------------|-------|
| 1                   | 254   |
| 2                   | 226   |
| 5                   | 264   |
| 10                  | 256   |

### Verdict Distribution (Test Split)

| verdict  | count |
|----------|-------|
| SUPPORTS | 332   |
| NEI      | 332   |
| REFUTES  | 336   |

Class proportions are approximately equal (~33.2% each), making 0.332 the majority-class
baseline accuracy.

### Disjoint ID Guarantee

The split algorithm in `splits.py` builds a single pool of `total = train + dev + test`
scenarios, assigns a unique `scenario_id` (formatted `scenario_{i:05d}`) to each, then
allocates to test first (guaranteed strata + random fill), then shuffles the leftover for
train/dev. IDs are never reused across splits.

### SHA-256 Checksums

```
e96f9dcbf74b2282ea56aa8fc080756e5a306cc5166d349583a66dcc19f3c53b  train.jsonl
7acdbf8ff2081a8614c2b1433ba62997232a56a32a77caa11a5536319a188921  dev.jsonl
32c8934de6f90c9627b22a874770b2103de27b19844d8b1a94abd2d8e24cc337  test.jsonl
```

Stored at `experiments/data/synthetic_v1/SHA256SUMS`. Verified: two independent generation
runs with `--seed 42` produce byte-identical files.

---

## 3. Ablation Matrix (Full Table)

Eight ablation cells were evaluated over 3 seeds each. Results are from
`experiments/results/ablation_matrix/aggregate.json`.

| Cell | polarity_acc_mean ± std | spearman_thinness | ece | brier | aurc |
|------|------------------------|------------------|-----|-------|------|
| canonical | 0.332 ± 0.000 | 0.000 | 0.378 | 0.923 | 0.290 |
| coherence_off | 0.332 ± 0.000 | 0.000 | 0.480 | 1.029 | 0.188 |
| dirichlet_edl_readout | 0.332 ± 0.000 | 0.000 | 0.269 | 0.850 | 0.399 |
| single_layer | 0.332 ± 0.000 | 0.000 | 0.399 | 0.941 | 0.269 |
| softmax_readout | 0.332 ± 0.000 | 0.000 | 0.656 | 1.312 | 0.012 |
| top1_fusion | 0.332 ± 0.000 | 0.000 | 0.412 | 0.953 | 0.256 |
| uniform_aggregator | 0.332 ± 0.000 | 0.000 | 0.382 | 0.927 | 0.285 |
| teacher_only | **0.664 ± 0.000** | **-0.739** | 0.321 | 0.673 | 0.015 |

Notes:
- All 8 cells have `n_seeds = 3`.
- `teacher_only` uses `no_training=True`; the GNN is bypassed and the raw DS-combined teacher
  mass is used directly as output.
- Polarity accuracy CIs collapse to a single point: all trained cells have CI = [0.332, 0.332]
  with std = 0.0. `teacher_only` has CI = [0.664, 0.664] with std = 0.0.
- `spearman_thinness` is the Spearman correlation between per-scenario m(Theta) and
  `planted_thinness`.
- The aurc and ece/brier values for trained cells are meaningful only as baseline artifacts;
  they do not reflect discriminative performance (see Section 6).

Cell configurations:

| Cell | aggregator | readout | n_layers | coherence_weight | decisive_top_k |
|------|-----------|---------|----------|-----------------|---------------|
| canonical | typed_ikl | ds_4mass | 2 | 1.0 | 2 |
| coherence_off | typed_ikl | ds_4mass | 2 | 0.0 | 2 |
| dirichlet_edl_readout | typed_ikl | dirichlet_edl | 2 | 1.0 | 2 |
| single_layer | typed_ikl | ds_4mass | 1 | 1.0 | 2 |
| softmax_readout | typed_ikl | softmax_temperature | 2 | 0.0 | 2 |
| top1_fusion | typed_ikl | ds_4mass | 2 | 1.0 | 1 |
| uniform_aggregator | uniform_mean | ds_4mass | 2 | 1.0 | 2 |
| teacher_only | typed_ikl | ds_4mass | 2 | 1.0 | 2 |

---

## 4. Headline Tables (H1, H2, AURC, ECE/Brier)

### H1: Accuracy by Hop Count

Source: `experiments/results/figures/h1_accuracy_by_hop.json`

| aggregator | hop_count | polarity_acc_mean | CI low | CI high | note |
|-----------|-----------|------------------|--------|---------|------|
| typed_ikl | 2 | 0.332 | 0.332 | 0.332 | single hop_count=2 (full range pending) |
| uniform_mean | 2 | 0.332 | 0.332 | 0.332 | single hop_count=2 (full range pending) |

H1 is untestable on `synthetic_v1`: the generator was called with fixed
`compositional_depth=2`, so all scenarios have `planted_hop_count=2`. There is no variation
in hop count to measure. See Finding-C6.a (Section 6).

### H2: Spearman rho(m_Theta, planted_thinness)

Source: `experiments/results/figures/h2_rho_table.json`

| cell | spearman_thinness_mean | CI low | CI high |
|------|----------------------|--------|---------|
| canonical | 0.000 | 0.000 | 0.000 |
| coherence_off | 0.000 | 0.000 | 0.000 |
| dirichlet_edl_readout | 0.000 | 0.000 | 0.000 |
| single_layer | 0.000 | 0.000 | 0.000 |
| softmax_readout | 0.000 | 0.000 | 0.000 |
| top1_fusion | 0.000 | 0.000 | 0.000 |
| uniform_aggregator | 0.000 | 0.000 | 0.000 |
| teacher_only | **-0.739** | -0.739 | -0.739 |

All trained cells: rho = 0.0 exactly. See Finding-C1.b (Section 6) for the root cause.
The teacher_only -0.739 is a confound, not an H2 signal (explained in Section 6).

### AURC Table

Source: `experiments/results/figures/aurc_table.json` (5-seed runs from canonical_cells
experiments)

| cell | readout | aurc_mean | aurc_std | n_seeds |
|------|---------|-----------|----------|---------|
| canonical | ds_4mass | 0.2868 | 0.0102 | 5 |
| softmax_readout | softmax_temperature | 0.0115 | 0.0015 | 5 |
| dirichlet_edl_readout | dirichlet_edl | 0.3929 | 0.0296 | 5 |

AURC (Area Under the Risk-Coverage curve) measures selective prediction quality. Higher AURC
indicates better risk-coverage tradeoff. Note: these AURC values are computed on the same
collapsed predictions; differences between readouts reflect confidence calibration artifacts
rather than discriminative performance.

### ECE and Brier Score

Source: `experiments/results/figures/ece_brier_table.json`

| cell | ece | brier |
|------|-----|-------|
| canonical | 0.381 | 0.925 |
| coherence_off | 0.481 | 1.031 |
| dirichlet_edl_readout | **0.275** | **0.853** |
| single_layer | 0.388 | 0.932 |
| softmax_readout | 0.656 | 1.313 |
| teacher_only | 0.321 | 0.673 |
| top1_fusion | 0.408 | 0.950 |
| uniform_aggregator | 0.382 | 0.927 |

ECE (Expected Calibration Error) and Brier score are lower-is-better. The dirichlet_edl
readout shows the best calibration among trained cells, and teacher_only is best overall.
Softmax_readout is badly calibrated (ECE=0.656), consistent with overconfident class
assignment.

---

## 5. Headline Figures

Four figures are in `experiments/results/figures/`.

### Figure 1: h1_accuracy_by_hop.png — H1 Accuracy Bar Chart

This bar chart displays polarity accuracy by hop count for `typed_ikl` vs `uniform_mean`
aggregators. Because `synthetic_v1` has only a single hop count value (2), the chart contains
only one bar per aggregator. Both bars are at 0.332 (chance level). The chart is included for
completeness but contains no useful H1 comparison. A full H1 evaluation requires
`compositional_depth ∈ {1, 2, 3, 4}` in `synthetic_v2` or natural hop-count variation in
AVeriTeC (Epic 03).

### Figure 2: h2_theta_vs_thinness.png — H2 m(Theta) vs evidence_redundancy Scatter

This scatter plot shows m(Theta) (vertical axis) against `evidence_redundancy` (horizontal
axis) for the canonical cell. The expected pattern (decreasing m(Theta) as evidence redundancy
increases, reflecting lower uncertainty with more supporting sentences) is entirely absent.
All plotted points cluster at a single m(Theta) value regardless of evidence_redundancy
stratum, confirming the rho = 0.0 result from the H2 table. The teacher_only cell would show
a step-function pattern (m_Theta ≈ 1.0 for REFUTES/NEI, ≈ 0.034 for SUPPORTS), but no
within-SUPPORTS gradient is present.

### Figure 3: reliability_diagrams.png — Calibration Curves for 3 Readouts

Reliability diagrams for `ds_4mass`, `softmax_temperature`, and `dirichlet_edl` readouts.
A well-calibrated model would follow the diagonal. The diagrams show the extent of
overconfidence or underconfidence for each readout. Consistent with the ECE table:
`dirichlet_edl` is closest to the diagonal, `softmax_temperature` is most deviant. Because
all cells collapse to the same predicted class (SUPPORTS), the reliability diagrams reflect
confidence distribution artifacts of each readout's parameterization rather than genuine
calibration learned from discriminative signal.

### Figure 4: risk_coverage.png — Risk-Coverage Curves for 3 Readouts

Risk-coverage curves for the same three readouts. Higher area under the curve indicates
better selective prediction (the model abstains on low-confidence predictions). The ordering
is `dirichlet_edl` (AURC ≈ 0.393) > `ds_4mass` (AURC ≈ 0.287) > `softmax_temperature`
(AURC ≈ 0.012). Again, these differences reflect the confidence distribution of each
readout on a fully-collapsed classifier, not genuine selective risk reduction. The near-zero
AURC for `softmax_temperature` indicates it assigns high confidence to all predictions
uniformly, leaving no room to selectively abstain.

---

## 6. Findings (Including Null and Negative)

### Finding-C1.a: H1 Null — All Trained Cells at Chance Accuracy

**Hypothesis**: The canonical `typed_ikl` aggregator should outperform `uniform_mean` at
`hop_count=2`.

**Result**: All 7 trained cells achieve exactly 0.332 polarity accuracy (chance level for a
balanced 3-way classification). Standard deviation is 0.0 across all 3 seeds. Confidence
intervals collapse to a single point: [0.332, 0.332].

The `teacher_only` cell, which bypasses GNN training and uses oracle-derived teacher masses
directly, achieves 0.664.

**Root cause**: The SPLADE encoder maps all "Entity N" tokens to the same BERT embedding.
Across all 1000 test scenarios, every trained cell outputs an identical 4-mass vector
(e.g., for canonical at seed=42: `m_S=0.703, m_Alignment=0.004, m_Distance=0.008,
m_Theta=0.285` for every scenario). With a single unique predicted verdict per cell (always
"SUPPORTS"), accuracy equals the SUPPORTS class frequency: 332/1000 = 0.332.

**Implication**: GNN training is a no-op on `synthetic_v1`. No ablation cell comparison
(aggregator, readout, depth, coherence) is meaningful because all differences are zero.

### Finding-C1.b: H2 Null — Spearman rho = 0.0

**Hypothesis**: m(Theta) should decrease monotonically as `planted_thinness` increases
(more supporting sentences → lower evidential uncertainty).

**Result**: Spearman rho = 0.0 for all 7 trained cells. For the canonical cell (seed=42),
m(Theta) is byte-identical across thinness levels 2, 4, 10, and 20:

| planted_thinness | n  | mean m_S | mean m_Theta |
|------------------|----|----------|--------------|
| 2 | 83 | 0.7027 | 0.2853 |
| 4 | 83 | 0.7027 | 0.2853 |
| 10 | 83 | 0.7027 | 0.2853 |
| 20 | 83 | 0.7027 | 0.2853 |

**Root cause**: Identical SPLADE encodings mean the GNN sees the same input regardless of
evidence_redundancy. The Spearman computation short-circuits to 0.0 because
`len(np.unique(m_theta_vals)) < 2`.

**teacher_only rho = -0.739**: This is a confound, not an H2 signal. NEI and REFUTES
scenarios have `planted_thinness=0` and `m_Theta=1.0` (vacuous mass — SPLADE finds no
matching infons for these templates). SUPPORTS scenarios have `planted_thinness > 0` and
`m_Theta ≈ 0.034`. The correlation measures the SUPPORTS vs. non-SUPPORTS distinction at
the binary level. Within the SUPPORTS class, m(Theta) is essentially constant across
evidence_redundancy levels 1, 2, 5, 10 — no within-class gradient exists.

**Conclusion**: H2 (thinness correlation) is untestable on `synthetic_v1` until the SPLADE
encoding issue is resolved in `synthetic_v2`.

### Finding-C6.a: H1 Untestable — Single hop_count=2

**Design limitation**: The generation command was run with `--compositional-depth 2` as a
fixed constant. All 10 000 scenarios (train/dev/test) have `planted_hop_count=2`.

**Consequence**: The h1_accuracy_by_hop table has exactly one row per aggregator. No curve
over hop counts can be constructed. The `h1_accuracy_by_hop.json` file itself notes:
`"note": "single hop_count=2 (full range pending)"`.

**Required fix for synthetic_v2**: Vary `compositional_depth ∈ {1, 2, 3, 4}` so the test
split spans multiple hop counts. Alternatively, use AVeriTeC (Epic 03) where hop count
varies naturally across real claims.

### Finding-C3.a: No Thinness Signal

Within-SUPPORTS m_Theta is byte-identical across all four thinness strata (2, 4, 10, 20).
The expected monotone decrease of m_Theta with evidence_redundancy is entirely absent.
Root cause: identical SPLADE encoding (same as C1.b).

### Finding-C2: Stratified Outliers

Per-verdict accuracy for the canonical cell (seed=42):

| Oracle verdict | Accuracy | n |
|----------------|----------|---|
| SUPPORTS | 1.0000 | 332 |
| NEI | 0.0000 | 332 |
| REFUTES | 0.0000 | 336 |

The model predicts "SUPPORTS" for every single scenario. SUPPORTS accuracy is therefore 1.0
(trivially, by mode collapse to majority class). NEI and REFUTES accuracy is 0.0. This
pattern is deterministic and holds across all 8 trained cells and all seeds.

**Root cause** (same as C1.a): The SPLADE encoder cannot distinguish entity tokens. REFUTES
and NEI scenarios return vacuous teacher mass; SUPPORTS scenarios return a consistent positive
polarity signal. The GNN learns only the SUPPORTS prior.

### REFUTES Schema Gap

The REFUTES sentence template is:

```
"Entity {doc_idx}: fact {doc_idx} is demonstrably false."
```

The phrase "demonstrably false" has no matching tokens in the synthetic schema. SPLADE
retrieves zero infons for REFUTES sentences, producing a fully vacuous teacher mass:
`(m_S=0.0, m_Alignment=0.0, m_Distance=0.0, m_Theta=1.0)`. With no gradient signal
distinguishing REFUTES from NEI, the model cannot learn to predict REFUTES. The result is
0% recall on the REFUTES class across the entire ablation matrix.

This is a corpus design flaw, not a model architecture failure. It is the primary target for
`synthetic_v2` remediation.

---

## 7. Follow-ups for Epic 03

The following issues from `synthetic_v1` are unresolved and must be addressed in
Epic 03 (AVeriTeC benchmarks) or `synthetic_v2`.

**1. H1 (aggregator ablation) — must test on real multi-hop queries (AVeriTeC)**

`synthetic_v1` provides zero evidence for or against the H1 hypothesis (typed_ikl outperforms
uniform_mean at higher compositional depth). The only valid path to H1 evaluation before
`synthetic_v2` is running both aggregator cells on AVeriTeC claims, which naturally span
multiple hop counts. Target: measure polarity accuracy as a function of AVeriTeC-annotated
hop count.

**2. H2 (thinness correlation) — verify rho(m_Theta, evidence_count) >= 0.1 on AVeriTeC**

The H2 hypothesis requires variation in m(Theta) correlated with evidence redundancy. On
real AVeriTeC evidence, the SPLADE encoder will produce distinct encodings for distinct
entity names, enabling the GNN to learn thinness-dependent mass. The acceptance criterion is
Spearman rho >= 0.1 on the AVeriTeC test split.

**3. Schema-text alignment — real corpus uses natural language SPLADE can parse**

`synthetic_v1` failed because template text ("Entity 0", "Entity 1", ...) does not align
with any SPLADE-indexable schema types. AVeriTeC claims and evidence sentences use natural
language entity names and relation descriptions that SPLADE was trained on. This is expected
to resolve the encoder collapse automatically.

**4. REFUTES recall — measure false negative rate on REFUTES in AVeriTeC**

`synthetic_v1` showed 0% recall on the REFUTES class across all cells. Epic 03 must measure
per-class precision and recall on AVeriTeC, with specific attention to REFUTES false negative
rate. Target: REFUTES recall >= 0.5 to demonstrate the schema can distinguish negating
evidence.

**5. synthetic_v2 design — fix corpus and vary compositional depth**

`synthetic_v2` (a separate task, not part of Epic 03) must address:

- **REFUTES template**: Replace "demonstrably false" with language containing explicit
  negation tokens that SPLADE can match to a "refutes" schema type (e.g., "did not achieve",
  "is not the case that"). This will give REFUTES scenarios a non-vacuous teacher mass.
- **Distinguishable entities**: Replace "Entity N" with proper nouns from a curated name list
  so SPLADE produces distinct per-entity embeddings, enabling genuine training signal.
- **Variable compositional depth**: Generate scenarios with
  `compositional_depth ∈ {1, 2, 3, 4}` and stratify the test split over hop counts.
- **Early detection**: Add a post-training assertion that `len(set(predicted_verdicts)) > 1`
  before committing any results. A collapsed model should fail fast.

---

## Summary

`synthetic_v1` is a validated, deterministic corpus with correct oracle labels and a clean
stratified test split. The infrastructure (generator, splits, ablation harness, aggregate
reporting) is correct and ready for `synthetic_v2`.

However, the corpus itself is not suitable for evaluating H1 or H2. The SPLADE encoder
produces indistinguishable representations for all synthetic entity tokens, causing complete
training signal collapse. All 7 trained cells predict "SUPPORTS" for every scenario, yielding
chance-level accuracy (0.332) with zero variance. The REFUTES class has 0% recall due to a
schema gap in the refuting sentence template.

These are corpus design failures, not architectural failures. The ablation matrix results
cannot be used to compare aggregator types, readouts, depth, or coherence weights — all
differences are zero. The only meaningful signal in `synthetic_v1` is the `teacher_only`
ceiling (0.664), which confirms that the DS combination layer and polarity teacher are
functioning correctly on SUPPORTS scenarios.

All H1 and H2 hypotheses remain open and must be evaluated on AVeriTeC (Epic 03) or
`synthetic_v2`.
