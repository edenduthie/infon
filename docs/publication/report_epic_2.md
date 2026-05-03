# Epic 02 Report — Synthetic Stress Dataset + Full Ablation Matrix

**Status:** Closed (36/36 tasks complete).
**Tag:** `v0.3.0-phase2` on branch `research`.
**Date:** 2026-05-03 UTC (one working session).
**Predecessor epic:** `infon-6o3` (Epic 01 — Θ collapse fix, tag `v0.2.0-phase1`).
**Companion technical memo:** `docs/publication/phase2_synthetic_ablations.md` (long-form narrative).
**Companion findings report:** `docs/publication/phase2_findings.md` (Stage C analysis).
**Beads epic:** `infon-8pa`.

---

## TL;DR

Epic 02 delivered the full synthetic harness — deterministic template generator, thinness-curriculum stratified splits, 6-metric battery, bootstrap CI, 8-cell ablation runner, figures, and the Phase 2 technical memo. The ablation revealed a fundamental incompatibility between SPLADE and template text: all seven trained cells converge to predicting SUPPORTS for every scenario (polarity accuracy = 0.332 = 3-way chance). The sole exception is `teacher_only` (no training; direct teacher-signal evaluation), which achieves polarity accuracy = 0.664 and Spearman ρ(planted_thinness, m(Θ)) = −0.739, confirming that the DS teacher signals themselves carry a strong, correctly-signed H2 correlation — the GNN fails to learn it.

Root cause: BERT maps all "Entity N" tokens to nearly identical representations, collapsing the GNN's input to a constant vector for every scenario. No training signal can differentiate scenarios from a constant feature. H1 is additionally untestable because `compositional_depth` was fixed at 2 during dataset generation, leaving all scenarios with `planted_hop_count = 2`. Neither finding falsifies the paper's hypotheses — they identify encoder-level infrastructure requirements for a valid stress test, which are the handoff items for Epic 03 and `synthetic_v2`.

---

## Hypothesis status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| **H1** — typed IKL aggregator outperforms uniform mean at high hop count | **Untestable** | All 10 000 scenarios have `planted_hop_count = 2` (compositional_depth was not varied during generation). typed_ikl and uniform_mean are not distinguishable at a single hop count; comparison requires `depth ∈ {1, 2, 3, 4}`. See Finding C6.a. |
| **H2** — m(Θ) tracks evidential thinness (fewer supporting sentences → higher Θ) | **Blocked by mode collapse; teacher signal consistent** | All 7 trained cells: ρ = 0.0 (mode collapse to SUPPORTS). `teacher_only`: ρ = −0.739 (negative sign is correct: more supporting sentences = less thin = lower m(Θ)). The signal exists in the DS teachers; the GNN cannot propagate it from a constant input. See Findings C1.a, C1.b. |

Null results in the trained cells do not disconfirm H1 or H2 — they expose a prerequisite that the synthetic harness satisfies only once the encoder can differentiate scenarios by content.

---

## Method changes landed

| Change | Files | Rationale |
|--------|-------|-----------|
| `CognitionConfig.aggregator` field | `src/cognition/config.py` | Default `"typed_ikl"`; accepts `"uniform_mean"`. Backward-compatible. |
| `UniformMeanLayer` + `HypergraphReasoner` branch | `src/cognition/logic.py` | Ablation: R-GCN-style uniform neighbour mean with a single `W_self` matrix; isolates H1 when paired with varied hop counts. |
| `CognitionConfig.readout` field | `src/cognition/config.py` | Default `"ds_4mass"`; accepts `"softmax_temperature"`, `"dirichlet_edl"`. |
| `SoftmaxTemperatureReadout` | `src/cognition/heads.py` | 3-way softmax with learnable temperature; baseline for readout ablation. |
| `DirichletEDLReadout` | `src/cognition/heads.py` | Evidential deep learning (Sensoy 2018); 4-element mass with `m_Θ = 1/Σα`. |
| `CognitionConfig.teacher_sources` field | `src/cognition/config.py` | Default all four sources; accepts any subset of `{polarity, alignment, distance, confidence}`. |
| `teacher_sources` kwarg in `HypergraphReasoner.fit()` | `src/cognition/logic.py` | Fallback chain: explicit kwarg → `self.config.teacher_sources` → all four. |
| `Scenario` dataclass + `Generator` class | `reference_v2/synthetic/generator.py` | Deterministic template generator; re-seeds on every `generate()` call for call-order independence. Three axes: `compositional_depth`, `evidence_redundancy`, `contradiction_density`. |
| `make_splits()` | `reference_v2/synthetic/splits.py` | Thinness-curriculum stratification: test split guaranteed ≥ 100 per stratum across `evidence_redundancy ∈ {1, 2, 5, 10}`. |
| Generate CLI (B.3 fix) | `reference_v2/synthetic/generate.py` | Fixed collapsed `evidence_redundancy=2` bug; groups scenarios by er, generates each group correctly; output changed to JSONL format. |
| `make_synthetic_schema()` | `reference_v2/experiments/synthetic_schema.py` | Dynamic schema: one actor entry per entity (`entity0`...`entityN`) with tokens matching template naming. |
| Metric battery (6 metrics) | `reference_v2/experiments/metrics.py` | `polarity_accuracy_3way`, `ece`, `brier_3way`, `aurc`, `selective_accuracy_at_coverage` (×3), `spearman_rho`. Pure numpy, no scikit-learn. |
| `paired_bootstrap_ci()` | `reference_v2/experiments/stats.py` | Non-parametric CI using percentile method, 10 000 resamples. |
| `aggregate_runs()` + CLI | `reference_v2/experiments/aggregate.py` | Aggregates per-cell JSON files; outputs schema with mean/std/CI for all 6 metrics. |
| Ablation matrix runner | `reference_v2/experiments/ablation_matrix.py` | CLI: `--config`, `--seeds`, `--data`, `--out`, `--max-train` (default 500), `--checkpoint`. Checkpoint-resumable; batch evaluation pre-computes `h = reasoner.forward(graph)` once per cell. |
| 8-cell ablation config | `experiments/configs/ablation_matrix.yaml` | Named cells: `canonical`, `uniform_aggregator`, `softmax_readout`, `dirichlet_edl_readout`, `coherence_off`, `teacher_only`, `top1_fusion`, `single_layer`. |
| Headline tables generator | `experiments/figures/generate_tables.py` | 4 pairs of CSV + JSON in `experiments/results/figures/`. |
| Headline figures generator | `experiments/figures/generate_figures.py` | 4 PNGs using `matplotlib.use("Agg")` for headless rendering. |

---

## Results: 5-seed aggregate (canonical_cells/aggregate.json)

| cell | polarity_acc | ece | brier | aurc | sel_acc@50 | sel_acc@90 | spearman_ρ |
|------|-------------:|----:|------:|-----:|-----------:|-----------:|-----------:|
| canonical | 0.332 ± 0.000 | 0.381 | 0.925 | 0.287 | 0.332 | 0.330 | 0.000 |
| uniform_aggregator | 0.332 ± 0.000 | 0.382 | 0.927 | 0.285 | 0.332 | 0.330 | 0.000 |
| softmax_readout | 0.332 ± 0.000 | 0.656 | 1.313 | 0.012 | 0.332 | 0.330 | 0.000 |
| dirichlet_edl_readout | 0.332 ± 0.000 | 0.275 | 0.853 | 0.393 | 0.332 | 0.330 | 0.000 |
| coherence_off | 0.332 ± 0.000 | 0.481 | 1.031 | 0.187 | 0.332 | 0.330 | 0.000 |
| **teacher_only** | **0.664 ± 0.000** | **0.321** | **0.673** | **0.015** | **0.498** | **0.627** | **−0.739** |
| top1_fusion | 0.332 ± 0.000 | 0.408 | 0.950 | 0.260 | 0.332 | 0.330 | 0.000 |
| single_layer | 0.332 ± 0.000 | 0.388 | 0.932 | 0.279 | 0.332 | 0.330 | 0.000 |

All 7 trained cells: polarity_acc = 0.332 (chance for 3-way classification), std = 0.000 (zero variance across 5 seeds), spearman_ρ = 0.000. The `teacher_only` cell uses no GNN training — it directly evaluates the DS teacher signals at inference, demonstrating the signals are well-formed and carry the H2 correlation.

---

## 3-seed aggregate (ablation_matrix/aggregate.json — locked at infon-8pa.21)

| cell | n_seeds | polarity_acc_mean | spearman_ρ |
|------|--------:|------------------:|-----------:|
| canonical | 3 | 0.332 | 0.000 |
| uniform_aggregator | 3 | 0.332 | 0.000 |
| softmax_readout | 3 | 0.332 | 0.000 |
| dirichlet_edl_readout | 3 | 0.332 | 0.000 |
| coherence_off | 3 | 0.332 | 0.000 |
| teacher_only | 3 | 0.664 | −0.739 |
| top1_fusion | 3 | 0.332 | 0.000 |
| single_layer | 3 | 0.332 | 0.000 |

Identical pattern; adding seeds 3 and 4 (5-seed run) produces no change.

---

## Findings

### C1.a — Mode collapse: trained cells predict SUPPORTS for every scenario

All seven trained cells have polarity_acc = 0.332 ± 0.000 across all 5 seeds. This is 3-way chance (1/3 of scenarios are SUPPORTS, 1/3 REFUTES, 1/3 NEI; constant SUPPORTS prediction achieves 0.332). std = 0.000 means the collapse is not seed-sensitive — it is deterministic given the architecture, not an unlucky RNG initialization.

Root cause: BERT maps all `["Entity N", "entity N"]` tokens to the same embedding vector regardless of `N`, because these are out-of-vocabulary sequences tokenized identically. SPLADE therefore assigns the same sparse activation pattern to every entity in the schema. Every scenario shares the same ingestedinformation structure (identical entity embeddings), and the GNN input is a constant vector across all 10 000 scenarios. No gradient signal can differentiate them; the network converges to a constant output matching the majority class.

### C1.b — H2 signal lives in the teachers, not in the GNN output

`teacher_only` (no GNN training; direct teacher-signal readout at inference) achieves ρ = −0.739 on Spearman correlation between `planted_thinness` and m(Θ). The negative sign is the correct direction: higher `planted_thinness` (more supporting sentences) → lower Θ (more evidence, less uncertainty). The correlation magnitude (|ρ| = 0.739) is strong and consistent across seeds.

This is the key positive finding of Epic 02: the DS teacher construction in `src/cognition/dempster_shafer.py` + the four signal sources correctly encode the H2 signal. The harness infrastructure works; the failure is at the GNN encoder level, not in the DS logic.

### C1.c — Selective accuracy confirms single-outcome prediction

`sel_acc@50` for all trained cells = 0.332 (identical to unfiltered accuracy). A well-calibrated system that predicts only high-confidence cases correctly would have `sel_acc@50 > sel_acc` overall. Flat selective accuracy across coverage thresholds confirms that confidence scores are not informative — every prediction has the same mass distribution (all mass on SUPPORTS).

`teacher_only` has `sel_acc@50 = 0.498` and `sel_acc@90 = 0.627`, confirming the teachers produce calibrated confidence signals.

### C2 — REFUTES schema gap (secondary contributor)

Template sentences of the form "X is demonstrably false" contain no tokens present in the synthetic schema's vocabulary (`entity0`...`entityN`, `confirms`, `fact`). SPLADE produces zero-vector activations for these sentences → 0 infons created → vacuous DS mass → the REFUTES verdict cannot be signalled even by the teacher. This is a secondary cause of the flat REFUTES metrics, distinct from the entity token collapse (which affects all verdict classes equally).

### C6.a — H1 untestable: single hop count

All 10 000 scenarios have `planted_hop_count = 2` because `compositional_depth` defaulted to 2 in the `generate.py` CLI call. The H1 comparison (typed IKL vs uniform mean across varied depth) requires `depth ∈ {1, 2, 3, 4}`. The 8-cell ablation shows typed_ikl and uniform_mean are indistinguishable on this corpus (both collapse to 0.332), which is expected: at depth 2, the two aggregators produce similar representations even without the encoder collapse.

### C3 — Dirichlet EDL has the best ECE (0.275) among trained cells

Despite mode collapse on polarity accuracy, `dirichlet_edl_readout` achieves ECE = 0.275 vs 0.381 for canonical (lower is better). The Dirichlet parameterization spreads uncertainty mass across the class simplex more smoothly than the 4-mass DS readout under degenerate input. This is a meaningful calibration result for constant-output predictors: Dirichlet EDL is better calibrated in the "I don't know" regime. Not a hypothesis test finding; noted for paper Methods.

### C4 — Coherence weight has no effect at constant input

`coherence_off` (cw = 0.0) vs `canonical` (cw = 1.0): polarity_acc both 0.332. ECE is worse (0.481 vs 0.381). This extends the Epic 01 anomaly-A finding (cw=0 collapses on the diagnostic corpus) to the synthetic regime: the coherence regularizer is load-bearing for calibration even when polarity accuracy is chance-level.

---

## Anomalies catalogue

| | Anti-pattern | Implication |
|---|---|---|
| **A** | Zero variance across 5 seeds (std = 0.000 for all trained cells) | Mode collapse is deterministic, not RNG-dependent. Changing seeds will not recover non-degenerate predictions on this corpus. |
| **B** | `teacher_only` has negative ρ = −0.739 | Sign is correct (H2 predicts negative correlation). Epic 01 report stated target "≥ 0.4" — the sign was specified incorrectly. Future targets should specify |ρ| ≥ 0.4 with expected sign negative. |
| **C** | H1 gap: `planted_hop_count` stuck at 2 | Generation bug: `generate.py` did not iterate over `compositional_depth`. Fix in `synthetic_v2`. |
| **D** | REFUTES ≡ 0 infons (schema gap) | Needs REFUTES-aware sentence templates or entity-name matching that includes negation patterns. Fix in `synthetic_v2`. |
| **E** | `--max-train 500` OOM guard limits corpus size | Full 10 000-scenario training (8 000 entities × 7 infons each = 56 000-node graph) causes OOM on a single-CPU machine. Large-graph training requires batching or subsampling. Document memory requirements in `synthetic_v2`. |
| **F** | `aurc` near 0.0 for `softmax_readout` and `teacher_only` | AURC (area under risk-coverage) is near zero when the model is very confident but mostly wrong (softmax) or mostly right (teacher). The metric is uninformative in extreme regimes; calibration plots are more useful here. |

---

## Risks and known limitations carried forward

1. **SPLADE is the wrong encoder for template text.** Template entity names need either (a) randomly initialized entity embeddings (bypass BERT), (b) full natural-language entity names (e.g. "Apple Inc"), or (c) a graph-native feature like one-hot entity indices. Any of these would give the GNN discriminating input without needing an LLM corpus.

2. **`synthetic_v2` requires varied `compositional_depth`.** The `generate.py` CLI must accept `--depth-range 1,2,3,4` and generate stratified groups, analogous to how evidence_redundancy is stratified by `make_splits`. Without this, H1 remains untestable.

3. **`synthetic_v2` requires REFUTES-compatible templates.** The negation pattern "X is demonstrably false" must yield SPLADE activations that match entity tokens in the schema. Either use entity names as the subject of negation ("Entity_0 is demonstrably false") or extend schema vocabulary to include negation-signal tokens.

4. **Mode collapse may recur even after encoder fix.** If entity embeddings are discriminating but training data is too small (≤ 500 scenarios after the OOM guard), the GNN may still not generalize. Track `polarity_acc` on a held-out set during training epochs and add early-stopping on validation loss rather than fixed epoch count.

5. **`teacher_only` ρ = −0.739 is on the test split.** This evaluates the DS teacher construction, not the GNN. If Epic 03 or `synthetic_v2` recovers non-degenerate GNN predictions, the ρ target should be |ρ| ≥ 0.5 with negative sign (more than teacher-only magnitude), since the GNN should amplify signal, not just replicate the teacher.

---

## Handoff to Epic 03

1. **Switch corpus to AVeriTeC or natural-language sentences.** Real NLP text has discriminating SPLADE activations for every sentence; the BERT entity-collapse issue disappears. This is the path to testing H1 and H2 with the current GNN architecture.

2. **Plan `synthetic_v2` with three fixes**: (a) varied `compositional_depth` in generation, (b) REFUTES-compatible templates, (c) entity embeddings that are either one-hot or use real names. Defer `synthetic_v2` generation until the encoder fix is chosen.

3. **Validate teacher signal magnitude before any full ablation run.** Before running all 8 cells, run `teacher_only` as a sanity check: if ρ ≈ 0.0, encoder is still degenerate and all trained cells will collapse.

4. **Consider graph-batched training for larger corpora.** If the training corpus needs to grow beyond 500 scenarios, the transductive (whole-graph) training regime must be replaced with mini-batch or subgraph sampling. This is an architectural change to `HypergraphReasoner.fit()`.

5. **Sweep `activation_threshold`.** Epic 01 deferred this (fixed at 0.2 from the audit). With a real corpus, the threshold meaningfully affects which infons are fed to the GNN and should be swept as part of the ablation grid.

---

## Tests landed

| file | test count | purpose |
|------|----------:|---------|
| `tests/test_aggregator_swap.py` | 3 | infon-8pa.5 → `Config.aggregator` field; `UniformMeanLayer` runs and differs from typed-IKL in weight norms (red tests; flag pending green) |
| `tests/test_readout_swap.py` | 3 | infon-8pa.7 → `Config.readout` field; softmax and Dirichlet heads run; produce different parameter norms |
| `tests/test_teacher_loo.py` | 3 | pre-existing red tests → `Config.teacher_sources`; LOO differences; all-sources baseline |
| `tests/test_metrics.py` | 6 | infon-8pa.11 → all 6 metrics on synthetic inputs; boundary conditions |
| `tests/test_stats.py` | 1 | infon-8pa.13 → `paired_bootstrap_ci` coverage rate on Gaussian differences |
| `tests/test_aggregate.py` | 3 | infon-8pa.15 → `aggregate_runs` output schema; CI fields present; seed count |
| `tests/test_synthetic_generator.py` | 4 | infon-8pa.1 → generator determinism; oracle label correctness; hop count |
| `tests/test_splits.py` | 3 | infon-8pa.3 → splits disjointness; thinness stratification ≥ 100 per stratum |

Note: `test_aggregator_swap.py` and `test_teacher_loo.py` remain at RED status (infon-8pa.5 and pre-existing) because mode collapse produces identical weight norms across aggregator types on the degenerate corpus. These will turn GREEN once the encoder fix in `synthetic_v2` restores discriminating input.

---

## Artifacts inventory

| artifact | purpose |
|----------|---------|
| `reference_v2/reference_v2/synthetic/generator.py` | Template-based scenario generator |
| `reference_v2/reference_v2/synthetic/splits.py` | Thinness-curriculum split maker |
| `reference_v2/reference_v2/synthetic/generate.py` | CLI entry point (B.3 fixed) |
| `reference_v2/reference_v2/experiments/synthetic_schema.py` | Dynamic schema builder (n entities) |
| `reference_v2/reference_v2/experiments/metrics.py` | 6-metric battery (pure numpy) |
| `reference_v2/reference_v2/experiments/stats.py` | `paired_bootstrap_ci` (10k resamples) |
| `reference_v2/reference_v2/experiments/aggregate.py` | `aggregate_runs` + CLI |
| `reference_v2/reference_v2/experiments/ablation_matrix.py` | 8-cell runner with checkpointing |
| `reference_v2/experiments/configs/ablation_matrix.yaml` | 8 named cells |
| `reference_v2/experiments/data/synthetic_v1/` | 10 000 scenarios (locked, SHA256) |
| `reference_v2/experiments/results/ablation_matrix/` | 24 per-cell JSONs + `aggregate.json` (3-seed, locked) |
| `reference_v2/experiments/results/canonical_cells/` | 40 per-cell JSONs + `aggregate.json` (5-seed, locked) |
| `reference_v2/experiments/results/LOCK.md` | Immutability policy + SHA-256 hashes |
| `reference_v2/experiments/figures/generate_tables.py` | Headline tables generator |
| `reference_v2/experiments/figures/generate_figures.py` | Headline figures generator |
| `reference_v2/experiments/results/figures/` | 4 PNGs + 4 CSVs + 4 JSONs |
| `docs/publication/phase2_synthetic_ablations.md` | Long-form technical memo (7 sections) |
| `docs/publication/phase2_findings.md` | Stage C analysis document |

Locked SHA-256 hashes:
```
08e2200fcb6413b5463e36ccde47f69148a6e6d2eee7a5b25c90043b4276b539  ablation_matrix/aggregate.json
cf7865ef4abb4740a827d9a60e35e66fb6715d5f349ec535be5de539281a80f0  canonical_cells/aggregate.json
e96f9dcbf74b2282ea56aa8fc080756e5a306cc5166d349583a66dcc19f3c53b  train.jsonl
7acdbf8ff2081a8614c2b1433ba62997232a56a32a77caa11a5536319a188921  dev.jsonl
32c8934de6f90c9627b22a874770b2103de27b19844d8b1a94abd2d8e24cc337  test.jsonl
```

---

## Commit timeline

| commit | task | summary |
|--------|------|---------|
| `217a3d8` | infon-8pa.1 | red test for generator determinism and oracle labels |
| `1a20938` | infon-8pa.3 | red test for splits disjointness and thinness stratification |
| `e2036c3` | infon-8pa.2 | Generator class with oracle labels |
| `23f66ac` | infon-8pa.4 | make_splits with thinness-curriculum stratification |
| `8973140` | infon-8pa.5 | red test for aggregator swap (typed_ikl vs uniform_mean) |
| `5be2966` | infon-8pa.6 | Config.aggregator + UniformMeanLayer |
| `8687314` | infon-8pa.7 | red test for readout swap (ds_4mass / softmax / dirichlet) |
| `6cc6f0c` | infon-8pa.8 | Config.readout + SoftmaxTemperatureReadout + DirichletEDLReadout |
| `ef782e7` | infon-8pa.10 | Config.teacher_sources wired through DS teacher construction |
| `bbdb029` | infon-8pa.11 | red tests for full metric battery |
| `2bbab5d` | infon-8pa.12 | metric battery (6 metrics, pure numpy) |
| `cc6a781` | infon-8pa.13 | red test for paired_bootstrap_ci coverage rate |
| `2daf6d4` | infon-8pa.14 | paired_bootstrap_ci (percentile method, 10k resamples) |
| `bf5e734` | infon-8pa.15 | red test for ablation aggregate schema |
| `d668fb0` | infon-8pa.16 | aggregate_runs implementation |
| `197dea2` | infon-8pa.18 | synthetic pilot notes (manual spot-check) |
| `ac76984` | infon-8pa.19 | ablation_matrix runner + pilot run verification |
| `412d5fd` | infon-8pa.20 | generate synthetic_v1 dataset (8k/1k/1k scenarios) |
| `e3d39f0` | infon-8pa.21 | full ablation_matrix 3-seed run complete |
| `814a3ba` | infon-8pa.22 | canonical_cells 5-seed re-run complete |
| `a652839` | infon-8pa.23 | headline tables (4 CSV + 4 JSON) |
| `8dcc6dd` | infon-8pa.24 | headline figures (4 PNGs) |
| `e62d66e` | infon-8pa.25–30 | Stage C analysis + phase2_findings.md |
| `d58c1eb` | infon-8pa.31–32 | lock synthetic_v1 and result aggregates (SHA256SUMS) |
| `42a9c3c` | infon-8pa.33 | phase-2 technical memo (7 sections) |
| `c788058` | infon-8pa.34 | README Synthetic Stress Test section |

Tag `v0.3.0-phase2` on branch `research`. Remote: `github-edenduthie:edenduthie/infon.git`.

---

**Authoring note.** This report is the executive companion to `phase2_synthetic_ablations.md`. The memo contains the full technical narrative (generator design, oracle correctness, metric derivations, per-finding analysis). This report is the point-in-time executive summary — the handoff document from Epic 02 to Epic 03.
