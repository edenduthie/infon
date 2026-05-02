# Spec: Epic 02 — Synthetic stress dataset + ablation matrix

## Hard Rules

- **TDD only:** integration test before implementation.
- **No mocks:** real generator, real reasoner runs, real metric computations on real numpy arrays.
- **Determinism is a correctness property:** every generator call is seeded; identical seed ⇒ identical output.
- **Stage-boundary review:** at the end of each stage in `tasks.md`, re-read this spec and run `pytest -v`.

## ADDED Requirements

### Requirement: Parametric Synthetic Generator

`reference_v2/synthetic/` SHALL provide a deterministic `Generator` class accepting an integer `seed` and producing `Scenario` objects via a `generate()` method. The generator's knobs SHALL include:

- `n_docs: int` — number of documents per scenario (default 5)
- `n_anchors_per_type: int` — anchors per type (default 5)
- `evidence_redundancy: int ∈ {1, 2, 5, 10}` — number of supporting sentences witnessing each claim
- `compositional_depth: int ∈ {1, 2, 3, 4}` — IKL chain length for the scenario's primary query
- `contradiction_density: float ∈ [0, 1]` — fraction of premises with a contradicting witness
- `nei_fraction: float ∈ [0, 1]` — fraction of queries with no supporting evidence in the corpus

Each `Scenario` SHALL carry: `corpus` (list of synthetic documents), `query_set` (list of queries), and `oracle_labels` (per query: `planted_verdict ∈ {SUPPORTS, REFUTES, NEI}`, `planted_thinness: int` = exact supporting-sentence count, `planted_hop_count: int`).

#### Scenario: Determinism
- **GIVEN** `Generator(seed=42)` constructed twice
- **WHEN** `generate()` is called with identical knob values
- **THEN** the two outputs are byte-identical when serialized

#### Scenario: Oracle correctness
- **GIVEN** a generated `Scenario` with `evidence_redundancy=3`
- **WHEN** the corpus is grep'd for the witnessing pattern of a query
- **THEN** the count equals the query's `planted_thinness`

**Testability:** integration tests in `tests/test_synthetic_generator.py` instantiate the real generator and check oracle invariants.

---

### Requirement: Stratified Splits

`reference_v2/synthetic/splits.py` SHALL provide `make_splits(seed, train, dev, test)` returning three disjoint scenario sets. The `test` split SHALL be a *thinness curriculum*: stratified across `evidence_redundancy ∈ {1, 2, 5, 10}` with at least 100 scenarios per stratum at the documented default sizes.

#### Scenario: Disjoint splits
- **WHEN** `make_splits(seed=42, train=8000, dev=1000, test=1000)` is called
- **THEN** the three returned sets have empty pairwise intersection on scenario ID

#### Scenario: Curriculum stratification
- **WHEN** the test split is grouped by `evidence_redundancy`
- **THEN** each group has ≥ 100 scenarios

**Testability:** real generator output, real grouping; no mocks.

---

### Requirement: Aggregator Ablation

`Config.aggregator: Literal["typed_ikl", "uniform_mean"]` SHALL be honoured by the network. When `uniform_mean` is selected, message passing uses a single `W_self` and a uniform-mean over neighbour embeddings (no per-relation weights, no IKL operators), with parameter count within 5% of the typed-IKL configuration at the same hidden size.

#### Scenario: Parameter parity
- **GIVEN** two `HypergraphReasoner` instances at the same hidden size, one per aggregator setting
- **THEN** their parameter counts differ by ≤ 5%

**Testability:** count parameters via `sum(p.numel() for p in model.parameters())` on real instances.

---

### Requirement: Readout Ablation

`Config.readout: Literal["ds_4mass", "softmax_temperature", "dirichlet_edl"]` SHALL select among:
- `ds_4mass` — current 4-element mass head
- `softmax_temperature` — 3-way softmax with post-hoc temperature scaling fit on a held-out 10% slice
- `dirichlet_edl` — evidential Dirichlet head per Sensoy et al. 2018

Each readout SHALL produce, on every query, a four-element confidence vector `[m(S), m(R), m(U), m(Θ)]` for downstream metric computation. For softmax, `m(U)` and `m(Θ)` are zero by construction; for Dirichlet, `m(Θ)` is the standard subjective-logic uncertainty mass.

#### Scenario: Readout produces 4-vector for each
- **GIVEN** a fitted reasoner under each readout
- **WHEN** `reason()` is called on any query
- **THEN** the result has a four-element mass with documented semantics

**Testability:** real fits, real outputs, structural assertions.

---

### Requirement: Teacher Leave-One-Out

`Config.teacher_sources: list[str]` SHALL be a subset of `{polarity, alignment, distance, confidence}`. The teacher mass is the Dempster combination of only the listed sources. The default is all four (current behaviour).

#### Scenario: LOO produces distinguishable training trajectories
- **GIVEN** four configurations each holding out one teacher source
- **THEN** their epoch-0 teacher mass distributions differ by KL ≥ ε across some queries

**Testability:** real teacher construction; KL is a real number computed from real masses.

---

### Requirement: Metric Battery

`reference_v2/experiments/metrics.py` SHALL implement, with no scikit-learn dependency:

- `polarity_accuracy_3way(pred, true)` — 3-way SUPPORTS/REFUTES/NEI accuracy where Θ-mass is collapsed into NEI by argmax over a thresholded decision.
- `ece(probs, labels, n_bins=15)` — Expected Calibration Error.
- `brier_3way(probs, labels)` — Brier score.
- `aurc(risks, coverages)` — area under the risk-coverage curve.
- `selective_accuracy_at_coverage(scores, preds, labels, coverage)` — selective accuracy at a fixed coverage level.
- `spearman_rho(x, y)` — Spearman rank correlation.

All metrics SHALL be deterministic, vectorised over numpy arrays, and have docstring references to a published source.

#### Scenario: Metrics match hand-computed values
- **GIVEN** small synthetic predictions whose metrics are computed by hand
- **THEN** each metric implementation matches to `1e-6`

**Testability:** `tests/test_metrics.py` enumerates the hand-computed cases.

---

### Requirement: Paired Bootstrap CI

`reference_v2/experiments/stats.py::paired_bootstrap_ci(differences, n_resamples=10_000, ci=0.95)` SHALL return `(low, high)` from a paired bootstrap over per-scenario differences. Coverage SHALL match the requested level on simulated data with known mean to within 2 percentage points over 1000 trials.

#### Scenario: Coverage holds
- **GIVEN** a simulator producing 1000 trials of differences with known mean
- **WHEN** paired bootstrap CIs are computed at 95%
- **THEN** the bracketing rate is in `[0.93, 0.97]`

**Testability:** integration test runs the simulator with a fixed seed.

---

### Requirement: Eight Named Cells

`reference_v2/experiments/configs/ablation_matrix.yaml` SHALL define exactly eight named cells: `canonical`, `uniform_aggregator`, `softmax_readout`, `dirichlet_readout`, `coherence_off`, `teacher_only`, `top1_fusion`, `single_layer`. Each cell SHALL be a complete config (not a diff) so the run is unambiguous in isolation. The phase-2 memo SHALL report results for these eight cells, with the `canonical` cell defined by reference to Epic 01's `canonical_v0_2.yaml`.

#### Scenario: Cells are complete
- **WHEN** any of the eight cells is loaded by the experiment runner
- **THEN** the runner does not have to merge against any other config to start a run

**Testability:** integration test instantiates each cell config and starts (and immediately cancels) a fit.

---

### Requirement: Aggregate Output Schema

`experiments/results/<run_name>/aggregate.json` SHALL be a list of rows, each row a JSON object with keys:
`{cell, n_seeds, polarity_acc_mean, polarity_acc_std, polarity_acc_ci_low, polarity_acc_ci_high, ece, brier, aurc, sel_acc_at_50, sel_acc_at_70, sel_acc_at_90, spearman_thinness, spearman_thinness_ci_low, spearman_thinness_ci_high}` plus the cell's full configuration nested under `config:`.

#### Scenario: Schema validates
- **WHEN** `aggregate.py` is run on a directory of per-cell JSONs
- **THEN** the output passes a JSON schema validator for the documented fields

**Testability:** synthetic per-cell JSONs feed the aggregator; output is validated.

---

### Requirement: Phase-2 Technical Memo

`docs/publication/phase2_synthetic_ablations.md` SHALL be authored before this epic closes and SHALL contain: (1) generator design and oracle invariants; (2) splits and curriculum; (3) full ablation matrix table; (4) four headline tables (H1 accuracy by hop, H2 ρ table, AURC, ECE/Brier); (5) four headline figures; (6) findings including null/negative; (7) explicit follow-ups for Epic 03 (e.g. "AVeriTeC must reproduce ρ ≥ X to support H2 on real data"). The memo SHALL be linked from `reference_v2/README.md`.

**Testability:** existence + link check in CI; the memo is content, not code, but it is a deliverable.
