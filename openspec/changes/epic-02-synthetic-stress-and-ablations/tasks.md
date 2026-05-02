# Tasks: Epic 02 — Synthetic stress dataset + full ablation matrix

> **Epic status (created 2026-05-02):** blocked by Epic 01. Acceptance criterion in `proposal.md`. Blocks Epic 03.

## Development Rules

- **Test-First:** write the integration test first, fail-then-pass.
- **No Mocks:** real generator output, real reasoner runs, real metric computations. No `MagicMock`.
- **Seed everything:** any test that draws from the generator pins `seed=42` and asserts byte-equal output.
- **Stage-Boundary Review:** at the end of each stage, run `pytest reference_v2/tests/ -v`; verify spec compliance; update beads.

---

## Stage A — Code (build the study harness)

### Synthetic generator

- [ ] A.1 Write `reference_v2/tests/test_synthetic_generator.py` first: instantiate `Generator(seed=42)`; call `generate(n_docs=10, evidence_redundancy=2, compositional_depth=2, contradiction_density=0.0, nei_fraction=0.0)`; assert deterministic output across two calls; assert oracle labels are populated; assert every query's `planted_thinness` matches the actual count of supporting sentences in the produced corpus. **Verify red.**
- [ ] A.1b Implement `reference_v2/synthetic/generator.py` with: template registry, anchor sampler, premise-chain construction, oracle labelling. Implement `Scenario` dataclass `(corpus, query_set, oracle_labels, generator_config)`. **Verify green.**
- [ ] A.2 Write `reference_v2/tests/test_synthetic_splits.py` first: call `make_splits(seed=42, train=8000, dev=1000, test=1000)`; assert disjoint scenario IDs; assert thinness-curriculum stratification on the test split (≥ 100 scenarios at each `evidence_redundancy ∈ {1, 2, 5, 10}`). **Verify red.**
- [ ] A.2b Implement `reference_v2/synthetic/splits.py`. **Verify green.**

### Ablation harness

- [ ] A.3 Write `reference_v2/tests/test_aggregator_swap.py` first: configure a reasoner with `aggregator="uniform_mean"`, fit on a tiny scenario, assert it runs without crashing and that the layer-2 norm-Δ behaviour differs from typed-IKL. **Verify red.**
- [ ] A.3b Add `Config.aggregator: Literal["typed_ikl", "uniform_mean"]` and a `UniformMeanLayer` in `reference_v2/src/cognition/logic.py` parameter-matched to `TypedMessagePassingLayer` (within 5% parameter count). **Verify green.**
- [ ] A.4 Write `reference_v2/tests/test_readout_swap.py` first: configure with `readout ∈ {ds_4mass, softmax_temperature, dirichlet_edl}`; fit; assert each readout produces a valid output and reasonable training curves. **Verify red.**
- [ ] A.4b Add `Config.readout` and implement `SoftmaxTemperatureReadout` (post-hoc Platt/temperature scaling on a held-out 10% slice) and `DirichletEDLReadout` (Sensoy 2018) in `reference_v2/src/cognition/heads.py`. **Verify green.**
- [ ] A.5 Write `reference_v2/tests/test_teacher_loo.py` first: fit with each of the four teacher sources held out individually; assert the resulting four loss trajectories are distinguishable. **Verify red.**
- [ ] A.5b Add `Config.teacher_sources: list[str]` controlling which of `{polarity, alignment, distance, confidence}` contribute. **Verify green.**

### Metric battery

- [ ] A.6 Write `reference_v2/tests/test_metrics.py` first: construct synthetic predictions with known properties; assert `polarity_accuracy_3way`, `ece(n_bins=15)`, `brier_3way`, `aurc(risk, coverage)`, `selective_accuracy_at_coverage(0.7)`, and `spearman_rho` match hand-computed values to 1e-6. **Verify red.**
- [ ] A.6b Implement `reference_v2/experiments/metrics.py` with all six metrics, references in docstrings, no scikit-learn dep (re-implement from numpy where possible). **Verify green.**
- [ ] A.7 Write `reference_v2/tests/test_paired_bootstrap.py` first: two known mean differences, assert paired-bootstrap 95% CI brackets the true difference at the documented coverage rate over 1000 simulated runs. **Verify red.**
- [ ] A.7b Implement `reference_v2/experiments/stats.py::paired_bootstrap_ci`. **Verify green.**

### Aggregation

- [ ] A.8 Write `reference_v2/tests/test_ablation_aggregate.py` first: feed 12 fake per-cell JSONs into `aggregate_runs()`; assert resulting `aggregate.json` has the documented schema with mean, std, n, ci_low, ci_high. **Verify red.**
- [ ] A.8b Implement `reference_v2/experiments/aggregate.py` and a CLI entrypoint. **Verify green.**

- [ ] **STAGE-A REVIEW:** run `pytest reference_v2/tests/ -v`; ensure no Epic 01 regression; commit checkpoint `epic-02-stage-a-complete`.

## Stage B — Run (execute and gather results)

- [ ] B.1 **Pilot**: run the generator with `n_scenarios=100, seeds=[42]`; sanity-check that planted thinness, hop count, and verdict labels look right by spot-checking 10 scenarios manually. Record findings in `experiments/results/synthetic_pilot/notes.md`.
- [ ] B.2 **Pilot ablation**: run the eight named cells (per `design.md`) on the 100-scenario pilot; verify the harness completes and metrics computed. Wall-clock estimate target ≤ 30 min.
- [ ] B.3 **Full generation**: run `python -m reference_v2.synthetic.generate --train 8000 --dev 1000 --test 1000 --seed 42 --out experiments/data/synthetic_v1/`. Wall-clock ≤ 60 min. Persist outputs to disk so re-runs are free.
- [ ] B.4 **Full ablation matrix**: run `python -m reference_v2.experiments.ablation_matrix --config experiments/configs/ablation_matrix.yaml --seeds 42,0,1 --data experiments/data/synthetic_v1 --out experiments/results/ablation_matrix/`. Wall-clock estimate: ~2-4 days CPU; run in background with checkpointing so partial completion is recoverable.
- [ ] B.5 **Canonical cells re-run with 5 seeds**: rerun the eight named cells with `--seeds 42,0,1,7,13` on the test split; record to `experiments/results/canonical_cells/`.
- [ ] B.6 Generate aggregate JSON and the four headline tables: H1 (accuracy by hop), H2 ρ table, AURC table, ECE/Brier table. Save to `experiments/results/figures/`.
- [ ] B.7 Generate four headline figures: (1) accuracy vs hop_count by aggregator; (2) `m(Θ)` vs planted thinness scatter (canonical config); (3) reliability diagrams for three readouts; (4) risk-coverage curves for three readouts.

## Stage C — Review and iterate on the unexpected

- [ ] C.1 Inspect each of the four headline tables. For any cell where the result is *opposite* to the H1/H2 prediction, do *not* tune it away — investigate root cause; add explanation to phase-2 memo as a finding.
- [ ] C.2 Check stratified outliers: are there `(hop_count=k, evidence_redundancy=r)` cells where ours fails badly? If so, file a beads ticket; do not silently exclude.
- [ ] C.3 Run a sanity check: ρ(`m(Θ)`, `planted_thinness`) on the *training* split should be larger than on the *test-thinness* split (because the model saw the training thinness). If not, the generator may be leaking information; debug.
- [ ] C.4 Validate paired-bootstrap CIs by reproducing one cell's CI with `seeds=42,0,1,7,13,21,33` and confirming overlap.
- [ ] C.5 If H2 metrics on synthetic are weak even after Epic 01's canonical config, re-examine whether the canonical config holds on synthetic (it was tuned on a 49-infon scenario — synthetic may need a re-tune). If a re-tune is needed, document this and update Epic 01's canonical config recommendation; note the dependency in the phase-2 memo.
- [ ] C.6 If H1 (typed-IKL > uniform-mean at hop=4) does not hold, file as a finding; reframe paper claim accordingly in Epic 04.

## Stage D — Finalize and produce phase-2 deliverable

- [ ] D.1 Lock the synthetic dataset in `experiments/data/synthetic_v1/` (treat as immutable; any change is `synthetic_v2`).
- [ ] D.2 Lock `experiments/results/ablation_matrix/aggregate.json` and `experiments/results/canonical_cells/aggregate.json`.
- [ ] D.3 Write `docs/publication/phase2_synthetic_ablations.md`: 1) generator design and oracle correctness; 2) splits and stratification; 3) ablation matrix table (full); 4) four headline tables; 5) four headline figures; 6) findings (including any null/negative); 7) follow-ups for Epic 03 (e.g. "verify on AVeriTeC: ρ ≥ X").
- [ ] D.4 Update `reference_v2/README.md` with a `## Synthetic Stress Test` section linking the memo and the generator CLI.
- [ ] D.5 Tag commit `v0.3.0-phase2`; push.
- [ ] **PHASE-BOUNDARY REVIEW Phase 2:** run `pytest reference_v2/tests/ -v`; verify acceptance criterion satisfied; verify phase-2 memo committed; update beads; mark `epic-02-synthetic-stress-and-ablations` complete; unblock `epic-03-benchmarks-and-baselines`.
