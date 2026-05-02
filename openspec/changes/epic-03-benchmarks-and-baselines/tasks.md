# Tasks: Epic 03 — Public benchmarks + baselines

> **Epic status (created 2026-05-02):** blocked by Epic 02. Acceptance criterion in `proposal.md`. Blocks Epic 04.

## Development Rules

- **Test-First:** write the integration test first, fail-then-pass.
- **No Mocks:** real benchmark loaders read real data; real baselines run real models. The only exception is the LLM caching layer — `record-and-replay` (we record real API calls once, then replay from disk on subsequent runs); the *replay* path is exercised by tests, the *record* path is exercised by the first live run.
- **Reproduce baselines before reporting**: every baseline must reproduce its published number on its native benchmark to within ±2 pp before we report any "ours vs that baseline" comparison.
- **Stage-Boundary Review:** at the end of each stage, run `pytest reference_v2/tests/ -v`; verify spec compliance; update beads.

---

## Stage A — Code (build the harness)

### Benchmark loaders

- [ ] A.1 Write `reference_v2/tests/benchmarks/test_hover.py` first: download the HoVer dev split (caching to `experiments/data/hover/`); load with `HoVerLoader`; assert dev-set claim count matches the published number; verify hop-count annotation is preserved per claim. **Verify red.**
- [ ] A.1b Implement `reference_v2/benchmarks/hover.py::HoVerLoader` reading the HoVer dev JSONL. **Verify green.**
- [ ] A.2 Write `reference_v2/tests/benchmarks/test_averitec.py` first: download AVeriTeC v2 dev (per the published shared-task data link); assert claim count matches; verify NEI labels present; verify the precompiled knowledge store is downloadable to `experiments/data/averitec/store/`. **Verify red.**
- [ ] A.2b Implement `reference_v2/benchmarks/averitec.py::AVeriTeCLoader`. **Verify green.**
- [ ] A.3 Write `reference_v2/tests/benchmarks/test_scifact.py` first: download SciFact dev; assert count; verify rationales preserved. **Verify red.**
- [ ] A.3b Implement `reference_v2/benchmarks/scifact.py::SciFactLoader`. **Verify green.**

### Baseline 1 — Symbolic floor

- [ ] A.4 Write `reference_v2/tests/baselines/test_symbolic_floor.py` first: build a synthetic case where retrieval returns nothing; assert the baseline emits NEI; build a case with one supporting passage matching surface negation; assert REFUTES. **Verify red.**
- [ ] A.4b Implement `reference_v2/baselines/symbolic_floor.py` using the existing SPLADE retrieval + a polarity-vote rule + a NEI-on-thin-retrieval threshold. **Verify green.**

### Baseline 2 — RoBERTa-NLI + temperature scaling

- [ ] A.5 Write `reference_v2/tests/baselines/test_nli_baseline.py` first: load a small NLI-pretrained checkpoint; predict on a known SUPPORTS/REFUTES/NEUTRAL claim; verify three-way logits; fit temperature on a tiny dev slice and assert the post-scaling ECE is no worse than pre-scaling on the slice. **Verify red.**
- [ ] A.5b Implement `reference_v2/baselines/nli.py` wrapping a published NLI checkpoint with a temperature-scaling fit step. **Verify green.**

### Baseline 3 — Architectural control (R-GCN + softmax)

- [ ] A.6 Write `reference_v2/tests/baselines/test_rgcn_control.py` first: build a tiny corpus; instantiate the control with `aggregator=uniform_mean, readout=softmax_temperature`; fit; assert parameter count within 5% of the canonical config. **Verify red.**
- [ ] A.6b Implement `reference_v2/baselines/rgcn_control.py` reusing Epic 02's aggregator/readout swap. **Verify green.**

### Baseline 4 — Evidential Dirichlet (matched uncertainty)

- [ ] A.7 Write `reference_v2/tests/baselines/test_dirichlet_baseline.py` first: build a tiny corpus; instantiate with `aggregator=typed_ikl, readout=dirichlet_edl`; fit; assert that on a high-confidence agreeing claim Dirichlet-uncertainty is small, on a thin claim it is larger. **Verify red.**
- [ ] A.7b Implement `reference_v2/baselines/dirichlet.py` reusing Epic 02's `dirichlet_edl` readout. **Verify green.**

### Baseline 5 — LLM zero-shot ceiling

- [ ] A.8 Write `reference_v2/tests/baselines/test_llm_zeroshot.py` first using the *replay* cache: pre-populate a fixture cache with a known prompt → completion mapping; run the baseline against the fixture; assert the parsed verdict and verbalized confidence match the cached completion. **Verify red.**
- [ ] A.8b Implement `reference_v2/baselines/llm_zeroshot.py` with: prompt template (carrying the document set + claim + abstention instruction), verbalized-confidence parser, on-disk record-and-replay cache keyed on `(model_name, prompt_sha256)`. Use Anthropic SDK by default; fall back to OpenAI for `gpt-4o`. Pin model versions. **Verify green.**

### Baseline 6 — RAG + Sufficient-Context selective generation

- [ ] A.9 Write `reference_v2/tests/baselines/test_llm_sufficient_context.py` first using replay cache: assert the sufficient-context judge produces `{sufficient, insufficient, partial}` for a set of fixture cases; assert the abstention gate fires on `insufficient`. **Verify red.**
- [ ] A.9b Implement `reference_v2/baselines/llm_sufficient_context.py` per Joren et al. ICLR 2025: a sufficient-context classifier prompt, a self-rated confidence elicitation, and a thresholded selective-generation gate. **Verify green.**

### Evaluation harness

- [ ] A.10 Write `reference_v2/tests/test_benchmark_eval.py` first: run a tiny end-to-end with two baselines × one benchmark × 3 seeds; assert per-cell JSONs and aggregate match the documented schema. **Verify red.**
- [ ] A.10b Implement `reference_v2/experiments/benchmark_eval.py` with parallelism over cells, checkpointing, resumability. **Verify green.**

### Significance tests + Pareto plot

- [ ] A.11 Reuse Epic 02's `paired_bootstrap_ci`; add `pairwise_significance_table()` for a triangular table of pairwise p-values across models per dataset.
- [ ] A.12 Implement `experiments/figures/pareto_front.py` producing the headline scatter (accuracy vs AURC) with Pareto-front line.

- [ ] **STAGE-A REVIEW:** run `pytest reference_v2/tests/ -v`; ensure no Epic 01/02 regression; commit checkpoint `epic-03-stage-a-complete`.

## Stage B — Run

- [ ] B.1 **Smoke**: 50-claim subset of each dataset × all six baselines + ours. Wall-clock ≤ 1 hour. Verify baselines do not crash and produce sensible verdicts on hand-checked claims.
- [ ] B.2 **Baseline reproduction gate**: for each baseline, evaluate on the dataset on which its published number was reported, and confirm reproduction within ±2 pp (within ±3 pp for the LLM Sufficient-Context baseline because of API drift). If a baseline fails to reproduce, return to Stage A and fix; do not proceed.
- [ ] B.3 **Full evaluation**: 3 datasets × (4 trainable models × 3 seeds + 2 LLM models × 1 seed) = 14 model-runs per dataset, 42 total. Estimate wall-clock: 1–2 weeks elapsed but <100 hours CPU; LLM API spend ≤ USD 200 capped via budget guard.
- [ ] B.4 Aggregate and produce: per-dataset accuracy table; per-dataset AURC table; per-dataset reliability diagrams; Pareto-front figure.
- [ ] B.5 Specifically for HoVer: stratified accuracy by hop count `{2, 3, 4}`, ours vs `rgcn_control` vs `llm_zeroshot` vs `llm_sufficient_context`. This is the H1 panel.
- [ ] B.6 Specifically for AVeriTeC: ρ(`m(Θ)`, `nei_label_proxy`); ρ(`m(Θ)`, `retrieval_coverage`); selective-accuracy-vs-coverage curves. This is the H2 panel. Report alongside the leaderboard winner's 33.17 AVeriTeC score.

## Stage C — Review and iterate

- [ ] C.1 For every cell where ours dominates a baseline by an *implausibly* large margin (>15 pp accuracy or >0.20 AURC gap), audit the baseline implementation. The default explanation is a baseline bug, not a system advantage.
- [ ] C.2 For every cell where ours loses, *do not tune to win*: file the result. Note in memo and in Epic 04's paper.
- [ ] C.3 Cross-check ρ across the three thinness proxies on AVeriTeC. If they disagree wildly, pre-register the primary one and report all three; do not silently choose the favourable one.
- [ ] C.4 Sanity-check: does temperature scaling for the NLI baseline actually reduce ECE? If not, the temperature fit is buggy (common: fitting on a tiny held-out set).
- [ ] C.5 Run a robustness check on Sufficient-Context: vary the abstention threshold across {0.5, 0.7, 0.9} and verify the AURC ranking is stable.
- [ ] C.6 If the Pareto front shows ours dominated by Sufficient-Context on every metric, the paper's contribution is *not* "calibrated abstention from a graph approach." Reframe in Epic 04 to focus on H1 (compositional reasoning) + interpretability.

## Stage D — Finalize

- [ ] D.1 Lock cached LLM API responses in `experiments/cache/llm/`; commit the cache to a separate artifact (large) referenced by SHA from the repo. Document the recompute cost.
- [ ] D.2 Lock `experiments/results/benchmark_eval/aggregate.json`.
- [ ] D.3 Write `docs/publication/phase3_benchmarks_and_baselines.md`: 1) loader provenance + reproduction-gate results; 2) baseline implementations and reproduced numbers; 3) full results table per dataset; 4) Pareto-front figure with commentary; 5) H1 panel (HoVer stratified); 6) H2 panel (AVeriTeC + SciFact); 7) findings, including any null/negative; 8) explicit follow-ups for Epic 04 (i.e. the paper's claim envelope).
- [ ] D.4 Update `reference_v2/README.md` with `## Public Benchmark Reproduction` section.
- [ ] D.5 Tag commit `v0.4.0-phase3`; push.
- [ ] **PHASE-BOUNDARY REVIEW Phase 3:** run `pytest reference_v2/tests/ -v`; verify acceptance criterion satisfied; verify phase-3 memo committed; update beads; mark `epic-03-benchmarks-and-baselines` complete; unblock `epic-04-paper-and-arxiv`.
