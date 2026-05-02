# Spec: Epic 03 — Public benchmarks + baselines

## Hard Rules

- **TDD only:** integration test before implementation.
- **No mocks for benchmarks or baselines:** real benchmark data, real model checkpoints, real torch fits. The single permitted exception is the LLM **record-and-replay cache**: production runs record real API calls; tests replay from on-disk fixtures keyed by `(model_name, prompt_sha256)`. The replay layer is not a mock — it is a deterministic transport over the same data.
- **Reproduce-before-report:** every baseline must reproduce its published number on its native benchmark to within ±2 pp (±3 pp for LLM Sufficient-Context) before any "ours vs baseline" number is reported in the memo or paper.
- **Stage-boundary review:** at each stage's end, run `pytest -v` and re-read this spec.

## ADDED Requirements

### Requirement: HoVer Loader

`reference_v2/benchmarks/hover.py` SHALL provide `HoVerLoader(split: Literal["train","dev"]) → list[HoVerClaim]`. Each `HoVerClaim` SHALL preserve: claim text, label `∈ {SUPPORTED, NOT_SUPPORTED}`, supporting facts (Wikipedia article + sentence ids), and `num_hops ∈ {2, 3, 4}`. The loader SHALL cache to `experiments/data/hover/` and SHALL not re-download if the cache is fresh.

#### Scenario: Dev count matches published
- **WHEN** the dev split is loaded
- **THEN** the count matches the HoVer paper's published dev-set count

**Testability:** count assertion against the published dataset card; structural assertions on the first claim.

---

### Requirement: AVeriTeC v2 Loader

`reference_v2/benchmarks/averitec.py` SHALL provide `AVeriTeCLoader(split, version="v2")`. Each claim SHALL preserve: claim text, verdict `∈ {SUPPORTED, REFUTED, NOT_ENOUGH_EVIDENCE, CONFLICTING_EVIDENCE}`, evidence questions and answers, and any retrieval-coverage metadata. The loader SHALL also surface a path to the precompiled knowledge store distributed with AVeriTeC v2.

#### Scenario: NEI labels surfaced
- **WHEN** the dev split is loaded
- **THEN** at least one claim has the `NOT_ENOUGH_EVIDENCE` verdict and the loader surfaces it as such

**Testability:** structural assertion on the loaded data.

---

### Requirement: SciFact Loader

`reference_v2/benchmarks/scifact.py` SHALL provide `SciFactLoader(split)` returning per-claim records with `claim`, `label ∈ {SUPPORTS, REFUTES, NEI}`, `cited_doc_ids`, and `rationales`.

#### Scenario: Rationales preserved
- **WHEN** a claim with rationales is loaded
- **THEN** the rationales are non-empty and aligned to specific abstracts

**Testability:** structural assertion.

---

### Requirement: Uniform Baseline Interface

Every baseline module under `reference_v2/baselines/` SHALL expose:

```python
class Baseline(Protocol):
    name: str
    def fit(self, train_corpus: Optional[Corpus], train_labels: Optional[list[Verdict]]) -> None: ...
    def predict(self, claim: Claim, corpus: Corpus) -> MassFunction: ...
```

`MassFunction` is the four-element belief mass `[m(S), m(R), m(U), m(Θ)]` already used by the cognition package. Baselines that natively produce 3-way softmax (NLI, R-GCN control) SHALL set `m(U) = m(Θ) = 0`. Baselines that produce evidential Dirichlet output SHALL map to the 4-element mass via the standard subjective-logic bijection. LLM baselines SHALL parse verbalized confidence and abstention into a 4-element mass with documented mapping rules.

#### Scenario: All baselines satisfy the interface
- **WHEN** every baseline class is instantiated
- **THEN** `predict()` returns a four-element non-negative vector summing to 1 within `1e-6`

**Testability:** parametric test over all baseline classes.

---

### Requirement: LLM Record-and-Replay Cache

`reference_v2/baselines/_llm_cache.py` SHALL provide a deterministic on-disk cache. Cache key: `sha256(model_name + "\x00" + prompt + "\x00" + system_prompt + "\x00" + temperature_str)`. On cache hit, the cached completion is returned without an API call. On cache miss in *replay-only* mode, the cache raises `LLMCacheMissError`. Tests SHALL run in replay-only mode against committed fixtures.

#### Scenario: Replay returns cached completion
- **GIVEN** a fixture cache entry for a known prompt
- **WHEN** the LLM baseline calls `chat()` with the same prompt
- **THEN** the cached completion is returned and no network I/O occurs

#### Scenario: Cache miss in replay mode raises
- **WHEN** a prompt not present in the cache is requested in replay-only mode
- **THEN** `LLMCacheMissError` is raised with a message including the missing key

**Testability:** real fixture cache, real cache logic, no network — but no MagicMock either.

---

### Requirement: Baseline Reproduction Gate

For each baseline, the phase-3 memo SHALL document a reproduction check: a published-number-on-its-native-benchmark, our-rerun-of-the-same-cell, and the delta. The deltas SHALL be ≤ 2 pp for non-LLM baselines and ≤ 3 pp for LLM Sufficient-Context. The reproduction-gate result SHALL be a table at the top of the memo's Results section.

#### Scenario: Gate enforced
- **WHEN** Stage B.2 runs
- **THEN** every baseline has a reproduced-number row with a delta within tolerance, OR the epic blocks at Stage B until the failing baseline is fixed

**Testability:** the reproduction check itself is a runnable script (`experiments/check_baseline_reproductions.py`) producing the table.

---

### Requirement: Evaluation Harness

`reference_v2/experiments/benchmark_eval.py` SHALL drive the Cartesian over `(dataset × baseline × seed)` with checkpointing (a partial run SHALL resume without recomputation), a per-cell timeout, an LLM API budget guard, and per-cell JSON output.

#### Scenario: Resumability
- **GIVEN** a run is killed mid-cell
- **WHEN** the runner is invoked again with the same config
- **THEN** completed cells are skipped and only the killed and unstarted cells run

**Testability:** integration test runs a tiny matrix, kills mid-stream, resumes, verifies no recomputation.

---

### Requirement: H1 Panel (HoVer Stratified)

For HoVer, the phase-3 memo SHALL report a table of label accuracy stratified by `num_hops ∈ {2, 3, 4}`, with one row per model `∈ {ours, rgcn_control, llm_zeroshot, llm_sufficient_context}` and per-cell paired-bootstrap 95% CI.

The H1 effect size SHALL be defined as `(ours.acc_at_4hop − rgcn_control.acc_at_4hop)` with its bootstrap CI; the value SHALL be reported regardless of sign.

#### Scenario: H1 panel produced
- **WHEN** Stage B.5 completes
- **THEN** the H1 panel table is committed under `experiments/results/h1_panel.json`

**Testability:** existence + schema check.

---

### Requirement: H2 Panel (AVeriTeC + SciFact)

For AVeriTeC and SciFact, the phase-3 memo SHALL report:

1. ρ(`m(Θ)`, `proxy`) for each of three proxies: `nei_label_indicator`, `retrieval_coverage`, `supporting_evidence_count`. The primary proxy SHALL be pre-registered in the memo (recommendation: `nei_label_indicator` for AVeriTeC; `supporting_evidence_count` for SciFact).
2. AURC across all six models with paired-bootstrap CIs.
3. Selective accuracy at coverage `∈ {0.5, 0.7, 0.9}` for all six models.
4. Reliability diagrams (15-bin) for all six models on at least one dataset.

#### Scenario: H2 panel produced
- **WHEN** Stage B.6 completes
- **THEN** the H2 panel JSON exists under `experiments/results/h2_panel.json`

**Testability:** existence + schema + monotonicity check (selective accuracy SHALL be non-decreasing as coverage decreases for any sane model).

---

### Requirement: Pareto Front Headline Figure

`experiments/figures/pareto_front.py` SHALL produce a single PDF figure: scatter of `mean_accuracy` vs `aurc` over all `(dataset × model)` cells, with the Pareto-optimal subset connected by a polyline. The figure SHALL be the headline figure for both the phase-3 memo and Epic 04's paper.

#### Scenario: Figure produced
- **WHEN** Stage B.4 completes
- **THEN** `experiments/figures/pareto_front.pdf` exists; its source script regenerates it deterministically from the aggregate JSON

**Testability:** existence; deterministic-rebuild check.

---

### Requirement: Phase-3 Technical Memo

`docs/publication/phase3_benchmarks_and_baselines.md` SHALL be authored before this epic closes and SHALL contain: (1) loader provenance and dataset statistics; (2) baseline reproduction-gate table; (3) per-dataset full results tables; (4) Pareto-front figure with commentary; (5) H1 panel; (6) H2 panel; (7) findings, including null/negative; (8) explicit claim envelope for Epic 04. The memo SHALL be linked from `reference_v2/README.md`.

**Testability:** existence + link check.
