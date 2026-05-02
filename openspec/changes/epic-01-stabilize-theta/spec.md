# Spec: Epic 01 — Stabilize Θ

## Hard Rules

- **TDD only:** integration test before implementation; red → green; no shortcuts.
- **No mocks, no stubs, no test doubles:** real torch tensors, real seeded RNGs, real mass functions, real reasoners. No `MagicMock`, `patch`, or fake DS algebra.
- **No isolated unit tests:** every test exercises at least one full call through `HypergraphReasoner` or the fusion pipeline.
- **Stage-boundary review:** at the end of each stage in `tasks.md`, re-read this spec, run the regression, and verify compliance.

## ADDED Requirements

### Requirement: Deterministic Reproduction

`HypergraphReasoner.fit()` SHALL accept an integer `seed` argument that pins all relevant random number generators (`torch`, `numpy`, `random`, CUDA where applicable). When `seed` is set, two invocations of `fit()` with the same data, same configuration, and same seed SHALL produce bit-identical loss traces and bit-identical post-training mass functions.

The experiment runner under `reference_v2/experiments/` SHALL refuse to start a sweep without an explicit `seed` (or `seeds:` list) in the configuration; an unseeded run is a configuration error.

#### Scenario: Same seed reproduces
- **GIVEN** a `HypergraphReasoner` configured identically twice
- **WHEN** `fit(seed=42)` is called on each
- **THEN** the loss-by-epoch traces are bit-identical and the diagnostic-query masses match to floating-point equality

#### Scenario: Missing seed errors
- **WHEN** `experiments/run.py` is invoked with a config lacking a `seed` field
- **THEN** the runner exits with a configuration-error message naming the missing field

**Testability:** see `tasks.md` A.2; integration test fits a real reasoner twice on the EV scenario and asserts byte-equal traces.

---

### Requirement: Per-Infon Mass Logging

The reasoner SHALL provide an opt-in diagnostic path that records, for each diagnostic query, the per-infon mass functions produced by `reason()` *before* `combine_multiple` is applied. The recorded structure SHALL contain at minimum: `query_id`, `infon_id`, the four-element mass vector `[m(S), m(R), m(U), m(Θ)]`, and the relevance score used to select that infon.

#### Scenario: Diagnostic record populated
- **GIVEN** a fitted reasoner with `Config.log_per_infon_masses=True`
- **WHEN** `reason(query)` is called on the Toyota query
- **THEN** the returned record contains the pre-fusion masses for every per-infon contributor, each summing to 1 within `1e-6`

**Testability:** real reasoner, real query, assert structural and numerical invariants on the record.

---

### Requirement: Alternative Fusion Rules

`reference_v2/src/cognition/dempster_shafer.py` SHALL provide four fusion rules accessible through `combine_multiple(masses, rule=...)`:

1. `dempster` — current behaviour, conflict mass renormalized
2. `yager` — conflict mass assigned to Θ instead of being renormalized away (Yager 1987)
3. `murphy` — average of pairwise Dempster combinations (Murphy 2000)
4. `top1` — return the single most-decisive focal mass, no fusion performed

Each rule SHALL produce a valid mass function (entries non-negative, sum to 1 within `1e-6`) on any input set of valid mass functions, including pathological inputs (all-conflict; all-agreeing; single mass).

#### Scenario: Yager preserves more Θ than Dempster
- **GIVEN** three mass functions, each placing 0.95 on SUPPORTS and 0.05 on Θ
- **WHEN** combined with `rule="yager"` and with `rule="dempster"`
- **THEN** the Yager result has strictly larger `m(Θ)` than the Dempster result

#### Scenario: Top-1 is non-degenerate on a single input
- **GIVEN** a single high-confidence mass on REFUTES
- **WHEN** combined with `rule="top1"`
- **THEN** the result equals the input within floating-point equality

**Testability:** see `tasks.md` A.4; constructive masses, real DS algebra.

---

### Requirement: Configurable Fusion Cap

`Config.decisive_top_k` SHALL default to `3` and SHALL be honoured by `HypergraphReasoner.reason()` as the maximum number of per-infon mass functions passed to `combine_multiple`. Setting `decisive_top_k=1` SHALL produce identical behaviour to `rule="top1"` for any fusion rule.

#### Scenario: Lower top-k yields larger Θ on agreeing evidence
- **GIVEN** a query with five high-confidence agreeing supporting infons
- **WHEN** `reason()` is called with `decisive_top_k=1` vs `decisive_top_k=5`
- **THEN** `m(Θ)` at `top_k=1` is strictly larger than at `top_k=5`

**Testability:** see `tasks.md` A.5; real reasoner; constructive query.

---

### Requirement: Configuration Sweeps

The experiment runner SHALL support a YAML `sweep:` block listing axes (e.g. `coherence_weight: [0, 0.2, 0.5]`, `fusion_rule: [dempster, yager]`, `decisive_top_k: [1, 3, 5]`, `seed: [42, 0, 1, 7, 13]`) and SHALL execute the full Cartesian product, writing one JSON report per cell to `experiments/results/<sweep_name>/<cell_id>.json`. The runner SHALL aggregate the per-cell reports into a single `aggregate.json` with one row per `(coherence_weight, fusion_rule, decisive_top_k)` combination summarized over seeds (mean, std, min, max for each metric).

#### Scenario: Sweep covers full grid
- **GIVEN** a sweep config with 6×4×4×5 axes
- **WHEN** `experiments/sweep.py --config sweep_collapse.yaml` is run to completion
- **THEN** exactly 480 per-cell JSONs and one `aggregate.json` exist; `aggregate.json` has 96 rows

**Testability:** integration test on a tiny sweep (2×2×2×2 = 16 cells) verifies file counts and aggregate row count.

---

### Requirement: Canonical Configuration

After Stage C of `tasks.md`, exactly one named configuration `experiments/configs/canonical_v0_2.yaml` SHALL exist and SHALL satisfy the acceptance criterion in `proposal.md`. The file SHALL include version metadata, the random-seed list, a one-line rationale referencing the phase-1 memo, and explicit `coherence_weight`, `fusion_rule`, `decisive_top_k`, and `activation_threshold` settings.

This configuration SHALL be the only one referenced by Epics 02–04. Any subsequent paper figure or table that depends on Epic-01 outputs SHALL cite this configuration by file path.

#### Scenario: Canonical run reproduces acceptance numbers
- **GIVEN** the canonical configuration
- **WHEN** the runner is invoked across seeds {42, 0, 1, 7, 13}
- **THEN** for both the Toyota and Honda queries, polarity matches corpus ground truth, and `m(Θ)` is in `[0.20, 0.40]` with across-seed standard deviation `≤ 0.05`

**Testability:** Stage D.2 commits a JSON report containing exactly these numbers; a CI test asserts the report file exists and the numbers parse to within tolerance.

---

### Requirement: Phase-1 Technical Memo

`docs/publication/phase1_collapse_fix.md` SHALL be authored before this epic closes and SHALL contain: (i) restatement of the audit finding; (ii) location of the collapse (training vs fusion vs both); (iii) sweep grid and a summary figure; (iv) the chosen canonical configuration; (v) reproduction instructions; (vi) the acceptance-criterion table with values; (vii) explicit follow-ups for Epic 02. The memo SHALL be linked from `reference_v2/README.md`.

#### Scenario: Memo present and linked
- **WHEN** the epic is marked complete
- **THEN** `docs/publication/phase1_collapse_fix.md` exists and is reachable from `reference_v2/README.md`

**Testability:** existence + link check in CI.
