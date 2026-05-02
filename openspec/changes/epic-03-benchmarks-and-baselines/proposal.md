# Epic 03 — Public benchmarks + baselines

> Phase 3 of 4. Blocked by Epic 02 (canonical synthetic results in hand). Unblocks Epic 04.

## Why

Synthetic results (Epic 02) establish *internal* validity: under controlled conditions, the typed-IKL aggregator and 4-mass DS readout do or do not produce the predicted effects. They do not establish *external* validity: reviewers will (correctly) ask whether the same effects appear on the public benchmarks the field actually publishes against, and whether ours sits anywhere on the Pareto frontier between LLM-class accuracy and calibrated abstention.

This epic puts the system on three public benchmarks chosen to test H1 and H2 directly, and pits it against a tiered set of baselines whose published numbers anchor the comparison:

- **HoVer** (Jiang et al., EMNLP-Findings 2020) — many-hop fact verification; primary H1 testbed because hop count is annotated.
- **AVeriTeC v2 dev** (Schlichtkrull et al., shared task 2024-2025; 2025 winning AVeriTeC score 33.17%) — real-world claim verification with NEI; primary H2 testbed.
- **SciFact** (Wadden et al., EMNLP 2020) — scientific claims with thin per-claim evidence; secondary H2 testbed.

Six baselines span the difficulty range from "is learning needed at all" to "GPT-class abstention":

1. **Symbolic floor** — SPLADE retrieval + rule-based polarity + abstain when retrieval coverage < threshold.
2. **NLI classifier** — RoBERTa-large pretrained on FEVER NLI, post-hoc temperature scaled.
3. **Architectural control** — same hypergraph and teacher, but uniform-mean R-GCN + softmax readout (the H1 control from Epic 02 ported to public data).
4. **Matched uncertainty competitor** — same network, evidential Dirichlet head (Sensoy 2018; E-NER 2023 lineage).
5. **LLM zero-shot ceiling** — Claude Sonnet 4.6 / GPT-4o-class with retrieval and verbalized confidence.
6. **RAG + Sufficient-Context selective generation** — Joren et al., ICLR 2025, the current published SOTA on calibrated abstention.

The story we want — and the story we will tell honestly even if the data does not support it — is: tier 5/6 wins on raw accuracy; ours sits on the Pareto front for the calibration/accuracy trade-off; tiers 1–4 are dominated by ours on at least one of {AURC, Spearman ρ, Brier, ECE}.

## What Changes

- Adds `reference_v2/benchmarks/` with deterministic, cached loaders for HoVer, AVeriTeC v2 dev, SciFact (and FEVEROUS as optional/stretch).
- Adds `reference_v2/baselines/` with one module per baseline; each module exposes a uniform `predict(corpus, query) → MassFunction` interface.
- Adds `reference_v2/experiments/benchmark_eval.py` driving the Cartesian product of (dataset × model × seed) and producing per-cell JSON reports.
- Adds significance testing (paired bootstrap on per-claim differences) and a Pareto-front visualisation script.
- Adds a documented LLM-API budget cap and offline-first design (cache LLM calls so reproduction doesn't re-spend).
- Produces a phase-3 technical memo at `docs/publication/phase3_benchmarks_and_baselines.md`.

## Phased Scope (within this epic)

- **Stage A — Code**: dataset loaders, six baselines, evaluation harness, significance tests, Pareto plot.
- **Stage B — Run**: per-baseline smoke test on 50 claims; full evaluation across 3 datasets × 6 baselines × 3 seeds (where applicable; LLM baselines are deterministic via temperature 0).
- **Stage C — Review and iterate**: validate that loaders match published statistics; investigate any cell where ours dominates by an implausible margin (likely a bug in the baseline, not a feature in ours); iterate baselines until their numbers reproduce published baselines on the same dataset to within reported tolerances.
- **Stage D — Finalize**: lock results JSON; commit canonical headline tables and Pareto-front figure; write phase-3 memo.

## Acceptance Criterion (gates Epic 04)

1. All six baselines run on all three datasets to completion under 3 seeds (LLM baselines: 1 seed at temperature 0; document the determinism caveat).
2. Each baseline's accuracy on at least one dataset reproduces the published number for that baseline-on-that-dataset to within ±2 percentage points (sanity check that the baselines are correctly implemented). For the LLM baseline: report the published Sufficient-Context numbers from Joren et al. on the same evaluation slice and verify our reproduction is within ±3 pp.
3. The phase-3 memo reports:
   - **H1 evidence on real data**: HoVer label accuracy stratified by hop count for ours vs `architectural control` vs `LLM zero-shot`.
   - **H2 evidence on real data**: ρ(`m(Θ)`, ground-truth-thinness-proxy) on AVeriTeC and SciFact; AURC across the six models; reliability diagrams.
   - **Pareto front**: scatter of mean accuracy vs AURC across all (dataset × model) cells; ours's position on the front is the headline figure.
4. Our system's reported accuracy on AVeriTeC v2 dev SHALL be reported alongside the 2025 leaderboard winning AVeriTeC score (33.17). It does not need to *beat* it (Epic 04 reframes appropriately if we don't); it must be reported.

## Impact

- Adds `reference_v2/benchmarks/`, `reference_v2/baselines/` (new subpackages).
- Adds `reference_v2/experiments/benchmark_eval.py`.
- Adds `docs/publication/phase3_benchmarks_and_baselines.md`.
- Adds optional dependencies under `[study]`: `transformers` (already direct), `sentence-transformers`, `anthropic`, `openai` (vendored cautiously), `requests`. LLM API keys via env vars, never committed.
- Disk: ~5 GB for cached benchmark data + LLM call cache. Documented in README.
- Approximate effort: ~15 working days of engineering + ~3 days of writing.
