# Epic 03 (v4) — Evidence Generation + Baselines

> Phase 3 of 4. Supersedes `epic-03-benchmarks-and-baselines/` (which targeted `reference_v2`). Blocked by Epics 01–02 (done). Unblocks Epic 04.

## Why

Epics 01 and 02 established that the Cognition system (`reference_v4`) works on real NLP text and that the `reference_v2` GNN training loop is blocked by a BERT entity-collapse at the synthetic harness level. The path to publication no longer runs through `synthetic_v2` or through the HypergraphReasoner training loop. It runs through `reference_v4`'s production system.

This epic produces all the evidence the paper needs: benchmark results on three public datasets, comparison against three tiered baselines, and quantitative support for three updated hypotheses:

- **H1 — Multi-hop advantage:** MCTS-guided multi-hop traversal with sheaf GNN rescoring achieves higher accuracy on multi-evidence claims than flat retrieval, and the advantage grows with chain depth.
- **H2 — Honest abstention:** On claims the corpus does not support, the system's residual uncertainty mass θ approaches 1.0 rather than returning a confident (wrong) verdict. This gives a measurably lower false-positive endorsement rate than LLM-class baselines.
- **H3 — Calibrated uncertainty:** Dempster–Shafer mass achieves lower expected calibration error (ECE) than neural 3-way softmax baselines on claim verification.

These replace the original H1 (typed IKL aggregator vs uniform mean) and H2 (ρ(m(Θ), planted thinness)) which were designed for a GNN training ablation and are no longer the critical test.

## What Changes

- Adds `reference_v4/benchmarks/` with loaders for HoVer (H1 testbed), AVeriTeC v2 dev (H2 testbed), and SciFact (H3 testbed).
- Adds `reference_v4/baselines/` with three baseline implementations plus a uniform interface.
- Adds `reference_v4/baselines/_llm_cache.py` for deterministic, offline-first LLM evaluation.
- Adds `reference_v4/experiments/benchmark_eval.py` driving the Cartesian product `(dataset × system × seed)`.
- Adds `reference_v4/experiments/metrics.py` (six-metric battery: polarity accuracy, ECE, Brier, AURC, selective accuracy at coverage {0.5, 0.7, 0.9}, Spearman ρ).
- Adds figure scripts for the three hypothesis panels and the Pareto front.
- Produces `docs/publication/phase3_evidence_generation.md`.

The `reference_v4/cognition/` package is the evaluation subject throughout; `reference_v2/` is not touched.

## Datasets

| Dataset | Primary hypothesis | Why |
|---------|------------------|-----|
| **HoVer** (Jiang et al., EMNLP-Findings 2020) | H1 | Multi-hop fact verification; `num_hops ∈ {2, 3, 4}` annotated, enabling a direct depth-stratified comparison of MCTS vs flat retrieval |
| **AVeriTeC v2 dev** (Schlichtkrull et al., ACL 2024–2025) | H2 | Real-world claim verification with explicit `NOT_ENOUGH_EVIDENCE` class; NEI label is the natural proxy for "corpus doesn't support this" where θ should approach 1.0; 2025 leaderboard score (33.17%) provides an external anchor |
| **SciFact** (Wadden et al., EMNLP 2020) | H3 | Scientific claims with thin per-claim evidence (1–3 supporting sentences max); calibration differences (ECE) are largest when evidence is sparse — exactly where DS mass vs softmax diverges |

## Baselines

Three baselines spanning difficulty:

| # | Name | Description |
|---|------|-------------|
| 1 | **Symbolic floor** | SPLADE retrieval + rule-based polarity (no GNN, no MCTS); the system-minus-learning ablation |
| 2 | **NLI classifier** | RoBERTa-large fine-tuned on FEVER; temperature-scaled post-hoc; the standard strong NLI baseline |
| 3 | **LLM zero-shot** | Claude claude-sonnet-4-6 with retrieval; verbalized confidence mapped to 4-element mass; the current practical ceiling for accuracy |

Six baselines from the old Epic 03 spec have been collapsed to three. The R-GCN control, Dirichlet EDL, and Sufficient-Context baselines were designed to probe the v2 GNN architecture. The three retained baselines cover the symbolic/neural/LLM spectrum that reviewers will ask about, without the infrastructure cost of the dropped baselines.

## Evaluation Setup

Each claim is evaluated in a **per-claim InfonStore**: the claim's associated evidence documents (Wikipedia articles for HoVer, Q&A pairs for AVeriTeC, abstracts for SciFact) are ingested into a fresh `InfonStore`. A schema is auto-generated per claim corpus via SPLADE anchor co-activation spectral clustering. The system is then queried with `ask()` or `connect()` and the verdict is compared to the ground-truth label.

This zero-shot setup (no schema pre-authoring, no dataset-specific tuning) is the honest evaluation surface for a system that claims to be general-purpose.

## Phased Scope

- **Stage A — Code**: loaders, baselines, LLM cache, evaluation harness, metrics, figure scripts.
- **Stage B — Run**: baseline reproduction check; full `(dataset × system × seed)` matrix; H1/H2/H3 panels.
- **Stage C — Review and iterate**: validate baseline numbers against published; investigate unexpected cells; lock results.
- **Stage D — Finalize**: SHA256-lock all result JSONs; write phase-3 memo; tag `v4.1.0-evidence-complete`.

## Acceptance Criterion (gates Epic 04)

1. All three datasets run to completion for all three baselines and the Cognition system (symbolic-only and symbolic+GNN variants), under 3 seeds where applicable (LLM baseline: 1 seed at temperature 0).
2. Each NLI baseline achieves within ±2 pp of its published accuracy on the dataset it was natively evaluated on (sanity gate). LLM baseline: ±3 pp on any published LLM-zero-shot number on the same split.
3. The H1 panel exists: HoVer label accuracy table stratified by `num_hops ∈ {2, 3, 4}`, one row per system, with 95% bootstrap CIs, committed as `experiments/results/h1_panel.json`.
4. The H2 panel exists: AVeriTeC θ distribution histogram (NEI vs SUPPORTED/REFUTED) + ρ(m(Θ), nei_indicator) for all systems, committed as `experiments/results/h2_panel.json`.
5. The H3 panel exists: ECE, Brier, AURC table across all datasets and systems, committed as `experiments/results/h3_panel.json`.
6. A Pareto-front figure exists (accuracy vs AURC across all dataset × system cells).
7. `docs/publication/phase3_evidence_generation.md` is authored and linked from `reference_v4/README.md`.

## Impact

- Adds `reference_v4/benchmarks/` and `reference_v4/baselines/` (new subpackages; no existing code modified).
- Adds `reference_v4/experiments/benchmark_eval.py`, `metrics.py`, `stats.py`, `figures/`.
- Optional dependencies under `[study]`: `transformers`, `anthropic`, `requests`, `scipy`. API keys via env vars, never committed.
- Disk: ~3 GB for cached benchmark data + LLM call cache.
- Approximate effort: ~8 working days of engineering + ~2 days of writing.
