# Epic 02 — Synthetic stress dataset + full ablation matrix

> Phase 2 of 4. Blocked by Epic 01 (`canonical_v0_2.yaml` must exist; H2 must not have been retracted). Unblocks Epic 03.

## Why

The current paper rests on a single 5-document, 49-infon hand-authored EV scenario. With one corpus and one configuration, neither H1 (compositional logic) nor H2 (Θ tracks evidential thinness) is statistically testable — the audit's three-seed reproduction shows numbers are stable to the third decimal *because there is no variance to vary*. Reviewers will (correctly) reject any paper that supports flagship claims with n=1.

The cheapest path to statistical power is a **parametric synthetic generator** with planted oracle labels. We can vary evidence thinness (the H2 axis), compositional depth (the H1 axis), and contradiction density (a confound for H2), generate 10k scenarios cheaply, and run the full ablation matrix from the experiment plan. This produces the H1 effect-size measurements (typed-IKL vs uniform-mean aggregator across hop counts) and the H2 correlation (`ρ(m(Θ), planted_thinness)`) that the paper will cite.

This epic does not yet involve public benchmarks (those are Epic 03) — synthetic data is faster, deterministic, and lets us isolate effects before contaminating the picture with the messiness of HoVer/AVeriTeC/SciFact.

## What Changes

- Adds a `reference_v2/synthetic/` package implementing a deterministic parametric corpus generator with knobs: `n_docs`, `n_anchors_per_type`, `evidence_redundancy ∈ {1, 2, 5, 10}`, `compositional_depth ∈ {1, 2, 3, 4}`, `contradiction_density ∈ {0.0, 0.1, 0.3}`, `nei_fraction ∈ {0.1, 0.3, 0.5}`. Output: a list of `Scenario` objects, each carrying its corpus, query set, and oracle labels (`planted_verdict`, `planted_thinness`, `planted_hop_count`).
- Adds `reference_v2/experiments/ablation_matrix.py` running the full Cartesian over the components named in `spec.md` (aggregator type × readout type × fusion rule × top-k × coherence weight × teacher leave-one-out × layer count) on the synthetic dev set.
- Adds metric battery `reference_v2/experiments/metrics.py`: polarity accuracy (3-way SUPPORTS/REFUTES/NEI), ECE, Brier score, AURC (area under risk-coverage curve), selective accuracy at fixed coverage levels {0.5, 0.7, 0.9}, Spearman ρ between `m(Θ)` and `planted_thinness`.
- Adds result aggregation, bootstrap CI estimation across seeds, and per-row significance tests (paired bootstrap).
- Produces a phase-2 technical memo at `docs/publication/phase2_synthetic_ablations.md`.

## Phased Scope (within this epic)

- **Stage A — Code**: synthetic generator, oracle labels, ablation harness, metric battery, result aggregator.
- **Stage B — Run**: pilot run (100 scenarios) for sanity; full run (10k scenarios × 3 seeds × ablation matrix).
- **Stage C — Review and iterate**: identify outliers, debug pathological cells, re-run; if any ablation shows that ours is *worse* than a stripped variant on H1 or H2 metrics, treat it as a finding (not a bug) and document.
- **Stage D — Finalize**: lock results JSON; commit canonical ablation tables; write phase-2 memo with figures.

## Acceptance Criterion (gates Epic 03)

1. The synthetic generator produces ≥ 10 000 scenarios under fixed seed `42`, total ingestion time ≤ 60 minutes on a single CPU.
2. Full ablation matrix runs to completion across 3 seeds for at least the eight starred cells in `tasks.md` Stage B.
3. The phase-2 memo reports:
   - **H1 evidence**: a table of polarity accuracy stratified by `planted_hop_count ∈ {1, 2, 3, 4}` for `aggregator ∈ {typed-IKL, uniform-mean}`, with bootstrap 95% CIs across seeds. The H1 effect size is the (typed − uniform) gap at `hop_count=4`. The acceptance criterion is *not* a particular sign — a null or negative result is also a publishable finding — but the test must produce a numerically defensible answer.
   - **H2 evidence**: Spearman ρ(`m(Θ)`, `planted_thinness`) on the held-out synthetic test set under the canonical configuration; an AURC table for `readout ∈ {DS-4-mass, softmax+T, Dirichlet-EDL}`; reliability diagrams for each.
4. The matrix's full `aggregate.json` is committed under `experiments/results/ablation_matrix/`.

## Impact

- Adds `reference_v2/synthetic/` (new subpackage).
- Adds `reference_v2/experiments/ablation_matrix.py`, `metrics.py`, `aggregate.py`.
- Adds `docs/publication/phase2_synthetic_ablations.md`.
- No public API removed.
- Adds `scipy.stats` (already a transitive dep) and one new dep `statsmodels` for paired bootstrap CIs (small, optional via `[study]` extra).
- Approximate effort: ~10 working days of engineering + ~2 days of writing.
