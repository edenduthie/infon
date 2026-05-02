# Publication-readiness program — four epics

These four epics convert the audit findings in `docs/publication/reproduction_audit.md` into an arXiv-submitted preprint.

| Epic | Phase | Goal | Blocked by | Acceptance gate |
|---|---|---|---|---|
| [`epic-01-stabilize-theta`](epic-01-stabilize-theta/) | 1 | Fix the Θ collapse from the audit (Path 2). Highest-risk phase; if it fails, H2 is retracted and Epics 02–04 are rescoped. | none | `m(Θ) ∈ [0.20, 0.40]` with correct polarity, std ≤ 0.05 across five seeds, on the EV scenario, under one named canonical config. |
| [`epic-02-synthetic-stress-and-ablations`](epic-02-synthetic-stress-and-ablations/) | 2 | Build a parametric synthetic generator, run the full ablation matrix, produce H1 and H2 effect sizes with statistical power. | Epic 01 | 10k scenarios; eight named cells × 5 seeds; H1/H2 panels and reliability diagrams. |
| [`epic-03-benchmarks-and-baselines`](epic-03-benchmarks-and-baselines/) | 3 | Evaluate on HoVer / AVeriTeC v2 / SciFact against six tiered baselines (symbolic, NLI, R-GCN control, Dirichlet-EDL, LLM zero-shot, Sufficient-Context). | Epic 02 | All baselines reproduce their published numbers; Pareto-front plot; full results tables; H1 + H2 panels on real data. |
| [`epic-04-paper-and-arxiv`](epic-04-paper-and-arxiv/) | 4 | Synthesize Epics 01–03 into an arXiv preprint with a deterministic reproducer and a numbers-audit trail. | Epic 03 | `make pdf` clean; `numbers_audit` validates; CHECKLIST ≥ 95% checked; arXiv ID assigned. |

Each epic follows the four-stage rhythm: **A** code the harness → **B** run the experiments → **C** review and iterate on the unexpected → **D** finalize and produce the deliverable. Each epic gates the next by a named acceptance criterion in its `proposal.md`.

The paper structure is locked to a hypothesis contract: two operationalized hypotheses (H1 compositional logic; H2 calibrated ignorance), the current pre-study baseline, and the evidence Epics 01–03 produce. Null and negative findings are reported, not buried — see `epic-04-paper-and-arxiv/spec.md` § Honest Treatment.
