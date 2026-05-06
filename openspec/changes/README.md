# Publication-readiness program — four epics

These four epics convert the audit findings and the Cognition v4 system into an arXiv-submitted preprint and a PyPI-published `infon` package.

| Epic | Phase | Goal | Blocked by | Status | Acceptance gate |
|---|---|---|---|---|---|
| [`epic-01-stabilize-theta`](epic-01-stabilize-theta/) | 1 | Fix the Θ collapse in `reference_v2`. | none | **COMPLETE** (`v0.2.0-phase1`) | `m(Θ) ∈ [0.20, 0.40]`, correct polarity, std ≤ 0.05 across five seeds, canonical config `top1/k=2/cw=1.0`. |
| [`epic-02-synthetic-stress-and-ablations`](epic-02-synthetic-stress-and-ablations/) | 2 | Synthetic stress test + ablation matrix on `reference_v2`. | Epic 01 | **COMPLETE** (`v0.3.0-phase2`) | 10k scenarios; 8-cell ablation; H2 teacher signal ρ = −0.739; encoder-collapse null result documented. |
| [`epic-03-v4-evidence-and-baselines`](epic-03-v4-evidence-and-baselines/) | 3 | Benchmark evaluation of the Cognition v4 system on HoVer / AVeriTeC / SciFact against three tiered baselines. Evidence for H1 (multi-hop), H2 (honest abstention), H3 (DS calibration). | Epics 01–02 | **not started** | All baselines reproduce; H1/H2/H3 panels committed; phase-3 memo authored. |
| [`epic-04-paper-and-release`](epic-04-paper-and-release/) | 4 | System paper + `infon` PyPI package + arXiv submission. | Epic 03 (v4) | **not started** | `make pdf` clean; numbers audit passes; `infon` on PyPI; arXiv ID recorded. |

---

## Direction change after Epic 02

Epics 01 and 02 operated on `reference_v2` (the HypergraphReasoner GNN training loop). Epic 02 uncovered a fundamental blocker: BERT maps out-of-vocabulary template entity names (`"Entity N"`) to identical embeddings, collapsing the GNN input to a constant vector across all 10,000 synthetic scenarios. All seven trained cells produced polarity accuracy at 3-way chance (0.332 ± 0.000). H1 was additionally untestable because `compositional_depth` was not varied during generation.

A colleague simultaneously developed `reference_v4` ("Cognition") — a production-ready cassette-native knowledge graph reasoner that:
- Uses real NLP text (no synthetic template collapse)
- Ships a pretrained sheaf GNN frozen at inference (no per-corpus GNN training)
- Validates H2 empirically: θ → 1.00 on unsupported claims vs LLM 0.11
- Achieves 100% actor-to-actor accuracy post-extraction-fix on real data

**Epics 03 and 04 therefore target `reference_v4`, not `reference_v2`.** The old `epic-03-benchmarks-and-baselines/` and `epic-04-paper-and-arxiv/` directories are superseded — they are preserved for reference but should not be executed. The new epics are:

- **`epic-03-v4-evidence-and-baselines/`** — evaluates Cognition v4 on the same three datasets (HoVer, AVeriTeC, SciFact) with three baselines, using per-claim `InfonStore` evaluation (zero-shot, no pre-authored schema).
- **`epic-04-paper-and-release/`** — frames the paper as a system paper (not a hypothesis paper); adds a code release stage (the `infon` PyPI package with cassette code bundled, not as a separate dependency); updates hypothesis framing to H1/H2/H3 as validation results rather than primary claims.

The analysis motivating this direction change is documented in `docs/publication/updated_direction.md`.

---

## Superseded epics (preserved for reference)

| Directory | Why superseded |
|-----------|---------------|
| [`epic-03-benchmarks-and-baselines/`](epic-03-benchmarks-and-baselines/) | Targets `reference_v2`; depends on HypergraphReasoner training loop that mode-collapses on synthetic data; six-baseline design includes baselines meaningful only for v2 architecture ablations. |
| [`epic-04-paper-and-arxiv/`](epic-04-paper-and-arxiv/) | Hypothesis paper framing (H1/H2 as primary claims) and `reference_v2` reproducer; superseded by the system paper framing and v4 evidence in `epic-04-paper-and-release/`. |

Each epic follows the four-stage rhythm: **A** code → **B** run → **C** review and iterate → **D** finalize. Each gates the next by a named acceptance criterion in its `proposal.md`.
