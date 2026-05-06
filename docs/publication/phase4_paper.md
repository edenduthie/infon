# Phase 4 Memo: Cognition System Paper and arXiv Submission

**Date**: 2026-05-06
**Status**: Paper complete; arXiv submission pending Epic 03 completion
**Epic**: Epic 04 — Cognition System Paper + arXiv Submission
**Tag**: v0.4.0-paper (commit 4f428904e8c806ce023d06b9397d6bf4d1302cf7)

---

## arXiv Submission Status

**Pending Epic 03 completion.**

The paper is structurally complete (all 11 sections, 17/17 checklist items, 44 numbers
audited, make lint passes, PDF builds cleanly at 24 pages). The arXiv tarball is
0.03 MB and passes all submission checks.

However, all H1/H2/H3 quantitative results in §5 (Experiments) are placeholder values
drawn from `paper/tests/fixtures/`. Real benchmark results require Epic 03 (reference_v4
evaluation pipeline) to complete before the paper can be submitted.

See `paper/SUBMISSION.md` for the full checklist of steps to complete at submission time.

---

## H1/H2/H3 Outcome Summary (Fixture Placeholders)

These values are structural placeholders, not real experimental results. They establish
the expected direction of each hypothesis and will be replaced by Epic 03 output.

### H1 — Multi-hop Advantage (HoVer dataset)

- **Hypothesis**: Cognition+GNN outperforms flat retrieval on multi-hop claims,
  with the gap widening as hop count increases.
- **Fixture result**: 24 pp gap at 4-hop between Cognition+GNN and flat retrieval
  (figure: figures/accuracy_by_hop.pdf).
- **Hypothesis direction**: Supported (pending real data).
- **What to verify with real data**: The 24 pp gap placeholder should be replaced
  by the actual accuracy-by-hop curve from the Epic 03 HoVer panel JSON.

### H2 — Honest Abstention (AVeriTeC dataset)

- **Hypothesis**: Cognition's uncertainty signal m(Theta) enables calibrated abstention
  on NEI (Not Enough Information) claims, outperforming the LLM baseline.
- **Fixture result**: m(Theta)=0.89 on NEI for Cognition vs 0.60 for LLM baseline;
  Spearman rho=0.95 vs 0.52 for confidence-accuracy correlation.
- **Hypothesis direction**: Supported (pending real data).
- **What to verify with real data**: Replace fixture values in numbers_audit.md
  with paths to real AVeriTeC panel JSONs.

### H3 — DS Calibration (SciFact dataset)

- **Hypothesis**: Cognition+GNN achieves better calibration (lower ECE) than the NLI
  baseline on the SciFact claim verification task.
- **Fixture result**: ECE=0.05 for Cognition+GNN vs 0.12 for NLI baseline.
- **Hypothesis direction**: Supported (pending real data).
- **What to verify with real data**: Replace fixture values in numbers_audit.md
  with paths to real SciFact panel JSONs.

---

## Phase Memo Index

| Phase | Document | Epic | Status |
|-------|----------|------|--------|
| Phase 1 | [docs/publication/report_epic_1.md](report_epic_1.md) | Epic 01 — Stabilize Theta (collapse fix) | Closed, tag v0.2.0-phase1 |
| Phase 1 (detail) | [docs/publication/phase1_collapse_fix.md](phase1_collapse_fix.md) | Epic 01 — technical detail | Closed |
| Phase 2 | [docs/publication/report_epic_2.md](report_epic_2.md) | Epic 02 — Synthetic Stress Dataset + Ablation Matrix | Closed, tag v0.3.0-phase2 |
| Phase 2 (detail) | [docs/publication/phase2_synthetic_ablations.md](phase2_synthetic_ablations.md) | Epic 02 — technical detail | Closed |
| Phase 3 | Epic 03 — reference_v4 benchmark evaluation | Real H1/H2/H3 results | IN PROGRESS (blocker) |
| Phase 4 | This document | Epic 04 — Paper + arXiv submission | Paper complete; submission pending |

Additional context documents in this directory:
- [reproduction_audit.md](reproduction_audit.md) — numeric consistency audit
- [updated_direction.md](updated_direction.md) — research direction notes

---

## Recommended Next Steps

### Priority 1 (immediate): Complete Epic 03

Epic 03 (reference_v4 benchmark evaluation) is the hard blocker for arXiv submission.
Once Epic 03 runs:

1. Run `bash paper/reproducer.sh` (stages 2-4).
2. Update `paper/numbers_audit.md` json_path columns to point to real panel JSONs
   (replacing all `paper/tests/fixtures/*.json` paths).
3. Verify: `make lint` — all 44 rows must pass.
4. Regenerate all four figures: `python3 paper/figures/*.py`.
5. Final build: `make pdf` and check output is correct.
6. Tarball: `make arxiv-tarball` — confirm size <= 50 MB.
7. Submit to arXiv: cs.CL primary, cs.LG + cs.IR cross-list.
8. Record arXiv ID in `paper/SUBMISSION.md` and update `README.md`.
9. Update `paper/sections/09_reproducibility.tex` with real arXiv ID.
10. Tag commit `v1.0.0-paper` and push.

### Priority 2 (post-arXiv): Conference submission

Target venues for the full paper:
- ACL 2026 (deadline ~February 2026)
- EMNLP 2026 (deadline ~June 2026)
- NAACL 2026 (deadline ~January 2026)

Select the venue that best fits the timeline after arXiv goes live.

### Priority 3 (future epic): PyPI package release

The `infon` Python package is not yet on PyPI. A separate epic should cover:
- Package polish and API stability
- Documentation site (ReadTheDocs or similar)
- PyPI release and versioning strategy

This is out of scope for Epic 04 and should be planned as a distinct future epic.
