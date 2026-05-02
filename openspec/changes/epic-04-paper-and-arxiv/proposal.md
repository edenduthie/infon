# Epic 04 — Publication-ready paper + arXiv submission

> Phase 4 of 4. Blocked by Epics 01–03. Final deliverable of the publication-readiness program.

## Why

Epics 01–03 produce the artifacts a publishable paper needs: a stable canonical configuration with non-trivial Θ (Epic 01); a parametric stress dataset with H1/H2 effect sizes and reliability diagrams (Epic 02); public-benchmark results against six tiered baselines and a Pareto front (Epic 03). Without Epic 04, those artifacts remain three internal memos and a results JSON tree — not a paper.

The aim of this epic is to produce a single arXiv-ready paper (preprint, non-anonymous, cs.CL primary with cs.LG cross-listing) that:

1. States the two operationalised hypotheses (H1 compositional logic; H2 calibrated ignorance) up front.
2. Reports the current baseline (the system as it stood before Epic 01).
3. Presents the SOTA reference points (AVeriTeC 2025 leaderboard 33.17; HoVer GNN-on-LLMs lineage; Sufficient-Context selective generation) so reviewers can place us.
4. Reports H1 and H2 evidence honestly, including null/negative findings from Epics 02 and 03.
5. Compiles deterministically from a single seeded reproducer that re-runs every reported number from a clean checkout.
6. Is *publication-ready for arXiv*, which is a substantially weaker bar than peer review — the goal is that the artefact is honest, complete, and reproducible, not that it would survive a top-venue review cycle without revision. Peer-review-ready is a follow-up.

## What Changes

- Adds `paper/` directory containing: LaTeX source (`main.tex`, sectioned files under `paper/sections/`), `references.bib`, figure scripts that regenerate every figure from the results JSONs, a `Makefile`, a `paper/reproducer.sh` that runs the entire pipeline end-to-end on a clean checkout.
- Adds `paper/CHECKLIST.md` reproducing the audit's pre-submission checklist (`docs/publication/reproduction_audit.md` lines 129–139) plus the additional items from Epics 01–03.
- Archives `docs/publication/draft2.txt` (the audited draft) under `docs/publication/_archive/` and supersedes it with the new draft.
- Updates `CHANGELOG.md` and `README.md` to point to the arXiv preprint URL after submission.
- Adds an `arxiv-submission/` tarball-prep script that produces an arXiv-compliant submission package (`.tar.gz` with `main.tex`, dependent `.tex`, `.bbl` (not `.bib`), figure files, and no auxiliary clutter).

## Phased Scope (within this epic)

- **Stage A — Code**: paper scaffold (LaTeX template, Makefile, figure scripts, reproducer); CHECKLIST; arXiv-submission packaging script.
- **Stage B — Run**: regenerate every figure and table from the locked JSONs of Epics 01–03; run the reproducer on a clean checkout; verify all numbers in the paper trace to a JSON (no orphan numbers).
- **Stage C — Review and iterate**: three review passes — (1) content & claim audit (every claim cites evidence; the Honda REFUTES claim from `draft2.txt` is removed per audit Critical row); (2) figures, tables, and notation; (3) related work, citations, and prior-art coverage. Iterate until the CHECKLIST is fully checked.
- **Stage D — Finalize**: produce the arXiv-submission tarball; submit to arXiv; record the assigned arXiv ID; tag commit `v1.0.0-paper`; update memos and READMEs with the preprint URL.

## Acceptance Criterion (final)

1. `paper/main.tex` compiles with `latexmk -pdf` to a single PDF with zero LaTeX errors and zero unresolved references.
2. Every numeric claim in the paper traces to a JSON file in `experiments/results/` (a `paper/numbers_audit.md` table maps each in-text number to its source).
3. `paper/reproducer.sh` runs end-to-end on a clean clone (with `pip install -e ".[study]"` plus the LLM cache from Epic 03) and reproduces every figure and number to within seed tolerance.
4. The audit's pre-submission checklist (`reproduction_audit.md` §Pre-submission, 9 items) is fully checked.
5. The paper is uploaded to arXiv. The arXiv ID is recorded in `paper/SUBMISSION.md` and in `README.md`. Failed submissions are recorded with their error messages and resolved before re-submission.
6. The paper does *not* claim H2 if Epic 01 retracted it; in the contingent case, the paper is reframed as an H1 + interpretability paper, the H2 claim is replaced with a documented null result, and acceptance criteria 1–5 still apply.

## Impact

- Adds `paper/` (new top-level directory).
- Adds `arxiv-submission/` build artefacts (gitignored).
- Adds `paper/reproducer.sh` and `paper/numbers_audit.md`.
- Archives `docs/publication/draft2.txt` (preserved, not deleted).
- Adds dev dependency `latexmk` (system-level, documented in `paper/README.md`); adds Python dep `arxiv-toolbox` or equivalent if needed for tarball validation.
- Approximate effort: ~10 working days of writing + ~3 days of figure work + ~2 days of submission engineering.
