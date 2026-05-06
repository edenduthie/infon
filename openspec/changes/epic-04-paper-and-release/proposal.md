# Epic 04 — System Paper + arXiv Submission

> Phase 4 of 4. Blocked by Epic 03 (v4). Final deliverable of the publication-readiness program.
> **Scope:** paper only. PyPI/package release is a separate future epic.

## Why

Epic 03 (v4) produces locked result JSONs covering H1, H2, and H3 on three public datasets. Without this epic those results remain internal memos. This epic converts them into a single arXiv-ready system paper describing the Cognition knowledge graph reasoner.

The code deliverable for this epic is a **tagged research release** of the existing `reference_v4/` directory — not a reorganized PyPI package. The paper's Reproducibility section links to a pinned GitHub tag. A production PyPI release (`infon`) is planned as a separate future epic after the community has had a chance to engage with the paper.

**Frame: system paper, not hypothesis paper.** The Cognition system — its cassette storage format, Kan-extension schema migration, Dempster–Shafer mass with first-class θ, and MCTS + sheaf GNN reasoner — is the contribution. H1, H2, and H3 are validation results that demonstrate measurable properties of the system.

## What Changes

- Adds `paper/` directory: `main.tex`, section files under `paper/sections/`, `references.bib`, `Makefile`, figure regeneration scripts, `paper/reproducer.sh`, `paper/numbers_audit.md`, `paper/CHECKLIST.md`, arXiv packaging script.
- Tags the current `reference_v4/` state as `v0.4.0-paper` — no reorganization of the codebase.
- Archives `docs/publication/draft2.txt` to `docs/publication/_archive/`.
- Updates `README.md` with arXiv preprint link after submission.

## Paper Structure

System paper. Section sequence:

```
Abstract → Introduction → Background →
System Architecture → Implementation →
Experiments (H1 / H2 / H3) → Discussion →
Related Work → Conclusion → Reproducibility → System Card
```

The Background section covers DS theory, situation semantics, SPLADE, sheaf GNNs.
The System Architecture section is the core contribution: cassette format, DSL, reasoner, Kan migration.
The Experiments section reports H1/H2/H3 findings honestly including null results.

## Phased Scope

- **Stage A — Tooling:** paper scaffold (LaTeX, Makefile, figure scripts, reproducer, numbers audit checker, arXiv packaging). All tooling is test-driven before any writing begins.
- **Stage B — First draft:** write all sections from scratch using Epic 03 locked results. Every section completed before Stage C begins.
- **Stage C — Review and revise:** three structured passes (claims + evidence, figures + notation, related work + citations). Revise until CHECKLIST ≥ 95%.
- **Stage D — Submit:** arXiv tarball, submission, record ID, tag commit.

## Acceptance Criterion

1. `make pdf` builds with zero LaTeX errors and zero unresolved references.
2. Every numeric in the paper traces to a locked Epic 03 JSON via `paper/numbers_audit.md`. `make lint` exits zero.
3. `paper/reproducer.sh` on a clean clone (after `git checkout v0.4.0-paper && pip install -e "reference_v4/[study]"`) regenerates every figure and number.
4. `paper/CHECKLIST.md` ≥ 95% checked.
5. Paper uploaded to arXiv; ID recorded in `paper/SUBMISSION.md` and `README.md`.
6. Null/negative findings from Epic 03 are present in Discussion; no result suppressed.

## Impact

- Adds `paper/` (new top-level directory).
- Tags `v0.4.0-paper` on current `reference_v4/` state — no code changes.
- Archives `docs/publication/draft2.txt`.
- Approximate effort: ~2 days tooling + ~8 days writing + ~2 days review + ~1 day submission.
