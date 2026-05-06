# Submission Record

## Paper baseline tag
Tag: v0.4.0-paper
Commit: 4f428904e8c806ce023d06b9397d6bf4d1302cf7
Date: 2026-05-06

## Paper completeness status
- All 11 sections written
- 17/17 checklist items checked (see paper/CHECKLIST.md)
- 44 numbers audited (make lint passes: all 44 rows PASS)
- make pdf: builds cleanly (24 pages, 384451 bytes)
- make arxiv-tarball: produces paper/arxiv-submission/paper.tar.gz (0.03 MB, well within 50 MB limit)

## arXiv submission status

**BLOCKED on Epic 03** — all experimental numbers in §5 (H1/H2/H3) are placeholders
from `paper/tests/fixtures/`; real benchmark results from Epic 03 are required before submission.

Do NOT tag v1.0.0-paper or submit to arXiv until Epic 03 is complete and real numbers
have replaced the fixture placeholders.

## Steps to complete when Epic 03 runs

1. Run: `bash paper/reproducer.sh` (stages 2-4) to execute the full benchmark pipeline
2. Update `paper/numbers_audit.md` json_path entries to point to real panel JSONs
   (replace `paper/tests/fixtures/*.json` paths with actual Epic 03 result files)
3. Verify: `make lint` — all 44 rows must pass with real values
4. Regenerate figures: `python3 paper/figures/*.py` (reads from real result JSONs)
5. Final build: `make pdf`
6. Tarball: `make arxiv-tarball` (confirm size <= 50 MB)
7. Submit to arXiv: cs.CL primary, cs.LG + cs.IR cross-list
8. Record arXiv ID in this SUBMISSION.md and update README.md
9. Update `paper/sections/09_reproducibility.tex` with real arXiv ID
10. Tag commit `v1.0.0-paper` and push

## arXiv submission (to be filled when submitted)
arXiv ID:
Submission URL:
Submission timestamp:
Model used in LLM evaluation: claude-sonnet-4-6
Warnings:

## Post-submission tag
Tag v1.0.0-paper: NOT YET CREATED — will be created at actual arXiv submission time.
