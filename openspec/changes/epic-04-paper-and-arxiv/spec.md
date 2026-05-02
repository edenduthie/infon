# Spec: Epic 04 — Publication-ready paper + arXiv submission

## Hard Rules

- **TDD for tooling:** figure scripts, reproducer, numbers-audit checker, and arXiv tarball builder are tested via integration tests before authoring. Tests use real LaTeX, real figure libraries, real reproducer invocations on small fixtures.
- **No Mocks:** no test stubs out LaTeX, the figure pipeline, or the reproducer.
- **No Orphan Numbers:** every numeric value in the rendered PDF traces to a JSON file under `experiments/results/` via `paper/numbers_audit.md`. CI fails when the audit doesn't validate.
- **No Unsupported Claims:** every textual claim in the paper is either (a) backed by an artifact from Epics 01–03, (b) a citation to published prior work, or (c) explicitly framed as opinion/discussion. Items lacking one of these three are removed before submission.
- **Stage-boundary review:** at each stage's end, run `pytest paper/tests/ -v`, `make pdf`, and `make lint`; verify spec compliance.

## ADDED Requirements

### Requirement: Paper Structure Mirrors Hypothesis Contract

`paper/main.tex` SHALL `\input` exactly the section sequence: abstract → introduction → background → method → hypotheses → current baseline → datasets → baselines → results → discussion → related work → conclusion → reproducibility → system card. The section files SHALL exist under `paper/sections/` with the names listed in `tasks.md` A.2.

#### Scenario: Section sequence enforced
- **WHEN** `paper/main.tex` is parsed
- **THEN** the documented `\input` sequence appears in the documented order

**Testability:** AST/text parse of `main.tex` against the expected sequence.

---

### Requirement: Numbers Audit Trail

`paper/numbers_audit.md` SHALL be a markdown table with at least the columns `value | section | location | json_path | json_key | tolerance | description`. Every numeric value that appears in the rendered PDF (excluding equation auto-numbering, page numbers, and reference numbers) SHALL have an entry. The validator `paper/tools/check_numbers_audit.py` SHALL parse the table, look up each `(json_path, json_key)`, and assert the value matches to the row's `tolerance`. The CI job `make lint` SHALL invoke the validator and fail on any miss.

#### Scenario: Validator catches drift
- **GIVEN** a `numbers_audit.md` row pointing at a JSON containing `{"x": 0.31}` with tolerance `0.005` and an in-text claim of `0.30`
- **WHEN** the validator runs
- **THEN** it reports a passing match (within tolerance)

#### Scenario: Validator fails on real drift
- **GIVEN** a row with tolerance `0.005` and an in-text claim of `0.50` against a JSON containing `0.30`
- **WHEN** the validator runs
- **THEN** it exits non-zero with a message naming the offending row and the actual delta

**Testability:** integration tests in `paper/tests/test_numbers_audit.py` cover both pass and fail.

---

### Requirement: Deterministic Reproducer

`paper/reproducer.sh` SHALL run the full pipeline end-to-end on a clean checkout (after `pip install -e ".[study]"` and unpacking the LLM cache from Epic 03). The script SHALL be:

- **Idempotent**: re-running on a partially-completed state SHALL skip completed stages and resume.
- **Deterministic**: a clean run from a fresh clone SHALL reproduce every figure and number to documented seed tolerance.
- **Documented**: each stage SHALL print, in order, the command being run and a one-line description.

#### Scenario: Dry-run lists every stage
- **WHEN** `paper/reproducer.sh --dry-run` is invoked
- **THEN** it prints the sequence of commands without executing them

#### Scenario: Resume skips completed stages
- **GIVEN** Stage 1 (Epic 01 canonical run) has completed and produced its JSON
- **WHEN** the reproducer is re-invoked
- **THEN** Stage 1 is skipped and Stage 2 is run

**Testability:** integration test in `paper/tests/test_reproducer_dry_run.py`; resumability test on a tiny fixture pipeline.

---

### Requirement: Figure Determinism

Every figure under `paper/figures/output/` SHALL be regenerable from the corresponding script + JSON inputs without manual intervention. Two regenerations from identical inputs on the same machine SHALL produce byte-identical (or near-identical, via PDF text-extract diff) outputs.

#### Scenario: Determinism check
- **GIVEN** a figure script and its JSON input
- **WHEN** the script is invoked twice in succession
- **THEN** the two output files are identical (or PDF-text-equivalent)

**Testability:** `paper/tests/test_figures_deterministic.py` runs each figure script twice and compares.

---

### Requirement: arXiv-Compliant Submission Tarball

`paper/arxiv-submission/build.sh` SHALL produce a `.tar.gz` containing exactly:

- `main.tex`
- every `\input`-ed `.tex` file
- `references.bbl` (the resolved bibliography; **not** `references.bib`)
- every figure file referenced by an `\includegraphics`
- `00README.txt` describing top-level layout

The tarball SHALL **not** contain: `.aux`, `.log`, `.out`, `.bib`, `.synctex.gz`, `.fls`, `.fdb_latexmk`, hidden files, editor swap files. Total size SHALL be ≤ 50 MB.

#### Scenario: Tarball passes manifest check
- **WHEN** `build.sh` runs to completion
- **THEN** the produced tarball satisfies all inclusions and exclusions; an internal manifest validator exits zero

**Testability:** `paper/tests/test_arxiv_tarball.py` runs the build and asserts contents.

---

### Requirement: Pre-Submission Checklist

`paper/CHECKLIST.md` SHALL contain the audit's pre-submission checklist (`docs/publication/reproduction_audit.md` lines 129–139) plus these additional items:

- [ ] `paper/numbers_audit.md` validates clean.
- [ ] `paper/reproducer.sh` produces every figure and number from a clean checkout.
- [ ] All Epic 01–03 acceptance criteria are documented in the paper as evidence.
- [ ] Honda REFUTES claim absent (audit Critical row #2).
- [ ] Honest reporting of any null/negative findings.
- [ ] arXiv tarball validated.
- [ ] Numbers audit and reproducer linked from `README.md`.
- [ ] Author list, affiliations, contact email, ORCID iDs present.
- [ ] License (`paper/LICENSE`) is CC BY 4.0 or compatible.
- [ ] System card (`paper/sections/system_card.tex`) included.

The checklist SHALL be at least 95% checked before Stage D (submission) begins.

**Testability:** parse `CHECKLIST.md` and count checked vs unchecked.

---

### Requirement: Honest Treatment of H1/H2 Outcomes

The paper SHALL faithfully represent the H1/H2 evidence produced by Epics 01–03:

1. If Epic 01 satisfied its acceptance criterion (Θ recovered): H2 is presented as a positive hypothesis with the supporting evidence from Epics 02 and 03.
2. If Epic 01 retracted H2: H2 is presented as a documented null result, the paper's contribution narrows accordingly, and §4 + §8 are rewritten without the H2 claim.
3. If Epic 03 produced null/negative results on a public benchmark: those results appear in §8 alongside positive results, and §9 (Discussion) addresses what they mean.

#### Scenario: Retraction handled
- **GIVEN** Epic 01's `decision-record.md` recommending H2 retraction
- **WHEN** Stage C.1.3 runs
- **THEN** §4 and §8 of the paper no longer contain the H2 claim, replaced by a documented null finding

**Testability:** review-pass checklist; not strictly automatable but enforced by the Stage-C process.

---

### Requirement: Submission Record

`paper/SUBMISSION.md` SHALL record: arXiv ID once assigned; submission URL; submission timestamp; the model and version of any LLMs used in production runs (e.g. `claude-sonnet-4-6`, `gpt-4o-2024-08`); any submission warnings; any rejection-then-resubmission events with their resolution. The file SHALL be created during Stage D.

#### Scenario: Record is complete
- **WHEN** the epic is closed
- **THEN** `paper/SUBMISSION.md` exists and contains an arXiv ID

**Testability:** existence + content check in CI.

---

### Requirement: Archival of Prior Draft

`docs/publication/draft2.txt` (the audited draft) SHALL be moved to `docs/publication/_archive/draft2_2026-04.txt` with a one-line preamble: "Superseded by arXiv:<id>; see `docs/publication/reproduction_audit.md` for the audit that motivated the rewrite." The original location SHALL contain a redirect note `docs/publication/draft2.txt.MOVED.md`.

#### Scenario: Archive present, original location redirected
- **WHEN** the epic is closed
- **THEN** the archived file exists, the redirect note exists, and `git log --follow` traces the rename history

**Testability:** existence + content checks; `git log --follow` test.

---

### Requirement: Phase-4 Memo

`docs/publication/phase4_paper.md` SHALL be the closing memo for the publication-readiness program. It SHALL contain: arXiv ID; brief reflection on H1/H2 outcomes (especially any retraction); pointers to all four phase memos; recommended follow-ups (peer-review submission target, response-to-reviewers process, additional benchmark coverage). The memo SHALL be linked from the top-level `README.md`.

**Testability:** existence + link check.
