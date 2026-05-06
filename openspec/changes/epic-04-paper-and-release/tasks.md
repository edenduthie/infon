# Tasks: Epic 04 — System Paper + arXiv

> **Epic status:** not started. Blocked by Epic 03 (v4). Acceptance criterion in `proposal.md`.

## Development Rules

- **Test-First for tooling:** figure scripts, reproducer, numbers-audit checker, tarball builder — all tested before implementation.
- **No Mocks:** real LaTeX (local), real figure libraries, real reproducer on fixture JSONs.
- **No Orphan Numbers:** `make lint` must pass before Stage D.
- **Complete before reviewing:** all sections written before Stage C review passes begin.
- **Stage-Boundary Review:** at the end of each stage, run `pytest paper/tests/ -v` + `make lint`; re-read spec; update beads.

---

## Stage A — Tooling

### Repository baseline

- [ ] A.0 Tag current HEAD: `git tag v0.4.0-paper && git push origin v0.4.0-paper`. Confirm `reference_v4/` has no uncommitted changes.

### LaTeX scaffold

- [ ] A.1 Create `paper/` with `main.tex`, `references.bib`, `Makefile` (`pdf`, `clean`, `lint`, `arxiv-tarball` targets), `LICENSE` (CC BY 4.0), `README.md`.
- [ ] A.2 Create section stubs under `paper/sections/`: `00_abstract.tex`, `01_introduction.tex`, `02_background.tex`, `03_architecture.tex`, `04_implementation.tex`, `05_experiments.tex`, `06_discussion.tex`, `07_related_work.tex`, `08_conclusion.tex`, `09_reproducibility.tex`, `system_card.tex`. Each stub contains only a `\section{}` header and a `% TODO` comment.
- [ ] A.3 Write `paper/tests/test_section_sequence.py`: parse `main.tex`; assert `\input` commands appear in the documented order. **Verify red → green.**
- [ ] A.4 Run `make pdf` on stubs; confirm LaTeX template compiles to a skeleton PDF.

### Figure scripts

- [ ] A.5 Write `paper/tests/test_figures.py`: load fixture JSON from Epic 03 results; invoke each figure script; assert PDF exists > 1 KB; invoke twice; assert byte-identical. **Verify red.**
- [ ] A.5b Implement `paper/figures/accuracy_by_hop.py` (H1), `paper/figures/theta_distribution.py` (H2), `paper/figures/reliability_diagrams.py` (H3), `paper/figures/pareto.py`. Each reads from the locked Epic 03 panel JSONs. **Verify green.**

### Reproducer

- [ ] A.6 Write `paper/tests/test_reproducer.py`: `--dry-run` lists all stages; resumability on a 3-claim fixture. **Verify red.**
- [ ] A.6b Author `paper/reproducer.sh`: stages = (1) `pip install -e "reference_v4/[study]"`, (2) Epic 03 evaluation matrix, (3) generate panels, (4) regenerate figures, (5) `make lint`, (6) `make pdf`. Checkpoint-resumable; idempotent. **Verify green.**

### Numbers audit

- [ ] A.7 Write `paper/tests/test_numbers_audit.py`: pass case (within tolerance); fail case (delta > tolerance → non-zero exit). **Verify red.**
- [ ] A.7b Implement `paper/tools/check_numbers_audit.py`. Wire into `make lint`. **Verify green.**

### arXiv packaging

- [ ] A.8 Write `paper/tests/test_arxiv_tarball.py`: run `build.sh`; assert inclusions and exclusions; size ≤ 50 MB. **Verify red.**
- [ ] A.8b Implement `paper/arxiv-submission/build.sh`. **Verify green.**

### Checklist

- [ ] A.9 Author `paper/CHECKLIST.md` with all items from `spec.md`. Mark none checked.

- [ ] **STAGE-A REVIEW:** all tests green; `make pdf` on stubs compiles; `make lint` runs (passes trivially on empty audit); commit checkpoint `epic-04-stage-a-complete`.

---

## Stage B — First Draft

> Write all sections to completion before beginning Stage C. Do not review or revise while drafting — forward momentum only.

### Numbers audit population (runs in parallel with writing)

- [ ] B.0 Create `paper/numbers_audit.md` header row. As each section is written, immediately add rows for every numeric introduced. Run `make lint` after each section; fix mismatches before moving to the next section.

### Section writing

- [ ] B.1 Write `00_abstract.tex`: ≤ 150 words; opens with what Cognition is; names three headline properties; includes top H2 quantitative result; states code availability. Does not open with "we hypothesize."
- [ ] B.2 Write `01_introduction.tex`: problem (fact-checking, hallucination on unsupported claims, rigid schemas); gap; contribution (five bullet properties); roadmap paragraph.
- [ ] B.3 Write `02_background.tex`: DS theory with full notation `(m_S, m_R, m_U, m_Θ)`; infons (Barwise & Perry 1983); SPLADE (Formal et al. 2021); sheaf GNNs (Bodnar et al. 2022). All symbols defined here are used without re-definition in subsequent sections.
- [ ] B.4 Write `03_architecture.tex`: four subsections (cassette substrate, DSL, reasoner, schema migration). Include the measured performance numbers from `reference_v4/README.md` for each subsection. Every number added to `numbers_audit.md`.
- [ ] B.5 Write `04_implementation.tex`: SPLADE-tiny encoder; anchor type system; extraction pipeline; executor variants; Strands Analyst (9 tools); include the one-minute start code block.
- [ ] B.6 Write `05_experiments.tex`: evaluation setup; H1 subsection with depth-stratified table and figure; H2 subsection with θ distribution stats and figure; H3 subsection with ECE/Brier/AURC table and reliability diagrams; baseline reproduction gate table. Every number added to `numbers_audit.md`. Run `make lint` after this section.
- [ ] B.7 Write `06_discussion.tex`: H1/H2/H3 interpretation; Epic 02 encoder-collapse in Limitations; schema coverage failure rate; future work.
- [ ] B.8 Write `07_related_work.tex`: KGAT, GEAR, DREAM, GraphCheck, STRIVE, AFEV, SelectLLM, Sufficient-Context, EDL, R-GCN, FEVER, AVeriTeC. Closing paragraph on Cognition's differentiators.
- [ ] B.9 Write `08_conclusion.tex`: 2–3 paragraphs; restate contribution; summarise H1/H2/H3 findings honestly; one sentence on future work.
- [ ] B.10 Write `09_reproducibility.tex`: GitHub URL, tag `v0.4.0-paper`, install command, reproducer invocation, expected wall-clock, compute used.
- [ ] B.11 Write `system_card.tex`: intended use; out-of-scope use; known limitations; training data.
- [ ] B.12 Populate `references.bib` with all required citations (see spec §Related Work and §Background). Run `bibtex` and `latexmk`; zero unresolved citations.

### Build and audit

- [ ] B.13 Run `make pdf`; fix any LaTeX errors. Confirm full draft PDF renders end-to-end.
- [ ] B.14 Run `make lint`; iterate `numbers_audit.md` until green.
- [ ] B.15 Generate all figures from locked Epic 03 panel JSONs. Confirm all figure files exist and are referenced in `main.tex`.

- [ ] **STAGE-B REVIEW:** `make pdf` clean; `make lint` green; all figures present and referenced; full draft PDF exists; commit checkpoint `epic-04-stage-b-draft`. Then re-read the full PDF (not the source) as a reviewer would. Write a one-paragraph self-assessment in `paper/stage_b_review_notes.md` noting: (a) any claim that feels unsupported, (b) any section that reads as a hypothesis paper rather than a system paper, (c) any null result not yet in Discussion. These notes drive Stage C Pass 1.

---

## Stage C — Review and Revise

> Three structured passes in fixed order. After all three passes, check CHECKLIST. If < 95%, identify which items are missing and do a targeted fourth pass — do not restart.

### Pass 1 — Claims and evidence

- [ ] C.1.1 Read every section. For each numeric: confirm row in `numbers_audit.md`; run `make lint`. For each qualitative claim: confirm it is backed by a locked Epic 03 artifact or a citation. Remove or reframe any unsupported claim.
- [ ] C.1.2 Confirm Honda REFUTES claim is absent everywhere (audit Critical row).
- [ ] C.1.3 Confirm null/negative H1/H2/H3 findings from Epic 03 Stage C review notes are in `06_discussion.tex`. If a hypothesis was not supported, confirm the corresponding Experiments subsection says so explicitly and Discussion explains why.
- [ ] C.1.4 Confirm Abstract does not open with "we hypothesize."
- [ ] C.1.5 Confirm Epic 02 encoder-collapse null result is in Discussion/Limitations.
- [ ] C.1.6 Confirm system paper framing is consistent: Architecture section reads as system description, not as experimental setup.

### Pass 2 — Figures, tables, notation

- [ ] C.2.1 Each figure: vector PDF, 300 DPI fallback if raster, axis labels and units present, legend explicit, color-blind-safe palette.
- [ ] C.2.2 Each table: consistent decimal alignment; CIs alongside means; significance markers explained in captions.
- [ ] C.2.3 Notation check: grep source for `theta`, `Theta`, `\theta` variants; ensure `m_\Theta` / `m(\Theta)` is used consistently throughout.
- [ ] C.2.4 All equations numbered; all `\ref{}` and `\cite{}` resolve; `make pdf` zero warnings.

### Pass 3 — Related work and citations

- [ ] C.3.1 Confirm citations for all required works listed in spec §Related Work and §Background.
- [ ] C.3.2 Confirm the closing paragraph of Related Work explains Cognition's differentiators relative to cited works.
- [ ] C.3.3 Run `bibtex` and `latexmk --pdf` clean; zero unresolved.
- [ ] C.3.4 Optional: share draft with one external reader. Capture feedback in `paper/external_review_notes.md`; address before Stage D.

### Checklist gate

- [ ] C.4 Tick all applicable items in `paper/CHECKLIST.md`. Confirm ≥ 95% checked. If < 95%, identify which items remain and execute a targeted pass — do not proceed to Stage D until gate is met.

- [ ] **STAGE-C REVIEW:** three passes complete; CHECKLIST ≥ 95%; `make lint` passes; `make pdf` zero warnings; commit checkpoint `epic-04-stage-c-revised`.

---

## Stage D — Submit

- [ ] D.1 Final `make pdf` and `make lint` — both clean.
- [ ] D.2 `make arxiv-tarball`; confirm size ≤ 50 MB; run manifest validator.
- [ ] D.3 Submit to arXiv (cs.CL primary; cs.LG, cs.IR cross-list). Record arXiv ID, submission URL, timestamp, any warnings in `paper/SUBMISSION.md`.
- [ ] D.4 If submission fails, capture error in `paper/SUBMISSION.md`; fix; re-tarball; re-submit.
- [ ] D.5 Once arXiv ID assigned: update `09_reproducibility.tex` with real ID; rebuild; re-tarball if needed.
- [ ] D.6 Update `README.md` with arXiv preprint link and badge.
- [ ] D.7 Archive `docs/publication/draft2.txt` to `docs/publication/_archive/draft2_2026-04.txt` with a one-line note: "Superseded by arXiv:<id>."
- [ ] D.8 Author `docs/publication/phase4_paper.md`: arXiv ID, H1/H2/H3 outcome summary, pointers to all four phase memos, recommended next steps (conference submission targets, PyPI release planning).
- [ ] D.9 Tag commit `v1.0.0-paper` on branch `research`; push.
- [ ] **PHASE-BOUNDARY REVIEW Phase 4:** all acceptance criteria met; CHECKLIST fully checked; arXiv ID recorded; tag pushed. Close the publication-readiness program.
