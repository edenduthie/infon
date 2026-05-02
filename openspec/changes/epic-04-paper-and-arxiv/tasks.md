# Tasks: Epic 04 — Publication-ready paper + arXiv submission

> **Epic status (created 2026-05-02):** blocked by Epic 03. Final phase. Acceptance criterion in `proposal.md`.

## Development Rules

- **Test-First (where applicable):** the figure scripts, reproducer, numbers-audit checker, and tarball builder are all tested via integration tests before they are written. The paper text itself is *content*, not tested code; reviews replace tests for prose.
- **No Mocks:** tests run real LaTeX builds, real figure regenerations, real reproducer invocations.
- **No Orphan Numbers:** every numeric value in the paper text traces to a JSON via `paper/numbers_audit.md`.
- **Stage-Boundary Review:** at each stage's end, run `pytest -v` plus the LaTeX build plus the numbers-audit checker; verify spec compliance; update beads.

---

## Stage A — Code (paper scaffold and tooling)

### LaTeX skeleton

- [ ] A.1 Create `paper/` with: `main.tex`, `references.bib`, `Makefile`, `LICENSE` (CC BY 4.0), `README.md`. Use a clean two-column or single-column class compatible with arXiv (e.g. `article` or `acmart` non-anon).
- [ ] A.2 Create section stubs under `paper/sections/`: `00_abstract.tex`, `01_introduction.tex`, `02_background.tex`, `03_method.tex`, `04_hypotheses.tex`, `05_baseline.tex`, `06_datasets.tex`, `07_baselines.tex`, `08_results.tex`, `09_discussion.tex`, `10_related_work.tex`, `11_conclusion.tex`, `12_reproducibility.tex`, `system_card.tex`.
- [ ] A.3 Configure `Makefile` with `pdf`, `clean`, `lint`, `arxiv-tarball` targets. `make pdf` SHALL be `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

### Figure scripts (regenerate from JSON)

- [ ] A.4 Write `paper/tests/test_figure_pareto.py` first: load a fixture aggregate JSON; invoke `paper/figures/pareto.py --in <fixture> --out /tmp/pareto.pdf`; assert the output PDF exists and has size > 1 KB. **Verify red.**
- [ ] A.4b Implement `paper/figures/pareto.py` (reuses Epic 03's `pareto_front.py` logic). **Verify green.**
- [ ] A.5 Repeat A.4/A.4b pattern for: `accuracy_by_hop.py` (HoVer H1 panel), `theta_vs_thinness.py` (H2 scatter), `reliability_diagrams.py` (multi-readout grid), `risk_coverage.py` (selective accuracy).
- [ ] A.6 Write `paper/tests/test_figures_deterministic.py`: regenerate every figure twice from the same JSON; assert byte-identical PDFs (or near-identical via PDF text-extraction comparison). **Verify red → green.**

### Reproducer

- [ ] A.7 Write `paper/tests/test_reproducer_dry_run.py` first: invoke `paper/reproducer.sh --dry-run`; assert it lists every command it would run; assert each command's invariants are documented. **Verify red.**
- [ ] A.7b Author `paper/reproducer.sh`: stages = (i) Epic 01 canonical run; (ii) Epic 02 ablation cells (the 8 named); (iii) Epic 03 evaluation matrix; (iv) figure regeneration; (v) `make pdf`. Each stage idempotent and resumable. **Verify green.**

### Numbers audit

- [ ] A.8 Write `paper/tests/test_numbers_audit.py` first: feed a `numbers_audit.md` row pointing to a fixture JSON containing `{"value": 33.17}` and an in-text claim "AVeriTeC 2025 score 33.17"; assert the validator passes. Feed a row whose target JSON disagrees by 0.5 pp; assert the validator fails with a precise diff. **Verify red.**
- [ ] A.8b Implement `paper/tools/check_numbers_audit.py` parsing `numbers_audit.md` rows, looking up each value in the named JSON, and asserting equality to documented tolerance. **Verify green.**
- [ ] A.9 Add `make lint` invoking the numbers-audit checker plus a `latex-check-warnings` script that fails on undefined references or warnings.

### arXiv submission tarball

- [ ] A.10 Write `paper/tests/test_arxiv_tarball.py` first: invoke `paper/arxiv-submission/build.sh`; assert the produced tarball contains `main.tex`, all `\input`-ed sections, `references.bbl`, every referenced figure file, no `.bib`, no `.aux`/`.log`/`.out`; total size ≤ 50 MB. **Verify red.**
- [ ] A.10b Implement `paper/arxiv-submission/build.sh`. **Verify green.**

- [ ] **STAGE-A REVIEW:** run all paper tests; run a minimal `make pdf` (template only); commit checkpoint `epic-04-stage-a-complete`.

## Stage B — Run (regenerate every number from the locked JSONs)

- [ ] B.1 Run `paper/reproducer.sh` (full, not `--dry-run`) on a clean checkout. Wall-clock estimate: 2–6 days; stage failures resume.
- [ ] B.2 Verify every figure under `paper/figures/output/` exists and matches the figure listing in `main.tex`.
- [ ] B.3 Inspect each generated table: H1 panel, H2 panel, ablation matrix, baseline-reproduction gate, Pareto front data table.
- [ ] B.4 Author `paper/numbers_audit.md` listing every numeric value that appears in the draft (page, location, JSON path, key, tolerance). Run `paper/tools/check_numbers_audit.py`; iterate until green.

## Stage C — Review and iterate (three passes, in order)

### Pass 1 — Content & claim audit

- [ ] C.1.1 Read every section. For each numeric or qualitative claim, ensure it is supported by Epic 01/02/03 evidence and (if numeric) appears in `numbers_audit.md`. Remove unsupported claims; reframe in Discussion if needed.
- [ ] C.1.2 Explicitly delete the Honda REFUTES claim if it survived from `draft2.txt` (audit Critical row). Replace with the corpus-correct SUPPORTS verdict and adjust narrative.
- [ ] C.1.3 If Epic 01 retracted H2, rewrite §4 (Hypotheses) and §8 (Results) accordingly: H2 becomes a documented null finding, paper's contribution narrows to H1 + interpretability.
- [ ] C.1.4 Cross-check that the "current baseline" section (§5) honestly describes the pre-study state.
- [ ] C.1.5 Verify that null/negative findings from Epic 03 are reported, not buried.

### Pass 2 — Figures, tables, notation

- [ ] C.2.1 Each figure: 300 DPI minimum, vector PDF preferred, axis labels and units present, legend explicit, color-blind-safe palette.
- [ ] C.2.2 Each table: consistent decimal alignment; CIs reported alongside means; significance markers explained in caption.
- [ ] C.2.3 Notation: pick one notation set (e.g. `m(S), m(R), m(U), m(Θ)` per draft1 — confirm consistency with Background section); search the source for inconsistent variants.
- [ ] C.2.4 Equations numbered; cross-references resolve.

### Pass 3 — Related work, citations, prior art

- [ ] C.3.1 Confirm citations for: AVeriTeC v2 shared task (2024 + 2025 ACL); HoVer (Findings-EMNLP 2020); SciFact (EMNLP 2020); FEVEROUS (NeurIPS 2021); KGAT/GEAR/DREAM (ACL 2020); GraphCheck/STRIVE/AFEV (2025); Sensoy 2018 EDL; E-NER ACL Findings 2023; Joren et al. *Sufficient Context* ICLR 2025; SelectLLM (ICLR 2025 OpenReview); Brier 1950, Dempster 1967, Shafer 1976, Yager 1987, Murphy 2000.
- [ ] C.3.2 Add Barwise & Perry 1983 (infon), Hayes & Menzel 2006 (IKRIS), Schlichtkrull et al. 2018 (R-GCN), Formal et al. 2021 (SPLADE) — already in `draft2.txt`, port over.
- [ ] C.3.3 Run `bibtex` and `latexmk` clean; verify zero unresolved citations.
- [ ] C.3.4 Optional: send to one external reader for sanity feedback before submission. Capture feedback in `paper/external_review_notes.md`; address before submission.

- [ ] **STAGE-C REVIEW:** all three passes complete; CHECKLIST (`paper/CHECKLIST.md`) ≥ 95% checked; numbers-audit passes; LaTeX builds with zero warnings.

## Stage D — Finalize and submit

- [ ] D.1 Final `make pdf` and `make lint` — both clean.
- [ ] D.2 `make arxiv-tarball` — produces `arxiv-submission/<arxiv_id_placeholder>.tar.gz`; size ≤ 50 MB; passes manifest validation.
- [ ] D.3 Submit to arXiv via the web UI (cs.CL primary; cross-list cs.LG and optionally cs.AI). Record the assigned arXiv ID, submission URL, and any reviewer warnings in `paper/SUBMISSION.md`.
- [ ] D.4 If submission fails (formatting/file issues), capture exact error in `paper/SUBMISSION.md`, fix, re-tarball, re-submit. Do not amend the timestamp on the original submission.
- [ ] D.5 Once arXiv ID assigned: update `README.md`, `CHANGELOG.md`, `docs/publication/phase4_paper.md` (this epic's memo), and the project's main documentation site to link the preprint URL.
- [ ] D.6 Archive the prior `docs/publication/draft2.txt` to `docs/publication/_archive/draft2_2026-04.txt` with a one-line preamble noting it is superseded by the arXiv preprint.
- [ ] D.7 Tag commit `v1.0.0-paper`; push.
- [ ] **PHASE-BOUNDARY REVIEW Phase 4:** all acceptance criteria from `proposal.md` met; CHECKLIST fully checked; `numbers_audit` clean; arXiv ID recorded; mark `epic-04-paper-and-arxiv` complete in beads. Close out the publication-readiness program.
