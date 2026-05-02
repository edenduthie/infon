# Design — Epic 04: Publication-ready paper + arXiv submission

## Context

The paper synthesizes outputs from Epics 01–03 into an arXiv preprint. Two non-trivial design decisions shape this epic: how the paper is structured around the two hypotheses, and how the reproducer is wired so that "the paper's number" is unambiguous.

The audit (`docs/publication/reproduction_audit.md`) identified concrete failure modes from `draft2.txt` (the prior draft):

- Numbers in the paper did not match any seeded run of the released code.
- The Honda REFUTES claim was factually inconsistent with the corpus.
- Verdict-calibration numbers may have been measured pre-training or against the teacher.
- No ablation table; no baselines; no calibration metrics.

Every one of those failure modes is a structural problem, not a polish problem; the reproducer-and-numbers-audit machinery in this epic exists specifically to prevent recurrence.

## Goals / Non-Goals

- **Goals**: arXiv-ready preprint; deterministic reproducer; numbers-audit trail; honest reporting including null results; clean LaTeX build; arXiv tarball.
- **Non-Goals**: peer-review submission (separate follow-up); response-to-reviewers cycle; venue-specific formatting beyond arXiv-acceptable; supplementary material beyond what the figures need; press release.

## Decisions

### Decision: Paper structure follows the H1/H2 contract

The paper is structured exactly around the user's request:

```
1. Abstract
2. Introduction         — motivation + the two hypotheses, stated explicitly
3. Background           — infon, hypergraph, Dempster-Shafer, IKL operators (kept tight; cite the existing draft heavily)
4. Method               — architecture, training, readout, fusion (canonical config from Epic 01)
5. Hypotheses           — H1 + H2 stated with operational definitions and pre-registered metrics
6. Current baseline     — what the system does before this study (Epic 01 reproduction state)
7. Datasets             — synthetic generator (Epic 02) + three public benchmarks (Epic 03)
8. Baselines            — six baselines + reproduced numbers (Epic 03)
9. Results              — H1 panel; H2 panel; ablation table; Pareto front
10. Discussion          — what the evidence shows; what it does not; limitations
11. Related work        — RAG abstention; selective prediction; EDL; GNN fact verification; KGAT/DREAM/GEAR family
12. Conclusion
13. Reproducibility     — pointer to reproducer.sh and seed list
References
```

**Why:** the user explicitly wants "2 hypotheses, current baseline, evidence" structure. Forcing the table of contents to mirror that contract removes the chief failure mode of `draft2.txt` — claims drifting away from the evidence that supports them.

---

### Decision: Numbers-audit table is mandatory

`paper/numbers_audit.md` is a table mapping every in-text number to: the page it appears on, the figure/table/equation it appears in, and the JSON file under `experiments/results/` that produced it. Building the paper without this table is a CI failure.

**Why:** the audit's central finding was "the paper's numbers are mutually consistent but unreproducible from the code." A numbers-audit table makes orphan numbers physically impossible — every digit in the PDF must point at a JSON.

---

### Decision: Reproducer is one shell script, not a notebook

`paper/reproducer.sh` is a plain bash script that runs a sequence of `python -m reference_v2.experiments.*` commands plus figure-script invocations plus `make pdf`. No Jupyter notebook in the critical path.

**Why:** notebooks are flaky for reproduction (kernel state, cell ordering, environment drift). A shell script with idempotent stages is the lowest-friction reproducer. We intentionally inherit the AVeriTeC 2025 reproducibility ethos: one script, one machine, one config.

---

### Decision: Honest reporting of null/negative findings

If H1 or H2 produces a null or negative result on real data (Epic 03), the paper reports it. The Discussion section is where we frame what the result means; the Results section is where we report the number.

**Why:** if we suppress null results we end up writing another `draft2.txt`. The Honda example is a cautionary tale — claims that look better than the data are not just unethical, they are eventually caught and they sink the paper harder than the original honest finding would have.

---

### Decision: Three review passes, in fixed order

Stage C is three discrete review passes:

1. **Content & claim audit** — every claim cites evidence; cross-reference against `paper/numbers_audit.md`; remove anything unsupported. Includes removing the Honda REFUTES claim from any inherited draft.
2. **Figures, tables, notation** — figure quality (300 DPI, vector PDFs preferred), table formatting, mathematical notation consistency, units.
3. **Related work, citations, prior art** — full BibTeX scrub; check for missing close-prior-art (the AVeriTeC, FEVER, HoVer, KGAT/DREAM, EDL, Sufficient-Context lineages); reach out to one external reader if budget permits.

**Why:** mixing these passes produces churn — fixing notation distracts from claim audits and vice versa. Doing them in this order means content is locked before figures are polished.

---

### Decision: arXiv tarball validated via the official manifest

`arxiv-submission/build.sh` produces a `.tar.gz` containing exactly: `main.tex`, every `\input`-ed `.tex` file, `references.bbl` (the resolved bibliography), all referenced figure files (PDF or PNG), and a top-level `00README.txt` per arXiv submission convention. The script validates: no `.bib` (only `.bbl`), no auxiliary files, total size ≤ 50 MB.

**Why:** arXiv quietly rejects submissions for many small reasons (missing `.bbl`, oversize, bad encoding). Validating against a deterministic manifest catches them before submission.

---

### Decision: Paper authorship + license

Authors: as listed in CHANGELOG/AGENTS at submission time. License: arXiv default (non-exclusive license to distribute). Code license unchanged (Apache-2.0). Paper-source license: CC BY 4.0 in `paper/LICENSE`.

**Why:** CC BY 4.0 is standard for research preprints and is compatible with arXiv. Apache-2.0 stays on the code.

## Risks / Trade-offs

- **arXiv submission rejection on first try**: format/file issues are common. Mitigation: dry-run the tarball through arXiv's offline validator (where available) before live submission; document failure modes in `paper/SUBMISSION.md`.
- **Numbers-audit drift**: as the paper evolves, in-text numbers may diverge from JSON. Mitigation: a CI job (`pytest paper/test_numbers_audit.py`) that reads the audit table and asserts each number appears verbatim in the corresponding JSON within a documented tolerance.
- **LaTeX environment drift**: `latexmk` works on author's machine but fails on CI. Mitigation: pin a TeX Live version or run via a `texlive/texlive` Docker image.
- **Honest null results may bury the paper's contribution**: we have to make sure the paper's framing handles a "we discovered X does not work but here is why and here is the design that does" story gracefully. The Discussion section is where this is done.
- **arXiv ID is forever**: a v1 with errors is hard to retract — only superseded. Mitigation: do not submit until CHECKLIST is fully green and `paper/numbers_audit.md` validates in CI.

## Open Questions

- Do we cross-list to `cs.AI` in addition to `cs.CL` and `cs.LG`? **Tentatively yes** (broadens visibility); decide at submission time.
- Do we include a model card / system card as supplementary? **Yes, brief** — one page covering intended use, limitations, evaluation gaps, in `paper/sections/system_card.tex`.
- Do we include a Broader Impact section? **Yes, short** — calibrated abstention has dual-use considerations (epistemic-confidence misuse) worth a paragraph.
