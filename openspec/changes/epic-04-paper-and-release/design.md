# Design: Epic 04 — System Paper + arXiv

## Context

No code reorganization happens in this epic. `reference_v4/` is the research artefact; it is tagged but not restructured. The paper's reproducer installs it with `pip install -e "reference_v4/[study]"`. PyPI packaging is deferred to a separate future epic.

The paper is written from the locked Epic 03 result JSONs. Every number in the paper traces to a specific JSON key; no number is typed by hand.

## Decisions

### Decision: System paper structure, not hypothesis paper

Section sequence: Abstract → Introduction → Background → System Architecture → Implementation → Experiments → Discussion → Related Work → Conclusion → Reproducibility → System Card.

**Why:** The contribution is the system (cassette format, Kan migration, DS mass, MCTS + sheaf GNN). H1/H2/H3 are validation results supporting the system's claimed properties — they are not the paper's primary claim. System paper structure is standard for infrastructure contributions at NLP venues and is more resilient to mixed hypothesis results.

The Experiments section leads with "we evaluate three properties of the system" rather than "we test two hypotheses." The hypotheses are operationalised there, not in the Introduction.

---

### Decision: Tag v0.4.0-paper, no reorganization

`git tag v0.4.0-paper` on the current HEAD of `research` branch. The paper's Reproducibility section cites this tag. No files are moved or renamed.

**Why:** Reorganizing the repo (renaming `cognition` → `infon`, archiving `reference_v2/`, restructuring `src/`) introduces risk — imports break, tests fail, the reproducer path changes. None of this is necessary for the paper. The research community is used to installing research code from a git tag; they are not expecting a polished PyPI package from an arXiv preprint.

---

### Decision: reproducer.sh installs from reference_v4/ directly

```bash
pip install -e "reference_v4/[study]"
```

This installs the `cognition` package (not yet renamed to `infon`) from its current location. All benchmark scripts under `reference_v4/experiments/` then import from `cognition.*`.

**Why:** The simplest reproducer is the correct reproducer. Adding a package rename step to make the import say `from infon import ...` instead of `from cognition import ...` has zero scientific value and non-zero breakage risk.

---

### Decision: arXiv categories cs.CL primary, cs.LG + cs.IR cross-list

**Why:** Primary contribution is an NLP fact-checking/reasoning system (cs.CL). GNN and DS components are ML (cs.LG). Retrieval pipeline is IR (cs.IR). All three audiences will find it relevant.

---

### Decision: No mocks in tooling tests; LaTeX build is local-only

CI runs `pytest paper/tests/ -v` and `make lint` (numbers audit). It does not run `latexmk`. LaTeX compilation is a developer step before submission, not a CI gate.

**Why:** Full TeX Live install is ~4 GB; adds minutes to CI. The numbers-audit checker catches drift between paper text and result JSONs without building the PDF. The reproducer (full PDF build) is documented for local execution.

---

### Decision: Write all sections before beginning review passes

Stage B completes every section stub into full prose before Stage C begins any review pass.

**Why:** Reviewing a partial draft is inefficient — feedback on the introduction changes the framing that every other section depends on. Writing all sections first gives the reviewer a complete picture of argument flow, and the three Stage C passes can then be applied to the whole document rather than piecemeal.

---

### Decision: Review stages are structured, not open-ended

Stage C has exactly three passes in fixed order: (1) claims and evidence, (2) figures and notation, (3) related work and citations. Each pass is a checklist, not a judgment call. This prevents "review creep" — the paper is not revised indefinitely, it is revised to the point where the checklist is satisfied.

After Stage C, the only changes permitted before submission are: LaTeX formatting errors surfaced by `make pdf`, citation resolution failures from `bibtex`, and items flagged by the numbers-audit checker. Content changes require re-entering Stage C.

---

### Decision: Honest treatment of hypothesis outcomes is non-negotiable

If H2 shows Cognition's θ on NEI claims does not exceed the LLM baseline's θ, this is reported as a null result in Discussion. It does not change the Architecture section (the DS mass is still the mechanism) or the system's value (the cassette format and Kan migration are valid regardless). Suppressing or reframing a null result is explicitly prohibited.

This is documented as a design decision so it cannot be overridden during Stage C review by arguing "this makes the paper weaker." The paper is weaker without the null result only in the short term; it is much stronger in the long term.
