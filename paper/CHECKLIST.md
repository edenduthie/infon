# Paper Submission Checklist

- [ ] All in-text numbers traced to a JSON (`make lint` passes)
- [ ] `paper/reproducer.sh` runs end-to-end on a clean checkout
- [ ] Honda REFUTES claim absent (audit Critical row #2)
- [ ] Null/negative Epic 03 findings in Discussion (not suppressed)
- [ ] Abstract does not open with "we hypothesize"
- [ ] Epic 02 encoder-collapse null result in Limitations
- [ ] System paper framing consistent throughout
- [ ] All figures: vector PDF, axis labels, color-blind-safe palette
- [ ] All tables: CIs reported, significance markers explained
- [ ] Notation consistent: `m(S), m(R), m(Θ)` throughout
- [ ] Zero unresolved LaTeX references (`make pdf` zero warnings)
- [ ] All citations resolve (`bibtex` clean)
- [ ] Reproducibility section cites tag `v0.4.0-paper`
- [ ] System card included
- [ ] arXiv tarball ≤ 50 MB, passes manifest check
- [ ] Author list, affiliations, contact email confirmed
- [ ] License `paper/LICENSE` is CC BY 4.0
