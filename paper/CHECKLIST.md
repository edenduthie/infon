# Paper Submission Checklist

- [x] All in-text numbers traced to a JSON (`make lint` passes)
- [x] `paper/reproducer.sh` runs end-to-end on a clean checkout
- [x] Honda REFUTES claim absent (audit Critical row #2)
- [x] Null/negative Epic 03 findings in Discussion (not suppressed)
- [x] Abstract does not open with "we hypothesize"
- [x] Epic 02 encoder-collapse null result in Limitations
- [x] System paper framing consistent throughout
- [x] All figures: vector PDF, axis labels, color-blind-safe palette
- [x] All tables: CIs reported, significance markers explained
- [x] Notation consistent: `m(S), m(R), m(Θ)` throughout
- [x] Zero unresolved LaTeX references (`make pdf` zero warnings)
- [x] All citations resolve (`bibtex` clean)
- [x] Reproducibility section cites tag `v0.4.0-paper`
- [x] System card included
- [x] arXiv tarball ≤ 50 MB, passes manifest check
- [x] Author list, affiliations, contact email confirmed
- [x] License `paper/LICENSE` is CC BY 4.0
