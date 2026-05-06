# Spec: Epic 04 — System Paper + arXiv

## Hard Rules

- **TDD for tooling only:** figure scripts, reproducer, numbers-audit checker, arXiv tarball builder are tested before implementation. Paper prose is reviewed, not unit-tested.
- **No Mocks:** tooling tests use real figure libraries, real JSON parsing, real reproducer invocations on fixture data.
- **No Orphan Numbers:** every numeric in the rendered PDF traces to a locked Epic 03 JSON via `paper/numbers_audit.md`. `make lint` fails when audit does not validate.
- **No Unsupported Claims:** every claim is (a) backed by a locked Epic 03 artifact, (b) a published citation, or (c) explicitly framed as discussion/opinion.
- **Honest Reporting:** null and negative findings appear in Discussion. No result is suppressed or reframed to appear stronger.
- **Stage-boundary review:** at each stage's end, run `pytest paper/tests/ -v` and `make lint`; re-read this spec; verify compliance before proceeding.

---

## ADDED Requirements

### Requirement: Paper Tag — No Code Changes

Before any paper work begins, tag the current repo state:

```bash
git tag v0.4.0-paper
git push origin v0.4.0-paper
```

No files in `reference_v4/` are moved, renamed, or modified as part of this epic. The tag is the reproducible anchor for the paper.

**Testability:** `git tag --list` contains `v0.4.0-paper`; `git diff v0.4.0-paper HEAD -- reference_v4/` is empty at epic close.

---

### Requirement: Paper Section Sequence

`paper/main.tex` SHALL `\input` sections in this exact order:

```
00_abstract, 01_introduction, 02_background, 03_architecture,
04_implementation, 05_experiments, 06_discussion,
07_related_work, 08_conclusion, 09_reproducibility, system_card
```

**Testability:** text parse of `main.tex`; assert ordered `\input` sequence.

---

### Requirement: Abstract

`paper/sections/00_abstract.tex` SHALL be ≤ 150 words. It SHALL:
- Open with what Cognition is (a knowledge graph reasoner) and what it does (extracts calibrated claims from text, answers with explicit uncertainty)
- Name the three headline system properties (cassette storage, honest θ, MCTS multi-hop)
- Include the top quantitative result from H2 (θ value on NEI claims vs LLM baseline)
- State code availability (GitHub tag)

It SHALL NOT open with "we hypothesize" or "in this paper we propose."

---

### Requirement: Introduction

`paper/sections/01_introduction.tex` SHALL cover: the problem (fact-checking over document corpora; existing systems either hallucinate on unsupported claims or require rigid schemas); the gap (no system combines calibrated abstention, multi-hop reasoning, and schema-free operation in a single deployable unit); the contribution (Cognition, listed as five bullet properties from `reference_v4/README.md` introduction); and a roadmap paragraph pointing to each subsequent section.

---

### Requirement: Background

`paper/sections/02_background.tex` SHALL establish the notation and theory used in the rest of the paper. Required subsections:
- **Dempster–Shafer theory:** define frame of discernment, basic probability assignment, vacuous element Θ, Dempster's combination rule, and the four-element mass `(m_S, m_R, m_U, m_Θ)`. This is the notation used throughout.
- **Situation semantics and infons:** Barwise & Perry (1983); define infon triple `⟨⟨pred, subj, obj; pol⟩⟩`; grounding to source.
- **SPLADE sparse retrieval:** Formal et al. (2021); how sparse activations map text to anchor vocabulary.
- **Sheaf GNNs:** Bodnar et al. (2022); restriction maps; H¹ cohomology as anomaly signal.

---

### Requirement: System Architecture

`paper/sections/03_architecture.tex` is the primary technical contribution section. Required subsections:

**Cassette substrate:** immutable append-only format (8-byte magic, JSON header, gzip frames, Parquet indexes); manifest pruner and bbox skip (measured: 7–16× at 300 cassettes, 278× on misses); time-travel via snapshot chain; delta ingest idempotency.

**Query DSL:** primitives (`where`, `mentioning`, `affirmed`, `negated`, temporal windows, hierarchy expansion); constraint pushdown; `run_any` for logical OR; NEXT-edge trajectory; measured: 500-infon compliance query resolves to 0 range gets.

**Reasoner:** per-infon DS mass construction from four sources (polarity, alignment, distance, confidence); top1 cautious fusion (canonical from Epic 01: `top1, k=2, cw=1.0`); MCTS chain traversal with polarity-aware `chain_mass` (conjunctive min/max, not Dempster); sheaf GNN chain-verdict head as MCTS prior.

**Schema migration:** `SchemaFunctor(rename, merge, delete)`; Kan pushforward; measured: 20ms migration vs seconds for reingest; time-travel intact post-migration.

---

### Requirement: Implementation

`paper/sections/04_implementation.tex` SHALL cover: SPLADE-tiny encoder (17 MB, no GPU, Apache 2.0, frozen); anchor type system (actor/relation/feature/market); extraction pipeline (activation thresholding, role assignment, dual-partition actor reranking); executor variants (SyncExecutor for testing, ProcessExecutor for production, Lambda-compatible container); Strands Analyst (9 tools, cross-session findings memory, schema bootstrap loop). Include the one-minute start code block from `reference_v4/README.md`.

---

### Requirement: Experiments Section

`paper/sections/05_experiments.tex` SHALL contain:

1. **Evaluation setup subsection:** datasets (HoVer, AVeriTeC v2, SciFact — brief provenance, claim counts, label distributions); evaluation surface (per-claim InfonStore, zero-shot schema bootstrap); systems evaluated (Cognition symbolic, Cognition+GNN, flat retrieval ablation, symbolic floor, NLI classifier, LLM zero-shot); metrics (polarity accuracy, ECE, AURC, Spearman ρ).

2. **H1 subsection (multi-hop advantage):** reproduce the HoVer depth-stratified accuracy table from `h1_panel.json`; include the `accuracy_by_hop` figure; state the H1 effect size and its CI; state the finding honestly (supported/partially supported/not supported).

3. **H2 subsection (honest abstention):** reproduce the AVeriTeC θ distribution stats from `h2_panel.json`; include the `theta_distribution` figure; state ρ(m_Θ, nei_indicator) for each system; report the false-positive endorsement rate comparison; state the finding.

4. **H3 subsection (DS calibration):** reproduce the ECE/Brier/AURC table from `h3_panel.json`; include reliability diagrams; state H3 effect size; state the finding.

5. **Baseline reproduction gate table** at the top of the section (compact; confirms baselines are correctly implemented).

Each subsection MUST state its finding honestly regardless of sign. If a hypothesis is not supported, the subsection says so and Discussion explains why.

---

### Requirement: Discussion

`paper/sections/06_discussion.tex` SHALL contain:

- Interpretation of each H1/H2/H3 finding, including null results
- The Epic 02 encoder-collapse null result (one paragraph in Limitations): BERT entity collapse in template text; what a valid synthetic harness requires; why it doesn't affect the v4 system
- Schema coverage failure rate from the Epic 03 coverage report; what it means for generalizability
- Future work: cross-cassette GNN training, multimodal infons, learned manifest pruning, Schema autodiscovery without spectral clustering

---

### Requirement: Related Work

`paper/sections/07_related_work.tex` SHALL cover and cite: KGAT, GEAR, DREAM (graph-based fact verification); GraphCheck, STRIVE, AFEV (2025 contemporaries); SelectLLM (ICLR 2025); Sufficient-Context selective generation (Joren et al., ICLR 2025); evidential deep learning (Sensoy 2018); E-NER (ACL Findings 2023); R-GCN (Schlichtkrull et al. 2018); FEVER dataset and NLI lineage; AVeriTeC shared task (2024–2025).

The section SHALL end with a paragraph explaining what Cognition offers that none of the above provide: schema-free operation, immutable cassette storage with time-travel, and DS mass with explicit Θ as a first-class output.

---

### Requirement: Reproducibility and System Card

`paper/sections/09_reproducibility.tex` SHALL state: GitHub repo URL, tag `v0.4.0-paper`, install command (`pip install -e "reference_v4/[study]"`), reproducer invocation (`bash paper/reproducer.sh`), expected wall-clock time, compute used.

`paper/sections/system_card.tex` SHALL state: intended use (research fact-checking, knowledge graph reasoning over text corpora); out-of-scope use (real-time systems, life-critical decisions); known limitations (schema coverage failures, SPLADE vocabulary mismatch on highly technical domains, transductive GNN not scalable beyond ~5K-node graphs); training data (synthgen for GNN, no personal data).

---

### Requirement: Numbers Audit Trail

`paper/numbers_audit.md` SHALL be a markdown table with columns `value | section | location | json_path | json_key | tolerance | description`. Every numeric in the paper's prose and tables (excluding equations and axis ticks) SHALL have a row.

`paper/tools/check_numbers_audit.py` SHALL parse the table, load each JSON, look up the key, and assert the value matches to tolerance. `make lint` invokes this checker. CI fails on any miss.

#### Scenario: Drift detected
- **GIVEN** a row pointing at a JSON containing `{"acc": 0.87}` with tolerance `0.005` and in-text value `87.0%`
- **WHEN** the validator runs
- **THEN** it passes

#### Scenario: Validator fails on real drift
- **GIVEN** the JSON is updated to `{"acc": 0.72}` without updating the paper
- **WHEN** the validator runs
- **THEN** it exits non-zero naming the offending row and actual delta

**Testability:** `paper/tests/test_numbers_audit.py`; pass and fail cases.

---

### Requirement: Deterministic Reproducer

`paper/reproducer.sh` stages:
1. `pip install -e "reference_v4/[study]"`
2. Run Epic 03 evaluation matrix (checkpoint-resumable)
3. Generate H1/H2/H3 panel JSONs
4. Regenerate all figures
5. `make lint`
6. `make pdf` (local only)

Accepts `--dry-run`; idempotent (re-running skips completed stages).

---

### Requirement: Pre-Submission Checklist

`paper/CHECKLIST.md` SHALL include:

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

≥ 95% checked before Stage D begins.

---

### Requirement: arXiv-Compliant Submission Tarball

`paper/arxiv-submission/build.sh` produces a `.tar.gz` containing `main.tex`, all `\input`-ed `.tex` files, `references.bbl` (not `.bib`), every figure referenced by `\includegraphics`, `00README.txt`. SHALL NOT contain `.aux`, `.log`, `.out`, `.bib`, `.synctex.gz`, hidden files. Size ≤ 50 MB.

---

### Requirement: Honest Treatment of Hypothesis Outcomes

The paper SHALL report H1/H2/H3 outcomes faithfully:
- If H1 shows MCTS + GNN significantly outperforms flat retrieval at depth ≥ 3: state this.
- If it does not: state the absence; discuss in Discussion (insufficient depth variation in dataset, or genuine null).
- If H2 shows Cognition θ on NEI > LLM θ: state this; quantify the gap.
- If H2 is not supported: remove the H2 claim from Experiments; report as null in Discussion.
- If H3 shows Cognition ECE < NLI classifier: state this.
- If H3 is reversed: report the actual ordering.

In all cases: the Architecture section is valid regardless of experimental outcomes.
