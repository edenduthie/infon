# Hyundai Mobis — First Customer Meeting Presentation (Outline)

**Meeting:** Friday 2026-05-15
**Presenter:** Eden Duthie, AgenticX
**Audience:** Hyundai Mobis (mixed technical / business; data engineering present)
**Goal:** Set up a credible 10-week PoC by showing (a) we understood the brief, (b) we have a system that fits it, (c) we have a sensible plan to deliver.

---

## Slide 0 — Title

- **안녕하세요** *(annyeonghaseyo — hello)*
- *Infon for Internal Knowledge — a Calibrated Knowledge Graph for Hyundai Mobis*
- Eden Duthie · AgenticX · 2026-05-15

---

## Slide 1 — Introduction (about me, about AgenticX)

- Eden Duthie — background in applied NLP and agent systems; author of the infon paper (Trusst.ai & AgenticX, May 2026).
- AgenticX — focus on grounded, auditable knowledge systems for the enterprise.
- One-line stance: *we build systems that cite their sources and tell you when they don't know.*

---

## Slide 2 — What we heard (replay of scope, single slide)

One slide, their words, mapped to seven bullets:

1. **Internal knowledge first** — search, summarise, reuse across teams; reduce duplicated research.
2. **Governance and structure** — metadata classification, security levels, author/topic, source traceability.
3. **Human-in-the-loop validation** — manual scoring and feedback loops in early stages.
4. **Document-centric** — unstructured docs (PPTs, reports, attachments), not only datasets.
5. **Historical tracking** — how a topic evolves over time, not one-shot summaries.
6. **Selective external data** — news, patents, subscription DBs complement internal.
7. **Measurable KPIs** — usage logs, search frequency, reuse, engagement → validate PoC.

> "Reducing duplicated research is the highest-stated priority." — quote back verbatim.

---

# Section 1 — Our approach (how infon works, tutorial-style)

*Goal: a data engineer with no prior context can follow this section. Diagrams over text. Keep math out of slides; reference the paper in the appendix.*

## Slide 3 — The problem with the obvious solution

- "Just put everything in ChatGPT / a RAG system" — two failure modes:
  - **Hallucination** — retrieves passages, fabricates the connection between them.
  - **No calibrated uncertainty** — sounds confident even when the corpus is silent.
- Knowledge-graph systems (the academic alternative) need rigid, hand-curated schemas, and can't say "we don't know."
- **What's missing**: a system that reasons across documents, *cites every step*, and tells you when the answer isn't in the corpus.

## Slide 4 — What infon is, in one sentence + one diagram

- **Infon = a knowledge-graph reasoner that fact-checks claims over your documents with calibrated uncertainty.**
- Diagram: Documents → Extracted facts (infons) → Cassette store → Reasoner → Cited verdict.
- Three properties that distinguish it:
  1. **Cassette-native storage** — every document becomes an immutable, content-addressed file with a full audit trail.
  2. **Honest uncertainty** — every answer carries a four-part mass `(supports, refutes, uncertain, θ)` where θ rises when the corpus has no evidence.
  3. **Multi-hop reasoning** — chains facts *across documents* using guided search, not flat retrieval.

## Slide 5 — The unit of knowledge: the *infon*

- An infon is a typed quadruple: `⟨ predicate, subject, object ; polarity ⟩`.
- Example: `⟨ supplies, LG_Energy_Solution, solid_state_cells ; +1 ⟩` extracted from a 2024 LG press release.
- Each infon is **grounded** to its source: document hash, byte offset, and timestamp.
- Diagram: a sentence in a PPT → highlighted span → extracted infon → arrow back to the source citation.

## Slide 6 — The cassette substrate (storage, simply)

- One document = one **cassette** (`.inf` file): immutable, content-addressed, append-only.
- Re-ingesting the same document is a no-op (idempotent).
- Sits on S3, any object store, or local disk — **same API**.
- Why engineers will care:
  - **Time-travel** — every ingest creates a new snapshot; old views remain queryable forever.
  - **Schema migration without reingestion** — change the ontology, rewrite the index in seconds, not hours (62× faster than re-extraction).

## Slide 7 — How a question gets answered (the pipeline, one diagram)

A horizontal flow with five boxes:

1. **Question** — plain English ("Who supplies solid-state cells to Hyundai?")
2. **Encode** — small bundled model (SPLADE-tiny, 17 MB, runs on CPU, no GPU, no API keys) turns the question into an anchor set.
3. **Prune** — manifest skips cassettes that *cannot* contain those anchors (7–16× fewer reads at scale).
4. **Reason** — guided multi-hop search (MCTS) walks the fact graph; a small frozen graph neural net scores chains.
5. **Verdict** — `(supports, refutes, uncertain, θ)` + every source citation that contributed.

## Slide 8 — Calibrated uncertainty in plain English

- Three outcomes per claim, with honest weights:
  - **Supports** — evidence in the corpus backs the claim.
  - **Refutes** — evidence in the corpus contradicts the claim.
  - **θ (theta)** — the corpus *doesn't speak to it*. This is the "I don't know" channel.
- Why this matters for Hyundai Mobis: an analyst running a query gets a *trustworthy abstention* when the answer isn't in your documents, instead of a fabricated-but-confident response.
- Single diagram: three bars (supports/refutes/θ) compared to a typical LLM's single "confidence" number.

## Slide 9 — Multi-hop reasoning, with an example

- Single-hop: "Does company A supply component X?" → answered from one document.
- **Multi-hop**: "Does company A supply component X *through any supplier*?" → answered by chaining facts across documents.
- Example walk-through (3 slides condensed to 1): A → partners with → B → develops → X. Each edge cites its source document.
- Why this matters: most useful Mobis questions are multi-hop — *partners of partners*, *supplier of a competitor's component*, *what changed since last review*.

## Slide 10 — The conversational layer (the *Analyst*)

- The user never writes queries. They ask in plain language.
- Diagram: chat input → Analyst agent → calls fixed tools (`ask`, `connect`, `any_of`, `extraction_report`) → returns a written answer + citation panel.
- **Hard constraints, built in**:
  - Never returns an answer without citing the sources `ask()` returned.
  - When θ > 0.7, explicitly says "the corpus does not answer this."
  - Cross-session memory: prior findings persist; the Analyst checks them at the start of each session.

## Slide 11 — Architecture at a glance (one diagram for the engineers)

Layered diagram (bottom-up):

- **Layer 0** — Documents (PPT, PDF, DOCX, XLSX, external feeds).
- **Layer 1** — Document preprocessor (format extraction → text + metadata).
- **Layer 2** — Cassette substrate (content-addressed `.inf` files + Parquet indexes; lives on S3 or local).
- **Layer 3** — Query DSL + Reasoner (Dempster–Shafer + MCTS + sheaf GNN).
- **Layer 4** — Human-in-the-loop validation queue.
- **Layer 5** — Access control + governance.
- **Layer 6** — KPI / audit / usage instrumentation.
- **Layer 7** — Analyst conversational UI + graph viewer.

Annotate which layers are **shipped today** vs **built during the PoC**.

---

# Section 2 — How this addresses your needs

*Goal: for each thing they said they wanted, show one concrete capability of infon and one example query. Keep one slide per requirement.*

## Slide 12 — Reduce duplicated research

- The problem: the same question is researched repeatedly across teams; reports sit unread.
- The capability: a single graph query returns every fact on a topic, *with every source document grouped*. A fact cited in 10 reports is stored once, returned with 10 citations.
- Example query: *"Show all internal knowledge on solid-state electrolyte suppliers, grouped by author team and date."*
- KPI implication: queries answered from internal knowledge vs queries that prompted new external research.

## Slide 13 — Source traceability (citations as a first-class object)

- Every infon carries: document hash, byte offset, ingestion timestamp, extraction confidence.
- The Analyst is *prompt-constrained* to never output a verdict without citing the `ask()` sources.
- Slide visual: an answer in the chat pane → click a citation → opens the exact passage in the source PPT/PDF.

## Slide 14 — Historical tracking (time-travel, native)

- Every ingest is a new snapshot. Nothing is ever rewritten.
- Example queries:
  - *"What did our internal documents say about Toyota's solid-state battery timeline in 2023 vs 2025?"*
  - *"Show the trajectory of how our cost estimate for LiDAR has changed over the last 18 months."*
- Slide visual: same anchor, two snapshots side-by-side, the divergence highlighted.

## Slide 15 — Honest uncertainty / "I don't know"

- The θ channel: when no evidence exists, the system *says so*, instead of guessing.
- Example: ask a question genuinely not in the corpus → analyst returns "the corpus does not contain a verified answer for this."
- Why it matters for governance: decisions in regulated/safety-critical contexts cannot be made on fabricated answers.

## Slide 16 — Surfacing contradictions

- The polarity channel (+1 affirmed / −1 negated) lets the system *find disagreements*.
- Example query: `where(s="supplier_X", p="meets", o="iso_26262").contradicting()` → returns documents where the answer disagrees.
- Use case: pre-decision risk review — find every internal claim that disagrees with the proposed conclusion.

## Slide 17 — Human-in-the-loop validation

- Low-confidence extractions go into a review queue. Domain experts see the source sentence, the extracted triple, the citation, and the confidence + reason.
- Three actions: **confirm**, **reject**, **correct**.
- Confirmed infons get `human_validated = True` and are treated as authoritative.
- Aggregate rejection patterns feed schema refinement — the system improves through use.
- Slide visual: review queue mockup (one row, three buttons).

## Slide 18 — Knowledge governance & metadata

- Document-level metadata is part of the cassette header: source, author, security classification, ingestion timestamp, schema version.
- Query-time filters on metadata + access control restrict the traversable cassette set by user role.
- Audit log: every query, every result, every validation action — timestamped and attributed.

## Slide 19 — Document-centric coverage

- Preprocessing layer (Layer 1, built in the PoC) handles: **PPTX, PDF, DOCX, XLSX**, with OCR fallback for image-embedded text.
- Speaker notes, table cells, chart titles, footnotes — all extracted.
- File metadata (author, dates, classification) carried into cassette headers automatically.

## Slide 20 — Selective external data

- External sources (patents, news, standards bodies) are ingested as separate cassettes with source-type and access-tier tags.
- Content-addressing deduplicates the same patent appearing in multiple feeds.
- Clear separation between confidential internal and public external content.

## Slide 21 — KPI & usage tracking

- Every query, result, and citation click is logged.
- PoC KPIs we propose to agree at engagement start:
  1. Validated facts in the graph (grows over time).
  2. Query response time vs estimated manual research time.
  3. Cross-document connections discovered.
  4. Duplication reduction (queries answered internally vs externally).
  5. Schema coverage (% of internal docs with ≥1 extracted infon).
- Slide visual: mock KPI dashboard.

## Slide 22 — Deployment posture (data security)

- Runs on commodity CPU. No GPU, no API keys, no external LLM call during extraction.
- Storage is S3-compatible — any object store, including air-gapped on-prem.
- Cassettes are portable: take the file, take the knowledge.
- Why this matters: a Hyundai Mobis deployment can be fully on-prem if required.

---

# Section 3 — Proposed 10-week PoC roadmap

*Iterative, risk-managed, every phase delivers a usable artefact.*

## Slide 23 — Roadmap philosophy

- **Iterative, not waterfall.** Every phase ends with something the client can actually use.
- **Schema-first.** A schema bootstrapped in week 1 from the Mobis glossary + a sample corpus — the system is useless without it, and it improves throughout the PoC.
- **Validation built in.** Human-in-the-loop is live by week 5 — feedback drives schema v2.
- **Honest scope.** The PoC is on a *curated* document set (150–300 docs). Full-corpus and external connectors are post-PoC.

## Slide 24 — The 10-week plan (one table)

| Weeks | Phase | Key deliverables | Milestone |
|-------|-------|------------------|-----------|
| 1–2 | **Bootstrap** | Sample corpus delivered (50–100 docs); glossary received; schema v1 drafted; preprocessor running on Mobis formats | First documents ingested under schema v1 |
| 3–4 | **Foundation** | 150–300 curated docs ingested; basic query API; first end-to-end hero query answered with citations | Live demo: a real Mobis question, answered with cited cross-document facts |
| 5–6 | **Governance + HITL** | Review queue UI; cassette-level security classification; role-based query filter; audit log | Domain experts using the review queue daily |
| 7–8 | **Synthesis + KPI** | LLM synthesis layer (prose answers over verified facts); KPI dashboard; schema v2 from review feedback | Validated-fact growth visible on the dashboard |
| 9 | **Stabilise** | Edge-case handling; schema v2 ingest; final KPI capture | Frozen system for evaluation week |
| 10 | **PoC report + decision** | Measured KPIs vs agreed targets; PoC report; production-contract recommendation | Joint go / no-go decision |

## Slide 25 — Week 1, in detail (what "iteration 1" actually looks like)

- Pre-engagement: receive sample corpus (50–100 docs) and internal glossary.
- Day 1–2: preprocessor running on Mobis sample docs; schema v1 drafted from glossary.
- Day 3–4: first ingest of sample; `extraction_report()` surfaces coverage gaps; schema patch.
- Day 5: first internal demo of a hero query on real Mobis content.
- This is the iteration cadence: ingest → report → refine → re-ingest.

## Slide 26 — What's in PoC scope vs deferred

**In PoC scope (10 weeks):**
- Document preprocessor (PPT/PDF/DOCX/XLSX).
- **Korean-language extraction** — Mobis documents are predominantly Korean. Encoder swap (multilingual or Korean-specific SPLADE variant) is week-1 work, not deferred.
- Schema v1 → v2 bootstrap + refinement (Korean anchor vocabulary from Mobis glossary).
- 150–300 curated internal documents ingested.
- HITL review queue (Korean-aware UI for reviewers).
- Security classification + role-based access.
- KPI dashboard + audit log.
- Live query UI (chat + graph viewer + citations).

**Deferred to post-PoC (production contract):**
- Full-corpus ingestion (thousands of docs).
- External connectors (patents, news, subscription DBs).
- SSO integration with Mobis IdP.
- Production-grade SLA / monitoring.

> Cut items are *scaling*, not *capability*. Every capability you'll evaluate the PoC on is in the 10-week scope.

## Slide 27 — Roles, ownership, working pattern

- **AgenticX** — schema design, ingestion, reasoner tuning, UI, KPI instrumentation.
- **Deloitte** — delivery partner; engagement management, on-site coordination with Mobis, change management, integration support.
- **Hyundai Mobis** — document curation (each phase), domain-expert reviewers (weeks 5–10), KPI target setting (week 1), final evaluation (week 10).
- Weekly checkpoint; schema review every two weeks; phase-end demo.

## Slide 28 — Risks and mitigations

| Risk | Mitigation |
|---|---|
| Schema v1 extracts poorly on real Mobis content | Run `extraction_report()` on week 1 sample; iterate before scaling ingest |
| Bundled SPLADE-tiny is English-trained; Korean documents need a different encoder | Week-1 task: swap to a multilingual or Korean-trained SPLADE variant (e.g. KoSPLADE / mSPLADE); validate anchor coverage on Korean sample corpus before scaling ingest |
| Korean tokenisation / morphological variation degrades anchor matching | Use a Korean-aware tokenizer in the encoder; build the schema's anchor vocabulary from the Mobis Korean glossary; verify with `extraction_report()` |
| Security review blocks document access | Start week 1 with a "safe" curated subset already agreed |
| Document corpus heterogeneity (formats, quality) | Curated 150–300 docs, not full corpus, for PoC |
| KPI targets disputed at week 10 | Agree targets *in writing* in week 1 |

## Slide 29 — Immediate next steps (the soft close)

1. **Scoping call** — agree corpus scope, security requirements, integration points (SAP, PLM, document store), KPI definitions.
2. **Sample corpus** — 50–100 representative documents across key knowledge areas.
3. **Internal glossary / taxonomy** — starting point for schema anchors.
4. **PoC success criteria in writing** — measurable KPIs with target values.
5. **Kick-off** — week 1 starts as soon as items 1–4 are agreed.

---

## Appendix slides (kept off the main deck, used in Q&A)

- A1 — Benchmark numbers (HoVer multi-hop, AVeriTeC NEI, SciFact calibration) for the technical track.
- A2 — Sheaf GNN internals (for the "how does the GNN actually work" question).
- A3 — Schema migration without reingestion (technical detail).
- A4 — On-prem / air-gapped deployment specifics.
- A5 — The paper — *Infon: A Knowledge Graph Reasoner with Calibrated Uncertainty*, Duthie et al. 2026 — preprint reference.

---

## Format & open decisions

**Confirmed format (2026-05-14):**
- **Video conference via Microsoft Teams.** No in-person component.
- **Total session: 1.5 hours.** Deloitte presents first; AgenticX slot is **~45 minutes**.
- **Presentation only — no live demo on Friday.** Static screenshots and mockups only.
- **No pre-recorded clips.** Not enough time to produce one of sufficient quality before Friday.
- **No side-by-side ChatGPT comparison clip** — dropped from slide 15.
- **Korean greeting** on the title slide — confirmed (slide 0).

**Implications for the deck:**
- Section 1 (slides 7, 10, 11) — show pipeline diagrams and static screenshots, not live runs.
- Section 2 (slides 12–17) — embed mock answer screenshots with citations rendered as cards instead of live chat output.
- Slides 14, 16, 17 — use annotated screenshots (time-travel snapshot pair, contradiction surfacing, HITL review row).
- No streamlit-agraph / PyVis viewer needed for Friday; that work moves to the PoC phase or a follow-up demo session.

**Pacing — 45 min for ~30 slides is tight.** Reserve ~10 min for Q&A → ~35 min of presentation → ~70 sec/slide. Two options to consider before the final draft:
- **Trim:** drop or merge 5–8 slides. Candidates: merge slides 5+6 (infon + cassette); merge slides 18+19 (governance + document coverage); fold slide 22 (deployment) into appendix; drop slide 27 if roles can be covered in a sentence on slide 24. Target ~22 slides + appendix.
- **Keep all and pace tightly:** rehearse a 35-min run; cut at the door rather than in the outline.

**Still open:**
- [ ] Final attendee list and roles — tailor the "what week 1 looks like" slide (slide 25) to whoever owns budget approval.
- [ ] Decide between *trim* vs *pace tightly* (above).
- [ ] Coordinate with Deloitte's slot — does their content overlap (e.g. they may already cover engagement-management roles on slide 27)?
