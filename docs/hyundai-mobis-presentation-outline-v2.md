# Hyundai Mobis — First Customer Meeting Presentation (Outline, v2 — trimmed)

**Meeting:** Friday 2026-05-15 · Microsoft Teams · 1.5h total session, Deloitte first, AgenticX slot ~45 min
**Presenter:** Eden Duthie, AgenticX
**Audience:** Hyundai Mobis (mixed technical + business; data engineering present)
**Format:** Presentation only — no live demo, no pre-recorded clips.
**Goal:** Set up a credible 10-week PoC by showing (a) we understood the brief, (b) we have a system that fits it, (c) we have a sensible plan to deliver.

**v2 changes from v1:** Section 2 reduced from 11 slides to 3 (top benefits only). Section 1 tightened by merging infon + cassette. Section 3 tightened. Closing summary slide added. Total: **22 slides + appendix**.

---

## Slide 0 — Title

- **안녕하세요** *(annyeonghaseyo — hello)*
- *Infon for Internal Knowledge — a Calibrated Knowledge Graph for Hyundai Mobis*
- Eden Duthie · AgenticX · 2026-05-15

## Slide 1 — Introduction

- Eden Duthie — applied NLP and agent systems; author of the infon paper (AgenticX, May 2026).
- AgenticX — grounded, auditable knowledge systems for the enterprise.
- One-line stance: *we build systems that cite their sources and tell you when they don't know.*

## Slide 2 — What we heard

One slide, their words, mapped to seven bullets:

1. **Internal knowledge first** — search, summarise, reuse; reduce duplicated research.
2. **Governance and structure** — metadata classification, security levels, source traceability.
3. **Human-in-the-loop validation** — manual scoring and feedback loops.
4. **Document-centric** — PPTs, reports, attachments; not only datasets.
5. **Historical tracking** — how a topic evolves over time.
6. **Selective external data** — patents, news, subscription DBs as complement.
7. **Measurable KPIs** — usage logs, reuse, engagement.

> "Reducing duplicated research is the highest-stated priority." — quote back verbatim.

---

# Section 1 — Our approach (how infon works)

*Goal: a data engineer with no prior context can follow this section. Diagrams over text. Math kept off-slide.*

## Slide 3 — The problem with the obvious solution

- **ChatGPT / RAG:** retrieves passages, fabricates the connection. Sounds confident even when the corpus is silent.
- **Classical knowledge graphs:** require rigid, hand-curated schemas. Can't say "we don't know."
- **What's missing:** a system that reasons *across documents*, cites every step, and is honest about absence of evidence.

## Slide 4 — What infon is, in one sentence + one diagram

- **Infon = a knowledge-graph reasoner that fact-checks claims over your documents with calibrated uncertainty.**
- Diagram: Documents → Extracted facts (infons) → Cassette store → Reasoner → Cited verdict.
- Three properties that distinguish it:
  1. **Cassette-native storage** — immutable, content-addressed, full audit trail.
  2. **Honest uncertainty** — every answer carries `(supports, refutes, uncertain, θ)`. θ rises when the corpus is silent.
  3. **Multi-hop reasoning** — chains facts *across documents* via guided search, not flat retrieval.

## Slide 5 — The unit of knowledge: the *infon*, stored as a *cassette*

- An **infon** is a typed quadruple: `⟨ predicate, subject, object ; polarity ⟩`.
  - Example: `⟨ supplies, LG_Energy_Solution, solid_state_cells ; +1 ⟩` extracted from a 2024 LG press release.
  - Grounded to: document hash, byte offset, ingestion timestamp.
- A **cassette** is one document, stored as a single immutable `.inf` file: content-addressed, append-only, S3-native.
- Why engineers will care:
  - **Time-travel** — every ingest creates a new snapshot; old views remain queryable forever.
  - **Schema migration without reingestion** — change the ontology, rewrite the index in seconds (62× faster than re-extraction).

## Slide 6 — How a question gets answered (pipeline diagram)

Horizontal flow, five boxes:

1. **Question** — plain English ("Who supplies solid-state cells to Hyundai?")
2. **Encode** — small bundled model turns the question into an anchor set. Runs on CPU; no GPU, no API keys.
3. **Prune** — manifest skips cassettes that *cannot* contain those anchors (7–16× fewer reads at scale).
4. **Reason** — guided multi-hop search (MCTS) walks the fact graph; a small frozen graph neural net scores chains.
5. **Verdict** — `(supports, refutes, uncertain, θ)` + every contributing source citation.

## Slide 7 — Calibrated uncertainty in plain English

- Three honest outcomes per claim:
  - **Supports** — evidence backs the claim.
  - **Refutes** — evidence contradicts the claim.
  - **θ (theta)** — the corpus *doesn't speak to it*. The "I don't know" channel.
- Why it matters for Mobis: analysts get *trustworthy abstention* when the answer isn't in the documents — not a fabricated-but-confident response.
- Visual: three-bar verdict beside a single "LLM confidence" number.

## Slide 8 — Multi-hop reasoning across documents

- Single-hop: "Does company A supply component X?" → answered from one document.
- **Multi-hop**: "Does company A supply component X *through any supplier*?" → answered by chaining facts across documents.
- Visual: 3-node chain A → partners with → B → develops → X. Each edge cites its source document.
- Why it matters: most useful Mobis questions are multi-hop — *partners of partners*, *supplier of a competitor's component*, *what changed since last review*.

## Slide 9 — The Analyst — conversational layer with hard guardrails

- The user never writes queries. They ask in plain language.
- Diagram: chat input → Analyst → calls fixed tools (`ask`, `connect`, `any_of`) → returns a written answer + citation panel.
- **Built-in non-negotiables:**
  - Never returns an answer without citing the sources that produced it.
  - When θ > 0.7, explicitly says "the corpus does not answer this."
  - Cross-session memory — prior findings persist; the Analyst checks them at the start of each session.

## Slide 10 — Architecture at a glance

Layered diagram, bottom-up, with annotations for **shipped today** vs **built during the PoC**:

- **Layer 0** — Documents (PPT, PDF, DOCX, XLSX, external feeds).
- **Layer 1** — Document preprocessor *(PoC)*.
- **Layer 2** — Cassette substrate *(shipped)*.
- **Layer 3** — Query DSL + Reasoner *(shipped)*.
- **Layer 4** — Human-in-the-loop validation *(PoC)*.
- **Layer 5** — Access control + governance *(PoC)*.
- **Layer 6** — KPI / audit / usage *(PoC)*.
- **Layer 7** — Analyst conversational UI *(shipped — extended in PoC)*.

---

# Section 2 — Top 3 benefits for Hyundai Mobis

*Three slides. Each maps directly to a stated Mobis priority and a concrete infon capability. Other capabilities (HITL, governance, metadata, KPIs, external data, deployment posture) appear in Section 3 as roadmap deliverables.*

## Slide 11 — Benefit 1: Reduce duplicated research *(your highest priority)*

- **The problem:** the same question is researched repeatedly across teams; reports sit unread; institutional knowledge is buried in PPTs.
- **The capability:** a single graph traversal returns every fact on a topic, *with every source document grouped*. A fact cited in 10 reports is stored once and returned with all 10 citations.
- **Example query:** *"Show all internal knowledge on solid-state electrolyte suppliers, grouped by author team and date."*
- **What changes:** analysts find existing knowledge before commissioning new work; cross-team duplication becomes visible and measurable.
- **KPI hook:** queries answered from internal knowledge vs queries that prompted new external research.

## Slide 12 — Benefit 2: Track how knowledge evolves over time

- **The problem:** internal reports are written as snapshots. There's no easy way to ask *"what did we say about X last year, and how has our view changed?"*
- **The capability:** every ingest is an immutable snapshot. Nothing is ever rewritten. Queries can be scoped to any point in time.
- **Example queries:**
  - *"What did internal documents say about Toyota's solid-state battery timeline in 2023 vs 2025?"*
  - *"Show the trajectory of how our cost estimate for LiDAR has changed over the last 18 months."*
- **Visual:** same anchor, two snapshots side-by-side, divergence highlighted.
- **Why this is unique to infon:** time-travel is native to the cassette store, not an add-on. Schema migrations preserve historical queries — old answers stay correct.

## Slide 13 — Benefit 3: Answers you can trust — citations, abstention, contradictions

This is one slide combining three trust mechanisms:

- **Every answer cites its sources.** Each infon carries document hash, byte offset, timestamp. The Analyst is prompt-constrained never to output a verdict without citing the contributing sources. Click → opens the exact passage.
- **The system says "I don't know."** When θ > 0.7, the answer is "the corpus does not contain a verified answer." No fabrication on absent evidence.
- **Contradictions surface automatically.** The polarity channel (+1 affirmed / −1 negated) lets the system *find disagreements* across documents — pre-decision risk review becomes a one-line query.

Why these three sit together: they all answer the same governance question — *can a decision-maker trust a result this system returns?*

---

# Section 3 — Proposed 10-week PoC roadmap

## Slide 14 — Roadmap philosophy

- **Iterative, not waterfall.** Every phase ends with something you can actually use.
- **Schema-first.** Bootstrapped in week 1 from the Mobis Korean glossary + a sample corpus — improves throughout the PoC.
- **Validation built in.** HITL queue live by week 5; reviewer feedback drives schema v2.
- **Honest scope.** PoC runs on a curated set of 150–300 docs. Full-corpus and external connectors are post-PoC.

## Slide 15 — The 10-week plan

| Weeks | Phase | Key deliverables | Milestone |
|-------|-------|------------------|-----------|
| 1–2 | **Bootstrap** | Sample corpus + glossary received; Korean encoder swap (multilingual / KoSPLADE); schema v1 drafted; preprocessor running on Mobis formats | First Korean documents ingested under schema v1 |
| 3–4 | **Foundation** | 150–300 curated docs ingested; basic query API; first end-to-end hero query answered with citations | A real Mobis question, answered with cited cross-document facts |
| 5–6 | **Governance + HITL** | Review queue UI; cassette-level security classification; role-based query filter; audit log | Domain experts using the review queue daily |
| 7–8 | **Synthesis + KPI** | LLM synthesis layer; KPI dashboard; schema v2 from review feedback | Validated-fact growth visible on the dashboard |
| 9 | **Stabilise** | Edge-case handling; schema v2 ingest; final KPI capture | Frozen system for evaluation week |
| 10 | **PoC report + decision** | Measured KPIs vs agreed targets; PoC report | Joint go / no-go decision |

## Slide 16 — Week 1, in detail

- Pre-engagement: sample corpus (50–100 docs) + Mobis Korean glossary received.
- Day 1–2: Korean-capable SPLADE variant swapped in and validated on sample; preprocessor running; schema v1 drafted from glossary.
- Day 3–4: first ingest of sample; `extraction_report()` surfaces coverage gaps; schema patch.
- Day 5: first internal demo of a hero query on real Mobis content.
- **The iteration cadence for the rest of the PoC:** ingest → report → refine → re-ingest.

## Slide 17 — In scope vs deferred

**In PoC scope (10 weeks):**
- Document preprocessor (PPT/PDF/DOCX/XLSX).
- Korean-language extraction — encoder swap is week-1 work.
- Schema v1 → v2 bootstrap + refinement (anchored on Mobis Korean glossary).
- 150–300 curated internal documents ingested.
- HITL review queue.
- Security classification + role-based access.
- KPI dashboard + audit log.
- Query UI (chat + graph viewer + citations).

**Deferred to post-PoC (production contract):**
- Full-corpus ingestion (thousands of docs).
- External connectors (patents, news, subscription DBs).
- SSO integration with Mobis IdP.
- Production-grade SLA / monitoring.

> Cut items are *scaling*, not *capability*. Every capability you'll evaluate the PoC on is in the 10-week scope.

## Slide 18 — Roles & working pattern

- **AgenticX** — schema design, ingestion, reasoner tuning, UI, KPI instrumentation.
- **Deloitte** — delivery partner; engagement management, on-site Mobis coordination, change management, integration support.
- **Hyundai Mobis** — document curation (each phase), domain-expert reviewers (weeks 5–10), KPI target setting (week 1), final evaluation (week 10).
- **Cadence:** weekly checkpoint; schema review every two weeks; phase-end demo.

## Slide 19 — Risks & mitigations

| Risk | Mitigation |
|---|---|
| Bundled SPLADE-tiny is English; Korean documents need a different encoder | Week-1 swap to a multilingual or Korean-trained SPLADE variant (KoSPLADE / mSPLADE); validate anchor coverage on Korean sample before scaling ingest |
| Korean tokenisation / morphology degrades anchor matching | Korean-aware tokenizer; anchor vocabulary seeded from Mobis Korean glossary; verify with `extraction_report()` |
| Schema v1 extracts poorly on real Mobis content | Run `extraction_report()` on week 1 sample; iterate before scaling ingest |
| Security review blocks document access | Start week 1 with a "safe" curated subset already agreed |
| KPI targets disputed at week 10 | Agree targets *in writing* in week 1 |

## Slide 20 — Immediate next steps

1. **Scoping call** — agree corpus scope, security requirements, integration points (SAP, PLM, document store), KPI definitions.
2. **Sample corpus** — 50–100 representative Korean documents across key knowledge areas.
3. **Internal glossary / taxonomy** — Korean anchor vocabulary seed.
4. **PoC success criteria in writing** — measurable KPIs with target values.
5. **Kick-off** — week 1 starts as soon as items 1–4 are agreed.

---

## Slide 21 — Summary + Q&A

- **What we heard** — internal knowledge, governance, HITL, document-centric, historical tracking, KPIs.
- **What infon brings** — cited multi-hop reasoning, time-travel, calibrated "I don't know."
- **What we propose** — 10-week iterative PoC; Korean-first; HITL live by week 5; measurable KPIs by week 10.
- **감사합니다** *(gamsahamnida — thank you)*. Questions?

---

## Appendix slides (kept off the main deck, used in Q&A)

- A1 — Benchmark numbers (HoVer multi-hop, AVeriTeC NEI, SciFact calibration) for the technical track.
- A2 — Sheaf GNN internals (for the "how does the GNN actually work" question).
- A3 — Schema migration without reingestion (technical detail).
- A4 — On-prem / air-gapped deployment specifics.
- A5 — Document-centric coverage details: PPT/PDF/DOCX/XLSX preprocessor, OCR fallback, metadata extraction.
- A6 — External-data strategy: patents (Espacenet, KIPO, USPTO), industry news, standards bodies.
- A7 — KPI dashboard mock: validated-fact growth, query response time vs manual research, cross-document connections, duplication reduction, schema coverage.
- A8 — HITL review queue mock: source sentence, extracted triple, confidence + reason, confirm/reject/correct.
- A9 — Governance & metadata: security classification model, role-based query filter, audit log structure.
- A10 — The paper — *Infon: A Knowledge Graph Reasoner with Calibrated Uncertainty*, Duthie et al. 2026.

---

## Format & open decisions

**Confirmed format (2026-05-14):**
- Microsoft Teams. 1.5h total; Deloitte first; AgenticX slot ~45 min.
- Presentation only — no live demo, no pre-recorded clips.
- No side-by-side ChatGPT comparison.
- Korean greeting on title slide (안녕하세요) and closing (감사합니다).

**Pacing — 22 slides for ~45 min** = ~10 min Q&A buffer + ~35 min presentation → ~95 sec/slide. Comfortable.

**Still open:**
- [ ] Final attendee list and roles — tailor slide 16 ("week 1 in detail") to whoever owns budget approval.
- [ ] Coordinate content with Deloitte's slot — engagement management may already be covered, freeing time on slide 18.
- [ ] Decide whether slide 10 (architecture) needs a simplified version for the business audience and a detailed one for appendix.
