# Hyundai Mobis: Infon Knowledge Intelligence — Recommendations

## Overview

This document provides technical and delivery recommendations for implementing an infon-based internal knowledge management system for Hyundai Mobis. It should be read alongside the *Infon SaaS Vision* document, which describes the broader platform that this engagement will simultaneously advance.

The core principle of this engagement: **every component built for Hyundai Mobis is designed as a reusable platform module, not a client-specific integration.** The schema and configuration are client-specific; the tooling is not. This means the Hyundai Mobis engagement funds the construction of the SaaS platform, and the SaaS platform makes future engagements faster and cheaper to deliver.

---

## Client Requirements Mapping

### 1. Internal knowledge search, summarisation, and reuse

**Infon fit: Strong**

Infon extracts structured facts from internal documents and stores them in a queryable knowledge graph. A natural language query is translated to the infon DSL, the graph is traversed to find relevant facts across all ingested documents, and an LLM synthesises a readable answer from the retrieved fact subgraph — with every sentence linked to its exact source document and passage.

This is architecturally superior to RAG-based search for this requirement because:
- The LLM synthesises *verified facts*, not fuzzy chunks — reducing hallucination
- Knowledge is connected *across* documents, not retrieved *per* document
- Reuse is explicit: the same fact, cited in 10 documents, is stored once and returned with all 10 sources

**Key capability:** The "reducing duplicated research" use case — the highest-stated priority — is answered by a single graph traversal: *"Show all internal knowledge on topic X, grouped by team and date."* This is achievable from day one with a correctly built schema.

### 2. Knowledge governance and metadata classification

**Infon fit: Requires extension (Layer 5 of the SaaS platform)**

Infon's cassette format already supports document-level metadata (author, source, timestamp, classification tags). The extension required is:
- A **security classification model** (public / internal / confidential / restricted) applied at cassette level during ingestion
- A **query access control layer** that filters the traversable cassette set based on the requesting user's role
- An **audit log** — every query, every result set, every human validation action — timestamped and attributed to a user

Source traceability is infon's strongest native capability. Every extracted fact carries its exact source document, byte offset within the document, and the sentence it was extracted from. The UI renders these as clickable reference links.

### 3. Document-centric coverage — PPTs, reports, attachments

**Infon fit: Gap — requires preprocessing layer (Layer 1 of the SaaS platform)**

The current infon pipeline ingests text. Hyundai Mobis documents will arrive as PPTX, PDF, DOCX, and potentially XLSX. A preprocessing module is required:

| Format | Extractor | Notes |
|---|---|---|
| PPTX | `python-pptx` | Slide text, speaker notes, table cells, chart titles |
| PDF | `pdfplumber` / `pymupdf` | Layout-aware; handles multi-column and table extraction |
| DOCX | `python-docx` | Body text, tables, headers |
| XLSX | `openpyxl` | Sheet data as structured text; column headers as field names |
| Images embedded in docs | `pytesseract` or vision model | OCR fallback for scanned content |

**Metadata extracted per document:** filename, author (from file properties), creation date, last modified date, document type (inferred from format + content), security classification (from filename conventions or folder path if a structured directory exists).

This preprocessing module is the first deliverable, built generically, and reused for every subsequent SaaS client.

### 4. Historical tracking — how topics evolve over time

**Infon fit: Strong — native capability**

Infon's snapshot chain and time-travel query DSL directly address this requirement. Every ingest creates an immutable snapshot. Queries can be scoped to any point in time:

- *"What did our internal documents say about solid-state battery suppliers in 2022 vs. 2024?"*
- *"When did our internal risk assessments of Supplier X change?"*
- *"Show the trajectory of how our understanding of ADAS lidar costs has evolved."*

These queries use the existing `before()`, `after()`, `between()`, and `trajectory()` DSL operators. No extension required. This is a significant differentiator from any alternative solution and should be demonstrated explicitly in the PoC.

### 5. Human-in-the-loop validation

**Infon fit: Gap — highest priority new capability (Layer 4 of the SaaS platform)**

This requirement is the most important new component to build, for two reasons: it directly satisfies a stated client requirement, and it is the primary mechanism by which the system improves over time.

**Proposed architecture:**

Extracted infons below a confidence threshold (suggested: joint score < 0.35) are placed in a review queue. Domain experts are presented with:
- The source sentence
- The extracted triple (subject → predicate → object)
- The document, page, and section of origin
- The confidence score and the reason it is low (low activation, ambiguous polarity, etc.)

The expert takes one of three actions:
- **Confirm:** Confidence is boosted; `human_validated = True` flag is set. This infon is treated as authoritative.
- **Reject:** Infon is excluded from query results. The extraction is logged as a schema refinement signal.
- **Correct:** The expert edits the triple. A new infon is written with the correction and `human_validated = True`; the original is preserved for audit purposes.

Aggregate rejection patterns feed back to the schema team: if a particular anchor or relation is consistently rejected in a context, the schema needs refinement.

**Product outcome:** The review queue is also the primary onboarding mechanism. Early in the engagement, domain experts spend time in the queue. This builds familiarity with the system while simultaneously improving it. The growing count of human-validated facts is a natural PoC KPI.

### 6. Selective use of external data

**Infon fit: Strong — clean architectural separation**

External sources (patents, news, subscription databases) are ingested as separate cassettes tagged with their source type and access tier. Content-addressing ensures the same patent appearing in two feeds is deduplicated. The security classification model keeps external content clearly separated from confidential internal content.

**Recommended external sources for Hyundai Mobis:**
- Patent databases: Espacenet (free API), Korean IPO, USPTO — relevant to R&D and competitive intelligence
- Industry news: RSS or licensed feeds from automotive trade publications
- Standards bodies: ISO, SAE — document updates tracked over time
- Supplier public filings: annual reports, sustainability disclosures (where relevant to procurement decisions)

External source connectors are built as part of the SaaS platform's ingestion pipeline and reused across clients.

### 7. KPI and usage tracking

**Infon fit: Gap — straightforward instrumentation (Layer 6 of the SaaS platform)**

A logging middleware layer on the query API records:
- Query log: timestamp, user, query text, query DSL, result count, latency
- Result utilisation: which source documents were opened from a query result
- Review queue: throughput, accuracy trends, schema refinement events
- Coverage metrics: infon density by topic (identifies knowledge gaps)

**Suggested PoC KPIs to agree with Hyundai Mobis at engagement start:**
1. Number of human-validated facts in the graph (grows over time)
2. Query response time vs. estimated analyst time for equivalent manual research
3. Cross-document connections discovered (facts linked across documents that were never explicitly related)
4. Duplication reduction: queries answered from internal knowledge vs. requiring new external research
5. Schema coverage: percentage of internal documents with at least one extracted infon

---

## Recommended Schema Design

### Domain anchor types for Hyundai Mobis

Based on the knowledge governance and internal knowledge focus, the schema should prioritise **research and knowledge concepts** rather than parts/supply chain structure. The parts taxonomy can be added in a later phase if operational data is also in scope.

```
ACTORS (entities that appear as subjects or objects in knowledge claims)
  - Technology         (e.g., solid-state battery, LiDAR, ADAS Level 3)
  - Component          (e.g., brake module, airbag ECU, headlamp assembly)
  - Supplier           (Tier 1/2/3 vendors)
  - Competitor         (other OEM Tier 1 suppliers)
  - Team / Division    (internal Hyundai Mobis groups)
  - Standard           (ISO, SAE, IATF, Korean regulations)
  - Material           (steel grades, polymers, rare earth materials)
  - Project            (internal R&D or development programmes)
  - Regulation         (REACH, RoHS, UN ECE regulations)

RELATIONS (typed predicates connecting actors)
  - researched_by      (Technology ← Team)
  - developed_by       (Component ← Team / Supplier)
  - supplied_by        (Component ← Supplier)
  - complies_with      (Component / Material → Standard / Regulation)
  - competes_with      (Supplier / Technology → Competitor)
  - depends_on         (Technology / Component → Component / Material)
  - superseded_by      (Technology → Technology)
  - risk_identified_in (Component / Supplier → Project / Report)
  - costs              (Component → cost range — Feature object)
  - tested_against     (Component → Standard)

FEATURES / OBJECTS (terminal nodes — typically attributes or values)
  - CostRange
  - PerformanceSpec    (tolerance, operating range, etc.)
  - RiskLevel
  - MaturityLevel      (TRL 1-9 for technology readiness)
  - GeographicRegion
```

### Schema bootstrap approach

1. Start with a curated list of ~200 anchor terms per type, drawn from:
   - Hyundai Mobis internal glossary (request from client)
   - Automotive industry standards vocabulary (SAE, ISO)
   - Supplier database export (company names)
2. Run the schema discovery tooling over a 50–100 document sample to identify high-frequency terms not yet in the schema
3. Domain expert review of proposed additions (first use of the human-in-the-loop workflow)
4. Expand to the full corpus

This schema is intentionally broad enough to be reusable as the **automotive vertical schema** in the SaaS platform, with Hyundai Mobis-specific anchors (internal team names, internal project codes) isolated as a client overlay.

---

## Delivery Plan

### Phase 1 — Foundation (Weeks 1–6)

**Goal:** Ingest a representative sample corpus; demonstrate the core value proposition.

| Deliverable | Scope | Notes |
|---|---|---|
| Document preprocessor | PPT, PDF, DOCX → text + metadata | Built as SaaS platform Layer 1 |
| Schema v1 | ~200 anchors per type, automotive domain | Bootstrapped from glossary + sample corpus |
| Ingestion pipeline | Batch ingest of 200–500 client-selected documents | Documents selected by Hyundai Mobis to represent key knowledge areas |
| Basic query API | DSL queries over ingested graph | REST API; no UI yet |
| PoC demo | Live demonstration of the "duplicated research" query | Pick a real question from a Hyundai Mobis analyst; answer it with infon in <10 seconds; show citations |

**Milestone:** Hyundai Mobis stakeholders see their own internal knowledge answered with citations, across documents that were never explicitly linked.

### Phase 2 — Governance and Validation (Weeks 7–12)

**Goal:** Add governance, human validation, and measurable KPIs.

| Deliverable | Scope | Notes |
|---|---|---|
| Human-in-the-loop review queue | Web UI for domain experts | Built as SaaS platform Layer 4 |
| Security classification | Cassette-level tags + query access control | Roles: admin, reviewer, analyst, reader |
| Audit log | Every query and validation action logged | Used for PoC KPI reporting |
| KPI dashboard | Search volume, validation throughput, coverage metrics | Built as SaaS platform Layer 6 |
| Schema v2 | Refined from Phase 1 review feedback | |
| LLM synthesis | Narrative answers over retrieved fact subgraphs | Built as SaaS platform Layer 3 |

**Milestone:** Domain experts are actively using the review queue; KPI dashboard shows measurable growth in validated facts and query utilisation.

### Phase 3 — Enrichment and Expansion (Weeks 13–20)

**Goal:** Extend corpus to full internal document set; add external data; production-ready.

| Deliverable | Scope | Notes |
|---|---|---|
| Full corpus ingestion | All Hyundai Mobis internal documents in scope | Volume TBD; likely thousands of documents |
| External connectors | Patents (Espacenet), industry news | Built as SaaS platform ingestion connectors |
| Historical tracking UI | Timeline view of topic evolution | Demonstrates native infon time-travel capability |
| Schema v3 | Expanded from full corpus analysis | |
| SSO integration | Integration with Hyundai Mobis identity provider | Scope depends on their IdP |
| PoC report | Quantified KPIs against agreed targets | Basis for production contract |

**Milestone:** Production-ready system with measurable KPIs; client has basis for moving from PoC to production contract.

---

## How This Engagement Builds the SaaS Platform

The following table maps each Hyundai Mobis deliverable to the SaaS platform component it produces. Nothing built for this engagement is throwaway.

| Hyundai Mobis Deliverable | SaaS Platform Component |
|---|---|
| Document preprocessor (PPT/PDF/DOCX) | Layer 1 — Document Preprocessing |
| Schema v1 bootstrap tooling | Layer 2 — Schema Construction Tooling |
| LLM synthesis over fact subgraphs | Layer 3 — LLM Synthesis Layer |
| Human-in-the-loop review queue | Layer 4 — Human-in-the-Loop Validation |
| Security classification + access control | Layer 5 — Access Control and Governance |
| KPI dashboard + query logging | Layer 6 — KPI and Analytics Layer |
| Query UI | Layer 7 — Query UI |
| Automotive schema (without client-specific overlays) | Automotive vertical schema template |
| External connectors (patents, news) | SaaS platform ingestion connectors |

The Hyundai Mobis automotive schema (stripped of proprietary anchor terms) becomes the starting schema for every subsequent automotive client. The document preprocessing, governance, and validation tooling is immediately deployable for Wave 2 (biomedical) with only schema and connector changes.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Client document corpus is highly heterogeneous (many formats, quality varies) | High | Scope Phase 1 to a curated 200–500 document set; expand later. Preprocessing handles the main formats. |
| Schema v1 has poor extraction quality on client documents | Medium | Run schema discovery over sample before finalising v1; domain expert review session early |
| Internal document access requires complex permissions negotiation | Medium | Agree data access scope and format in pre-engagement scoping; start with a "safe" subset |
| Hyundai Mobis wants parts/supply chain graph (not just internal knowledge) | Low-Medium | Architecture supports it; parts ontology can be added as Phase 4 with BOM direct-insert pipeline |
| Client wants on-premise deployment | Medium | Infon uses S3-compatible storage (any object store); cassettes are portable; containerise the full stack |
| LLM synthesis introduces hallucination into answers | Low | LLM only synthesises facts retrieved by the graph; citations are infon-provenance; fact-checking is bounded |

---

## Immediate Next Steps

1. **Scoping call with Hyundai Mobis:** Agree document corpus scope, security requirements, existing systems (SAP, PLM, document store), and KPI definitions for PoC success
2. **Request sample corpus:** 50–100 documents across key knowledge areas; representative of formats and topics
3. **Request internal glossary or taxonomy:** Starting point for schema anchor vocabulary
4. **Agree PoC success criteria in writing:** Specific KPIs, target values, measurement method
5. **Set up development environment:** Separate infon instance for Hyundai Mobis; schema version-controlled from day one
6. **Begin document preprocessor development:** First platform component; unblocks everything else
