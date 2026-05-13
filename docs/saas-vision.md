# Infon: Vertical Knowledge Intelligence Platform — SaaS Vision

## Executive Summary

Infon is a structured knowledge graph engine that extracts, stores, and reasons over facts drawn from document corpora. It occupies a distinct position between traditional search (returns documents) and large language models (returns confident prose that may be wrong): infon returns **verified, cited, uncertainty-tagged facts**, assembled via symbolic graph traversal with honest "I don't know" reporting when the corpus is insufficient.

This document describes a vertical-first SaaS strategy for commercialising infon as an enterprise knowledge intelligence platform — starting with automotive (Hyundai Mobis) and expanding across industries where document-heavy knowledge work drives high costs and high risk.

The business thesis: **the value of any knowledge base is not the documents it contains, but the connections between facts across those documents, and the ability to know when information is absent, contradictory, or outdated.** No existing enterprise search or RAG product provides this. Infon does.

---

## The Problem

Large organisations accumulate vast internal knowledge — engineering reports, research summaries, specifications, meeting records, procurement documents — and struggle with three recurring failures:

1. **Rediscovery:** Teams repeat research that was already done, because they cannot find it in document silos.
2. **Contradiction blindness:** Conflicting information exists across documents and systems; no tool surfaces these conflicts before they cause failures.
3. **Temporal drift:** Knowledge changes over time, but systems snapshot documents rather than tracking how facts evolve. Audit and compliance questions ("what did we know, and when?") are expensive to answer.

Existing solutions fail because they treat documents as the unit of knowledge. Vector search and RAG retrieve *chunks of text*; an LLM synthesises a confident-sounding answer. This works for simple factual recall but fails for multi-document reasoning, contradiction detection, and temporal fact tracking — and produces hallucinations the user cannot detect.

---

## The Infon Advantage

Infon's unit of knowledge is the **infon**: a typed, grounded, uncertainty-tagged triple `<<subject, predicate, object; polarity>>` with exact provenance (source document, byte offset, sentence, timestamp, confidence score). Infons are stored in content-addressed cassettes and indexed for graph traversal.

### What this enables that no RAG system can replicate

| Capability | RAG / LLM Search | Infon |
|---|---|---|
| Multi-hop graph reasoning | LLM loop (hallucination-prone) | Symbolic MCTS (grounded chains) |
| Contradiction detection | None | Native — surfaces conflicting facts across any documents |
| Uncertainty quantification | None — LLM sounds confident | Dempster-Shafer mass per answer |
| Temporal fact tracking | None | Time-travel queries; snapshot chains; `before/after/between` |
| Source provenance | Approximate | Exact: byte offset, sentence, timestamp per fact |
| Cross-document connection | Similarity only | Graph traversal across any connected facts |
| Honest "I don't know" | Never | `m_θ → 1.0` when corpus is insufficient |
| Deduplication at scale | None | Content-addressed; same fact from 1000 sources stored once |

### Cost efficiency

The extraction pipeline uses a frozen 4.4M parameter SPLADE model (ships bundled, no GPU required, ~10–20ms per sentence on CPU). There are no per-token LLM API costs for extraction. At scale, an LLM synthesis layer is used only for prose generation over already-verified facts — a much cheaper and more accurate use of LLM compute than using it for both retrieval and synthesis.

---

## Market Positioning

Infon is not a search engine and not a chatbot. It is a **knowledge graph layer** that sits between a document corpus and any query interface (human or AI agent).

**Primary competitors and their gaps:**

- **Microsoft SharePoint Copilot / Microsoft 365 AI:** Document-level search with LLM synthesis. No fact-level graph, no contradiction detection, no temporal reasoning. Locked to Microsoft ecosystem.
- **Elastic + RAG (any vendor):** Strong document retrieval; no cross-document fact reasoning; no uncertainty quantification; hallucination on synthesis.
- **Notion AI / Confluence AI:** Wiki-layer LLM. Same limitations. No structured reasoning.
- **Palantir Foundry:** Powerful but requires extensive data engineering, very expensive, targets defence/government. Not self-service.
- **Traditional Knowledge Graphs (Neo4j + manual curation):** Accurate but require expert curation of every fact. Infon automates extraction at scale.

**Infon's position:** Automated, provenance-tracked, uncertainty-aware knowledge graph extraction. Self-service for a vertical domain, priced for enterprise, defensible via schema and domain expertise compounding over time.

---

## Vertical-First Go-to-Market Strategy

### Why vertical-first

A general-purpose internet-scale knowledge graph is a decade-long infrastructure problem. Vertical focus solves three things simultaneously:

1. **Schema-at-scale:** Every vertical has existing domain ontologies (UMLS for biomedical, FIBO for finance, industry standards for automotive). The schema bootstrap becomes an engineering task (weeks), not a research problem.
2. **Sales motion:** Domain-specific demos (e.g., "supply chain disruption impact query" for automotive; "conflicting clinical evidence" for pharma) are immediately legible to buyers.
3. **Compounding moat:** A schema tuned for a vertical improves with each client and each document ingested. The graph for Automotive client 3 is better than for client 1 because it has been refined by 2 clients' feedback.

### Vertical expansion roadmap

**Wave 1 — Automotive (now)**
- Client: Hyundai Mobis (see companion document)
- Domain: Parts management, internal engineering knowledge, supplier intelligence
- Existing ontologies: Industry standards (IATF 16949), OEM part schemas, REACH/RoHS compliance taxonomies
- Killer queries: Supply chain disruption impact; cross-team research deduplication; temporal spec tracking

**Wave 2 — Biomedical / Pharma**
- Domain: Drug discovery, clinical evidence, regulatory documents
- Existing ontologies: UMLS (3M concepts), MeSH, Gene Ontology, ChEBI, RxNorm, SNOMED CT
- Killer queries: "What does the evidence say about drug X for condition Y, with uncertainty?" across internal trials + published literature; contradiction surfacing between internal and published data
- Market: CROs, pharma R&D divisions, hospital systems

**Wave 3 — Legal / Compliance**
- Domain: Contract intelligence, regulatory compliance, case law
- Existing ontologies: Legal taxonomy (Black's Law, jurisdiction-specific), regulatory code structures
- Killer queries: "What precedents support this argument chain?"; "Which of our contracts are exposed to this regulatory change?"
- Market: Law firms, in-house legal teams, financial institutions

**Wave 4 — Financial / Market Intelligence**
- Domain: Investment research, supply chain risk, ESG
- Existing ontologies: FIBO, company registries, SIC/NAICS codes
- Killer queries: Multi-hop supply chain exposure; contradiction between analyst reports; temporal tracking of market claims
- Market: Asset managers, corporate strategy teams

---

## What Needs to Be Built

The core infon library is the reasoning and storage engine. The SaaS product requires additional layers. Below is a gap analysis.

### Layer 1 — Document Preprocessing (Gap: High Priority)

The current infon pipeline ingests text. Enterprise documents are binary formats.

**Required:**
- PPTX → text extractor (slide text, speaker notes, table content) via `python-pptx`
- PDF → text extractor with layout awareness via `pdfplumber` / `pymupdf`
- DOCX → text via `python-docx`
- XLSX → structured-to-text for table data via `openpyxl`
- Image/diagram OCR fallback via `pytesseract` or vision model
- Metadata extraction: author, creation date, modification date, document type, security classification from file properties

Output: normalised `(text, metadata)` pairs fed to the existing SPLADE pipeline.

**Estimated scope:** 2–3 weeks. Vertical-specific format quirks add time (e.g., automotive CAD-linked PDFs).

### Layer 2 — Schema Construction Tooling (Gap: Medium Priority)

Currently schemas are hand-authored JSON. At vertical scale, schemas must be bootstrapped from existing ontologies and refined iteratively.

**Required:**
- Ontology importer: convert UMLS / FIBO / domain-specific taxonomy → infon anchor schema
- Schema discovery assistant: analyse a sample corpus and propose anchor candidates (frequency + SPLADE activation analysis)
- Schema editor UI: domain experts add/remove/merge anchors; version controlled
- Schema migration: the existing `SchemaFunctor` handles cassette rewriting — expose this in the product as "schema upgrade" workflow

**Estimated scope:** 3–4 weeks for ontology importer + basic editor. Schema discovery assistant is ongoing.

### Layer 3 — LLM Synthesis Layer (Gap: Medium Priority)

Infon returns fact chains, not readable prose. Enterprise users need narrative answers.

**Architecture:**
```
User query
  → infon graph traversal → verified fact subgraph (with provenance)
  → LLM prompt: "Synthesise this into a readable answer. Cite each fact."
  → Response: prose + inline citations linking to source documents
```

**Key constraint:** The LLM sees only facts the graph retrieved — it cannot hallucinate beyond the corpus. Citations are infon provenance (exact source + sentence), not LLM-generated references.

**Estimated scope:** 1–2 weeks. Model selection (GPT-4o, Claude, Gemini) is a configuration option.

### Layer 4 — Human-in-the-Loop Validation (Gap: High Priority for Enterprise)

Enterprise clients require human validation before trusting automated extractions. This is also the primary feedback mechanism for improving the schema.

**Architecture:**
```
Extracted infon (confidence < threshold → e.g. 0.35)
  → Review queue (prioritised by domain importance weight)
  → Domain expert UI: source sentence shown, triple displayed
      → Confirm (confidence boosted, human_validated=True flag)
      → Reject (flagged, excluded from queries, contributes to schema refinement)
      → Correct (new infon written with correction, original preserved)
  → Aggregate feedback → schema improvement suggestions
```

**Product value:** Creates measurable PoC KPI ("human-validated fact count grows over time"). Creates stickiness — the more experts interact, the more accurate the system becomes for their domain. The review queue doubles as an onboarding mechanism for domain experts.

**Estimated scope:** 3–4 weeks (queue, review UI, flagging, feedback aggregation).

### Layer 5 — Access Control and Knowledge Governance (Gap: Medium Priority)

Enterprise requirements: security classification, access control, audit trail.

**Architecture:**
- Cassette-level classification tags: `public | internal | confidential | restricted`
- User roles map to classification clearance levels
- Query API filters cassette set by user clearance before traversal
- Audit log: every query, every result set accessed, every human-validated action — timestamped and attributed
- Retention policy: cassettes can be marked for expiry; manifest pruner skips expired cassettes

**Estimated scope:** 2–3 weeks. Identity integration (SSO, LDAP) varies by client — add 1–2 weeks per integration.

### Layer 6 — KPI and Analytics Layer (Gap: Required for PoC)

Clients need to see measurable value, especially during PoC.

**Metrics to track:**
- Query volume and latency over time
- Result utilisation rate (queries that led to a document being opened/cited)
- Knowledge coverage: infon density by topic/domain area (shows gaps)
- Duplication reduction: queries answered from existing knowledge vs. referred to external research
- Human validation throughput and accuracy trends
- Schema growth over time

**Implementation:** Query API middleware logs to a time-series store. Dashboard in Grafana or Metabase. PoC report auto-generated from logs.

**Estimated scope:** 1–2 weeks logging; 1–2 weeks dashboard.

### Layer 7 — Query UI (Gap: Required for Product)

The current infon library exposes a Python DSL. Enterprise users need a UI.

**Minimum viable UI:**
- Natural language query box → translated to infon DSL (LLM-assisted)
- Result display: fact chain with provenance links, uncertainty score shown plainly
- Document view: source document with infon-sourced sentences highlighted
- Topic browser: navigate the knowledge graph by anchor/entity
- Human validation queue (for reviewers)
- Admin panel: schema management, security classification, user roles

**Estimated scope:** 6–8 weeks for a functional product-grade UI. A minimal PoC UI is 2–3 weeks.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│   Natural language query | Document browser | Review queue  │
│   Admin panel | KPI dashboard                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   QUERY & SYNTHESIS API                     │
│   NL → DSL translation (LLM) | Access control filter       │
│   Infon graph traversal (MCTS + Sheaf GNN)                  │
│   LLM synthesis over fact subgraph | Provenance rendering   │
│   Query logging | KPI instrumentation                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  INFON KNOWLEDGE STORE                      │
│   Content-addressed cassettes (S3)                          │
│   Multi-column indexes (by_triple | by_time | by_anchor)    │
│   Manifest pruner | Time-travel snapshots                   │
│   Schema versioning (SchemaFunctor)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 INGESTION PIPELINE                          │
│   Document preprocessor (PPT|PDF|DOCX|XLSX → text)         │
│   SPLADE encoder (frozen, CPU, 10-20ms/sentence)            │
│   Anchor projector + triple formation                       │
│   Human-in-the-loop review queue                           │
│   External source connectors (patents, news, databases)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Build Strategy: SaaS Platform and Vertical Simultaneously

The critical architectural decision is to build the **SaaS platform and the Hyundai Mobis vertical in parallel**, with the following separation:

- **Infon core library:** Domain-agnostic. All improvements (document preprocessing, human-in-the-loop, access control) go here as reusable components.
- **Hyundai Mobis vertical:** Schema + configuration + client-specific connectors (SAP, PLM, internal document stores). No Hyundai-specific logic in the platform layer.
- **Platform product:** Schema editor, review queue, KPI dashboard, query UI — built generically, configured per vertical.

This means every week of work on the Hyundai engagement also advances the SaaS product. The Hyundai schema becomes the template for the automotive vertical. The document preprocessing built for their PPTs works for every other client's PPTs.

See the companion document *Hyundai Mobis: Recommendations* for the specific delivery plan.

---

## Revenue Model

**Tier 1 — PoC / Pilot** (per engagement)
- Fixed-price delivery of schema bootstrap + ingestion of a defined corpus + query interface
- Typically 8–12 weeks
- Price: $80k–$150k depending on corpus size and schema complexity

**Tier 2 — SaaS Subscription** (post-PoC)
- Monthly per-seat or per-query pricing
- Includes: managed ingestion pipeline, hosted cassette store, query API, UI
- Price: $5k–$25k/month depending on corpus size, query volume, user seats

**Tier 3 — Enterprise License**
- On-premise or private cloud deployment
- Includes schema migration support, SSO integration, SLA
- Price: $150k–$500k/year

**Services revenue** (ongoing)
- Schema maintenance and refinement as domain evolves
- New data source connectors
- Custom query development

---

## Defensibility and Moat

1. **Schema quality compounds:** A well-tuned vertical schema is months of domain expert feedback. It cannot be replicated quickly by a new entrant.
2. **Human-validated fact graph:** The reviewed, confirmed infon graph is a unique asset per client. It cannot be reconstructed from documents alone — it embeds expert judgment.
3. **Switching cost:** Clients' workflows (queries, review processes, API integrations) are built against the infon graph. Migrating means rebuilding the graph from scratch.
4. **Technical depth:** The Dempster-Shafer reasoning, MCTS traversal, and Sheaf GNN are non-trivial to replicate. The academic foundation (published paper) provides credibility.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Schema bootstrap is harder than expected | Start with existing ontologies; use schema discovery tooling; accept imperfect v1 schema |
| LLM synthesis adds hallucination back | LLM sees only retrieved facts, not the full corpus; citations are infon-sourced, not LLM-generated |
| Enterprise sales cycles are long | PoC-first motion with fixed-price engagement de-risks for client; fast time-to-value via first demo |
| Large vendors (Microsoft, Google) add similar features | Depth of provenance, uncertainty, and temporal reasoning is not replicable by adding a RAG wrapper to a document store |
| Document format complexity (CAD files, proprietary formats) | Scope preprocessing to supported formats in v1; treat unsupported formats as text fallback |

---

## Next Steps

1. Complete Hyundai Mobis PoC delivery (see companion document) — this is the reference implementation
2. Generalise PoC components into the SaaS platform layer
3. Identify Wave 2 vertical (biomedical recommended) based on Hyundai Mobis learnings
4. Publish case study from Hyundai Mobis engagement (with client approval) to drive inbound interest
