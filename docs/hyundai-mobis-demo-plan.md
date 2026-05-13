# Hyundai Mobis — Friday Demo Plan

**Demo date:** Friday 2026-05-15 (T-2 days from drafting)
**Slot:** 30 minutes
**Audience:** Mixed (technical + business)
**Goal:** Convince Hyundai Mobis to start the engagement. Not "show off infon."

---

## What they must walk away believing

All three of these, in this order of emotional weight:

1. **The tech is real** and works on their kind of documents — live query, citations, no slides.
2. **Infon does things ChatGPT/Copilot fundamentally cannot** — contradiction surfacing, time-travel, calibrated uncertainty.
3. **We have a credible plan** to deliver this in **10 weeks** — phased, schema-driven, with measurable PoC KPIs.

Anchor the framing in their own stated pain (from the recommendations doc):
> *"Reducing duplicated research is our highest-stated priority."*

Quote it back to them in the opening minute.

---

## Hero theme: Solid-state batteries and suppliers

**Why this theme:**
- Active Hyundai Mobis investment area (publicly disclosed JV with LG Energy Solution, internal R&D programs).
- Rich public corpus: supplier press releases, trade press, patents (Espacenet, USPTO).
- Clear 2022–2025 evolution — perfect for time-travel demo (early Toyota optimism → 2024 delays → 2025 production guidance shifts).
- Real public contradictions in the wild (Toyota's solid-state production timelines have been revised multiple times; CATL vs QuantumScape energy density claims).

**Anchors to add to schema** (extending `data/automotive_schema.json`):
- Suppliers: CATL, LG Energy Solution, Samsung SDI, SK On, Panasonic, QuantumScape, SES AI, Solid Power, ProLogium
- OEMs: Toyota, Hyundai, Hyundai Mobis, Volkswagen, Stellantis, BMW, Nissan
- Features: solid_state, sulfide_electrolyte, oxide_electrolyte, polymer_electrolyte, lithium_metal_anode, energy_density, cycle_life, TRL
- Relations: invest, partner, supply, develop, certify, delay, ramp_production, license

---

## 30-minute flow

| Min  | Segment                          | Win served            | Notes |
|------|----------------------------------|-----------------------|-------|
| 0–3  | **Their pain, their words**      | Framing               | One slide. Quote them back. "Imagine an analyst saving 3 hours a week not redoing research." |
| 3–5  | **What infon is, in one diagram**| Tech credibility      | `Infon-Engine.png` + one-line positioning vs RAG. |
| 5–14 | **Hero query (live)**            | Tech is real          | Chat: *"Who are Hyundai's solid-state battery partners and what do we know about their timelines?"* Graph viewer updates live with the fact subgraph. Click a citation. |
| 14–18| **Time-travel**                  | Only-infon-can-do     | Same anchor `solid_state_battery` at 2023 snapshot vs 2025 snapshot. Show `trajectory("toyota")` — early production claims → delays. |
| 18–21| **Contradiction + "I don't know"** | Only-infon-can-do   | Two real public sources that disagree on Toyota's SSB timeline. Infon flags it with `(supports, refutes, θ)`. Then ask an out-of-corpus question — show θ → 1.0. *Optional 30s insert: same query to ChatGPT, watch it fabricate confidently.* |
| 21–26| **Path to your 10-week PoC**     | Plan is credible      | Phased table (rescoped to 10 weeks — see below). One slide: "what week 1 actually looks like" (preprocessor + glossary-driven schema bootstrap + 50-doc sample ingest). Show HITL review queue mockup. |
| 26–30| Q&A → soft close                 | Conversion            | Cue them on next steps from recs doc §"Immediate Next Steps". |

---

## 10-week PoC rescope (supersedes 20-week plan in recommendations doc)

The original recommendations doc had three phases summing to 20 weeks. Rescoped to 10:

| Weeks | Phase | Scope |
|-------|-------|-------|
| 1–4   | **Foundation** | Document preprocessor (PPT/PDF/DOCX). Schema v1 bootstrap from Mobis glossary + sample-corpus discovery. Ingestion of **150–300 client-curated documents** (smaller than the original 200–500). Basic query API. **Milestone:** first hero query answered on their docs with citations. |
| 5–8   | **Governance + Validation + KPI** | HITL review queue UI. Cassette-level security classification + role-based query filter. Audit log. LLM synthesis layer (prose answers over verified facts). KPI dashboard. Schema v2 refinement from review feedback. **Milestone:** domain experts using review queue daily; KPI dashboard shows validated-fact growth. |
| 9–10  | **PoC report + decision** | Measured KPIs against agreed targets. PoC report. Production-contract decision point. |

**Explicitly out of scope for the 10-week PoC** (push to production contract / Phase 4):
- Full corpus ingestion (stays at 150–300 curated docs).
- External connectors (patents, news feeds).
- SSO integration with Mobis IdP.
- Historical-tracking UI (the capability exists; the dedicated timeline UI is a Phase 4 deliverable).
- Korean-language extraction (English-only for PoC; clearly flag this for them).

**Why this is honest, not a downgrade pitch:** the 10-week scope still delivers everything they need to evaluate the system on their own knowledge — including human-in-the-loop validation, governance, and KPIs. The cut items are *scaling* items, not *capability* items. Frame it that way.

**Risk this introduces:** schema v2 quality. With only 4 weeks of usage feedback (weeks 5–8), schema refinement is shallower than the original plan. Mitigations: aggressive sample-corpus schema discovery in week 1; weekly schema review checkpoint; defer non-critical anchor additions to post-PoC.

---

## Stage layout: chat + live graph viewer

Two-pane app on a single screen:

```
┌────────────────────────────┬──────────────────────────────────┐
│  ANALYST CHAT              │  FACT SUBGRAPH                    │
│  (natural language input)  │  (interactive, click nodes)       │
│                            │                                   │
│  > Who are Hyundai's SSB   │      ┌──────────┐                 │
│    partners?               │      │ Hyundai  │                 │
│                            │      └────┬─────┘                 │
│  Hyundai partners with     │           │ partner               │
│  LG Energy Solution on     │      ┌────▼─────┐                 │
│  solid-state cells [1].    │      │  LG ES   │                 │
│  SK On supplies pouch...   │      └────┬─────┘                 │
│                            │           │ supply                │
│  [1] LG Press, 2024-03-12  │      ┌────▼─────┐                 │
│  [2] Reuters, 2024-09-04   │      │   SSB    │                 │
├────────────────────────────┴──────────────────────────────────┤
│  CITATIONS: [click to open source]                            │
└───────────────────────────────────────────────────────────────┘
```

### Recommended stack

**Decided: Streamlit + streamlit-agraph.** Considered and ruled out: React (would seed Layer 7 but eats the week given current build resources), vanilla HTML + FastAPI (good middle ground but no reusability win under throwaway constraint), native macOS (wrong audience — Mobis is Windows-heavy, plus pure throwaway).

- Streamlit chat component for the left pane.
- `streamlit-agraph` wraps vis.js — interactive, physics-based.
- After each `Analyst.ask()` returns, render the returned fact-chain triples as nodes/edges.
- Single Python file, deployable locally or via Streamlit Cloud for the live demo.

**Non-optional polish (Streamlit's default look is the main risk):**
- Custom CSS hides hamburger menu and "Made with Streamlit" footer.
- Wide layout, hand-tuned columns (~40% chat / ~60% graph).
- Custom theme: clean palette, no defaults; real font (Inter or similar from Google Fonts).
- streamlit-agraph styling: node colors per anchor type (actor / feature / relation), physics tuned to settle quickly without bouncing, sensible default zoom.
- Citation panel styled as cards, not a bullet list.
- Replace default Streamlit spinner with a custom "thinking…" indicator.

**Fallback (if streamlit-agraph integration breaks by Wed EOD):**
- Jupyter notebook with chat-style markdown output + PyVis writing `graph.html` to a second browser tab that auto-refreshes. Less polished but ships in hours.

**What we are NOT building:**
- A full UI. The roadmap slide shows the "real product UI" as a Figma mock.
- A persistent app. Demo-only; tear down or keep as `reference_v4/mobis_demo/`.

---

## Prep workstreams (Mon → Thu)

Tracked as Beads tasks #1–9. Critical path:

| Day | Workstream |
|-----|-----------|
| Mon | Corpus curation (#2). Schema extension (#3). |
| Tue | Two-snapshot ingest (#4). Validate `trajectory()` and contradiction queries hit. Start demo app (#5). |
| Wed | Demo app polished (#5). Slides + script draft (#6). HITL mockup (#7). |
| Thu | Dry run with timer (#9). Cut overrun. Optional LLM comparison clip (#8). |
| Fri | Demo. |

**Gating decision Wed EOD:** Streamlit app working end-to-end? If not, drop to Jupyter+PyVis fallback. Do not push into Thursday — Thursday is rehearsal.

---

## Queries to script (draft — to be refined)

These need to *actually run* on the curated corpus. Validation that they return non-empty results is the gating check before Friday.

1. **Hero (cross-document discovery):**
   `"Who are Hyundai Mobis's solid-state battery partners and what timelines have they publicly committed to?"`
   → Expected: ≥3 partners, ≥2 timeline claims, citations across ≥4 docs.

2. **Time-travel A (2023 view):**
   `store.trajectory("toyota").before("2024-01-01")` framed as *"What did the public record say about Toyota's solid-state plans in 2023?"*
   → Expected: optimistic 2025 production targets.

3. **Time-travel B (2025 view):**
   Same anchor, current snapshot. → Expected: pushed to 2027–2030, partnerships shifted.

4. **Contradiction:**
   `Query().where(s="toyota", p="ramp_production", o="solid_state").contradicting()`
   → Expected: surfaces 2 docs with opposing polarity. Verdict shows non-zero refute mass.

5. **"I don't know":**
   `"What is Hyundai Mobis's planned solid-state cathode chemistry?"` (something genuinely not in the corpus)
   → Expected: θ → 1.0. Analyst responds "the corpus does not contain a verified answer."

6. **Connectivity (multi-hop):**
   `store.connect("hyundai_mobis", "quantumscape")` → expected: chain via Volkswagen or shared standard, OR honest no-path.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Schema doesn't extract well on real auto news (synthetic-corpus overfit) | Run `extraction_report()` Monday after first ingest; budget time Tue for anchor/token tuning |
| Graph viewer doesn't update cleanly per query | Fallback to static graph rendered after each query; commit to one path Wed |
| Live demo fails in front of client | Pre-record a 60-second backup screencast of each key segment; switch to it gracefully if needed |
| Contradiction query returns empty | Pick the contradicting docs **first**, build the schema around them, not the other way around |
| Out-of-corpus query *accidentally* returns a hit | Pre-test the θ→1.0 query the day before; have a known-safe backup question ready |
| Korean-language docs requested | We don't support them yet. Pre-empt: "current corpus is English; multilingual is Phase 3 with their internal terminology". |
| On-prem / data security comes up | Crisp answers ready: S3-compatible (any object store), cassettes are portable, no LLM API calls in extraction (frozen 4.4M-param SPLADE on CPU), full air-gap deployable |

---

## What we are deliberately NOT showing

- **The benchmark numbers (HoVer, FEVER, SciFact)** — academic, not their world. Mention only if asked.
- **Sheaf GNN internals** — paper-level detail. Reference if a technical person asks "how does the calibration work."
- **Layer 4–6 (HITL queue, access control, KPI dashboard)** as *live* features — they're gaps. Show only as mockups on the roadmap slide.
- **Multiple verticals (biomedical, legal)** — keep the SaaS pitch implicit; this meeting is about *their* engagement.

---

## Soft-close: next steps (from recommendations doc §"Immediate Next Steps")

Land these in Q&A or in the follow-up email:

1. Scoping call: agree document corpus scope, security requirements, existing systems (SAP, PLM), KPI definitions.
2. Request sample corpus: 50–100 documents across key knowledge areas.
3. Request internal glossary or taxonomy.
4. Agree PoC success criteria in writing.

---

## Open decisions / still to resolve

- [ ] Korean-language anything needed on Friday? (Slides? Greeting?)
- [ ] Are we presenting in person or video call? Affects whether we can use a second monitor for the graph view.
- [ ] Final call on whether to include the side-by-side ChatGPT comparison clip — it's leverage but it can come across as combative if mis-pitched.
- [ ] Who's on the line from our side? Solo demo or partner-driven?
- [ ] What's the actual list of Hyundai Mobis attendees and their roles? Tailor the "what week 1 looks like" slide to whoever owns budget approval.
