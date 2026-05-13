"""Analyst — a Strands-powered conversational layer over InfonStore.

Commit 2 of the UX plan: minimum viable agent. Three tools, one system
prompt. The agent orchestrates; the cassette stack does the work.

The split is deliberate:
  • LLM never touches belief calculation. It can call `ask()`, but the
    Verdict returned is the reasoner's, not the model's interpretation.
  • LLM never fabricates triples. All claims cite `sources[]` from the
    tool's output; the system prompt demands it.
  • LLM does the fuzzy work: translating "Did Nvidia partner with TSMC?"
    to ask(subject="nvidia", predicate="partner", object="tsmc"),
    spotting schema gaps in the extraction report, proposing reformulated
    queries when θ is high.

Usage:
    from cognition.cassette import InfonStore
    from cognition.cassette.analyst import Analyst

    store = InfonStore("./data/chips", schema_path="schema.json")
    a = Analyst(store)
    print(a.chat("Here are 15 news snippets: [...]. Help me analyze them."))

If you already ingested, just ask:
    print(a.chat("Does Nvidia partner with TSMC?"))
"""

from __future__ import annotations

import json
from typing import Any

try:
    from strands import tool, Agent
    _STRANDS_AVAILABLE = True
except ImportError:
    _STRANDS_AVAILABLE = False
    def tool(fn):
        fn.is_tool = True
        return fn

from .store import InfonStore
from .dsl import Query


# Module-level store reference — each tool call reaches through here.
# This matches the existing agent_tools.py pattern and avoids closures
# over the store (which Strands doesn't serialize cleanly).
_STORE: InfonStore | None = None


# ═══════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════

@tool
def ingest(documents_json: str) -> str:
    """Ingest a batch of documents into the store.

    Pass a JSON-encoded array of {"id", "text", "timestamp"} objects.
    Returns a summary including the extraction report (coverage
    diagnostics). If coverage is poor (many zero-infon docs, unused
    anchors, overfit objects), use extraction_report() for detail and
    suggest schema edits to the user — do NOT proceed with queries until
    coverage is acceptable.

    Args:
        documents_json: JSON array of document objects, each with keys
            'id' (str), 'text' (str), 'timestamp' (optional, ISO date).
    """
    docs = json.loads(documents_json)
    result = _STORE.ingest(docs)
    report = result.get("report")
    out = {
        "ingested": len(result["ingested"]),
        "skipped": len(result["skipped"]),
        "errors": len(result["errors"]),
        "n_infons": result["n_infons"],
        "snapshot_id": result["snapshot_id"],
    }
    if report:
        out["report_summary"] = report.summary()
    return json.dumps(out, indent=2)


@tool
def extraction_report() -> str:
    """Coverage + quality diagnostics for the current store.

    Use this immediately after ingest, or anytime a query returns NEI
    unexpectedly — the report will surface whether the schema failed to
    match the corpus. The four categories to watch:

      docs_with_zero_infons — sentences with no extracted triples
      unused_anchors         — schema anchors that never fired
      overfit_objects        — one anchor dominating the object role
      role_imbalance         — anchors appearing in the wrong slot type

    Returns a JSON object; the 'summary' field is a human-readable line
    per issue.
    """
    report = _STORE.extraction_report()
    body = report.to_dict()
    body["summary"] = report.summary()
    return json.dumps(body, indent=2)


@tool
def set_schema(ontology_json: str) -> str:
    """Activate a new ontology on the store. Does NOT re-extract prior docs.

    Use this when:
      • Starting fresh with a proposed ontology.
      • Revising after extraction_report() flagged coverage issues.

    After calling set_schema(), any NEW ingest() calls will tag cassettes
    with the new schema. To re-extract existing docs under the new schema,
    call reingest() next.

    Ontology shape — a JSON object mapping anchor name → {type, tokens}:

        {
          "toyota":   {"type": "actor",    "tokens": ["toyota"]},
          "invest":   {"type": "relation", "tokens": ["invest", "invested"]},
          "batteries":{"type": "feature",  "tokens": ["battery", "batteries"]}
        }

    Valid types: actor (subjects), relation (predicates), feature / market /
    location (objects). Tokens are surface forms found in text.

    Design guidance:
      • Lists of tokens should include realistic surface variants (verb
        tenses, plural forms, common abbreviations).
      • Multiple feature anchors relieve object-slot competition better
        than one catch-all.
      • Actors and features never share the same name — pick one role.

    Args:
        ontology_json: JSON object as a string. Must parse to a dict mapping
            anchor name to an object with keys 'type' and 'tokens' (list).
    """
    schema = json.loads(ontology_json)
    path = _STORE.set_schema_from_dict(schema)
    n_anchors = len(_STORE.schema.names)
    by_type: dict[str, int] = {}
    for t in _STORE.schema.types.values():
        by_type[t] = by_type.get(t, 0) + 1
    return json.dumps({
        "schema_ref": _STORE.schema_ref,
        "schema_path": path,
        "n_anchors": n_anchors,
        "by_type": by_type,
    }, indent=2)


@tool
def reingest() -> str:
    """Re-extract every known document under the currently-active schema.

    Use after set_schema() to migrate previously-ingested docs to a new
    ontology. Old cassettes stay (tagged with the old schema_ref); new
    cassettes appear tagged with the new schema. The active manifest
    tracks both, so queries run against whichever matches the active
    schema when the reasoner looks up anchors.

    Returns ingest counts + a fresh extraction_report so you can compare
    coverage before vs. after the schema change.
    """
    result = _STORE.reingest()
    report = result.get("report")
    out = {
        "ingested": len(result["ingested"]),
        "skipped": len(result["skipped"]),
        "errors": len(result["errors"]),
        "n_infons": result["n_infons"],
        "snapshot_id": result["snapshot_id"],
    }
    if report:
        out["report_summary"] = report.summary()
    return json.dumps(out, indent=2)


@tool
def connect(source: str, target: str, max_hops: int = 3) -> str:
    """Find whether two entities are linked via a chain of relations.

    Use when the user asks about indirect relationships — "Is X connected
    to Y?", "Does A touch B somehow?", "Is there a path from P to Q?".
    The reasoner searches for chains up to `max_hops` long (default 3)
    where each hop is a connective edge (supply/partner/license/etc.).
    It's polarity-aware: a retracted edge on the chain makes the verdict
    REFUTES, not SUPPORTS.

    Do NOT use this for one-hop relationships — ask() is cheaper and
    more direct for those (~2 range gets vs. ~8-20).

    Args:
        source: starting entity anchor (e.g. "nvidia").
        target: ending entity anchor (e.g. "anthropic").
        max_hops: maximum chain length. Stays 3 or less in practice;
            each hop bleeds confidence to θ, so 4+ hops usually return NEI.

    Returns: Verdict with label, DS mass, and the sequence of infons
    along the best chain to target (each element is one edge).
    """
    verdict = _STORE.connect(source, target, max_hops=max_hops)
    m = verdict.mass
    return json.dumps({
        "label": verdict.label,
        "mass": {
            "supports": round(m.supports, 3),
            "refutes": round(m.refutes, 3),
            "theta": round(m.theta, 3),
        },
        "n_hops": len(verdict.sources),
        "chain": [
            {"sentence": s.sentence, "doc_id": s.doc_id,
             "polarity": s.polarity,
             "timestamp": s.timestamp,
             "triple": f"{s.subject}/{s.predicate}/{s.object}"}
            for s in verdict.sources
        ],
        "range_gets": verdict.range_gets,
    }, indent=2)


@tool
def any_of(source: str, targets_json: str, max_hops: int = 3) -> str:
    """Find which of multiple targets is reachable from source.

    Use when the user asks "which of {A, B, C, ...} does X relate to?" —
    for example "which battery supplier is Toyota connected to?" or
    "which of my competitors touches this chain?". One tree walk
    resolves all targets at roughly the same cost as a single connect().

    Don't call this for a single target — use connect() instead.

    Args:
        source: starting entity anchor.
        targets_json: JSON array of target anchor names, e.g.
            '["catl", "lg", "samsung", "sk_hynix"]'.
        max_hops: chain length cap. Same semantics as connect().

    Returns: per-target Verdict dict with label + mass + chain. Targets
    with no reachable chain come back as NOT_ENOUGH_INFO with θ=1.
    """
    targets = set(json.loads(targets_json))
    verdicts = _STORE.any_of(source, targets, max_hops=max_hops)
    # Sort by decisiveness so the most interesting targets surface first.
    items = [
        (t, v, abs(v.mass.supports - v.mass.refutes))
        for t, v in verdicts.items()
    ]
    items.sort(key=lambda x: -x[2])

    out = {
        "source": source,
        "shared_range_gets": next(iter(verdicts.values())).range_gets
                             if verdicts else 0,
        "results": [
            {
                "target": t,
                "label": v.label,
                "mass": {
                    "supports": round(v.mass.supports, 3),
                    "refutes": round(v.mass.refutes, 3),
                    "theta": round(v.mass.theta, 3),
                },
                "n_hops": len(v.sources),
                "chain": [
                    {"sentence": s.sentence,
                     "triple": f"{s.subject}/{s.predicate}/{s.object}",
                     "polarity": s.polarity,
                     "timestamp": s.timestamp}
                    for s in v.sources
                ],
            }
            for t, v, _ in items
        ],
    }
    return json.dumps(out, indent=2)


@tool
def record_finding(title: str, body: str, tags_json: str = "[]",
                    cites_json: str = "[]") -> str:
    """Save a persistent note under <root>/findings/ for future sessions.

    Use for synthesis, not raw tool output. Good findings:
      • "Main supply chain in this corpus: TSMC makes Nvidia chips;
         SK Hynix + Samsung supply HBM; Samsung deal failed later."
      • "Schema lesson: the 'partner' relation needs 'venture' and
         'alliance' tokens — without them we miss ~20% of partnerships."

    Bad findings (just recompute these):
      • Raw ask() output
      • Full extraction reports
      • Trajectories

    Args:
      title: short human-readable title.
      body: markdown OK; explain what you found and why it matters.
      tags_json: JSON array of tag strings (e.g. '["supply_chain","q2"]').
      cites_json: JSON array of citation objects, each with 'infon_id',
        'cassette_id', and 'sentence'. Let the user connect the finding
        back to the evidence that supports it.
    """
    tags = json.loads(tags_json) if tags_json else []
    cites = json.loads(cites_json) if cites_json else []
    f = _STORE.record_finding(title=title, body=body, tags=tags, cites=cites)
    return json.dumps({
        "id": f.id,
        "title": f.title,
        "tags": f.tags,
        "n_cites": len(f.cites),
        "snapshot_id": f.snapshot_id,
    }, indent=2)


@tool
def list_findings(tag: str = "", limit: int = 20) -> str:
    """Read back previously-recorded findings (cross-session memory).

    Call this at the START of a new investigation to see what's been
    learned before — don't re-derive conclusions the user already wrote.

    Args:
      tag: filter to findings with this tag (empty = all).
      limit: how many to return, newest first.
    """
    tag_filter = tag if tag else None
    findings = _STORE.findings(tag=tag_filter, limit=limit)
    return json.dumps({
        "count": len(findings),
        "findings": [
            {"id": f.id, "created_at": f.created_at,
             "title": f.title, "tags": f.tags,
             "snapshot_id": f.snapshot_id,
             "body_preview": f.body[:240]
                              + ("..." if len(f.body) > 240 else "")}
            for f in findings
        ],
    }, indent=2)


@tool
def ask(subject: str = "", predicate: str = "", object: str = "",
        polarity: int = 1) -> str:
    """Ask a single-claim question and get a calibrated verdict.

    Returns a Verdict with: label (SUPPORTS/REFUTES/NOT_ENOUGH_INFO),
    a Dempster-Shafer mass (supports/refutes/theta), and the source
    infons that drove the decision.

    Guidelines when consuming the result:
      • θ > 0.7 means the corpus doesn't answer the question. Say so —
        do NOT fill in with the LLM's own knowledge.
      • When REFUTES, the sources will include a negated infon. Cite it
        as the reason.
      • When SUPPORTS, cite the top 1-3 source sentences verbatim.

    Args:
        subject: Subject anchor name (e.g. "nvidia"). At least one of
            subject/predicate/object must be provided.
        predicate: Predicate anchor name (e.g. "partner").
        object: Object anchor name (e.g. "tsmc").
        polarity: 1 for affirmed claim (default), 0 for negated form.
    """
    q = Query()
    if subject: q = q.where(subject=subject)
    if predicate: q = q.where(predicate=predicate)
    if object: q = q.where(object=object)
    if polarity == 0: q = q.negated()
    else: q = q.affirmed()

    verdict = _STORE.ask(q)
    m = verdict.mass
    return json.dumps({
        "label": verdict.label,
        "mass": {
            "supports": round(m.supports, 3),
            "refutes": round(m.refutes, 3),
            "theta": round(m.theta, 3),
        },
        "n_sources": len(verdict.sources),
        "sources": [
            {"sentence": s.sentence, "doc_id": s.doc_id,
             "confidence": round(s.confidence, 2),
             "polarity": s.polarity,
             "timestamp": s.timestamp}
            for s in verdict.sources[:5]
        ],
        "range_gets": verdict.range_gets,
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an analyst helping a user investigate a corpus of documents through \
a calibrated knowledge graph. You have nine tools:

1. set_schema(ontology_json) — activate a new ontology.
2. ingest(documents_json) — extract triples from a batch of documents.
3. reingest() — re-extract all known docs under the current schema.
4. extraction_report() — coverage diagnostics; call after every ingest.
5. ask(subject, predicate, object, polarity) — single claim (one hop) → \
calibrated verdict + cited sources.
6. connect(source, target, max_hops) — multi-hop chain between two \
entities → verdict + the edge sequence.
7. any_of(source, targets_json, max_hops) — which of many targets is \
reachable from source → per-target verdicts from ONE tree walk.
8. record_finding(title, body, tags_json, cites_json) — save a cross- \
session note to <root>/findings/ (synthesis, not raw output).
9. list_findings(tag, limit) — read previously-recorded findings.

Tool-selection guidance for questions:

  • "Did X verb Y?" → ask(X, verb, Y). Cheapest, most direct.
  • "Is X connected/linked to Y?" / "Does X touch Y (somehow)?" → \
connect(X, Y). Follows partnership/supply/acquire/license chains.
  • "Which of {A, B, C, ...} does X relate to?" → any_of(X, [A,B,C]). \
Resolves all at once; one tree walk, NOT N separate calls.
  • "Show all infons about X" → ask() with only subject=X.

Never use connect() for a one-hop question that ask() would answer \
directly — it's several times more expensive.

BEFORE ANYTHING ELSE:
  1. Call list_findings() to see what past sessions concluded. If the \
current question overlaps a past finding, reference that finding instead \
of re-deriving from scratch. Old findings may be out-of-date; verify \
briefly with a fresh ask() if the snapshot has moved.
  2. Call extraction_report() to see what corpus is loaded. If cassettes \
exist, proceed to ANSWER-QUESTIONS — do not re-ask for documents. If \
empty (0 infons), ask the user for documents.

AT THE END of a meaningful investigation, call record_finding() with the \
synthesized conclusion, relevant citations, and tags. This is what \
makes the store feel like it remembers — future sessions will see your \
finding on list_findings(). Do NOT record trivial findings (single-ask \
verdicts, raw extraction reports).

Your workflow has two phases: BUILD-SCHEMA and ANSWER-QUESTIONS.

══ PHASE 1: BUILD-SCHEMA ══

  If the user has NO schema or an obviously-broken one, propose one by \
reading their documents and identifying:
    • entities (companies, people, products) → 'actor' type
    • relationships (verbs, predicates) → 'relation' type
    • features/targets/objects → 'feature' type

  For each anchor, include realistic surface forms in 'tokens':
    • verb tenses: 'partner', 'partnered', 'partnership'
    • abbreviations: 'catl', '3nm', 'tpu'
    • common misspellings or casing variants

  Set AT LEAST 2 feature anchors even if the corpus seems narrow — a \
single feature hoards the object slot and blocks discrimination. Add \
sibling features to relieve competition.

  Then: set_schema → ingest → extraction_report.

  If the report shows coverage problems:
    • >25% docs with zero infons
    • >40% unused anchors
    • any object anchor >70% of object slots

  Revise the schema and call set_schema → reingest → extraction_report \
again. Continue until coverage is acceptable or you've iterated 3 times \
(don't loop forever — report limitations instead).

  Announce each iteration clearly so the user sees progress: \
"Iteration 2: added 'license' and 'acquire' tokens, coverage up from \
40% to 73%."

══ PHASE 2: ANSWER-QUESTIONS ══

  When answering a user's question:
    1. Translate their natural-language claim into a triple (s, p, o).
    2. Call ask() with those anchors.
    3. Report the label, mass, and SOURCES. Cite sentences verbatim.
    4. If θ > 0.7, explicitly say the corpus doesn't answer the question. \
Do NOT fill in from your own knowledge.

  If the first ask() returns NEI with high θ, try ONE reformulation \
(swap subject/object, drop the object). If still NEI, report honestly.

══ UNIVERSAL RULES ══

  • Never invent facts.
  • Never output a verdict without citing the sources returned by ask().
  • When grading your own output, distinguish 'what the graph knows' from \
'what the docs say' — if coverage failed, tell the user the graph can't \
answer even when the answer exists in the raw text.
"""


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

class Analyst:
    """Strands Agent wrapping an InfonStore.

    Spawns a Strands `Agent` with the three cassette tools and the
    investigator system prompt. Each Analyst instance is bound to one
    InfonStore — swap stores by making a new Analyst.

    Args:
      store: InfonStore (local or S3-backed).
      model: optional Strands model or model-id string. Defaults to
        whatever Strands picks (BedrockModel if AWS creds are present,
        otherwise raises).
      extra_tools: additional @tool functions to expose. The three
        cassette tools are always included.
    """

    def __init__(self, store: InfonStore, *, model: Any = None,
                 extra_tools: list | None = None,
                 stream: bool = True):
        global _STORE
        if not _STRANDS_AVAILABLE:
            raise ImportError(
                "Analyst requires strands-agents. "
                "Install with: pip install strands-agents"
            )
        _STORE = store
        self.store = store

        tools = [set_schema, ingest, reingest, extraction_report,
                 ask, connect, any_of,
                 record_finding, list_findings]
        if extra_tools:
            tools.extend(extra_tools)

        # stream=True (default): Strands prints tokens to stdout as they
        # arrive, showing tool calls and thinking live. Good for REPL.
        # stream=False: no stdout side-effects — chat() returns the text
        # only. Set False when piping output or using Analyst as a library.
        agent_kwargs: dict = dict(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        if not stream:
            # Disable the default callback handler's streaming prints.
            agent_kwargs["callback_handler"] = lambda **_: None

        self.agent = Agent(**agent_kwargs)

    def chat(self, message: str) -> str:
        """Send a single turn to the agent and return its response.

        Strands streams tool calls + tool results + text blocks to stdout
        via the default callback handler. We capture only the final text
        response so this function's return is programmatically usable.
        Silence the side-channel printing with a null callback if the
        caller wants clean stdout.
        """
        result = self.agent(message)
        # AgentResult.message is {"content": [blocks], "role": "assistant"}.
        # Final text lives in the text blocks; tool calls are separate types.
        if hasattr(result, "message"):
            msg = result.message
            if isinstance(msg, dict) and "content" in msg:
                parts = [b["text"] for b in msg["content"]
                         if isinstance(b, dict) and "text" in b]
                return "\n".join(parts)
            return str(msg)
        return str(result)

    def __call__(self, message: str) -> str:
        return self.chat(message)
