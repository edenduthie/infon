"""Agent tools for LLM-based navigation of the cognition knowledge graph.

Provides @tool-decorated functions that an LLM agent can call to query
infons, constraints, timelines, and evidence.

Usage:
    from cognition.agent_tools import create_tools

    tools, system_prompt = create_tools(cognition_instance)
    agent = Agent(model=model, system_prompt=system_prompt, tools=tools)
"""

from __future__ import annotations

from collections import defaultdict

try:
    from strands import tool
except ImportError:
    # Fallback: define a no-op @tool decorator
    def tool(fn):
        fn.is_tool = True
        return fn


# Module-level state — set by create_tools()
_cog = None


@tool
def list_anchors(anchor_type: str = "") -> str:
    """List all anchors in the schema, optionally filtered by type.

    Types: actor, relation, feature, market, location, temporal, variant.

    Args:
        anchor_type: Filter by type. Empty = all types.
    """
    schema = _cog.schema
    by_type = defaultdict(list)
    for name in schema.names:
        atype = schema.types.get(name, "unknown")
        if anchor_type and atype != anchor_type:
            continue
        by_type[atype].append(name)

    if not by_type:
        return f"No anchors found{' of type ' + anchor_type if anchor_type else ''}."

    lines = []
    for atype in sorted(by_type):
        names = ", ".join(sorted(by_type[atype]))
        lines.append(f"{atype} ({len(by_type[atype])}): {names}")
    return "\n".join(lines)


@tool
def find_infons(subject: str = "", predicate: str = "", object: str = "",
                min_importance: float = 0.0, limit: int = 20) -> str:
    """Find infons matching anchor role filters.

    Args:
        subject: Filter by subject anchor (e.g. "toyota").
        predicate: Filter by predicate anchor (e.g. "invest").
        object: Filter by object anchor (e.g. "solid_state").
        min_importance: Minimum importance score.
        limit: Maximum results.
    """
    infons = _cog.store.query_infons(
        subject=subject or None,
        predicate=predicate or None,
        object=object or None,
        min_importance=min_importance,
        limit=limit,
    )
    if not infons:
        parts = []
        if subject: parts.append(f"subject={subject}")
        if predicate: parts.append(f"predicate={predicate}")
        if object: parts.append(f"object={object}")
        return f"No infons found matching {', '.join(parts) if parts else 'criteria'}."

    lines = [f"Infons ({len(infons)} results):"]
    for inf in infons:
        pol = "+" if inf.polarity else "-"
        lines.append(
            f"  {pol}<<{inf.predicate}, {inf.subject}, {inf.object}>> "
            f"conf={inf.confidence:.3f} imp={inf.importance:.3f}"
        )
        if inf.sentence:
            sent_preview = inf.sentence[:80] + "..." if len(inf.sentence) > 80 else inf.sentence
            lines.append(f"    \"{sent_preview}\"")
    return "\n".join(lines)


@tool
def get_evidence(subject: str, predicate: str, object: str) -> str:
    """Get evidence sentences for a specific (subject, predicate, object) triple.

    Args:
        subject: Subject anchor name.
        predicate: Predicate anchor name.
        object: Object anchor name.
    """
    infons = _cog.store.query_infons(
        subject=subject, predicate=predicate, object=object, limit=50,
    )
    if not infons:
        return f"No evidence for ({subject}, {predicate}, {object})."

    lines = [
        f"Evidence for ({subject}, {predicate}, {object}): {len(infons)} infons",
        "",
    ]
    for inf in infons:
        lines.append(f"  [{inf.doc_id}] conf={inf.confidence:.3f}")
        lines.append(f"    \"{inf.sentence}\"")
        support = inf.support
        if support:
            lines.append(f"    grounding: S={support.get('subject','?')} "
                        f"P={support.get('predicate','?')} O={support.get('object','?')}")
    return "\n".join(lines)


@tool
def get_constraints(subject: str = "", predicate: str = "", object: str = "",
                    min_score: float = 0.0, limit: int = 20) -> str:
    """Get aggregated constraints (claims across multiple infons).

    Constraints represent corpus-level assertions with evidence counts.

    Args:
        subject: Filter by subject.
        predicate: Filter by predicate.
        object: Filter by object.
        min_score: Minimum constraint score.
        limit: Maximum results.
    """
    constraints = _cog.store.get_constraints(
        subject=subject or None,
        predicate=predicate or None,
        object=object or None,
        min_score=min_score,
        limit=limit,
    )
    if not constraints:
        return "No constraints found."

    lines = [f"Constraints ({len(constraints)} results):"]
    for c in constraints:
        lines.append(
            f"  ({c.subject}, {c.predicate}, {c.object}) "
            f"score={c.score:.3f} evidence={c.evidence} "
            f"docs={c.doc_count} persistence={c.persistence}"
        )
    return "\n".join(lines)


@tool
def get_timeline(anchor: str, limit: int = 30) -> str:
    """Get the temporal timeline for an anchor — infons ordered by time.

    Shows how an entity/relation evolves over time through its infons.

    Args:
        anchor: Anchor name to get timeline for.
        limit: Maximum infons to show.
    """
    infons = _cog.store.get_infons_for_anchor(anchor, limit=limit)
    timestamped = [inf for inf in infons if inf.timestamp]
    timestamped.sort(key=lambda inf: inf.timestamp)

    if not timestamped:
        return f"No timestamped infons for '{anchor}'."

    lines = [f"Timeline for '{anchor}' ({len(timestamped)} infons):"]
    for inf in timestamped:
        pol = "+" if inf.polarity else "-"
        lines.append(
            f"  {inf.timestamp}  {pol}<<{inf.predicate}, {inf.subject}, {inf.object}>> "
            f"conf={inf.confidence:.3f}"
        )
    return "\n".join(lines)


@tool
def compare_entities(entity_a: str, entity_b: str) -> str:
    """Compare two entities by their infon profiles.

    Shows constraints unique to each and shared between them.

    Args:
        entity_a: First entity name.
        entity_b: Second entity name.
    """
    cons_a = {(c.predicate, c.object): c
              for c in _cog.store.get_constraints(subject=entity_a, limit=50)}
    cons_b = {(c.predicate, c.object): c
              for c in _cog.store.get_constraints(subject=entity_b, limit=50)}

    shared = set(cons_a.keys()) & set(cons_b.keys())
    only_a = set(cons_a.keys()) - shared
    only_b = set(cons_b.keys()) - shared

    lines = [f"Comparison: {entity_a} vs {entity_b}", ""]

    if shared:
        lines.append(f"Shared ({len(shared)}):")
        for pred, obj in sorted(shared):
            sa = cons_a[(pred, obj)].score
            sb = cons_b[(pred, obj)].score
            lines.append(f"  ({pred}, {obj}): {entity_a}={sa:.3f} vs {entity_b}={sb:.3f}")

    if only_a:
        lines.append(f"\nOnly {entity_a} ({len(only_a)}):")
        for pred, obj in sorted(only_a):
            lines.append(f"  ({pred}, {obj}) score={cons_a[(pred, obj)].score:.3f}")

    if only_b:
        lines.append(f"\nOnly {entity_b} ({len(only_b)}):")
        for pred, obj in sorted(only_b):
            lines.append(f"  ({pred}, {obj}) score={cons_b[(pred, obj)].score:.3f}")

    if not shared and not only_a and not only_b:
        lines.append("No constraints found for either entity as subject.")

    return "\n".join(lines)


@tool
def ask_cognition(question: str, persona: str = "") -> str:
    """Ask a natural language question and get a grounded answer.

    Runs the full cognition query pipeline: encode → retrieve → score.
    Returns top infons with valence scoring.

    Args:
        question: Natural language question.
        persona: Override persona (investor, engineer, executive, regulator, analyst).
    """
    result = _cog.query(
        text=question,
        persona=persona or None,
        top_k=15,
    )

    lines = [
        f"Query: {result.query}",
        f"Persona: {result.persona}",
        f"Anchors activated: {len(result.anchors_activated)}",
        "",
        f"Top infons ({len(result.infons)}):",
    ]

    for inf in result.infons[:15]:
        pol = "+" if inf.polarity else "-"
        v = result.valence.get(inf.infon_id, 0)
        v_label = "pos" if v > 0.1 else "neg" if v < -0.1 else "neutral"
        lines.append(
            f"  {pol}<<{inf.predicate}, {inf.subject}, {inf.object}>> "
            f"imp={inf.importance:.3f} valence={v:+.2f}({v_label})"
        )
        if inf.sentence:
            sent_preview = inf.sentence[:80] + "..." if len(inf.sentence) > 80 else inf.sentence
            lines.append(f"    \"{sent_preview}\"")

    if result.constraints:
        lines.append(f"\nTop constraints ({len(result.constraints)}):")
        for c in result.constraints[:10]:
            lines.append(
                f"  ({c.subject}, {c.predicate}, {c.object}) "
                f"score={c.score:.3f} evidence={c.evidence}"
            )

    return "\n".join(lines)


@tool
def get_stats() -> str:
    """Get statistics about the knowledge graph.

    Shows infon count, constraint count, anchor count, and backend info.
    """
    s = _cog.stats()
    lines = [
        "Knowledge Graph Statistics:",
        f"  Infons: {s['infon_count']}",
        f"  Constraints: {s['constraint_count']}",
        f"  Sequences: {'yes' if s['has_sequences'] else 'no'}",
        f"  Anchors: {s['anchors']}",
        f"  Backend: {s['backend']}",
        f"  Model: {s['model']}",
    ]
    return "\n".join(lines)


# ── Tool list and builder ───────────────────────────────────────────────

TOOLS = [
    list_anchors,
    find_infons,
    get_evidence,
    get_constraints,
    get_timeline,
    compare_entities,
    ask_cognition,
    get_stats,
]


def create_tools(cognition_instance):
    """Initialize tools with a Cognition instance and return (tools, system_prompt).

    Args:
        cognition_instance: A Cognition instance with loaded model and store.

    Returns:
        (tools_list, system_prompt)
    """
    global _cog
    _cog = cognition_instance

    schema = cognition_instance.schema
    type_counts = defaultdict(int)
    for name in schema.names:
        type_counts[schema.types.get(name, "unknown")] += 1
    type_desc = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

    stats = cognition_instance.stats()

    prompt = f"""\
You are an analyst with access to a cognition knowledge graph built from
document analysis using SPLADE sparse encoding with anchor projection.

The graph contains {stats['infon_count']} infons (situation-semantic triples)
extracted from documents, with {stats['constraint_count']} aggregated constraints.

Schema: {type_desc}

Key concepts:
- Infon: a grounded <<predicate, subject, object; polarity>> triple extracted
  from a specific sentence, with confidence scores and importance ranking
- Constraint: an aggregated claim across multiple infons with evidence count,
  document count, persistence (time windows), and composite score
- Anchor: a typed vocabulary entry (actor, relation, feature, market, etc.)
  with hierarchy (parent/child) relationships
- Valence: query-time scoring relative to a persona (investor, engineer, etc.)
- NEXT edges: temporal chains linking infons through shared anchors

Your tools:
1. list_anchors — browse the anchor vocabulary by type
2. find_infons — search infons by subject, predicate, object
3. get_evidence — drill into evidence sentences for a triple
4. get_constraints — find aggregated corpus-level assertions
5. get_timeline — see temporal evolution of an anchor
6. compare_entities — compare constraint profiles of two entities
7. ask_cognition — natural language query with persona valence
8. get_stats — knowledge graph statistics

Strategy: start broad (list_anchors, get_stats), then drill down
(find_infons, get_evidence). Use ask_cognition for open-ended questions.
Always ground assertions in evidence sentences.
"""
    return TOOLS, prompt
