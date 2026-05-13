#!/usr/bin/env python3
"""ask.py — Query a cognition knowledge graph from the command line.

Two modes, both default-on so you see the full story for every query:

  1. Reasoner   — calibrated DS verdict with (S, R, U, θ). θ is explicit
                  ignorance: near 1.0 on claims the corpus doesn't support.
  2. Retrieval  — the older persona-valence view (same infons, ranked
                  by relevance + persona preference). Use --retrieval-only
                  to skip the reasoner.

Usage:
    python ask.py "Did Toyota invest in batteries?"
    python ask.py "Did Tesla acquire CATL?"          # high-θ answer
    python ask.py "What is Toyota investing in?" --persona investor
    python ask.py --ingest data/documents.json "What happened?"
    python ask.py --stats
    python ask.py --retrieval-only "Who works on batteries?"

If no knowledge graph exists yet, ingests the bundled demo corpus first.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from cognition import Cognition, CognitionConfig

# ── Demo data ──────────────────────────────────────────────────────────

DEMO_SCHEMA = {
    "toyota":      {"type": "actor", "tokens": ["toyota"], "country_code": "JP"},
    "tesla":       {"type": "actor", "tokens": ["tesla"], "country_code": "US"},
    "ford":        {"type": "actor", "tokens": ["ford"], "country_code": "US"},
    "bmw":         {"type": "actor", "tokens": ["bmw"], "country_code": "DE"},
    "byd":         {"type": "actor", "tokens": ["byd"], "country_code": "CN"},
    "panasonic":   {"type": "actor", "tokens": ["panasonic"], "country_code": "JP"},
    "invest":      {"type": "relation", "tokens": ["invest", "investment", "investing"]},
    "launch":      {"type": "relation", "tokens": ["launch", "unveil", "introduce", "release"]},
    "partner":     {"type": "relation", "tokens": ["partner", "partnership", "collaborate", "alliance"]},
    "expand":      {"type": "relation", "tokens": ["expand", "expansion", "grow", "growth"]},
    "decline":     {"type": "relation", "tokens": ["decline", "drop", "fall", "decrease"]},
    "delay":       {"type": "relation", "tokens": ["delay", "postpone", "push back"]},
    "battery":     {"type": "feature", "tokens": ["battery", "batteries"]},
    "ev":          {"type": "feature", "tokens": ["ev", "electric vehicle", "electric"]},
    "solid_state": {"type": "feature", "tokens": ["solid-state", "solid state"]},
    "autonomous":  {"type": "feature", "tokens": ["autonomous", "self-driving", "autopilot"]},
    "us":          {"type": "market", "tokens": ["us", "united states", "america"]},
    "europe":      {"type": "market", "tokens": ["europe", "european"]},
    "china":       {"type": "market", "tokens": ["china", "chinese"]},
}

DEMO_DOCS = [
    {"id": "d01", "timestamp": "2024-01-15", "text": "Toyota announced a $13.5 billion investment in solid-state battery technology for electric vehicles."},
    {"id": "d02", "timestamp": "2024-02-20", "text": "Tesla expanded its Gigafactory in Shanghai to increase battery production capacity in the Chinese market."},
    {"id": "d03", "timestamp": "2024-03-10", "text": "Ford and BMW announced a partnership to develop next-generation electric vehicle platforms and battery standardization."},
    {"id": "d04", "timestamp": "2024-04-05", "text": "BYD launched its new solid-state battery pack for the European market, undercutting competitors on price."},
    {"id": "d05", "timestamp": "2024-05-20", "text": "Tesla invested $500 million in autonomous driving research at its new facility in the United States."},
    {"id": "d06", "timestamp": "2024-06-15", "text": "Toyota partnered with Panasonic to expand solid-state battery manufacturing in Europe."},
    {"id": "d07", "timestamp": "2024-07-01", "text": "Ford delayed its next-generation EV platform launch by six months due to battery supply constraints."},
    {"id": "d08", "timestamp": "2024-08-10", "text": "BYD expanded aggressively into the European market, growing its EV sales by 40 percent."},
    {"id": "d09", "timestamp": "2024-09-01", "text": "Tesla launched its full self-driving system in China after regulatory approval."},
    {"id": "d10", "timestamp": "2024-10-15", "text": "Toyota launched its first solid-state battery vehicle in the United States to strong initial demand."},
    {"id": "d11", "timestamp": "2024-11-01", "text": "BMW invested $2 billion in autonomous driving technology development across Europe."},
    {"id": "d12", "timestamp": "2024-12-01", "text": "Ford's EV sales declined 15 percent in the US market amid increasing competition from BYD and Tesla."},
    {"id": "d13", "timestamp": "2025-01-15", "text": "Toyota expanded its solid-state battery production to China, partnering with CATL for local manufacturing."},
    {"id": "d14", "timestamp": "2025-02-10", "text": "Tesla expanded its autonomous driving fleet in the United States to over 1 million vehicles."},
    {"id": "d15", "timestamp": "2025-03-01", "text": "BYD launched a next-generation battery technology in Europe that doubles energy density."},
]

DB_PATH = "data/cognition.db"
SCHEMA_PATH = "data/demo_schema.json"


def setup(schema_path=None, db_path=None, ingest_path=None):
    """Initialize cognition, optionally ingesting documents."""
    sp = schema_path or SCHEMA_PATH
    dp = db_path or DB_PATH

    # Write demo schema if it doesn't exist
    sp_file = Path(sp)
    if not sp_file.exists():
        sp_file.parent.mkdir(parents=True, exist_ok=True)
        sp_file.write_text(json.dumps(DEMO_SCHEMA, indent=2))
        print(f"  Wrote demo schema: {sp}")

    config = CognitionConfig(schema_path=sp, db_path=dp)
    cog = Cognition(config)

    # Ingest if DB is empty or custom ingest requested
    if cog.store.count_infons() == 0 or ingest_path:
        if ingest_path:
            with open(ingest_path) as f:
                docs = json.load(f)
            print(f"  Ingesting {len(docs)} documents from {ingest_path}...")
        else:
            docs = DEMO_DOCS
            print(f"  Ingesting {len(docs)} demo documents...")

        t0 = time.time()
        n = cog.ingest(docs, consolidate_now=True)
        dt = time.time() - t0
        print(f"  Extracted {n} infons in {dt:.1f}s")

    return cog


def display_reasoning(result, query, elapsed_ms):
    """Pretty-print a ReasoningResult from `reasoner.reason(...)`.

    Leads with the verdict and θ — the two numbers the user is actually
    there for. Then an ASCII mass bar so you can see the distribution.
    """
    m = result.mass
    print(f"\n{'='*70}")
    print(f"  Q: {query}")
    print(f"{'='*70}")

    # Verdict + mass summary — the headline
    print(f"\n  verdict:  {result.verdict}")
    print(f"  θ:        {m.theta:.2f}   " + ("(high ignorance)" if m.theta > 0.5
                                              else "(committed)"))

    # ASCII mass bar: S / R / U / θ
    def _bar(label, val, color_tag=""):
        w = max(1, int(round(val * 30)))
        bar = "█" * w + "░" * (30 - w)
        print(f"    {label:<8s}  {val:>5.2f}  {bar}")

    print(f"\n  belief mass  (sums to 1):")
    _bar("supports", m.supports)
    _bar("refutes",  m.refutes)
    _bar("uncertain", m.uncertain)
    _bar("θ (ign)",  m.theta)

    # Diagnostic — how many infons actually supported this verdict?
    if hasattr(result, "n_relevant") and result.n_relevant is not None:
        print(f"\n  {result.n_relevant} relevant infon(s) passed the "
              f"role-overlap filter")
    print(f"  reasoning time: {elapsed_ms:.0f}ms")


def display_retrieval(result):
    """Pretty-print a retrieval QueryResult (the older API)."""
    print(f"\n{'='*70}")
    print(f"  Query:   {result.query}")
    print(f"  Persona: {result.persona}")
    print(f"{'='*70}")

    # Anchor spectrum
    top_anchors = sorted(result.anchors_activated.items(), key=lambda x: -x[1])[:8]
    if top_anchors:
        print("\n  Anchor spectrum:")
        for name, score in top_anchors:
            bar = '█' * int(score * 15)
            print(f"    {name:15s} {score:.3f} {bar}")

    # Top infons
    print(f"\n  Infons ({len(result.infons)} total):")
    for inf in result.infons[:10]:
        v = result.valence.get(inf.infon_id, 0)
        arrow = "▲" if v > 0.1 else "▼" if v < -0.1 else "─"
        grounding = []
        for role in ["subject", "predicate", "object"]:
            g = inf.support.get(role, "?")
            grounding.append(g[0].upper() if g else "?")
        g_str = "|".join(grounding)

        print(f"    {arrow} <<{inf.predicate}, {inf.subject}, {inf.object}>>  "
              f"[{g_str}]  conf={inf.confidence:.3f}  valence={v:+.2f}")
        if inf.sentence:
            preview = inf.sentence[:75] + "..." if len(inf.sentence) > 75 else inf.sentence
            print(f"      \"{preview}\"")

    # Constraints
    if result.constraints:
        print(f"\n  Constraints ({len(result.constraints)}):")
        for c in result.constraints[:8]:
            print(f"    ({c.subject}, {c.predicate}, {c.object})  "
                  f"evidence={c.evidence}  docs={c.doc_count}  score={c.score:.3f}")

    # Timeline
    if result.timeline:
        print(f"\n  Timeline ({len(result.timeline)} events):")
        from collections import defaultdict
        months = defaultdict(list)
        for inf in result.timeline:
            months[inf.timestamp[:7]].append(inf)
        for month in sorted(months):
            print(f"    {month}:")
            for inf in months[month][:3]:
                v = result.valence.get(inf.infon_id, 0)
                arrow = "▲" if v > 0.1 else "▼" if v < -0.1 else "─"
                print(f"      {arrow} <<{inf.predicate}, {inf.subject}, {inf.object}>>")

    print()


def main():
    parser = argparse.ArgumentParser(description="Query a cognition knowledge graph")
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument("--persona", "-p", default=None,
                        help="Persona: investor, engineer, executive, regulator, analyst")
    parser.add_argument("--schema", "-s", default=None, help="Path to schema JSON")
    parser.add_argument("--db", "-d", default=None, help="Path to SQLite database")
    parser.add_argument("--ingest", "-i", default=None, help="Path to documents JSON to ingest")
    parser.add_argument("--top-k", "-k", type=int, default=15, help="Max results")
    parser.add_argument("--stats", action="store_true", help="Show knowledge graph stats")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip the reasoner and show persona-valence retrieval only")
    parser.add_argument("--reasoning-only", action="store_true",
                        help="Skip the retrieval view and show only the calibrated verdict")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run analyze_corpus() before the query — flags hubs, "
                             "low-entropy data, missing temporal edges, etc.")

    args = parser.parse_args()

    if not args.query and not (args.stats or args.diagnose):
        parser.print_help()
        sys.exit(1)

    print("Loading cognition...")
    t0 = time.time()
    cog = setup(schema_path=args.schema, db_path=args.db, ingest_path=args.ingest)
    dt = time.time() - t0
    print(f"  Ready in {dt:.1f}s")

    if args.stats:
        s = cog.stats()
        print(f"\n  Infons:      {s['infon_count']}")
        print(f"  Constraints: {s['constraint_count']}")
        print(f"  Sequences:   {'yes' if s['has_sequences'] else 'no'}")
        print(f"  Anchors:     {s['anchors']}")
        print(f"  Backend:     {s['backend']}")
        print(f"  Model:       {s['model']}")

    if args.diagnose:
        from cognition.diagnostics import analyze_corpus
        print()
        print(analyze_corpus(cog).summary())

    if args.query:
        # 1. Reasoning view — calibrated verdict with (S, R, U, θ)
        if not args.retrieval_only:
            t0 = time.time()
            reasoner = cog.reasoner()
            r_result = reasoner.reason(args.query)
            dt = time.time() - t0
            display_reasoning(r_result, args.query, dt * 1000)

        # 2. Retrieval view — persona-valence (optional, default-on)
        if not args.reasoning_only:
            t0 = time.time()
            result = cog.query(
                args.query,
                persona=args.persona,
                top_k=args.top_k,
            )
            dt = time.time() - t0
            display_retrieval(result)
            print(f"  retrieval time: {dt*1000:.0f}ms")

    cog.close()


if __name__ == "__main__":
    main()
