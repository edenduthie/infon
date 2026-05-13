"""Integration test: full pipeline from documents to query results.

Uses SPLADE (prithivida/Splade_PP_en_v2) with AnchorProjector — no
trained model weights needed. Just a schema.json defining your anchors.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "cognition" / "src"))

from cognition import Cognition, CognitionConfig

# ── Test schema: automotive domain ──────────────────────────────────────
TEST_SCHEMA = {
    # Actors
    "toyota":     {"type": "actor", "tokens": ["toyota"], "country_code": "JP"},
    "bmw":        {"type": "actor", "tokens": ["bmw"], "country_code": "DE"},
    "tesla":      {"type": "actor", "tokens": ["tesla"], "country_code": "US"},
    "panasonic":  {"type": "actor", "tokens": ["panasonic"], "country_code": "JP"},
    "honda":      {"type": "actor", "tokens": ["honda"], "country_code": "JP"},
    "volkswagen": {"type": "actor", "tokens": ["volkswagen", "vw"], "country_code": "DE"},
    # Relations
    "invest":     {"type": "relation", "tokens": ["invest", "investment", "investing"]},
    "launch":     {"type": "relation", "tokens": ["launch", "launching", "launched", "introduce"]},
    "partner":    {"type": "relation", "tokens": ["partner", "partnership", "partnered", "collaborate"]},
    "compete":    {"type": "relation", "tokens": ["compete", "competing", "competition", "rival"]},
    "expand":     {"type": "relation", "tokens": ["expand", "expanding", "expansion", "grow"]},
    "develop":    {"type": "relation", "tokens": ["develop", "developing", "development"]},
    "sell":       {"type": "relation", "tokens": ["sell", "selling", "sold", "deliver", "delivery"]},
    # Features
    "solid_state":     {"type": "feature", "tokens": ["solid-state", "solid state"], "parent": "battery"},
    "battery":         {"type": "feature", "tokens": ["battery", "batteries"]},
    "ev_platform":     {"type": "feature", "tokens": ["electric vehicle", "ev", "electric"]},
    "autonomous":      {"type": "feature", "tokens": ["autonomous", "self-driving", "autopilot"]},
    "manufacturing":   {"type": "feature", "tokens": ["manufacturing", "factory", "gigafactory", "plant"]},
    "electrification": {"type": "feature", "tokens": ["electrification", "electrify"]},
    # Markets
    "china":         {"type": "market", "tokens": ["china", "chinese"], "country_code": "CN"},
    "europe":        {"type": "market", "tokens": ["europe", "european"], "macro_region": "Europe"},
    "north_america": {"type": "market", "tokens": ["us", "united states", "america", "texas"], "country_code": "US"},
    "global_market": {"type": "market", "tokens": ["global", "worldwide"]},
}

# Test documents
DOCS = [
    {
        "text": "Toyota is investing heavily in solid-state battery technology. "
                "The Japanese automaker plans to launch solid-state batteries by 2027. "
                "Toyota has partnered with Panasonic on battery development.",
        "id": "doc_001",
        "timestamp": "2025-06-15",
    },
    {
        "text": "BMW has announced a new electric vehicle platform for 2026. "
                "The German automaker is competing with Tesla in the luxury EV segment. "
                "BMW plans to invest over two billion dollars in EV manufacturing.",
        "id": "doc_002",
        "timestamp": "2025-07-20",
    },
    {
        "text": "Tesla has expanded its Gigafactory in Texas. "
                "The company reported strong Q2 2025 delivery numbers. "
                "Tesla is developing new autonomous driving capabilities.",
        "id": "doc_003",
        "timestamp": "2025-08-01",
    },
    {
        "text": "The global EV market is expected to grow 25% in 2026. "
                "China leads in EV adoption with over 60% market share. "
                "European automakers are accelerating their electrification strategies.",
        "id": "doc_004",
        "timestamp": "2025-09-10",
    },
]


def test_full_pipeline():
    """Test the complete ingest → consolidate → query pipeline."""

    # Write test schema to temp file
    schema_file = tempfile.mktemp(suffix=".json")
    with open(schema_file, "w") as f:
        json.dump(TEST_SCHEMA, f, indent=2)

    db_path = tempfile.mktemp(suffix=".db")

    config = CognitionConfig(
        backend="local",
        schema_path=schema_file,
        db_path=db_path,
        activation_threshold=0.5,
        top_k_per_role=3,
        min_confidence=0.05,
    )

    print("=" * 78)
    print("COGNITION INTEGRATION TEST (SPLADE + AnchorProjector)")
    print("=" * 78)

    # 1. Initialize
    print("\n1. Initializing Cognition...")
    cog = Cognition(config)
    print(f"   Encoder: {cog.config.model_name}")
    print(f"   Anchors: {len(cog.schema.names)}")
    print(f"   Anchor types: {dict(sorted(((t, len(ns)) for t, ns in cog.schema.by_type.items())))}")
    print(f"   Backend: {cog.config.backend}")
    print(f"   Device: {cog.encoder.device}")

    # 2. Quick SPLADE sanity check
    print("\n2. SPLADE sanity check...")
    test_acts = cog.encoder.encode_single("Toyota invests in solid-state batteries")
    top5 = sorted(test_acts.items(), key=lambda x: -x[1])[:10]
    print(f"   Activations for 'Toyota invests in solid-state batteries':")
    for name, score in top5:
        atype = cog.schema.types.get(name, "?")
        bar = "█" * int(score * 5) + "░" * (25 - int(score * 5))
        print(f"   {name:<20s} [{atype:<8s}] {bar} {score:.3f}")

    # 3. Ingest documents
    print(f"\n3. Ingesting {len(DOCS)} documents...")
    count = cog.ingest(DOCS, consolidate_now=True)
    print(f"   Extracted {count} infons")

    # 4. Stats
    stats = cog.stats()
    print(f"\n4. Knowledge graph stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # 5. Queries with different personas
    queries = [
        ("What is Toyota investing in?", None),
        ("How is the EV market evolving?", "investor"),
        ("What are the latest battery technologies?", "engineer"),
    ]

    print("\n5. Running queries...")
    for query_text, persona in queries:
        print(f"\n   {'─' * 72}")
        print(f"   Query: \"{query_text}\"" + (f" (persona={persona})" if persona else ""))
        result = cog.query(query_text, persona=persona, top_k=10)
        print(f"   Persona: {result.persona}")
        print(f"   Anchors activated: {len(result.anchors_activated)}")
        print(f"   Infons returned: {len(result.infons)}")
        print(f"   Constraints: {len(result.constraints)}")

        # Top anchors
        top_anchors = sorted(result.anchors_activated.items(), key=lambda x: -x[1])[:5]
        if top_anchors:
            print(f"   Top anchors: {', '.join(f'{n}={p:.2f}' for n, p in top_anchors)}")

        # Top infons
        for inf in result.infons[:5]:
            pol = "+" if inf.polarity else "-"
            v = result.valence.get(inf.infon_id, 0)
            sup = inf.support
            sup_str = ""
            if sup:
                tags = []
                for r in ("subject", "predicate", "object"):
                    s = sup.get(r, "?")
                    tags.append(f"{r[0].upper()}:{'D' if s == 'direct' else 'S' if s == 'semantic' else 'H'}")
                sup_str = f" [{','.join(tags)}]"
            print(f"   {pol}<<{inf.predicate}, {inf.subject}, {inf.object}>> "
                  f"conf={inf.confidence:.3f} valence={v:+.2f}{sup_str}")

        # Top constraints
        for c in result.constraints[:3]:
            print(f"   Constraint: ({c.subject}, {c.predicate}, {c.object}) "
                  f"score={c.score:.3f} evidence={c.evidence}")

    # 6. Timeline
    print(f"\n6. Timeline for 'toyota'...")
    infons = cog.store.get_infons_for_anchor("toyota", limit=20)
    timestamped = sorted([inf for inf in infons if inf.timestamp],
                         key=lambda inf: inf.timestamp)
    print(f"   {len(timestamped)} timestamped infons:")
    for inf in timestamped[:8]:
        print(f"   {inf.timestamp}  <<{inf.predicate}, {inf.subject}, {inf.object}>>")

    # 7. Decay
    print("\n7. Importance decay test...")
    cog.decay(reference_date="2026-06-01")
    stats_after = cog.stats()
    print(f"   Infons after decay: {stats_after['infon_count']}")

    # Cleanup
    cog.close()
    import os
    os.unlink(db_path)
    os.unlink(schema_file)

    print("\n" + "=" * 78)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    test_full_pipeline()
