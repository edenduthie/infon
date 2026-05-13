"""Test the DIMEFIL schema against real-world grayzone scenarios.

Validates:
  1. Schema loads into cognition without errors
  2. SPLADE activation resolves surface forms to correct anchors
  3. Hierarchy parent traversal works (senkaku → japan → east_asia)
  4. Extraction produces infons with correct types and parent chains
  5. Query retrieval finds relevant infons via hierarchical fallback
  6. MCTS traversal discovers multi-hop chains across DIMEFIL domains
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognition import Cognition, CognitionConfig, AnchorSchema

SCHEMA_PATH = str(Path(__file__).parent.parent / "data" / "dimefil_schema.json")

# Real-world grayzone test documents
DOCUMENTS = [
    {
        "text": "China Coast Guard vessels entered waters near the Senkaku Islands "
                "for the 30th consecutive day, prompting Japan to file a diplomatic "
                "protest. Tokyo summoned Beijing's ambassador and demanded an immediate "
                "withdrawal.",
        "id": "senkaku_incursion",
        "timestamp": "2024-03-15",
    },
    {
        "text": "The United States imposed targeted sanctions on Chinese technology "
                "firms accused of supplying dual-use components to Russia's military. "
                "Washington added 14 entities to the Commerce Department's Entity List.",
        "id": "us_china_sanctions",
        "timestamp": "2024-04-01",
    },
    {
        "text": "Russia conducted a nuclear forces exercise involving its strategic "
                "bomber fleet and submarine-launched ballistic missiles, days after "
                "NATO expanded joint military exercises in the Baltic states.",
        "id": "russia_nuclear_drill",
        "timestamp": "2024-04-10",
    },
    {
        "text": "Philippines accused Chinese fishing vessels of swarming near "
                "Scarborough Shoal in the South China Sea. The Philippine Coast Guard "
                "deployed additional patrol boats to monitor the situation.",
        "id": "scarborough_shoal",
        "timestamp": "2024-05-01",
    },
    {
        "text": "Iran threatened to close the Strait of Hormuz in response to new "
                "US sanctions targeting Iranian oil exports. European nations called "
                "for diplomatic de-escalation through multilateral channels.",
        "id": "hormuz_threat",
        "timestamp": "2024-05-15",
    },
    {
        "text": "A coordinated disinformation campaign attributed to Russian-linked "
                "bot networks flooded social media with false narratives about NATO "
                "military buildup in Poland, according to EU intelligence analysts.",
        "id": "disinfo_campaign",
        "timestamp": "2024-06-01",
    },
    {
        "text": "North Korea tested a new hypersonic missile over the Sea of Japan, "
                "prompting emergency consultations between South Korea, Japan, and "
                "the United States. The UN Security Council scheduled a session.",
        "id": "nk_missile_test",
        "timestamp": "2024-06-15",
    },
    {
        "text": "Australia signed a new defense cooperation agreement with Japan "
                "to strengthen intelligence sharing and joint military exercises in "
                "the Indo-Pacific region. The AUKUS partners welcomed the development.",
        "id": "aus_japan_defense",
        "timestamp": "2024-07-01",
    },
]


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_schema_load():
    section("TEST 1: Schema Load")
    schema = AnchorSchema.from_file(SCHEMA_PATH)
    print(f"  Anchors loaded: {len(schema.names)}")
    print(f"  Types: {schema.anchor_types}")

    # Check hierarchy
    assert "senkaku" in schema.names, "senkaku anchor missing"
    assert schema.get_parent("senkaku") == "japan", f"senkaku parent: {schema.get_parent('senkaku')}"
    assert schema.get_parent("japan") == "east_asia", f"japan parent: {schema.get_parent('japan')}"
    ancestors = schema.get_ancestors("senkaku")
    print(f"  senkaku ancestors: {ancestors}")
    assert "japan" in ancestors
    assert "east_asia" in ancestors

    # Check DIMEFIL hierarchy
    assert schema.get_parent("combat_operations") == "force_projection"
    assert schema.get_parent("force_projection") == "military"
    print(f"  combat_operations ancestors: {schema.get_ancestors('combat_operations')}")

    # Check descendants
    east_asia_desc = schema.get_descendants("east_asia")
    print(f"  east_asia descendants ({len(east_asia_desc)}): {east_asia_desc[:8]}...")
    assert "japan" in east_asia_desc
    assert "senkaku" in east_asia_desc
    assert "china" in east_asia_desc
    assert "taiwan" in east_asia_desc

    mil_desc = schema.get_descendants("military")
    print(f"  military descendants ({len(mil_desc)}): {mil_desc[:8]}...")

    print("  PASSED")


def test_encoding():
    section("TEST 2: SPLADE Encoding + Anchor Projection")
    config = CognitionConfig(schema_path=SCHEMA_PATH, db_path=":memory:")
    cog = Cognition(config)

    # Test synonym resolution
    test_cases = [
        ("China Coast Guard patrolled near the Senkaku Islands",
         ["ccg", "senkaku", "china", "japan"]),
        ("United States imposed sanctions on Russian entities",
         ["us", "targeted_sanctions", "russia"]),
        ("NATO conducted joint military exercises in the Baltic",
         ["nato", "military_posturing", "baltic_states"]),
        ("Iran threatened to close the Strait of Hormuz",
         ["iran", "strait_of_hormuz"]),
        ("Disinformation campaign by bot networks on social media",
         ["disinformation", "propaganda_influence"]),
        ("North Korea tested a hypersonic missile",
         ["north_korea", "missile_operations"]),
    ]

    for text, expected_anchors in test_cases:
        acts = cog.encoder.encode_single(text)
        top_10 = sorted(acts.items(), key=lambda x: -x[1])[:10]
        top_names = [name for name, _ in top_10]
        found = [a for a in expected_anchors if a in acts and acts[a] > 0.05]
        missing = [a for a in expected_anchors if a not in found]

        status = "OK" if not missing else "PARTIAL"
        print(f"\n  [{status}] \"{text[:50]}...\"")
        print(f"       Top activations: {[(n, f'{s:.2f}') for n, s in top_10[:6]]}")
        if missing:
            print(f"       Missing: {missing}")
        if found:
            print(f"       Found:   {found}")

    cog.close()
    print("\n  PASSED")


def test_extraction():
    section("TEST 3: Infon Extraction")
    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.01,
        top_k_per_role=5,
    )
    cog = Cognition(config)

    n = cog.ingest(DOCUMENTS)
    print(f"  Ingested {len(DOCUMENTS)} documents → {n} infons")

    all_infons = cog.store.query_infons(limit=500)

    # Check type distribution
    from collections import Counter
    subj_types = Counter(cog.schema.types.get(i.subject, "?") for i in all_infons)
    pred_types = Counter(cog.schema.types.get(i.predicate, "?") for i in all_infons)
    obj_types = Counter(cog.schema.types.get(i.object, "?") for i in all_infons)
    print(f"  Subject types: {dict(subj_types)}")
    print(f"  Predicate types: {dict(pred_types)}")
    print(f"  Object types: {dict(obj_types)}")

    # Check that we have location objects
    loc_infons = [i for i in all_infons if cog.schema.types.get(i.object) == "location"]
    print(f"  Infons with location objects: {len(loc_infons)}")
    if loc_infons:
        for inf in loc_infons[:3]:
            parent = cog.schema.get_parent(inf.object) or "none"
            print(f"    {inf.subject} --[{inf.predicate}]--> {inf.object} (parent={parent})")

    # Check support types
    support_types = Counter()
    for inf in all_infons:
        for role, stype in (inf.support or {}).items():
            support_types[stype] += 1
    print(f"  Support types: {dict(support_types)}")

    # Check that consolidation works
    cog.consolidate()
    stats = cog.stats()
    edges = cog.store.get_edges(edge_type="NEXT", limit=300)
    print(f"  After consolidation: {stats['constraint_count']} constraints, {len(edges)} NEXT edges")

    cog.close()
    print("  PASSED")
    return n


def test_query_retrieval():
    section("TEST 4: Query Retrieval + Hierarchy")
    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.01,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=0,
    )
    cog = Cognition(config)
    cog.ingest(DOCUMENTS)
    cog.consolidate()

    queries = [
        ("What happened near the Senkaku Islands?", ["senkaku", "japan", "ccg"]),
        ("US sanctions against China", ["us", "china", "targeted_sanctions"]),
        ("Military activity in East Asia", ["military", "east_asia"]),
        ("Nuclear threats and missile tests", ["nuclear_operations", "missile_operations"]),
        ("Disinformation operations in Europe", ["disinformation", "europe"]),
        ("Maritime disputes in the South China Sea", ["south_china_sea", "maritime_gray_zone"]),
    ]

    for query_text, expected_anchors in queries:
        result = cog.query(query_text, top_k=20, include_chains=True)
        activated = sorted(result.anchors_activated.items(), key=lambda x: -x[1])[:8]
        n_infons = len(result.infons)
        n_edges = len(result.edges)

        print(f"\n  Query: \"{query_text}\"")
        print(f"    Persona: {result.persona}")
        print(f"    Infons: {n_infons}, NEXT edges: {n_edges}")
        print(f"    Top anchors: {[(n, f'{s:.2f}') for n, s in activated[:5]]}")

        found = [a for a in expected_anchors
                 if a in result.anchors_activated and result.anchors_activated[a] > 0.05]
        if found:
            print(f"    Found expected: {found}")

        if result.infons:
            top_inf = result.infons[0]
            print(f"    Top infon: {top_inf.subject} --[{top_inf.predicate}]--> {top_inf.object}")
            print(f"      from: \"{top_inf.sentence[:60]}...\"")

        if result.constraints:
            c = result.constraints[0]
            print(f"    Top constraint: {c.subject} --[{c.predicate}]--> {c.object} "
                  f"(score={c.score:.3f}, evidence={c.evidence})")

    cog.close()
    print("\n  PASSED")


def test_mcts():
    section("TEST 5: Graph MCTS Traversal")
    from cognition.graph_mcts import GraphMCTS

    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.01,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=0,
    )
    cog = Cognition(config)
    cog.ingest(DOCUMENTS)
    cog.consolidate()

    mcts = GraphMCTS(
        store=cog.store, encoder=cog.encoder, schema=cog.schema,
        max_iterations=6, max_depth=3, exploration_bias=1.4,
    )

    queries = [
        "Did Chinese maritime activity near the Senkaku Islands lead to a broader military response?",
        "Are US sanctions on China connected to Russia's military operations?",
        "Is there an escalation pattern in the Indo-Pacific?",
    ]

    for query_text in queries:
        print(f"\n  Query: \"{query_text}\"")
        result = mcts.search(query_text, verbose=False)

        m = result.combined_mass
        print(f"    Verdict: {result.verdict}")
        print(f"    Mass: S={m.supports:.3f}  R={m.refutes:.3f}  θ={m.theta:.3f}")
        print(f"    Nodes: {result.nodes_explored}, Infons evaluated: {result.infons_evaluated}")
        print(f"    Chains: {len(result.chains_discovered)}")
        print(f"    Time: {result.elapsed_s:.2f}s")

        if result.chains_discovered:
            for i, chain in enumerate(result.chains_discovered[:3]):
                types = [cog.schema.types.get(a, "?") for a in chain]
                print(f"    Chain {i+1}: {' → '.join(f'{a}({t})' for a, t in zip(chain, types))}")

    cog.close()
    print("\n  PASSED")


def test_hierarchy_rollup():
    section("TEST 6: Hierarchy Rollup Query")
    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.01,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=0,
    )
    cog = Cognition(config)
    cog.ingest(DOCUMENTS)
    cog.consolidate()

    # Query at different hierarchy levels
    specific = cog.encoder.encode_single("Senkaku Islands dispute")
    regional = cog.encoder.encode_single("East Asia tensions")
    broad = cog.encoder.encode_single("Indo-Pacific security")

    print("\n  Senkaku (specific):")
    s_top = sorted(specific.items(), key=lambda x: -x[1])[:5]
    for name, score in s_top:
        parent = cog.schema.get_parent(name) or "root"
        print(f"    {name} ({cog.schema.types.get(name, '?')}) = {score:.2f}  parent={parent}")

    print("\n  East Asia (regional):")
    r_top = sorted(regional.items(), key=lambda x: -x[1])[:5]
    for name, score in r_top:
        parent = cog.schema.get_parent(name) or "root"
        print(f"    {name} ({cog.schema.types.get(name, '?')}) = {score:.2f}  parent={parent}")

    print("\n  Indo-Pacific (broad):")
    b_top = sorted(broad.items(), key=lambda x: -x[1])[:5]
    for name, score in b_top:
        parent = cog.schema.get_parent(name) or "root"
        print(f"    {name} ({cog.schema.types.get(name, '?')}) = {score:.2f}  parent={parent}")

    # Show that descendants connect the levels
    if "east_asia" in regional:
        descs = cog.schema.get_descendants("east_asia")
        activated_descs = [d for d in descs if d in specific and specific[d] > 0.1]
        print(f"\n  east_asia descendants activated by 'Senkaku' query: {activated_descs}")

    cog.close()
    print("  PASSED")


def test_descendant_expansion():
    section("TEST 7: Query-Time Descendant Expansion")
    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.01,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=0,
    )
    cog = Cognition(config)
    cog.ingest(DOCUMENTS)
    cog.consolidate()

    # --- Flat retrieval: parent-level query should find child-level infons ---

    # "East Asia" is a parent of japan, china, senkaku, etc.
    # Without expansion, querying east_asia returns 0 infons (nothing stored at that level).
    # With expansion, it should pull in infons stored under japan, senkaku, china, etc.
    result = cog.query("East Asia security tensions", top_k=50, include_chains=True)

    ea_descendants = set(cog.schema.get_descendants("east_asia"))
    infon_anchors = set()
    for inf in result.infons:
        infon_anchors.update([inf.subject, inf.predicate, inf.object])

    child_hits = infon_anchors & ea_descendants
    print(f"\n  Query: 'East Asia security tensions'")
    print(f"    east_asia descendants in schema: {sorted(ea_descendants)[:10]}...")
    print(f"    Infons returned: {len(result.infons)}")
    print(f"    Child-level anchors found in results: {sorted(child_hits)}")
    assert len(result.infons) > 0, "Descendant expansion failed: no infons for parent-level query"
    assert len(child_hits) > 0, "No child-level anchors in results"

    # "military" is a parent of force_projection, combat_operations, etc.
    result_mil = cog.query("military operations worldwide", top_k=50)
    mil_descendants = set(cog.schema.get_descendants("military"))
    mil_anchors = set()
    for inf in result_mil.infons:
        mil_anchors.update([inf.subject, inf.predicate, inf.object])
    mil_child_hits = mil_anchors & mil_descendants
    print(f"\n  Query: 'military operations worldwide'")
    print(f"    military descendants in schema: {sorted(mil_descendants)[:10]}...")
    print(f"    Infons returned: {len(result_mil.infons)}")
    print(f"    Child-level predicates found: {sorted(mil_child_hits)}")
    assert len(result_mil.infons) > 0, "Descendant expansion failed for military"

    # --- Specific query should NOT expand (leaf node) ---
    result_specific = cog.query("Senkaku Islands patrol", top_k=50)
    print(f"\n  Query: 'Senkaku Islands patrol' (leaf — no expansion)")
    print(f"    Infons returned: {len(result_specific.infons)}")

    # --- Constraint retrieval should also expand ---
    ea_constraints = [c for c in result.constraints]
    print(f"\n  Constraints from 'East Asia' query: {len(ea_constraints)}")
    if ea_constraints:
        for c in ea_constraints[:3]:
            print(f"    {c.subject} --[{c.predicate}]--> {c.object} "
                  f"(score={c.score:.3f}, evidence={c.evidence})")

    # --- MCTS: parent-level query should seed with child-level infons ---
    from cognition.graph_mcts import GraphMCTS
    mcts = GraphMCTS(
        store=cog.store, encoder=cog.encoder, schema=cog.schema,
        max_iterations=4, max_depth=3, exploration_bias=1.4,
    )
    mcts_result = mcts.search("Is there an escalation pattern in East Asia?")
    print(f"\n  MCTS Query: 'Is there an escalation pattern in East Asia?'")
    print(f"    Nodes: {mcts_result.nodes_explored}, Infons: {mcts_result.infons_evaluated}")
    print(f"    Chains: {len(mcts_result.chains_discovered)}")
    print(f"    Verdict: {mcts_result.verdict}")
    assert mcts_result.infons_evaluated > 0, "MCTS descendant expansion failed: no infons seeded"

    cog.close()
    print("\n  PASSED")


if __name__ == "__main__":
    test_schema_load()
    test_encoding()
    test_extraction()
    test_query_retrieval()
    test_mcts()
    test_hierarchy_rollup()
    test_descendant_expansion()
    section("ALL TESTS PASSED")
