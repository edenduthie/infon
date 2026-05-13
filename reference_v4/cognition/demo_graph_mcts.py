"""Demo: Graph MCTS traversal over an automotive knowledge graph.

Ingests a corpus of documents about Toyota's battery investments and
EV strategy, builds the hypergraph, then uses MCTS to traverse it
and answer a multi-hop question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognition import Cognition, CognitionConfig
from cognition.graph_mcts import GraphMCTS, format_mcts_result


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC CORPUS: Toyota battery investment narrative
# ═══════════════════════════════════════════════════════════════════════

DOCUMENTS = [
    # Investment announcements
    {
        "text": "Toyota announced a $13.6 billion investment in battery technology through 2030, with a focus on solid-state batteries that promise twice the range of conventional lithium-ion cells.",
        "id": "doc_invest_1",
        "timestamp": "2023-06-01",
    },
    {
        "text": "The Japanese automaker plans to invest heavily in solid-state battery development, partnering with Panasonic and CATL to accelerate production timelines.",
        "id": "doc_invest_2",
        "timestamp": "2023-08-15",
    },
    {
        "text": "Toyota's battery investment strategy includes building a new gigafactory in North Carolina, expected to produce 30 GWh annually by 2025.",
        "id": "doc_invest_3",
        "timestamp": "2023-09-20",
    },
    # Technology progress
    {
        "text": "Toyota revealed a breakthrough in solid-state battery technology, achieving a 10-minute charge time and 745-mile range in laboratory tests.",
        "id": "doc_tech_1",
        "timestamp": "2024-01-10",
    },
    {
        "text": "The solid-state batteries developed by Toyota use a sulfide-based electrolyte that eliminates the risk of thermal runaway, making them significantly safer than lithium-ion alternatives.",
        "id": "doc_tech_2",
        "timestamp": "2024-03-05",
    },
    # Market impact - positive
    {
        "text": "Toyota's global market share in the EV segment grew from 3.2% to 5.1% in 2024, driven by strong demand for the bZ4X and new battery-electric models.",
        "id": "doc_market_1",
        "timestamp": "2024-06-15",
    },
    {
        "text": "Investors reacted positively to Toyota's battery roadmap, pushing the stock price up 18% year-over-year as the company overtook Volkswagen in EV market capitalization.",
        "id": "doc_market_2",
        "timestamp": "2024-07-20",
    },
    {
        "text": "Toyota reported record quarterly profits in Q3 2024, with EV sales contributing 22% of revenue compared to just 8% the previous year.",
        "id": "doc_market_3",
        "timestamp": "2024-09-30",
    },
    # Market impact - negative/counter-evidence
    {
        "text": "Despite investments, Toyota recalled 12,000 bZ4X vehicles due to battery cooling issues, temporarily halting production at the Kentucky plant.",
        "id": "doc_recall_1",
        "timestamp": "2024-04-12",
    },
    {
        "text": "Critics argue Toyota's late entry into the EV market has cost it significant ground to Tesla and BYD in China, where market share fell from 14% to 11%.",
        "id": "doc_negative_1",
        "timestamp": "2024-05-08",
    },
    {
        "text": "Toyota's solid-state battery commercialization was delayed from 2025 to 2027, raising concerns about the company's ability to compete in the next-generation battery race.",
        "id": "doc_delay_1",
        "timestamp": "2024-08-01",
    },
    # Partnerships and ecosystem
    {
        "text": "Toyota expanded its partnership with Panasonic to form Prime Planet Energy Solutions, a joint venture focused on prismatic battery cell production for EVs.",
        "id": "doc_partner_1",
        "timestamp": "2023-11-10",
    },
    {
        "text": "The alliance between Toyota and Idemitsu Kosan aims to mass-produce solid-state batteries by 2027, leveraging Idemitsu's expertise in sulfide solid electrolytes.",
        "id": "doc_partner_2",
        "timestamp": "2024-02-28",
    },
    # Competitive context
    {
        "text": "Tesla's battery cost per kWh dropped to $80 in 2024, maintaining its cost advantage over Toyota's $95 per kWh despite the Japanese firm's heavy investment.",
        "id": "doc_compete_1",
        "timestamp": "2024-10-15",
    },
    {
        "text": "Ford and GM announced they would license Toyota's solid-state battery technology, validating Toyota's research investment and positioning it as an industry standard.",
        "id": "doc_compete_2",
        "timestamp": "2024-11-20",
    },
]


def main():
    print("=" * 70)
    print("GRAPH MCTS DEMO: Toyota Battery Investment → Market Share")
    print("=" * 70)

    # 1. Build the knowledge graph
    print("\n[1] Building hypergraph from 15 documents...")
    schema_path = str(Path(__file__).parent.parent / "data" / "automotive_schema.json")
    config = CognitionConfig(
        schema_path=schema_path,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.02,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=5,
    )
    cog = Cognition(config)

    n_infons = cog.ingest(DOCUMENTS, consolidate_now=True)
    stats = cog.stats()
    print(f"    Infons extracted: {n_infons}")
    print(f"    Constraints: {stats['constraint_count']}")
    print(f"    Has sequences: {stats['has_sequences']}")
    print(f"    Anchors: {stats['anchors']}")

    # Show some extracted infons
    all_infons = cog.store.query_infons(limit=200)
    print(f"\n    Sample infons:")
    for inf in all_infons[:8]:
        print(f"      <<{inf.predicate}, {inf.subject}, {inf.object}; "
              f"pol={inf.polarity}>> conf={inf.confidence:.3f} [{inf.doc_id}]")

    # Show NEXT edges
    edges = cog.store.get_edges(edge_type="NEXT", limit=100)
    print(f"\n    NEXT edges: {len(edges)}")
    for edge in edges[:5]:
        print(f"      {edge.source[:12]}... → {edge.target[:12]}... "
              f"({edge.metadata.get('shared_anchor', '?')})")

    # 2. Run MCTS traversal
    print("\n" + "=" * 70)
    print("[2] Running Graph MCTS...")
    print("=" * 70)

    mcts = GraphMCTS(
        store=cog.store,
        encoder=cog.encoder,
        schema=cog.schema,
        max_iterations=8,
        max_depth=4,
        exploration_bias=1.4,
    )

    query = "Did Toyota's battery investment lead to market share gains?"
    result = mcts.search(query, verbose=True)

    # 3. Print formatted result
    print("\n" + "=" * 70)
    print("[3] RESULT")
    print("=" * 70)
    print()
    print(format_mcts_result(result))

    # 4. Compare with flat retrieval
    print("\n" + "=" * 70)
    print("[4] COMPARISON: Flat Retrieval vs Graph MCTS")
    print("=" * 70)

    flat_result = cog.query(query, top_k=20, include_chains=True)
    from cognition.dempster_shafer import verify_claim
    flat_verdict = verify_claim(
        flat_result.infons,
        claim_anchors=flat_result.anchors_activated,
        schema_types=cog.schema.types,
    )

    print(f"\n  Flat retrieval (top-k):")
    print(f"    Verdict: {flat_verdict.label}")
    print(f"    Belief: S={flat_verdict.belief_supports:.3f} "
          f"R={flat_verdict.belief_refutes:.3f}")
    print(f"    Infons used: {flat_verdict.n_evidence}")
    print(f"    Chains: {len([e for e in flat_result.edges if e.edge_type == 'NEXT'])}")

    print(f"\n  Graph MCTS:")
    print(f"    Verdict: {result.verdict}")
    print(f"    Belief: S={result.combined_mass.supports:.3f} "
          f"R={result.combined_mass.refutes:.3f}")
    print(f"    Infons evaluated: {result.infons_evaluated}")
    print(f"    Chains discovered: {len(result.chains_discovered)}")
    print(f"    Nodes explored: {result.nodes_explored}")
    print(f"    Time: {result.elapsed_s:.2f}s")

    cog.close()


if __name__ == "__main__":
    main()
