"""BlackMagic quickstart — end-to-end smoke test of ingest/search/verify/reason/imagine."""
from blackmagic import BlackMagic, BlackMagicConfig

DOCS = [
    {"text": "Toyota announced a 13.6 billion dollar investment in battery "
             "production by 2030, targeting 200 GWh of capacity across Japan "
             "and North America.",
     "id": "d1", "timestamp": "2026-03-01"},
    {"text": "Ford recalled over one million vehicles in Q1 due to faulty "
             "transmission systems affecting Ford F-150 and Explorer models.",
     "id": "d2", "timestamp": "2026-03-15"},
    {"text": "Honda launched a new electric SUV in partnership with CATL, "
             "targeting the Chinese and European markets in 2026.",
     "id": "d3", "timestamp": "2026-04-02"},
    {"text": "Stellantis postponed its planned electric vehicle launch in "
             "Japan from 2025 to 2027, citing supply chain disruptions.",
     "id": "d4", "timestamp": "2026-04-10"},
    {"text": "BMW expanded its investment in solid-state battery research, "
             "committing 500 million euros to European R&D centres.",
     "id": "d5", "timestamp": "2026-04-18"},
]


def main():
    cfg = BlackMagicConfig(
        schema_path="examples/automotive_schema.json",
        db_path=":memory:",
        activation_threshold=0.3,
        min_confidence=0.05,
        consolidation_interval=0,  # we'll consolidate explicitly
    )
    bm = BlackMagic(cfg)
    print(f"Loaded schema: {len(bm.schema.names)} anchors")
    print(f"Device: {bm.encoder.device}")

    n = bm.ingest(DOCS, consolidate_now=True)
    print(f"\nIngested {len(DOCS)} docs → {n} infons")
    print(f"Stats: {bm.stats()}")

    # ── Search ──
    print("\n── Search ──")
    result = bm.search("Which automakers are investing in batteries?",
                       persona="investor", top_k=5)
    print(f"Query: '{result.query}'  Persona: {result.persona}")
    print(f"Anchors: {list(result.anchors_activated.keys())[:8]}")
    for inf in result.infons[:3]:
        print(f"  {inf.subject}:{inf.predicate}:{inf.object}  "
              f"(conf={inf.confidence:.2f}, imp={inf.importance:.2f})")

    # ── Verify ──
    print("\n── Verify ──")
    v = bm.verify_claim("Toyota is aggressively investing in battery technology.")
    print(f"Claim verdict: {v.label}  "
          f"S={v.belief_supports:.2f} R={v.belief_refutes:.2f} U={v.belief_uncertain:.2f}")

    # ── Reason (MCTS) ──
    print("\n── Reason (MCTS) ──")
    m = bm.reason("Does the automotive industry face battery supply risks?",
                  max_iterations=4, max_depth=3)
    print(f"MCTS verdict: {m.verdict}  "
          f"nodes={m.nodes_explored}  chains={len(m.chains_discovered)}")

    # ── Imagine (GA) ──
    print("\n── Imagine (GA) ──")
    im = bm.imagine("What partnerships might emerge between OEMs and battery suppliers?",
                    n_generations=5, population_size=30, persona="investor")
    print(f"Imagination verdict: {im.verdict}  (MCTS-compatible: {im.mcts_verdict})")
    print(f"  elapsed={im.elapsed_s:.2f}s  generations={im.generations}  "
          f"nodes_explored={im.nodes_explored}")
    print(f"  combined_mass: S={im.combined_mass.supports:.2f} "
          f"R={im.combined_mass.refutes:.2f} θ={im.combined_mass.theta:.2f}")
    print("\nTop imagined infons (fitness-ranked):")
    for inf in im.imagined_infons[:8]:
        parents = inf.parent_infon_ids[:2] if inf.parent_infon_ids else []
        print(f"  fit={inf.fitness:.3f}  <<{inf.predicate}, {inf.subject}, {inf.object}; "
              f"{inf.polarity}>>  parents={parents}")

    bm.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
