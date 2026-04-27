"""End-to-end integration: ingest → search → verify → reason → imagine."""


def test_full_pipeline(bm_populated):
    # stats
    stats = bm_populated.stats()
    assert stats["infon_count"] > 0
    assert stats["document_count"] > 0
    assert stats["anchors"] > 10

    # search
    r = bm_populated.search("Toyota battery investment")
    assert len(r.infons) > 0

    # verify
    v = bm_populated.verify_claim("Toyota invests in batteries.")
    assert v.label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

    # imagine
    im = bm_populated.imagine("new OEM partnerships",
                              n_generations=2, population_size=10,
                              store_imagined=False)
    assert im.verdict in ("PLAUSIBLE", "CONTRADICTED", "SPECULATIVE")

    # clean up imagined
    bm_populated.clear_imagined()
    assert bm_populated.store.count_infons(kind="imagined") == 0


def test_reingest_is_idempotent(bm, tiny_corpus):
    bm.ingest(tiny_corpus, consolidate_now=True)
    n1 = bm.store.count_infons()
    bm.ingest(tiny_corpus, consolidate_now=True)
    n2 = bm.store.count_infons()
    # Deterministic infon_id + INSERT OR REPLACE → same count
    assert n1 == n2
