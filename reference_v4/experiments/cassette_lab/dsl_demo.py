"""Fire every primitive in the DSL, plus three persona queries end-to-end.

Personas:
  analyst     — "What is Toyota investing in, this quarter?"
  compliance  — "Is any claim about CATL contradicted elsewhere?"
  auditor     — "What was our view of Toyota as of the first snapshot?"
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes
from cognition.cassette import (
    Query, run_any, run_all,
    first_seen, last_seen, timeline, count_by,
)
from cognition.cassette.reader import LocalFetcher, hydrate_locs


def mk(i, s, p, o, ts, pol=1, conf=0.8, sent=None):
    return Infon(
        infon_id=f"i{i:04d}", subject=s, predicate=p, object=o,
        polarity=pol, confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{i}", sent_id=f"d{i}_{i:04d}", timestamp=ts,
    )


def write(root, cid, infons):
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{cid}.inf")
    with open(path, "wb") as f:
        w = CassetteWriter(f, cassette_id=cid, schema_ref="dsl-v1")
        for inf in infons:
            w.add(inf)
        footer = w.close()
    idx = build_indexes(footer, os.path.join(root, "index"))
    return path, footer, idx


def build_store(root):
    # Three cassettes → three snapshots. Later cassettes refute earlier
    # claims to test contradiction queries.
    a = [
        mk(1, "toyota",  "invest",  "solid_state",  "2026-01-04"),
        mk(2, "toyota",  "partner", "panasonic",    "2026-01-05"),
        mk(3, "honda",   "invest",  "lithium_ion",  "2026-01-06"),
        mk(4, "catl",    "supply",  "honda",        "2026-01-07"),
    ]
    b = [
        mk(10, "toyota", "invest",  "batteries",    "2026-02-03"),
        mk(11, "vw",     "invest",  "solid_state",  "2026-02-04"),
        mk(12, "catl",   "supply",  "vw",           "2026-02-05"),
    ]
    # Cassette C includes a refutation of the Honda/lithium_ion claim and
    # a fresh CATL contradiction — for the compliance persona.
    c = [
        mk(20, "honda",  "invest",  "lithium_ion",  "2026-03-02", pol=0,
           sent="Honda has shelved its lithium-ion program entirely."),
        mk(21, "catl",   "supply",  "honda",        "2026-03-03", pol=0,
           sent="CATL no longer supplies batteries to Honda."),
        mk(22, "byd",    "invest",  "solid_state",  "2026-03-04"),
    ]

    pa, fa, ia = write(root, "cass_a", a)
    m0 = Manifest.new(root)
    m0.add_cassette(fa, pa, ia)
    m0.save()
    snap0 = m0.snapshot_id

    pb, fb, ib = write(root, "cass_b", b)
    m1 = Manifest.new(root, parent=m0)
    m1.add_cassette(fb, pb, ib)
    m1.save()

    pc, fc, ic = write(root, "cass_c", c)
    m2 = Manifest.new(root, parent=m1)
    m2.add_cassette(fc, pc, ic)
    m2.save()

    return m2, snap0


def show(label: str, hits):
    print(f"\n{label}  → {len(hits)} hits")
    for h in hits[:8]:
        polarity_mark = "¬" if h.loc.polarity == 0 else " "
        print(f"  {polarity_mark}{h.loc.subject:>8} {h.loc.predicate:>8} "
              f"{h.loc.object:<14}  t={h.loc.timestamp}  (from {h.cassette_id})")


def main():
    root = tempfile.mkdtemp(prefix="dsl_demo_")
    m, snap0 = build_store(root)
    fetcher = LocalFetcher()

    n = sum(c["n_records"] for c in m.cassettes)
    print(f"store: {len(m.cassettes)} cassettes, {n} infons\n")

    # ═══════════════════════════════════════════════════════════════════
    # GRAMMAR
    # ═══════════════════════════════════════════════════════════════════
    print("━" * 72)
    print("GRAMMAR")
    print("━" * 72)

    show("where(subject=toyota)",
         Query().where(subject="toyota").run(m))
    show("where(subject=toyota, predicate=invest)",
         Query().where(subject="toyota", predicate="invest").run(m))
    show("mentioning('catl')  ← role-free",
         Query().mentioning("catl").run(m))
    show("where(subject=toyota).affirmed()",
         Query().where(subject="toyota").affirmed().run(m))
    show("where(subject=honda).negated()",
         Query().where(subject="honda").negated().run(m))
    show("where(subject=toyota).min_conf(0.85)",
         Query().where(subject="toyota").min_conf(0.85).run(m))

    # ═══════════════════════════════════════════════════════════════════
    # TIMELINE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "━" * 72)
    print("TIMELINE")
    print("━" * 72)

    show("where(subject=toyota).between('2026-02-01','2026-02-28')",
         Query().where(subject="toyota").between("2026-02-01", "2026-02-28").run(m))
    show("mentioning('solid_state').after('2026-02-01')",
         Query().mentioning("solid_state").after("2026-02-01").run(m))

    print(f"\nfirst_seen('toyota')   = {first_seen(m, 'toyota')}")
    print(f"last_seen('toyota')    = {last_seen(m, 'toyota')}")
    print(f"first_seen('byd')      = {first_seen(m, 'byd')}")

    print("\ntimeline('catl'):")
    for ts, h in timeline(m, "catl"):
        mark = "¬" if h.loc.polarity == 0 else " "
        print(f"  {ts}  {mark}{h.loc.subject}/{h.loc.predicate}/{h.loc.object}")

    # ═══════════════════════════════════════════════════════════════════
    # LOGIC
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "━" * 72)
    print("LOGIC")
    print("━" * 72)

    # AND (chain)
    q_and = Query().where(subject="toyota", predicate="invest").after("2026-02-01")
    show(f"AND (chain): {q_and.describe()}", q_and.run(m))

    # OR (run_any) — anything about either toyota or vw
    q_a = Query().where(subject="toyota")
    q_b = Query().where(subject="vw")
    show("OR: subject=toyota | subject=vw", run_any(m, [q_a, q_b]))

    # NOT (contradicting) — flip polarity on a pinned triple.
    # Useful for refutation search: 'find negations of the claim X invested Y'.
    claim = Query().where(subject="honda", predicate="invest", object="lithium_ion")
    show("affirmed claim (honda invest lithium_ion)", claim.affirmed().run(m))
    show("contradicting() of that claim", claim.affirmed().contradicting().run(m))

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE (pushdown — no hydration)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "━" * 72)
    print("AGGREGATE (no hydration)")
    print("━" * 72)
    counts = count_by(m, Query().where(subject="toyota"), groupby="predicate")
    print(f"count by predicate for subject=toyota: {counts}")
    counts = count_by(m, Query().mentioning("solid_state"), groupby="subject")
    print(f"count by subject for mentions(solid_state): {counts}")

    # ═══════════════════════════════════════════════════════════════════
    # PERSONA QUERIES (end-to-end with hydration + SPLADE-free ranking)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "━" * 72)
    print("PERSONAS")
    print("━" * 72)

    # Analyst — "Toyota investments this quarter, ranked by confidence"
    q = (Query().where(subject="toyota", predicate="invest")
                 .between("2026-01-01", "2026-03-31")
                 .affirmed())
    hits = q.run(m)
    infons = hydrate_locs(fetcher, m, hits)
    infons.sort(key=lambda i: -i.confidence)
    print(f"\n[ANALYST] {q.describe()}")
    for inf in infons:
        print(f"  {inf.confidence:.2f}  {inf.sentence}")

    # Compliance — "any CATL claim with a later contradiction in the store?"
    print(f"\n[COMPLIANCE] CATL claims whose triple is refuted later")
    fetcher.reset_counters()
    for aff in Query().where(subject="catl").affirmed().run(m):
        # Construct the exact contradiction query.
        contra = Query().where(subject=aff.loc.subject,
                               predicate=aff.loc.predicate,
                               object=aff.loc.object).negated()
        later = [h for h in contra.run(m)
                 if (h.loc.timestamp or "") > (aff.loc.timestamp or "")]
        if later:
            print(f"  affirmed {aff.loc.subject}/{aff.loc.predicate}/{aff.loc.object} "
                  f"@ {aff.loc.timestamp}  →  refuted @ "
                  f"{[h.loc.timestamp for h in later]}")
    print(f"  (no hydration used: {fetcher.requests} range gets)")

    # Auditor — "what did we know about toyota at snapshot 0?"
    print(f"\n[AUDITOR] snapshot {snap0} — view of 'toyota' as of then")
    m_old = Manifest.load_at(root, snap0)
    for h in Query().where(subject="toyota").run(m_old):
        print(f"  {h.loc.timestamp}  {h.loc.subject}/{h.loc.predicate}/"
              f"{h.loc.object}  (from {h.cassette_id})")
    print(f"  snapshots available: {Manifest.list_snapshots(root)}")

    # ═══════════════════════════════════════════════════════════════════
    # SANITY: pruner still fires
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "━" * 72)
    print("PRUNER SANITY")
    print("━" * 72)
    from cognition.cassette.index import _prune_by_anchors
    paths_toyota = _prune_by_anchors(m, "by_triple", {"toyota"}, ("subjects",))
    paths_byd    = _prune_by_anchors(m, "by_triple", {"byd"},    ("subjects",))
    paths_miss   = _prune_by_anchors(m, "by_triple", {"nobody"}, ("subjects",))
    print(f"  toyota subject pruner: {len(paths_toyota)}/{len(m.cassettes)} shards")
    print(f"  byd    subject pruner: {len(paths_byd)}/{len(m.cassettes)} shards")
    print(f"  missing anchor:        {len(paths_miss)}/{len(m.cassettes)} shards "
          f"(expect 0)")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()
