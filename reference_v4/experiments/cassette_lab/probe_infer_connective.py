"""Auto-infer connective predicates from corpus structure.

Corpus = same planted scale store: entities + connective relations +
decoy relations (predicates whose objects are terminal attributes).

Expected inference output:
  connective:     partner, supply, license, acquire, invest
  non-connective: advertise, sponsor, review, mention, host, visit,
                  lobby, criticize  (decoy predicates whose objects
                  are countries/events, not entities)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette import (
    infer_connective_predicates,
    reason_connectivity,
    reason_any_target,
)

# Reuse the scale store from the earlier probe.
from probe_mcts_scale import build_scale_store, PLANTED_CHAINS, UNCONNECTED


def main():
    root = tempfile.mkdtemp(prefix="infer_")
    m = build_scale_store(root, seed=7, decoys_per_entity=12)
    n = sum(c["n_records"] for c in m.cassettes)
    print(f"store: {len(m.cassettes)} cassettes, {n} infons\n")

    inferred = infer_connective_predicates(m)
    expected_connective = {"partner", "supply", "license", "invest"}
    expected_non = {"advertise", "sponsor", "review", "mention",
                    "host", "visit", "lobby", "criticize"}

    print("── INFERENCE ────────────────────────────────────────────")
    print(f"inferred connective: {sorted(inferred)}")
    print(f"expected connective: {sorted(expected_connective)}")

    false_pos = inferred - expected_connective
    false_neg = expected_connective - inferred
    print(f"false positives: {sorted(false_pos)}  "
          f"(non-connective mislabeled connective)")
    print(f"false negatives: {sorted(false_neg)}  "
          f"(connective missed)")

    # Are all expected non-connective predicates correctly excluded?
    wrongly_included = expected_non & inferred
    print(f"wrongly included non-connective: {sorted(wrongly_included)}")

    # ── E2E: do probes still pass when inference is the default? ─────
    print("\n── USAGE: reason_connectivity with auto-inferred set ────────")
    # Toyota ↔ CATL via panasonic — must still resolve.
    gold_cases = [(s, t, "SUPPORTS", f"{s}↔{t} via {mid}")
                  for s, mid, t, _, _ in PLANTED_CHAINS]
    nei_cases = [(s, t, "NOT_ENOUGH_INFO", f"{s}↔{t} (disconnected)")
                 for s, t in UNCONNECTED]

    correct = 0
    for src, tgt, gold, desc in gold_cases + nei_cases:
        # No connective_predicates arg — triggers inference.
        v = reason_connectivity(m, src, tgt, budget=20)
        mark = "✓" if v.label == gold else "✗"
        if v.label == gold:
            correct += 1
        print(f"  {mark} {desc:<40}  {v.label:<16}  "
              f"S={v.mass.supports:.2f} R={v.mass.refutes:.2f} "
              f"gets={v.range_gets}")

    total = len(gold_cases) + len(nei_cases)
    print(f"\n  accuracy (auto-inferred connective): {correct}/{total} "
          f"= {correct/total:.0%}")

    # Multi-target with inference
    print("\n── reason_any_target with auto-inferred set ────────")
    targets = {s[2] for s in PLANTED_CHAINS} | \
              {t for _, t in UNCONNECTED}
    vs = reason_any_target(m, "toyota", targets, budget=20)
    # The gold: only catl should resolve (via panasonic chain from toyota).
    for t in sorted(targets):
        v = vs[t]
        expected = "SUPPORTS" if t == "catl" else "NOT_ENOUGH_INFO"
        mark = "✓" if v.label == expected else "✗"
        print(f"  {mark} {t:<14} {v.label:<16} S={v.mass.supports:.2f}")
    shared_gets = next(iter(vs.values())).range_gets
    print(f"  shared gets: {shared_gets}")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()
