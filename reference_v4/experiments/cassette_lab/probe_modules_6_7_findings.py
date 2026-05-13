"""End-to-end probe: Module 6 (trajectory + constraint), Module 7
(hierarchy expansion), and findings persistence across sessions.

Schema has a small hierarchy tree so Module 7 actually has something to
expand:

    chip_maker (parent)
    ├── tsmc
    ├── intel
    ├── samsung
    └── sk_hynix

    us_maker (parent)
    ├── nvidia
    ├── amd
    └── intel   (deliberate multi-parent — intel is both chip_maker
                  and us_maker, mirroring real ontologies)

Flow:
  1. Ingest docs.
  2. Module 6: trajectory for nvidia; constraint for a disputed triple.
  3. Module 7: Query.where(subject="chip_maker").expand_hierarchy()
     should match any cassette mentioning tsmc/intel/samsung/sk_hynix.
  4. Agent session A records findings.
  5. Agent session B lists findings from a fresh Analyst (same store).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette import (
    InfonStore, Query, Analyst,
)


# Hierarchy: parents are schema-only anchors (no direct surface form).
# The tokens list is a single "_" placeholder that won't match anything,
# so parent anchors stay silent while descendants carry the signal.
SCHEMA = {
    "chip_maker":  {"type": "actor", "tokens": ["_"]},
    "us_maker":    {"type": "actor", "tokens": ["_"]},
    "tsmc":        {"type": "actor", "tokens": ["tsmc"], "parent": "chip_maker"},
    "samsung":     {"type": "actor", "tokens": ["samsung"], "parent": "chip_maker"},
    "sk_hynix":    {"type": "actor", "tokens": ["hynix"], "parent": "chip_maker"},
    "intel":       {"type": "actor", "tokens": ["intel"], "parent": "chip_maker"},
    "nvidia":      {"type": "actor", "tokens": ["nvidia"], "parent": "us_maker"},
    "amd":         {"type": "actor", "tokens": ["amd"], "parent": "us_maker"},

    "openai":      {"type": "actor", "tokens": ["openai"]},
    "anthropic":   {"type": "actor", "tokens": ["anthropic"]},
    "microsoft":   {"type": "actor", "tokens": ["microsoft", "azure"]},
    "google":      {"type": "actor", "tokens": ["google"]},

    "partner":     {"type": "relation",
                     "tokens": ["partner", "partnered", "partnership", "venture"]},
    "supply":      {"type": "relation",
                     "tokens": ["supply", "supplies", "supplier", "sources"]},
    "invest":      {"type": "relation",
                     "tokens": ["invest", "invested", "investment"]},
    "acquire":     {"type": "relation", "tokens": ["acquire", "acquired"]},
    "compete":     {"type": "relation",
                     "tokens": ["compete", "competing"]},

    "hbm":         {"type": "feature", "tokens": ["hbm", "high-bandwidth memory"]},
    "b200":        {"type": "feature", "tokens": ["b200"]},
    "tpu":         {"type": "feature", "tokens": ["tpu"]},
    "3nm":         {"type": "feature", "tokens": ["3nm"]},
    "compute":     {"type": "feature",
                     "tokens": ["compute", "datacenter", "data center"]},
}

DOCS = [
    {"id": "d1",  "text": "Nvidia partnered with TSMC to produce the B200 chip on 3nm.", "timestamp": "2026-01-05"},
    {"id": "d2",  "text": "SK Hynix supplies HBM memory to Nvidia.", "timestamp": "2026-01-08"},
    {"id": "d3",  "text": "Samsung supplies HBM to Nvidia, competing with SK Hynix.", "timestamp": "2026-01-12"},
    {"id": "d4",  "text": "Microsoft invested in datacenter for OpenAI partnership.", "timestamp": "2026-01-15"},
    {"id": "d5",  "text": "Anthropic partnered with Google for TPU compute.", "timestamp": "2026-02-05"},
    {"id": "d6",  "text": "Intel announced a 3nm foundry push, competing with TSMC.", "timestamp": "2026-02-10"},
    {"id": "d7",  "text": "Nvidia acquired a datacenter networking startup.", "timestamp": "2026-02-20"},
    {"id": "d8",  "text": "Samsung's HBM supply to Nvidia fell through after quality issues.", "timestamp": "2026-03-01"},
    {"id": "d9",  "text": "TSMC's 3nm bottleneck is delaying Nvidia's B200 ramp.", "timestamp": "2026-03-05"},
    {"id": "d10", "text": "AMD invested in HBM with SK Hynix.", "timestamp": "2026-03-15"},
]


def rule(label: str):
    print("\n" + "═" * 72)
    print(f"  {label}")
    print("═" * 72)


def main():
    tmp = tempfile.mkdtemp(prefix="modules_67_")
    try:
        schema_path = os.path.join(tmp, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)
        root = os.path.join(tmp, "store")
        store = InfonStore(root, schema_path=schema_path)

        rule("Setup: ingest")
        r = store.ingest(DOCS)
        covered = r["report"].n_docs - len(r["report"].docs_with_zero_infons)
        print(f"  {r['n_infons']} infons, {covered}/{r['report'].n_docs} docs")

        # ── MODULE 6: trajectory + constraint ─────────────────────────
        rule("Module 6: nvidia trajectory (time-ordered, NEXT at read time)")
        traj = store.trajectory("nvidia", hydrate=True)
        for inf in traj:
            mark = "¬" if inf.polarity == 0 else " "
            print(f"  {inf.timestamp}  {mark}{inf.subject:>10}/"
                  f"{inf.predicate:<8}/{inf.object:<10}  "
                  f"{inf.sentence[:48]}")

        edges = store.next_edges("nvidia")
        print(f"\n  NEXT edges ({len(edges)}):")
        for e in edges:
            gap = f"{e.gap_days}d" if e.gap_days is not None else "?"
            print(f"    {e.from_timestamp} → {e.to_timestamp}  "
                  f"gap={gap:<6} {e.from_infon_id[:12]} → {e.to_infon_id[:12]}")

        rule("Module 6: constraint (samsung/supply/hbm — contested triple)")
        c = store.constraint("samsung", "supply", "hbm")
        print(f"  evidence_count: {c.evidence_count}")
        print(f"  affirmed:       {c.n_affirmed}")
        print(f"  refuted:        {c.n_refuted}")
        print(f"  mean_conf:      {c.mean_confidence}")
        print(f"  t_min → t_max:  {c.t_min} → {c.t_max}  "
              f"(span {c.span_days}d)")
        print(f"  balance:        {c.polarity_balance:+.2f}  "
              f"(contested={c.is_contested})")

        # ── MODULE 7: hierarchy expansion ──────────────────────────────
        rule("Module 7: query at parent level (chip_maker)")
        # Baseline: singleton query matches literally "chip_maker" → nothing.
        q_base = Query().where(subject="chip_maker")
        n_base = len(q_base.run(store.manifest))
        print(f"  without expansion:  {n_base} hits (expected 0 — "
              f"'chip_maker' is a parent anchor)")

        # Expanded: walks descendants tsmc/samsung/sk_hynix/intel.
        q_exp = q_base.expand_hierarchy(store.schema)
        hits = q_exp.run(store.manifest)
        print(f"  with expansion:     {len(hits)} hits "
              f"(subject ∈ {sorted(set(h.loc.subject for h in hits))})")

        # Prune check: manifest pruner should still fire on the set.
        from cognition.cassette.index import _prune_by_anchors
        subj_set = set(q_exp.subject_set)
        paths = _prune_by_anchors(store.manifest, "by_triple",
                                   subj_set, ("subjects",))
        n_cass = len(store.manifest.cassettes)
        print(f"  pruner: {len(paths)}/{n_cass} shards kept — "
              f"still prunes, just over an anchor set")

        # ── Agent session A: investigate + record ──────────────────────
        rule("Agent session A: investigate and record_finding")
        from strands.models.bedrock import BedrockModel
        model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-west-2",
        )
        analyst_a = Analyst(store, model=model, stream=False)
        response = analyst_a(
            "Investigate Nvidia's supply chain and the key relationships. "
            "Use connect() and any_of() as appropriate. When you find "
            "something meaningful, record a finding so future sessions "
            "can reuse it. Keep it concise."
        )
        print(response[:1500] + ("..." if len(response) > 1500 else ""))

        # Show what the filesystem looks like after session A
        findings_dir = os.path.join(root, "findings")
        if os.path.isdir(findings_dir):
            files = sorted(os.listdir(findings_dir))
            print(f"\n  findings on disk: {len(files)}")
            for name in files[:5]:
                print(f"    {name}")

        # ── Agent session B: fresh Analyst, same store ─────────────────
        rule("Agent session B: fresh Analyst reads findings")
        analyst_b = Analyst(store, model=model, stream=False)
        response = analyst_b(
            "What have previous investigations concluded about this corpus? "
            "Just summarize what's in the findings — don't re-run queries."
        )
        print(response[:1500] + ("..." if len(response) > 1500 else ""))

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
