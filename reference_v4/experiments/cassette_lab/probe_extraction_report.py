"""Commit 1: extraction_report diagnostic.

Re-runs the journalist walkthrough with the same schema and docs, but
now the report surfaces each failure mode automatically after ingest.
Compare the diagnostics to the user's hand-written complaints from the
prior session — everything they noticed manually should appear in the
report.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette import InfonStore


SCHEMA = {
    # Actors
    "nvidia":    {"type": "actor",    "tokens": ["nvidia"]},
    "tsmc":      {"type": "actor",    "tokens": ["tsmc"]},
    "intel":     {"type": "actor",    "tokens": ["intel"]},
    "amd":       {"type": "actor",    "tokens": ["amd"]},
    "openai":    {"type": "actor",    "tokens": ["openai"]},
    "anthropic": {"type": "actor",    "tokens": ["anthropic"]},
    "microsoft": {"type": "actor",    "tokens": ["microsoft"]},
    "google":    {"type": "actor",    "tokens": ["google", "alphabet"]},
    "samsung":   {"type": "actor",    "tokens": ["samsung"]},
    "sk_hynix":  {"type": "actor",    "tokens": ["hynix", "sk hynix"]},
    # Relations
    "supply":    {"type": "relation", "tokens": ["supply", "supplies", "supplier"]},
    "invest":    {"type": "relation", "tokens": ["invest", "invested", "investment"]},
    "partner":   {"type": "relation", "tokens": ["partner", "partnered", "partnership"]},
    "acquire":   {"type": "relation", "tokens": ["acquire", "acquired", "buy", "bought"]},
    "compete":   {"type": "relation", "tokens": ["compete", "rival"]},
    # Features — the "overfit object" issue comes from having only these.
    "h200":      {"type": "feature",  "tokens": ["h200"]},
    "b200":      {"type": "feature",  "tokens": ["b200"]},
    "hbm":       {"type": "feature",  "tokens": ["hbm", "high-bandwidth memory"]},
    "datacenter":{"type": "feature",  "tokens": ["datacenter", "data center"]},
    "3nm":       {"type": "feature",  "tokens": ["3nm", "3-nanometer"]},
}

DOCS = [
    {"id": "d1",  "text": "Nvidia partnered with TSMC to produce the new B200 chip on the 3nm process.",
     "timestamp": "2026-01-05"},
    {"id": "d2",  "text": "SK Hynix supplies HBM memory to Nvidia for its H200 datacenter GPU line.",
     "timestamp": "2026-01-08"},
    {"id": "d3",  "text": "Samsung announced it also supplies HBM to Nvidia, intensifying competition with SK Hynix.",
     "timestamp": "2026-01-12"},
    {"id": "d4",  "text": "Microsoft invested heavily in datacenter capacity for its OpenAI partnership.",
     "timestamp": "2026-01-15"},
    {"id": "d5",  "text": "OpenAI partnered with Microsoft for exclusive Azure compute, rumored worth 10 billion.",
     "timestamp": "2026-01-20"},
    {"id": "d6",  "text": "Google invested in custom TPU development, competing directly with Nvidia's datacenter GPUs.",
     "timestamp": "2026-02-01"},
    {"id": "d7",  "text": "Anthropic partnered with Google for TPU compute, diversifying away from Nvidia.",
     "timestamp": "2026-02-05"},
    {"id": "d8",  "text": "Intel announced a 3nm foundry push, aiming to compete with TSMC for AI chip orders.",
     "timestamp": "2026-02-10"},
    {"id": "d9",  "text": "AMD invested in HBM supply contracts with SK Hynix for its MI300X line.",
     "timestamp": "2026-02-15"},
    {"id": "d10", "text": "Nvidia acquired a small startup specializing in datacenter networking.",
     "timestamp": "2026-02-20"},
    {"id": "d11", "text": "Samsung's HBM deal with Nvidia fell through after quality issues were reported.",
     "timestamp": "2026-03-01"},
    {"id": "d12", "text": "TSMC's 3nm production bottleneck is delaying Nvidia's B200 ramp.",
     "timestamp": "2026-03-05"},
    {"id": "d13", "text": "OpenAI and Microsoft extended their partnership with a multi-year compute commitment.",
     "timestamp": "2026-03-10"},
    {"id": "d14", "text": "Anthropic also partnered with AWS, not just Google, for additional compute redundancy.",
     "timestamp": "2026-03-15"},
    {"id": "d15", "text": "Intel scrapped its 3nm foundry plans, citing insufficient customer commitments.",
     "timestamp": "2026-04-01"},
]


def main():
    tmp = tempfile.mkdtemp(prefix="report_probe_")
    schema_path = os.path.join(tmp, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f)

    store = InfonStore(os.path.join(tmp, "store"), schema_path=schema_path)

    print("─" * 72)
    print("INGEST")
    print("─" * 72)
    result = store.ingest(DOCS)
    print(f"  ingested={len(result['ingested'])}  "
          f"skipped={len(result['skipped'])}  "
          f"errors={len(result['errors'])}  "
          f"n_infons={result['n_infons']}")

    # THE NEW PART — the report comes back from ingest() for free.
    print("\n─" * 36 + " REPORT " + "─" * 36)
    report = result["report"]
    assert report is not None, "ingest() should attach a report"
    print(report.summary())

    # ── Cross-check against the journalist's hand-written complaints ────
    print("\n─" * 72)
    print("\nCROSS-CHECK: did the report surface the user's manual findings?")
    print("─" * 72)

    # Pain #1: d1 produced 0 infons
    zero = set(report.docs_with_zero_infons)
    check("d1 flagged as extracting nothing", "d1" in zero)

    # Pain #2: many anchors never fired (b200, h200, 3nm, tsmc, intel, ...)
    unused = set(report.unused_anchors)
    for expected in ("b200", "h200", "3nm", "tsmc"):
        check(f"unused anchor '{expected}' flagged", expected in unused)

    # Pain #3: 'datacenter' hoarding the object role
    overfit_names = {name for name, _ in report.overfit_objects}
    check("'datacenter' flagged as overfit object", "datacenter" in overfit_names)

    # Pain #4: multiple docs produced nothing
    check("at least 3 docs flagged as zero-infon",
          len(report.docs_with_zero_infons) >= 3)

    # Show the report's JSON form — what the agent layer will consume.
    print("\n─" * 72)
    print("REPORT AS JSON (what Commit 2's agent will see)")
    print("─" * 72)
    import json as _json
    print(_json.dumps(report.to_dict(), indent=2)[:1200] + "...")

    shutil.rmtree(tmp)


def check(label: str, ok: bool):
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}")


if __name__ == "__main__":
    main()
