"""Commit 3 demo: Analyst bootstraps a schema from scratch and refines it.

Two scenarios:

  Scenario A — cold start, no schema at all.
    User hands over 15 AI-chip docs. Agent proposes ontology, ingests,
    reads coverage report, revises, reingests. Target: >=70% doc coverage
    within 3 iterations.

  Scenario B — schema exists but is minimal/broken.
    User provides only 4 anchors. Agent adds missing anchors based on
    corpus inspection and the extraction report.

Target metric: comparing coverage (docs with ≥1 infon / total docs) before
and after the refinement loop.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette import InfonStore, Analyst


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


# Deliberately-minimal schema for Scenario B — 4 anchors total so we can
# watch the agent extend it. Has the same failures as no schema at all.
MINIMAL_SCHEMA = {
    "nvidia":   {"type": "actor",    "tokens": ["nvidia"]},
    "partner":  {"type": "relation", "tokens": ["partner", "partnered"]},
    "supply":   {"type": "relation", "tokens": ["supply", "supplies"]},
    "hbm":      {"type": "feature",  "tokens": ["hbm"]},
}


def divider(label: str):
    print("\n" + "═" * 72)
    print(f"  {label}")
    print("═" * 72)


def coverage_fraction(store):
    """How many docs produced ≥1 infon / total known docs."""
    r = store.extraction_report()
    if r.n_docs == 0:
        return 0.0
    return (r.n_docs - len(r.docs_with_zero_infons)) / r.n_docs


def scenario_a_cold_start(tmp_root):
    divider("SCENARIO A: no schema — agent must propose one from scratch")

    # Bootstrap the store WITHOUT a schema. The agent will call set_schema.
    # We do have to give InfonStore a path-free seed since set_schema is
    # required before ingest — use a trivial stub that the agent will
    # replace on its first tool call.
    stub_path = os.path.join(tmp_root, "stub.json")
    with open(stub_path, "w") as f:
        json.dump({"_": {"type": "actor", "tokens": ["_"]}}, f)

    store = InfonStore(os.path.join(tmp_root, "store_a"),
                       schema_path=stub_path)

    from strands.models.bedrock import BedrockModel
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name="us-west-2",
    )
    analyst = Analyst(store, model=model, stream=False)

    prompt = (
        f"I have {len(DOCS)} news snippets about the AI chip industry, but "
        f"no schema yet. Can you design one, ingest the docs, and iterate "
        f"on the schema until coverage is good? Aim for at least 70% of "
        f"docs producing infons.\n\n"
        f"Here are the docs as JSON:\n\n{json.dumps(DOCS)}"
    )
    print(analyst(prompt))

    print("\n" + "─" * 72)
    print(f"  final coverage: {coverage_fraction(store):.0%}")
    print(f"  final cassettes: {len(store.manifest.cassettes)}")
    r = store.extraction_report()
    print(f"  final infons: {r.n_infons}")
    print(f"  docs with 0 infons: {len(r.docs_with_zero_infons)}/{r.n_docs}")
    print(f"  unused anchors: {len(r.unused_anchors)}/{r.total_anchors}")


def scenario_b_minimal_schema(tmp_root):
    divider("SCENARIO B: minimal (broken) schema — agent must extend it")

    schema_path = os.path.join(tmp_root, "minimal.json")
    with open(schema_path, "w") as f:
        json.dump(MINIMAL_SCHEMA, f)

    store = InfonStore(os.path.join(tmp_root, "store_b"),
                       schema_path=schema_path)

    # First, ingest with the minimal schema so the agent sees the broken
    # state via extraction_report, not from a fresh empty corpus.
    result = store.ingest(DOCS)
    baseline = coverage_fraction(store)
    print(f"\nbaseline coverage (minimal schema): {baseline:.0%}")
    print(f"baseline infons: {result['n_infons']}")

    from strands.models.bedrock import BedrockModel
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name="us-west-2",
    )
    analyst = Analyst(store, model=model, stream=False)

    prompt = (
        f"I already ingested {len(DOCS)} AI-chip news docs, but I only had "
        f"a minimal schema. Check the extraction_report and extend the "
        f"schema to capture what I missed. Aim for at least 70% doc "
        f"coverage after reingestion.\n\n"
        f"Original docs as JSON:\n\n{json.dumps(DOCS)}"
    )
    print(analyst(prompt))

    final = coverage_fraction(store)
    print("\n" + "─" * 72)
    print(f"  baseline → final coverage: {baseline:.0%} → {final:.0%}")


def main():
    tmp = tempfile.mkdtemp(prefix="analyst_schema_")
    try:
        scenario_a_cold_start(tmp)
        scenario_b_minimal_schema(tmp)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
