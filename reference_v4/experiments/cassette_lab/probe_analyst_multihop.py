"""Commit 4 demo: Analyst handles multi-hop and any-of-many questions.

Seeds a store with good coverage (the schema is pre-tuned so we're not
re-running Commit 3's refinement loop), then hands the agent three
question types:

  Single-claim  → should pick ask()
  Connectivity  → should pick connect()
  One-of-many   → should pick any_of()

Target behaviors:
  • Agent selects the right tool for each question shape.
  • Never uses connect() for a question ask() can answer.
  • any_of() is called ONCE for the target list, not N times.
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


# Pre-tuned schema — what Commit 3's agent converged to, minus the
# things that didn't matter. ~20 anchors, covers all 15 docs.
SCHEMA = {
    "nvidia":    {"type": "actor", "tokens": ["nvidia"]},
    "tsmc":      {"type": "actor", "tokens": ["tsmc"]},
    "intel":     {"type": "actor", "tokens": ["intel"]},
    "amd":       {"type": "actor", "tokens": ["amd"]},
    "openai":    {"type": "actor", "tokens": ["openai"]},
    "anthropic": {"type": "actor", "tokens": ["anthropic"]},
    "microsoft": {"type": "actor", "tokens": ["microsoft", "azure"]},
    "google":    {"type": "actor", "tokens": ["google", "alphabet"]},
    "samsung":   {"type": "actor", "tokens": ["samsung"]},
    "sk_hynix":  {"type": "actor", "tokens": ["hynix"]},
    "aws":       {"type": "actor", "tokens": ["aws"]},

    "partner":   {"type": "relation",
                   "tokens": ["partner", "partnered", "partnership", "venture"]},
    "supply":    {"type": "relation",
                   "tokens": ["supply", "supplies", "supplier", "sources"]},
    "invest":    {"type": "relation",
                   "tokens": ["invest", "invested", "investment"]},
    "acquire":   {"type": "relation",
                   "tokens": ["acquire", "acquired"]},
    "compete":   {"type": "relation",
                   "tokens": ["compete", "competing", "rival"]},

    "hbm":       {"type": "feature", "tokens": ["hbm", "high-bandwidth memory"]},
    "b200":      {"type": "feature", "tokens": ["b200"]},
    "tpu":       {"type": "feature", "tokens": ["tpu"]},
    "3nm":       {"type": "feature", "tokens": ["3nm"]},
    "foundry":   {"type": "feature", "tokens": ["foundry"]},
    "compute":   {"type": "feature", "tokens": ["compute", "datacenter", "data center"]},
}


DOCS = [
    {"id": "d1",  "text": "Nvidia partnered with TSMC to produce the new B200 chip on the 3nm process.", "timestamp": "2026-01-05"},
    {"id": "d2",  "text": "SK Hynix supplies HBM memory to Nvidia for its datacenter GPU line.", "timestamp": "2026-01-08"},
    {"id": "d3",  "text": "Samsung announced it also supplies HBM to Nvidia, intensifying competition with SK Hynix.", "timestamp": "2026-01-12"},
    {"id": "d4",  "text": "Microsoft invested heavily in datacenter capacity for its OpenAI partnership.", "timestamp": "2026-01-15"},
    {"id": "d5",  "text": "OpenAI partnered with Microsoft for Azure compute, worth billions.", "timestamp": "2026-01-20"},
    {"id": "d6",  "text": "Google invested in custom TPU development, competing with Nvidia's datacenter GPUs.", "timestamp": "2026-02-01"},
    {"id": "d7",  "text": "Anthropic partnered with Google for TPU compute, diversifying away from Nvidia.", "timestamp": "2026-02-05"},
    {"id": "d8",  "text": "Intel announced a foundry push, aiming to compete with TSMC for AI chip orders.", "timestamp": "2026-02-10"},
    {"id": "d9",  "text": "AMD invested in HBM supply contracts with SK Hynix.", "timestamp": "2026-02-15"},
    {"id": "d10", "text": "Nvidia acquired a small startup specializing in datacenter networking.", "timestamp": "2026-02-20"},
    {"id": "d11", "text": "Samsung's HBM supply to Nvidia fell through after quality issues.", "timestamp": "2026-03-01", },
    {"id": "d12", "text": "TSMC's 3nm production bottleneck is delaying Nvidia's B200 ramp.", "timestamp": "2026-03-05"},
    {"id": "d13", "text": "OpenAI and Microsoft extended their partnership with a multi-year compute commitment.", "timestamp": "2026-03-10"},
    {"id": "d14", "text": "Anthropic also partnered with AWS for additional compute redundancy.", "timestamp": "2026-03-15"},
    {"id": "d15", "text": "Intel scrapped its foundry plans, citing insufficient customer commitments.", "timestamp": "2026-04-01"},
]


def divider(label: str):
    print("\n" + "═" * 72)
    print(f"  {label}")
    print("═" * 72)


def main():
    tmp = tempfile.mkdtemp(prefix="analyst_mh_")
    try:
        schema_path = os.path.join(tmp, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)

        store = InfonStore(os.path.join(tmp, "store"),
                            schema_path=schema_path)
        print("ingesting docs...")
        r = store.ingest(DOCS)
        zero = len(r["report"].docs_with_zero_infons)
        print(f"  {r['n_infons']} infons, {r['report'].n_docs - zero}/"
              f"{r['report'].n_docs} docs covered\n")

        from strands.models.bedrock import BedrockModel
        model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-west-2",
        )
        analyst = Analyst(store, model=model, stream=False)

        # Turn 1 — clearly single-hop. Agent should pick ask().
        divider("TURN 1 (single-hop): 'Does Nvidia supply HBM?'")
        print(analyst("Does Nvidia supply HBM?"))

        # Turn 2 — explicitly connectivity. Agent should pick connect().
        divider("TURN 2 (connectivity): 'Is Anthropic connected to Nvidia "
                "through any chain of partnerships or supply relationships?'")
        print(analyst(
            "Is Anthropic connected to Nvidia through any chain of "
            "partnerships or supply relationships?"
        ))

        # Turn 3 — one-of-many. Agent should pick any_of().
        divider("TURN 3 (one-of-many): 'Which chip companies is OpenAI "
                "linked to — nvidia, tsmc, intel, amd, samsung, sk_hynix?'")
        print(analyst(
            "Which chip companies is OpenAI linked to? Consider these: "
            "nvidia, tsmc, intel, amd, samsung, sk_hynix."
        ))

        # Turn 4 — trap: sounds multi-hop but is actually single-hop.
        # "Does Microsoft partner with OpenAI?" → ask(), not connect().
        divider("TURN 4 (trap — one hop phrased like connectivity): "
                "'Is there a relationship between Microsoft and OpenAI?'")
        print(analyst(
            "Is there a relationship between Microsoft and OpenAI?"
        ))

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
