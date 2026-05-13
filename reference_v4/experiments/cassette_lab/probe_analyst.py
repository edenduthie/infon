"""Commit 2 demo: Analyst handles the journalist's workflow conversationally.

Compare the interaction below to /tmp/chip_analyst/analyze.py (the
journalist's hand-written script). The agent should:
  1. Ingest, then self-report coverage issues.
  2. Propose schema fixes without being asked.
  3. Translate NL questions into ask() calls.
  4. Refuse to answer when θ is high (no filling in from LLM knowledge).
  5. Cite sources verbatim.
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


# Same schema + docs as probe_extraction_report.py — reuse so we can
# observe the agent reacting to the same pathological extraction.
SCHEMA = {
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
    "supply":    {"type": "relation", "tokens": ["supply", "supplies", "supplier"]},
    "invest":    {"type": "relation", "tokens": ["invest", "invested", "investment"]},
    "partner":   {"type": "relation", "tokens": ["partner", "partnered", "partnership"]},
    "acquire":   {"type": "relation", "tokens": ["acquire", "acquired", "buy", "bought"]},
    "compete":   {"type": "relation", "tokens": ["compete", "rival"]},
    "h200":      {"type": "feature",  "tokens": ["h200"]},
    "b200":      {"type": "feature",  "tokens": ["b200"]},
    "hbm":       {"type": "feature",  "tokens": ["hbm", "high-bandwidth memory"]},
    "datacenter":{"type": "feature",  "tokens": ["datacenter", "data center"]},
    "3nm":       {"type": "feature",  "tokens": ["3nm", "3-nanometer"]},
}

DOCS = [
    {"id": "d1",  "text": "Nvidia partnered with TSMC to produce the new B200 chip on the 3nm process.", "timestamp": "2026-01-05"},
    {"id": "d2",  "text": "SK Hynix supplies HBM memory to Nvidia for its H200 datacenter GPU line.", "timestamp": "2026-01-08"},
    {"id": "d3",  "text": "Samsung announced it also supplies HBM to Nvidia, intensifying competition with SK Hynix.", "timestamp": "2026-01-12"},
    {"id": "d4",  "text": "Microsoft invested heavily in datacenter capacity for its OpenAI partnership.", "timestamp": "2026-01-15"},
    {"id": "d5",  "text": "OpenAI partnered with Microsoft for exclusive Azure compute, rumored worth 10 billion.", "timestamp": "2026-01-20"},
    {"id": "d6",  "text": "Google invested in custom TPU development, competing directly with Nvidia's datacenter GPUs.", "timestamp": "2026-02-01"},
    {"id": "d7",  "text": "Anthropic partnered with Google for TPU compute, diversifying away from Nvidia.", "timestamp": "2026-02-05"},
    {"id": "d8",  "text": "Intel announced a 3nm foundry push, aiming to compete with TSMC for AI chip orders.", "timestamp": "2026-02-10"},
    {"id": "d9",  "text": "AMD invested in HBM supply contracts with SK Hynix for its MI300X line.", "timestamp": "2026-02-15"},
    {"id": "d10", "text": "Nvidia acquired a small startup specializing in datacenter networking.", "timestamp": "2026-02-20"},
    {"id": "d11", "text": "Samsung's HBM deal with Nvidia fell through after quality issues were reported.", "timestamp": "2026-03-01"},
    {"id": "d12", "text": "TSMC's 3nm production bottleneck is delaying Nvidia's B200 ramp.", "timestamp": "2026-03-05"},
    {"id": "d13", "text": "OpenAI and Microsoft extended their partnership with a multi-year compute commitment.", "timestamp": "2026-03-10"},
    {"id": "d14", "text": "Anthropic also partnered with AWS, not just Google, for additional compute redundancy.", "timestamp": "2026-03-15"},
    {"id": "d15", "text": "Intel scrapped its 3nm foundry plans, citing insufficient customer commitments.", "timestamp": "2026-04-01"},
]


def divider(label: str):
    print("\n" + "═" * 72)
    print(f"  {label}")
    print("═" * 72)


def main():
    tmp = tempfile.mkdtemp(prefix="analyst_demo_")
    schema_path = os.path.join(tmp, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f)

    store = InfonStore(os.path.join(tmp, "store"), schema_path=schema_path)

    # The Strands default model is deprecated; pin to Sonnet 4.5 which is
    # the current Claude generally available on Bedrock. Users with their
    # own Bedrock access can swap this for whatever they've got enabled.
    from strands.models.bedrock import BedrockModel
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name="us-west-2",
    )
    analyst = Analyst(store, model=model, stream=False)

    # Turn 1: hand over the corpus. Agent should ingest + run extraction_report
    # + flag coverage issues + suggest fixes.
    divider("TURN 1: journalist hands over the corpus")
    turn1 = (
        f"I have {len(DOCS)} news snippets about AI chip industry. "
        f"Here they are as JSON:\n\n{json.dumps(DOCS)}\n\n"
        f"Please ingest them and tell me what you found — including any "
        f"issues with the extraction."
    )
    print(analyst(turn1))

    # Turn 2: natural-language query that requires translation to triple form.
    divider("TURN 2: 'Does Nvidia partner with TSMC?'")
    print(analyst("Does Nvidia partner with TSMC?"))

    # Turn 3: question the agent probably cannot answer honestly.
    divider("TURN 3: 'Who does OpenAI compete with?'")
    print(analyst("Who does OpenAI compete with?"))

    # Turn 4: a claim that's present AND later refuted — do sources show both?
    divider("TURN 4: 'Does Samsung supply HBM to Nvidia?'")
    print(analyst("Does Samsung supply HBM to Nvidia?"))

    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
