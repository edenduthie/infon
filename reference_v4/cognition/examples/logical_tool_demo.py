"""Agent-style logical query evaluation.

Simulates the pattern an LLM agent would use: receive a natural-language
question, emit a JSON logical expression, pass it to the tool, inspect
the verdict + mass + trace.

We're not calling an actual LLM here — we hand-write the JSON the LLM
would emit — but the shape is exactly what a Claude / GPT tool call
would produce. `cognition.agent_tools.evaluate_logic` is the `@tool`
wrapper an LLM framework plugs into.

Run:
    python examples/logical_tool_demo.py
"""
from __future__ import annotations

import json
import os
import tempfile

from cognition import Cognition, CognitionConfig
from cognition.logical_tool import evaluate


# ═══════════════════════════════════════════════════════════════════════
# Corpus
# ═══════════════════════════════════════════════════════════════════════

SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "panasonic":{"type": "actor",    "tokens": ["panasonic"]},
    "catl":     {"type": "actor",    "tokens": ["catl"]},
    "invests":  {"type": "relation", "tokens": ["invest", "invests", "investment"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners", "partnership"]},
    "produces": {"type": "relation", "tokens": ["produce", "produces"]},
    "acquires": {"type": "relation", "tokens": ["acquire", "acquires"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    "factory":  {"type": "feature",  "tokens": ["factory", "plant"]},
    "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
    "china":    {"type": "market",   "tokens": ["china", "chinese"]},
}

DOCS = [
    {"id": "d01", "text": "Toyota invests in battery technology in Japan."},
    {"id": "d02", "text": "Toyota partners with Panasonic on battery development."},
    {"id": "d03", "text": "Tesla invests in battery factories."},
    {"id": "d04", "text": "Honda partners with CATL on battery supply in China."},
    {"id": "d05", "text": "CATL produces batteries for Japanese automakers."},
    {"id": "d06", "text": "Panasonic invests in a battery factory in Japan."},
]


# ═══════════════════════════════════════════════════════════════════════
# Four agent queries, translated into JSON — this is what the LLM emits
# ═══════════════════════════════════════════════════════════════════════

AGENT_QUERIES = [
    (
        "Did Toyota invest in batteries?",
        # LLM-emitted JSON (a simple triple):
        {
            "op": "triple",
            "s": "toyota", "p": "invests", "o": "battery",
        },
    ),

    (
        "Did Toyota invest in batteries AND partner with Panasonic?",
        # Conjunction — both must be supported
        {
            "op": "and", "args": [
                {"op": "triple", "s": "toyota", "p": "invests", "o": "battery"},
                {"op": "triple", "s": "toyota", "p": "partners", "o": "panasonic"},
            ],
        },
    ),

    (
        "Did Toyota invest in batteries BUT NOT acquire CATL?",
        # Conjunction with a negated clause — honest test of the
        # "confident on one claim, ignorant on the other" behaviour
        {
            "op": "and", "args": [
                {"op": "triple", "s": "toyota", "p": "invests", "o": "battery"},
                {"op": "not", "args": [
                    {"op": "triple", "s": "toyota", "p": "acquires", "o": "catl"},
                ]},
            ],
        },
    ),

    (
        "Is there ANY actor that invests in batteries?",
        # Existential — the LLM's "does there exist..." becomes an exists
        {
            "op": "exists",
            "type": "actor",
            "body": {"op": "triple", "s": "?x", "p": "invests", "o": "battery"},
        },
    ),

    (
        "Would every actor that partners with CATL also produce batteries?",
        # Material conditional over a quantifier — deep nesting.
        # "For every actor x, (partners(x, catl) → produces(x, battery))"
        {
            "op": "forall", "type": "actor", "body": {
                "op": "if", "args": [
                    {"op": "triple", "s": "?x", "p": "partners", "o": "catl"},
                    {"op": "triple", "s": "?x", "p": "produces", "o": "battery"},
                ],
            },
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Logical-tool demo — the JSON below is what an LLM agent would emit")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp:
        schema_path = os.path.join(tmp, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)

        cog = Cognition(CognitionConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmp, "cog.db"),
            random_state=42,
        ))
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()

        reasoner = cog.reasoner()

        for i, (nl, expr) in enumerate(AGENT_QUERIES, 1):
            print(f"\n── query {i}/{len(AGENT_QUERIES)} ─────────────────────")
            print(f"  natural language:")
            print(f"    {nl}")
            print(f"\n  LLM-emitted JSON (pretty-printed):")
            for line in json.dumps(expr, indent=2).splitlines():
                print(f"    {line}")

            result = evaluate(cog, expr, reasoner=reasoner)
            m = result["mass"]
            print(f"\n  verdict:  {result['verdict']}")
            print(f"  supports: {m['supports']:.3f}   "
                  f"refutes:  {m['refutes']:.3f}   "
                  f"θ:        {m['theta']:.3f}")

            # Show the trace (bottom-up)
            print(f"\n  trace ({len(result['trace'])} sub-expressions):")
            for step in result["trace"]:
                op = step["op"]
                verdict = step.get("verdict", "")
                theta = step.get("mass", {}).get("theta", 0.0)
                if op == "triple":
                    q = step["query"]
                    print(f"    {op:<8s} {q!r:<50s} → {verdict} (θ={theta:.2f})")
                else:
                    n_args = step.get("n_args", step.get("n_candidates", ""))
                    extra = f" [{n_args} children]" if n_args != "" else ""
                    print(f"    {op:<8s}{extra:<15s} → {verdict} (θ={theta:.2f})")

        cog.close()

    print("\n" + "=" * 72)
    print("Notes:")
    print("  - An agent chooses the JSON structure based on the question's")
    print("    logical shape — conjunction, negation, quantifier, etc.")
    print("  - Every sub-expression has its own verdict and θ, visible in")
    print("    the trace. The agent can reject a compound claim whose")
    print("    leaves all have θ > 0.5 even if the combined verdict says")
    print("    SUPPORTS.")
    print("  - To use inside an LLM framework, see the `evaluate_logic`")
    print("    @tool in cognition.agent_tools — it wraps evaluate() with")
    print("    JSON-string I/O and an LLM-facing docstring.")


if __name__ == "__main__":
    main()
