# BlackMagic

Sparse retrieval + reasoning + GA imagination on `splade-tiny`. English-only
fork of the `cognition` package, stripped of multilingual complexity and
extended with query-time counterfactual imagination.

## Install

```bash
pip install -e .          # development
pip install -e '.[dev]'   # with pytest
```

The 17MB splade-tiny model is bundled — no downloads required.

## Quickstart

```python
from blackmagic import BlackMagic, BlackMagicConfig

bm = BlackMagic(BlackMagicConfig(
    schema_path="examples/automotive_schema.json",
    db_path=":memory:",
))

bm.ingest([
    {"text": "Toyota announced a $13.6B investment in battery production.",
     "id": "d1", "timestamp": "2026-03-01"},
    {"text": "Honda launches new EV in partnership with CATL.",
     "id": "d2", "timestamp": "2026-03-15"},
])

# Sparse retrieval with persona valence
result = bm.search("automakers investing in batteries", persona="investor")
for inf in result.infons[:5]:
    print(inf.subject, inf.predicate, inf.object, inf.confidence)

# Dempster-Shafer claim verification
v = bm.verify_claim("Toyota is aggressively investing in batteries.")
print(v.label, v.belief_supports, v.belief_refutes)

# MCTS multi-hop reasoning
m = bm.reason("Does the industry face supply risks?")
print(m.verdict, m.chains_discovered)

# GA imagination — MCTS-shaped output with dual verdicts
im = bm.imagine("What OEM–supplier partnerships might emerge?")
print(im.verdict, im.mcts_verdict)
for inf in im.imagined_infons[:5]:
    print(inf.subject, inf.predicate, inf.object,
          "fitness=", inf.fitness,
          "parents=", inf.parent_infon_ids)
```

## Features

- **Sparse retrieval** via splade-tiny → typed anchor projection
- **Persona valence** — investor / engineer / executive / regulator / analyst
- **Contrary views** — invert the evidential lens at query time
- **Temporal graph** — NEXT edges link facts across time per shared anchor
- **Constraint aggregation** — cross-document infon fusion
- **Dempster-Shafer** claim verification
- **Graph MCTS** for multi-hop reasoning
- **GA imagination** (new) — query-scoped genetic algorithm that proposes
  plausible counterfactual infons scored by grammar × logic × health,
  with output isomorphic to `MCTSResult`

## When to use cognition vs BlackMagic

| | cognition | BlackMagic |
|---|---|---|
| Languages | EN + JA/KO/ZH/...  | English only |
| Encoder | splade-tiny or multilingual XLM-R | splade-tiny only |
| Structural analysis (Kano, Kan, etc.) | Yes | No |
| Category theory extensions | Yes | No |
| Cloud backend (DynamoDB, Lambda) | Yes | No |
| MCP / agent tooling | Yes | No |
| GA imagination | No | Yes |
| Line count | ~5,700 | ~3,900 |

## Testing

```bash
PYTHONPATH=src pytest tests/
```

## License

Apache 2.0.
