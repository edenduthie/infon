# legacy/

Prior artifacts archived during the migration from the SQLite/DynamoDB
`Cognition` class to the cassette-native `InfonStore`. Nothing here is
broken — it's been superseded by newer paths that are documented in the
main [README.md](../README.md).

## What's here and why it moved

| Path | What it is | Superseded by |
|---|---|---|
| `notebooks/09_self_supervised_mdp.ipynb` | Research arc: Barlow → JEPA → intrinsic reward, reconstructing an MDP above SPLADE | The synthgen + sheaf-GNN pipeline (`cognition/src/cognition/cassette/synthgen.py` + `gnn_encoder.py`). The "self-supervised corpus reconstruction" story landed as a trained prior, not a live MDP. |
| `notebooks/10_automl.ipynb` | AutoML pitch: `analyze_corpus`, `sweep`, `cross_val_score`, `ensemble_top_k` | `bootstrap_gnn()` + `extraction_report()` — the product's version of "data in, best recipe out". Full estimator-contract AutoML was deferred as aspirational. |
| `scripts/cognition_tui.py` | Terminal UI for the old `Cognition` class | `ask.py` at the repo root, which now uses the cassette store. |
| `archives/BlackMagic.zip` | Prior vendor artifact | n/a — historical reference only. |
| `archives/Ontology-Workshop-V1.zip` | Snapshot of the workshop before the cassette substrate | The current repo. |

## What's still in use and NOT archived

- The `Cognition` class itself in `cognition/src/cognition/` — the legacy
  Python API still works and is imported by some experiments. It coexists
  with `cognition/src/cognition/cassette/`, the new substrate.
- Notebooks 01, 02, 04, 05, 11 in the repo root — patched in place rather
  than archived; their teaching content is still current after minor API
  updates.
- `cognition-workshop/*.md` prose modules — being rewritten in place to
  match the cassette substrate.

## If you're looking for

- **"How does the self-supervised MDP reconstruction work?"** → `legacy/notebooks/09_self_supervised_mdp.ipynb` (the theory) + `cognition/src/cognition/cassette/synthgen.py` (the shipped version).
- **"Is there an AutoML loop?"** → Not shipped as a full estimator contract. See `cognition/src/cognition/cassette/analyst.py::bootstrap_gnn` for the per-corpus training the product does instead.
- **"Why isn't SQLite/DynamoDB the primary backend anymore?"** → [`README.md`](../README.md) — the cassette format is S3-native, immutable, delta-ingestible, and time-travelable in ways the relational path wasn't. DynamoDB is still supported in `cognition/src/cognition/store/cloud.py` for legacy integrations.
