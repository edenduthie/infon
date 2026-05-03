# Baseline ("before-state") audit reproducer

This directory holds the committed baseline reports referenced by Stage B/C/D
of Epic 01 (`infon-6o3`). The JSON files capture the released `reference_v2/`
behaviour that `docs/publication/reproduction_audit.md` flagged as a
Θ-collapse: every diagnostic query returns `m(Θ) ≈ 0.002` rather than the
paper's claimed `m(Θ) ≈ 0.30`.

These artifacts are the **anchor** against which Stage C/D fixes are diffed;
they should not be regenerated casually.

## What is here

```
baseline__seed=42.json
baseline__seed=0.json
baseline__seed=1.json
README.md          (this file)
```

Each JSON conforms to the runner schema in `experiments/run.py` (top-level
keys: `config`, `seed`, `loss_trace`, `queries`).

## How to reproduce

From `reference_v2/`:

```bash
PYTHONPATH=src python3 -c "from experiments.run import run; \
    run('experiments/configs/baseline.yaml', 'experiments/results/baseline/')"
```

The runner is deterministic for a fixed seed (A.2b), so this command
overwrites the JSONs byte-for-byte. CPU runtime: ~90 seconds for all
three seeds.

The exact configuration used is committed at
`experiments/configs/baseline.yaml`. Note: `decisive_top_k: 5` in that
YAML is **deliberately the audit-time default** (pre-A.5b) so that the
audit numbers reproduce verbatim. Do not "fix" it to the new default of
3 — that would silently detach the anchor from the audit.

## Cross-seed Θ table (the "collapse")

| Seed | Toyota m(Θ) | Honda m(Θ) | Tesla m(Θ) | CATL m(Θ) | Final loss |
|-----:|------------:|-----------:|-----------:|----------:|-----------:|
|  42  |    0.0015   |   0.0023   |   0.0009   |   0.0003  |   0.069687 |
|   0  |    0.0017   |   0.0020   |   0.0012   |   0.0003  |   0.071861 |
|   1  |    0.0015   |   0.0021   |   0.0007   |   0.0003  |   0.067456 |

All twelve verdicts (4 queries × 3 seeds) are `SUPPORTS`. Toyota Θ is in
`[0.0015, 0.0017]`; Honda Θ is in `[0.0020, 0.0023]` — both ~150× below
the paper's claimed `m(Θ) ≈ 0.30`. The collapse is stable across seeds,
ruling out random-init noise as the explanation.

## Cross-check vs `paper_scenario_report.json`

The audit's original report (committed as
`reference_v2/paper_scenario_report.json`) was produced by the seed=42
run of the released code. The seed=42 JSON in this directory matches it
to four-decimal precision on every (S, R, U, Θ) value and on the final
loss (`0.069687`). The runner reproduces the audit anchor bit-for-bit,
so any post-fix change in Stage C/D can be attributed to the
intervention rather than to runner drift.

## Downstream consumers

- **B.2** (`infon-6o3.15`): per-infon mass diagnostic — reads
  `baseline__seed=42.json` to confirm the Θ-collapse is mass-level, not
  fusion-level.
- **B.3 / B.4** (`infon-6o3.16`, `.17`): the 480-cell sweep aggregates
  Θ-recovery deltas from this baseline.
- **C.1+** and **D.1+**: the "before-state" column of every comparison
  plot and the phase-1 memo's quantitative claims.
