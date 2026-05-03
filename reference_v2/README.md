# reference_v2

Reference implementation of the cognition stack reviewed in
`docs/publication/draft2.txt` and audited in
`docs/publication/reproduction_audit.md`. The package under
`src/cognition/` is the seeded-deterministic version; the experiment
runner under `experiments/` exercises it.

## Phase 1: Θ collapse fix

The audit (`docs/publication/reproduction_audit.md`, 2026-05-01) found
that the released code's diagnostic-query Θ output had collapsed to
`≈ 0.002` rather than the paper's claimed `≈ 0.30`, sign-reversing the
central H2 claim that residual mass on Θ tracks evidential thinness.
Stage B.2's per-infon diagnostic (`experiments/results/diagnostic/`)
located the collapse to fusion (not training): pre-fusion per-infon
`m(Θ)` had mean `≈ 0.235` already inside the acceptance band, and
Dempster's rule on five agreeing decisive masses was shrinking it
multiplicatively by ~100–200×. The 480-cell sweep at
`experiments/configs/sweep_collapse.yaml` (6 cw × 4 fusion × 4 top-k ×
5 seeds) produced 10 acceptance-passing cells; the chosen canonical
configuration is `top1, decisive_top_k=2, coherence_weight=1.0`,
committed at `experiments/configs/canonical_v0_2.yaml`. Across
seeds `{42, 0, 1, 7, 13}` it produces Toyota Θ = 0.2330 ± 0.0082 and
Honda Θ = 0.2247 ± 0.0092, both in `[0.20, 0.40]` with polarity
SUPPORTS-correct on all four diagnostic queries (Toyota, Honda, Tesla,
CATL).

The full audit-→-fix story, sweep figures, anomaly catalogue, and
Epic-02 follow-up list are in
[`docs/publication/phase1_collapse_fix.md`](../docs/publication/phase1_collapse_fix.md).

## Reproducing the canonical run

```bash
cd reference_v2
PYTHONPATH=src python3 -c "from experiments.run import run; \
    run('experiments/configs/canonical_v0_2.yaml', \
        'experiments/results/canonical_v0_2/')"
```

Total wall-clock ~17 seconds on a single CPU; outputs five JSON
reports at `experiments/results/canonical_v0_2/canonical_v0_2__seed=*.json`.
The runner is byte-deterministic for a fixed seed.

## Synthetic Stress Test

The `synthetic_v1` dataset (10 000 scenarios) and the eight-cell ablation matrix exercise the
cognition stack on oracle-labelled template corpora. The generator is deterministic, LLM-free,
and produces exact oracle labels for three axes: H1 (compositional depth / hop count), H2
(evidence thinness / `evidence_redundancy`), and a contradiction-density confound. Full design
rationale and findings are in
[`docs/publication/phase2_synthetic_ablations.md`](../docs/publication/phase2_synthetic_ablations.md).

### Generator CLI

```bash
cd reference_v2
PYTHONPATH=src .venv/bin/python -m reference_v2.synthetic.generate \
  --train 8000 --dev 1000 --test 1000 --seed 42 \
  --out experiments/data/synthetic_v1/
```

### Ablation matrix CLI

```bash
cd reference_v2
PYTHONPATH=src .venv/bin/python -m reference_v2.experiments.ablation_matrix \
  --config experiments/configs/ablation_matrix.yaml \
  --seeds 42,0,1 \
  --data experiments/data/synthetic_v1 \
  --out experiments/results/ablation_matrix/ \
  --max-train 500 --checkpoint
```

Paper headline numbers (8 cells × 5 seeds) are at
`experiments/results/canonical_cells/aggregate.json`.

## Test suite

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

The Epic-01 test family (`test_canonical_config.py`, `test_logic.py`,
`test_seed_pinning.py`, `test_fusion_rules.py`, `test_decisive_top_k.py`,
`test_per_infon_mass_logger.py`, `test_experiment_runner.py`) is 41 / 41
green as of v0.2.0-phase1.
