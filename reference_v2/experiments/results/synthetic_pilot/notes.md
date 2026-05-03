# Synthetic Pilot Spot-Check Notes

## Configuration

| Parameter | Value |
|-----------|-------|
| train | 80 |
| dev | 10 |
| test | 10 |
| seed | 42 |
| evidence_redundancy | 2 (default) |
| compositional_depth | 2 (default) |
| contradiction_density | 0.33 (default) |
| nei_fraction | 0.33 (default) |

Command run:

```bash
cd /home/ubuntu/infon/reference_v2
PYTHONPATH=src python3 -m reference_v2.synthetic.generate \
    --train 80 --dev 10 --test 10 --seed 42 \
    --out experiments/results/synthetic_pilot/data/
```

## Generation Rate

- Total scenarios: 100
- Generation time: ~0.010 s
- Rate: ~9800 scenarios/sec

## Spot-Check Results (10 randomly selected scenarios)

Random seed 99 was used to select 10 scenarios from the combined 100-scenario pool.

| # | verdict | planted_thinness | actual_thinness | hops | thinness_ok |
|---|---------|-----------------|-----------------|------|-------------|
| 0 | NEI     | 0               | 0               | 2    | yes |
| 1 | REFUTES | 0               | 0               | 2    | yes |
| 2 | SUPPORTS| 4               | 4               | 2    | yes |
| 3 | NEI     | 0               | 0               | 2    | yes |
| 4 | NEI     | 0               | 0               | 2    | yes |
| 5 | NEI     | 0               | 0               | 2    | yes |
| 6 | SUPPORTS| 4               | 4               | 2    | yes |
| 7 | REFUTES | 0               | 0               | 2    | yes |
| 8 | SUPPORTS| 4               | 4               | 2    | yes |
| 9 | REFUTES | 0               | 0               | 2    | yes |

### Were oracle labels correct in all 10 spot-checked scenarios?

**Yes.** All 10 scenarios had:
- `planted_verdict` correctly set to one of SUPPORTS / REFUTES / NEI
- `planted_thinness` exactly matching the count of `supports_query=True` sentences in `corpus_sentences`
- `planted_hop_count` equal to the `compositional_depth` parameter (2)

### Malformed templates?

**None found.** All sentence texts followed the expected template patterns:

- SUPPORTS supporting: `"Entity {id} hop {hop} witness {redundancy_idx}: fact {id} is confirmed true."`
- REFUTES contradicting: `"Entity {id}: fact {id} is demonstrably false."`
- Distractor (all verdicts): `"Entity {id} distractor {idx}: unrelated context {noise_val}."`

No empty, truncated, or malformed sentences were observed.

## Output Files

- `data/train.json` — 80 scenarios
- `data/dev.json` — 10 scenarios
- `data/test.json` — 10 scenarios

Each file is a JSON array of dicts with keys: `corpus_sentences`, `planted_verdict`, `planted_thinness`, `planted_hop_count`.

## Conclusion

The generator produces correct oracle labels at high speed (~9800 scenarios/sec). No issues found. Safe to proceed to the full 10k dataset generation.
