# synthetic_v1 Generation Notes

IMMUTABLE: do not modify — any change becomes synthetic_v2

## Generation Parameters

| Parameter             | Value  |
|-----------------------|--------|
| --train               | 8000   |
| --dev                 | 1000   |
| --test                | 1000   |
| --seed                | 42     |
| --compositional-depth | 2      |
| --contradiction-density | 0.33 |
| --nei-fraction        | 0.33   |

Command used:
```bash
cd /home/ubuntu/infon/reference_v2
PYTHONPATH=src .venv/bin/python -m reference_v2.synthetic.generate \
  --train 8000 --dev 1000 --test 1000 --seed 42 \
  --out experiments/data/synthetic_v1/
```

## Timing

Wall-clock time: ~0.192s (generation 0.111s + I/O 0.081s)
Generation rate: ~89,743 scenarios/sec

## Thinness Stratification (evidence_redundancy)

Evidence redundancy is assigned by `make_splits()` using thinness-curriculum
stratification over {1, 2, 5, 10}. The test split is exactly balanced at 250
per stratum.

### Train split (8000 scenarios)
| evidence_redundancy | count |
|---------------------|-------|
| 1                   | 1996  |
| 2                   | 2024  |
| 5                   | 1986  |
| 10                  | 1994  |

### Dev split (1000 scenarios)
| evidence_redundancy | count |
|---------------------|-------|
| 1                   | 254   |
| 2                   | 226   |
| 5                   | 264   |
| 10                  | 256   |

### Test split (1000 scenarios)
| evidence_redundancy | count |
|---------------------|-------|
| 1                   | 250   |
| 2                   | 250   |
| 5                   | 250   |
| 10                  | 250   |

## Verdict Distribution

### Train split
| verdict  | count |
|----------|-------|
| SUPPORTS | 2736  |
| REFUTES  | 2600  |
| NEI      | 2664  |

### Dev split
| verdict  | count |
|----------|-------|
| SUPPORTS | 380   |
| REFUTES  | 332   |
| NEI      | 288   |

### Test split
| verdict  | count |
|----------|-------|
| SUPPORTS | 332   |
| REFUTES  | 336   |
| NEI      | 332   |

## SHA-256 Hashes (for reproducibility)

```
e96f9dcbf74b2282ea56aa8fc080756e5a306cc5166d349583a66dcc19f3c53b  train.jsonl
7acdbf8ff2081a8614c2b1433ba62997232a56a32a77caa11a5536319a188921  dev.jsonl
32c8934de6f90c9627b22a874770b2103de27b19844d8b1a94abd2d8e24cc337  test.jsonl
```

Verified deterministic: two independent runs with `--seed 42` produce byte-identical files.

## Design Notes

- Evidence redundancy per scenario is determined by `make_splits()` (task infon-8pa.20 fix)
- Scenarios are grouped by `evidence_redundancy` and generated with `Generator.generate(n_docs=count, evidence_redundancy=er, ...)` so each scenario gets the correct supporting-sentence count
- `Generator` re-seeds from `self._seed` at each `generate()` call; strata are iterated in canonical order (1, 2, 5, 10) for determinism
- Output format: JSONL (one JSON object per line), NOT a JSON array
