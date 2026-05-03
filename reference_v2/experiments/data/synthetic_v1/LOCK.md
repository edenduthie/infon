# LOCK — synthetic_v1 Dataset

**Status: IMMUTABLE**

Locked: 2026-05-03  
Generator task: infon-8pa.20  
Generator seed: 42  
Generator version: reference_v2/reference_v2/synthetic/generate.py (post-B.3 fix)

## Counts

| Split | Scenarios |
|-------|-----------|
| train | 8000 |
| dev   | 1000 |
| test  | 1000 |
| **total** | **10000** |

## Thinness curriculum (test split)

| evidence_redundancy | count |
|--------------------|-------|
| 1 | 250 |
| 2 | 250 |
| 5 | 250 |
| 10 | 250 |

All strata ≥ 100 as required by spec.

## SHA-256 Hashes

```
e96f9dcbf74b2282ea56aa8fc080756e5a306cc5166d349583a66dcc19f3c53b  train.jsonl
7acdbf8ff2081a8614c2b1433ba62997232a56a32a77caa11a5536319a188921  dev.jsonl
32c8934de6f90c9627b22a874770b2103de27b19844d8b1a94abd2d8e24cc337  test.jsonl
```

## Immutability Policy

Any modification to this dataset **must** use the name `synthetic_v2`.  
Do not edit files in this directory. Any downstream re-run that changes these files invalidates the locked aggregate results.
