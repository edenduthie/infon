# LLM vs Cognition comparison

Twenty claims drawn from a 4-document EV corpus. Each claim has a gold verdict (SUPPORTS, REFUTES, NOT_ENOUGH_INFO). We compare our cognition pipeline against a deterministic mock LLM baseline.

## Summary

| System | Accuracy | Mean latency (ms) | θ on NEI claims | Verdict distribution |
|---|---|---|---|---|
| cognition | 85% | 177 | 1.00 | S:8 / R:0 / NEI:12 |
| mock-llm | 45% | 301 | 0.11 | S:20 / R:0 / NEI:0 |

## Per-claim results

| Claim | Gold | Cognition | θ | LLM | LLM θ |
|---|---|---|---|---|---|
| Did Toyota invest in battery technology? | SUPPORTS | SUPPORTS ✓ | 0.00 | SUPPORTS ✓ | 0.10 |
| Does Toyota partner with Panasonic? | SUPPORTS | NOT_ENOUGH_INFO ✗ | 1.00 | SUPPORTS ✓ | 0.10 |
| Does Tesla produce batteries? | SUPPORTS | SUPPORTS ✓ | 0.05 | SUPPORTS ✓ | 0.10 |
| Does Honda partner with CATL? | SUPPORTS | NOT_ENOUGH_INFO ✗ | 1.00 | SUPPORTS ✓ | 0.10 |
| Does CATL produce batteries? | SUPPORTS | SUPPORTS ✓ | 0.00 | SUPPORTS ✓ | 0.10 |
| Did Toyota invest in solid-state batteries? | SUPPORTS | SUPPORTS ✓ | 0.00 | SUPPORTS ✓ | 0.10 |
| Does Tesla expand its factory? | SUPPORTS | SUPPORTS ✓ | 0.04 | SUPPORTS ✓ | 0.10 |
| Does CATL produce batteries in China? | SUPPORTS | SUPPORTS ✓ | 0.00 | SUPPORTS ✓ | 0.10 |
| Is Honda accelerating its EV production? | REFUTES | SUPPORTS ✗ | 0.03 | SUPPORTS ✗ | 0.10 |
| Did Tesla acquire CATL? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |
| Is Toyota producing EVs in China? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.15 |
| Does Panasonic partner with CATL? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |
| Is Honda investing in factories in Japan? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.15 |
| Is CATL expanding in North America? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |
| Did Tesla partner with Panasonic? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |
| Is Toyota acquiring battery startups? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.15 |
| Does Honda invest in solid-state batteries? | SUPPORTS | SUPPORTS ✓ | 0.00 | SUPPORTS ✓ | 0.10 |
| Is Tesla expanding in China? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |
| Does Panasonic produce EVs? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |
| Did CATL delay any projects? | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO ✓ | 1.00 | SUPPORTS ✗ | 0.10 |

## Reading

**Accuracy** is the fraction of claims where the predicted verdict matched the gold verdict. Higher is better.

**θ on NEI claims** is the mean ignorance mass assigned to claims whose gold verdict is NOT_ENOUGH_INFO. **Higher is better** — a well-calibrated system should know when the corpus doesn't answer a question. A system that answers NEI claims with confident SUPPORTS/REFUTES has low θ and is hallucinating.
