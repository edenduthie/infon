# Module 10: Evaluation and Improvement


> **Update for the cassette substrate.** The evaluation framework below is still valid for held-out infon scoring. In practice the shipping product surfaces corpus-quality issues via `extraction_report()` (automatic after every ingest), so most of the manual evaluation work happens before a sweep rather than as a sweep. See module 14 for the shipped `bootstrap_gnn` + `extraction_report` pair.

## What You'll Learn

- How to measure projection quality — how much information survives the change of basis
- Identifying schema gaps: concept dimensions the current basis is missing
- Strategies for improving anchor coverage and accuracy
- When to expand the basis vs. adjust thresholds

## Prompt 1: Audit Extraction Quality

---

```
Run extraction on 20 diverse documents from my corpus and help me audit:

1. Precision check: for 50 random infons, show me the sentence and triple. 
   I'll mark each as correct/incorrect. Calculate precision.

2. Recall check: for 10 sentences I choose, manually identify what triples 
   should be extracted. Compare against what the system actually extracted. 
   Calculate recall.

3. Support type analysis:
   - What fraction of infons are direct vs semantic vs hierarchical?
   - For semantic infons (no lexical match), are they mostly correct?
   - Are hierarchical infons appropriate?

4. Error patterns:
   - Are certain anchor types consistently wrong?
   - Are there sentences that produce too many/too few infons?
   - Does the activation threshold need adjustment?

Show results as a structured report with recommendations.
```

---

## Prompt 2: Schema Gap Analysis

---

```
Analyze the extracted infons and identify schema gaps:

1. Frequent unmatched patterns: sentences where the model activates 
   anchors strongly but no good triple forms (missing anchor type)

2. Token coverage: for each anchor, what fraction of its actual 
   occurrences in text are captured by its token list?

3. Hierarchy gaps: are there obvious parent-child relationships 
   missing from the schema?

4. Missing relations: are there actions/relationships that appear 
   in text but have no matching relation anchor?

5. Overloaded anchors: are any anchors activated by too many 
   unrelated concepts (need to be split)?

Generate a list of recommended schema additions and modifications.
```

---

## Prompt 3: Iterative Improvement Plan

---

```
Based on the audit and gap analysis, create an improvement plan:

1. Quick wins (no retraining needed):
   - Token list expansions
   - Threshold adjustments
   - New anchors that can be matched by keywords

2. Schema changes (requires retraining):
   - New anchors to add
   - Anchors to split or merge
   - Hierarchy restructuring

3. Training data improvements:
   - Sentences to add to training set
   - Label corrections needed
   - Hard negatives to include

4. Architecture considerations:
   - Is 160 anchors enough for this domain?
   - Should activation_threshold be per-anchor-type?
   - Would a larger model (ModernBERT-large) help?

Prioritize by impact and effort. Which changes give the most 
improvement for the least work?
```

---

## Measuring Projection Quality

### Triple-Level Precision
Not "did the change of basis activate the right anchors" but "did we extract the right triples." A sentence might correctly project to toyota and invest coordinates but incorrectly pair them.

### Constraint Quality
Are the top constraints actually true? A constraint with high evidence but low precision is worse than one with moderate evidence and high precision.

### Temporal Coherence
Do NEXT chains make narrative sense? If Toyota's chain goes "invest → divest → invest" in the same month, something is off.

### Query Satisfaction
Give the system 20 questions you'd want answered. Rate the results 1-5. This is the ultimate test — does it actually help you find knowledge?

## Checkpoint

- [ ] Manual precision audit on 50+ infons
- [ ] Recall audit on 10+ manually annotated sentences
- [ ] Schema gap analysis with concrete recommendations
- [ ] Prioritized improvement plan
- [ ] At least one iteration of improvement applied and measured

---

## Reasoning-level metrics

Extraction metrics tell you *what went into the graph*. Reasoning metrics
tell you *whether the verdicts are honest*. The current system ships
with three gold-set metrics, exercised in `tests/test_theta_calibration.py`
and `tests/test_sheaf.py`:

### 1. θ on NEI claims (the calibration metric)

On claims the corpus doesn't support, the mean θ should be **above 0.5**
— ideally near 1.0. Before the relevance-filter fix, the baseline
systematically overcommitted with mean θ ≈ 0.00 on NEI.

```python
nei = [r for r in cog_results.results if r.gold == "NOT_ENOUGH_INFO"]
mean_theta = sum(r.theta for r in nei) / len(nei)
assert mean_theta > 0.5   # shipped: ≈ 1.00 on the LLM-comparison gold set
```

### 2. θ on confidently-SUPPORTS claims

On claims the corpus *does* support and the system predicts correctly,
θ should be **below 0.3** — the system can commit.

### 3. End-to-end accuracy

On the 20-claim LLM comparison gold set, overall accuracy ≥ 0.60
(shipped: 85%). This guards against the "perfect calibration,
useless accuracy" degenerate fix — we want honesty *and* commitment.

### Sheaf vs R-GCN comparison

The sheaf message-passing layer (see Module 11) is a drop-in for
R-GCN. `tests/test_sheaf.py` runs a head-to-head benchmark: both
variants are fit on the same graph with the same teacher, then asked
the same four queries. Current result on the 4-doc EV benchmark:

| Metric | R-GCN | Sheaf |
|---|---|---|
| Accuracy (4 queries) | 2/4 | 2/4 |
| Mean θ on SUPPORTS | 0.05 | 0.19 |
| Mean θ on NEI | 1.00 | 1.00 |

On this tiny corpus there's no accuracy lift — expected, since near-
identity restriction maps don't yet have asymmetry to exploit. The
calibration invariant (NEI > SUPPORTS) holds for both.

The sheaf layer is expected to pay off on larger, messier graphs — see
§Part 5 of `08_category_theory.ipynb`.

### Full test gate

```bash
pytest tests/ -q
# → 170 passed in ~2.5 min
```

Any production deployment should block on this gate. Use the LLM
comparison tests as canaries — if θ-on-NEI drops below 0.5 or accuracy
below 60%, the calibration is broken.
