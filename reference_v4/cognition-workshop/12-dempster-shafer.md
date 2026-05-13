# Module 12: Dempster-Shafer Belief Calculus and Contrary View


> **Update for the cassette substrate.** The Dempster–Shafer belief calculus below is unchanged and load-bearing in the cassette reasoner. `reason()`, `reason_connectivity()`, and the MCTS backprop all compose per-infon masses via Dempster's rule. One notable change: for *chains* (not single claims) we use min/max conjunction instead of Dempster across edges — a chain is a conjunction, and Dempster's additive combine across hops wrongly amplifies S. See module 13.

## What You'll Learn

- How Dempster-Shafer theory assigns belief masses over {SUPPORTS, REFUTES, UNCERTAIN} instead of single-point probabilities
- How multiple independent evidence sources combine via Dempster's rule of combination
- Why mass on the full frame (theta/ignorance) captures "I genuinely don't know" — something probability can't express
- How the pignistic probability transform converts belief masses to actionable decisions
- How NLI classification heads provide calibrated mass functions for evidence evaluation
- How `MassFunction.invert()` enables contrary-view analysis — same evidence, opposite thesis

## Background

Traditional fact-checking returns a binary or ternary verdict: true, false, or unknown. But real evidence is messy — one document supports a claim, another refutes it, and a third is irrelevant. How do you combine conflicting evidence into a principled verdict?

Dempster-Shafer theory solves this. Instead of a probability distribution over outcomes, it assigns **mass** to subsets of the frame of discernment. For fact verification:

- **Frame**: Θ = {SUPPORTS, REFUTES}
- **Mass on {SUPPORTS}**: evidence that directly supports the claim
- **Mass on {REFUTES}**: evidence that directly refutes the claim
- **Mass on {UNCERTAIN}**: evidence that is ambiguous or irrelevant
- **Mass on Θ (theta)**: total ignorance — we genuinely don't have evidence

The key insight: `m(Θ) = 0.8` means "I'm 80% ignorant" — not "I'm 80% uncertain between support and refute." This distinction is critical for calibrated verdicts.

## Prompt 1: Mass Functions and Combination

---

```
Build Dempster-Shafer belief calculus in cognition/src/cognition/dempster_shafer.py.

MassFunction dataclass:
  - supports: float   # m({SUPPORTS})
  - refutes: float    # m({REFUTES})
  - uncertain: float  # m({UNCERTAIN}) — ambiguous evidence
  - theta: float      # m(Θ) — total ignorance (no evidence)
  - Must sum to 1.0 (normalize in __post_init__)
  - belief(h), plausibility(h), entropy(), verdict() methods

Dempster's Rule — combine_dempster(m1, m2) → MassFunction:
  - For each pair (A, B) where A∩B ≠ ∅: mass(A∩B) += m1(A) * m2(B)
  - Conflict K = sum of m1(A)*m2(B) where A∩B = ∅
  - Normalize by 1/(1-K) (redistribute conflict)
  - Handle edge case: K ≈ 1.0 → return ignorance mass

Design the intersection rules:
  - {S} ∩ {S} = {S}     (agreement strengthens)
  - {R} ∩ {R} = {R}     (agreement strengthens)
  - {S} ∩ {R} = ∅       (conflict → normalized away)
  - {S} ∩ Θ = {S}       (evidence + ignorance = evidence)
  - {R} ∩ Θ = {R}       (evidence + ignorance = evidence)
  - {U} ∩ anything = {U} or target (U is dominated)
  - Θ ∩ Θ = Θ           (ignorance + ignorance = ignorance)

Test:
- Two supporting sources → S > 0.8
- Support + refute → conflict resolved, moderate belief
- All ignorance → theta stays high
- Three sources combined → entropy decreases
```

---

### Why Not Just Use Probabilities?

Consider a claim with no evidence at all. A probabilistic system must assign P(true) + P(false) = 1 — forcing a stance from nothing. With DS theory, you assign m(Θ) = 1.0: "I have no evidence." As evidence arrives, mass moves from Θ to specific hypotheses. The verdict only firms up when enough mass concentrates on SUPPORTS or REFUTES.

This is exactly what you want for fact verification: the system should say "not enough evidence" when there genuinely isn't enough, rather than guessing.

## Prompt 2: Evidence Sources and Claim Verification

---

```
Add four evidence sources that produce mass functions from different signals:

1. NLI Source (primary):
   - Takes claim + evidence text → runs through NLI classification head
   - Head outputs [entail, neutral, contradict] probabilities
   - Calibration: theta = max(0.0, 0.5 * (1.0 - max_prob))
   - Scale remaining mass by (1.0 - theta)
   - Maps: entail → supports, neutral → uncertain, contradict → refutes

2. Lexical Overlap Source:
   - Jaccard similarity between claim tokens and evidence tokens
   - High overlap → low theta (relevant evidence)
   - Very low overlap → high theta (irrelevant)

3. Temporal Recency Source:
   - Recent evidence gets lower theta (more trustworthy)
   - Old evidence gets higher theta (may be outdated)
   - Controlled by half-life parameter

4. Source Agreement Source:
   - If multiple infons from different documents agree → lower theta
   - Single source → moderate theta
   - Conflicting sources → mass split between S and R

Build verify_claim(claim, evidences, encoder, heads) → MassFunction:
  - For each evidence: compute all 4 source masses
  - Combine sources per-evidence via Dempster's rule
  - Combine across evidences via Dempster's rule
  - Return final combined mass with verdict

Test with automotive corpus:
- "Toyota invested in batteries" (clear support) → SUPPORTS
- "Toyota has no EV plans" (contradicted by evidence) → REFUTES  
- "Toyota will merge with Honda" (no evidence) → NOT ENOUGH INFO
```

---

### The Calibration Problem

A raw softmax output [0.45, 0.35, 0.20] sums to 1.0 — there's no room for "I'm not confident in these numbers." The calibration step carves out theta (ignorance) proportional to how uncertain the head is. When max_prob is 0.95, theta ≈ 0.025 (confident). When max_prob is 0.4, theta ≈ 0.30 (uncertain). This prevents the DS system from treating a confused classifier as confident evidence.

## Prompt 3: Pignistic Transform and Decision Making

---

```
Add pignistic probability transform to MassFunction:

pignistic() → dict with keys "supports", "refutes", "uncertain":
  - BetP(A) = sum over B⊇A: m(B) / |B| * 1/(1-m(∅))
  - For our frame: distributes theta equally among focal elements
  - Result is a proper probability distribution (sums to 1.0)

verdict_with_confidence() → (verdict: str, confidence: float):
  - Apply pignistic transform
  - Verdict is argmax of BetP
  - Confidence is max(BetP) - second_max(BetP)
  - Thresholds:
    * confidence < 0.1 → "NOT ENOUGH INFO"
    * Otherwise report verdict with confidence score

Add batch_verify(claims, corpus, encoder, heads) for benchmarking:
  - Process multiple claims efficiently
  - Return list of (verdict, MassFunction, confidence) tuples
  - Report timing and accuracy if gold labels provided

Test the full pipeline end-to-end with 10 automotive claims,
comparing DS verdict vs. simple majority voting.
```

---

### Why Pignistic?

The raw mass function gives rich information but isn't directly comparable to probability. When you need to make a decision (SUPPORTS vs REFUTES vs NOT_ENOUGH_INFO), the pignistic transform converts mass to probability by distributing ignorance equally. A mass function with m(S)=0.4, m(Θ)=0.6 becomes BetP(S)=0.4+0.3=0.7, BetP(R)=0.3 — the ignorance splits evenly, but the evidence tips the balance.

## Prompt 4: Integration with Cognition Pipeline

---

```
Integrate Dempster-Shafer into the main cognition query flow:

1. In cognition/src/cognition/query.py, add verify mode:
   - query(claim, mode="verify") → uses DS pipeline
   - Retrieves relevant infons via standard anchor projection
   - For each infon: generates mass from all 4 sources
   - Combines and returns verdict + mass + evidence chain

2. Add DS results to the query response:
   - QueryResult gets new field: belief_mass: Optional[MassFunction]
   - Includes breakdown: which evidence contributed what mass
   - Includes conflict level (K from Dempster combination)

3. Test with the automotive schema:
   - Ingest 15 documents about Toyota/Tesla/Ford battery investments
   - Verify claims that require combining multiple pieces of evidence
   - Show that DS gives calibrated verdicts where simple retrieval fails:
     * "Toyota invested in batteries" → high support, low theta
     * "Toyota abandoned battery research" → high refute (contradicted)
     * "Toyota and BMW partnered on batteries" → NOT ENOUGH INFO (no evidence)

Verify the mass always sums to 1.0 and that conflict (K) is reported.
```

---

## Contrary View: Inverting the Evidential Lens

Once you have Dempster-Shafer verdicts, a natural question follows: "What does the evidence look like from the opposite perspective?" An analyst who finds `SUPPORTS` for "China is escalating" should immediately ask "What if I'm wrong?" — not by finding new data, but by re-examining the same evidence through the contrary frame.

`MassFunction.invert()` swaps SUPPORTS and REFUTES:

```python
m = MassFunction(supports=0.7, refutes=0.1, uncertain=0.1, theta=0.1)
m_contrary = m.invert()
# → MassFunction(supports=0.1, refutes=0.7, uncertain=0.1, theta=0.1)
```

This is a query-time operation — nothing stored changes. The same infons, the same evidence chain, but the lens flips. Evidence that confirmed the original claim now refutes the contrary, and vice versa. Combined with the `contrary=True` parameter on `verify_claim` and `query`, this enables:

- **Red-team analysis**: automatically surface counter-evidence for any claim
- **Devil's advocate**: what's the strongest case against my thesis?
- **Bias detection**: if normal and contrary verdicts are both weak, the evidence is genuinely ambiguous

## Prompt 5: Contrary View via Mass Inversion

---

```
Add contrary view support to the Dempster-Shafer module:

1. MassFunction.invert() → MassFunction:
   - Swap supports ↔ refutes
   - Keep uncertain and theta unchanged
   - Returns a new MassFunction (immutable)

2. Extend verify_claim with a contrary parameter:
   verify_claim(infons, claim_anchors, schema_types, contrary=False)
   - When contrary=True, call .invert() on each per-infon mass
     BEFORE combining via Dempster's rule
   - The final combined mass reflects the contrary perspective
   - Verdict labels flip accordingly: what was SUPPORTS becomes REFUTES

Test with opposing evidence:
- Claim: "Chinese Coast Guard patrols near the Senkaku Islands are escalatory"
- Normal verify_claim → SUPPORTS (escalatory evidence dominates)
- Contrary verify_claim → REFUTES or SUPPORTS (de-escalation evidence
  now dominates — patrols described as "routine law enforcement")
- Show both verdicts side by side with their belief distributions
- Verify: normal.supports ≈ contrary.refutes (they should be close)
```

---

## Prompt 6: Contrary Query Ranking

---

```
The contrary view also applies at the query level. In Module 07 we
built the query engine with a contrary parameter. Test it here with
the DS integration:

1. Query: "Has China escalated military operations in the East China Sea?"

2. Normal ranking (contrary=False):
   - Affirmed infons (polarity=1) about escalation rank high
   - Valence is positive for escalatory evidence

3. Contrary ranking (contrary=True):
   - Negated infons (polarity=0) get a +0.25 rank boost
   - Valence flips: what was positive is now negative
   - De-escalation and restraint evidence floats to the top

Show side-by-side for the top 5 results:
  - Which infons appear in each ranking
  - How valence signs flip
  - How negated-polarity infons (polarity=0) rise in contrary mode

Then run verify_claim on both result sets:
  - Normal infons → DS verdict
  - Contrary infons → DS verdict with contrary=True
  - The two verdicts should be opposite or near-opposite

This completes the red-team loop: query contrary → verify contrary →
surface the strongest counter-narrative from existing evidence.
```

---

## The Mathematical View

Dempster-Shafer theory is a generalization of Bayesian probability:

| Bayesian | Dempster-Shafer |
|----------|----------------|
| P(A) single probability | Bel(A) ≤ P(A) ≤ Pl(A) — interval |
| Prior required | m(Θ) = 1.0 is valid (no prior needed) |
| Single update rule (Bayes) | Dempster combination (multiple sources) |
| Forces a stance | Allows "I don't know" |
| One source at a time | Combines independent sources directly |

The belief function Bel(A) is the total mass on subsets of A (definitely supports A). The plausibility Pl(A) is 1 - Bel(¬A) (doesn't rule out A). The gap Pl(A) - Bel(A) measures ignorance about A specifically.

## Calibration: why θ on NEI *must* stay high

Dempster's rule has a subtle failure mode that bit us in production.
Applied to many weakly-relevant masses, the combined mass concentrates
fast — even when no single infon deserved confidence.

On the 20-claim LLM-comparison gold set, the baseline showed:

```
mean θ on NEI claims: 0.00   ← system confidently wrong on unsupported claims
accuracy:             45%
```

Individual infons had reasonable θ. But `reason()` was pulling every
infon with any anchor overlap and running Dempster's rule over the
whole pool. With 10+ weakly-relevant infons combined, θ collapsed
toward zero.

### The fix: a strict relevance filter *before* Dempster's rule

Inside `HypergraphReasoner.reason()`, each candidate infon is scored by
role-wise overlap with the query and filtered before combine:

```python
subj_s = query_activations.get(infon.subject, 0.0)
pred_s = query_activations.get(infon.predicate, 0.0)
obj_s  = query_activations.get(infon.object, 0.0)

min_role = min(subj_s, pred_s, obj_s)
if min_role <= 0.05:
    continue                    # any role scoring near zero → drop

rel = (subj_s * pred_s * obj_s) ** (1/3)   # geometric mean
```

On true NEI queries, no infon passes the filter. Dempster's rule is
never invoked; `reason()` returns the uninformative prior
`(0, 0, 0, 1)` and `.verdict = NOT_ENOUGH_INFO`.

Result on the same gold set:

```
mean θ on NEI claims:    1.00   ← honest about not knowing
accuracy:                85%
mean θ on SUPPORTS hits: ~0.25  ← still commits when it should
```

The lesson: **Dempster's rule is not a replacement for relevance
filtering.** It's a combiner for evidence you already believe applies.
Feed it junk and it concentrates mass *on* the junk.

`tests/test_theta_calibration.py` is the shipped regression gate — if
NEI mean θ drops below 0.5 or accuracy below 0.60, the calibration is
broken.

## Checkpoint

- [ ] MassFunction normalizes to 1.0 and handles edge cases
- [ ] Dempster combination works for 2, 3, N sources
- [ ] Conflict K is computed and reported
- [ ] High-conflict combination doesn't crash (K ≈ 1.0 case)
- [ ] NLI head provides calibrated mass (theta reflects uncertainty)
- [ ] verify_claim combines 4 sources × N evidences correctly
- [ ] Pignistic transform produces valid probability distribution
- [ ] DS gives "NOT ENOUGH INFO" when evidence is absent (not a forced guess)
- [ ] Batch verification works for benchmarking against FEVER/HoVer
- [ ] `MassFunction.invert()` swaps supports ↔ refutes, keeps uncertain/theta
- [ ] `verify_claim(contrary=True)` produces opposite verdict from normal
- [ ] Normal `belief_supports` ≈ contrary `belief_refutes`
- [ ] Contrary query ranking promotes negated-polarity infons (polarity=0)
- [ ] Valence signs flip in contrary mode
- [ ] **Relevance filter blocks Dempster from seeing off-topic infons** — core of the θ-calibration fix
- [ ] `test_nei_claims_have_high_theta` gate: mean θ on NEI > 0.5
- [ ] `test_supported_claims_have_low_theta` gate: mean θ on correct-SUPPORTS < 0.3
- [ ] Accuracy on the 20-claim LLM-comparison set ≥ 0.60
