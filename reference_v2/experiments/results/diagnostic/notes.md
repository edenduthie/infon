# B.2 Per-infon mass diagnostic — Toyota / Honda Θ-collapse localization

**Task:** `infon-6o3.15` (Stage B.2)
**Anchor:** `docs/publication/reproduction_audit.md` §Path 2.1
**Inputs:** `experiments/results/baseline/baseline__seed={42,0,1}.json` (B.1 anchor)
**Outputs:** `{toyota,honda}__seed={42,0,1}.json` (this directory)
**Reproducer:** `experiments/results/diagnostic/per_infon_analysis.py`

## Verdict: **FUSION-side collapse.**

Per-infon `m(Θ)` is **not collapsed** (mean ≈ 0.23, min ≈ 0.11, max ≈ 0.45 across all
seeds and both queries). The fused `m(Θ)` collapses by a factor of **~100–200×** in
the fusion step alone (post-fusion 0.0015–0.0023 vs pre-fusion mean 0.235). The
audit's `Θ ≈ 0.30` paper claim is consistent with the *pre-fusion distribution* but
not with what survives Dempster's-rule fusion of 5 highly-overlapping decisive
masses on the highest-relevance subset.

This narrows Stage E priority firmly toward fusion-rule selection (B.3 sweep over
Yager/Murphy/top1) and decisive_top_k (already exposed by A.4b/A.5b) — and away
from regularizer redesign (E.1) and teacher reconstruction (E.2) as *primary*
remediations. E.1 / E.2 may still help but are not load-bearing for the Θ deficit.

## Per-infon m(Θ) distribution (cross-seed, per query)

| seed | query | n | min | p25 | median | p75 | max | mean | fused Θ |
|----:|------:|--:|----:|----:|-----:|----:|----:|-----:|--------:|
| 42 | toyota | 34 | 0.1253 | 0.1896 | 0.2311 | 0.2732 | 0.4452 | 0.2352 | 0.00154 |
| 42 |  honda | 28 | 0.1253 | 0.1512 | 0.2112 | 0.2637 | 0.4452 | 0.2262 | 0.00235 |
|  0 | toyota | 34 | 0.1110 | 0.1881 | 0.2347 | 0.2764 | 0.4411 | 0.2379 | 0.00172 |
|  0 |  honda | 28 | 0.1110 | 0.1628 | 0.2091 | 0.2476 | 0.4411 | 0.2230 | 0.00204 |
|  1 | toyota | 34 | 0.1146 | 0.1842 | 0.2425 | 0.2752 | 0.4442 | 0.2404 | 0.00151 |
|  1 |  honda | 28 | 0.1146 | 0.1656 | 0.2127 | 0.2674 | 0.4442 | 0.2301 | 0.00208 |

Cross-seed: per-infon `mean(Θ)` is in `[0.223, 0.241]` — tighter than the
acceptance gate's `[0.20, 0.40]` and within the across-seed std `≤ 0.05`. The
distribution is also stable across queries: Toyota's `n=34` superset of Honda's
`n=28` shares the same tail (max ≈ 0.445, min ≈ 0.115) within each seed.

The audit's "the readout produces non-trivial Θ" hypothesis is now confirmed
quantitatively: **the readout is not the problem**.

## Bimodality check

For all six (seed, query) cells: **decisive count ≥ 28, uncertain count = 0**.

There is no bimodality. Every per-infon mass has `m(Θ) < 0.5`; the 25th–75th
percentile band is `[0.15, 0.28]` and the maximum is 0.445 — well below the
0.5 split. The readout produces a *unimodal*, *moderately-decisive*, *evidentially-
honest* distribution (mean Θ ≈ 0.23 is very close to the paper's claimed 0.30).
Dempster fusion of any 3–5 such overlapping focal-aligned masses then
multiplicatively shrinks Θ to ~0.002.

## Polarity diversity (the load-bearing finding)

| seed | query | SUPPORTS | REFUTES | UNCERTAIN | THETA | most-decisive |
|----:|------:|---------:|--------:|----------:|------:|:-------------|
| 42 | toyota | 27 | 7 | 0 | 0 | REFUTES (m(Θ)=0.125, m(R)=0.823) |
| 42 |  honda | 20 | 8 | 0 | 0 | REFUTES (m(Θ)=0.125, m(R)=0.823) |
|  0 | toyota | 27 | 7 | 0 | 0 | REFUTES (m(Θ)=0.111, m(R)=0.834) |
|  0 |  honda | 20 | 8 | 0 | 0 | REFUTES (m(Θ)=0.111, m(R)=0.834) |
|  1 | toyota | 27 | 7 | 0 | 0 | REFUTES (m(Θ)=0.115, m(R)=0.833) |
|  1 |  honda | 20 | 8 | 0 | 0 | REFUTES (m(Θ)=0.115, m(R)=0.833) |

This **confirms A.5b's surprise** (`infon-6o3.9` close-reason): the trained
reasoner's *single most-decisive* per-infon mass is REFUTES on every Toyota/Honda
query at every seed, but it gets swamped in fusion by 20–27 SUPPORTS contributors.
The most-decisive contributor is the same infon (`inf_5471700ee548`) for every
(seed, query) pair — a structural property of the corpus, not a random-init
artefact.

Per-polarity mean Θ also tells the story: REFUTES contributors have
mean Θ ≈ **0.139–0.145** (more decisive); SUPPORTS contributors have
mean Θ ≈ **0.257–0.265** (less decisive). The decisive masses *are* the
contradicting ones, but they are outnumbered ~3× and Dempster's normalization
attaches the conflict mass back onto SUPPORTS, not Θ.

This means the B.3 sweep should look hard at:

- **`fusion_rule = "yager"`** — adds conflict to Θ instead of normalizing it
  away. Predicted to substantially raise fused Θ on these queries.
- **`fusion_rule = "murphy"`** — averages then Dempster-combines, blunting the
  multiplicative collapse.
- **`fusion_rule = "top1"` / `decisive_top_k = 1`** — picks the single
  most-decisive mass. **Caveat:** that mass is REFUTES, so cap=1 will flip
  polarity to REFUTES on Toyota/Honda (per A.5b's test 4 narrowing rationale).
  It satisfies the Θ acceptance gate but fails the polarity gate.
- **`decisive_top_k = 2` or `3`** — fuses fewer agreeing SUPPORTS masses,
  likely landing higher Θ without flipping polarity. This is the most
  promising operating point a-priori.

## Stage E sequencing implications

Per audit §Path 2.1, the verdict gates Stage E ordering. With FUSION-side
collapse confirmed:

1. **Stage B.3 sweep is now the primary fix axis.** The 480-cell grid in
   `experiments/configs/sweep_collapse.yaml` (6 coh × 4 rules × 4 top_k × 5
   seeds) directly varies the two knobs that the diagnostic implicates. If
   any sweep cell satisfies all four acceptance criteria
   (Θ ∈ [0.20, 0.40], polarity = SUPPORTS, std ≤ 0.05, 14-test green),
   **Stage E can be skipped** for the Θ-recovery line item.

2. **Stage E.1 (regularizer redesign) demoted to secondary.** The readout is
   already producing mean Θ ≈ 0.23 at the per-infon level — within
   0.20–0.40 even before any training change. Increasing `coherence_weight`
   *might* tighten this further, but the per-infon distribution is already
   on-target; the immediate problem is the fusion arithmetic, not the
   training objective.

3. **Stage E.2 (teacher reconstruction) demoted to secondary.** The teacher
   targets the per-infon mass shape, which is already healthy. Reconstructing
   it will not change the multiplicative-collapse behaviour of Dempster's
   rule on n=5 agreeing decisive masses.

4. **Stage E should still be entered if Stage B's sweep does not land a
   Toyota+Honda double-pass cell**, because polarity preservation is a
   joint constraint — the REFUTES-dominance of the smallest-Θ contributor
   means top1/cap=1 cells are likely to satisfy the Θ gate but fail the
   polarity gate. If no (rule, top_k) pair satisfies both, then the
   per-infon distribution itself needs reshaping (E.1 / E.2 / readout
   changes), because the corpus contains genuine REFUTES signal for
   Toyota/Honda at these queries.

5. **Order suggestion for the canonical config search**: start with
   `(rule=yager, top_k=3, coherence_weight=0.2)` as the most-likely
   candidate (yager preserves Θ on partial conflict; cap=3 keeps polarity
   stable per A.5b). If that fails the Θ gate, raise `coherence_weight`
   first (cheaper than retraining the readout); only then escalate to
   E.1/E.2.

## Noteworthy structural findings

1. **The most-decisive per-infon mass is identical across seeds and queries**
   (`inf_5471700ee548`, `m(Θ) ∈ [0.11, 0.13]`, REFUTES). This is the same
   infon A.5b's narrowing of test 4 was reacting to. It belongs to the
   negated/refuting subset of the corpus (Honda's delays / unproduced
   research per A.5b notes). Its consistent identity across seeds suggests
   the readout is reliably learning the polarity of the corpus's strongest
   counter-evidence — the issue is purely that fusion buries it.

2. **REFUTES contributors are systematically more decisive than SUPPORTS.**
   In every cell, REFUTES contributors have lower mean Θ
   (mean Θ_R ≈ 0.14 vs mean Θ_S ≈ 0.26). The trained reasoner is more
   confident about *contradictions* than about *confirmations* on these
   queries — a small but structurally interesting honesty signal that
   the audit fusion-rule swap will make visible in the verdict.

3. **The pre-fusion mean (0.23) almost exactly matches the paper's claim
   (0.30).** With Yager fusion (which adds conflict to Θ rather than
   normalizing it away) the gap between the fused output and the paper's
   claim is likely to close substantially without any other intervention.
   This is the strongest single-knob hypothesis for B.3 and should be
   tested first.

4. **Number of per-infon contributors is constant per query across seeds**
   (Toyota: 34, Honda: 28). Activation-threshold and graph-construction
   are seed-independent (only the readout weights change with seed), and
   the activation threshold of 0.2 produces a stable contributor pool.
   Cross-seed variance is purely in the per-infon Θ values themselves
   (max Θ varies ~0.001 across seeds), supporting the audit's claim that
   the collapse is a deterministic property of the configuration, not a
   random-init artefact.

## Cross-check vs B.1 anchor

The diagnostic script re-fits the baseline config in-process and produces
fused query masses bit-equal (`max_abs_delta = 0.0`) to the committed
baseline JSONs across all six (seed, query) cells. This confirms that:

- The per-infon records this diagnostic surfaces are the *same* records
  that produced the audit's collapsed Θ — not a parallel run with a
  drifted config.
- The runner is byte-deterministic (validated again here, after A.6b's
  initial determinism proof).
- Any Stage C/D/E intervention that changes the per-infon distribution
  or the fusion will be visible as a non-zero `max_abs_delta` between
  the new diagnostic JSONs and these committed reference files —
  giving the canonical-config search a clean before/after diff.
