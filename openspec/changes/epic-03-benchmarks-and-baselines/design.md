# Design — Epic 03: Public benchmarks + baselines

## Context

Epic 02 generated synthetic evidence for H1 and H2 under controlled conditions. Without external validity, the paper is unpublishable. This epic adds three public benchmarks, each chosen to test a specific hypothesis on data the field already publishes against:

- **HoVer** is annotated with hop count (2-, 3-, 4-hop), making it the only public benchmark that supports the H1 stratification we did synthetically. Recent work (GraphCheck 2025; STRIVE 2025; AFEV 2025) reports +20 pp gains on HoVer specifically when graph structure is added to LLMs — exactly the niche our typed message passing is supposed to occupy.
- **AVeriTeC v2** has a documented 2025 leaderboard (winning AVeriTeC score = Ev2R recall = 33.17%, CTU AIC) and an explicit "not enough evidence" verdict class — the load-bearing testbed for H2's "knows what it doesn't know" claim.
- **SciFact** has thin per-claim evidence (often a single supporting abstract) and rationale annotations — the load-bearing testbed for ρ(m(Θ), thinness).

The six baselines span from "no learning" to "current SOTA on calibrated abstention". Each is the strongest published representative of its tier.

## Goals / Non-Goals

- **Goals**: deterministic public-benchmark loaders; six baselines whose numbers reproduce published values; full Pareto-front analysis of accuracy vs AURC; phase-3 memo with reliability diagrams and stratified tables.
- **Non-Goals**: training new models from scratch on the benchmarks (we evaluate; we do not fine-tune to win the leaderboard); FEVEROUS tabular path (stretch only); training new LLMs; building a new retrieval index (use the precompiled stores published with each benchmark).

## Decisions

### Decision: Three benchmarks, not all five

Use HoVer, AVeriTeC v2 dev, and SciFact. Make FEVEROUS a documented stretch goal; do not block the epic on it. Skip FEVER (subsumed by AVeriTeC for our purposes — same task, AVeriTeC has cleaner real-world claims and a current leaderboard).

**Why:** five benchmarks at six baselines is 30 cells per seed; three benchmarks at six baselines is 18 cells per seed. The H1/H2 hypotheses do not gain external validity from a fourth benchmark that does not stratify by hop count or thinness.

---

### Decision: Reproduce published numbers as a sanity gate

Before reporting any "ours vs baseline X" number, verify that baseline X reproduces *its own published number* on a benchmark to within ±2 pp. If it doesn't, either the loader is wrong or the baseline implementation is wrong; fix, don't paper over.

**Why:** the most common failure mode of "method beats baselines" papers is a quietly broken baseline. Reviewers know this and check. A reproduced-baseline gate up-front means our comparison is honest.

---

### Decision: LLM baselines run at temperature 0, single seed, with cached calls

The two LLM baselines (zero-shot ceiling; RAG + Sufficient-Context) are run at temperature 0, single deterministic seed (the API call ID + a timestamp recorded for audit), with all (prompt → completion) pairs cached on disk. Reproduction does not re-call the API.

**Why:** non-determinism across LLM API calls would invalidate cross-seed comparisons. Caching also caps cost: total LLM spend is bounded by the *first* run; subsequent re-runs are free. Document the budget cap in the phase-3 memo.

**Alternatives considered:**
- *Self-host an open-weights LLM (Llama 3.1 70B)*: matches the AVeriTeC 2025 single-GPU constraint and removes API spend. Tentatively a stretch goal — start with API for fast iteration, add open-weights run as a robustness check if budget allows.

---

### Decision: Architectural control = R-GCN + softmax + temperature scaling

The "architectural control" baseline shares everything with ours except: aggregator (uniform-mean instead of typed-IKL) and readout (softmax + temperature scaling instead of 4-mass DS + coherence regularizer). Same teacher, same encoder, same retrieval, same training schedule. Parameter count within 5%.

**Why:** this is the cleanest H1 + H2 control on real data. If our system loses to this control, *both* hypotheses lose simultaneously, and the paper's contribution collapses to "we used SPLADE and tree-sitter cleverly."

---

### Decision: Matched uncertainty baseline = evidential Dirichlet head

Replace only the readout with Dirichlet-EDL (Sensoy 2018). Keep typed-IKL aggregator and DS teacher. The teacher mass is mapped to a Dirichlet prior via the standard belief→Dirichlet bijection.

**Why:** this is the *strongest* H2 control. If Dirichlet matches DS-4-mass on AURC and ρ, then H2's contribution is "we re-derived EDL with Dempster's rule," which is true but not interesting on its own.

---

### Decision: One memo, not three

Write a single phase-3 memo covering all three benchmarks, not one per benchmark. The benchmarks are not independent stories — they jointly test H1 and H2, and forcing the reader to read three memos to see the joint picture defeats the point.

**Why:** the paper will have one Methods section and one Results section, not three. The memo should be shaped like the paper.

---

### Decision: Pareto front is the headline figure

The phase-3 memo and Epic 04's paper share one headline figure: a scatter of `mean accuracy` (x-axis) vs `AURC` (y-axis, lower is better) across all (dataset × model) cells, with the Pareto front drawn. Ours's position on the front (or off it) is the visual summary of the contribution.

**Why:** AURC is a single-number summary of the entire risk-coverage curve and the only way to make a six-model comparison legible in one figure.

---

### Decision: No fine-tuning, only evaluation (mostly)

For our system, "evaluation" means: load benchmark documents into the hypergraph, build the hypergraph, fit the network transductively as the system normally does, then evaluate. This is the system as designed; transductive training is part of the system, not pretraining for the benchmark.

For RoBERTa-NLI: use a publicly checkpointed model (e.g. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`); do not fine-tune on the benchmark train split.

For LLM baselines: zero-shot only.

**Why:** the question is whether the *architecture* generalises, not whether we can squeeze out a few extra points by fine-tuning. Fine-tuning would also blow up the experiment time and cost.

## Risks / Trade-offs

- **HoVer hop annotations are coarse**: hop count is annotated at the claim level, not per-step. Mitigation: report stratified numbers but flag the coarseness in the memo.
- **AVeriTeC's NEI label is not exactly "thinness"**: AVeriTeC's "not enough evidence" is annotator-judged sufficiency, not redundancy count. Mitigation: report ρ against multiple proxies (NEI label, retrieval-coverage, supporting-evidence count) and pre-register the primary one.
- **SciFact is small (~300 dev claims)**: bootstrap CIs will be wide. Mitigation: report CIs honestly; treat SciFact as a directional check, not a pivotal test.
- **LLM baselines change over time**: APIs deprecate models. Mitigation: pin model name and version (e.g. `claude-sonnet-4-6`, `gpt-4o-2024-08`); cache calls; document in memo.
- **Architectural control might be very weak**: untuned R-GCN + softmax may underperform on real benchmarks. Mitigation: also run a temperature-scaled variant; report the higher of the two as the strongest "structural-but-not-IKL" comparator. Don't strawman.
- **API spend**: LLM baselines on three benchmark dev sets ≈ low thousands of claims × low hundreds of tokens each. Estimate: USD 50–150 total spend. Document the actual spend in the memo.

## Open Questions

- Should we evaluate against the *blind test* sets (AVeriTeC) or only the dev sets? **Dev sets** — blind test requires submitting to a leaderboard; we will reference the leaderboard winner (33.17) but not contend.
- Should we report results on the AVeriTeC 2025 efficiency constraint (≤ 1 min/claim, ≤ 23 GB GPU)? **Optional**; report wall-clock and peak memory for our system, leave the comparison as a Discussion-section note.
- For the H1 stratified analysis on HoVer, do we re-train per stratum or evaluate one model on all strata? **One model** — that is the stratified-eval protocol HoVer expects.
