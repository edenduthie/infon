# Tasks: Epic 03 (v4) — Evidence Generation + Baselines

> **Epic status:** not started. Blocked by Epics 01–02 (complete). Acceptance criterion in `proposal.md`. Unblocks Epic 04.

## Development Rules

- **Test-First:** write the integration test before the implementation; verify it fails (red), implement, verify it passes (green).
- **No Mocks:** all tests use real InfonStore, real SPLADE activations, real DS mass functions, real cassette ingests. LLM tests run in replay-only mode against committed cache fixtures — the cache is real data, not a mock.
- **No Dataset-Specific Branching in the Harness:** `benchmark_eval.py` iterates over `EvalClaim` and `EvalSystem`; all dataset-specific logic lives in the loader and the metadata dict.
- **Stage-Boundary Review:** at the end of each stage, run `pytest reference_v4/tests/ -v`, re-read this file and `spec.md`, verify compliance, update beads tasks.

---

## Stage A — Code

### Dataset loaders and types

- [ ] A.1 Create `reference_v4/benchmarks/__init__.py` and `reference_v4/benchmarks/types.py` with the `EvalClaim` dataclass and `MassFunction` namedtuple as specified in `spec.md`.
- [ ] A.2 Write `reference_v4/tests/test_loaders.py` first: parametric test over all three loaders (fixture cache); assert uniform `EvalClaim` structure; assert HoVer dev count == 4,000; assert AVeriTeC NEI present; assert SciFact rationales non-empty. **Verify red.**
- [ ] A.2b Implement `reference_v4/benchmarks/hover.py`, `averitec.py`, `scifact.py`. Download and cache datasets to `reference_v4/experiments/data/<dataset>/`; commit `data/SHA256SUMS`. **Verify green.**

### System interface and baselines

- [ ] A.3 Write `reference_v4/tests/test_system_interface.py` first: instantiate each system; call `evaluate()` on a shared fixture claim; assert `MassFunction` sums to 1 within `1e-6` for all systems. **Verify red.**
- [ ] A.3b Implement `reference_v4/baselines/types.py` (`EvalSystem` protocol, `MassFunction`). **Verify green.**
- [ ] A.4 Implement `reference_v4/baselines/symbolic_floor.py` (`SymbolicFloor`): SPLADE retrieval + rule-based polarity + abstain-on-weak-retrieval. Tests from A.3 cover this.
- [ ] A.5 Write `reference_v4/tests/test_cognition_system.py` first: `CognitionSystem("symbolic").evaluate(ev_fixture_claim)` → valid mass, `supports > 0`. **Verify red.** Then implement `reference_v4/baselines/cognition_system.py` wrapping `InfonStore` with the two-iteration schema bootstrap. **Verify green.**
- [ ] A.6 Implement `reference_v4/baselines/nli_classifier.py` (`NLIClassifier`). Tests from A.3 cover the interface; add one targeted test for the temperature-scaling path.
- [ ] A.7 Write `reference_v4/tests/test_llm_cache.py` first: replay hit → cached `response_text`; replay miss → `LLMCacheMissError` with 12-char key prefix in message; record mode appends new line to JSONL; token counter accumulates across calls. **Verify red.** Then implement `reference_v4/baselines/_llm_cache.py` (JSONL-backed, append-only, modes `record`/`replay_only`). Commit `reference_v4/tests/fixtures/llm_cache_fixture.jsonl` with 5 pre-recorded responses (SUPPORTS 0.82, REFUTES 0.71, NEI 0.95, malformed JSON, NEI with low-stated confidence 0.30). **Verify green.**
- [ ] A.7b Write `reference_v4/tests/test_verdict_to_mass.py` first: all four rows of the mapping table (SUPPORTS, REFUTES, NEI, parse-failure); edge cases confidence=0.0 and confidence=1.0; assert sum == 1 within 1e-6; assert NEI ignores stated confidence. **Verify red.** Implement `verdict_to_mass()` pure function in `reference_v4/baselines/llm_zeroshot.py`. **Verify green.**
- [ ] A.7c Write `reference_v4/tests/test_evidence_truncation.py` first: single long doc → ≤600 words + `[truncated]` marker; six 600-word docs → five docs + omission marker; empty docs → `[none provided]`. **Verify red.** Implement `LLMZeroShot._truncate_evidence()`. **Verify green.**
- [ ] A.7d Write `reference_v4/tests/test_llm_prompt.py` first: `_build_prompt(fixture_claim)` returns byte-identical output on two calls; system prompt contains the NEI confidence instruction; user prompt contains `[1]` prefix for the first evidence doc. **Verify red.** Implement `_build_prompt()` using `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` constants. **Verify green.**
- [ ] A.7e Write `reference_v4/tests/test_llm_zeroshot.py` covering the three scenarios in the spec: SUPPORTS fixture → correct mass; malformed response → theta=1.0; NEI with low confidence → theta=1.0 regardless. Run in `replay_only=True` mode against the committed fixture. Also add a test confirming `ANTHROPIC_API_KEY` absence raises `EnvironmentError` in `replay_only=False` mode. **Verify red → green.**
- [ ] A.7f Write `reference_v4/tests/test_cognition_no_llm.py`: patch `anthropic.Anthropic` at module level; call `CognitionSystem("gnn").evaluate(ev_fixture_claim)`; assert `anthropic.Anthropic` was never instantiated. **Verify red → green** (confirms Analyst is not invoked during evaluation).

### Evaluation harness

- [ ] A.8 Write `reference_v4/tests/test_benchmark_eval.py` first: (a) run a 3-cell matrix; kill after 2; resume; assert no recomputation. (b) run a matrix with `--max-input-tokens 100` and fixture cache entries totalling 80 tokens each; assert first cell completes, remaining cells recorded as `budget_exhausted`. **Verify red.**
- [ ] A.8b Implement `reference_v4/experiments/benchmark_eval.py`. Wire `LLMCache.total_input_tokens` into the budget guard; write `budget_exhausted` cell JSON on guard trigger. **Verify green.**

### Metrics and statistics

- [ ] A.9 Copy `reference_v2/experiments/metrics.py` and `stats.py` verbatim to `reference_v4/experiments/`. Copy associated tests. Run them; confirm green (no changes needed).
- [ ] A.10 Write `reference_v4/tests/test_panels.py` first: feed fixture mass arrays; assert H1, H2, H3 panel builders produce the documented JSON schemas. **Verify red.** Then implement `reference_v4/experiments/panels.py`. **Verify green.**

### Figure scripts

- [ ] A.11 Write `reference_v4/tests/test_figures.py` first: load fixture aggregate JSON; invoke `pareto.py`; assert output PDF exists and is > 1 KB; invoke twice; assert byte-identical. **Verify red.**
- [ ] A.11b Implement `reference_v4/experiments/figures/pareto.py`. Implement `accuracy_by_hop.py` (H1 panel figure), `theta_distribution.py` (H2 panel figure), `reliability_diagrams.py` (H3 panel). **Verify green.**

### Baseline reproduction check

- [ ] A.12 Implement `reference_v4/experiments/check_baseline_reproductions.py`: runs each baseline on its native benchmark (small dev slice), computes published-vs-ours delta, exits non-zero if delta > tolerance. Commit a fixture run that confirms the NLI baseline is within ±2 pp on SciFact.

- [ ] **STAGE-A REVIEW:** run `pytest reference_v4/tests/ -v`; no existing tests regressed; all new tests green; commit checkpoint `epic-03v4-stage-a-complete`.

---

## Stage B — Run

- [ ] B.1 Run `check_baseline_reproductions.py` for all three baselines on their native benchmarks. Record delta table in `experiments/results/baseline_reproduction_gate.json`. **Gate: all deltas within tolerance before proceeding.**
- [ ] B.2 Run `benchmark_eval.py` for the H1 evaluation: HoVer dev, all systems (symbolic floor, Cognition symbolic, Cognition+GNN, NLI), 3 seeds. Includes `CognitionSystem` in flat-retrieval mode (MCTS disabled). Commit per-cell JSONs.
- [ ] B.3 Run H2 evaluation: AVeriTeC v2 dev, all systems including LLM zero-shot (1 seed, temperature 0). Commit per-cell JSONs and LLM cache.
- [ ] B.4 Run H3 evaluation: SciFact test, all systems, 3 seeds. Commit per-cell JSONs.
- [ ] B.5 Run `panels.py` to generate `h1_panel.json`, `h2_panel.json`, `h3_panel.json`. Run `check_baseline_reproductions.py`; confirm it exits zero.
- [ ] B.6 Run all figure scripts; confirm all PDFs generate cleanly. Visual inspect each figure.
- [ ] B.7 SHA256-lock all result JSONs. Append hashes to `experiments/results/SHA256SUMS`. Commit.

---

## Stage C — Review and iterate

- [ ] C.1 **Baseline reproduction audit.** For any baseline outside tolerance, diagnose root cause (model checkpoint, tokenisation, evaluation split). Fix and re-run B.1; re-lock.
- [ ] C.2 **H1 audit.** Inspect the hop-stratified accuracy table. If Cognition+GNN is not monotonically improving with depth (or flat retrieval is not monotonically degrading), investigate: is schema coverage at fault? Is the MCTS depth bound too low? Iterate `CognitionSystem` parameters if needed; re-run B.2; re-lock.
- [ ] C.3 **H2 audit.** Inspect θ distributions on NEI vs SUPPORTS claims for each system. If Cognition's `theta_on_nei_mean ≤ LLMZeroShot.theta_on_nei_mean`, the H2 effect is not present — flag in the phase-3 memo's claim envelope as "H2 not supported at this scale" rather than re-engineering the system to force the result.
- [ ] C.4 **H3 audit.** Inspect ECE values. If Cognition's ECE > NLI baseline's ECE on SciFact, H3 is falsified on this dataset — report honestly.
- [ ] C.5 **Coverage failure audit.** Review the list of claims flagged as < 80% schema coverage. If coverage failures are concentrated in one dataset (>20%), investigate schema bootstrap for that dataset's evidence format; adjust the tokenisation or the spectral clustering threshold; re-run.
- [ ] C.6 **Null and negative finding documentation.** For any hypothesis not supported by the evidence, draft the exact text that will appear in the paper's Discussion. This text is the primary output of Stage C for falsified hypotheses — do not suppress negative results.
- [ ] **STAGE-C REVIEW:** all baselines within tolerance; all three panel JSONs reflect the C-stage findings; all figures regenerate deterministically; results SHA256-locked.

---

## Stage D — Finalize

- [ ] D.1 Re-lock all result JSONs and figures (SHA256SUMS updated to final state).
- [ ] D.2 Author `docs/publication/phase3_evidence_generation.md` (10 required sections per `spec.md`). Every number in the memo traces to a committed JSON. Link from `reference_v4/README.md`.
- [ ] D.3 Run `pytest reference_v4/tests/ -v` one final time; confirm all tests green.
- [ ] D.4 Tag commit `v4.1.0-evidence-complete` on branch `research`. Record the tag in the phase-3 memo.
- [ ] **PHASE-BOUNDARY REVIEW Phase 3:** all acceptance criteria from `proposal.md` met; phase-3 memo authored and linked; result JSONs SHA256-locked; tag pushed. Mark this epic complete.
