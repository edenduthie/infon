# Tasks: Epic 01 — Stabilize Θ

> **Epic status (created 2026-05-02):** not started. Acceptance criterion in `proposal.md`. Blocks Epics 02–04.

## Development Rules (applied to every task)

- **Test-First:** write the integration test before the implementation; verify it fails (red), implement, verify it passes (green).
- **No Mocks:** all tests exercise the real reasoner, real masses, real torch tensors, real seeded RNGs. No `MagicMock`, no `patch`.
- **Stage-Boundary Review:** at the end of each stage (A/B/C/D below), re-read this `tasks.md` and the `spec.md`, run the full `pytest -v` regression, confirm spec compliance, and update the openspec plan and beads tasks.

---

## Stage A — Code (write the supporting code for the study)

- [ ] A.1 Create `reference_v2/experiments/__init__.py`, `reference_v2/experiments/configs/`, `reference_v2/experiments/run.py`. Add module to `reference_v2/pyproject.toml` if needed.
- [ ] A.2 Write `reference_v2/tests/test_seed_pinning.py` first: build a `HypergraphReasoner`, fit with `seed=42`, snapshot the loss-by-epoch trace and the final mass for one query; rebuild and re-fit with `seed=42`; assert traces and masses are bit-identical. **Verify red.**
- [ ] A.2b Add `seed: int | None = None` parameter to `HypergraphReasoner.fit()` in `reference_v2/src/cognition/logic.py`; pin `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.use_deterministic_algorithms(True)` when set; document determinism caveats. **Verify green.**
- [ ] A.3 Write `reference_v2/tests/test_per_infon_mass_logger.py` first: fit a small reasoner; call `reason()` with `log_per_infon_masses=True`; assert the returned record contains a list of pre-fusion masses with shape and DS-axiom checks (each mass sums to 1, all entries non-negative). **Verify red.**
- [ ] A.3b Add `log_per_infon_masses: bool` to `Config`; wire through `HypergraphReasoner.reason()` so the diagnostic record is populated when set. **Verify green.**
- [ ] A.4 Write `reference_v2/tests/test_fusion_rules.py` first: construct three high-confidence agreeing focal masses; call each of `combine_dempster`, `combine_yager`, `combine_murphy`, `combine_top1` on them; assert that Yager and `top1` preserve more `m(Θ)` than Dempster; assert all return valid mass functions. **Verify red.**
- [ ] A.4b Implement `combine_yager`, `combine_murphy`, `combine_top1` in `reference_v2/src/cognition/dempster_shafer.py`; expose a `combine_multiple(masses, rule="dempster"|"yager"|"murphy"|"top1")` dispatcher preserving the existing default. **Verify green.**
- [ ] A.5 Write `reference_v2/tests/test_decisive_top_k.py` first: fit a small reasoner; call `reason()` with `decisive_top_k=1` and `decisive_top_k=5`; assert that the result for `decisive_top_k=1` has strictly larger `m(Θ)` than for `decisive_top_k=5` on a query with 5 supporting infons. **Verify red.**
- [ ] A.5b Add `decisive_top_k: int = 3` to `Config` (default reduced from 5); use it in `HypergraphReasoner.reason()` to cap the number of fused per-infon masses. **Verify green.**
- [ ] A.6 Write `reference_v2/tests/test_experiment_runner.py` first: load `experiments/configs/baseline.yaml`; call `experiments.run.run(config_path, output_dir)`; assert a JSON report appears at `output_dir/<config_name>__seed=<seed>.json` with the documented schema. **Verify red.**
- [ ] A.6b Implement `experiments/run.py`: load YAML, build `Config`, run `fit()` → `reason()` for each diagnostic query, dump JSON. Implement `experiments/sweep.py` that iterates over a `sweep:` block in the YAML and invokes `run.run` per cell. **Verify green.**
- [ ] A.7 Author `experiments/configs/baseline.yaml` (matches current default behaviour) and `experiments/configs/sweep_collapse.yaml` (sweeps `coherence_weight × fusion_rule × decisive_top_k × seed`).
- [ ] **STAGE-A REVIEW:** run `pytest reference_v2/tests/ -v`; ensure no existing tests regressed; ensure all new tests pass; commit checkpoint `epic-01-stage-a-complete`.

## Stage B — Run (execute experiments and gather results)

- [ ] B.1 Run `experiments/run.py --config baseline.yaml --seed 42` to confirm the reproducer matches `paper_scenario_report.json` numbers (sanity check that we have not silently changed defaults).
- [ ] B.2 Run the per-infon mass diagnostic on the Toyota and Honda queries; record the pre-fusion masses to `experiments/results/diagnostic/`. Determine: is `m(Θ)` already collapsed pre-fusion (training problem) or only post-fusion (fusion problem)?
- [ ] B.3 Run `experiments/sweep.py --config sweep_collapse.yaml`. Expected ~96 cells (6 coherence × 4 fusion × 4 top-k) × 5 seeds = 480 runs; ~30 s/run = ~4 hours wall-clock on a single CPU. Run in background; aggregate results into `experiments/results/sweep_collapse/aggregate.json`.
- [ ] B.4 Plot a sweep summary figure (`experiments/results/sweep_collapse/sweep_summary.png`) of `m(Θ)` vs `coherence_weight` faceted by `fusion_rule`, with `decisive_top_k` as line style; mark the polarity-correct cells.

## Stage C — Review and iterate on the unexpected

- [ ] C.1 Identify all configurations that satisfy acceptance criterion 1–3 from `proposal.md`. If zero satisfy, file a `decision-record.md` recommending H2 retraction; close epic with that recommendation; STOP.
- [ ] C.2 If multiple satisfy, rank by: (i) polarity accuracy, (ii) `m(Θ)` stability across seeds, (iii) loss convergence speed, (iv) simplicity (prefer `dempster` + low `decisive_top_k` over `yager` + high coherence weight if both work).
- [ ] C.3 For the top candidate, run on the 14-test regression suite (`pytest reference_v2/tests/test_logic.py -v`); fix any failures. If a fix requires reverting a knob, re-rank.
- [ ] C.4 Capture three plots for the memo: per-infon mass distribution before/after fusion; `m(Θ)` vs coherence weight under chosen fusion rule; loss curves under each fusion rule.
- [ ] C.5 If unexpected behaviour appears (e.g. Yager produces high `m(Θ)` purely from conflict), file follow-up beads tickets — do not paper over in the canonical config.

## Stage D — Finalize and produce phase-1 deliverable

- [ ] D.1 Commit `experiments/configs/canonical_v0_2.yaml` with the chosen settings. Lock the file's mtime in CI.
- [ ] D.2 Re-run the canonical config across all 5 seeds; commit the resulting JSON reports to `experiments/results/canonical_v0_2/`.
- [ ] D.3 Write `docs/publication/phase1_collapse_fix.md`: 1) what the audit found; 2) where the collapse was located; 3) the sweep grid and surface plot; 4) the canonical configuration; 5) how to reproduce; 6) acceptance criterion table filled in; 7) follow-ups for Epic 2 (e.g. activation-threshold full ablation).
- [ ] D.4 Update `reference_v2/README.md` (or create one) with a `## Reproducing the Paper` section referencing `canonical_v0_2.yaml`.
- [ ] D.5 Tag the commit `v0.2.0-phase1`; push.
- [ ] **PHASE-BOUNDARY REVIEW Phase 1:** run `pytest reference_v2/tests/ -v` (all green); re-read `proposal.md`, `design.md`, `spec.md`; confirm acceptance criterion satisfied; confirm `phase1_collapse_fix.md` written; update beads tasks; mark `epic-01-stabilize-theta` as completed in beads; unblock `epic-02-synthetic-stress-and-ablations`.
