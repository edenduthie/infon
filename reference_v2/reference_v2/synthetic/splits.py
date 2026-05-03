"""Train/dev/test split generation with thinness-curriculum stratification.

The test split is stratified over ``evidence_redundancy`` values so that
every curriculum stratum (1, 2, 5, 10) is represented at least
``max(1, int(test_size / 400 * 100))`` times.  This mirrors the
proportional scaling used in the full 400-scenario benchmark.
"""

from __future__ import annotations

import numpy as np

# Curriculum strata for the thinness axis (evidence_redundancy).
_CURRICULUM_STRATA: tuple[int, ...] = (1, 2, 5, 10)

# Scaling constants: a full test set (400 scenarios) requires 100 per stratum.
_FULL_TEST_SIZE: int = 400
_MIN_PER_STRATUM_FULL: int = 100


def _min_per_stratum(test_size: int) -> int:
    """Return minimum required scenarios per stratum for *test_size* scenarios.

    Scales proportionally from the full benchmark (400 → 100 per stratum).
    Always at least 1.
    """
    return max(1, int(test_size / _FULL_TEST_SIZE * _MIN_PER_STRATUM_FULL))


def _make_scenario(scenario_id: str, evidence_redundancy: int) -> dict:
    """Return a minimal scenario dict with required keys."""
    return {
        "scenario_id": scenario_id,
        "evidence_redundancy": evidence_redundancy,
    }


def make_splits(
    seed: int,
    train: int,
    dev: int,
    test: int,
) -> dict[str, list[dict]]:
    """Generate a stratified train/dev/test split of synthetic scenarios.

    Parameters
    ----------
    seed:
        Integer seed for ``numpy.random.default_rng``.  Identical seeds
        produce identical outputs (fully deterministic).
    train:
        Number of scenarios in the training split.
    dev:
        Number of scenarios in the development split.
    test:
        Number of scenarios in the test split.

    Returns
    -------
    dict with keys ``"train"``, ``"dev"``, ``"test"``, each mapping to a
    list of scenario dicts.  Every scenario dict has at minimum:

    * ``scenario_id`` — unique string identifier
    * ``evidence_redundancy`` — int in ``{1, 2, 5, 10}``
    """
    rng = np.random.default_rng(seed)

    total = train + dev + test
    n_strata = len(_CURRICULUM_STRATA)

    # ── Build the full scenario pool ─────────────────────────────────────────
    # Assign evidence_redundancy by cycling through strata so the pool is
    # naturally balanced, then shuffle for randomness.
    strata_cycle = [_CURRICULUM_STRATA[i % n_strata] for i in range(total)]
    scenario_pool = [
        _make_scenario(f"scenario_{i:05d}", strata_cycle[i])
        for i in range(total)
    ]
    rng.shuffle(scenario_pool)  # type: ignore[arg-type]

    # ── Build the stratified test split ──────────────────────────────────────
    min_count = _min_per_stratum(test)

    # Bucket the pool by stratum.
    stratum_buckets: dict[int, list[dict]] = {s: [] for s in _CURRICULUM_STRATA}
    leftover: list[dict] = []

    for scenario in scenario_pool:
        er = scenario["evidence_redundancy"]
        if er in stratum_buckets and len(stratum_buckets[er]) < min_count:
            stratum_buckets[er].append(scenario)
        else:
            leftover.append(scenario)

    # Take min_count per stratum as the guaranteed test sample.
    guaranteed_test: list[dict] = []
    for s in _CURRICULUM_STRATA:
        guaranteed_test.extend(stratum_buckets[s])

    # Fill the remaining test slots from the leftover pool.
    remaining_test_slots = test - len(guaranteed_test)
    extra_test = list(rng.choice(len(leftover), size=remaining_test_slots, replace=False))  # type: ignore[arg-type]
    extra_test_scenarios = [leftover[i] for i in sorted(extra_test)]

    # Remove extra_test scenarios from leftover (preserve order).
    extra_set = set(id(s) for s in extra_test_scenarios)
    leftover_after_test = [s for s in leftover if id(s) not in extra_set]

    # ── Build train / dev from leftover ──────────────────────────────────────
    rng.shuffle(leftover_after_test)  # type: ignore[arg-type]

    train_scenarios = leftover_after_test[:train]
    dev_scenarios = leftover_after_test[train : train + dev]

    return {
        "train": train_scenarios,
        "dev": dev_scenarios,
        "test": guaranteed_test + extra_test_scenarios,
    }
