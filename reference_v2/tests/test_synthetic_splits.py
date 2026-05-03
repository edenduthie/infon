"""Red-phase TDD tests for make_splits in reference_v2.synthetic.splits.

This file is intentionally written BEFORE the implementation exists.
All tests below MUST fail today with ImportError; the failure mode is
spelled out in each test's docstring so the green-phase task knows what
to make pass.

Spec:
- Epic 02 — Synthetic stress dataset + full ablation matrix
- Task A.2 (red) → green phase lands reference_v2/synthetic/splits.py

Contract for make_splits(seed, train, dev, test):
  - Returns a dict with keys {"train", "dev", "test"}, each mapping to a
    list of scenario dicts.
  - Each scenario dict has at minimum the key "evidence_redundancy" (int).
  - Scenario IDs are unique across the three splits (disjoint).
  - The "test" split is a thinness-curriculum: at least 2 scenarios per
    stratum in evidence_redundancy ∈ {1, 2, 5, 10}, scaled proportionally
    for small test sizes (floor(test / 50) * 1, minimum 1).
  - Split sizes exactly match the requested train/dev/test counts.

Phase A.2 (RED): expected failure mode is
    ImportError: No module named 'reference_v2'
    (or: cannot import name 'make_splits' from 'reference_v2.synthetic.splits')
because reference_v2/synthetic/splits.py does not exist yet.
"""

from __future__ import annotations

# NOTE: This import is the load-bearing red-phase failure.
# Once reference_v2/synthetic/splits.py is created with make_splits,
# all tests below become the green-phase contract.
from reference_v2.synthetic.splits import make_splits  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────

# Small sizes for fast testing; total = 100 scenarios.
_TRAIN = 80
_DEV = 10
_TEST = 10
_SEED = 42

# The four evidence_redundancy strata that define the thinness curriculum.
_CURRICULUM_STRATA = {1, 2, 5, 10}

# Minimum scenarios per stratum for a test split.
# Full-size test (>=400) needs >=100 per stratum; scale proportionally
# with floor(test_size / 400) * 100, minimum 1.
_MIN_PER_STRATUM_FULL = 100
_FULL_TEST_SIZE = 400


def _min_per_stratum(test_size: int) -> int:
    """Return the minimum required scenarios per stratum for a given test size.

    Scales proportionally: full size (400) requires 100 per stratum.
    For any smaller size we require at least 1, floor-scaled from the ratio.
    """
    ratio = test_size / _FULL_TEST_SIZE
    return max(1, int(ratio * _MIN_PER_STRATUM_FULL))


# ──────────────────────────────────────────────────────────────────────────
# Test 1 — split sizes match requested counts
# ──────────────────────────────────────────────────────────────────────────


def test_split_sizes_match_requested():
    """make_splits returns exactly train/dev/test scenarios in each split.

    Phase A.2 (RED): ImportError today.

    Phase green: splits dict must have exactly the requested counts.
    """
    splits = make_splits(seed=_SEED, train=_TRAIN, dev=_DEV, test=_TEST)

    assert "train" in splits, "splits dict missing 'train' key"
    assert "dev" in splits, "splits dict missing 'dev' key"
    assert "test" in splits, "splits dict missing 'test' key"

    assert len(splits["train"]) == _TRAIN, (
        f"train split has {len(splits['train'])} scenarios; expected {_TRAIN}"
    )
    assert len(splits["dev"]) == _DEV, (
        f"dev split has {len(splits['dev'])} scenarios; expected {_DEV}"
    )
    assert len(splits["test"]) == _TEST, (
        f"test split has {len(splits['test'])} scenarios; expected {_TEST}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 2 — train/dev/test scenario IDs are disjoint
# ──────────────────────────────────────────────────────────────────────────


def test_splits_are_disjoint():
    """No scenario ID appears in more than one split.

    Phase A.2 (RED): ImportError today.

    Phase green: each scenario must carry a unique "scenario_id" and no ID
    may appear in more than one of {train, dev, test}.
    """
    splits = make_splits(seed=_SEED, train=_TRAIN, dev=_DEV, test=_TEST)

    def extract_ids(scenarios: list[dict]) -> set:
        ids = set()
        for s in scenarios:
            assert "scenario_id" in s, (
                f"scenario dict missing 'scenario_id' key: {s!r}"
            )
            ids.add(s["scenario_id"])
        return ids

    train_ids = extract_ids(splits["train"])
    dev_ids = extract_ids(splits["dev"])
    test_ids = extract_ids(splits["test"])

    train_dev = train_ids & dev_ids
    assert not train_dev, (
        f"train and dev share {len(train_dev)} scenario IDs: "
        f"{sorted(train_dev)[:10]}"
    )

    train_test = train_ids & test_ids
    assert not train_test, (
        f"train and test share {len(train_test)} scenario IDs: "
        f"{sorted(train_test)[:10]}"
    )

    dev_test = dev_ids & test_ids
    assert not dev_test, (
        f"dev and test share {len(dev_test)} scenario IDs: "
        f"{sorted(dev_test)[:10]}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 3 — test split has thinness-curriculum stratification
# ──────────────────────────────────────────────────────────────────────────


def test_test_split_has_curriculum_stratification():
    """The test split must contain at least min_per_stratum scenarios for
    each evidence_redundancy value in {1, 2, 5, 10}.

    This ensures H2 correlation results generalise across the thinness
    curriculum: full-size test (>=400) requires >=100 per stratum; smaller
    test sizes scale proportionally, minimum 1.

    Phase A.2 (RED): ImportError today.

    Phase green: make_splits must deliberately stratify the test split so
    that every stratum of the thinness curriculum is represented.
    """
    splits = make_splits(seed=_SEED, train=_TRAIN, dev=_DEV, test=_TEST)

    test_scenarios = splits["test"]
    min_count = _min_per_stratum(_TEST)

    # Build a count per stratum.
    stratum_counts: dict[int, int] = {s: 0 for s in _CURRICULUM_STRATA}
    for scenario in test_scenarios:
        assert "evidence_redundancy" in scenario, (
            f"scenario dict missing 'evidence_redundancy' key: {scenario!r}"
        )
        er = scenario["evidence_redundancy"]
        if er in stratum_counts:
            stratum_counts[er] += 1

    for stratum, count in stratum_counts.items():
        assert count >= min_count, (
            f"test split has only {count} scenarios with evidence_redundancy="
            f"{stratum}; expected at least {min_count} "
            f"(scaled from {_MIN_PER_STRATUM_FULL} for full size {_FULL_TEST_SIZE}, "
            f"test_size={_TEST})"
        )


# ──────────────────────────────────────────────────────────────────────────
# Test 4 — all four curriculum strata are covered in test split
# ──────────────────────────────────────────────────────────────────────────


def test_test_split_covers_all_curriculum_strata():
    """All four evidence_redundancy strata {1, 2, 5, 10} appear in the test split.

    Phase A.2 (RED): ImportError today.

    Phase green: even for very small test sizes, the curriculum strata must
    all be represented (at least 1 scenario each).
    """
    splits = make_splits(seed=_SEED, train=_TRAIN, dev=_DEV, test=_TEST)

    test_scenarios = splits["test"]
    observed_strata = {s["evidence_redundancy"] for s in test_scenarios}

    missing = _CURRICULUM_STRATA - observed_strata
    assert not missing, (
        f"test split is missing curriculum strata: {sorted(missing)}. "
        f"Observed evidence_redundancy values: {sorted(observed_strata)}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 5 — splits are reproducible with same seed
# ──────────────────────────────────────────────────────────────────────────


def test_splits_are_reproducible_with_same_seed():
    """Calling make_splits with the same seed twice produces identical splits.

    Phase A.2 (RED): ImportError today.

    Phase green: the seed parameter must fully determine the output.
    """
    splits_a = make_splits(seed=_SEED, train=_TRAIN, dev=_DEV, test=_TEST)
    splits_b = make_splits(seed=_SEED, train=_TRAIN, dev=_DEV, test=_TEST)

    for split_name in ("train", "dev", "test"):
        ids_a = [s["scenario_id"] for s in splits_a[split_name]]
        ids_b = [s["scenario_id"] for s in splits_b[split_name]]
        assert ids_a == ids_b, (
            f"{split_name} split differs between two calls with seed={_SEED}: "
            f"first={ids_a[:5]}..., second={ids_b[:5]}..."
        )


if __name__ == "__main__":
    test_split_sizes_match_requested()
    test_splits_are_disjoint()
    test_test_split_has_curriculum_stratification()
    test_test_split_covers_all_curriculum_strata()
    test_splits_are_reproducible_with_same_seed()
    print("PASS: all splits tests passed")
