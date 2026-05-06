"""Tests for panels.py: each panel builder produces schema-matching output.

The panel schemas are defined by paper/tests/fixtures/*.json.
These tests verify:
  (a) All top-level keys are present
  (b) The "summary" dict has all expected scalar keys

Run from repo root:
    pytest reference_v4/tests/test_panels.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAPER_FIXTURES = REPO_ROOT / "paper" / "tests" / "fixtures"
V4_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers: build synthetic result files matching panel expectations
# ---------------------------------------------------------------------------

def _write_hover_results(results_dir: Path) -> None:
    """Write synthetic HoVer results for H1 panel testing."""
    systems = ["cognition_symbolic", "cognition_gnn", "flat_retrieval", "symbolic_floor"]
    # Create results for hops 2, 3, 4
    hop_claims = {
        2: [
            {"claim_id": f"h2_{i}", "ground_truth": "SUPPORTS",
             "mass": {"m_s": 0.72, "m_r": 0.08, "m_u": 0.1, "m_theta": 0.1},
             "predicted_label": "SUPPORTS", "metadata": {"num_hops": 2}}
            for i in range(5)
        ],
        3: [
            {"claim_id": f"h3_{i}", "ground_truth": "SUPPORTS",
             "mass": {"m_s": 0.65, "m_r": 0.1, "m_u": 0.15, "m_theta": 0.1},
             "predicted_label": "SUPPORTS", "metadata": {"num_hops": 3}}
            for i in range(5)
        ],
        4: [
            {"claim_id": f"h4_{i}", "ground_truth": "NEI",
             "mass": {"m_s": 0.0, "m_r": 0.0, "m_u": 0.0, "m_theta": 1.0},
             "predicted_label": "NEI", "metadata": {"num_hops": 4}}
            for i in range(5)
        ],
    }

    for system in systems:
        for seed in (0, 1, 2):
            all_results = []
            for hop_val, claims in hop_claims.items():
                for claim in claims:
                    all_results.append(claim)
            out = results_dir / "hover" / system / f"seed_{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "dataset": "hover",
                "system": system,
                "seed": seed,
                "results": all_results,
            }))


def _write_averitec_results(results_dir: Path) -> None:
    """Write synthetic AVeriTeC results for H2 panel testing."""
    systems = ["cognition_symbolic", "cognition_gnn", "llm_zeroshot"]
    # Mix of SUPPORTS, REFUTES, NEI
    results = [
        {"claim_id": "a0", "ground_truth": "SUPPORTS",
         "mass": {"m_s": 0.6, "m_r": 0.1, "m_u": 0.0, "m_theta": 0.3},
         "predicted_label": "SUPPORTS", "metadata": {}},
        {"claim_id": "a1", "ground_truth": "REFUTES",
         "mass": {"m_s": 0.1, "m_r": 0.7, "m_u": 0.0, "m_theta": 0.2},
         "predicted_label": "REFUTES", "metadata": {}},
        {"claim_id": "a2", "ground_truth": "NEI",
         "mass": {"m_s": 0.0, "m_r": 0.0, "m_u": 0.0, "m_theta": 1.0},
         "predicted_label": "NEI", "metadata": {}},
        {"claim_id": "a3", "ground_truth": "NEI",
         "mass": {"m_s": 0.6, "m_r": 0.1, "m_u": 0.0, "m_theta": 0.3},
         "predicted_label": "SUPPORTS", "metadata": {}},  # false positive for NEI
    ]
    for system in systems:
        for seed in (0, 1, 2):
            out = results_dir / "averitec" / system / f"seed_{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "dataset": "averitec",
                "system": system,
                "seed": seed,
                "results": results,
            }))


def _write_scifact_results(results_dir: Path) -> None:
    """Write synthetic SciFact results for H3 panel testing."""
    systems = ["cognition_symbolic", "cognition_gnn", "nli_classifier"]
    results = [
        {"claim_id": "s0", "ground_truth": "SUPPORTS",
         "mass": {"m_s": 0.8, "m_r": 0.1, "m_u": 0.05, "m_theta": 0.05},
         "predicted_label": "SUPPORTS", "metadata": {}},
        {"claim_id": "s1", "ground_truth": "REFUTES",
         "mass": {"m_s": 0.1, "m_r": 0.7, "m_u": 0.1, "m_theta": 0.1},
         "predicted_label": "REFUTES", "metadata": {}},
        {"claim_id": "s2", "ground_truth": "NEI",
         "mass": {"m_s": 0.0, "m_r": 0.0, "m_u": 0.0, "m_theta": 1.0},
         "predicted_label": "NEI", "metadata": {}},
        {"claim_id": "s3", "ground_truth": "SUPPORTS",
         "mass": {"m_s": 0.9, "m_r": 0.05, "m_u": 0.03, "m_theta": 0.02},
         "predicted_label": "SUPPORTS", "metadata": {}},
    ]
    for system in systems:
        for seed in (0, 1, 2):
            out = results_dir / "scifact" / system / f"seed_{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "dataset": "scifact",
                "system": system,
                "seed": seed,
                "results": results,
            }))


@pytest.fixture
def synthetic_results_dir(tmp_path):
    """Create synthetic results for all three datasets."""
    _write_hover_results(tmp_path)
    _write_averitec_results(tmp_path)
    _write_scifact_results(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# H1 panel tests
# ---------------------------------------------------------------------------

class TestH1Panel:
    def test_h1_panel_top_level_keys(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h1_panel
        panel = build_h1_panel(str(synthetic_results_dir))
        assert panel["hypothesis"] == "H1"
        assert isinstance(panel["systems"], list)
        assert isinstance(panel["depth_results"], list)
        assert isinstance(panel["summary"], dict)

    def test_h1_panel_summary_keys_match_fixture(self, synthetic_results_dir):
        """Summary must have same key pattern as paper fixture."""
        from reference_v4.experiments.panels import build_h1_panel

        with open(PAPER_FIXTURES / "h1_panel.json") as f:
            fixture = json.load(f)

        panel = build_h1_panel(str(synthetic_results_dir))

        # The summary key pattern: {system}_{N}hop
        fixture_systems = fixture["systems"]
        fixture_hops = [row["num_hops"] for row in fixture["depth_results"]]
        for system in fixture_systems:
            for hop in fixture_hops:
                key = f"{system}_{hop}hop"
                assert key in panel["summary"], (
                    f"H1 summary missing key: {key!r}"
                )

    def test_h1_depth_results_have_num_hops(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h1_panel
        panel = build_h1_panel(str(synthetic_results_dir))
        for row in panel["depth_results"]:
            assert "num_hops" in row

    def test_h1_summary_values_are_floats(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h1_panel
        panel = build_h1_panel(str(synthetic_results_dir))
        for k, v in panel["summary"].items():
            assert isinstance(v, (int, float)), f"Summary value for {k!r} must be numeric"


# ---------------------------------------------------------------------------
# H2 panel tests
# ---------------------------------------------------------------------------

class TestH2Panel:
    def test_h2_panel_top_level_keys(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h2_panel
        panel = build_h2_panel(str(synthetic_results_dir))
        assert panel["hypothesis"] == "H2"
        assert isinstance(panel["systems"], list)
        assert isinstance(panel["theta_by_class"], dict)
        assert isinstance(panel["summary"], dict)

    def test_h2_panel_summary_keys_match_fixture(self, synthetic_results_dir):
        """Summary must have same key pattern as paper fixture."""
        from reference_v4.experiments.panels import build_h2_panel

        with open(PAPER_FIXTURES / "h2_panel.json") as f:
            fixture = json.load(f)

        panel = build_h2_panel(str(synthetic_results_dir))

        fixture_systems = fixture["systems"]
        expected_suffixes = [
            "theta_supports", "theta_refutes", "theta_nei",
            "spearman_rho", "fpr_nei",
        ]
        for system in fixture_systems:
            for suffix in expected_suffixes:
                key = f"{system}_{suffix}"
                assert key in panel["summary"], (
                    f"H2 summary missing key: {key!r}"
                )

    def test_h2_theta_by_class_structure(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h2_panel
        panel = build_h2_panel(str(synthetic_results_dir))
        for system, class_data in panel["theta_by_class"].items():
            for cls in ("SUPPORTS", "REFUTES", "NEI"):
                assert cls in class_data, f"theta_by_class[{system!r}] missing class {cls!r}"
                assert isinstance(class_data[cls], list)

    def test_h2_fpr_nei_is_fraction(self, synthetic_results_dir):
        """False positive rate for NEI must be in [0, 1]."""
        from reference_v4.experiments.panels import build_h2_panel
        panel = build_h2_panel(str(synthetic_results_dir))
        for key, val in panel["summary"].items():
            if "fpr_nei" in key:
                assert 0.0 <= val <= 1.0, f"{key}={val} must be in [0,1]"


# ---------------------------------------------------------------------------
# H3 panel tests
# ---------------------------------------------------------------------------

class TestH3Panel:
    def test_h3_panel_top_level_keys(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h3_panel
        panel = build_h3_panel(str(synthetic_results_dir))
        assert panel["hypothesis"] == "H3"
        assert isinstance(panel["systems"], list)
        assert isinstance(panel["calibration"], dict)
        assert isinstance(panel["summary"], dict)

    def test_h3_panel_summary_keys_match_fixture(self, synthetic_results_dir):
        """Summary must have same key pattern as paper fixture."""
        from reference_v4.experiments.panels import build_h3_panel

        with open(PAPER_FIXTURES / "h3_panel.json") as f:
            fixture = json.load(f)

        panel = build_h3_panel(str(synthetic_results_dir))

        fixture_systems = fixture["systems"]
        for system in fixture_systems:
            for metric in ("ece", "brier", "aurc"):
                key = f"{system}_{metric}"
                assert key in panel["summary"], (
                    f"H3 summary missing key: {key!r}"
                )

    def test_h3_calibration_structure(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h3_panel
        panel = build_h3_panel(str(synthetic_results_dir))
        for system, cal in panel["calibration"].items():
            assert "bins" in cal
            assert "accuracies" in cal
            assert "counts" in cal
            assert len(cal["bins"]) == len(cal["accuracies"]) == len(cal["counts"])

    def test_h3_metric_values_in_range(self, synthetic_results_dir):
        from reference_v4.experiments.panels import build_h3_panel
        panel = build_h3_panel(str(synthetic_results_dir))
        for key, val in panel["summary"].items():
            assert isinstance(val, (int, float)), f"{key} must be numeric"
            assert val >= 0.0, f"{key}={val} must be non-negative"


# ---------------------------------------------------------------------------
# Aggregate panel tests
# ---------------------------------------------------------------------------

class TestAggregatePanel:
    def test_aggregate_panel_structure(self, synthetic_results_dir):
        from reference_v4.experiments.panels import (
            build_h1_panel, build_h2_panel, build_h3_panel, build_aggregate_panel
        )
        h1 = build_h1_panel(str(synthetic_results_dir))
        h2 = build_h2_panel(str(synthetic_results_dir))
        h3 = build_h3_panel(str(synthetic_results_dir))
        agg = build_aggregate_panel(h1, h2, h3)

        assert "systems" in agg
        assert "summary" in agg
        assert isinstance(agg["systems"], list)
        assert isinstance(agg["summary"], dict)

    def test_aggregate_panel_summary_keys_match_fixture(self, synthetic_results_dir):
        """Summary must contain all keys from the paper aggregate fixture."""
        from reference_v4.experiments.panels import (
            build_h1_panel, build_h2_panel, build_h3_panel, build_aggregate_panel
        )

        with open(PAPER_FIXTURES / "aggregate_panel.json") as f:
            fixture = json.load(f)

        h1 = build_h1_panel(str(synthetic_results_dir))
        h2 = build_h2_panel(str(synthetic_results_dir))
        h3 = build_h3_panel(str(synthetic_results_dir))
        agg = build_aggregate_panel(h1, h2, h3)

        for key in fixture["summary"]:
            assert key in agg["summary"], (
                f"Aggregate summary missing fixture key: {key!r}"
            )

    def test_aggregate_systems_list(self, synthetic_results_dir):
        from reference_v4.experiments.panels import (
            build_h1_panel, build_h2_panel, build_h3_panel, build_aggregate_panel
        )
        h1 = build_h1_panel(str(synthetic_results_dir))
        h2 = build_h2_panel(str(synthetic_results_dir))
        h3 = build_h3_panel(str(synthetic_results_dir))
        agg = build_aggregate_panel(h1, h2, h3)

        for entry in agg["systems"]:
            assert "name" in entry
            assert "accuracy" in entry
            assert "aurc" in entry
            assert "ece" in entry
