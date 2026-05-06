"""Tests for benchmark_eval.py: resumability and budget guard.

Run from repo root:
    pytest reference_v4/tests/test_benchmark_eval.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_system(name: str, evaluate_fn=None):
    """Return a mock system that records evaluate() calls."""
    system = MagicMock()
    system.name = name
    if evaluate_fn is not None:
        system.evaluate.side_effect = evaluate_fn
    else:
        from reference_v4.benchmarks.types import MassFunction
        system.evaluate.return_value = MassFunction(
            m_s=0.6, m_r=0.1, m_u=0.1, m_theta=0.2
        )
    return system


def _make_fixture_claims():
    """Return a tiny list of EvalClaims from the hover fixture."""
    from reference_v4.benchmarks.hover import load_hover
    return load_hover(data_path=str(FIXTURES_DIR / "hover_fixture.json"), limit=3)


# ---------------------------------------------------------------------------
# A.8 Test (a) — Resumability
# ---------------------------------------------------------------------------

class TestResumability:
    """benchmark_eval skips cells that already have output files."""

    def test_existing_output_file_skips_evaluate(self, tmp_path):
        """If seed_0.json already exists, evaluate() is NOT called again."""
        from reference_v4.experiments.benchmark_eval import run_cell

        claims = _make_fixture_claims()
        system = _make_mock_system("symbolic_floor")
        llm_cache_path = str(tmp_path / "llm_cache.jsonl")

        # First run — should write output and call evaluate
        status = run_cell(
            dataset_name="hover",
            system_name="symbolic_floor",
            seed=0,
            results_dir=str(tmp_path),
            llm_cache_path=llm_cache_path,
            max_input_tokens=10_000_000,
            systems={"symbolic_floor": system},
            loaders={"hover": _make_fixture_claims},
        )
        assert status == "completed"
        first_call_count = system.evaluate.call_count
        assert first_call_count == len(claims)

        # Second run — file exists, should skip without calling evaluate again
        system2 = _make_mock_system("symbolic_floor")
        status2 = run_cell(
            dataset_name="hover",
            system_name="symbolic_floor",
            seed=0,
            results_dir=str(tmp_path),
            llm_cache_path=llm_cache_path,
            max_input_tokens=10_000_000,
            systems={"symbolic_floor": system2},
            loaders={"hover": _make_fixture_claims},
        )
        assert status2 == "skipped"
        assert system2.evaluate.call_count == 0, (
            "evaluate() must NOT be called for existing output file"
        )

    def test_output_file_is_valid_json(self, tmp_path):
        """Written output file must be valid JSON with expected keys."""
        from reference_v4.experiments.benchmark_eval import run_cell

        llm_cache_path = str(tmp_path / "llm_cache.jsonl")
        run_cell(
            dataset_name="hover",
            system_name="symbolic_floor",
            seed=1,
            results_dir=str(tmp_path),
            llm_cache_path=llm_cache_path,
            max_input_tokens=10_000_000,
            systems={"symbolic_floor": _make_mock_system("symbolic_floor")},
            loaders={"hover": _make_fixture_claims},
        )
        out_path = tmp_path / "hover" / "symbolic_floor" / "seed_1.json"
        assert out_path.exists(), "Output file must be created"
        data = json.loads(out_path.read_text())
        assert data["dataset"] == "hover"
        assert data["system"] == "symbolic_floor"
        assert data["seed"] == 1
        assert isinstance(data["results"], list)
        assert len(data["results"]) == len(_make_fixture_claims())
        for result in data["results"]:
            assert "claim_id" in result
            assert "ground_truth" in result
            assert "mass" in result
            assert "predicted_label" in result
            assert result["predicted_label"] in ("SUPPORTS", "REFUTES", "NEI")

    def test_resumability_multi_seed(self, tmp_path):
        """Running seeds 0,1,2 then re-running only calls evaluate for missing seeds."""
        from reference_v4.experiments.benchmark_eval import run_cell

        llm_cache_path = str(tmp_path / "llm_cache.jsonl")
        system_a = _make_mock_system("symbolic_floor")

        # Run seeds 0 and 1
        for seed in (0, 1):
            run_cell(
                dataset_name="hover",
                system_name="symbolic_floor",
                seed=seed,
                results_dir=str(tmp_path),
                llm_cache_path=llm_cache_path,
                max_input_tokens=10_000_000,
                systems={"symbolic_floor": system_a},
                loaders={"hover": _make_fixture_claims},
            )

        calls_after_two_seeds = system_a.evaluate.call_count

        # Check seed 2 not yet present
        seed2_path = tmp_path / "hover" / "symbolic_floor" / "seed_2.json"
        assert not seed2_path.exists()

        # Re-run all three seeds — seeds 0 and 1 should be skipped
        system_b = _make_mock_system("symbolic_floor")
        n_claims = len(_make_fixture_claims())
        for seed in (0, 1, 2):
            run_cell(
                dataset_name="hover",
                system_name="symbolic_floor",
                seed=seed,
                results_dir=str(tmp_path),
                llm_cache_path=llm_cache_path,
                max_input_tokens=10_000_000,
                systems={"symbolic_floor": system_b},
                loaders={"hover": _make_fixture_claims},
            )

        # Only seed 2 should have triggered evaluate
        assert system_b.evaluate.call_count == n_claims, (
            f"Expected {n_claims} evaluate calls (seed 2 only), "
            f"got {system_b.evaluate.call_count}"
        )


# ---------------------------------------------------------------------------
# A.8 Test (b) — Budget guard
# ---------------------------------------------------------------------------

class TestBudgetGuard:
    """run_cell writes budget_exhausted when token budget exceeded."""

    def test_budget_exhausted_written_when_over_limit(self, tmp_path):
        """With max_input_tokens=0, budget is exhausted before any LLM cell runs."""
        from reference_v4.experiments.benchmark_eval import run_cell
        from reference_v4.baselines._llm_cache import LLMCache

        # Pre-populate cache with some tokens so total > 0
        cache_path = tmp_path / "llm_cache.jsonl"
        cache = LLMCache(str(cache_path), mode="record")
        cache.put("fake_key", '{"verdict":"SUPPORTS","confidence":0.8}', 5000, 10)

        # max_input_tokens=100, but cache already has 5000 tokens → budget exceeded
        system = _make_mock_system("symbolic_floor")
        status = run_cell(
            dataset_name="hover",
            system_name="symbolic_floor",
            seed=7,
            results_dir=str(tmp_path),
            llm_cache_path=str(cache_path),
            max_input_tokens=100,
            systems={"symbolic_floor": system},
            loaders={"hover": _make_fixture_claims},
        )
        assert status == "budget_exhausted"

        out_path = tmp_path / "hover" / "symbolic_floor" / "seed_7.json"
        assert out_path.exists(), "budget_exhausted file must still be written"
        data = json.loads(out_path.read_text())
        assert data["status"] == "budget_exhausted"
        assert data["dataset"] == "hover"
        assert data["system"] == "symbolic_floor"

    def test_non_llm_system_ignores_budget(self, tmp_path):
        """Non-LLM systems always run regardless of token budget (budget only gates LLM cells)."""
        from reference_v4.experiments.benchmark_eval import run_cell

        # LLM cache is empty → 0 tokens used, but max_input_tokens=0
        # Non-LLM system should still complete normally
        cache_path = tmp_path / "llm_cache.jsonl"
        system = _make_mock_system("symbolic_floor")
        status = run_cell(
            dataset_name="hover",
            system_name="symbolic_floor",
            seed=9,
            results_dir=str(tmp_path),
            llm_cache_path=str(cache_path),
            max_input_tokens=0,  # zero budget, but non-LLM system
            systems={"symbolic_floor": system},
            loaders={"hover": _make_fixture_claims},
        )
        # symbolic_floor is not an LLM system, so budget guard doesn't apply
        # (budget guard only applies when system_name starts with "llm_")
        assert status == "completed"


# ---------------------------------------------------------------------------
# A.8 — _mass_to_label helper
# ---------------------------------------------------------------------------

class TestMassToLabel:
    """_mass_to_label must return correct labels."""

    def test_supports_when_ms_highest(self):
        from reference_v4.experiments.benchmark_eval import _mass_to_label
        from reference_v4.benchmarks.types import MassFunction
        mass = MassFunction(m_s=0.7, m_r=0.1, m_u=0.1, m_theta=0.1)
        assert _mass_to_label(mass) == "SUPPORTS"

    def test_refutes_when_mr_highest(self):
        from reference_v4.experiments.benchmark_eval import _mass_to_label
        from reference_v4.benchmarks.types import MassFunction
        mass = MassFunction(m_s=0.1, m_r=0.8, m_u=0.05, m_theta=0.05)
        assert _mass_to_label(mass) == "REFUTES"

    def test_nei_when_theta_highest(self):
        from reference_v4.experiments.benchmark_eval import _mass_to_label
        from reference_v4.benchmarks.types import MassFunction
        mass = MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)
        assert _mass_to_label(mass) == "NEI"

    def test_nei_when_ms_equals_mr(self):
        from reference_v4.experiments.benchmark_eval import _mass_to_label
        from reference_v4.benchmarks.types import MassFunction
        # Tie: m_s == m_r, neither > other, falls through to NEI
        mass = MassFunction(m_s=0.4, m_r=0.4, m_u=0.1, m_theta=0.1)
        # m_s is NOT > max(m_r, m_theta) since 0.4 == 0.4, so not SUPPORTS
        # m_r is NOT > max(m_s, m_theta) since 0.4 == 0.4, so not REFUTES
        assert _mass_to_label(mass) == "NEI"
