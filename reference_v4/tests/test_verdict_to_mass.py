"""Tests for verdict_to_mass pure function.

Run from repo root:
    pytest reference_v4/tests/test_verdict_to_mass.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_v4.benchmarks.types import MassFunction
from reference_v4.baselines.llm_zeroshot import verdict_to_mass


def test_supports():
    mass = verdict_to_mass("SUPPORTS", 0.82)
    assert mass.m_s == pytest.approx(0.82)
    assert abs(sum(mass) - 1.0) < 1e-6


def test_refutes():
    mass = verdict_to_mass("REFUTES", 0.71)
    assert mass.m_r == pytest.approx(0.71)
    assert abs(sum(mass) - 1.0) < 1e-6


def test_nei_ignores_confidence():
    mass = verdict_to_mass("NEI", 0.30)
    assert mass.m_theta == pytest.approx(1.0)
    assert mass.m_s == 0.0
    assert mass.m_r == 0.0


def test_nei_always_full_theta():
    """NEI must always return m_theta=1.0 regardless of confidence."""
    for conf in [0.0, 0.5, 1.0, 0.30, 0.95]:
        mass = verdict_to_mass("NEI", conf)
        assert mass.m_theta == pytest.approx(1.0), f"NEI with conf={conf} should give m_theta=1.0"


def test_parse_failure():
    mass = verdict_to_mass(None, None)
    assert mass.m_theta == pytest.approx(1.0)
    assert mass.m_s == 0.0
    assert mass.m_r == 0.0


def test_unknown_verdict():
    mass = verdict_to_mass("UNKNOWN_LABEL", 0.5)
    assert mass.m_theta == pytest.approx(1.0)


def test_edge_confidence_zero():
    mass = verdict_to_mass("SUPPORTS", 0.0)
    assert mass.m_theta == pytest.approx(1.0)
    assert mass.m_s == pytest.approx(0.0)


def test_edge_confidence_one():
    mass = verdict_to_mass("SUPPORTS", 1.0)
    assert mass.m_s == pytest.approx(1.0)
    assert mass.m_theta == pytest.approx(0.0)


def test_supports_theta_complement():
    """m_s + m_theta must equal 1.0 for SUPPORTS."""
    mass = verdict_to_mass("SUPPORTS", 0.65)
    assert mass.m_s == pytest.approx(0.65)
    assert mass.m_theta == pytest.approx(0.35)
    assert mass.m_r == 0.0
    assert mass.m_u == 0.0


def test_refutes_theta_complement():
    """m_r + m_theta must equal 1.0 for REFUTES."""
    mass = verdict_to_mass("REFUTES", 0.43)
    assert mass.m_r == pytest.approx(0.43)
    assert mass.m_theta == pytest.approx(0.57)
    assert mass.m_s == 0.0
    assert mass.m_u == 0.0


def test_returns_mass_function_type():
    mass = verdict_to_mass("SUPPORTS", 0.5)
    assert isinstance(mass, MassFunction)
