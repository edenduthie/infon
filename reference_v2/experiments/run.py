"""Experiment runner entry point.

This is a skeleton stub committed in task A.1 (infon-6o3.1). The real
implementation lands in A.6b (infon-6o3.11) and is driven test-first by
A.6 (infon-6o3.10).

Subsequent test scaffolding can already import this module so that the
red phase of A.6 can fail with a clear ``NotImplementedError`` rather
than an ``ImportError``.
"""

from __future__ import annotations


def run(config_path: str, output_dir: str) -> None:
    """Run an experiment described by a YAML config.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file under
        ``reference_v2/experiments/configs/``.
    output_dir:
        Directory in which the JSON report(s) will be written.

    Raises
    ------
    NotImplementedError
        Always — implemented in A.6b.
    """

    raise NotImplementedError("Implemented in A.6b")
