#!/usr/bin/env python3
"""
Checkpoint-resumable evaluation harness for the Infon benchmark matrix.

Usage:
    python3 experiments/benchmark_eval.py \
        --datasets hover averitec scifact \
        --systems infon_symbolic infon_gnn flat_retrieval symbolic_floor nli_classifier llm_zeroshot \
        --seeds 0 1 2 \
        --results-dir experiments/results \
        --llm-cache experiments/results/llm_cache.jsonl \
        --max-input-tokens 4000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.types import MassFunction



def _mass_to_label(mass: MassFunction) -> str:
    """Convert a MassFunction to a predicted label string.

    Rules:
      - "SUPPORTS" if m_s  > max(m_r, m_theta)
      - "REFUTES"  if m_r  > max(m_s, m_theta)
      - "NEI"      otherwise (tie or theta dominates)
    """
    if mass.m_s > max(mass.m_r, mass.m_theta):
        return "SUPPORTS"
    if mass.m_r > max(mass.m_s, mass.m_theta):
        return "REFUTES"
    return "NEI"


def _load_llm_token_count(llm_cache_path: str) -> int:
    """Return total input tokens already consumed per the LLM cache file.

    Reads all valid JSONL entries and sums their input_tokens fields.
    Returns 0 if the file does not exist or has no valid entries.
    """
    path = Path(llm_cache_path)
    if not path.exists():
        return 0
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total += entry.get("input_tokens", 0)
            except (json.JSONDecodeError, KeyError):
                continue
    return total


def run_cell(
    dataset_name: str,
    system_name: str,
    seed: int,
    results_dir: str,
    llm_cache_path: str,
    max_input_tokens: int,
    systems: dict,
    loaders: dict,
) -> str:
    """Run one (dataset, system, seed) evaluation cell.

    Args:
        dataset_name:     Name of the dataset (e.g. "hover").
        system_name:      Name of the system (e.g. "symbolic_floor").
        seed:             Integer seed for reproducibility.
        results_dir:      Root directory for result files.
        llm_cache_path:   Path to the LLM JSONL cache file.
        max_input_tokens: Token budget. LLM cells exceeding this are skipped.
        systems:          Dict mapping system_name -> system object.
        loaders:          Dict mapping dataset_name -> callable returning list[EvalClaim].

    Returns:
        "skipped"          — output file already existed (checkpoint resume)
        "budget_exhausted" — LLM token budget exceeded
        "completed"        — evaluation ran and results saved
    """
    out_path = Path(results_dir) / dataset_name / system_name / f"seed_{seed}.json"

    # Checkpoint resume: skip if output already exists
    if out_path.exists():
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Budget guard: check if total tokens consumed exceeds the max budget
    used_tokens = _load_llm_token_count(llm_cache_path)
    if used_tokens > max_input_tokens:
        out_path.write_text(json.dumps({
            "status": "budget_exhausted",
            "dataset": dataset_name,
            "system": system_name,
            "seed": seed,
        }, indent=2))
        return "budget_exhausted"

    system = systems[system_name]
    claims = loaders[dataset_name]()

    from baselines._llm_cache import LLMCacheMissError

    results = []
    for claim in claims:
        try:
            mass = system.evaluate(claim)
        except LLMCacheMissError:
            # LLM cache miss in replay_only mode — treat as vacuous
            mass = MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)
        results.append({
            "claim_id": claim.claim_id,
            "ground_truth": claim.ground_truth,
            "mass": {
                "m_s": mass.m_s,
                "m_r": mass.m_r,
                "m_u": mass.m_u,
                "m_theta": mass.m_theta,
            },
            "predicted_label": _mass_to_label(mass),
            "metadata": claim.metadata,
        })

    out_path.write_text(json.dumps({
        "dataset": dataset_name,
        "system": system_name,
        "seed": seed,
        "results": results,
    }, indent=2))
    return "completed"


def _build_systems(
    system_names: list[str],
    llm_cache_path: str,
    max_input_tokens: int,
    llm_mode: str = "replay_only",
) -> dict:
    """Instantiate all requested systems by name.

    Args:
        system_names:      List of system names to instantiate.
        llm_cache_path:    Path to LLM cache (only used by llm_zeroshot).
        max_input_tokens:  Token budget (passed to llm_zeroshot).
        llm_mode:          LLM cache mode: "replay_only" (CI) or "record" (live calls).

    Returns:
        Dict mapping system_name -> system instance.
    """
    from baselines.symbolic_floor import SymbolicFloor
    from baselines.infon_system import InfonSystem
    from baselines.nli_classifier import NLIClassifier
    from baselines.llm_zeroshot import LLMZeroShot

    constructors = {
        "symbolic_floor": lambda: SymbolicFloor(),
        "infon_symbolic": lambda: InfonSystem("symbolic"),
        "infon_gnn": lambda: InfonSystem("gnn"),
        "flat_retrieval": lambda: InfonSystem("symbolic"),
        "nli_classifier": lambda: NLIClassifier(),
        "llm_zeroshot": lambda: LLMZeroShot(
            cache_path=llm_cache_path,
            mode=llm_mode,
            max_input_tokens=max_input_tokens,
        ),
    }

    systems = {}
    for name in system_names:
        if name not in constructors:
            raise ValueError(f"Unknown system: {name!r}. Valid: {sorted(constructors)}")
        systems[name] = constructors[name]()
    return systems


def _build_loaders(dataset_names: list[str], limit: int | None = None) -> dict:
    """Build dataset loader callables for each requested dataset.

    Args:
        dataset_names: Dataset names to load.
        limit: If set, each dataset is capped at this many claims.

    Returns:
        Dict mapping dataset_name -> zero-argument callable returning list[EvalClaim].
    """
    from benchmarks.hover.loader import load_hover
    from benchmarks.averitec import load_averitec
    from benchmarks.scifact import load_scifact

    all_loaders: dict[str, object] = {
        # HoVer: dev split, fetch Wikipedia text for evidence
        "hover": lambda: load_hover(limit=limit),
        # AVeriTeC: dev split, 500 claims
        "averitec": lambda: load_averitec(limit=limit),
        # SciFact: dev split (has evidence labels; test split does not)
        "scifact": lambda: load_scifact(split="dev", limit=limit),
    }

    loaders = {}
    for name in dataset_names:
        if name not in all_loaders:
            raise ValueError(f"Unknown dataset: {name!r}. Valid: {sorted(all_loaders)}")
        loaders[name] = all_loaders[name]
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint-resumable evaluation harness for the Infon benchmark matrix."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["hover", "averitec", "scifact"],
        help="Datasets to evaluate on.",
    )
    parser.add_argument(
        "--systems", nargs="+",
        default=["infon_symbolic", "infon_gnn", "flat_retrieval",
                 "symbolic_floor", "nli_classifier", "llm_zeroshot"],
        help="Systems to evaluate.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Random seeds for repeated evaluation.",
    )
    parser.add_argument(
        "--results-dir", default="experiments/results",
        help="Root directory for result files.",
    )
    parser.add_argument(
        "--llm-cache", default="experiments/results/llm_cache.jsonl",
        help="Path to the LLM JSONL cache file.",
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=4_000_000,
        help="Maximum total input tokens for LLM systems (budget guard).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap each dataset at this many claims. Default: None (use full dataset). "
             "Recommended: 500 for the full benchmark run to keep wall-time under 2 hours.",
    )
    parser.add_argument(
        "--llm-mode", default="replay_only", choices=["replay_only", "record"],
        help="LLM cache mode: 'replay_only' (CI, raises on miss) or 'record' (live Bedrock calls).",
    )
    args = parser.parse_args()

    systems = _build_systems(
        args.systems, args.llm_cache, args.max_input_tokens, llm_mode=args.llm_mode
    )
    loaders = _build_loaders(args.datasets, limit=args.limit)

    total = len(args.datasets) * len(args.systems) * len(args.seeds)
    done = 0
    for dataset in args.datasets:
        for system_name in args.systems:
            for seed in args.seeds:
                done += 1
                status = run_cell(
                    dataset_name=dataset,
                    system_name=system_name,
                    seed=seed,
                    results_dir=args.results_dir,
                    llm_cache_path=args.llm_cache,
                    max_input_tokens=args.max_input_tokens,
                    systems=systems,
                    loaders=loaders,
                )
                print(f"[{done}/{total}] {dataset}/{system_name}/seed_{seed}: {status}")


if __name__ == "__main__":
    main()
