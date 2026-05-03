"""CLI for generating synthetic scenario splits and saving them as JSON.

Usage::

    python3 -m reference_v2.synthetic.generate \\
        --train 80 --dev 10 --test 10 --seed 42 \\
        --out experiments/results/synthetic_pilot/data/

Each split is saved as ``{out}/train.json``, ``{out}/dev.json``, and
``{out}/test.json``.  Every file is a JSON array of scenario dicts with
the following keys:

* ``corpus_sentences`` — list of ``{"text": str, "supports_query": bool}``
* ``planted_verdict`` — one of ``"SUPPORTS"``, ``"REFUTES"``, ``"NEI"``
* ``planted_thinness`` — int (count of sentences where supports_query is True)
* ``planted_hop_count`` — int (compositional depth / number of inference hops)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from reference_v2.synthetic.generator import Generator, Scenario
from reference_v2.synthetic.splits import make_splits


# Default generation parameters for the ablation study.
_DEFAULT_EVIDENCE_REDUNDANCY = 2
_DEFAULT_COMPOSITIONAL_DEPTH = 2
_DEFAULT_CONTRADICTION_DENSITY = 0.33
_DEFAULT_NEI_FRACTION = 0.33


def _scenario_to_dict(scenario: Scenario) -> dict:
    """Convert a Scenario dataclass to a JSON-serialisable dict."""
    return {
        "corpus_sentences": [
            {"text": s.text, "supports_query": s.supports_query}
            for s in scenario.corpus_sentences
        ],
        "planted_verdict": scenario.planted_verdict,
        "planted_thinness": scenario.planted_thinness,
        "planted_hop_count": scenario.planted_hop_count,
    }


def generate_and_save(
    train: int,
    dev: int,
    test: int,
    seed: int,
    out: Path,
    evidence_redundancy: int = _DEFAULT_EVIDENCE_REDUNDANCY,
    compositional_depth: int = _DEFAULT_COMPOSITIONAL_DEPTH,
    contradiction_density: float = _DEFAULT_CONTRADICTION_DENSITY,
    nei_fraction: float = _DEFAULT_NEI_FRACTION,
) -> None:
    """Generate scenarios, split them, and write JSON files to *out*.

    Args:
        train: Number of training scenarios.
        dev: Number of development scenarios.
        test: Number of test scenarios.
        seed: Random seed for reproducibility.
        out: Output directory path.
        evidence_redundancy: Supporting sentences per SUPPORTS scenario.
        compositional_depth: Inference hops; stored as planted_hop_count.
        contradiction_density: Fraction of scenarios assigned REFUTES verdict.
        nei_fraction: Fraction of scenarios assigned NEI verdict.
    """
    total = train + dev + test
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {total} scenarios "
        f"(train={train}, dev={dev}, test={test}, seed={seed}) ...",
        flush=True,
    )

    t0 = time.monotonic()

    # Generate all scenarios using the Generator.
    generator = Generator(seed=seed)
    scenarios = generator.generate(
        n_docs=total,
        evidence_redundancy=evidence_redundancy,
        compositional_depth=compositional_depth,
        contradiction_density=contradiction_density,
        nei_fraction=nei_fraction,
    )

    elapsed = time.monotonic() - t0
    rate = total / elapsed if elapsed > 0 else float("inf")
    print(f"Generated {total} scenarios in {elapsed:.3f}s ({rate:.1f} scenarios/sec)")

    # Obtain stratified split indices via make_splits.
    # make_splits returns dicts with scenario_id and evidence_redundancy.
    # We use the scenario index (parsed from scenario_id) to map back to
    # the generated Scenario objects.
    splits = make_splits(seed=seed, train=train, dev=dev, test=test)

    split_names = ("train", "dev", "test")
    expected_sizes = {"train": train, "dev": dev, "test": test}

    for split_name in split_names:
        split_meta = splits[split_name]
        # Extract integer indices from scenario_id strings like "scenario_00042".
        indices = [int(m["scenario_id"].split("_")[1]) for m in split_meta]
        split_scenarios = [scenarios[i] for i in indices]

        assert len(split_scenarios) == expected_sizes[split_name], (
            f"Expected {expected_sizes[split_name]} scenarios for {split_name}, "
            f"got {len(split_scenarios)}"
        )

        records = [_scenario_to_dict(s) for s in split_scenarios]
        out_path = out / f"{split_name}.json"
        out_path.write_text(json.dumps(records, indent=2))
        print(f"Wrote {len(records)} scenarios -> {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fact-checking scenario splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train", type=int, required=True, help="Number of training scenarios."
    )
    parser.add_argument(
        "--dev", type=int, required=True, help="Number of development scenarios."
    )
    parser.add_argument(
        "--test", type=int, required=True, help="Number of test scenarios."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for train.json, dev.json, test.json.",
    )
    parser.add_argument(
        "--evidence-redundancy",
        type=int,
        default=_DEFAULT_EVIDENCE_REDUNDANCY,
        help="Number of supporting sentences per SUPPORTS scenario.",
    )
    parser.add_argument(
        "--compositional-depth",
        type=int,
        default=_DEFAULT_COMPOSITIONAL_DEPTH,
        help="Number of inference hops (planted_hop_count).",
    )
    parser.add_argument(
        "--contradiction-density",
        type=float,
        default=_DEFAULT_CONTRADICTION_DENSITY,
        help="Fraction of scenarios assigned REFUTES verdict.",
    )
    parser.add_argument(
        "--nei-fraction",
        type=float,
        default=_DEFAULT_NEI_FRACTION,
        help="Fraction of scenarios assigned NEI verdict.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI module."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    generate_and_save(
        train=args.train,
        dev=args.dev,
        test=args.test,
        seed=args.seed,
        out=args.out,
        evidence_redundancy=args.evidence_redundancy,
        compositional_depth=args.compositional_depth,
        contradiction_density=args.contradiction_density,
        nei_fraction=args.nei_fraction,
    )


if __name__ == "__main__":
    main()
