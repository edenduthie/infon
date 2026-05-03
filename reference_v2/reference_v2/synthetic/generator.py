"""Template-based synthetic corpus generator for ablation experiments.

Generates deterministic Scenario objects with planted oracle labels
(planted_verdict, planted_thinness, planted_hop_count) without calling
any model or LLM.

Axes studied:
- H1: compositional_depth  → planted_hop_count
- H2: evidence_redundancy  → planted_thinness (supporting sentence count)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Sentence:
    """A single sentence in a scenario's corpus.

    Attributes:
        text: The sentence string.
        supports_query: Whether this sentence provides direct support for the
            scenario's query claim.
    """

    text: str
    supports_query: bool


@dataclass(frozen=True)
class Scenario:
    """A single fact-checking scenario with planted oracle labels.

    Attributes:
        corpus_sentences: All sentences in the generated corpus for this
            scenario (both supporting and distractor sentences).
        planted_verdict: Oracle verdict string — one of "SUPPORTS",
            "REFUTES", or "NEI".
        planted_thinness: Count of sentences where supports_query is True.
            This is the oracle label for evidence thinness (H2 axis).
        planted_hop_count: Number of compositional inference hops required
            to verify the query.  Equal to the compositional_depth parameter
            (H1 axis).
    """

    corpus_sentences: tuple[Sentence, ...]
    planted_verdict: str
    planted_thinness: int
    planted_hop_count: int


class Generator:
    """Deterministic template-based scenario factory.

    Two ``Generator`` instances constructed with the same seed will produce
    byte-equal output from ``generate()``, regardless of call order or
    Python process state.

    Args:
        seed: Integer seed for numpy's default RNG.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_docs: int,
        evidence_redundancy: int,
        compositional_depth: int,
        contradiction_density: float,
        nei_fraction: float,
    ) -> list[Scenario]:
        """Generate ``n_docs`` scenarios with planted oracle labels.

        Args:
            n_docs: Number of scenarios to generate.
            evidence_redundancy: Number of supporting sentences planted in
                each SUPPORTS scenario (H2 axis).
            compositional_depth: Number of inference hops; stored directly
                as planted_hop_count (H1 axis).
            contradiction_density: Fraction of scenarios assigned a
                "REFUTES" verdict.  Must be in [0.0, 1.0].
            nei_fraction: Fraction of scenarios assigned a "NEI" verdict.
                Must be in [0.0, 1.0].  contradiction_density + nei_fraction
                must not exceed 1.0.

        Returns:
            A list of ``n_docs`` Scenario objects with fully populated
            oracle labels.

        Raises:
            ValueError: If contradiction_density + nei_fraction > 1.0 or
                if n_docs < 1.
        """
        if n_docs < 1:
            raise ValueError(f"n_docs must be >= 1, got {n_docs}")
        if contradiction_density + nei_fraction > 1.0:
            raise ValueError(
                f"contradiction_density ({contradiction_density}) + "
                f"nei_fraction ({nei_fraction}) must not exceed 1.0"
            )

        # Re-seed at the start of every call for determinism.
        rng = np.random.default_rng(self._seed)

        # Determine per-scenario verdict assignments.
        verdicts = self._assign_verdicts(
            n_docs, contradiction_density, nei_fraction, rng
        )

        scenarios: list[Scenario] = []
        for doc_idx, verdict in enumerate(verdicts):
            scenario = self._build_scenario(
                doc_idx=doc_idx,
                verdict=verdict,
                evidence_redundancy=evidence_redundancy,
                compositional_depth=compositional_depth,
                rng=rng,
            )
            scenarios.append(scenario)

        return scenarios

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_verdicts(
        self,
        n_docs: int,
        contradiction_density: float,
        nei_fraction: float,
        rng: np.random.Generator,
    ) -> list[str]:
        """Assign verdict labels to each scenario index.

        The assignment is deterministic given the rng state.  Labels are
        drawn by sampling from a categorical distribution defined by the
        density parameters.
        """
        supports_fraction = 1.0 - contradiction_density - nei_fraction
        probabilities = [supports_fraction, contradiction_density, nei_fraction]
        labels = ["SUPPORTS", "REFUTES", "NEI"]
        choices = rng.choice(labels, size=n_docs, p=probabilities)
        return list(choices)

    def _build_scenario(
        self,
        doc_idx: int,
        verdict: str,
        evidence_redundancy: int,
        compositional_depth: int,
        rng: np.random.Generator,
    ) -> Scenario:
        """Construct a single Scenario with planted sentences.

        Supporting sentences are generated only for SUPPORTS verdicts.
        Distractor sentences are added to pad the corpus.

        The number of distractor sentences equals ``compositional_depth``
        (one per hop level) so the corpus size scales with depth.
        """
        sentences: list[Sentence] = []

        if verdict == "SUPPORTS":
            # Plant evidence_redundancy supporting sentences.
            for hop in range(compositional_depth):
                for redundancy_idx in range(evidence_redundancy):
                    text = (
                        f"Entity {doc_idx} hop {hop} witness {redundancy_idx}: "
                        f"fact {doc_idx} is confirmed true."
                    )
                    sentences.append(Sentence(text=text, supports_query=True))

            # Add distractor sentences that do not support the query.
            n_distractors = max(1, compositional_depth)
            for dist_idx in range(n_distractors):
                noise_val = int(rng.integers(0, 1_000_000))
                text = (
                    f"Entity {doc_idx} distractor {dist_idx}: "
                    f"unrelated context {noise_val}."
                )
                sentences.append(Sentence(text=text, supports_query=False))

        elif verdict == "REFUTES":
            # Plant a contradicting sentence (supports_query=False).
            text = (
                f"Entity {doc_idx}: fact {doc_idx} is demonstrably false."
            )
            sentences.append(Sentence(text=text, supports_query=False))

            # Add neutral distractors.
            n_distractors = max(1, compositional_depth)
            for dist_idx in range(n_distractors):
                noise_val = int(rng.integers(0, 1_000_000))
                text = (
                    f"Entity {doc_idx} distractor {dist_idx}: "
                    f"unrelated context {noise_val}."
                )
                sentences.append(Sentence(text=text, supports_query=False))

        else:  # NEI
            # No supporting or contradicting evidence.
            n_distractors = max(1, compositional_depth)
            for dist_idx in range(n_distractors):
                noise_val = int(rng.integers(0, 1_000_000))
                text = (
                    f"Entity {doc_idx} distractor {dist_idx}: "
                    f"unrelated context {noise_val}."
                )
                sentences.append(Sentence(text=text, supports_query=False))

        # Compute planted_thinness by counting supports_query=True sentences.
        planted_thinness = sum(1 for s in sentences if s.supports_query)

        return Scenario(
            corpus_sentences=tuple(sentences),
            planted_verdict=verdict,
            planted_thinness=planted_thinness,
            planted_hop_count=compositional_depth,
        )
