"""NLI (Natural Language Inference) baseline for fact-checking.

Uses a cross-encoder DeBERTa-v3 NLI model to classify claim-evidence pairs.
The model output (ENTAILMENT / CONTRADICTION / NEUTRAL) is mapped to the
Dempster-Shafer MassFunction.

Mapping:
  ENTAILMENT   → m_s = confidence, m_theta = 1 - confidence
  CONTRADICTION → m_r = confidence, m_theta = 1 - confidence
  NEUTRAL      → m_theta = 1.0  (consistent with DS semantics: no commitment)

Temperature scaling:
  confidence = sigmoid(logit / temperature)
  temperature=1.0 (default) leaves raw model confidence unchanged.
  temperature>1.0 makes the distribution softer (more uncertain).
"""

from __future__ import annotations

import math

from benchmarks.types import EvalClaim, MassFunction

# Maximum characters for the combined evidence premise (truncate for speed)
_MAX_PREMISE_CHARS = 2048


def _join_evidence(docs: list[str]) -> str:
    """Concatenate evidence docs into a single premise string."""
    joined = " ".join(docs)
    return joined[:_MAX_PREMISE_CHARS]


def _apply_temperature(raw_score: float, temperature: float) -> float:
    """Temperature-scale a probability score via logit space.

    Converts: score → logit → scaled logit → probability.
    """
    if temperature == 1.0:
        return raw_score
    # Clamp to avoid log(0)
    score = max(1e-7, min(1.0 - 1e-7, raw_score))
    logit = math.log(score / (1.0 - score))
    scaled_logit = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled_logit))


class NLIClassifier:
    """DeBERTa-v3 cross-encoder NLI baseline.

    Args:
        model_name: HuggingFace model identifier.
        temperature: logit temperature for calibration (1.0 = no scaling).
    """

    name = "nli_classifier"

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        temperature: float = 1.0,
    ) -> None:
        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # CPU inference
        )
        self.temperature = temperature

    def evaluate(self, claim: EvalClaim) -> MassFunction:
        """Classify the claim against the concatenated evidence premise.

        The cross-encoder takes (premise, hypothesis) where:
          premise    = joined evidence docs
          hypothesis = claim text
        """
        if not claim.evidence_docs:
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        premise = _join_evidence(claim.evidence_docs)
        result = self.pipe(premise, text_pair=claim.claim_text)

        # Handle both single-dict and list-of-dict output formats
        if isinstance(result, list):
            result = result[0]

        label = result["label"].lower()
        raw_score = float(result["score"])
        confidence = _apply_temperature(raw_score, self.temperature)

        if label == "entailment":
            return MassFunction(
                m_s=confidence,
                m_r=0.0,
                m_u=0.0,
                m_theta=1.0 - confidence,
            )
        elif label == "contradiction":
            return MassFunction(
                m_s=0.0,
                m_r=confidence,
                m_u=0.0,
                m_theta=1.0 - confidence,
            )
        else:
            # NEUTRAL → full ignorance (no commitment in DS semantics)
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)
