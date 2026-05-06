"""EvalClaim types for the evidence generation evaluation harness."""

from dataclasses import dataclass, field
from typing import Literal, NamedTuple


class MassFunction(NamedTuple):
    m_s: float      # SUPPORTS mass
    m_r: float      # REFUTES mass
    m_u: float      # mixed/uncertain mass
    m_theta: float  # vacuous mass (total ignorance)


@dataclass
class EvalClaim:
    claim_id: str
    claim_text: str
    evidence_docs: list[str]          # text of associated evidence docs/passages
    ground_truth: Literal["SUPPORTS", "REFUTES", "NEI"]
    metadata: dict = field(default_factory=dict)  # dataset-specific: num_hops, rationales, etc.
