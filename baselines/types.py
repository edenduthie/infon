"""Protocol defining the EvalSystem interface for all baseline systems."""

from typing import Protocol, runtime_checkable

from benchmarks.types import EvalClaim, MassFunction


@runtime_checkable
class EvalSystem(Protocol):
    name: str

    def evaluate(self, claim: EvalClaim) -> MassFunction:
        ...
