"""Compute backends: protocol + local (threads) and cloud (Lambda+SQS).

All compute backends implement the ComputeBackend protocol — code above
this layer submits work and gets results back regardless of execution venue.
"""

from __future__ import annotations

from typing import Protocol, Callable, Any, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol that all compute backends implement."""

    def map(self, fn: Callable, items: list[Any], **kwargs) -> list[Any]:
        """Apply fn to each item in parallel. Returns results in order."""
        ...

    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """Submit a single task. Returns a future-like object."""
        ...

    def shutdown(self) -> None:
        """Clean up resources."""
        ...
