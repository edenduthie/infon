"""Fan-out executor — pywren-style map for cassette workloads.

One protocol (`Executor.map(fn, items) → list[R]`), multiple implementations.
The cassette pipeline was designed for this: one doc → one cassette → one
footer summary. Workers are pure, idempotent (content-addressed output),
and return small summaries (≤1KB) that the driver aggregates into a
single manifest commit.

Implementations:
  SyncExecutor     — in-process sequential. Baseline, always available.
  ProcessExecutor  — concurrent.futures ProcessPoolExecutor. CPU parallel
                      on a single machine. Good for 10–100 docs.
  LambdaExecutor   — AWS Lambda fan-out. Placeholder here; real impl lives
                      in a separate module to keep boto3 optional.

Design choices:
  • map only — no reduce, no futures. Each doc's output is independent
    and written directly to the cassette/ + index/ paths. The driver's
    only post-map work is collecting footer summaries and writing one
    manifest snapshot.
  • Functions must be picklable. For ProcessExecutor that's enforced by
    the stdlib; for Lambda it's enforced by the handler contract.
  • Errors per item are collected into Result objects, not raised, so a
    single bad doc doesn't abort a batch of 10k.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Result:
    """One map result. Either `value` is set or `error` is set, never both."""
    index: int
    value: object | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Executor(Protocol):
    """Minimal interface: map a function over an iterable of items."""

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[Result]:
        """Apply `fn` to each item. Returns list[Result] in input order.

        Individual item failures are captured in Result.error; the whole
        batch does not abort. Ordering is preserved.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# IN-PROCESS (baseline)
# ═══════════════════════════════════════════════════════════════════════

class SyncExecutor:
    """Runs fn on the calling thread. Zero overhead, one-at-a-time."""

    def map(self, fn, items):
        out = []
        for i, item in enumerate(items):
            try:
                out.append(Result(index=i, value=fn(item)))
            except Exception as exc:
                out.append(Result(index=i, error=f"{type(exc).__name__}: {exc}"))
        return out


# ═══════════════════════════════════════════════════════════════════════
# PROCESS POOL — multi-core on one machine
# ═══════════════════════════════════════════════════════════════════════

class ProcessExecutor:
    """Fan-out over a concurrent.futures ProcessPoolExecutor.

    Every worker initializes from scratch on the first call; the pool is
    reused across map() calls on the same instance. Use this when a
    single machine has unused cores and your items are CPU-bound
    (extraction is).

    Pitfalls:
      • `fn` must be picklable (module-level function, not closure).
      • Heavy imports in `fn` (torch, transformers) pay cold-start ONCE
        per worker, not per item. Keep the function focused.
      • Don't pass 500MB arguments — workers serialize them.
    """

    def __init__(self, workers: int = 4, mp_context: str | None = "spawn"):
        import multiprocessing as mp
        self.workers = workers
        self.mp_context = mp.get_context(mp_context) if mp_context else None
        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=self.mp_context,
            )
        return self._pool

    def map(self, fn, items):
        items = list(items)
        pool = self._get_pool()
        results: list[Result] = [Result(index=i) for i in range(len(items))]
        futures = {pool.submit(_safe_call, fn, item): i
                   for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                ok, payload = fut.result()
                if ok:
                    results[i].value = payload
                else:
                    results[i].error = payload
            except Exception as exc:
                results[i].error = f"{type(exc).__name__}: {exc}"
        return results

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _safe_call(fn, item):
    """Worker-side: run fn(item), catch exceptions, return (ok, payload).

    Returning instead of raising means traceback strings cross the
    process boundary cleanly — pickling a live exception's __traceback__
    is lossy."""
    try:
        return True, fn(item)
    except Exception:
        return False, traceback.format_exc(limit=3)


# ═══════════════════════════════════════════════════════════════════════
# LAMBDA (stub — real impl lives in executor_lambda.py to keep boto3 optional)
# ═══════════════════════════════════════════════════════════════════════

class LambdaExecutor:
    """AWS Lambda fan-out. Requires `boto3`; see executor_lambda.py.

    The shape is intentionally identical to ProcessExecutor: same `map()`
    signature, same Result type. Swapping executors changes only the
    backend, not the pipeline code."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LambdaExecutor not yet implemented — use SyncExecutor or "
            "ProcessExecutor for now. Will land once the layer packaging "
            "is tested."
        )
