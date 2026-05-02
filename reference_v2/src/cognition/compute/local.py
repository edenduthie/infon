"""LocalCompute: thread/process pool for single-machine parallelism."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Callable, Any


class LocalCompute:
    """Local compute backend using thread or process pools.

    Default is threads (good for I/O-bound work like encoding batches
    on MPS/CUDA where the GIL isn't the bottleneck). Switch to processes
    for CPU-bound consolidation work.
    """

    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self._pool = None

    @property
    def pool(self):
        if self._pool is None:
            cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            self._pool = cls(max_workers=self.max_workers)
        return self._pool

    def map(self, fn: Callable, items: list[Any], **kwargs) -> list[Any]:
        """Apply fn to each item in parallel."""
        if not items:
            return []
        # For small batches, sequential is faster than pool overhead
        if len(items) <= 2:
            return [fn(item, **kwargs) for item in items]
        futures = [self.pool.submit(fn, item, **kwargs) for item in items]
        return [f.result() for f in futures]

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a single task, returns a Future."""
        return self.pool.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
