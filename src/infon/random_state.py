"""Central seed management — sklearn-style random_state.

Sets every source of stochasticity we touch: torch, torch.cuda, numpy,
and Python's `random` module. Called once at `Cognition.__init__` when
`config.random_state` is set, so downstream fit/train/sample calls are
bit-identical across runs.

We expose a helper that returns a *per-call* generator derived from the
same seed — useful for places where we don't want to disturb the global
state (e.g. view corruption in SSL, synthetic corpus generation).
"""
from __future__ import annotations

import random
import hashlib


def set_global_seed(seed: int) -> None:
    """Pin every global RNG to the given seed.

    Called once at Cognition.__init__ when random_state is non-None.
    Safe to call multiple times — last call wins.
    """
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base: int | None, *tags: str) -> int | None:
    """Return a deterministic sub-seed from `base` and a string path.

    Lets multiple components coexist without one consuming another's
    RNG stream — e.g. `derive_seed(42, "extract")` vs
    `derive_seed(42, "embedder")` give different but reproducible
    seeds off the same root.

    Returns None if base is None so callers can short-circuit.
    """
    if base is None:
        return None
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode())
    for t in tags:
        h.update(b"\x00")
        h.update(t.encode())
    # Python int from the first 4 bytes — stays in int32 range
    return int.from_bytes(h.digest()[:4], "big", signed=False)


def make_python_rng(base: int | None, *tags: str) -> random.Random:
    """A seeded Python Random instance scoped to a particular call site."""
    return random.Random(derive_seed(base, *tags))


def make_torch_generator(base: int | None, *tags: str):
    """A seeded torch.Generator scoped to a particular call site."""
    import torch
    g = torch.Generator()
    seed = derive_seed(base, *tags)
    if seed is not None:
        g.manual_seed(seed)
    return g
