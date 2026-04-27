"""BlackMagic — sparse retrieval + reasoning + GA imagination on splade-tiny.

English-only fork of the `cognition` package. Preserves the full cognition
reasoning surface (persona valence, contrary views, NEXT edges, constraint
aggregation, Dempster-Shafer, MCTS) and adds a GA imagination layer whose
output is isomorphic to MCTSResult for renderer compatibility.

Quickstart:
    from blackmagic import BlackMagic, BlackMagicConfig

    bm = BlackMagic(BlackMagicConfig(schema_path="schemas/automotive.json"))
    bm.ingest([{"text": "Toyota invests in batteries.", "id": "d1",
                "timestamp": "2026-04-20"}])
    result = bm.search("Which automakers are investing in batteries?")
    for hit in result.hits:
        print(hit.score, hit.infon)

    imagined = bm.imagine("What partnerships might emerge in the EV market?")
    print(imagined.verdict, imagined.mcts_verdict)
"""

from __future__ import annotations

from .config import BlackMagicConfig
from .schema import AnchorSchema
from .infon import (
    Infon, Edge, Constraint, Span,
    Hit, QueryResult, SearchResult,
    ImaginationNode, ImaginationResult,
    IMAGINATION_TO_MCTS_VERDICT,
)

__version__ = "0.1.0"

__all__ = [
    "BlackMagic",
    "BlackMagicConfig",
    "AnchorSchema",
    "Infon",
    "Edge",
    "Constraint",
    "Span",
    "Hit",
    "QueryResult",
    "SearchResult",
    "ImaginationNode",
    "ImaginationResult",
    "IMAGINATION_TO_MCTS_VERDICT",
    "__version__",
]


# BlackMagic facade is imported lazily to avoid torch import on package load
def __getattr__(name):
    if name == "BlackMagic":
        from .facade import BlackMagic as _BlackMagic
        return _BlackMagic
    raise AttributeError(f"module 'blackmagic' has no attribute {name!r}")
