"""Infon: the universal knowledge unit.

An infon is a grounded <<predicate, subject, object; polarity>> extracted
from text, carrying hierarchy metadata, spatial/temporal context, importance
scoring, and edges to other infons.

See INFON_SPEC.md for the full specification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Span:
    """A character span in source text."""
    text: str
    start: int
    end: int


@dataclass
class Infon:
    """A grounded unit of information extracted from text."""

    # Identity
    infon_id: str = ""

    # Core triple
    subject: str = ""
    predicate: str = ""
    object: str = ""
    polarity: int = 1               # 1=affirmed, 0=negated
    direction: str = "forward"       # forward, reverse, neutral
    confidence: float = 0.0          # geometric mean of role probs

    # Grounding
    sentence: str = ""
    doc_id: str = ""
    sent_id: str = ""
    spans: dict = field(default_factory=dict)       # {role: {text, start, end}}
    support: dict = field(default_factory=dict)      # {role: direct|semantic|hierarchical}

    # Hierarchy metadata (from anchor schema)
    subject_meta: dict = field(default_factory=dict)
    predicate_meta: dict = field(default_factory=dict)
    object_meta: dict = field(default_factory=dict)

    # Spatial
    locations: list = field(default_factory=list)    # [{name, level, macro_region, ...}]

    # Temporal
    timestamp: str | None = None
    precision: str = "unknown"
    temporal_refs: list = field(default_factory=list)
    tense: str = "unknown"
    aspect: str = "unknown"

    # Metrics — McKinsey-style metadata for structural analysis
    # Populated by structural.py engines (Kano, conjoint, gap functor, etc.)
    # Default keys set by StructuralAnalyzer.enrich():
    #   kano_class: must_be | one_dimensional | attractive | indifferent | reverse
    #   kano_score: float in [-1, 1] (dysfunctional → functional)
    #   conjoint_utility: float — part-worth utility contribution
    #   conjoint_importance: float in [0, 1] — relative attribute importance
    #   signal_count: int — discourse signal references
    #   spec_present: bool — whether the feature exists in spec/engineering data
    #   discourse_present: bool — whether discussed in market signals
    #   gap_type: silent | hallucinated | overlap | none
    #   polarization_h1: float — sheaf cohomology H¹ (contradiction density)
    #   bull_count: int — bullish signal count
    #   bear_count: int — bearish signal count
    #   narrative_beta0: float — connected components (coverage fragmentation)
    #   narrative_beta1: float — persistent loops (recurring narrative cycles)
    #   contagion_exposure: float — bearish signal fan-out degree
    #   ghost: bool — zero market signals (ghost product)
    #   driver_node: str — position in driver tree (e.g., "revenue.demand.organic")
    metrics: dict = field(default_factory=dict)

    # Importance
    activation: float = 0.0          # model confidence
    coherence: float = 0.0           # sheaf NPMI consistency
    specificity: float = 0.0         # inverse doc frequency
    novelty: float = 1.0             # how new vs existing
    importance: float = 0.0          # composite score
    reinforcement_count: int = 0
    last_reinforced: str | None = None
    decay_rate: float = 0.01

    # Imagination provenance (populated for infons produced by imagine.py)
    kind: str = "observed"                          # "observed" | "imagined"
    parent_infon_ids: list = field(default_factory=list)
    fitness: float | None = None                    # GA score; None for observed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Infon:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def triple_key(self) -> tuple[str, str, str]:
        """Canonical (subject, predicate, object) key for deduplication."""
        return (self.subject, self.predicate, self.object)

    def __repr__(self):
        pol = "1" if self.polarity else "0"
        return (f"Infon(<<{self.predicate}, {self.subject}, {self.object}; {pol}>> "
                f"conf={self.confidence:.3f})")


@dataclass
class Edge:
    """A typed directed edge in the knowledge graph."""
    source: str
    target: str
    edge_type: str           # INITIATES, ASSERTS, TARGETS, NEXT, ENTAILS, etc.
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Constraint:
    """An aggregated claim across multiple infons."""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    evidence: int = 0            # number of supporting infons
    doc_count: int = 0           # number of distinct documents
    strength: float = 0.0        # mean confidence across evidence
    persistence: int = 0         # number of distinct time windows
    score: float = 0.0           # composite score
    infon_ids: list = field(default_factory=list)

    def triple_key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Constraint:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Hit:
    """One scored, ranked retrieval hit."""
    infon: Infon
    score: float                                     # final retrieval score
    snippet: str = ""                                # highlighted sentence fragment
    valence: float = 0.0                             # persona-signed valence

    def to_dict(self) -> dict:
        return {
            "infon": self.infon.to_dict(),
            "score": self.score,
            "snippet": self.snippet,
            "valence": self.valence,
        }


@dataclass
class QueryResult:
    """Result of a BlackMagic query (cognition-compatible)."""
    query: str = ""
    persona: str = ""
    infons: list[Infon] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    valence: dict = field(default_factory=dict)      # {infon_id: float}
    timeline: list = field(default_factory=list)       # chronologically ordered infons
    anchors_activated: dict = field(default_factory=dict)  # {name: prob}
    hits: list[Hit] = field(default_factory=list)    # new: fitness-ranked view


# Alias for API clarity — SearchResult is what bm.search() returns.
SearchResult = QueryResult


# ── Imagination (GA) result types ───────────────────────────────────────
# ImaginationResult is deliberately isomorphic to MCTSResult so the same
# renderers work for both — dual verdict fields let callers either use
# imagination-native labels (PLAUSIBLE/CONTRADICTED/SPECULATIVE) or the
# MCTS-compatible mapping (SUPPORTS/REFUTES/UNCERTAIN).

# Stable mapping imagination-native → MCTS-compatible
IMAGINATION_TO_MCTS_VERDICT = {
    "PLAUSIBLE":    "SUPPORTS",
    "CONTRADICTED": "REFUTES",
    "SPECULATIVE":  "UNCERTAIN",
}


@dataclass
class ImaginationNode:
    """One node in the imagination GA traversal tree — mirrors MCTSNode."""
    node_id: str = ""
    anchor_path: list = field(default_factory=list)       # anchors touched
    infons: list[Infon] = field(default_factory=list)    # GA population members
    parent: ImaginationNode | None = None
    children: list = field(default_factory=list)         # list[ImaginationNode]
    visit_count: int = 0                                  # generation depth
    belief_mass: object | None = None                     # MassFunction — DS-shaped
    fitness: float = 0.0                                  # GA fitness at this node
    mutator_used: str | None = None                       # name of mutator
    is_terminal: bool = False

    def __repr__(self):
        return (f"ImaginationNode({self.node_id[:10]} "
                f"anchors={self.anchor_path} "
                f"fit={self.fitness:.3f} children={len(self.children)})")


@dataclass
class ImaginationResult:
    """Query-scoped GA imagination output. MCTS-shaped."""
    query: str = ""
    seed_anchors: dict = field(default_factory=dict)     # {anchor: activation}

    # Dual verdicts
    verdict: str = "SPECULATIVE"                          # imagination-native
    mcts_verdict: str = "UNCERTAIN"                       # MCTS-compatible

    combined_mass: object | None = None                   # MassFunction
    traversal_tree: ImaginationNode | None = None

    iteration_log: list = field(default_factory=list)
    chains_discovered: list = field(default_factory=list)
    imagined_infons: list[Infon] = field(default_factory=list)

    nodes_explored: int = 0
    infons_evaluated: int = 0
    generations: int = 0
    iterations: int = 0                                   # alias of generations
    elapsed_s: float = 0.0
