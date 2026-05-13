"""Infon: the universal knowledge unit.

An infon is a grounded <<predicate, subject, object; polarity>> extracted
from text, carrying hierarchy metadata, spatial/temporal context, importance
scoring, and edges to other infons.

See INFON_SPEC.md for the full specification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


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
class QueryResult:
    """Result of a cognition query."""
    query: str = ""
    persona: str = ""
    infons: list[Infon] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    valence: dict = field(default_factory=dict)      # {infon_id: float}
    timeline: list = field(default_factory=list)       # chronologically ordered infons
    anchors_activated: dict = field(default_factory=dict)  # {name: prob}
