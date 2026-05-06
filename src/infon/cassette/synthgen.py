"""Synthetic hypergraph generator — ground truth for GNN training.

Flow at step 0:
  1. Agent peeks at real docs → writes a SynthGenConfig (JSON).
  2. generate(config, seed) → list of (hypergraph, labels).
  3. Training uses the labels as supervisor for two heads:
      • chain-verdict (did SUPPORTS / REFUTES / NEI hold for this chain?)
      • next-anchor (given a node, which anchor came next in the graph?)
  4. Frozen weights become the query-time prior for the cassette reasoner.

Key design decisions:
  • JSON-first config. The agent writes it; Python consumes it. No Python
    callbacks in the config — keeps it serializable and editable by hand.
  • Ground truth is computed from generator rules, not from a model. So
    the label fidelity probe can assert the symbolic chain_mass matches.
  • Distributional levers are explicit (chain_length, polarity_flip_rate,
    cycle_rate, anomaly_rate). The agent tunes these; the structural
    invariants (conjunctive chains, polarity cancels, etc.) are fixed.
  • Deterministic given (config, seed). Regen should reproduce exactly.

What the config does NOT include:
  • Neural architecture choices. Those live in gnn_encoder.py.
  • Training hyperparameters. Those live in the training script.
  • The synthetic sentences themselves. We generate structured triples,
    not prose — the GNN doesn't need natural-language inputs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════
# CONFIG  (agent writes; generator consumes)
# ═══════════════════════════════════════════════════════════════════════

RelationKind = Literal[
    "connective",   # chains compose: a→b→c implies a→c (supply, partner, license)
    "terminal",     # relation attaches an attribute, not another entity (invest→feature)
    "reportive",    # non-connective, mention-like (describe, discuss)
]


@dataclass
class RelationSpec:
    """One relation's structural semantics.

    Agent fills this after reading a few real predicates in the corpus.
    The `kind` field is what matters for the GNN — it distinguishes
    chain-propagating relations from attribute ones."""
    name: str                  # e.g. "supply"
    kind: RelationKind         # chain semantics
    symmetric: bool = False    # a R b ⇒ b R a (e.g. "compete")
    can_retract: bool = True   # polarity flips allowed for this relation
    typical_gap_days: int = 7  # mean temporal gap between NEXT events


@dataclass
class SynthGenConfig:
    """What the agent produces after peeking at real docs.

    Sizes and rates are distributional — aim to match what the real
    corpus roughly looks like, not match it exactly. The goal is to
    teach the GNN structural invariants, not to memorize the user's
    specific graph.
    """
    # ── vocabulary ───────────────────────────────────────────────────
    n_actors: int = 20
    n_features: int = 10
    relations: list[RelationSpec] = field(default_factory=list)

    # ── graph shape ──────────────────────────────────────────────────
    n_samples: int = 2000          # training examples
    chain_length_min: int = 1
    chain_length_max: int = 4
    chain_length_bias: float = 0.5  # 0 = uniform, 1 = strong preference for short chains

    # ── noise parameters (each in [0, 1]) ────────────────────────────
    polarity_flip_rate: float = 0.15   # fraction of edges that are retractions
    anomaly_rate: float = 0.05          # fraction of chains with an "impossible" edge
    cycle_rate: float = 0.10            # fraction with cycles (tests loop handling)
    noise_edge_rate: float = 0.20       # irrelevant edges added per subgraph

    # ── reproducibility ──────────────────────────────────────────────
    seed: int = 42
    description: str = ""  # freeform note from the agent: "battery supply chain"

    def to_json(self) -> str:
        d = asdict(self)
        d["relations"] = [asdict(r) for r in self.relations]
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "SynthGenConfig":
        d = json.loads(blob)
        rels = [RelationSpec(**r) for r in d.pop("relations", [])]
        return cls(relations=rels, **d)


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC HYPERGRAPH
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SynthEdge:
    """One triple with polarity and timestamp. Same shape as an Infon."""
    subject: str
    predicate: str
    object: str
    polarity: int = 1
    t: int = 0  # synthetic time index; maps to dates if needed
    confidence: float = 0.85


@dataclass
class SynthGraph:
    """A synthetic subgraph + the labels we want the GNN to predict."""
    edges: list[SynthEdge]

    # Task 1 supervisor — chain verdict at the endpoints.
    chain_source: str
    chain_target: str
    chain_verdict: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    # Task 2 supervisor — for each node in the chain, the "real" next anchor.
    next_map: dict[str, str] = field(default_factory=dict)

    # Metadata for diagnostics (not a training signal).
    kind: str = "clean"   # clean | retracted | anomaly | disconnected | cyclic


# ═══════════════════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def _default_config() -> SynthGenConfig:
    """Baseline config — plausible chip/entity domain.
    Used when the agent hasn't run; probes use this for testing."""
    return SynthGenConfig(
        n_actors=15,
        n_features=8,
        relations=[
            RelationSpec("supply", "connective", can_retract=True),
            RelationSpec("partner", "connective", symmetric=True),
            RelationSpec("acquire", "connective", can_retract=False),
            RelationSpec("license", "connective"),
            RelationSpec("invest", "terminal"),
            RelationSpec("mention", "reportive"),
        ],
        description="default synthetic config",
    )


def generate(config: SynthGenConfig) -> list[SynthGraph]:
    """Produce `config.n_samples` synthetic hypergraphs with ground truth.

    Sample distribution (approximate):
      • clean connective chains          → majority; supervisor=SUPPORTS
      • chains with a retraction         → polarity_flip_rate; → REFUTES
      • chains with an anomaly edge      → anomaly_rate; → NEI
      • disconnected source/target pairs → always; → NEI  (negative examples)
      • cyclic chains                    → cycle_rate; tests cycle handling
    """
    rng = random.Random(config.seed)

    actors = [f"a{i:03d}" for i in range(config.n_actors)]
    features = [f"f{i:03d}" for i in range(config.n_features)]
    connective = [r for r in config.relations if r.kind == "connective"]
    terminal = [r for r in config.relations if r.kind == "terminal"]
    reportive = [r for r in config.relations if r.kind == "reportive"]

    if not connective:
        raise ValueError(
            "Need at least one connective relation to generate chains."
        )

    out: list[SynthGraph] = []
    for _ in range(config.n_samples):
        # Pick chain length with a bias toward shorter chains (realistic).
        length = _biased_choice(
            rng,
            range(config.chain_length_min, config.chain_length_max + 1),
            bias=config.chain_length_bias,
        )

        # Roll for example kind. Order matters: "disconnected" first,
        # because a chain that never reaches the target is always NEI
        # regardless of polarity.
        roll = rng.random()
        if roll < 0.20:
            g = _gen_disconnected(rng, actors, features, connective,
                                   terminal, reportive, length, config)
        elif roll < 0.20 + config.polarity_flip_rate:
            g = _gen_retracted(rng, actors, features, connective,
                                terminal, reportive, length, config)
        elif roll < 0.20 + config.polarity_flip_rate + config.anomaly_rate:
            g = _gen_anomaly(rng, actors, features, connective,
                              terminal, reportive, length, config)
        elif roll < 0.20 + config.polarity_flip_rate + config.anomaly_rate + config.cycle_rate:
            g = _gen_cyclic(rng, actors, features, connective,
                             terminal, reportive, length, config)
        else:
            g = _gen_clean(rng, actors, features, connective,
                            terminal, reportive, length, config)

        # Always inject a small number of noise edges — teaches the GNN
        # to ignore irrelevant context.
        n_noise = int(len(g.edges) * config.noise_edge_rate) + 1
        for _ in range(n_noise):
            g.edges.append(_noise_edge(rng, actors, features,
                                         connective + terminal + reportive))
        out.append(g)
    return out


# ── helpers ─────────────────────────────────────────────────────────────

def _biased_choice(rng: random.Random, choices, bias: float = 0.5) -> int:
    """Choice with geometric bias toward the first element. bias=0 uniform,
    bias=1 deterministic first."""
    choices = list(choices)
    if not choices:
        raise ValueError("empty choices")
    if bias <= 0:
        return rng.choice(choices)
    # Weight ∝ (1 - bias) ** i
    weights = [(1 - bias) ** i for i in range(len(choices))]
    return rng.choices(choices, weights=weights, k=1)[0]


def _gen_clean(rng, actors, features, connective, terminal, reportive,
               length, config):
    """SUPPORTS: a proper connective chain from source to target."""
    path_actors = rng.sample(actors, length + 1)
    edges = []
    t = 0
    for i in range(length):
        r = rng.choice(connective)
        edges.append(SynthEdge(
            subject=path_actors[i],
            predicate=r.name,
            object=path_actors[i + 1],
            polarity=1,
            t=t,
            confidence=round(rng.uniform(0.75, 0.95), 2),
        ))
        t += max(1, int(rng.gauss(r.typical_gap_days, 2)))
    next_map = {path_actors[i]: path_actors[i + 1] for i in range(length)}
    return SynthGraph(
        edges=edges,
        chain_source=path_actors[0],
        chain_target=path_actors[-1],
        chain_verdict="SUPPORTS",
        next_map=next_map,
        kind="clean",
    )


def _gen_retracted(rng, actors, features, connective, terminal, reportive,
                   length, config):
    """REFUTES: an otherwise-clean chain with a later retraction on one edge."""
    g = _gen_clean(rng, actors, features, connective, terminal, reportive,
                   length, config)
    # Pick an edge to retract; append a polarity=0 twin at a later time.
    idx = rng.randrange(len(g.edges))
    e = g.edges[idx]
    retraction = SynthEdge(
        subject=e.subject,
        predicate=e.predicate,
        object=e.object,
        polarity=0,
        t=e.t + rng.randint(5, 30),
        confidence=round(rng.uniform(0.80, 0.95), 2),
    )
    g.edges.append(retraction)
    g.chain_verdict = "REFUTES"
    g.kind = "retracted"
    return g


def _gen_anomaly(rng, actors, features, connective, terminal, reportive,
                 length, config):
    """NEI: chain contains a reportive edge in a position the conjunctive
    chain can't cross — graph coincidentally touches target but no real
    chain exists."""
    g = _gen_clean(rng, actors, features, connective, terminal, reportive,
                   length, config)
    if reportive and g.edges:
        # Replace one connective edge with a reportive one.
        idx = rng.randrange(len(g.edges))
        r = rng.choice(reportive)
        g.edges[idx] = SynthEdge(
            subject=g.edges[idx].subject,
            predicate=r.name,
            object=g.edges[idx].object,
            polarity=1,
            t=g.edges[idx].t,
            confidence=round(rng.uniform(0.60, 0.85), 2),
        )
    g.chain_verdict = "NOT_ENOUGH_INFO"
    g.kind = "anomaly"
    return g


def _gen_disconnected(rng, actors, features, connective, terminal, reportive,
                       length, config):
    """NEI: source and target named, but no path of connective edges
    connects them. Some noise edges exist."""
    source, target = rng.sample(actors, 2)
    # Build a chain from source that leads somewhere else.
    path_actors = rng.sample(
        [a for a in actors if a != target], length
    )
    path_actors.insert(0, source)
    edges = []
    t = 0
    for i in range(len(path_actors) - 1):
        r = rng.choice(connective)
        edges.append(SynthEdge(
            subject=path_actors[i],
            predicate=r.name,
            object=path_actors[i + 1],
            polarity=1,
            t=t,
            confidence=round(rng.uniform(0.75, 0.95), 2),
        ))
        t += max(1, int(rng.gauss(r.typical_gap_days, 2)))
    return SynthGraph(
        edges=edges,
        chain_source=source,
        chain_target=target,
        chain_verdict="NOT_ENOUGH_INFO",
        next_map={},
        kind="disconnected",
    )


def _gen_cyclic(rng, actors, features, connective, terminal, reportive,
                length, config):
    """SUPPORTS: chain that loops back through an intermediate and still
    reaches target. Tests the GNN's ability to not get stuck on cycles."""
    g = _gen_clean(rng, actors, features, connective, terminal, reportive,
                   max(2, length), config)
    # Inject a cycle: add an edge from the middle back to the second node.
    if len(g.edges) >= 2:
        src = g.edges[-1].object
        tgt = g.edges[0].object
        r = rng.choice(connective)
        g.edges.append(SynthEdge(
            subject=src, predicate=r.name, object=tgt,
            polarity=1,
            t=g.edges[-1].t + rng.randint(1, 10),
            confidence=round(rng.uniform(0.70, 0.90), 2),
        ))
    g.kind = "cyclic"
    return g


def _noise_edge(rng, actors, features, relations):
    """An irrelevant edge that doesn't touch the chain."""
    r = rng.choice(relations)
    subj = rng.choice(actors)
    if r.kind == "terminal":
        obj = rng.choice(features)
    else:
        obj = rng.choice([a for a in actors if a != subj])
    return SynthEdge(
        subject=subj, predicate=r.name, object=obj,
        polarity=1 if rng.random() > 0.05 else 0,
        t=rng.randint(0, 100),
        confidence=round(rng.uniform(0.5, 0.95), 2),
    )


# ═══════════════════════════════════════════════════════════════════════
# STATS — for the probe and the agent to inspect
# ═══════════════════════════════════════════════════════════════════════

def summarize(graphs: list[SynthGraph]) -> dict:
    """Distributional summary of a generated corpus."""
    from collections import Counter
    kinds = Counter(g.kind for g in graphs)
    verdicts = Counter(g.chain_verdict for g in graphs)
    lengths = Counter(len([e for e in g.edges if e.polarity == 1])
                      for g in graphs)
    total_edges = sum(len(g.edges) for g in graphs)
    return {
        "n_graphs": len(graphs),
        "total_edges": total_edges,
        "mean_edges_per_graph": round(total_edges / max(1, len(graphs)), 1),
        "kinds": dict(kinds),
        "verdicts": dict(verdicts),
        "lengths": dict(sorted(lengths.items())),
    }
