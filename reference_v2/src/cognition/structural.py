"""Structural analysis engines for McKinsey-style intelligence.

Nine composable engines that turn hypergraph topology into business metrics,
driver trees, and unit economics assumptions — all computed from graph
structure alone, no financial data required.

Engines:
    1. KanoClassifier       — feature importance classification
    2. ConjointEstimator    — part-worth utility from adoption curves
    3. FeatureGapFunctor    — F: Engineering → Perception distortion
    4. GhostDetector        — products/anchors with zero market signal
    5. PolarizationIndex    — sheaf H¹ contradiction density
    6. NarrativeAnalyzer    — persistent homology β₀/β₁ on temporal chains
    7. ContagionAnalyzer    — bearish signal fan-out over COMPETES_WITH
    8. KanExtension         — market-transfer estimates (left=optimistic, right=conservative)
    9. DriverTree           — composable tree from structural signals

Usage:
    from cognition.structural import StructuralAnalyzer

    sa = StructuralAnalyzer(store, schema, encoder)
    results = sa.run_all(infons)          # dict of engine name → result
    enriched = sa.enrich(infons)          # populates infon.metrics in-place
    tree = sa.driver_tree(infons)         # returns DriverNode root
"""

from __future__ import annotations

import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime

from .infon import Infon, Edge, Constraint
from .schema import AnchorSchema


# ═══════════════════════════════════════════════════════════════════════
# RESULT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KanoResult:
    """Result of Kano classification for an anchor."""
    anchor: str
    kano_class: str         # must_be | one_dimensional | attractive | indifferent | reverse
    adoption_rate: float    # fraction of products/docs that mention this anchor
    satisfaction_delta: float  # change in positive polarity when present vs absent
    score: float            # mapped to [-1, 1]: dysfunctional → functional

    @property
    def is_table_stakes(self) -> bool:
        return self.kano_class == "must_be"


@dataclass
class ConjointResult:
    """Conjoint part-worth utility for an anchor."""
    anchor: str
    utility: float          # part-worth contribution
    importance: float       # relative importance in [0, 1]
    levels: dict = field(default_factory=dict)  # {level_name: utility}


@dataclass
class FeatureGapResult:
    """Feature gap functor result for an anchor."""
    anchor: str
    spec_present: bool
    discourse_present: bool
    gap_type: str           # silent | hallucinated | overlap | absent
    spec_score: float       # engineering/spec activation strength
    discourse_score: float  # market signal activation strength
    discourse_count: int    # number of signal mentions


@dataclass
class GhostResult:
    """Ghost detection result for an anchor."""
    anchor: str
    anchor_type: str
    signal_count: int
    is_ghost: bool
    carrying_cost_proxy: float  # reinforcement_count = engineering effort spent


@dataclass
class PolarizationResult:
    """Polarization index for an anchor."""
    anchor: str
    bull_count: int
    bear_count: int
    total_count: int
    h1: float               # min(bull, bear) — irreducible contradiction
    polarization_ratio: float  # h1 / total if total > 0
    direction: str           # positive | negative | balanced | silent


@dataclass
class NarrativeResult:
    """Narrative lifecycle analysis for an anchor."""
    anchor: str
    chain_length: int        # total NEXT edges in the chain
    beta0: float             # connected components (fragmentation)
    beta1: float             # persistent loops (recurring narrative)
    gap_distribution: dict = field(default_factory=dict)  # {bucket: count}
    phase: str = "unknown"   # emerging | ramping | mature | decaying | dead
    velocity: float = 0.0    # signals per time unit at peak


@dataclass
class ContagionResult:
    """Supply chain contagion analysis for an anchor."""
    anchor: str
    bearish_signals: int
    direct_hits: list = field(default_factory=list)   # anchors directly hit
    fan_out_degree: int = 0   # how many competitors benefit
    beneficiaries: list = field(default_factory=list)  # anchors that gain
    contagion_score: float = 0.0


@dataclass
class KanExtensionResult:
    """Kan extension: market transfer estimate."""
    source_context: str      # e.g., "US_suv"
    target_context: str      # e.g., "EU_suv"
    left_estimate: dict = field(default_factory=dict)   # optimistic {anchor: score}
    right_estimate: dict = field(default_factory=dict)  # conservative {anchor: score}
    uncertainty_budget: dict = field(default_factory=dict)  # left - right gap
    shared_anchors: list = field(default_factory=list)
    source_only: list = field(default_factory=list)
    target_only: list = field(default_factory=list)


@dataclass
class DriverNode:
    """A node in the driver tree."""
    name: str
    path: str                # dot-separated path, e.g., "revenue.demand.organic"
    value: float = 0.0       # computed metric
    unit: str = ""           # e.g., "signals", "ratio", "index"
    source_engine: str = ""  # which structural engine computed this
    children: list = field(default_factory=list)  # list[DriverNode]
    evidence: list = field(default_factory=list)   # supporting infon_ids

    def leaf_values(self) -> dict[str, float]:
        if not self.children:
            return {self.path: self.value}
        result = {}
        for child in self.children:
            result.update(child.leaf_values())
        return result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "value": round(self.value, 4),
            "unit": self.unit,
            "source_engine": self.source_engine,
            "children": [c.to_dict() for c in self.children],
            "evidence_count": len(self.evidence),
        }


# ═══════════════════════════════════════════════════════════════════════
# 1. KANO CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

class KanoClassifier:
    """Classify anchors into Kano categories from infon co-occurrence patterns.

    Kano model maps features to customer satisfaction impact:
    - must_be: expected by all — absence causes dissatisfaction, presence doesn't delight
    - one_dimensional: linear — more is better, less is worse
    - attractive: delighter — presence excites, absence is OK
    - indifferent: nobody cares
    - reverse: presence causes dissatisfaction

    We compute this from the hypergraph: adoption_rate (how many docs mention it)
    crossed with satisfaction_delta (polarity shift when present vs absent).
    """

    def classify(self, infons: list[Infon], schema: AnchorSchema,
                 anchor_type: str | None = None) -> list[KanoResult]:
        anchor_docs = defaultdict(set)       # anchor → {doc_ids where it appears}
        anchor_positive = defaultdict(int)   # anchor → count of polarity=1
        anchor_negative = defaultdict(int)   # anchor → count of polarity=0
        all_docs = set()

        for inf in infons:
            all_docs.add(inf.doc_id)
            for role_anchor in [inf.subject, inf.predicate, inf.object]:
                if anchor_type and schema.types.get(role_anchor) != anchor_type:
                    continue
                anchor_docs[role_anchor].add(inf.doc_id)
                if inf.polarity == 1:
                    anchor_positive[role_anchor] += 1
                else:
                    anchor_negative[role_anchor] += 1

        n_docs = max(len(all_docs), 1)
        results = []

        for anchor in sorted(anchor_docs.keys()):
            adoption = len(anchor_docs[anchor]) / n_docs
            pos = anchor_positive[anchor]
            neg = anchor_negative[anchor]
            total = pos + neg
            sat_delta = (pos - neg) / max(total, 1)

            kano_class = self._classify_single(adoption, sat_delta)
            score = sat_delta * (1 - abs(0.5 - adoption))  # peak near 50% adoption

            results.append(KanoResult(
                anchor=anchor,
                kano_class=kano_class,
                adoption_rate=round(adoption, 4),
                satisfaction_delta=round(sat_delta, 4),
                score=round(score, 4),
            ))

        results.sort(key=lambda r: -abs(r.score))
        return results

    def _classify_single(self, adoption: float, sat_delta: float) -> str:
        if adoption >= 0.9:
            return "must_be"
        if adoption < 0.05:
            return "indifferent"
        if sat_delta < -0.3:
            return "reverse"
        if adoption >= 0.3:
            return "one_dimensional"
        return "attractive"


# ═══════════════════════════════════════════════════════════════════════
# 2. CONJOINT ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class ConjointEstimator:
    """Estimate conjoint part-worth utilities from infon patterns.

    In a traditional conjoint, you'd run a survey with product profiles
    and estimate utilities via logistic regression. Here, the hypergraph
    IS the revealed preference: which features co-occur with positive
    polarity in high-importance infons.

    Part-worth utility ≈ mean(importance × polarity_sign) for infons
    mentioning this anchor. Importance captures both the structural
    (NPMI coherence, SPLADE activation) and temporal (reinforcement,
    recency) signals.
    """

    def estimate(self, infons: list[Infon], schema: AnchorSchema,
                 anchor_type: str | None = None) -> list[ConjointResult]:
        anchor_utils = defaultdict(list)

        for inf in infons:
            sign = 1.0 if inf.polarity == 1 else -1.0
            val = inf.importance * sign * inf.confidence
            for role_anchor in [inf.subject, inf.predicate, inf.object]:
                if anchor_type and schema.types.get(role_anchor) != anchor_type:
                    continue
                anchor_utils[role_anchor].append(val)

        results = []
        all_utilities = []
        for anchor, vals in sorted(anchor_utils.items()):
            utility = sum(vals) / max(len(vals), 1)
            all_utilities.append(abs(utility))
            results.append(ConjointResult(
                anchor=anchor,
                utility=round(utility, 4),
                importance=0.0,
            ))

        # Normalize importance to [0, 1]
        max_u = max(all_utilities) if all_utilities else 1.0
        for r in results:
            r.importance = round(abs(r.utility) / max(max_u, 1e-6), 4)

        results.sort(key=lambda r: -r.importance)
        return results


# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE GAP FUNCTOR
# ═══════════════════════════════════════════════════════════════════════

class FeatureGapFunctor:
    """Compute F: Engineering → Perception and measure the distortion.

    The functor maps spec features (what a product HAS) to discourse
    features (what the market DISCUSSES). The gap between these sets
    reveals silent value (spec - discourse) and hallucinated value
    (discourse - spec).

    In the infon graph, we can detect this by comparing:
    - Spec set: anchors present in infons with HAS_FEATURE or direct
      support type (the product literally has it)
    - Discourse set: anchors that appear in signal-linked infons
      (the market discusses it)

    If no explicit spec/discourse labels exist, we approximate:
    - spec_anchors = anchors in the schema definition (the ontology says it exists)
    - discourse_anchors = anchors that actually fire on ingested documents
    """

    def compute(self, infons: list[Infon], schema: AnchorSchema,
                spec_anchors: set[str] | None = None,
                discourse_anchors: set[str] | None = None,
                target_type: str = "feature") -> list[FeatureGapResult]:
        # Build discourse set from actual infon mentions
        anchor_mentions = defaultdict(int)
        anchor_max_score = defaultdict(float)
        for inf in infons:
            for role_anchor in [inf.subject, inf.predicate, inf.object]:
                if schema.types.get(role_anchor) == target_type or not target_type:
                    anchor_mentions[role_anchor] += 1
                    anchor_max_score[role_anchor] = max(
                        anchor_max_score[role_anchor], inf.confidence)

        if discourse_anchors is None:
            discourse_anchors = set(anchor_mentions.keys())

        if spec_anchors is None:
            spec_anchors = {n for n in schema.names
                           if schema.types.get(n) == target_type}

        all_anchors = spec_anchors | discourse_anchors
        results = []

        for anchor in sorted(all_anchors):
            in_spec = anchor in spec_anchors
            in_disc = anchor in discourse_anchors
            mentions = anchor_mentions.get(anchor, 0)
            disc_score = anchor_max_score.get(anchor, 0.0)

            if in_spec and in_disc:
                gap_type = "overlap"
            elif in_spec and not in_disc:
                gap_type = "silent"
            elif not in_spec and in_disc:
                gap_type = "hallucinated"
            else:
                gap_type = "absent"

            results.append(FeatureGapResult(
                anchor=anchor,
                spec_present=in_spec,
                discourse_present=in_disc,
                gap_type=gap_type,
                spec_score=1.0 if in_spec else 0.0,
                discourse_score=round(disc_score, 4),
                discourse_count=mentions,
            ))

        results.sort(key=lambda r: -(r.discourse_count + (10 if r.gap_type in ("silent", "hallucinated") else 0)))
        return results

    def overlap_ratio(self, results: list[FeatureGapResult]) -> float:
        overlap = sum(1 for r in results if r.gap_type == "overlap")
        total = max(len(results), 1)
        return round(overlap / total, 4)

    def summary(self, results: list[FeatureGapResult]) -> dict:
        counts = Counter(r.gap_type for r in results)
        total = max(len(results), 1)
        return {
            "total_features": len(results),
            "overlap": counts.get("overlap", 0),
            "silent": counts.get("silent", 0),
            "hallucinated": counts.get("hallucinated", 0),
            "overlap_pct": round(counts.get("overlap", 0) / total * 100, 1),
            "silent_pct": round(counts.get("silent", 0) / total * 100, 1),
            "hallucinated_pct": round(counts.get("hallucinated", 0) / total * 100, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. GHOST DETECTOR
# ═══════════════════════════════════════════════════════════════════════

class GhostDetector:
    """Detect anchors with zero market signal (ghosts).

    A ghost is an anchor that exists in the schema but has zero infons
    referencing it. In product terms: it was engineered, certified, and
    dealer-stocked, but generates zero market conversation. CAC → ∞.

    From the graph Laplacian perspective, ghost nodes are disconnected
    (Fiedler vector component = 0) — they contribute nothing to the
    algebraic connectivity of the product-signal bipartite graph.
    """

    def detect(self, infons: list[Infon], schema: AnchorSchema,
               anchor_type: str | None = None) -> list[GhostResult]:
        anchor_counts = Counter()
        anchor_effort = defaultdict(float)

        for inf in infons:
            for role_anchor in [inf.subject, inf.predicate, inf.object]:
                anchor_counts[role_anchor] += 1
                anchor_effort[role_anchor] += inf.reinforcement_count

        results = []
        for name in schema.names:
            atype = schema.types.get(name, "")
            if anchor_type and atype != anchor_type:
                continue
            count = anchor_counts.get(name, 0)
            effort = anchor_effort.get(name, 0.0)
            results.append(GhostResult(
                anchor=name,
                anchor_type=atype,
                signal_count=count,
                is_ghost=(count == 0),
                carrying_cost_proxy=effort,
            ))

        results.sort(key=lambda r: (r.signal_count, -r.carrying_cost_proxy))
        return results

    def ghost_rate(self, results: list[GhostResult]) -> float:
        ghosts = sum(1 for r in results if r.is_ghost)
        return round(ghosts / max(len(results), 1), 4)

    def by_type(self, results: list[GhostResult]) -> dict[str, dict]:
        type_groups = defaultdict(lambda: {"total": 0, "ghosts": 0})
        for r in results:
            type_groups[r.anchor_type]["total"] += 1
            if r.is_ghost:
                type_groups[r.anchor_type]["ghosts"] += 1
        return {
            k: {**v, "ghost_pct": round(v["ghosts"] / max(v["total"], 1) * 100, 1)}
            for k, v in sorted(type_groups.items())
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. POLARIZATION INDEX
# ═══════════════════════════════════════════════════════════════════════

class PolarizationIndex:
    """Measure sentiment contradiction density per anchor.

    A sheaf assigns sentiment (bullish/bearish) to local patches (individual
    signals about a product). The gluing axiom asks: can local observations
    be assembled into one consistent global view?

    H¹ ≈ min(bull, bear) — the irreducible contradiction that can't be
    resolved by majority vote. High H¹ = fractured brand identity or
    strategic polarization.
    """

    def compute(self, infons: list[Infon], schema: AnchorSchema,
                anchor_type: str | None = None) -> list[PolarizationResult]:
        anchor_bull = defaultdict(int)
        anchor_bear = defaultdict(int)

        for inf in infons:
            for role_anchor in [inf.subject, inf.predicate, inf.object]:
                if anchor_type and schema.types.get(role_anchor) != anchor_type:
                    continue
                if inf.polarity == 1:
                    anchor_bull[role_anchor] += 1
                else:
                    anchor_bear[role_anchor] += 1

        all_anchors = set(anchor_bull.keys()) | set(anchor_bear.keys())
        results = []

        for anchor in sorted(all_anchors):
            bull = anchor_bull.get(anchor, 0)
            bear = anchor_bear.get(anchor, 0)
            total = bull + bear
            h1 = min(bull, bear)
            ratio = h1 / max(total, 1)

            if total == 0:
                direction = "silent"
            elif bear == 0:
                direction = "positive"
            elif bull == 0:
                direction = "negative"
            elif abs(bull - bear) <= max(total * 0.1, 1):
                direction = "balanced"
            elif bull > bear:
                direction = "positive"
            else:
                direction = "negative"

            results.append(PolarizationResult(
                anchor=anchor,
                bull_count=bull,
                bear_count=bear,
                total_count=total,
                h1=h1,
                polarization_ratio=round(ratio, 4),
                direction=direction,
            ))

        results.sort(key=lambda r: -r.h1)
        return results


# ═══════════════════════════════════════════════════════════════════════
# 6. NARRATIVE ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class NarrativeAnalyzer:
    """Analyze narrative lifecycle via persistent homology on temporal chains.

    Treats Signal --NEXT--> Signal chains as a directed simplicial complex.
    As the time filtration increases:
    - β₀ (connected components): coverage bursts — high β₀ = fragmented
    - β₁ (loops/cycles): recurring narrative — long-lived cycle = keeps coming back

    A real market shift shows: high β₀ early → low β₀ (consensus) with persistent β₁.
    A fad shows: β₀ spike → dies → no β₁.
    """

    def analyze(self, infons: list[Infon], edges: list[Edge],
                schema: AnchorSchema,
                anchor_type: str | None = None) -> list[NarrativeResult]:
        # Build per-anchor NEXT chains
        anchor_chain_edges = defaultdict(list)
        for e in edges:
            if e.edge_type != "NEXT":
                continue
            anchor = e.metadata.get("anchor", "")
            if anchor_type and schema.types.get(anchor) != anchor_type:
                continue
            if anchor:
                anchor_chain_edges[anchor].append(e)

        # Build per-anchor temporal infon sequences
        anchor_infons = defaultdict(list)
        for inf in infons:
            if not inf.timestamp:
                continue
            for role_anchor in [inf.subject, inf.predicate, inf.object]:
                if anchor_type and schema.types.get(role_anchor) != anchor_type:
                    continue
                anchor_infons[role_anchor].append(inf)

        results = []
        for anchor in sorted(set(anchor_chain_edges.keys()) | set(anchor_infons.keys())):
            chain_edges = anchor_chain_edges.get(anchor, [])
            infon_seq = sorted(anchor_infons.get(anchor, []),
                               key=lambda i: i.timestamp or "")

            chain_length = len(chain_edges)

            # Gap distribution
            gaps = []
            for e in chain_edges:
                g = e.metadata.get("gap_days", 0)
                if isinstance(g, (int, float)):
                    gaps.append(int(g))

            gap_dist = {"0-7": 0, "8-30": 0, "31-60": 0, "61-90": 0, "90+": 0}
            for g in gaps:
                if g <= 7:
                    gap_dist["0-7"] += 1
                elif g <= 30:
                    gap_dist["8-30"] += 1
                elif g <= 60:
                    gap_dist["31-60"] += 1
                elif g <= 90:
                    gap_dist["61-90"] += 1
                else:
                    gap_dist["90+"] += 1

            # β₀: approximate connected components from gap analysis
            # Large gaps (>30 days) fragment the narrative into separate bursts
            large_gaps = sum(1 for g in gaps if g > 30)
            beta0 = 1.0 + large_gaps  # at least 1 component

            # β₁: approximate persistent loops from recurring anchors
            # If the same anchor appears in multiple time windows, it's a cycle
            time_windows = set()
            for inf in infon_seq:
                if inf.timestamp:
                    time_windows.add(inf.timestamp[:7])  # monthly
            beta1 = max(0, len(time_windows) - chain_length * 0.5) if chain_length > 0 else 0

            # Phase detection
            phase = self._detect_phase(infon_seq, gaps)

            # Velocity: peak signals per month
            monthly = Counter()
            for inf in infon_seq:
                if inf.timestamp:
                    monthly[inf.timestamp[:7]] += 1
            velocity = max(monthly.values()) if monthly else 0

            results.append(NarrativeResult(
                anchor=anchor,
                chain_length=chain_length,
                beta0=round(beta0, 2),
                beta1=round(max(beta1, 0), 2),
                gap_distribution=gap_dist,
                phase=phase,
                velocity=velocity,
            ))

        results.sort(key=lambda r: -r.chain_length)
        return results

    def _detect_phase(self, infons: list[Infon], gaps: list[int]) -> str:
        if not infons:
            return "dead"

        n = len(infons)
        if n < 3:
            return "emerging"

        # Split into halves and compare density
        mid = n // 2
        first_half = infons[:mid]
        second_half = infons[mid:]

        if not first_half or not second_half:
            return "unknown"

        try:
            t0 = datetime.strptime(first_half[0].timestamp[:10], "%Y-%m-%d")
            t_mid = datetime.strptime(first_half[-1].timestamp[:10], "%Y-%m-%d")
            t_end = datetime.strptime(second_half[-1].timestamp[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return "unknown"

        d1 = max((t_mid - t0).days, 1)
        d2 = max((t_end - t_mid).days, 1)
        density1 = len(first_half) / d1
        density2 = len(second_half) / d2

        if density2 > density1 * 1.5:
            return "ramping"
        if density2 < density1 * 0.3:
            return "decaying"
        if density1 > 0.01 and density2 > 0.01:
            return "mature"
        return "emerging"


# ═══════════════════════════════════════════════════════════════════════
# 7. CONTAGION ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class ContagionAnalyzer:
    """Map risk propagation: bearish signal → direct hit → competitor opportunity.

    The full contagion path is 4 hops in the general case:
      Actor --EMITS--> Signal(bearish) --ABOUT--> Product(hit)
        --COMPETES_WITH--> Product(opportunity) --MADE_BY--> Actor(beneficiary)

    In the infon graph, we approximate this:
    - Bearish = polarity=0 infons about an anchor
    - Direct hit = the anchor itself
    - Competitors = other anchors of the same type sharing predicates
    - Beneficiaries = actors in competing infons with polarity=1
    """

    def analyze(self, infons: list[Infon], schema: AnchorSchema,
                anchor_type: str | None = None) -> list[ContagionResult]:
        # Count bearish signals per anchor
        anchor_bearish = defaultdict(list)
        anchor_bullish = defaultdict(list)

        for inf in infons:
            targets = []
            if anchor_type:
                for a in [inf.subject, inf.predicate, inf.object]:
                    if schema.types.get(a) == anchor_type:
                        targets.append(a)
            else:
                targets = [inf.subject, inf.object]

            for anchor in targets:
                if inf.polarity == 0:
                    anchor_bearish[anchor].append(inf)
                else:
                    anchor_bullish[anchor].append(inf)

        # Find competitors: anchors of same type sharing a predicate
        pred_anchors = defaultdict(set)
        for inf in infons:
            for a in [inf.subject, inf.object]:
                atype = schema.types.get(a, "")
                if anchor_type and atype != anchor_type:
                    continue
                pred_anchors[(inf.predicate, atype)].add(a)

        results = []
        for anchor, bearish_infons in sorted(anchor_bearish.items()):
            atype = schema.types.get(anchor, "")
            n_bearish = len(bearish_infons)

            # Find direct competitors via shared predicates
            competitors = set()
            for inf in bearish_infons:
                key = (inf.predicate, atype)
                for comp in pred_anchors.get(key, set()):
                    if comp != anchor:
                        competitors.add(comp)

            # Beneficiaries: competitors with bullish signals
            beneficiaries = []
            for comp in competitors:
                if anchor_bullish.get(comp):
                    beneficiaries.append(comp)

            # Contagion score: bearish density × fan-out
            fan_out = len(competitors)
            contagion_score = n_bearish * math.log1p(fan_out)

            results.append(ContagionResult(
                anchor=anchor,
                bearish_signals=n_bearish,
                direct_hits=[anchor],
                fan_out_degree=fan_out,
                beneficiaries=sorted(beneficiaries),
                contagion_score=round(contagion_score, 4),
            ))

        results.sort(key=lambda r: -r.contagion_score)
        return results


# ═══════════════════════════════════════════════════════════════════════
# 8. KAN EXTENSION
# ═══════════════════════════════════════════════════════════════════════

class KanExtension:
    """Market-transfer estimates via left and right Kan extensions.

    A Kan extension is: given partial data in context A, what's the best
    completion for context B?

    Left Kan extension (optimistic): extend the signal profile from
    source → target along the functor mapping shared anchors. Where
    data exists in both contexts, use target data. Where only source
    data exists, project the source value.

    Right Kan extension (conservative): where source data doesn't
    transfer, assume the worst case (indifferent / zero signal).

    The gap between left and right = uncertainty budget for entering
    a new market/segment.

    In infon terms:
    - Context = a predicate filter (e.g., doc_id prefix, location, segment)
    - Anchors = features/products
    - Signal profile = {anchor: importance_sum} per context
    """

    def extend(self, infons: list[Infon], schema: AnchorSchema,
               source_filter: callable,
               target_filter: callable,
               source_label: str = "source",
               target_label: str = "target") -> KanExtensionResult:
        source_profile = defaultdict(float)
        target_profile = defaultdict(float)

        for inf in infons:
            anchors = [inf.subject, inf.predicate, inf.object]
            if source_filter(inf):
                for a in anchors:
                    source_profile[a] += inf.importance
            if target_filter(inf):
                for a in anchors:
                    target_profile[a] += inf.importance

        source_anchors = set(source_profile.keys())
        target_anchors = set(target_profile.keys())
        shared = source_anchors & target_anchors
        source_only = source_anchors - target_anchors
        target_only = target_anchors - source_anchors

        # Left Kan: optimistic — project source values into target gaps
        left = {}
        for a in shared:
            left[a] = target_profile[a]
        for a in source_only:
            # Scale by ratio of shared anchor means
            if shared:
                src_shared_mean = sum(source_profile[s] for s in shared) / len(shared)
                tgt_shared_mean = sum(target_profile[s] for s in shared) / len(shared)
                ratio = tgt_shared_mean / max(src_shared_mean, 1e-6)
                left[a] = source_profile[a] * ratio
            else:
                left[a] = source_profile[a]
        for a in target_only:
            left[a] = target_profile[a]

        # Right Kan: conservative — assume zero for untransferrable
        right = {}
        for a in shared:
            right[a] = min(source_profile[a], target_profile[a])
        for a in source_only:
            right[a] = 0.0  # no evidence in target — assume worst
        for a in target_only:
            right[a] = target_profile[a]

        # Uncertainty budget: left - right
        uncertainty = {}
        for a in set(left.keys()) | set(right.keys()):
            uncertainty[a] = round(left.get(a, 0) - right.get(a, 0), 4)

        return KanExtensionResult(
            source_context=source_label,
            target_context=target_label,
            left_estimate={k: round(v, 4) for k, v in sorted(left.items())},
            right_estimate={k: round(v, 4) for k, v in sorted(right.items())},
            uncertainty_budget={k: v for k, v in sorted(uncertainty.items(), key=lambda x: -x[1])},
            shared_anchors=sorted(shared),
            source_only=sorted(source_only),
            target_only=sorted(target_only),
        )


# ═══════════════════════════════════════════════════════════════════════
# 9. DRIVER TREE
# ═══════════════════════════════════════════════════════════════════════

class DriverTree:
    """Build a composable driver tree from structural analysis results.

    The tree decomposes a top-level metric into causal/compositional branches,
    each computed by a structural engine. Every leaf is directly computable
    from the hypergraph topology — no financial data needed for the structure.

    Plug in actual costs per unit to get dollar values; the shape and
    relative weights come from the topology.
    """

    def build(self,
              infons: list[Infon],
              kano: list[KanoResult] | None = None,
              conjoint: list[ConjointResult] | None = None,
              gaps: list[FeatureGapResult] | None = None,
              ghosts: list[GhostResult] | None = None,
              polarization: list[PolarizationResult] | None = None,
              narrative: list[NarrativeResult] | None = None,
              contagion: list[ContagionResult] | None = None,
              ) -> DriverNode:

        root = DriverNode(name="Portfolio Value", path="root", unit="index")

        # ── Revenue branch ─────────────────────────────────────────
        revenue = DriverNode(name="Revenue", path="root.revenue", unit="index")

        # Demand: signal count = attention = consideration
        total_signals = len(infons)
        organic_signals = sum(1 for i in infons if i.polarity == 1)
        paid_proxy = total_signals - organic_signals

        demand = DriverNode(name="Demand", path="root.revenue.demand",
                            value=math.log1p(total_signals), unit="log_signals",
                            source_engine="store")
        demand.children = [
            DriverNode(name="Organic", path="root.revenue.demand.organic",
                       value=math.log1p(organic_signals), unit="log_signals",
                       source_engine="polarization"),
            DriverNode(name="Paid Proxy", path="root.revenue.demand.paid",
                       value=math.log1p(paid_proxy), unit="log_signals",
                       source_engine="ghost_detector"),
        ]

        # Conversion: Kano must-be coverage
        if kano:
            must_be = [k for k in kano if k.kano_class == "must_be"]
            total_kano = max(len(kano), 1)
            must_be_coverage = len(must_be) / total_kano
        else:
            must_be_coverage = 0.0

        conversion = DriverNode(name="Conversion", path="root.revenue.conversion",
                                value=round(must_be_coverage, 4), unit="ratio",
                                source_engine="kano")

        # Pricing Power: attractive feature count × exclusivity
        if kano:
            attractive = [k for k in kano if k.kano_class == "attractive"]
            pricing_power = len(attractive) * (1.0 / max(len(kano), 1))
        else:
            pricing_power = 0.0

        pricing = DriverNode(name="Pricing Power", path="root.revenue.pricing",
                             value=round(pricing_power, 4), unit="index",
                             source_engine="kano+conjoint")

        revenue.children = [demand, conversion, pricing]
        revenue.value = round(demand.value * (0.3 + 0.7 * must_be_coverage), 4)

        # ── Cost branch ────────────────────────────────────────────
        cost = DriverNode(name="Cost", path="root.cost", unit="index")

        # Engineering waste: silent features
        if gaps:
            gap_summary = FeatureGapFunctor().summary(gaps)
            silent_pct = gap_summary["silent_pct"] / 100.0
        else:
            silent_pct = 0.0

        engineering = DriverNode(name="Engineering Waste", path="root.cost.engineering",
                                 value=round(silent_pct, 4), unit="ratio",
                                 source_engine="feature_gap_functor")

        # Marketing cost: H¹ × polarization direction
        if polarization:
            avg_h1 = sum(p.h1 for p in polarization) / max(len(polarization), 1)
            neg_polar = sum(1 for p in polarization if p.direction == "negative")
            marketing_cost = avg_h1 * (1 + neg_polar * 0.5)
        else:
            marketing_cost = 0.0

        marketing = DriverNode(name="Marketing Burden", path="root.cost.marketing",
                               value=round(marketing_cost, 4), unit="h1_index",
                               source_engine="polarization")

        # Portfolio drag: ghost products × carrying cost
        if ghosts:
            ghost_count = sum(1 for g in ghosts if g.is_ghost)
            drag = ghost_count * max(sum(g.carrying_cost_proxy for g in ghosts if g.is_ghost), 1)
        else:
            ghost_count = 0
            drag = 0.0

        portfolio_drag = DriverNode(name="Portfolio Drag", path="root.cost.portfolio_drag",
                                    value=round(math.log1p(drag), 4), unit="log_cost",
                                    source_engine="ghost_detector")

        cost.children = [engineering, marketing, portfolio_drag]
        cost.value = round(engineering.value + marketing.value * 0.1 + portfolio_drag.value * 0.01, 4)

        # ── Risk branch ────────────────────────────────────────────
        risk = DriverNode(name="Risk", path="root.risk", unit="index")

        # Contagion exposure
        if contagion:
            max_contagion = max((c.contagion_score for c in contagion), default=0)
            total_bearish = sum(c.bearish_signals for c in contagion)
        else:
            max_contagion = 0.0
            total_bearish = 0

        contagion_node = DriverNode(name="Contagion Exposure", path="root.risk.contagion",
                                    value=round(max_contagion, 4), unit="score",
                                    source_engine="contagion")

        # Concentration: inverse of unique anchor diversity
        unique_subjects = len(set(i.subject for i in infons))
        concentration = 1.0 / max(unique_subjects, 1)

        concentration_node = DriverNode(name="Concentration", path="root.risk.concentration",
                                        value=round(concentration, 4), unit="inverse_diversity",
                                        source_engine="store")

        # Narrative decay
        if narrative:
            decaying = sum(1 for n in narrative if n.phase == "decaying")
            decay_ratio = decaying / max(len(narrative), 1)
        else:
            decay_ratio = 0.0

        decay_node = DriverNode(name="Narrative Decay", path="root.risk.narrative_decay",
                                value=round(decay_ratio, 4), unit="ratio",
                                source_engine="narrative")

        risk.children = [contagion_node, concentration_node, decay_node]
        risk.value = round(contagion_node.value * 0.01 + concentration_node.value + decay_node.value, 4)

        # ── Assemble root ──────────────────────────────────────────
        root.children = [revenue, cost, risk]
        root.value = round(revenue.value - cost.value - risk.value * 0.5, 4)

        return root


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class StructuralAnalyzer:
    """Orchestrates all structural engines and enriches infons with metrics.

    Usage:
        sa = StructuralAnalyzer(schema)
        results = sa.run_all(infons, edges)
        enriched = sa.enrich(infons, results)
        tree = sa.driver_tree(infons, results)
    """

    def __init__(self, schema: AnchorSchema):
        self.schema = schema
        self.kano = KanoClassifier()
        self.conjoint = ConjointEstimator()
        self.feature_gap = FeatureGapFunctor()
        self.ghost = GhostDetector()
        self.polarization = PolarizationIndex()
        self.narrative = NarrativeAnalyzer()
        self.contagion = ContagionAnalyzer()
        self.kan = KanExtension()
        self.driver = DriverTree()

    def run_all(self, infons: list[Infon],
                edges: list[Edge] | None = None,
                spec_anchors: set[str] | None = None) -> dict:
        """Run all structural engines. Returns dict of engine_name → results."""
        edges = edges or []
        return {
            "kano": self.kano.classify(infons, self.schema),
            "conjoint": self.conjoint.estimate(infons, self.schema),
            "feature_gap": self.feature_gap.compute(infons, self.schema,
                                                     spec_anchors=spec_anchors),
            "ghosts": self.ghost.detect(infons, self.schema),
            "polarization": self.polarization.compute(infons, self.schema),
            "narrative": self.narrative.analyze(infons, edges, self.schema),
            "contagion": self.contagion.analyze(infons, self.schema),
        }

    def enrich(self, infons: list[Infon],
               results: dict | None = None,
               edges: list[Edge] | None = None) -> list[Infon]:
        """Populate infon.metrics dict from structural analysis results.

        Modifies infons in-place and returns them.
        """
        if results is None:
            results = self.run_all(infons, edges)

        # Build lookup indexes from results
        kano_map = {r.anchor: r for r in results.get("kano", [])}
        conjoint_map = {r.anchor: r for r in results.get("conjoint", [])}
        gap_map = {r.anchor: r for r in results.get("feature_gap", [])}
        ghost_map = {r.anchor: r for r in results.get("ghosts", [])}
        polar_map = {r.anchor: r for r in results.get("polarization", [])}
        narrative_map = {r.anchor: r for r in results.get("narrative", [])}
        contagion_map = {r.anchor: r for r in results.get("contagion", [])}

        for inf in infons:
            m = inf.metrics if inf.metrics else {}
            anchors = [inf.subject, inf.predicate, inf.object]

            # Kano — use the object anchor (typically the feature)
            for a in anchors:
                k = kano_map.get(a)
                if k:
                    m["kano_class"] = k.kano_class
                    m["kano_score"] = k.score
                    m["kano_adoption"] = k.adoption_rate
                    break

            # Conjoint — use highest-importance anchor
            best_conjoint = None
            for a in anchors:
                c = conjoint_map.get(a)
                if c and (best_conjoint is None or c.importance > best_conjoint.importance):
                    best_conjoint = c
            if best_conjoint:
                m["conjoint_utility"] = best_conjoint.utility
                m["conjoint_importance"] = best_conjoint.importance

            # Feature gap — use object anchor
            for a in anchors:
                g = gap_map.get(a)
                if g:
                    m["gap_type"] = g.gap_type
                    m["spec_present"] = g.spec_present
                    m["discourse_present"] = g.discourse_present
                    m["signal_count"] = g.discourse_count
                    break

            # Ghost
            for a in anchors:
                gh = ghost_map.get(a)
                if gh:
                    m["ghost"] = gh.is_ghost
                    break

            # Polarization — aggregate across anchors
            total_bull = 0
            total_bear = 0
            for a in anchors:
                p = polar_map.get(a)
                if p:
                    total_bull += p.bull_count
                    total_bear += p.bear_count
            m["bull_count"] = total_bull
            m["bear_count"] = total_bear
            m["polarization_h1"] = min(total_bull, total_bear)

            # Narrative — use the anchor with longest chain
            best_nar = None
            for a in anchors:
                n = narrative_map.get(a)
                if n and (best_nar is None or n.chain_length > best_nar.chain_length):
                    best_nar = n
            if best_nar:
                m["narrative_beta0"] = best_nar.beta0
                m["narrative_beta1"] = best_nar.beta1
                m["narrative_phase"] = best_nar.phase

            # Contagion — use anchor with highest score
            best_cont = None
            for a in anchors:
                ct = contagion_map.get(a)
                if ct and (best_cont is None or ct.contagion_score > best_cont.contagion_score):
                    best_cont = ct
            if best_cont:
                m["contagion_exposure"] = best_cont.contagion_score
                m["contagion_fan_out"] = best_cont.fan_out_degree

            inf.metrics = m

        return infons

    def driver_tree(self, infons: list[Infon],
                    results: dict | None = None,
                    edges: list[Edge] | None = None) -> DriverNode:
        """Build the full driver tree from structural analysis results."""
        if results is None:
            results = self.run_all(infons, edges)

        return self.driver.build(
            infons=infons,
            kano=results.get("kano"),
            conjoint=results.get("conjoint"),
            gaps=results.get("feature_gap"),
            ghosts=results.get("ghosts"),
            polarization=results.get("polarization"),
            narrative=results.get("narrative"),
            contagion=results.get("contagion"),
        )

    def summary(self, results: dict) -> dict:
        """Compact summary across all engines for dashboarding."""
        kano = results.get("kano", [])
        ghosts = results.get("ghosts", [])
        gaps = results.get("feature_gap", [])
        polarization = results.get("polarization", [])
        narrative = results.get("narrative", [])
        contagion = results.get("contagion", [])
        conjoint = results.get("conjoint", [])

        kano_dist = Counter(k.kano_class for k in kano)

        return {
            "kano_distribution": dict(kano_dist),
            "kano_must_be_count": kano_dist.get("must_be", 0),
            "kano_attractive_count": kano_dist.get("attractive", 0),
            "conjoint_top3": [(c.anchor, c.utility, c.importance) for c in conjoint[:3]],
            "feature_gap": self.feature_gap.summary(gaps) if gaps else {},
            "ghost_rate": self.ghost.ghost_rate(ghosts),
            "ghost_count": sum(1 for g in ghosts if g.is_ghost),
            "ghost_by_type": self.ghost.by_type(ghosts),
            "max_polarization_h1": max((p.h1 for p in polarization), default=0),
            "polarized_anchors": [p.anchor for p in polarization if p.h1 > 0][:10],
            "narrative_phases": Counter(n.phase for n in narrative),
            "max_chain_length": max((n.chain_length for n in narrative), default=0),
            "max_contagion_score": max((c.contagion_score for c in contagion), default=0),
            "total_contagion_fan_out": sum(c.fan_out_degree for c in contagion),
        }
