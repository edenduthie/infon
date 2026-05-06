"""Experiment 5: Temporal Kan Extension Predictor.

Uses NEXT chains and the sheaf NPMI co-activation matrix to predict
which triples are most likely to appear NEXT beyond the observed data.
This is "structural autocomplete" for geopolitical events.

The algorithm:
  1. Build infons and NEXT edges from a 75% training split of the corpus.
  2. Fit a SheafCoherence model on the training sentences.
  3. Find the frontier — the most recent infon per NEXT chain (per anchor).
  4. For each frontier infon, use NPMI neighborhoods to generate candidate
     future triples, scored by coherence, recency, and chain momentum.
  5. Evaluate predictions against a 25% holdout split.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np

from infon import (
    AnchorSchema, Encoder, InfonConfig, extract_infons, split_sentences,
)
from infon.category import SheafCoherence
from infon.consolidate import build_next_edges


# ═══════════════════════════════════════════════════════════════════════
# DATA — 30-anchor geopolitical schema and 24-document corpus
# ═══════════════════════════════════════════════════════════════════════

GEO_SCHEMA = {
    # Actors
    "china":         {"type": "actor", "tokens": ["china", "chinese", "beijing"], "country_code": "CN"},
    "us":            {"type": "actor", "tokens": ["united states", "us", "washington", "american"], "country_code": "US"},
    "israel":        {"type": "actor", "tokens": ["israel", "israeli"], "country_code": "IL"},
    "palestine":     {"type": "actor", "tokens": ["palestine", "palestinian"], "country_code": "PS"},
    "russia":        {"type": "actor", "tokens": ["russia", "russian", "moscow"], "country_code": "RU"},
    "japan":         {"type": "actor", "tokens": ["japan", "japanese", "tokyo"], "country_code": "JP"},
    "india":         {"type": "actor", "tokens": ["india", "indian"], "country_code": "IN"},
    "iran":          {"type": "actor", "tokens": ["iran", "iranian", "tehran"], "country_code": "IR"},
    "nato":          {"type": "actor", "tokens": ["nato", "alliance"]},
    "un":            {"type": "actor", "tokens": ["un", "united nations", "security council"]},
    "eu":            {"type": "actor", "tokens": ["eu", "european union", "brussels"]},
    "african_union": {"type": "actor", "tokens": ["african union", "au"]},

    # Relations
    "sanction":    {"type": "relation", "tokens": ["sanction", "sanctions", "embargo", "restrict"]},
    "negotiate":   {"type": "relation", "tokens": ["negotiate", "talks", "diplomacy", "summit", "discuss"]},
    "deploy":      {"type": "relation", "tokens": ["deploy", "deployment", "station", "mobilize", "troops"]},
    "cooperate":   {"type": "relation", "tokens": ["cooperate", "cooperation", "collaborate", "partner", "aid"]},
    "condemn":     {"type": "relation", "tokens": ["condemn", "denounce", "protest", "urge", "demand"]},
    "trade":       {"type": "relation", "tokens": ["trade", "tariff", "import", "export", "economic"]},
    "attack":      {"type": "relation", "tokens": ["attack", "strike", "bomb", "clash", "military"]},
    "invest":      {"type": "relation", "tokens": ["invest", "investment", "fund", "billion"]},

    # Features
    "military":    {"type": "feature", "tokens": ["military", "defense", "armed forces", "weapons"]},
    "nuclear":     {"type": "feature", "tokens": ["nuclear", "atomic", "missile", "warhead"]},
    "humanitarian":{"type": "feature", "tokens": ["humanitarian", "aid", "relief", "refugee"]},
    "territory":   {"type": "feature", "tokens": ["territory", "border", "occupation", "settlement"]},
    "maritime":    {"type": "feature", "tokens": ["maritime", "naval", "sea", "coast", "waters"]},
    "technology":  {"type": "feature", "tokens": ["technology", "cyber", "digital", "innovation"]},

    # Markets/regions
    "middle_east": {"type": "market", "tokens": ["middle east", "gulf", "levant"]},
    "east_asia":   {"type": "market", "tokens": ["east asia", "pacific", "indo-pacific", "asia"]},
    "europe":      {"type": "market", "tokens": ["europe", "european"]},
    "africa":      {"type": "market", "tokens": ["africa", "african"]},
}

GEO_DOCS = [
    {"id": "geo-001", "timestamp": "2023-07-20", "text": "Palestine urges Israel to halt settlement expansion and end the occupation of Palestinian territories in the West Bank."},
    {"id": "geo-002", "timestamp": "2017-07-02", "text": "China organized the Disability and Sustainable Development Forum in Beijing with UNESCO cooperation."},
    {"id": "geo-003", "timestamp": "2008-03-26", "text": "African Union troops clashed with forces loyal to Mohamed Bacar on the island of Anjouan in a military operation."},
    {"id": "geo-004", "timestamp": "2019-07-11", "text": "Israel and India launched a joint venture to manufacture missile defense systems through Rafael Advanced Defense Systems."},
    {"id": "geo-005", "timestamp": "2020-08-20", "text": "The United States rejected the UN Security Council snapback mechanism on Iran sanctions in a diplomatic breakdown."},
    {"id": "geo-006", "timestamp": "2024-09-01", "text": "Taiwan plays a critical role in the US-China competition for critical minerals and technology supply chains."},
    {"id": "geo-007", "timestamp": "2022-03-09", "text": "China sent the first batch of humanitarian aid to Ukraine including food and medical supplies."},
    {"id": "geo-008", "timestamp": "2015-07-05", "text": "Greeks voted in a referendum amid pressure from the European Commission over economic bailout conditions."},
    {"id": "geo-009", "timestamp": "2020-01-15", "text": "North Korea launched missile tests near the Korean Peninsula, drawing condemnation from Japan and the United States."},
    {"id": "geo-010", "timestamp": "2018-01-18", "text": "Malaysia and Singapore negotiated over border congestion at the Johor Causeway crossing."},
    {"id": "geo-011", "timestamp": "2023-10-15", "text": "Iran and Russia deepened military cooperation with joint naval exercises in the maritime region."},
    {"id": "geo-012", "timestamp": "2024-03-10", "text": "NATO deployed additional troops to Eastern Europe amid Russian military buildup near the border."},
    {"id": "geo-013", "timestamp": "2021-09-15", "text": "The United States and Japan signed a bilateral trade agreement covering technology and defense cooperation in East Asia."},
    {"id": "geo-014", "timestamp": "2022-06-20", "text": "The African Union condemned the military coup and deployed peacekeeping forces to restore order."},
    {"id": "geo-015", "timestamp": "2023-04-10", "text": "China invested billions in infrastructure across Africa as part of the Belt and Road Initiative."},
    {"id": "geo-016", "timestamp": "2019-12-05", "text": "Iran rejected nuclear inspections demanded by the United Nations Security Council."},
    {"id": "geo-017", "timestamp": "2024-06-15", "text": "The European Union imposed sanctions on Russian energy exports amid the Ukraine conflict."},
    {"id": "geo-018", "timestamp": "2020-09-10", "text": "Israel and the United States signed the Abraham Accords normalizing diplomatic relations in the Middle East."},
    {"id": "geo-019", "timestamp": "2021-03-20", "text": "India deployed naval vessels in the maritime region of East Asia for joint exercises with Japan."},
    {"id": "geo-020", "timestamp": "2023-11-01", "text": "China and the European Union held trade negotiations over tariffs and technology transfer in Brussels."},
    {"id": "geo-021", "timestamp": "2024-01-20", "text": "NATO condemned Russian military deployments near the European border as provocative."},
    {"id": "geo-022", "timestamp": "2022-11-10", "text": "The United Nations delivered humanitarian aid to Afghanistan after the Taliban takeover."},
    {"id": "geo-023", "timestamp": "2004-08-02", "text": "The African Union deployed troops to Darfur in a peacekeeping mission to protect civilians."},
    {"id": "geo-024", "timestamp": "2025-01-01", "text": "China invested in nuclear energy technology cooperation with Iran despite international sanctions."},
]


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def parse_date(ts: str) -> datetime:
    """Parse an ISO date string."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(ts[:len(fmt.replace("%", "0"))], fmt)
        except ValueError:
            continue
    return datetime.strptime(ts[:4], "%Y")


def get_anchor_type(name: str) -> str:
    """Return the type of an anchor from the schema."""
    return GEO_SCHEMA.get(name, {}).get("type", "unknown")


VALID_ROLE_TYPES = {
    "subject":   {"actor"},
    "predicate": {"relation"},
    "object":    {"feature", "market", "actor"},
}


def is_valid_triple(subj: str, pred: str, obj: str) -> bool:
    """Check whether (subj, pred, obj) has valid type assignments."""
    s_type = get_anchor_type(subj)
    p_type = get_anchor_type(pred)
    o_type = get_anchor_type(obj)
    return (
        s_type in VALID_ROLE_TYPES["subject"]
        and p_type in VALID_ROLE_TYPES["predicate"]
        and o_type in VALID_ROLE_TYPES["object"]
    )


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL KAN EXTENSION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════

class TemporalKanPredictor:
    """Predict future triples using NEXT-chain structure and NPMI neighborhoods.

    The temporal right Kan extension lifts the observed NEXT chains to
    predicted continuations by extending along the NPMI co-activation
    functor.
    """

    def __init__(
        self,
        infons,
        next_edges,
        sheaf: SheafCoherence,
        anchor_names: list[str],
        recency_half_life_days: float = 365.0,
        top_n_npmi: int = 5,
    ):
        self.infons = infons
        self.next_edges = next_edges
        self.sheaf = sheaf
        self.anchor_names = anchor_names
        self.name_to_idx = {n: i for i, n in enumerate(anchor_names)}
        self.recency_half_life = recency_half_life_days
        self.top_n = top_n_npmi

        # Index infons by ID for fast lookup
        self.infon_by_id = {inf.infon_id: inf for inf in infons}

        # Parse all timestamps and find the reference date (latest timestamp)
        self.timestamps = {}
        for inf in infons:
            if inf.timestamp:
                self.timestamps[inf.infon_id] = parse_date(inf.timestamp)
        self.reference_date = max(self.timestamps.values()) if self.timestamps else datetime.now()

        # Index: anchor -> list of infon IDs mentioning it (any role)
        self.anchor_to_infons: dict[str, list[str]] = defaultdict(list)
        for inf in infons:
            self.anchor_to_infons[inf.subject].append(inf.infon_id)
            self.anchor_to_infons[inf.predicate].append(inf.infon_id)
            self.anchor_to_infons[inf.object].append(inf.infon_id)

    def find_frontier(self) -> dict[str, list]:
        """Find the frontier: the most recent infon per anchor NEXT chain.

        Returns {anchor: [(infon, role)]} for each anchor's chain terminus.
        """
        # Build adjacency: for each anchor, track which infon IDs have
        # outgoing NEXT edges vs. which have only incoming.
        # The frontier is the set of infon IDs that have no outgoing NEXT
        # edge for that anchor.

        # anchor -> set of infon_ids that have an outgoing NEXT for that anchor
        has_outgoing: dict[str, set] = defaultdict(set)
        # anchor -> set of all infon_ids in the chain
        in_chain: dict[str, set] = defaultdict(set)

        for edge in self.next_edges:
            anchor = edge.metadata.get("anchor", "")
            if not anchor:
                continue
            has_outgoing[anchor].add(edge.source)
            in_chain[anchor].add(edge.source)
            in_chain[anchor].add(edge.target)

        frontier: dict[str, list] = defaultdict(list)
        for anchor, chain_ids in in_chain.items():
            terminal_ids = chain_ids - has_outgoing.get(anchor, set())
            for iid in terminal_ids:
                inf = self.infon_by_id.get(iid)
                if inf:
                    # Determine what role this anchor plays in this infon
                    role = "unknown"
                    if inf.subject == anchor:
                        role = "subject"
                    elif inf.predicate == anchor:
                        role = "predicate"
                    elif inf.object == anchor:
                        role = "object"
                    frontier[anchor].append((inf, role))

        return dict(frontier)

    def chain_momentum(self, anchor: str, window_pct: float = 0.25) -> float:
        """How active has this anchor been in the recent window?

        Count of infons mentioning this anchor in the last window_pct
        of the overall timeline, normalized by total mentions.
        """
        all_dates = sorted(self.timestamps.values())
        if not all_dates:
            return 0.0

        total_span = (all_dates[-1] - all_dates[0]).days
        if total_span <= 0:
            return 1.0

        cutoff_days = total_span * (1.0 - window_pct)
        cutoff_date = all_dates[0] + timedelta(days=cutoff_days)

        infon_ids = self.anchor_to_infons.get(anchor, [])
        total = len(infon_ids)
        if total == 0:
            return 0.0

        recent = sum(
            1 for iid in infon_ids
            if iid in self.timestamps and self.timestamps[iid] >= cutoff_date
        )
        return recent / total

    def recency_score(self, infon) -> float:
        """Score how recent an infon is. More recent = higher score.

        Exponential decay from the reference date with configurable half-life.
        """
        if infon.infon_id not in self.timestamps:
            return 0.5  # unknown date gets middle score
        dt = self.timestamps[infon.infon_id]
        days_ago = (self.reference_date - dt).days
        # Exponential decay
        return math.exp(-0.693 * days_ago / self.recency_half_life)

    def npmi_neighbors(self, anchor: str, role_filter: set[str] | None = None) -> list[tuple[str, float]]:
        """Find the top-N NPMI neighbors of an anchor.

        Optionally filter to anchors of specific types (e.g. only relations).
        """
        if self.sheaf.npmi is None:
            return []

        idx = self.name_to_idx.get(anchor)
        if idx is None:
            return []

        row = self.sheaf.npmi[idx]
        neighbors = []
        for j, val in enumerate(row):
            if j == idx:
                continue
            name = self.anchor_names[j]
            if role_filter and get_anchor_type(name) not in role_filter:
                continue
            if val > 0:
                neighbors.append((name, float(val)))

        neighbors.sort(key=lambda x: -x[1])
        return neighbors[:self.top_n]

    def triple_npmi_coherence(self, subj: str, pred: str, obj: str) -> float:
        """Average pairwise NPMI for a candidate (S, P, O) triple."""
        if self.sheaf.npmi is None:
            return 0.0

        indices = []
        for a in [subj, pred, obj]:
            idx = self.name_to_idx.get(a)
            if idx is not None:
                indices.append(idx)

        if len(indices) < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total += self.sheaf.npmi[indices[i], indices[j]]
                count += 1

        return total / count if count > 0 else 0.0

    def generate_candidates(self, frontier: dict[str, list]) -> list[dict]:
        """Generate candidate future triples from frontier infons.

        For each frontier infon:
          - From its subject: find likely co-activating (pred, obj) pairs
          - From its object: find likely co-activating (subj, pred) pairs
        Filter to valid type combinations.
        """
        candidates = []
        seen = set()

        for anchor, frontier_entries in frontier.items():
            for inf, role in frontier_entries:
                # --- Strategy A: extend from subject anchor ---
                if role in ("subject", "predicate", "object"):
                    subj = inf.subject
                    # Find top relation neighbors of the subject
                    rel_neighbors = self.npmi_neighbors(subj, role_filter={"relation"})
                    # Find top object neighbors of the subject
                    obj_neighbors = self.npmi_neighbors(
                        subj, role_filter={"feature", "market", "actor"}
                    )

                    for pred_name, pred_npmi in rel_neighbors:
                        for obj_name, obj_npmi in obj_neighbors:
                            if not is_valid_triple(subj, pred_name, obj_name):
                                continue
                            key = (subj, pred_name, obj_name)
                            if key in seen:
                                continue
                            seen.add(key)

                            npmi_coh = self.triple_npmi_coherence(subj, pred_name, obj_name)
                            recency = self.recency_score(inf)
                            momentum = self.chain_momentum(subj)

                            candidates.append({
                                "subject": subj,
                                "predicate": pred_name,
                                "object": obj_name,
                                "npmi_coherence": npmi_coh,
                                "recency": recency,
                                "momentum": momentum,
                                "source_infon": inf,
                                "source_anchor": anchor,
                                "source_role": role,
                                "generation_strategy": "extend_subject",
                                "npmi_links": {
                                    f"{subj}->{pred_name}": pred_npmi,
                                    f"{subj}->{obj_name}": obj_npmi,
                                },
                            })

                # --- Strategy B: extend from object anchor ---
                if role in ("subject", "predicate", "object"):
                    obj = inf.object
                    # Find top subject (actor) neighbors of the object
                    subj_neighbors = self.npmi_neighbors(obj, role_filter={"actor"})
                    # Find top relation neighbors of the object
                    rel_neighbors = self.npmi_neighbors(obj, role_filter={"relation"})

                    for subj_name, subj_npmi in subj_neighbors:
                        for pred_name, pred_npmi in rel_neighbors:
                            if not is_valid_triple(subj_name, pred_name, obj):
                                continue
                            key = (subj_name, pred_name, obj)
                            if key in seen:
                                continue
                            seen.add(key)

                            npmi_coh = self.triple_npmi_coherence(subj_name, pred_name, obj)
                            recency = self.recency_score(inf)
                            momentum = self.chain_momentum(obj)

                            candidates.append({
                                "subject": subj_name,
                                "predicate": pred_name,
                                "object": obj,
                                "npmi_coherence": npmi_coh,
                                "recency": recency,
                                "momentum": momentum,
                                "source_infon": inf,
                                "source_anchor": anchor,
                                "source_role": role,
                                "generation_strategy": "extend_object",
                                "npmi_links": {
                                    f"{obj}->{subj_name}": subj_npmi,
                                    f"{obj}->{pred_name}": pred_npmi,
                                },
                            })

        return candidates

    def score_and_rank(self, candidates: list[dict],
                       w_npmi: float = 0.50,
                       w_recency: float = 0.25,
                       w_momentum: float = 0.25) -> list[dict]:
        """Score candidates by weighted combination and rank."""
        for c in candidates:
            # Normalize NPMI coherence from [-1,1] to [0,1]
            npmi_norm = (c["npmi_coherence"] + 1.0) / 2.0
            c["score"] = (
                w_npmi * npmi_norm
                + w_recency * c["recency"]
                + w_momentum * c["momentum"]
            )

        candidates.sort(key=lambda c: -c["score"])

        # Deduplicate: keep highest-scored per triple
        seen = set()
        deduped = []
        for c in candidates:
            key = (c["subject"], c["predicate"], c["object"])
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped

    def predict(self, top_k: int = 15) -> tuple[dict, list[dict]]:
        """Run the full prediction pipeline.

        Returns (frontier_dict, ranked_predictions).
        """
        frontier = self.find_frontier()
        candidates = self.generate_candidates(frontier)
        ranked = self.score_and_rank(candidates)
        return frontier, ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# BASELINE: FREQUENCY IN RECENT WINDOW
# ═══════════════════════════════════════════════════════════════════════

def frequency_baseline(infons, window_pct: float = 0.25, top_k: int = 10) -> list[tuple]:
    """Baseline: most frequent triples in the last window_pct of the timeline.

    Returns [(subject, predicate, object, count), ...] ranked by count.
    """
    dated = [(inf, parse_date(inf.timestamp)) for inf in infons if inf.timestamp]
    if not dated:
        return []

    dated.sort(key=lambda x: x[1])
    total_span = (dated[-1][1] - dated[0][1]).days
    if total_span <= 0:
        # All same date: return all triples
        counter = Counter(inf.triple_key() for inf, _ in dated)
        return [(s, p, o, c) for (s, p, o), c in counter.most_common(top_k)]

    cutoff_days = total_span * (1.0 - window_pct)
    cutoff_date = dated[0][1] + timedelta(days=cutoff_days)

    recent = [inf for inf, dt in dated if dt >= cutoff_date]
    counter = Counter(inf.triple_key() for inf in recent)
    return [(s, p, o, c) for (s, p, o), c in counter.most_common(top_k)]


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run_experiment():
    """Run the temporal Kan extension prediction experiment."""

    print("=" * 72)
    print("  EXPERIMENT 5: Temporal Kan Extension Predictor")
    print("  Structural autocomplete for geopolitical events")
    print("=" * 72)

    # ── Step 0: Split corpus 75/25 by timestamp ─────────────────────
    sorted_docs = sorted(GEO_DOCS, key=lambda d: d["timestamp"])
    split_idx = int(len(sorted_docs) * 0.75)
    train_docs = sorted_docs[:split_idx]
    holdout_docs = sorted_docs[split_idx:]

    print(f"\n--- Corpus split ---")
    print(f"  Total documents:   {len(GEO_DOCS)}")
    print(f"  Training (75%):    {len(train_docs)}  [{train_docs[0]['timestamp']} .. {train_docs[-1]['timestamp']}]")
    print(f"  Holdout (25%):     {len(holdout_docs)}  [{holdout_docs[0]['timestamp']} .. {holdout_docs[-1]['timestamp']}]")

    # ── Step 1: Build schema & encoder ──────────────────────────────
    schema_path = Path(tempfile.mktemp(suffix=".json"))
    schema_path.write_text(json.dumps(GEO_SCHEMA, indent=2))

    schema = AnchorSchema.from_file(schema_path)
    config = InfonConfig(schema_path=str(schema_path))
    encoder = Encoder(schema=schema)

    print(f"\n--- Schema ---")
    print(f"  Anchors: {len(schema.names)}")
    for atype in sorted(set(GEO_SCHEMA[k]["type"] for k in GEO_SCHEMA)):
        names = [k for k, v in GEO_SCHEMA.items() if v["type"] == atype]
        print(f"    {atype:10s}: {len(names):2d}  {names}")

    # ── Step 2: Extract infons from TRAINING set ────────────────────
    print(f"\n--- Extracting infons from training set ---")
    train_infons, train_doc_edges = extract_infons(
        train_docs, encoder, schema, config,
    )
    print(f"  Extracted {len(train_infons)} infons from {len(train_docs)} documents")

    # Also extract holdout infons for evaluation
    holdout_infons, _ = extract_infons(
        holdout_docs, encoder, schema, config,
    )
    holdout_triples = set(inf.triple_key() for inf in holdout_infons)
    print(f"  Holdout: {len(holdout_infons)} infons, {len(holdout_triples)} unique triples")

    # ── Step 3: Build NEXT edges ────────────────────────────────────
    next_edges = build_next_edges(train_infons)
    print(f"\n--- NEXT edges ---")
    print(f"  Built {len(next_edges)} NEXT edges from training infons")

    # Count chains per anchor
    anchors_in_chains = Counter(e.metadata.get("anchor", "") for e in next_edges)
    print(f"  Anchors with chains: {len(anchors_in_chains)}")
    for anchor, count in anchors_in_chains.most_common(10):
        atype = get_anchor_type(anchor)
        print(f"    {atype:10s} {anchor:15s}  {count} edges")

    # ── Step 4: Build sheaf (NPMI co-activation) ────────────────────
    print(f"\n--- Sheaf coherence (NPMI co-activation) ---")
    sheaf = SheafCoherence(encoder.anchor_names)

    train_sentences = []
    for doc in train_docs:
        train_sentences.extend(split_sentences(doc["text"]))

    activations = encoder.encode(train_sentences)
    sheaf.observe(activations)
    sheaf.fit()

    print(f"  Sentences observed: {len(train_sentences)}")
    print(f"  NPMI matrix shape:  {sheaf.npmi.shape}")
    print(f"  Fiedler value:      {sheaf.fiedler_value:.4f}")

    # Show top NPMI pairs
    n_anchors = len(encoder.anchor_names)
    npmi_pairs = []
    for i in range(n_anchors):
        for j in range(i + 1, n_anchors):
            val = sheaf.npmi[i, j]
            if val > 0.05:
                npmi_pairs.append((encoder.anchor_names[i], encoder.anchor_names[j], val))
    npmi_pairs.sort(key=lambda x: -x[2])

    print(f"\n  Top NPMI co-activation pairs:")
    for a, b, v in npmi_pairs[:12]:
        ta = get_anchor_type(a)
        tb = get_anchor_type(b)
        print(f"    {ta:8s} {a:15s} <-> {tb:8s} {b:15s}  NPMI={v:.3f}")

    # ── Step 5: Run the temporal Kan extension predictor ────────────
    print(f"\n{'=' * 72}")
    print(f"  TEMPORAL KAN EXTENSION PREDICTIONS")
    print(f"{'=' * 72}")

    predictor = TemporalKanPredictor(
        infons=train_infons,
        next_edges=next_edges,
        sheaf=sheaf,
        anchor_names=encoder.anchor_names,
        recency_half_life_days=365.0 * 2,
        top_n_npmi=6,
    )

    frontier, predictions = predictor.predict(top_k=15)

    # Print frontier
    print(f"\n--- Frontier (current state per anchor chain) ---")
    frontier_sorted = sorted(
        frontier.items(),
        key=lambda kv: max(
            (predictor.timestamps.get(inf.infon_id, datetime.min) for inf, _ in kv[1]),
            default=datetime.min,
        ),
        reverse=True,
    )
    for anchor, entries in frontier_sorted:
        atype = get_anchor_type(anchor)
        momentum = predictor.chain_momentum(anchor)
        for inf, role in entries:
            ts = inf.timestamp or "?"
            print(f"  {atype:10s} {anchor:15s} [{role:9s}]  "
                  f"{ts}  <<{inf.predicate}, {inf.subject}, {inf.object}>>  "
                  f"momentum={momentum:.2f}")

    # Print predictions
    print(f"\n--- Top {len(predictions)} predicted future triples ---")
    for i, pred in enumerate(predictions, 1):
        s, p, o = pred["subject"], pred["predicate"], pred["object"]
        score = pred["score"]
        npmi_c = pred["npmi_coherence"]
        recency = pred["recency"]
        momentum = pred["momentum"]
        strategy = pred["generation_strategy"]

        # Check if this prediction matches a holdout triple
        match = "MATCH" if (s, p, o) in holdout_triples else ""

        print(f"\n  {i:2d}. <<{p}, {s}, {o}>>")
        print(f"      score={score:.3f}  "
              f"(NPMI_coh={npmi_c:+.3f}  recency={recency:.3f}  momentum={momentum:.3f})")
        print(f"      strategy: {strategy}")

        # Show WHY: which frontier infon and which NPMI links
        src = pred["source_infon"]
        print(f"      source:   <<{src.predicate}, {src.subject}, {src.object}>>  "
              f"[{pred['source_anchor']}/{pred['source_role']}]  "
              f"ts={src.timestamp}")
        links = pred["npmi_links"]
        link_str = "  ".join(f"{k}: {v:.3f}" for k, v in links.items())
        print(f"      NPMI:     {link_str}")
        if match:
            print(f"      >>> {match} in holdout set! <<<")

    # ── Step 6: Baseline comparison ─────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  BASELINE: Most frequent triples in last 25% of training window")
    print(f"{'=' * 72}")

    baseline = frequency_baseline(train_infons, window_pct=0.25, top_k=10)
    for i, (s, p, o, count) in enumerate(baseline, 1):
        match = "MATCH" if (s, p, o) in holdout_triples else ""
        print(f"  {i:2d}. <<{p}, {s}, {o}>>  count={count}  {match}")

    # ── Step 7: Evaluation ──────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  EVALUATION: Predictions vs. holdout triples")
    print(f"{'=' * 72}")

    pred_triples = set((p["subject"], p["predicate"], p["object"]) for p in predictions)
    baseline_triples = set((s, p, o) for s, p, o, _ in baseline)

    kan_hits = pred_triples & holdout_triples
    baseline_hits = baseline_triples & holdout_triples

    print(f"\n  Holdout triples:        {len(holdout_triples)}")
    print(f"  Kan predictions:        {len(pred_triples)}")
    print(f"  Kan hits:               {len(kan_hits)}")
    if kan_hits:
        for s, p, o in sorted(kan_hits):
            print(f"    HIT: <<{p}, {s}, {o}>>")
    kan_precision = len(kan_hits) / len(pred_triples) if pred_triples else 0
    kan_recall = len(kan_hits) / len(holdout_triples) if holdout_triples else 0
    print(f"  Kan precision:          {kan_precision:.3f}")
    print(f"  Kan recall:             {kan_recall:.3f}")

    print(f"\n  Baseline predictions:   {len(baseline_triples)}")
    print(f"  Baseline hits:          {len(baseline_hits)}")
    if baseline_hits:
        for s, p, o in sorted(baseline_hits):
            print(f"    HIT: <<{p}, {s}, {o}>>")
    base_precision = len(baseline_hits) / len(baseline_triples) if baseline_triples else 0
    base_recall = len(baseline_hits) / len(holdout_triples) if holdout_triples else 0
    print(f"  Baseline precision:     {base_precision:.3f}")
    print(f"  Baseline recall:        {base_recall:.3f}")

    # Qualitative analysis
    print(f"\n--- Qualitative analysis ---")
    if predictions:
        top = predictions[0]
        print(f"  Strongest prediction: <<{top['predicate']}, {top['subject']}, {top['object']}>>")
        print(f"    This says: '{top['subject']}' will '{top['predicate']}' "
              f"regarding '{top['object']}'")
        print(f"    Driven by: NPMI co-activation from frontier infon at {top['source_infon'].timestamp}")

    # Show holdout triples that were NOT predicted
    missed = holdout_triples - pred_triples
    if missed:
        print(f"\n  Holdout triples NOT predicted ({len(missed)}):")
        for s, p, o in sorted(missed)[:10]:
            print(f"    <<{p}, {s}, {o}>>")

    # Novel predictions not in training or holdout
    all_train_triples = set(inf.triple_key() for inf in train_infons)
    novel = pred_triples - all_train_triples - holdout_triples
    if novel:
        print(f"\n  Novel predictions (not in training or holdout): {len(novel)}")
        for s, p, o in sorted(novel)[:10]:
            print(f"    <<{p}, {s}, {o}>>")

    # Cleanup
    schema_path.unlink(missing_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Experiment complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    run_experiment()
