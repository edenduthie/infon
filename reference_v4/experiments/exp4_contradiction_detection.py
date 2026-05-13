"""Experiment 4: Sheaf-based contradiction detection.

Extends sheaf coherence to detect contradictions -- pairs of infons that
share anchors but whose NPMI co-activation neighborhoods are structurally
incompatible.  Two infons that claim the same (S, O) pair but via different
predicates should show up as contradictions when the predicates live in
very different neighborhoods of the co-activation graph.

The "gluing error" measures exactly this: for two infons sharing >= 2
anchors, how different are the NPMI row-vectors of their non-shared
anchors?  High L2 distance = the local sections cannot be glued into a
consistent global section = structural contradiction.

We also flag polarity conflicts: same (S, P, O) with polarity=1 vs 0.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cognition import (
    AnchorSchema,
    CognitionConfig,
    Encoder,
    extract_infons,
    split_sentences,
)
from cognition.category import SheafCoherence
from cognition.infon import Infon


# ── Geopolitical schema & corpus (from cognition/tests/test_category.py) ──

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


# ── Contradiction detection ────────────────────────────────────────────

@dataclass
class ContradictionPair:
    """A pair of infons flagged as potentially contradictory."""
    infon_a: Infon
    infon_b: Infon
    shared_anchors: set[str]
    non_shared_a: set[str]          # anchors unique to infon_a
    non_shared_b: set[str]          # anchors unique to infon_b
    gluing_error: float             # L2 distance between NPMI neighborhoods
    polarity_conflict: bool         # same (S,P,O) but opposite polarity?
    reason: str                     # human-readable explanation


def _anchor_set(infon: Infon) -> set[str]:
    """Return the set of anchors involved in an infon's triple."""
    return {infon.subject, infon.predicate, infon.object}


def find_shared_anchor_pairs(
    infons: list[Infon],
    min_shared: int = 2,
) -> list[tuple[Infon, Infon, set[str]]]:
    """Find all infon pairs sharing at least `min_shared` anchors."""
    # Index infons by each anchor they touch
    anchor_index: dict[str, list[int]] = defaultdict(list)
    for idx, inf in enumerate(infons):
        for a in _anchor_set(inf):
            anchor_index[a].append(idx)

    # Candidate pairs: share at least min_shared anchors
    pair_shared: dict[tuple[int, int], set[str]] = defaultdict(set)
    for anchor, indices in anchor_index.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                key = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                pair_shared[key].add(anchor)

    results = []
    for (i, j), shared in pair_shared.items():
        if len(shared) >= min_shared:
            # Skip identical triples with same polarity (that is reinforcement, not contradiction)
            if (infons[i].triple_key() == infons[j].triple_key()
                    and infons[i].polarity == infons[j].polarity):
                continue
            results.append((infons[i], infons[j], shared))

    return results


def compute_gluing_error(
    infon_a: Infon,
    infon_b: Infon,
    shared_anchors: set[str],
    sheaf: SheafCoherence,
) -> float:
    """Compute the L2 gluing error between two infons' NPMI neighborhoods.

    For the non-shared anchors between the two infons, compare their NPMI
    row vectors restricted to all other anchors.  High distance means the
    two infons live in structurally incompatible neighborhoods of the
    co-activation graph.
    """
    if sheaf.npmi is None:
        return 0.0

    anchors_a = _anchor_set(infon_a)
    anchors_b = _anchor_set(infon_b)
    non_shared = (anchors_a | anchors_b) - shared_anchors

    if not non_shared:
        # All anchors are shared -- use the full NPMI rows of shared anchors
        # to see if the context neighborhoods differ for the two infons
        non_shared_indices = list(range(sheaf.n))
    else:
        non_shared_indices = [
            sheaf.name_to_idx[a]
            for a in non_shared
            if a in sheaf.name_to_idx
        ]

    if not non_shared_indices:
        return 0.0

    shared_indices_a = [
        sheaf.name_to_idx[a]
        for a in anchors_a
        if a in sheaf.name_to_idx
    ]
    shared_indices_b = [
        sheaf.name_to_idx[a]
        for a in anchors_b
        if a in sheaf.name_to_idx
    ]

    # Build NPMI neighborhood vectors for the non-shared dimensions
    # For infon A: average NPMI row across its anchors, restricted to non-shared cols
    vec_a = np.mean(
        [sheaf.npmi[i, non_shared_indices] for i in shared_indices_a],
        axis=0,
    )
    vec_b = np.mean(
        [sheaf.npmi[i, non_shared_indices] for i in shared_indices_b],
        axis=0,
    )

    return float(np.linalg.norm(vec_a - vec_b))


def detect_contradictions(
    infons: list[Infon],
    sheaf: SheafCoherence,
    min_shared: int = 2,
) -> list[ContradictionPair]:
    """Detect contradictory infon pairs via sheaf gluing error.

    Two kinds of contradiction:
    1. Polarity conflict: same (S, P, O) with polarity=1 vs polarity=0
    2. Structural incompatibility: shared anchors but non-shared anchors
       live in very different NPMI neighborhoods (high gluing error)
    """
    pairs = find_shared_anchor_pairs(infons, min_shared=min_shared)
    contradictions = []

    for infon_a, infon_b, shared in pairs:
        anchors_a = _anchor_set(infon_a)
        anchors_b = _anchor_set(infon_b)
        non_shared_a = anchors_a - shared
        non_shared_b = anchors_b - shared

        # Check polarity conflict
        polarity_conflict = (
            infon_a.triple_key() == infon_b.triple_key()
            and infon_a.polarity != infon_b.polarity
        )

        # Compute gluing error
        error = compute_gluing_error(infon_a, infon_b, shared, sheaf)

        # Determine reason
        if polarity_conflict:
            reason = "Polarity conflict: same triple, opposite polarity"
        elif infon_a.subject == infon_b.subject and infon_a.object == infon_b.object:
            reason = (
                f"Same subject+object, different predicate: "
                f"{infon_a.predicate} vs {infon_b.predicate}"
            )
        elif infon_a.predicate == infon_b.predicate:
            diff_roles = []
            if infon_a.subject != infon_b.subject:
                diff_roles.append(f"subject: {infon_a.subject} vs {infon_b.subject}")
            if infon_a.object != infon_b.object:
                diff_roles.append(f"object: {infon_a.object} vs {infon_b.object}")
            reason = (
                f"Same predicate ({infon_a.predicate}), different "
                + ", ".join(diff_roles)
            )
        else:
            reason = f"Shared anchors {shared} with divergent neighborhoods"

        contradictions.append(ContradictionPair(
            infon_a=infon_a,
            infon_b=infon_b,
            shared_anchors=shared,
            non_shared_a=non_shared_a,
            non_shared_b=non_shared_b,
            gluing_error=error,
            polarity_conflict=polarity_conflict,
            reason=reason,
        ))

    # Sort by gluing error descending (strongest contradictions first)
    contradictions.sort(key=lambda c: c.gluing_error, reverse=True)
    return contradictions


# ── Report ──────────────────────────────────────────────────────────────

def _fmt_infon(inf: Infon) -> str:
    """Format an infon for display."""
    pol = "+" if inf.polarity else "-"
    return (
        f"  <<{inf.predicate}, {inf.subject}, {inf.object}; {pol}>>"
        f"  (conf={inf.confidence:.3f}, doc={inf.doc_id})"
    )


def print_report(contradictions: list[ContradictionPair]) -> None:
    """Print a clear contradiction detection report."""
    sep = "=" * 72

    print()
    print(sep)
    print("  SHEAF CONTRADICTION DETECTION REPORT")
    print(sep)

    # ── Summary statistics ──
    n = len(contradictions)
    polarity_conflicts = sum(1 for c in contradictions if c.polarity_conflict)
    structural = n - polarity_conflicts
    errors = [c.gluing_error for c in contradictions]

    print()
    print(f"  Total contradictory pairs found : {n}")
    print(f"    Polarity conflicts            : {polarity_conflicts}")
    print(f"    Structural incompatibilities   : {structural}")
    print()

    if errors:
        errors_arr = np.array(errors)
        print(f"  Gluing error distribution:")
        print(f"    min    = {errors_arr.min():.4f}")
        print(f"    median = {np.median(errors_arr):.4f}")
        print(f"    mean   = {errors_arr.mean():.4f}")
        print(f"    max    = {errors_arr.max():.4f}")
        print(f"    std    = {errors_arr.std():.4f}")

        # Histogram buckets
        bins = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, float("inf")]
        labels = ["[0.0, 0.1)", "[0.1, 0.2)", "[0.2, 0.5)", "[0.5, 1.0)", "[1.0, 2.0)", "[2.0+)"]
        print()
        print(f"  Error histogram:")
        for i in range(len(bins) - 1):
            count = sum(1 for e in errors if bins[i] <= e < bins[i + 1])
            bar = "#" * count
            print(f"    {labels[i]:12s} | {count:3d} {bar}")

    # ── Top 10 contradictions ──
    print()
    print("-" * 72)
    print("  TOP 10 CONTRADICTIONS (by gluing error)")
    print("-" * 72)

    for rank, c in enumerate(contradictions[:10], 1):
        print()
        print(f"  #{rank}  Gluing error = {c.gluing_error:.4f}"
              f"  {'[POLARITY CONFLICT]' if c.polarity_conflict else ''}")
        print(f"  Reason: {c.reason}")
        print(f"  Shared anchors: {c.shared_anchors}")
        print(f"  Infon A:{_fmt_infon(c.infon_a)}")
        print(f"  Infon B:{_fmt_infon(c.infon_b)}")

    print()
    print(sep)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Setting up schema and encoder...")

    # 1. Build schema and encoder
    schema_path = Path(tempfile.mktemp(suffix=".json"))
    schema_path.write_text(json.dumps(GEO_SCHEMA, indent=2))

    config = CognitionConfig(schema_path=str(schema_path))
    schema = AnchorSchema.from_file(schema_path)
    encoder = Encoder(schema=schema, max_length=config.max_length, device=config.device)

    # 2. Extract infons
    print(f"Extracting infons from {len(GEO_DOCS)} documents...")
    infons, edges = extract_infons(GEO_DOCS, encoder, schema, config)
    print(f"  Extracted {len(infons)} infons, {len(edges)} edges")

    # 3. Build sheaf
    print("Building sheaf co-activation model...")
    sheaf = SheafCoherence(encoder.anchor_names)
    all_sentences = []
    for doc in GEO_DOCS:
        all_sentences.extend(split_sentences(doc["text"]))
    print(f"  {len(all_sentences)} sentences for co-activation analysis")

    activations = encoder.encode(all_sentences)
    sheaf.observe(activations)
    sheaf.fit()

    print(f"  Fiedler value (algebraic connectivity): {sheaf.fiedler_value:.4f}")
    components = sheaf.component_structure()
    print(f"  Connected components: {len(components)}")

    # Score infons
    for inf in infons:
        inf.coherence = sheaf.score_infon(inf)

    scores = [inf.coherence for inf in infons]
    print(f"  Infon coherence: mean={np.mean(scores):.3f}, "
          f"std={np.std(scores):.3f}, "
          f"range=[{min(scores):.3f}, {max(scores):.3f}]")

    # 4. Detect contradictions
    print("\nDetecting contradictions...")
    contradictions = detect_contradictions(infons, sheaf, min_shared=2)

    # 5. Print report
    print_report(contradictions)

    # 6. Bonus: show the anchor NPMI neighborhoods for the top contradiction
    if contradictions:
        top = contradictions[0]
        print("\n  NPMI NEIGHBORHOOD DETAIL for top contradiction:")
        print(f"  Infon A:{_fmt_infon(top.infon_a)}")
        print(f"  Infon B:{_fmt_infon(top.infon_b)}")
        print()

        anchors_a = _anchor_set(top.infon_a)
        anchors_b = _anchor_set(top.infon_b)
        all_involved = sorted(anchors_a | anchors_b)

        # Show NPMI values between all involved anchors
        header = f"{'':>15s}"
        for a in all_involved:
            header += f" {a:>12s}"
        print(header)

        for a in all_involved:
            if a not in sheaf.name_to_idx:
                continue
            row = f"{a:>15s}"
            i = sheaf.name_to_idx[a]
            for b in all_involved:
                if b not in sheaf.name_to_idx:
                    row += f" {'N/A':>12s}"
                    continue
                j = sheaf.name_to_idx[b]
                val = sheaf.npmi[i, j]
                marker = ""
                if a in anchors_a and b in anchors_a:
                    marker = " (A)"
                elif a in anchors_b and b in anchors_b:
                    marker = " (B)"
                row += f" {val:>8.3f}{marker:4s}"
            print(row)

    # Cleanup temp file
    schema_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
