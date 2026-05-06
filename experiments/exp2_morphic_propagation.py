"""Experiment 2: Morphic weight propagation through IsA/PartOf hierarchy.

When a child anchor activates (e.g., "iran"), propagate a fraction of its
weight upward to its parent anchor (e.g., "middle_east_actors"). This tests
whether hierarchy-aware activation enriches infon extraction by surfacing
category-level patterns that individual anchors miss.

Compares:
  - Baseline: raw SPLADE + AnchorProjector activations (no propagation)
  - Propagated: single-pass upward propagation with configurable decay
"""

from __future__ import annotations

import json
import tempfile
import numpy as np
from pathlib import Path

from infon import AnchorSchema, Encoder, InfonConfig, extract_infons, split_sentences
from infon.encoder import SpladeEncoder


# ── Decay factor for upward propagation ───────────────────────────────
DECAY_FACTOR = 0.5


# ── Geopolitical corpus (24 docs from test_category.py) ──────────────

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


# ── Schema with hierarchy ────────────────────────────────────────────
#
# Leaf anchors have a .parent field pointing to a category anchor.
# Category anchors are included in the schema so they get columns in
# the activation matrix and can participate in infon extraction.

GEO_SCHEMA_HIERARCHICAL = {
    # ── Category anchors (parents) ────────────────────────────────
    "middle_east_actors": {
        "type": "actor",
        "tokens": ["middle east", "gulf states", "levant"],
    },
    "western_powers": {
        "type": "actor",
        "tokens": ["western", "west", "allies"],
    },
    "east_asia_actors": {
        "type": "actor",
        "tokens": ["east asia", "pacific"],
    },
    "weapons": {
        "type": "feature",
        "tokens": ["weapons", "arms", "armament"],
    },
    "coercive_actions": {
        "type": "relation",
        "tokens": ["coerce", "coercion", "punish", "pressure"],
    },
    "diplomatic_actions": {
        "type": "relation",
        "tokens": ["diplomacy", "diplomatic", "dialogue"],
    },

    # ── Leaf actors ───────────────────────────────────────────────
    "iran": {
        "type": "actor",
        "tokens": ["iran", "iranian", "tehran"],
        "country_code": "IR",
        "parent": "middle_east_actors",
    },
    "israel": {
        "type": "actor",
        "tokens": ["israel", "israeli"],
        "country_code": "IL",
        "parent": "middle_east_actors",
    },
    "us": {
        "type": "actor",
        "tokens": ["united states", "us", "washington", "american"],
        "country_code": "US",
        "parent": "western_powers",
    },
    "nato": {
        "type": "actor",
        "tokens": ["nato", "alliance"],
        "parent": "western_powers",
    },
    "eu": {
        "type": "actor",
        "tokens": ["eu", "european union", "brussels"],
        "parent": "western_powers",
    },
    "china": {
        "type": "actor",
        "tokens": ["china", "chinese", "beijing"],
        "country_code": "CN",
        "parent": "east_asia_actors",
    },
    "japan": {
        "type": "actor",
        "tokens": ["japan", "japanese", "tokyo"],
        "country_code": "JP",
        "parent": "east_asia_actors",
    },

    # ── Leaf features (with parent) ──────────────────────────────
    "nuclear": {
        "type": "feature",
        "tokens": ["nuclear", "atomic", "missile", "warhead"],
        "parent": "weapons",
    },
    "military": {
        "type": "feature",
        "tokens": ["military", "defense", "armed forces", "weapons"],
        "parent": "weapons",
    },

    # ── Leaf relations (with parent) ─────────────────────────────
    "sanction": {
        "type": "relation",
        "tokens": ["sanction", "sanctions", "embargo", "restrict"],
        "parent": "coercive_actions",
    },
    "condemn": {
        "type": "relation",
        "tokens": ["condemn", "denounce", "protest", "urge", "demand"],
        "parent": "coercive_actions",
    },
    "attack": {
        "type": "relation",
        "tokens": ["attack", "strike", "bomb", "clash", "military"],
        "parent": "coercive_actions",
    },
    "negotiate": {
        "type": "relation",
        "tokens": ["negotiate", "talks", "diplomacy", "summit", "discuss"],
        "parent": "diplomatic_actions",
    },
    "cooperate": {
        "type": "relation",
        "tokens": ["cooperate", "cooperation", "collaborate", "partner", "aid"],
        "parent": "diplomatic_actions",
    },

    # ── Other anchors (no parent, kept for breadth) ──────────────
    "palestine": {
        "type": "actor",
        "tokens": ["palestine", "palestinian"],
        "country_code": "PS",
    },
    "russia": {
        "type": "actor",
        "tokens": ["russia", "russian", "moscow"],
        "country_code": "RU",
    },
    "india": {
        "type": "actor",
        "tokens": ["india", "indian"],
        "country_code": "IN",
    },
    "un": {
        "type": "actor",
        "tokens": ["un", "united nations", "security council"],
    },
    "african_union": {
        "type": "actor",
        "tokens": ["african union", "au"],
    },
    "deploy": {
        "type": "relation",
        "tokens": ["deploy", "deployment", "station", "mobilize", "troops"],
    },
    "trade": {
        "type": "relation",
        "tokens": ["trade", "tariff", "import", "export", "economic"],
    },
    "invest": {
        "type": "relation",
        "tokens": ["invest", "investment", "fund", "billion"],
    },
    "humanitarian": {
        "type": "feature",
        "tokens": ["humanitarian", "aid", "relief", "refugee"],
    },
    "territory": {
        "type": "feature",
        "tokens": ["territory", "border", "occupation", "settlement"],
    },
    "maritime": {
        "type": "feature",
        "tokens": ["maritime", "naval", "sea", "coast", "waters"],
    },
    "technology": {
        "type": "feature",
        "tokens": ["technology", "cyber", "digital", "innovation"],
    },
    "middle_east": {
        "type": "market",
        "tokens": ["middle east", "gulf", "levant"],
    },
    "east_asia": {
        "type": "market",
        "tokens": ["east asia", "pacific", "indo-pacific", "asia"],
    },
    "europe": {
        "type": "market",
        "tokens": ["europe", "european"],
    },
    "africa": {
        "type": "market",
        "tokens": ["africa", "african"],
    },
}


# ── Propagation logic ────────────────────────────────────────────────

def propagate_upward(
    activations: np.ndarray,
    schema: AnchorSchema,
    anchor_names: list[str],
    decay: float = DECAY_FACTOR,
) -> np.ndarray:
    """Single-pass upward propagation: child -> parent.

    For each activated child anchor, push (child_weight * decay) to its
    parent.  The parent takes max(own_activation, propagated_value).

    Args:
        activations: (n_sentences, n_anchors) raw activation matrix
        schema: AnchorSchema with parent/child relationships
        anchor_names: column ordering matching the activation matrix
        decay: fraction of child weight propagated upward

    Returns:
        (n_sentences, n_anchors) propagated activation matrix (copy)
    """
    prop = activations.copy()
    name_to_idx = {name: i for i, name in enumerate(anchor_names)}

    for child_name, child_idx in name_to_idx.items():
        parent_name = schema.get_parent(child_name)
        if parent_name is None:
            continue
        parent_idx = name_to_idx.get(parent_name)
        if parent_idx is None:
            continue

        # For every sentence, propagate child -> parent
        child_col = prop[:, child_idx]
        propagated = child_col * decay
        # Parent gets max of its own activation and the propagated value
        prop[:, parent_idx] = np.maximum(prop[:, parent_idx], propagated)

    return prop


def compare_activations(
    baseline: np.ndarray,
    propagated: np.ndarray,
    anchor_names: list[str],
    threshold: float = 0.3,
) -> dict:
    """Compare baseline vs propagated activations.

    Returns summary statistics and per-sentence detail for reporting.
    """
    n_sents = baseline.shape[0]

    baseline_active = (baseline > threshold).astype(int)
    propagated_active = (propagated > threshold).astype(int)

    # Newly activated anchors (were below threshold, now above)
    newly_active = propagated_active - baseline_active
    newly_active = np.clip(newly_active, 0, 1)  # only gains

    total_new = int(newly_active.sum())
    per_sentence_new = newly_active.sum(axis=1)  # (n_sents,)

    # Which anchors gained activations?
    anchor_gains = {}
    for j, name in enumerate(anchor_names):
        gain = int(newly_active[:, j].sum())
        if gain > 0:
            anchor_gains[name] = gain

    # Per-sentence detail: which sentences gained which anchors
    sentence_details = []
    for i in range(n_sents):
        gained = []
        for j in range(len(anchor_names)):
            if newly_active[i, j] > 0:
                gained.append((anchor_names[j], float(propagated[i, j])))
        if gained:
            sentence_details.append({"sent_idx": i, "gained": gained})

    return {
        "total_new_activations": total_new,
        "mean_new_per_sentence": float(per_sentence_new.mean()),
        "max_new_per_sentence": int(per_sentence_new.max()),
        "sentences_with_gains": int((per_sentence_new > 0).sum()),
        "anchor_gains": anchor_gains,
        "sentence_details": sentence_details,
    }


# ── Infon extraction wrapper ─────────────────────────────────────────

def extract_with_matrix(
    sentences: list[str],
    docs: list[dict],
    activation_matrix: np.ndarray,
    encoder: Encoder,
    schema: AnchorSchema,
    config: InfonConfig,
) -> list:
    """Extract infons using a pre-computed activation matrix.

    Monkey-patches encoder.encode to return the given matrix, runs
    extract_infons, then restores the original method.
    """
    original_encode = encoder.encode

    # Build document list matching the sentence split
    # extract_infons re-splits docs, so we need docs whose sentences
    # match our pre-split list.  Simplest: one doc per sentence.
    pseudo_docs = []
    for i, sent in enumerate(sentences):
        pseudo_docs.append({
            "id": f"sent-{i:03d}",
            "text": sent,
            "timestamp": docs[min(i, len(docs) - 1)].get("timestamp"),
        })

    call_count = [0]

    def patched_encode(texts, batch_size=32):
        call_count[0] += 1
        # extract_infons calls encode once with all sentences
        return activation_matrix

    encoder.encode = patched_encode
    try:
        infons, edges = extract_infons(pseudo_docs, encoder, schema, config)
    finally:
        encoder.encode = original_encode

    return infons


# ── Main experiment ──────────────────────────────────────────────────

def run_experiment():
    print("=" * 72)
    print("EXPERIMENT 2: Morphic Weight Propagation")
    print("=" * 72)
    print()

    # 1. Build schema
    schema = AnchorSchema(GEO_SCHEMA_HIERARCHICAL)
    anchor_names = schema.names
    print(f"Schema: {len(anchor_names)} anchors")
    print(f"  Parent anchors: {list(schema._children.keys())}")
    for parent_name in sorted(schema._children.keys()):
        children = schema.get_children(parent_name)
        print(f"    {parent_name} <- {children}")
    print()

    # 2. Encode corpus
    print("Encoding corpus through SPLADE + AnchorProjector ...")
    encoder = Encoder(schema=schema)
    config = InfonConfig(
        activation_threshold=0.3,
        top_k_per_role=3,
        min_confidence=0.05,
    )

    all_sentences = []
    for doc in GEO_DOCS:
        all_sentences.extend(split_sentences(doc["text"]))
    n_sents = len(all_sentences)
    print(f"  {len(GEO_DOCS)} documents -> {n_sents} sentences")

    baseline_activations = encoder.encode(all_sentences)
    print(f"  Activation matrix: {baseline_activations.shape}")
    print()

    # 3. Propagate
    print(f"Running upward propagation (decay={DECAY_FACTOR}) ...")
    propagated_activations = propagate_upward(
        baseline_activations, schema, anchor_names, decay=DECAY_FACTOR,
    )
    print()

    # 4. Compare activations
    print("-" * 72)
    print("ACTIVATION COMPARISON (threshold = {:.1f})".format(config.activation_threshold))
    print("-" * 72)

    comparison = compare_activations(
        baseline_activations,
        propagated_activations,
        anchor_names,
        threshold=config.activation_threshold,
    )

    print(f"  Total new anchor activations from propagation: {comparison['total_new_activations']}")
    print(f"  Mean new activations per sentence:             {comparison['mean_new_per_sentence']:.2f}")
    print(f"  Max new activations in a single sentence:      {comparison['max_new_per_sentence']}")
    print(f"  Sentences with at least one new activation:    {comparison['sentences_with_gains']}/{n_sents}")
    print()

    if comparison["anchor_gains"]:
        print("  Anchors that gained activations (via propagation):")
        for name, count in sorted(comparison["anchor_gains"].items(), key=lambda x: -x[1]):
            atype = schema.types.get(name, "?")
            print(f"    {atype:10s}  {name:22s}  +{count} sentences")
        print()

    # 5. Show specific examples
    print("-" * 72)
    print("DETAILED EXAMPLES: Sentences with propagated activations")
    print("-" * 72)
    name_to_idx = {name: i for i, name in enumerate(anchor_names)}

    shown = 0
    for detail in comparison["sentence_details"]:
        if shown >= 8:
            break
        idx = detail["sent_idx"]
        sent = all_sentences[idx]
        print(f"\n  Sentence {idx}: \"{sent[:100]}{'...' if len(sent) > 100 else ''}\"")

        # Show which children fired and caused propagation
        for gained_anchor, gained_score in detail["gained"]:
            children = schema.get_children(gained_anchor)
            firing_children = []
            for child in children:
                cidx = name_to_idx.get(child)
                if cidx is not None:
                    child_score = baseline_activations[idx, cidx]
                    if child_score > config.activation_threshold:
                        firing_children.append((child, float(child_score)))

            base_score = float(baseline_activations[idx, name_to_idx[gained_anchor]])
            print(f"    + {gained_anchor} activated: {base_score:.3f} -> {gained_score:.3f}")
            if firing_children:
                for ch_name, ch_score in firing_children:
                    print(f"      (child '{ch_name}' fired at {ch_score:.3f}, propagated {ch_score * DECAY_FACTOR:.3f})")
        shown += 1
    print()

    # 6. Extract infons with and without propagation
    print("-" * 72)
    print("INFON EXTRACTION COMPARISON")
    print("-" * 72)

    baseline_infons = extract_with_matrix(
        all_sentences, GEO_DOCS, baseline_activations,
        encoder, schema, config,
    )
    propagated_infons = extract_with_matrix(
        all_sentences, GEO_DOCS, propagated_activations,
        encoder, schema, config,
    )

    baseline_triples = set(inf.triple_key() for inf in baseline_infons)
    propagated_triples = set(inf.triple_key() for inf in propagated_infons)
    new_triples = propagated_triples - baseline_triples
    lost_triples = baseline_triples - propagated_triples

    print(f"  Baseline infons:    {len(baseline_infons)}  ({len(baseline_triples)} unique triples)")
    print(f"  Propagated infons:  {len(propagated_infons)}  ({len(propagated_triples)} unique triples)")
    print(f"  New triples:        +{len(new_triples)}")
    print(f"  Lost triples:       -{len(lost_triples)}")
    delta = len(propagated_infons) - len(baseline_infons)
    pct = (delta / max(len(baseline_infons), 1)) * 100
    print(f"  Delta:              {'+' if delta >= 0 else ''}{delta} infons ({pct:+.1f}%)")
    print()

    # Show new triples involving parent anchors
    parent_names = set(schema._children.keys())
    new_parent_triples = [t for t in new_triples if t[0] in parent_names or t[2] in parent_names]
    new_child_only = [t for t in new_triples if t not in set(new_parent_triples)]

    if new_parent_triples:
        print("  New triples involving parent (category) anchors:")
        for s, p, o in sorted(new_parent_triples)[:15]:
            stype = schema.types.get(s, "?")
            otype = schema.types.get(o, "?")
            marker_s = " [PARENT]" if s in parent_names else ""
            marker_o = " [PARENT]" if o in parent_names else ""
            print(f"    <<{p}, {s}{marker_s}, {o}{marker_o}>>")
        if len(new_parent_triples) > 15:
            print(f"    ... and {len(new_parent_triples) - 15} more")
        print()

    if new_child_only:
        print("  New triples NOT involving parent anchors (indirect effect):")
        for s, p, o in sorted(new_child_only)[:10]:
            print(f"    <<{p}, {s}, {o}>>")
        if len(new_child_only) > 10:
            print(f"    ... and {len(new_child_only) - 10} more")
        print()

    # 7. Qualitative analysis: infons that showcase the hierarchy
    print("-" * 72)
    print("QUALITATIVE: How propagation enriches the knowledge graph")
    print("-" * 72)

    # Find infons where a parent anchor participates as subject
    parent_as_subject = [inf for inf in propagated_infons if inf.subject in parent_names]
    parent_as_object = [inf for inf in propagated_infons if inf.object in parent_names]

    baseline_parent_subj = [inf for inf in baseline_infons if inf.subject in parent_names]
    baseline_parent_obj = [inf for inf in baseline_infons if inf.object in parent_names]

    print(f"  Parent anchors as SUBJECT: {len(baseline_parent_subj)} -> {len(parent_as_subject)} (baseline -> propagated)")
    print(f"  Parent anchors as OBJECT:  {len(baseline_parent_obj)} -> {len(parent_as_object)} (baseline -> propagated)")
    print()

    # Show some new parent-subject infons
    new_parent_subj_keys = (
        set(inf.triple_key() for inf in parent_as_subject)
        - set(inf.triple_key() for inf in baseline_parent_subj)
    )
    if new_parent_subj_keys:
        print("  New infons with parent anchor as subject (from propagation):")
        for key in sorted(new_parent_subj_keys)[:8]:
            matching = [inf for inf in propagated_infons if inf.triple_key() == key]
            if matching:
                inf = matching[0]
                print(f"    <<{inf.predicate}, {inf.subject}, {inf.object}>>  "
                      f"conf={inf.confidence:.3f}  \"{inf.sentence[:80]}...\"")
        print()

    # 8. Summary
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Schema anchors:            {len(anchor_names)}")
    print(f"    with parent:             {len(schema._parent)}")
    print(f"    parent categories:       {len(schema._children)}")
    print(f"  Corpus:                    {len(GEO_DOCS)} docs, {n_sents} sentences")
    print(f"  Propagation decay:         {DECAY_FACTOR}")
    print(f"  New activations:           +{comparison['total_new_activations']} across {comparison['sentences_with_gains']} sentences")
    print(f"  Infon delta:               {'+' if delta >= 0 else ''}{delta} ({pct:+.1f}%)")
    print(f"  New unique triples:        +{len(new_triples)}")
    print(f"  New parent-subject infons: +{len(new_parent_subj_keys) if new_parent_subj_keys else 0}")
    print()
    if comparison["total_new_activations"] > 0:
        print("  CONCLUSION: Upward morphic propagation successfully surfaces")
        print("  category-level anchors that would not activate on their own,")
        print("  enriching the infon graph with hierarchical structure.")
    else:
        print("  CONCLUSION: Parent anchors already activate sufficiently from")
        print("  their own token matches. Propagation adds marginal value in")
        print("  this domain -- the parent tokens may overlap with child tokens.")
    print()


if __name__ == "__main__":
    run_experiment()
