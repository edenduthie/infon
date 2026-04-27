"""Imagination: genetic-algorithm proposal of counterfactual infons.

Query-scoped. Given a query, seeds a GA population from query-matching
observed infons, then iterates mutation + selection over typed triples,
scored by:

    fitness = grammar(hard) × (1 + logic_penalty) × health

Output is isomorphic to MCTSResult with dual verdict fields
(imagination-native + MCTS-compatible) so existing MCTS renderers work.

Mutation operators:
    HierarchyWalkMutator       — replace subject or object with parent/sibling/child
    PredicateSubstitutionMutator — swap predicate with a same-type relation
    PolarityFlipMutator        — invert polarity (counterfactual)
    RoleRecombinationMutator   — cross two observed infons
    TemporalProjectionMutator  — project a triple forward in time

Fitness components:
    grammar           — hard 0/1 filter on typed roles
    logic_penalty     — soft ∈ [-1, 0], penalises contradictions
    health = whisper × bridge × persona_align
       whisper        — every role anchor must exist in corpus
       bridge         — reward for connecting disjoint subgraphs
       persona_align  — persona predicate-alignment weight
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

from .infon import (
    Infon, ImaginationNode, ImaginationResult,
    IMAGINATION_TO_MCTS_VERDICT,
)
from .dempster_shafer import MassFunction, combine_multiple


# ── Mutators ────────────────────────────────────────────────────────────

class Mutator:
    """Base class. Mutators take an Infon (or two, for recombination) and
    return a mutated copy, or None if the mutation doesn't apply."""
    name: str = "base"

    def apply(self, infon: Infon, *, rng: random.Random,
              schema=None, all_observed: list[Infon] = ()) -> Infon | None:
        raise NotImplementedError


class HierarchyWalkMutator(Mutator):
    """Replace subject or object with its parent, sibling, or child."""
    name = "hierarchy_walk"

    def __init__(self, schema):
        self.schema = schema

    def apply(self, infon, *, rng, schema=None, all_observed=()):
        role = rng.choice(["subject", "object"])
        current = getattr(infon, role)

        candidates = []
        parent = self.schema.get_parent(current)
        if parent:
            candidates.append(parent)
            for sib in self.schema.get_children(parent):
                if sib != current:
                    candidates.append(sib)
        for child in self.schema.get_children(current):
            candidates.append(child)

        # Require correct type for the role being replaced
        required_type = infon.subject_meta.get("type") if role == "subject" \
                        else infon.object_meta.get("type")
        if required_type:
            candidates = [c for c in candidates
                          if self.schema.types.get(c) == required_type]

        if not candidates:
            return None

        replacement = rng.choice(candidates)
        if replacement == current:
            return None

        return _clone_with(infon, **{role: replacement})


class PredicateSubstitutionMutator(Mutator):
    """Replace predicate with another relation-type anchor."""
    name = "predicate_sub"

    def __init__(self, schema):
        self.schema = schema
        self._rel_anchors = [n for n, t in schema.types.items() if t == "relation"]

    def apply(self, infon, *, rng, schema=None, all_observed=()):
        if not self._rel_anchors:
            return None
        candidates = [n for n in self._rel_anchors if n != infon.predicate]
        if not candidates:
            return None
        return _clone_with(infon, predicate=rng.choice(candidates))


class PolarityFlipMutator(Mutator):
    """Invert polarity."""
    name = "polarity_flip"

    def apply(self, infon, *, rng, schema=None, all_observed=()):
        new_pol = 0 if infon.polarity == 1 else 1
        return _clone_with(infon, polarity=new_pol)


class RoleRecombinationMutator(Mutator):
    """Cross two observed infons — subject from one, predicate/object from another."""
    name = "recombination"

    def apply(self, infon, *, rng, schema=None, all_observed=()):
        if len(all_observed) < 2:
            return None
        other = rng.choice(all_observed)
        if other.infon_id == infon.infon_id:
            return None
        choice = rng.choice(["sub_from_a", "pred_from_a", "obj_from_a"])
        if choice == "sub_from_a":
            return _clone_with(infon, predicate=other.predicate,
                               object=other.object,
                               predicate_meta=other.predicate_meta,
                               object_meta=other.object_meta)
        elif choice == "pred_from_a":
            return _clone_with(infon, subject=other.subject,
                               object=other.object,
                               subject_meta=other.subject_meta,
                               object_meta=other.object_meta)
        else:
            return _clone_with(infon, subject=other.subject,
                               predicate=other.predicate,
                               subject_meta=other.subject_meta,
                               predicate_meta=other.predicate_meta)


class TemporalProjectionMutator(Mutator):
    """Propose the same triple at a future timestamp.

    Uses the subject's NEXT-edge gap distribution when available; falls back
    to a fixed 30-day shift otherwise. No timestamp → returns None.
    """
    name = "temporal_projection"

    def __init__(self, store):
        self.store = store

    def apply(self, infon, *, rng, schema=None, all_observed=()):
        if not infon.timestamp:
            return None

        # Simple projection: add 30 days
        from datetime import datetime, timedelta
        try:
            dt = datetime.strptime(infon.timestamp, "%Y-%m-%d")
        except ValueError:
            return None
        delta_days = rng.choice([7, 14, 30, 60])
        future = (dt + timedelta(days=delta_days)).strftime("%Y-%m-%d")
        return _clone_with(infon, timestamp=future)


# ── Helpers ─────────────────────────────────────────────────────────────

def _clone_with(infon: Infon, **overrides) -> Infon:
    """Shallow-copy an infon with overrides, producing an imagined child."""
    d = infon.to_dict()
    d.update(overrides)
    # Reset provenance fields
    d["kind"] = "imagined"
    parents = list(d.get("parent_infon_ids") or [])
    if infon.infon_id and infon.infon_id not in parents:
        parents.append(infon.infon_id)
    d["parent_infon_ids"] = parents
    # Recompute id from the new triple (deterministic for dedup)
    d["infon_id"] = _imagined_id(d["subject"], d["predicate"], d["object"],
                                  d.get("polarity", 1))
    d["fitness"] = None   # will be set by fitness scorer
    d["doc_id"] = "imagined"
    d["sent_id"] = ""
    d["sentence"] = (f"imagined: <<{d['predicate']}, {d['subject']}, "
                     f"{d['object']}; {d['polarity']}>>")
    return Infon.from_dict(d)


def _imagined_id(subj: str, pred: str, obj: str, polarity: int) -> str:
    raw = f"imagined:{subj}:{pred}:{obj}:{polarity}"
    return "img_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── Fitness ──────────────────────────────────────────────────────────────

PERSONAS_DEFAULT = {
    "investor":  {"positive_predicates": ["invest", "grow", "launch", "expand", "partner"],
                  "negative_predicates": ["decline", "divest", "delay", "cancel", "lose"]},
    "engineer":  {"positive_predicates": ["launch", "develop", "patent", "innovate", "improve"],
                  "negative_predicates": ["delay", "cancel", "fail", "recall"]},
    "executive": {"positive_predicates": ["grow", "expand", "launch", "dominate", "partner"],
                  "negative_predicates": ["decline", "lose", "exit", "shrink", "cancel"]},
    "regulator": {"positive_predicates": ["regulate", "comply", "standardize", "certify"],
                  "negative_predicates": ["violate", "evade", "monopolize", "lobby"]},
    "analyst":   {"positive_predicates": [], "negative_predicates": []},
}


class FitnessFunction:
    """grammar × (1 + logic_penalty) × health.

    Caches corpus-level aggregates (anchor frequencies, adjacency) across
    calls within one GA run since they're invariant across mutations.
    """

    def __init__(self, store, schema, config,
                 weights: dict | None = None, persona: str | None = None):
        self.store = store
        self.schema = schema
        self.config = config
        self.weights = weights or {"grammar": 1.0, "logic": 1.0, "health": 1.0}
        self.persona = persona
        self.persona_info = PERSONAS_DEFAULT.get(persona, {}) if persona else {}

        self._anchor_freq: dict[str, int] = {}
        self._max_freq = 1
        self._observed_triples: set[tuple] = set()
        self._adjacency: dict[str, set[str]] = {}
        self._build_cache()

    def _build_cache(self):
        """Precompute anchor frequencies and adjacency over observed infons."""
        observed = self.store.query_infons(limit=50000)
        from collections import defaultdict
        freq = defaultdict(int)
        adj = defaultdict(set)
        for inf in observed:
            freq[inf.subject] += 1
            freq[inf.predicate] += 1
            freq[inf.object] += 1
            adj[inf.subject].add(inf.object)
            adj[inf.object].add(inf.subject)
            self._observed_triples.add(
                (inf.subject, inf.predicate, inf.object)
            )
        self._anchor_freq = dict(freq)
        self._max_freq = max(freq.values()) if freq else 1
        self._adjacency = {k: set(v) for k, v in adj.items()}

    def grammar(self, infon: Infon) -> float:
        """Hard 0/1: subject is actor-type, predicate is relation, object not-actor."""
        types = self.schema.types
        subj_t = types.get(infon.subject, "")
        pred_t = types.get(infon.predicate, "")
        obj_t = types.get(infon.object, "")
        if subj_t != "actor":
            return 0.0
        if pred_t != "relation":
            return 0.0
        if obj_t == "actor" or obj_t == "":
            return 0.0
        return 1.0

    def logic_penalty(self, infon: Infon) -> float:
        """Soft ∈ [-1, 0]. Penalise triples contradicting high-confidence constraints."""
        penalty = 0.0
        # Same triple, opposite polarity in existing constraints
        existing = self.store.get_constraints(
            subject=infon.subject, predicate=infon.predicate,
            object=infon.object, limit=5,
        )
        for c in existing:
            # Constraint records don't track polarity per se; use triple-match
            # as a "this has been observed" signal. If we're flipping an
            # observed polarity, that's a contradiction.
            if infon.polarity == 0 and (infon.subject, infon.predicate,
                                         infon.object) in self._observed_triples:
                penalty -= 0.5 * min(1.0, c.strength) * min(
                    1.0, c.persistence / 5.0
                )
        return max(-1.0, penalty)

    def whisper(self, infon: Infon) -> float:
        """Each role anchor must have observed corpus presence."""
        if self._max_freq == 0:
            return 0.0
        f_s = self._anchor_freq.get(infon.subject, 0)
        f_p = self._anchor_freq.get(infon.predicate, 0)
        f_o = self._anchor_freq.get(infon.object, 0)
        # Zero whisper if any role is never observed — prevents hallucination
        if f_s == 0 or f_p == 0 or f_o == 0:
            return 0.0
        # Geometric mean, normalised
        import math
        g = math.pow(f_s * f_p * f_o, 1.0 / 3.0) / self._max_freq
        return min(1.0, g)

    def bridge(self, infon: Infon) -> float:
        """Reward imagined triples that connect non-adjacent anchors.

        Score = 1.0 if subject and object are NOT currently adjacent in the
        observed graph (a novel bridge), 0.3 if they're already adjacent
        (restating known adjacency is less informative).
        """
        adj_s = self._adjacency.get(infon.subject, set())
        if infon.object in adj_s:
            return 0.3
        return 1.0

    def persona_align(self, infon: Infon) -> float:
        """Persona-weighted: positive predicate → 1.0, negative → 0.3, neutral → 0.6."""
        if not self.persona_info:
            return 1.0
        pos = self.persona_info.get("positive_predicates", [])
        neg = self.persona_info.get("negative_predicates", [])
        if infon.predicate in pos:
            return 1.0
        if infon.predicate in neg:
            return 0.3
        return 0.6

    def health(self, infon: Infon) -> float:
        return (self.whisper(infon)
                * self.bridge(infon)
                * self.persona_align(infon))

    def __call__(self, infon: Infon) -> tuple[float, dict]:
        """Returns (fitness, component_scores) for transparent scoring."""
        g = self.grammar(infon)
        if g <= 0:
            return 0.0, {"grammar": 0.0}
        lp = self.logic_penalty(infon)
        h = self.health(infon)

        fit = (self.weights["grammar"] * g
               * max(0.0, 1.0 + self.weights["logic"] * lp)
               * self.weights["health"] * h)
        components = {"grammar": g, "logic_penalty": lp, "health": h,
                      "whisper": self.whisper(infon),
                      "bridge": self.bridge(infon),
                      "persona_align": self.persona_align(infon)}
        return fit, components


# ── GA runner ───────────────────────────────────────────────────────────

class Imagination:
    """Query-scoped GA over counterfactual infons."""

    def __init__(self, store, schema, encoder, config):
        self.store = store
        self.schema = schema
        self.encoder = encoder
        self.config = config
        self.mutators: list[Mutator] = [
            HierarchyWalkMutator(schema),
            PredicateSubstitutionMutator(schema),
            PolarityFlipMutator(),
            RoleRecombinationMutator(),
            TemporalProjectionMutator(store),
        ]

    def run(self, query: str, *,
            persona: str | None = None,
            n_generations: int = 10,
            population_size: int = 50,
            mutation_rate: float = 0.7,
            elitism: float = 0.2,
            cost_weights: dict | None = None,
            top_k: int = 20,
            seed: int | None = None) -> ImaginationResult:
        """Run the GA loop and return an MCTS-shaped ImaginationResult."""
        t0 = time.perf_counter()
        rng = random.Random(seed)

        # Seed from query-activated observed infons
        query_anchors = self.encoder.encode_single(query)
        activated = {
            name: score for name, score in query_anchors.items()
            if score >= self.config.activation_threshold
        }

        seed_infons = self._seed_infons(activated, limit=population_size)
        if not seed_infons:
            # Couldn't find anything to seed from — return an empty speculative result
            return _empty_result(query, activated, t0)

        fitness_fn = FitnessFunction(
            store=self.store, schema=self.schema, config=self.config,
            weights=cost_weights or {"grammar": 1.0, "logic": 1.0, "health": 1.0},
            persona=persona,
        )

        # Build root node
        root = _new_node("root", list(activated.keys())[:3],
                         seed_infons[:10], mutator_used=None,
                         parent=None, visit_count=0)

        # Initial population: seed infons (re-clone to mark imagined) + a few random mutations
        population: list[Infon] = []
        all_observed = seed_infons + self.store.query_infons(limit=2000)
        dedup: set[str] = set()

        for s in seed_infons[:population_size // 2]:
            # We use seed observed infons as initial candidates for mutation,
            # but we don't emit them directly as imagined — they're real.
            # Spawn an initial mutation from each to seed generation 0.
            mut = rng.choice(self.mutators).apply(
                s, rng=rng, schema=self.schema, all_observed=all_observed,
            )
            if mut and mut.infon_id not in dedup:
                population.append(mut)
                dedup.add(mut.infon_id)

        iter_log: list[dict] = []
        best_fitness_prev = -1.0
        converged_count = 0
        nodes_explored = 0

        # GA main loop
        for gen in range(n_generations):
            scored: list[tuple[Infon, float, dict]] = []
            for ind in population:
                fit, comp = fitness_fn(ind)
                if fit > 0:
                    ind.fitness = fit
                    scored.append((ind, fit, comp))
                    nodes_explored += 1

            if not scored:
                break

            scored.sort(key=lambda x: -x[1])
            top = scored[:max(5, int(elitism * population_size))]
            mean_fit = sum(s[1] for s in scored) / len(scored)
            max_fit = scored[0][1]

            iter_log.append({
                "generation": gen,
                "n_candidates": len(scored),
                "max_fitness": max_fit,
                "mean_fitness": mean_fit,
                "new_unique": len(population) - len(dedup) + len(dedup),
            })

            # Attach elites as children of root
            for ind, fit, comp in top[:8]:
                child_node = _new_node(
                    node_id=f"gen{gen}_{ind.infon_id[:8]}",
                    anchor_path=[ind.subject, ind.predicate, ind.object],
                    infons=[ind], mutator_used=None,
                    parent=root, visit_count=gen + 1,
                )
                child_node.fitness = fit
                # Infer a mass function from logic_penalty — treats contradictions
                # as refutation signal.
                lp = comp.get("logic_penalty", 0.0)
                if lp < -0.3:
                    child_node.belief_mass = MassFunction(
                        refutes=min(0.9, abs(lp)), theta=1.0 - min(0.9, abs(lp))
                    )
                else:
                    child_node.belief_mass = MassFunction(
                        supports=min(0.9, fit), theta=1.0 - min(0.9, fit)
                    )
                root.children.append(child_node)

            # Convergence
            if abs(max_fit - best_fitness_prev) < 0.005:
                converged_count += 1
                if converged_count >= 3:
                    break
            else:
                converged_count = 0
            best_fitness_prev = max_fit

            # Produce next generation
            new_pop: list[Infon] = [ind for ind, _, _ in top]
            while len(new_pop) < population_size:
                parent = rng.choice([ind for ind, _, _ in top])
                mutator = rng.choice(self.mutators)
                child = mutator.apply(parent, rng=rng, schema=self.schema,
                                      all_observed=all_observed)
                if not child:
                    continue
                if child.infon_id in dedup:
                    continue
                dedup.add(child.infon_id)
                new_pop.append(child)
            population = new_pop

        # Build result — dedup children by canonical triple_id so the same
        # (subject, predicate, object, polarity) doesn't appear twice in the
        # elite set across generations.
        seen_ids: set[str] = set()
        unique_children = []
        for n in sorted(root.children, key=lambda c: -c.fitness):
            if n.infons and n.infons[0].infon_id not in seen_ids:
                unique_children.append(n)
                seen_ids.add(n.infons[0].infon_id)
        final_top = unique_children[:top_k]

        imagined_infons: list[Infon] = []
        seen_inf_ids: set[str] = set()
        for node in final_top:
            for inf in node.infons:
                if inf.fitness is None or inf.infon_id in seen_inf_ids:
                    continue
                imagined_infons.append(inf)
                seen_inf_ids.add(inf.infon_id)

        combined_mass = self._combine_masses(
            [n.belief_mass for n in final_top if n.belief_mass]
        )

        verdict = self._derive_verdict(imagined_infons, fitness_fn)
        mcts_verdict = IMAGINATION_TO_MCTS_VERDICT[verdict]

        chains = [[n.anchor_path[0], n.anchor_path[1], n.anchor_path[2]]
                  for n in final_top if len(n.anchor_path) >= 3]

        elapsed = time.perf_counter() - t0
        return ImaginationResult(
            query=query, seed_anchors=activated,
            verdict=verdict, mcts_verdict=mcts_verdict,
            combined_mass=combined_mass, traversal_tree=root,
            iteration_log=iter_log, chains_discovered=chains,
            imagined_infons=imagined_infons,
            nodes_explored=nodes_explored,
            infons_evaluated=len(imagined_infons),
            generations=len(iter_log),
            iterations=len(iter_log),
            elapsed_s=elapsed,
        )

    def _seed_infons(self, activated: dict, limit: int) -> list[Infon]:
        """Gather observed infons whose anchors appear in the query activations."""
        found: dict[str, Infon] = {}
        for anchor in sorted(activated, key=lambda a: -activated[a]):
            hits = self.store.get_infons_for_anchor(anchor, limit=limit)
            for inf in hits:
                if inf.infon_id not in found:
                    found[inf.infon_id] = inf
                if len(found) >= limit:
                    break
            if len(found) >= limit:
                break
        return list(found.values())

    @staticmethod
    def _combine_masses(masses: list[MassFunction]) -> MassFunction:
        if not masses:
            return MassFunction(theta=1.0)
        return combine_multiple(masses)

    @staticmethod
    def _derive_verdict(infons: list[Infon], fitness_fn: FitnessFunction) -> str:
        """PLAUSIBLE / CONTRADICTED / SPECULATIVE per spec."""
        if not infons:
            return "SPECULATIVE"

        fits = [inf.fitness for inf in infons[:10] if inf.fitness is not None]
        if not fits:
            return "SPECULATIVE"

        # Compute logic_penalty for each
        penalties = [fitness_fn.logic_penalty(inf) for inf in infons[:10]]
        mean_penalty = sum(penalties) / len(penalties)
        top_penalty = penalties[0] if penalties else 0.0

        if top_penalty < -0.5 or mean_penalty < -0.4:
            return "CONTRADICTED"

        strong = sum(1 for f in fits if f >= 0.6)
        no_contradiction = all(p >= -0.2 for p in penalties[:5])
        if strong >= 5 and no_contradiction:
            return "PLAUSIBLE"

        return "SPECULATIVE"


def _new_node(node_id: str, anchor_path: list, infons: list,
              mutator_used: str | None, parent: ImaginationNode | None,
              visit_count: int = 0) -> ImaginationNode:
    return ImaginationNode(
        node_id=node_id, anchor_path=list(anchor_path),
        infons=list(infons), mutator_used=mutator_used,
        parent=parent, visit_count=visit_count,
    )


def _empty_result(query: str, activated: dict, t0: float) -> ImaginationResult:
    root = _new_node("root", [], [], None, None, 0)
    return ImaginationResult(
        query=query, seed_anchors=activated,
        verdict="SPECULATIVE",
        mcts_verdict=IMAGINATION_TO_MCTS_VERDICT["SPECULATIVE"],
        combined_mass=MassFunction(theta=1.0),
        traversal_tree=root, iteration_log=[],
        chains_discovered=[], imagined_infons=[],
        nodes_explored=0, infons_evaluated=0,
        generations=0, iterations=0,
        elapsed_s=time.perf_counter() - t0,
    )
