"""Synthetic corpus generator for training the multi-task embedder.

Generates fully-labeled training examples. Each sample carries:
    - sentence (raw text)
    - token_roles: per-token label in {O, NSUBJ, VERB, DOBJ, POBJ, MARKER}
    - token_anchors: per-token schema-anchor name (or empty string)
    - ds_mass: (S, R, U, theta)  — supervised by template type
    - evidentiality: assertion | hedge | attribution
    - modality: certain | possible | obligation
    - template: id of the generating template

Templates are grouped by structural pattern; each template has a fill
slot schema and a gold-label recipe. The generator is deterministic
given a seed so corpora are reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# ── Role labels ─────────────────────────────────────────────────────
ROLE_O       = 0
ROLE_NSUBJ   = 1
ROLE_VERB    = 2
ROLE_DOBJ    = 3
ROLE_POBJ    = 4
ROLE_MARKER  = 5   # hedge / modal / attribution cue
ROLE_LABELS = ["O", "NSUBJ", "VERB", "DOBJ", "POBJ", "MARKER"]
N_ROLES = len(ROLE_LABELS)


@dataclass
class SynthExample:
    sentence: str
    tokens: list[str]
    token_roles: list[int]             # one int per token
    token_anchors: list[str]           # "" if no anchor
    ds_mass: tuple[float, float, float, float]  # (S, R, U, theta)
    evidentiality: str
    modality: str
    template_id: str
    gold_triple: tuple[str, str, str]  # (subject, predicate, object)


def _split_tokens(text: str) -> list[str]:
    """Very simple whitespace+punctuation tokenizer."""
    out = []
    for chunk in text.split():
        stripped = chunk.rstrip(".,;:!?")
        if stripped:
            out.append(stripped)
        punct = chunk[len(stripped):]
        for ch in punct:
            out.append(ch)
    return out


@dataclass
class Schema:
    actors: list[str]
    relations: list[str]
    features: list[str]
    markets: list[str]
    attribution_sources: list[str] = field(default_factory=lambda: [
        "Reuters", "Bloomberg", "analysts", "reports",
    ])

    def all_anchor_names(self) -> list[str]:
        return self.actors + self.relations + self.features + self.markets

    def types(self) -> dict[str, str]:
        t = {}
        for a in self.actors:    t[a] = "actor"
        for r in self.relations: t[r] = "relation"
        for f in self.features:  t[f] = "feature"
        for m in self.markets:   t[m] = "market"
        return t

    @classmethod
    def from_anchor_schema(cls, anchor_schema) -> "Schema":
        """Bridge from infon.schema.AnchorSchema → synth.Schema.

        Partitions the anchor names by their type. Requires at least one
        anchor of each of actor/relation/feature/market; types without
        any anchors fall back to a single empty list — generator
        templates referencing that type will simply produce no data.
        """
        actors, relations, features, markets = [], [], [], []
        for name in anchor_schema.names:
            atype = anchor_schema.types.get(name, "feature")
            if atype == "actor":
                actors.append(name)
            elif atype == "relation":
                relations.append(name)
            elif atype == "market":
                markets.append(name)
            else:
                features.append(name)
        # The generator needs at least one of each to produce templates.
        # Fill empty buckets with a placeholder so generation doesn't
        # crash — the output will just be biased toward the populated
        # types, which is fine for training.
        if not actors:
            actors = ["entity"]
        if not relations:
            relations = ["relates"]
        if not features:
            features = ["thing"]
        if not markets:
            markets = ["place"]
        return cls(
            actors=actors,
            relations=relations,
            features=features,
            markets=markets,
        )


# ── Default schema used by the generator in tests ──────────────────

DEFAULT_SCHEMA = Schema(
    actors=["Toyota", "Honda", "Tesla", "Panasonic", "CATL"],
    relations=["invests", "partners", "produces", "expands",
               "delays", "acquires"],
    features=["batteries", "EVs", "factories", "supply"],
    markets=["Japan", "China", "America"],
)


# ── Template definitions ────────────────────────────────────────────
# Each template is a function taking schema + rng and returning a
# (sentence, token_roles, token_anchors, ds_mass, evid, modl, triple)
# tuple. Roles align to tokens produced by _split_tokens.

def _find_token_idx(tokens: list[str], needle: str) -> int | None:
    """Return the first index where needle appears, case-insensitive."""
    lo = needle.lower()
    for i, t in enumerate(tokens):
        if t.lower() == lo:
            return i
    return None


def _relation_past_participle(rel: str) -> str:
    """Map present-tense relation to past participle (naive)."""
    mapping = {
        "invests": "invested",
        "partners": "partnered",
        "produces": "produced",
        "expands": "expanded",
        "delays": "delayed",
        "acquires": "acquired",
    }
    return mapping.get(rel, rel + "ed")


def _anchor_from_text(schema: Schema, text: str) -> str:
    """Return the schema anchor name matching a surface form.

    Surface forms like 'batteries' and 'EVs' map to anchor names as-is
    because our schema uses the surface-plural form.
    """
    low = text.lower()
    for name in schema.all_anchor_names():
        if name.lower() == low:
            return name
    return ""


def _make_example(schema: Schema, rng: random.Random,
                  template_id: str) -> SynthExample:
    """Dispatch by template id. Each branch builds its own sentence +
    token-aligned gold labels."""
    actor = rng.choice(schema.actors)
    others = [a for a in schema.actors if a != actor]
    actor2 = rng.choice(others) if others else actor
    rel = rng.choice(schema.relations)
    feat = rng.choice(schema.features)
    market = rng.choice(schema.markets)
    source = rng.choice(schema.attribution_sources)

    if template_id == "svo":
        # "Toyota invests batteries."
        sentence = f"{actor} {rel} {feat}."
        triple = (actor, rel, feat)
        mass = (0.70, 0.02, 0.03, 0.25)
        evid, modl = "assertion", "certain"

    elif template_id == "svo_prep":
        # "Toyota partners with Panasonic in Japan."
        sentence = f"{actor} {rel} with {actor2} in {market}."
        triple = (actor, rel, actor2)
        mass = (0.68, 0.02, 0.03, 0.27)
        evid, modl = "assertion", "certain"

    elif template_id == "svo_for":
        # "CATL produces batteries for Honda."
        sentence = f"{actor} {rel} {feat} for {actor2}."
        triple = (actor, rel, feat)
        mass = (0.65, 0.03, 0.03, 0.29)
        evid, modl = "assertion", "certain"

    elif template_id == "hedge":
        # "Toyota may possibly invest in batteries."
        sentence = f"{actor} may possibly {rel} {feat}."
        triple = (actor, rel, feat)
        mass = (0.20, 0.02, 0.05, 0.73)
        evid, modl = "hedge", "possible"

    elif template_id == "obligation":
        # "Honda must expand factories in America."
        sentence = f"{actor} must {rel} {feat} in {market}."
        triple = (actor, rel, feat)
        mass = (0.15, 0.02, 0.48, 0.35)
        evid, modl = "assertion", "obligation"

    elif template_id == "attribution":
        # "According to Reuters, Tesla produces batteries."
        sentence = f"According to {source}, {actor} {rel} {feat}."
        triple = (actor, rel, feat)
        mass = (0.35, 0.03, 0.05, 0.57)
        evid, modl = "attribution", "certain"

    elif template_id == "passive":
        # "Batteries are produced by Tesla in America."
        past = _relation_past_participle(rel)
        sentence = f"{feat} are {past} by {actor} in {market}."
        triple = (actor, rel, feat)
        mass = (0.62, 0.03, 0.04, 0.31)
        evid, modl = "assertion", "certain"

    elif template_id == "neg":
        # "Toyota does not produce batteries."
        sentence = f"{actor} does not {rel} {feat}."
        triple = (actor, rel, feat)
        mass = (0.03, 0.72, 0.03, 0.22)
        evid, modl = "assertion", "certain"

    else:
        raise ValueError(f"unknown template_id {template_id!r}")

    tokens = _split_tokens(sentence)
    token_roles = [ROLE_O] * len(tokens)
    token_anchors = [""] * len(tokens)

    # Label the subject, verb, and (dobj | pobj) positions
    # by locating the surface string inside the tokens.
    def _tag(tok_text: str, role: int, anchor: str):
        idx = _find_token_idx(tokens, tok_text)
        if idx is not None:
            token_roles[idx] = role
            token_anchors[idx] = anchor

    if template_id == "passive":
        # Subject in surface grammar is the patient; logical subject is actor.
        # We label by *logical* role so downstream heads learn agentive S.
        _tag(feat, ROLE_DOBJ, feat)
        _tag(past, ROLE_VERB, rel)
        _tag(actor, ROLE_NSUBJ, actor)
        _tag(market, ROLE_POBJ, market)
    elif template_id == "svo_prep":
        _tag(actor, ROLE_NSUBJ, actor)
        _tag(rel, ROLE_VERB, rel)
        _tag(actor2, ROLE_POBJ, actor2)
        _tag(market, ROLE_POBJ, market)
    elif template_id == "svo_for":
        _tag(actor, ROLE_NSUBJ, actor)
        _tag(rel, ROLE_VERB, rel)
        _tag(feat, ROLE_DOBJ, feat)
        _tag(actor2, ROLE_POBJ, actor2)
    elif template_id == "obligation":
        _tag(actor, ROLE_NSUBJ, actor)
        _tag("must", ROLE_MARKER, "")
        _tag(rel, ROLE_VERB, rel)
        _tag(feat, ROLE_DOBJ, feat)
        _tag(market, ROLE_POBJ, market)
    elif template_id == "hedge":
        _tag(actor, ROLE_NSUBJ, actor)
        _tag("may", ROLE_MARKER, "")
        _tag("possibly", ROLE_MARKER, "")
        _tag(rel, ROLE_VERB, rel)
        _tag(feat, ROLE_DOBJ, feat)
    elif template_id == "attribution":
        _tag(source, ROLE_MARKER, "")
        _tag(actor, ROLE_NSUBJ, actor)
        _tag(rel, ROLE_VERB, rel)
        _tag(feat, ROLE_DOBJ, feat)
    elif template_id == "neg":
        _tag(actor, ROLE_NSUBJ, actor)
        _tag("not", ROLE_MARKER, "")
        _tag(rel, ROLE_VERB, rel)
        _tag(feat, ROLE_DOBJ, feat)
    else:  # svo
        _tag(actor, ROLE_NSUBJ, actor)
        _tag(rel, ROLE_VERB, rel)
        _tag(feat, ROLE_DOBJ, feat)

    return SynthExample(
        sentence=sentence,
        tokens=tokens,
        token_roles=token_roles,
        token_anchors=token_anchors,
        ds_mass=mass,
        evidentiality=evid,
        modality=modl,
        template_id=template_id,
        gold_triple=triple,
    )


DEFAULT_TEMPLATES = [
    "svo", "svo_prep", "svo_for",
    "hedge", "obligation", "attribution",
    "passive", "neg",
]


def generate_corpus(schema: Schema = DEFAULT_SCHEMA,
                    n: int = 5000,
                    seed: int = 42,
                    templates: list[str] | None = None
                    ) -> list[SynthExample]:
    """Generate n labeled synthetic examples.

    Templates are sampled uniformly. Each example carries full token-
    level gold labels plus sentence-level DS mass.
    """
    rng = random.Random(seed)
    templates = templates or DEFAULT_TEMPLATES
    return [_make_example(schema, rng, rng.choice(templates))
            for _ in range(n)]
