# Module 01: Domain Schema


> **Update for the cassette substrate.** The schema concepts below are unchanged — typed anchors, hierarchy, tokens. The user-facing API now activates schemas via `InfonStore(root, schema_path="...")` or `store.set_schema_from_dict({...})`. See module 05 for how schemas tag cassettes with a content-hash `schema_ref`.

## What You'll Learn

- What anchors are and why typed vocabularies matter
- How to design an ontology for your specific domain — **the target basis** for a change of basis from token space to concept space
- The role hierarchy plays in multi-level queries

## Background

A **schema** is a typed vocabulary of **anchors** — the coordinate axes of your concept space. In linear algebra terms, you are defining the **target basis**: the *n*-dimensional coordinate system that every sentence will be projected into. Module 03 will show how SPLADE performs the actual change of basis from BERT's 30,522-dim token space into these coordinates. For now, the job is to choose those coordinates well.

Every piece of knowledge extracted from text maps to exactly three anchors filling three roles:

```
subject (actor) ── predicate (relation) ── object (feature/market/entity)
```

For example, in automotive intelligence:
- **Actors**: toyota, bmw, tesla, panasonic
- **Relations**: invest, launch, partner, compete, acquire
- **Features**: solid_state, autonomous_driving, ev_platform
- **Markets**: north_america, china, europe

Each anchor has:
- A **type** (actor, relation, feature, market, location, temporal)
- **Tokens** — the words that trigger this anchor in text
- Optional **hierarchy** — parent/child relationships for multi-level queries

## Prompt 1: Design Your Schema

Copy this prompt to Claude Code. Replace `[YOUR DOMAIN]` with your domain.

---

```
I want to build a knowledge extraction system for [YOUR DOMAIN].

Help me design a typed anchor schema with these requirements:
1. Actor anchors (15-30): the key entities/organizations/people in this domain
2. Relation anchors (8-15): the actions/relationships between entities
3. Feature anchors (20-40): technologies, products, capabilities, concepts
4. Market anchors (5-15): geographic or segment markets
5. Location anchors (5-10): key geographic locations

For each anchor, define:
- name: lowercase_with_underscores
- type: actor | relation | feature | market | location
- tokens: list of words/phrases that should trigger this anchor in text
- parent: optional parent anchor for hierarchy

Save this as data/schema.json in the format:
{
  "anchor_name": {
    "type": "actor",
    "tokens": ["word1", "word2"],
    "parent": "parent_anchor_or_null"
  }
}

Start with ~80 anchors total. We can expand later.
```

---

## Prompt 2: Validate and Expand

After Claude generates the schema, review it and refine:

---

```
Read data/schema.json and check:
1. Are there any anchors with overlapping tokens that could cause confusion?
2. Are the hierarchy relationships forming clean trees (no cycles)?
3. Are there important domain concepts missing?
4. Do the relation anchors cover both positive and negative actions?

Fix any issues and add 10-20 more anchors for concepts we missed.
Show me the final anchor count by type.
```

---

## What Good Looks Like

A well-designed schema has:
- **Distinct tokens**: "invest" and "investment" go on one anchor, not two
- **Balanced types**: roughly 4:1:3:1 ratio of actors:relations:features:markets
- **Useful hierarchy**: `solid_state → batteries → energy_storage` enables "show me everything about energy storage"
- **Negation pairs**: if you have `invest`, consider whether `divest` needs its own anchor
- **Token coverage**: each anchor should have 2-5 token variants (singular, plural, abbreviation)

## Checkpoint

Before moving on, verify:
- [ ] `data/schema.json` exists and is valid JSON
- [ ] At least 60 anchors across all types
- [ ] Every anchor has at least one token
- [ ] No circular parent references
- [ ] You can describe what each anchor means in your domain
