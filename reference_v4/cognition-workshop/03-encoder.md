# Module 03: The Change of Basis — SPLADE Encoder + Anchor Projection


> **Update for the cassette substrate.** The encoder itself is unchanged — SPLADE-tiny plus AnchorProjector, the same change-of-basis operator. The 17 MB bundled model now runs inside Lambda containers on demand; see module 09. For the fix to actor-as-object role competition (which unblocked actor-to-actor triple extraction), see module 04.

## What You'll Learn

- Why encoding is a **change of basis** from token space to concept space
- How [splade-tiny-msmarco](https://github.com/rasyosef/splade-tiny-msmarco) produces sparse vocabulary expansions via its MLM head
- How the AnchorProjector maps 30,522 vocab activations to your typed anchors
- Why this needs zero training — define the target basis and project

## Background: Encoding as Change of Basis

In linear algebra, a change of basis re-expresses a vector in a new coordinate system. The vector itself doesn't change — only its representation. That is exactly what happens in the cognition encoder:

```
Token basis (dim 30,522)          Concept basis (dim n)
──────────────────────────        ─────────────────────
BERT WordPiece vocabulary    ──→  Your typed anchors
"toyota" at position 16323        toyota (actor)
"invest" at position 15697        invest (relation)
"##ing" at position 2075          solid_state (feature)
...30,522 entries                 ...n entries (your schema)
```

The **change-of-basis operator** has two stages:

1. **Vocabulary expansion** (SPLADE): a learned MLM head activates semantically related vocabulary terms beyond the literal input. "The automaker plans to invest in next-gen power storage" activates `battery`, `toyota`, `electric` even though none appear in the text. This is the key: the MLM head learned, during pretraining on 1.2M passages, which vocabulary terms are *structurally implied* by a context.

2. **Anchor projection**: max-pool the expanded vocabulary vector over your schema's token IDs. Each anchor collects the strongest activation from its mapped tokens. The 30,522-dim vector collapses to your *n*-dim concept space.

```
sentence
  │
  ▼
BERT-Tiny + MLM head → 30,522-dim logits per position
  │
  ▼ log(1 + ReLU(logits)), max-pool across positions
  │
sparse vocabulary vector ∈ ℝ³⁰⁵²²   (~99.9% zeros)
  │
  ▼ AnchorProjector: max-pool by schema token IDs
  │
anchor activation vector ∈ ℝⁿ
```

The result: every sentence is now a point in your concept coordinate system. Two sentences that mean the same thing in your domain will have similar concept coordinates, even if they share no words.

## The Foundation: splade-tiny-msmarco

[splade-tiny-msmarco](https://github.com/rasyosef/splade-tiny-msmarco) is the model that makes this work. Key facts:

| Property | Value |
|----------|-------|
| Architecture | BERT-Tiny (2 layers, 128 hidden, 2 heads) + MLM head |
| Parameters | **4.4M** (15x smaller than distilbert SPLADE) |
| Training | Knowledge distillation on 1.2M MS MARCO passages |
| Teacher | ms-marco-MiniLM-L6-v2 cross-encoder |
| Vocab | 30,522 (standard BERT WordPiece) |
| Sparsity | ~99.94% zeros (avg ~18 active dims per query) |
| License | Apache 2.0 |
| Size on disk | 17MB |
| Performance | MRR@10 = 0.649 on MS MARCO (beats BM25 by 65%) |

Why this model specifically:

1. **Apache 2.0** — commercially usable (unlike Naver's SPLADE models, CC-BY-NC)
2. **Ships bundled** — 17MB, no model download, no GPU required
3. **Concept expansion** — the MLM head's superpower. It learned which vocabulary terms are *implied* by a context, not just which are *present*. This is what makes projection onto arbitrary schemas work across any domain.
4. **Distilled knowledge** — trained via knowledge distillation from a cross-encoder teacher, meaning it inherited the teacher's understanding of semantic relevance while being 15x smaller.

### Why Not a Larger SPLADE?

The 4.4M parameter model is intentionally small. The vocabulary expansion quality is what matters for schema projection — not retrieval ranking precision. Bigger models activate more vocabulary terms more aggressively, but the anchor projector only needs the *right* terms to fire, not all of them. The tiny model is the right tool: fast inference, low memory, broad semantic coverage.

## The Projection Step

The AnchorProjector is the second half of the change of basis. It bridges SPLADE's 30,522-dim vocabulary space to your typed anchor space:

```
SPLADE vector: [... toyota=3.2, car=1.8, invest=2.5, battery=2.1, ...]
                              ↓ max-pool by schema tokens
Anchor vector:  {toyota: 3.2, invest: 2.5, solid_state: 0.0, battery: 2.1, ...}
```

For each anchor in your schema, the projector looks up its token strings' positions in the BERT vocabulary and takes the **max activation**. Multi-word tokens like "solid-state" are handled by tokenizing them and taking the max across all sub-tokens.

This is a linear projection — a matrix multiply where the projection matrix is determined entirely by your schema definition. No training, no gradients, no optimization. Just token ID lookups.

## Prompt 1: Build the Encoder

---

```
Create cognition/src/cognition/encoder.py with three classes:

1. SpladeEncoder — wraps rasyosef/splade-tiny (BertForMaskedLM, 4.4M params):
   - Forward: input_ids → MLM logits → log(1 + ReLU(logits)) → max-pool
   - encode_sparse(texts, batch_size) → (n_texts, 30522) numpy array
   - Supports cuda, mps, cpu device selection
   - Ship the model bundled in cognition/src/cognition/model/

2. AnchorProjector — the change-of-basis projection matrix:
   - __init__(schema, tokenizer): for each anchor, map its tokens to
     BERT vocab IDs. Try direct conversion, subword (##token), and
     full tokenization for multi-word tokens.
   - project(sparse_vec) → {anchor_name: score}: max-pool over token IDs
   - project_batch(sparse_matrix) → list of dicts
   - project_to_matrix(sparse_matrix, anchor_names) → (n, n_anchors) array

3. Encoder — the main interface combining SPLADE + projection:
   - __init__(schema, model_name, max_length, device)
   - encode(texts) → (n_texts, n_anchors) anchor activation matrix
   - encode_sparse(texts) → raw SPLADE vectors (for debugging)
   - encode_single(text) → {anchor_name: score}
   - from_dir(path) → load from saved config directory
   - find_spans(text, anchor_name, anchor_defs) → character spans

Key: SPLADE scores are in ~[0, 5], not [0, 1]. Higher = stronger activation.
The downstream pipeline normalizes per-sentence for confidence scoring.
```

---

## Prompt 2: Visualize the Change of Basis

---

```
Load the SPLADE encoder with my schema and test on 5 diverse sentences.
For each sentence, show:
1. Top 10 anchor activations with scores and types
2. Which activations are "direct" (token literally in text) vs
   "expanded" (SPLADE inferred semantically — the basis expansion)
3. The raw SPLADE vocabulary's top 20 activated terms (not just anchors)

This reveals both what the model sees (raw vocab = token basis) and what
reaches our schema (projected anchors = concept basis). The gap between
them is the information lost in projection — schema coverage we're missing.

Also show the projection as a literal change-of-basis matrix:
a (vocab_size × n_anchors) binary matrix where entry [i,j] = 1 if
vocab token i is mapped to anchor j. Print its sparsity and rank.
```

---

## Understanding Concept Basis Expansion

This is SPLADE's superpower — and the reason schema projection works without training. For the sentence "The automaker plans to invest in next-generation power storage":

**Token basis** (raw SPLADE vocab activations, top terms):
```
invest=3.1, battery=2.8, car=2.5, automaker=2.3, energy=2.1,
storage=1.9, power=1.8, toyota=1.2, electric=1.1, ...
```

**Concept basis** (after anchor projection):
```
invest=3.1 (direct: "invest" in schema tokens)
battery=2.8 (expanded: "battery" not in text, but MLM head inferred it)
ev=1.1 (expanded: "electric" activated)
toyota=1.2 (expanded: "automaker" context → "toyota" implied)
```

Notice: "battery" isn't in the sentence, but the MLM head learned during pretraining that "power storage" implies batteries. And "toyota" activates weakly because the head learned that "the automaker" in automotive context often refers to Toyota. This learned vocabulary expansion is the change-of-basis at work — the model re-expresses the sentence in the broader vocabulary before we project to anchors.

## The Algebra

For the mathematically inclined: let **V** ∈ ℝ^{30522} be the SPLADE vocabulary space and **A** ∈ ℝ^n be your anchor space. The projection matrix **P** ∈ {0,1}^{30522 × n} maps each anchor to its vocab token IDs. Then:

```
a = P^T · max-pool(log(1 + ReLU(MLM(BERT(x)))))
```

where `a ∈ ℝ^n` is the anchor activation vector. The max-pool is applied column-wise through **P** (not a standard matmul — it's a grouped max reduction). This is a nonlinear projection, but the anchor mapping **P** itself is purely defined by your schema — no learned parameters.

The SPLADE model learns the expansion (the MLM head), and your schema defines the projection. Neither requires the other to train.

## Checkpoint

- [ ] `SpladeEncoder` loads splade-tiny and produces (n, 30522) sparse arrays
- [ ] `AnchorProjector` maps vocab → anchors with no missing critical anchors
- [ ] `Encoder.encode_single()` returns {anchor: score} dicts
- [ ] Direct mentions score higher than semantic expansions
- [ ] You can see concept basis expansion working on test sentences
- [ ] The projection matrix sparsity and shape match expectations
