# BlackMagic — Theory and Inspirations

This document threads together the theoretical lineage behind BlackMagic's
design: where each load-bearing idea comes from, why it's here, and what
it commits us to. None of these ideas are original to this project — the
contribution is the *combination*.

Organised by architectural layer, bottom up:

1. [Sparse coding — the brain analogy](#1-sparse-coding--the-brain-analogy)
2. [SPLADE — learned lexical sparsity](#2-splade--learned-lexical-sparsity)
3. [Typed anchor projection — concept bottleneck](#3-typed-anchor-projection--concept-bottleneck)
4. [Situation semantics — infons as atoms](#4-situation-semantics--infons-as-atoms)
5. [Dempster-Shafer — belief under contradiction](#5-dempstershafer--belief-under-contradiction)
6. [Graph MCTS — deliberation as tree search](#6-graph-mcts--deliberation-as-tree-search)
7. [GA imagination — combinatorial creativity](#7-ga-imagination--combinatorial-creativity)
8. [JEPA relationship — what we do and don't borrow](#8-jepa-relationship)
9. [Whitebox commitment — why not dense](#9-whitebox-commitment)

---

## 1. Sparse coding — the brain analogy

**The idea**: your brain doesn't represent a concept by activating every
neuron that could possibly relate to it. For any given stimulus, roughly
1–5% of cortical neurons fire. The rest are silent. This is sparse coding,
measured since the 1990s (Olshausen & Field 1996, Willmore & Tolhurst
2001), and it holds up across species and cortical regions.

**Why it matters computationally**:

- **Energy efficiency.** The brain runs on ~20W. If every neuron fired
  for every input, cortical metabolism would exceed what the skull can
  dissipate. Sparsity is not an optimisation — it's a biological
  precondition.
- **Generalisation.** Sparse representations filter jitter. By committing
  to a small number of dimensions per stimulus, the system ignores
  irrelevant variation and recovers invariants.
- **Interpretability.** When only a few dimensions are active, the
  representation names itself — you can read off which features the system
  "thinks" matter.

**What BlackMagic borrows**: the activation pattern. Every sentence
projects to a typed anchor space where only ~5–15 anchors fire above
threshold. The 90%+ silent anchors are the majority, exactly as in cortex.

**What BlackMagic does NOT claim**: that we're simulating neurons. The
brain's sparse code is *learned* under metabolic competitive pressure;
BlackMagic's is *imposed* via the schema + thresholding. The activation
pattern is brain-like; the underlying mechanism is not.

**Useful references**:
- Olshausen, B. A. & Field, D. J. (1996). Emergence of simple-cell
  receptive field properties by learning a sparse code for natural images.
  *Nature* 381, 607–609.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience* 11, 127–138. (The predictive-coding /
  surprise-minimisation framing that motivates why sparsity *matters*
  beyond just efficiency.)

---

## 2. SPLADE — learned lexical sparsity

**The idea**: use a masked language model (BERT) to produce a sparse
activation pattern over its vocabulary for each input, then do retrieval
on these sparse vectors as if they were bags-of-words. Combines the
recall of dense retrieval with the interpretability and inverted-index
speed of BM25.

SPLADE's activation: `log(1 + ReLU(MLM_logits))` max-pooled across token
positions. The logits are the predictions the MLM would make for each
vocab token; log-ReLU pushes uninformative predictions toward zero while
preserving the activation pattern for predictions the model considers
plausible.

**Why the log-ReLU?** Without it, activation counts are dominated by a
handful of high-mass tokens and the "expansion" vocabulary stays silent.
The log compresses, the ReLU sparsifies. FLOPS regularization during
training completes the picture by explicitly penalising dense output.

**Variants and lineage** used as context by BlackMagic:

- **SPLADE v1/v2/v3** (Naver) — the canonical research checkpoints.
  CC-BY-NC-SA licensed, EN-only. The v3 recipe uses MarginMSE + KL
  distillation from a cross-encoder teacher, 8 hard negatives per query,
  aggressive FLOPS regularization.
- **Splade_PP_en_v2** (prithivida) — Apache-2.0, v3-quality, fewer
  negatives (5/query), tighter FLOPS schedule, inspired by Google's
  SparseEmbed.
- **opensearch-neural-sparse-encoding-multilingual-v1** — Apache-2.0,
  XLM-R base, but its sparse activations stay in the input language's
  vocabulary. Cross-lingual alignment requires a dedicated fine-tune.
- **rasyosef/splade-tiny** — 4.4M-param distilled student. The bundled
  model. Surprisingly competitive given its size: 30.9 MRR@10 on MS MARCO,
  within 80% of splade-v3's quality at 15× smaller.

**What BlackMagic ships**: splade-tiny bundled as-is (17MB). The EN-only
commitment means we can use this tiny model directly without the
multilingual vocab tax.

**Useful references**:
- Formal, T., Lassance, C., Piwowarski, B., & Clinchant, S. (2022).
  From Distillation to Hard Negative Sampling: Making Sparse Neural IR
  Models More Effective. *SIGIR*. (arXiv:2205.04733)
- Lassance, C. et al. (2024). SPLADE-v3. arXiv:2403.06789.

---

## 3. Typed anchor projection — concept bottleneck

**The idea**: SPLADE gives you ~30k vocab dimensions of sparse activation.
That's a *literal* change of basis from dense embedding space to sparse
vocabulary space. BlackMagic takes one more step: project those 30k
vocab activations onto a much smaller typed anchor space (usually
~100–400 anchors), where each anchor has:

- a **name** (human-readable concept label)
- a **type** (actor / relation / feature / location / market)
- a list of **vocab tokens** that it's defined by

The projection is max-pooling: `anchor_score = max(sparse_vec[tok_id]
for tok_id in anchor.token_ids)`.

**Why the types matter**: downstream reasoning (triple extraction, MCTS,
imagination grammar) depends on role-typed anchors. An `actor` anchor can
be a triple subject; a `relation` is the predicate; a `feature` or
`location` is the object. Without types, you can't even state the
grammar that says "subject must be an actor."

**Lineage — concept bottleneck models**:

- Koh et al. (2020), *Concept Bottleneck Models*. Supervised training of
  a network through an intermediate layer of human-labeled concepts.
  Interpretable by construction.
- Marconato et al. (2023), *Concept Bottleneck Networks*. Extensions to
  unsupervised concept discovery.

BlackMagic's typed-anchor layer is essentially a *hand-specified* concept
bottleneck: the user writes the schema, and the model is forced to route
activations through those concepts before anything downstream sees them.
This is interpretability-by-construction, not post-hoc explanation.

**What we give up**: the flexibility of learned representations. The schema
is fixed at ingestion time; if a new concept emerges in the corpus that
isn't in the schema, the system can't represent it directly — only as
weak activations over adjacent anchors. This is a deliberate tradeoff.

---

## 4. Situation semantics — infons as atoms

**The idea**: extract structured facts from text, not embeddings. Each
"infon" is a typed tuple `<<predicate, subject, object; polarity>>`
representing one atomic assertion in the situation described.

**Lineage**:

- Barwise & Perry (1983), *Situations and Attitudes*. Introduced situation
  semantics as an alternative to possible-world semantics — meaning is
  anchored to situations (partial worlds), not truth-in-every-world.
- Devlin (1991), *Logic and Information*. Formalised the infon notation
  `<< R, a₁, ..., aₙ ; pol >>` where R is a relation, aᵢ are its
  arguments, and pol is the polarity.
- Seligman & Moss (1997). Situated information theory: infons as the
  quanta of information carried by a situation.

**What BlackMagic borrows**: the infon format and the commitment to
*grounding* — every infon in BlackMagic has a pointer back to the
sentence and character span it was extracted from. This is the
cognition package's "extractive" commitment: nothing in the knowledge
graph exists without a source.

**What's different from classical situation semantics**: BlackMagic infons
carry confidence scores, activation levels, and are linked into a temporal
graph. Classical situation theory is logical; our infons are statistical.
The spirit (structured, typed, grounded) is preserved; the metatheory
(model-theoretic vs probabilistic) is different.

---

## 5. Dempster-Shafer — belief under contradiction

**The idea**: when multiple sources provide conflicting evidence for a
claim, you can't just average their opinions. Averaging treats
"one says yes, one says no" as "maybe" — but if both sources are
high-confidence, the right conclusion is "we have a genuine
disagreement," not "we have uncertainty."

Dempster-Shafer theory (DST) gives each source a **mass function**
distributing belief over the power set of outcomes:

```
m : 2^Θ → [0, 1],   Σ m(A) = 1
```

For BlackMagic's frame Θ = {SUPPORTS, REFUTES, UNCERTAIN}, each piece of
evidence produces a 4-tuple `(m(S), m(R), m(U), m(θ))` where m(θ) is the
mass on the whole frame (total ignorance). Dempster's rule combines two
mass functions into one, redistributing conflict mass.

**Why DST and not Bayes**: Bayesian methods require a prior and conflate
ignorance with uniform belief. DST lets you start from "I don't know"
(m(θ) = 1) and build up mass as evidence arrives, without committing to
a prior distribution. It's the right tool when you have *some* evidence
but not enough for a proper posterior.

**Where it lives in BlackMagic**:

- `verify_claim()` — for a claim, retrieve supporting/refuting infons,
  derive a mass function from each, combine via Dempster's rule, emit a
  verdict.
- `MCTSResult.combined_mass` — MCTS's tree search backpropagates DS
  masses, not scalar scores.
- **Imagination's logic_penalty** — an imagined infon that contradicts a
  high-confidence constraint gets its fitness multiplicatively reduced
  by the DS refutation signal.

**Lineage**:
- Shafer, G. (1976). *A Mathematical Theory of Evidence*. The founding
  text. Dempster's rule is Chapter 3.
- Smets & Kennes (1994). *The Transferable Belief Model*. A refined
  version that drops DST's normalisation step and handles conflict more
  gracefully in high-disagreement cases.

---

## 6. Graph MCTS — deliberation as tree search

**The idea**: given a query against a structured knowledge graph, use
AlphaGo-style Monte Carlo Tree Search to explore multi-hop chains of
inference. The four-phase loop — SELECT (UCB1), EXPAND (follow edges),
EVALUATE (score the leaf), BACKPROP (update ancestors) — operates over
anchor paths rather than board positions.

**Why MCTS specifically**:

- Scales to branching factors and depths where full exhaustive search is
  intractable.
- Doesn't require a heuristic — UCB1 balances exploration and
  exploitation automatically.
- Produces a **traversal tree** as output, not just a verdict. You can
  inspect which paths were explored, which were pruned, and why.
- Naturally composes with Dempster-Shafer: leaf evaluations produce mass
  functions, which backpropagate as DS combinations.

**Cognitive analogy**:

The MCTS loop maps onto what psychology calls System 2 deliberation
(Kahneman 2011):

- SELECT ↔ attention allocation — which thread to explore next
- EXPAND ↔ spreading activation (Collins & Loftus 1975) — which
  associations to consider
- EVALUATE ↔ coherence-checking — does this chain make sense given what
  I believe?
- BACKPROP ↔ belief updating — propagate new evidence up the dependency
  tree

The analogy breaks in one important way: real deliberation is massively
parallel, not a single-trajectory tree walk. Multiple candidate
explanations are entertained in parallel cortical circuits.

---

## 7. GA imagination — combinatorial creativity

**The idea**: imagination is not generation from scratch — it's
*recombination* of existing structure. Given a knowledge graph of
observed infons, propose plausible counterfactuals by mutating and
recombining real triples under a fitness function that rewards novelty
within coherence.

This is BlackMagic's one genuinely novel contribution: a query-scoped
genetic algorithm over typed infons with a three-term cost function.

**The cost function**:

```
fitness = grammar × (1 + logic_penalty) × health
```

- **grammar** (hard 0/1): the proposed triple must be schema-well-formed.
  Subject is actor-type, predicate is relation-type, object is non-actor.
- **logic_penalty** (soft [-1, 0]): imagined triples that contradict
  high-confidence existing constraints lose fitness, but don't go to
  zero — a strong contradiction *is* informative.
- **health** (soft [0, 1]): the product of three components:
  - **whisper**: every role anchor must have at least one corpus
    occurrence. Prevents pure hallucination.
  - **bridge**: reward imagined triples that connect previously
    non-adjacent subgraphs. Creativity is about novel connections.
  - **persona_align**: predicates that match the active persona's
    positive_predicates score higher. A persona lens for imagination.

**Mutation operators**:

1. **Hierarchy walk** — replace a role with its parent, sibling, or child
   in the schema
2. **Predicate substitution** — swap predicate with another relation
3. **Polarity flip** — invert polarity (counterfactual negation)
4. **Role recombination** — cross two observed infons
5. **Temporal projection** — propose the same triple at a future timestamp

**Lineage**:

- **Genetic algorithms**: Holland (1975), *Adaptation in Natural and
  Artificial Systems*. The original framework. Crossover, mutation,
  selection.
- **Combinatorial creativity**: Boden (1990), *The Creative Mind*. Her
  H-creativity vs P-creativity distinction maps cleanly: "novel to the
  knowledge graph" (H) vs "novel to the algorithm's current exploration"
  (P). BlackMagic's bridge score rewards H-creativity.
- **Counterfactual reasoning**: Pearl (2009), *Causality*. The three-rung
  ladder — association, intervention, counterfactual. BlackMagic's
  imagination operates at rung 3: "what if this hadn't happened?" via
  polarity flip; "what else might have happened instead?" via mutation.
- **Conceptual blending**: Fauconnier & Turner (2002). Novel concepts
  arise from the selective combination of features from two source
  spaces. This is what role_recombination implements.
- **Mental simulation / imagination**: Dennett (1991), *Consciousness
  Explained*; Schank (1982), *Dynamic Memory*. Imagination as structured
  recombination of memory traces.

**Why "genetic" and not pure enumeration**: the search space is
combinatorially large (all possible typed triples). Uniform random
sampling is wasteful; GA uses the fitness landscape to concentrate
exploration on promising regions. Importantly, fitness here isn't
single-objective: grammar is a gate, logic is penalty, health is reward.
A triple can score highly on health but contradict the KB (high
logic_penalty) — that's a genuinely interesting counterfactual.

**Why the MCTS-compatible output shape**: imagination and deliberation
are cognitively adjacent — both explore trees over structured knowledge.
By producing `ImaginationResult` isomorphic to `MCTSResult`, the same
renderers, audit tools, and downstream analysis work on both. The dual
verdicts (imagination-native PLAUSIBLE/CONTRADICTED/SPECULATIVE +
MCTS-compatible SUPPORTS/REFUTES/UNCERTAIN) let either vocabulary be used.

---

## 8. JEPA relationship

**JEPA** (Joint-Embedding Predictive Architecture, LeCun 2022) is a
framework for learning world models by predicting future embeddings from
past ones. Recent examples — LeWorldModel, V-JEPA 2, DINO-WM — train
end-to-end from raw pixel sequences to learn a compact latent space in
which dynamics are easy.

**What BlackMagic shares with JEPA**:

- Commitment to operating in a compact representation space, not raw
  input space
- Valuing predictability of the representation as a design constraint
- Treating "interesting" as "not yet known but coherent"

**What BlackMagic does NOT share**:

- JEPA learns its latent space end-to-end; BlackMagic's space is
  hand-specified via the schema.
- JEPA predicts next-state embeddings; BlackMagic has no action or
  timestep notion at the embedding level.
- JEPA uses anti-collapse regularisation (SIGReg, EMA, stop-gradient)
  because the latent space is learned; BlackMagic doesn't need this
  because our "collapse" mode (all anchors fire equally) is prevented
  constructively by the schema + threshold.

**The honest comparison**: JEPA is the right architecture for perceptual
world modelling from raw sensor streams. BlackMagic is the right
architecture for structured reasoning over human-readable concepts. They
are complementary, not competing. A future system might use a JEPA-style
perception layer feeding into a BlackMagic-style reasoning layer.

**Reference**:
- Balestriero & LeCun (2025). *LeJEPA: provable and scalable
  self-supervised learning without the heuristics.* arXiv:2511.08544.

---

## 9. Whitebox commitment

A design commitment that runs through every module: every output of
BlackMagic should trace back to its input without post-hoc explanation.

Concretely:

- An infon traces to a sentence and character spans.
- An anchor activation traces to vocab tokens that lit up.
- A constraint traces to the infons that aggregate into it.
- An MCTS verdict traces to the traversal tree that produced it.
- An imagined triple traces to the observed triples it mutated from,
  plus the fitness components that scored it.

This is **not the same as full mechanistic interpretability**. We still
don't know *why* the transformer activated `battery` rather than `motor`
on a particular sentence — that's a function of MLM weights and remains
opaque. What we do know is *which* tokens drove *which* anchors, and we
can surface that trace to a user or auditor.

**Why not dense**: dense embeddings (LaBSE, E5, sentence-transformers)
give you a 384- or 768-dim vector where each dimension is uninterpretable.
You can compute cosine similarity, but you can't answer "which
dimensions fired for this input and what do they mean?" The dimensions
don't mean anything individually.

**Why sparse-lexical**: SPLADE dimension 4892 *is* "battery." You can
run `tokenizer.convert_ids_to_tokens(4892)` and get the word back. The
representation names itself. This makes every downstream decision
auditable in a way dense representations simply cannot be.

**What this commits us to**:
- No dense fallbacks. If sparse activation fails, that's a failure to be
  investigated, not a signal to add a dense backup layer.
- No learned re-ranking that loses the provenance trace.
- No representation operations that require post-hoc explanation.
- Yes to Anchor hierarchy expansion (a parent-anchor query expands to
  its descendants — traceable).
- Yes to imagination (every imagined triple records its parents and
  fitness components — traceable).

This commitment is what distinguishes BlackMagic from the retrieval
mainstream. Most production retrieval is dense. The few sparse systems
that exist (BM25, SPLADE, Pyserini) mostly treat sparsity as a speed
optimisation. BlackMagic treats it as an epistemic commitment.

---

## Summary

| Layer | Idea | Lineage | What BlackMagic commits to |
|---|---|---|---|
| Activation | Sparse coding | Olshausen-Field; Friston | ~5–15 anchors fire per sentence |
| Encoding | SPLADE + FLOPS | Formal-Lassance 2022 | splade-tiny bundled, English only |
| Representation | Typed concept bottleneck | Koh et al 2020 | Schema is the contract |
| Atoms | Infons | Barwise-Perry; Devlin | Every fact is grounded |
| Belief | Dempster-Shafer | Shafer 1976 | Mass functions, not probabilities |
| Reasoning | MCTS | Silver et al 2016 | Tree search over anchor paths |
| Imagination | GA + fitness | Holland 1975; Boden 1990; Pearl 2009 | Query-scoped, MCTS-shaped output |
| Whole system | Whitebox interpretability | — | Every output traces to inputs |

None of these choices are optimal in isolation. Learned dense retrieval is
often higher recall than sparse. JEPA-style world models handle temporal
dynamics better. Bayesian methods are better when you can commit to
priors. The contribution of BlackMagic is the *combination*: a system
where sparse + typed + grounded + interpretable compose to produce a
retrieval-and-reasoning stack that is inspectable end-to-end.

The cost: narrower than unconstrained ML. The benefit: every output is
auditable by construction.
