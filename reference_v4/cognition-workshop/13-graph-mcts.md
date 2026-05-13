# Module 13: Graph MCTS — Tree Search Over the Knowledge Graph


> **Update for the cassette substrate.** MCTS shipped in `cognition.cassette.reason_path` with three design changes from this module's original spec: (1) chain mass uses min/max conjunction not Dempster across edges; (2) edges are grouped by (s, p, o) triple so later retractions cancel earlier affirmations via Dempster's conflict normalization; (3) connective-predicate filter at expansion time, not scoring time — prevents graph-coincidental chains through `mention` edges. See `cassette.reason_path`'s BUILD NOTES for rationale, and notebook 12 for the sheaf GNN terminal scorer wired on top.

## What You'll Learn

- How Monte Carlo Tree Search (MCTS) explores the infon hypergraph to find evidence chains
- How UCB1 selection balances exploitation (following strong evidence) with exploration (checking new paths)
- How NEXT edges and shared anchors define the expansion frontier for graph traversal
- How Dempster-Shafer combination during backpropagation produces calibrated multi-hop verdicts
- Why graph traversal beats flat retrieval for claims that require reasoning across documents

## Background

Flat retrieval answers "which single document best matches this claim?" — but many claims require chaining facts across multiple documents:

> "Did Toyota's battery investment lead to market share gains?"

No single document states this directly. One document says Toyota invested in batteries. Another says their EV market share grew. A third mentions the timeline. The answer emerges by **traversing** the graph — following NEXT edges and shared anchors between infons.

This is exactly what AlphaGo's MCTS does, adapted from game trees to knowledge graphs:

1. **SELECT**: Pick the most promising unexplored path (UCB1 score)
2. **EXPAND**: Follow NEXT edges and shared anchors to discover new infons
3. **EVALUATE**: Run NLI head to assess each infon against the claim
4. **BACKPROPAGATE**: Combine child masses into parent via Dempster's rule

The result: a tree of explored evidence paths, with belief masses propagated from leaves to root, giving a calibrated verdict that accounts for supporting *and* refuting evidence chains.

## Prompt 1: MCTS Node and Tree Structure

---

```
Build Graph MCTS in cognition/src/cognition/graph_mcts.py.

MCTSNode dataclass:
  - infon_id: str
  - infon_text: str (the sentence this infon was extracted from)
  - belief_mass: MassFunction (DS mass from evaluation)
  - children: list[MCTSNode]
  - visits: int
  - parent: Optional[MCTSNode]
  - depth: int
  - expansion_source: str ("next_edge" | "shared_anchor" | "root")

MCTSResult dataclass:
  - query: str
  - verdict: str ("SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO")
  - combined_mass: MassFunction (root's belief after backprop)
  - iterations: int
  - nodes_explored: int
  - infons_evaluated: int
  - max_depth_reached: int
  - chains_discovered: list[list[str]] (evidence chains as text)

GraphMCTS class:
  - __init__(store, encoder, schema, heads=None,
             max_iterations=10, max_depth=4,
             exploration_weight=1.414)
  - search(query: str) → MCTSResult
  - Configurable stopping: early-stop when root entropy < 0.3
    for 3 consecutive iterations (minimum 5 iterations)

UCB1 score for node selection:
  ucb1 = exploitation + exploration_weight * sqrt(ln(parent.visits) / visits)
  exploitation = max(belief_mass.supports, belief_mass.refutes)

Test: build a small graph with 5 Toyota/battery documents,
run MCTS, verify it explores > 1 node and produces a verdict.
```

---

### Why MCTS for Knowledge Graphs?

The infon hypergraph can have thousands of nodes. Exhaustive evaluation is expensive (each node requires an NLI forward pass). MCTS focuses computation on the most promising paths — spending more time exploring branches with strong evidence and less time on dead ends. The UCB1 formula ensures we don't get stuck in a single branch but still prioritize known good paths.

## Prompt 2: Expand and Evaluate

---

```
Implement the EXPAND and EVALUATE phases:

EXPAND — _expand(node, query_cls) → list[MCTSNode]:
  Two expansion strategies:

  1. NEXT edges: follow temporal chains from this infon
     - Query store for edges where source = node.infon_id, type = "NEXT"
     - Each target infon becomes a child node
     - These represent "what happened next" in the timeline

  2. Shared anchors: find infons that share anchors with this one
     - Extract anchors from current infon (subject, predicate, object)
     - Query store for other infons containing those anchors
     - Filter to avoid already-visited infons (no cycles)
     - These represent "related facts from different documents"

  Combine both strategies. Limit to top 5 children per expansion
  (scored by anchor overlap with query).

EVALUATE — _evaluate(node, query_cls) → updates node.belief_mass:
  - Encode the node's infon text via encoder
  - Run NLI head: (query_cls, infon_cls) → mass function
  - Apply calibration: theta reflects NLI confidence
  - Sort evaluation results by max(supports, refutes) descending
  - Pick the most "decisive" mass (highest directional signal)

Important: evaluate the ROOT NODE immediately after creation.
The root is the initial retrieval result — it has direct evidence.

Test:
- Verify NEXT edges produce children
- Verify shared anchors produce children  
- Verify no duplicate nodes (cycle prevention)
- Verify evaluate produces non-trivial mass (not all theta)
```

---

### The Expansion Frontier

NEXT edges give you temporal chains — "Toyota invested in batteries in 2023" → NEXT → "Toyota's EV share grew in 2024." This is chronological reasoning.

Shared anchors give you lateral connections — "Toyota invested in batteries" shares the "battery" anchor with "Tesla maintains cost advantage in battery production." This is cross-document reasoning.

Together, they define a rich frontier for MCTS to explore.

## Prompt 3: Select and Backpropagate

---

```
Implement SELECT and BACKPROPAGATE:

SELECT — _select(root) → MCTSNode:
  - Start at root
  - While node has children: pick child with highest UCB1 score
  - Return the selected leaf (for expansion)
  - If leaf already expanded and all children visited: select deepest
    unvisited child via UCB1 recursively

BACKPROPAGATE — _backpropagate(node):
  - After evaluating children, propagate their masses up the tree
  - For each child: combine child.belief_mass into parent via Dempster's rule
  - BUT: filter children before combining:
    * Only accept children where max(supports, refutes) > uncertain * 0.3
    * This prevents "noise" children (mostly uncertain) from diluting
      the parent's signal
  - Increment visits on all ancestors

  The filtering is critical: without it, neutral/uncertain children
  wash out strong evidence at the parent via Dempster combination.

Early stopping:
  - Track root entropy across iterations
  - If entropy < 0.3 for 3 consecutive iterations AND iterations >= 5:
    stop early (verdict is stable)
  - Maximum iterations cap always applies

Test:
- Build a tree manually, verify backprop combines correctly
- Verify filtering prevents dilution (parent with S=0.5 shouldn't
  become S=0.3 after adding a child with U=0.8)
- Verify UCB1 balances: visited nodes get lower scores
```

---

### The Dilution Problem

Without filtering, MCTS degrades:

```
Root: S=0.49, R=0.12  (strong support from direct retrieval)
  └── Child1: S=0.05, U=0.84  (neutral/irrelevant)
  └── Child2: S=0.03, U=0.90  (neutral/irrelevant)
  
After naive backprop: Root → S=0.22  (signal destroyed!)
```

The fix: only combine children that have actual directional evidence. A child with max(S,R) < uncertain*0.3 is noise — skip it during backprop.

## Prompt 4: Output Formatting and Comparison

---

```
Build the output formatter and flat-retrieval comparison:

format_mcts_result(result: MCTSResult) → str:
  Produce a tree-format display:

  ═══ MCTS Traversal ═══
  Query: "Did Toyota's battery investment lead to market share gains?"
  
  Root: toyota_battery_invest (NEXT→ 2 children, ANCHOR→ 1)
    ├── toyota_ev_share_growth (via NEXT)
    │   └── tesla_cost_advantage (via anchor: battery)
    └── toyota_recall_cooling (via NEXT)
  
  Evaluation:
    Belief: S=0.287  R=0.199  U=0.321  Θ=0.193
    Verdict: SUPPORTS (pignistic: 0.41)
    Confidence: low (spread across hypotheses)
    Iterations: 5, Nodes: 4, Depth: 2

  Evidence chains:
    1. Toyota invested 13B in battery tech → EV share grew 3%→5%
    2. Toyota recalled vehicles (cooling issues) — refuting signal

Also build compare_with_flat(query, store, encoder, schema):
  - Flat: retrieve top-5 infons by cosine similarity, average their polarity
  - MCTS: full tree search with DS combination
  - Report both verdicts side by side, showing:
    * Flat tends to be overconfident (no refuting evidence found)
    * MCTS explores refuting branches and gives calibrated belief

Test end-to-end:
- "Did Toyota invest in batteries?" → both SUPPORT, MCTS more calibrated
- "Did Toyota's battery research fail?" → flat may miss refuting evidence,
  MCTS finds both supporting and refuting chains
```

---

### Why MCTS Beats Flat Retrieval

Flat retrieval finds the top-K most similar documents and treats them all as supporting evidence. It has two failure modes:

1. **Overconfidence**: if 4/5 retrieved docs mention batteries positively, flat says "strongly supports" — even if a refuting doc exists elsewhere in the graph
2. **Single-hop**: flat can't chain facts. "Invested in batteries" + "EV share grew" requires two hops.

MCTS fixes both: it actively explores the graph including paths that lead to refuting evidence, and it follows multi-hop chains via NEXT edges. The result is a calibrated belief that reflects the true state of evidence.

## Prompt 5: Integration and Benchmarking

---

```
Integrate Graph MCTS into the cognition package:

1. Add to cognition.__init__.py exports:
   - GraphMCTS, MCTSResult, format_mcts_result

2. Add query mode to the main API:
   - cog.verify(claim) → MCTSResult
   - Internally: builds MCTS, runs search, returns result

3. Build a demo script (demo_graph_mcts.py):
   - Ingest 15 automotive documents covering battery investments,
     market share changes, recalls, and partnerships
   - Run 5 verification queries showing different verdict types
   - Compare MCTS vs flat for each
   - Print the tree-format output

4. Benchmark preparation:
   - evaluate_fever_mcts(n=500): run MCTS on FEVER claims
   - evaluate_hover_mcts(n=200): run MCTS on HoVer multi-hop claims
   - Report: accuracy, average iterations, average depth,
     comparison with flat retrieval accuracy
   - HoVer should show bigger improvement (multi-hop claims
     benefit most from graph traversal)

Test the full integration: ingest → consolidate → verify → verdict.
```

---

## The Complete Evidence Pipeline

After this module, claims flow through the full stack:

```
Claim
  ↓
SPLADE encode → anchor projection (change of basis)
  ↓
Retrieve seed infons (standard query)
  ↓
MCTS SELECT → UCB1 picks promising branch
  ↓
MCTS EXPAND → follow NEXT edges + shared anchors
  ↓
MCTS EVALUATE → NLI head → calibrated mass function
  ↓
MCTS BACKPROPAGATE → Dempster combination (filtered)
  ↓
Repeat until convergence or max iterations
  ↓
Pignistic transform → verdict + confidence
```

Each layer adds structure: SPLADE gives you concept-space coordinates, the graph gives you connections, MCTS gives you multi-hop reasoning, DS gives you calibrated belief, and the NLI head provides the evaluation signal. No LLM needed — 17MB of parameters does it all.

## Checkpoint

- [ ] MCTSNode and MCTSResult dataclasses work correctly
- [ ] EXPAND finds children via both NEXT edges and shared anchors
- [ ] EVALUATE produces calibrated mass functions (theta reflects confidence)
- [ ] SELECT uses UCB1 to balance exploration/exploitation
- [ ] BACKPROPAGATE filters noise children before combining
- [ ] Early stopping triggers when entropy stabilizes
- [ ] Root node is evaluated immediately (not left at theta=1.0)
- [ ] format_mcts_result produces readable tree output
- [ ] MCTS gives more calibrated verdicts than flat retrieval
- [ ] Multi-hop claims (HoVer-style) benefit from graph traversal
- [ ] Full pipeline: ingest → consolidate → verify → verdict works end-to-end
