"""Graph MCTS: AlphaGo-style traversal over the infon hypergraph.

Instead of flat top-k retrieval, uses Monte Carlo Tree Search to explore
the knowledge graph, following NEXT edges and anchor links to discover
supporting/refuting evidence paths.

The "game" is: given a query, find the subgraph that maximizes belief
while minimizing uncertainty. Each MCTS node represents a cluster of
infons reached via a specific anchor path.

Key differences from Overlord's web-search MCTS:
  - No API calls: all evaluation via NLI head on local 17MB backbone
  - Graph structure: follows typed edges (NEXT, shared anchors) not web links
  - Evidence is pre-computed: infons already extracted and stored
  - Belief is grounded: DS mass functions from NLI, not LLM-assigned

MCTS Loop:
  SELECT  → UCB1 over anchor clusters (exploit: belief, explore: entropy)
  EXPAND  → follow NEXT edges & shared-anchor links from selected infons
  EVALUATE → NLI head scores each discovered infon against the query
  BACKPROP → Dempster-Shafer combination up the traversal tree
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .dempster_shafer import MassFunction, combine_dempster, combine_multiple
from .heads import CognitionHeads
from .infon import Edge, Infon

# ═══════════════════════════════════════════════════════════════════════
# MCTS NODE (traversal state, not graph node)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """A node in the MCTS traversal tree.

    Each node represents a cluster of infons discovered via a specific
    anchor path. Children are reached by following edges from this cluster.
    """
    node_id: str
    anchor_path: list[str] = field(default_factory=list)  # anchors traversed to get here
    infons: list[Infon] = field(default_factory=list)     # infons in this cluster
    edges_followed: list[Edge] = field(default_factory=list)

    # MCTS statistics
    visit_count: int = 0
    belief_mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    per_infon_masses: list[MassFunction] = field(default_factory=list)

    # Tree structure
    parent: MCTSNode | None = field(default=None, repr=False)
    children: list[MCTSNode] = field(default_factory=list)

    # Expansion state
    is_expanded: bool = False
    is_terminal: bool = False  # no more edges to follow

    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief mass (higher = more uncertain)."""
        m = self.belief_mass
        values = [v for v in [m.supports, m.refutes, m.uncertain, m.theta] if v > 0]
        if not values:
            return 0.0
        return -sum(v * math.log2(v) for v in values)

    @property
    def ucb1_score(self) -> float:
        """UCB1 score balancing exploitation and exploration."""
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.belief_mass.supports  # prefer supporting evidence
        exploration = math.sqrt(2 * math.log(parent_visits) / self.visit_count)
        return exploitation + 1.4 * exploration


# ═══════════════════════════════════════════════════════════════════════
# GRAPH MCTS ENGINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSResult:
    """Result of Graph MCTS traversal."""
    query: str
    verdict: str  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    combined_mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    iterations: int = 0
    nodes_explored: int = 0
    infons_evaluated: int = 0
    chains_discovered: list[list[str]] = field(default_factory=list)
    traversal_tree: MCTSNode | None = None
    elapsed_s: float = 0.0
    iteration_log: list[dict] = field(default_factory=list)


class GraphMCTS:
    """MCTS-based traversal over the infon hypergraph.

    Usage:
        from cognition import Cognition, CognitionConfig
        from cognition.graph_mcts import GraphMCTS

        cog = Cognition(config)
        cog.ingest(documents)

        mcts = GraphMCTS(cog.store, cog.encoder, cog.schema)
        result = mcts.search("Did Toyota's battery investment lead to gains?")
        print(result.verdict, result.combined_mass)
    """

    def __init__(self, store, encoder, schema,
                 heads: CognitionHeads | None = None,
                 max_iterations: int = 8,
                 exploration_bias: float = 1.4,
                 max_depth: int = 4,
                 min_entropy_change: float = 0.01,
                 contrary: bool = False):
        self.store = store
        self.encoder = encoder
        self.schema = schema
        self.max_iterations = max_iterations
        self.exploration_bias = exploration_bias
        self.max_depth = max_depth
        self.min_entropy_change = min_entropy_change
        self.contrary = contrary

        # Load heads for NLI evaluation. The bundled heads.pt was trained on
        # splade-tiny's 128-dim hidden state; if we're running XLM-R (768-dim)
        # the checkpoint shapes won't match and we fall back to the heuristic
        # evaluator (_evaluate_heuristic handles self.heads is None).
        if heads is not None:
            self.heads = heads
        else:
            from pathlib import Path
            heads_path = Path(__file__).parent / "model"
            self.heads = None
            if (heads_path / "heads.pt").exists():
                backbone = encoder.splade if hasattr(encoder, "splade") else encoder
                encoder_hidden = getattr(backbone.model.config, "hidden_size", None)
                try:
                    candidate = CognitionHeads.load(heads_path)
                    if encoder_hidden is None or candidate.hidden_dim == encoder_hidden:
                        self.heads = candidate
                except Exception:
                    self.heads = None

        # Cache for CLS embeddings
        self._cls_cache: dict[str, torch.Tensor] = {}

    def search(self, query: str, verbose: bool = False) -> MCTSResult:
        """Run MCTS traversal to answer a query over the hypergraph."""
        t0 = time.time()

        # Encode query
        query_activations = self.encoder.encode_single(query)
        query_cls = self._encode_cls([query])[0]

        # Initialize root: seed with top activated anchors
        root = self._create_root(query_activations)

        # Evaluate root's infons against query (initial belief)
        self._evaluate(root, query_cls)

        if verbose:
            print(f"\n  Query: \"{query}\"")
            print(f"  Root anchors: {root.anchor_path}")
            print(f"  Root infons: {len(root.infons)}")
            print(f"  Root initial: S={root.belief_mass.supports:.3f} "
                  f"R={root.belief_mass.refutes:.3f} θ={root.belief_mass.theta:.3f}")

        iteration_log = []
        entropy_history = []

        for iteration in range(self.max_iterations):
            # SELECT: pick the most promising unexpanded leaf
            selected = self._select(root)
            if selected is None:
                if verbose:
                    print(f"  Iteration {iteration+1}: no expandable nodes remaining")
                break

            # EXPAND: follow edges from selected node
            new_children = self._expand(selected, query_activations)

            # EVALUATE: NLI head scores each child's infons against query
            for child in new_children:
                self._evaluate(child, query_cls)

            # BACKPROP: DS combination back up to root
            self._backpropagate(selected)

            # Log iteration
            iter_info = {
                "iteration": iteration + 1,
                "selected_path": selected.anchor_path,
                "new_children": len(new_children),
                "root_belief": {
                    "supports": root.belief_mass.supports,
                    "refutes": root.belief_mass.refutes,
                    "theta": root.belief_mass.theta,
                },
                "root_entropy": root.entropy,
            }
            iteration_log.append(iter_info)
            entropy_history.append(root.entropy)

            if verbose:
                self._print_iteration(iteration + 1, selected, new_children, root)

            # Early stopping: entropy equilibrium (only after at least 4 iterations)
            if len(entropy_history) >= 5:
                recent = entropy_history[-3:]
                if max(recent) - min(recent) < self.min_entropy_change:
                    if verbose:
                        print(f"  → Entropy equilibrium reached at iteration {iteration+1}")
                    break

        # Final verdict via pignistic transform
        m = root.belief_mass
        total_focal = m.supports + m.refutes + m.uncertain
        if total_focal > 0:
            pig_s = m.supports + m.theta * (m.supports / total_focal)
            pig_r = m.refutes + m.theta * (m.refutes / total_focal)
        else:
            pig_s = m.theta / 3
            pig_r = m.theta / 3

        if pig_r > 0.15 and pig_r > pig_s:
            verdict = "REFUTES"
        elif pig_s > 0.25 and pig_s > pig_r:
            verdict = "SUPPORTS"
        else:
            verdict = "NOT ENOUGH INFO"

        # Collect discovered chains
        chains = self._collect_chains(root)

        elapsed = time.time() - t0
        total_infons = self._count_infons(root)

        return MCTSResult(
            query=query,
            verdict=verdict,
            combined_mass=root.belief_mass,
            iterations=len(iteration_log),
            nodes_explored=self._count_nodes(root),
            infons_evaluated=total_infons,
            chains_discovered=chains,
            traversal_tree=root,
            elapsed_s=elapsed,
            iteration_log=iteration_log,
        )

    # ── MCTS PHASES ────────────────────────────────────────────────────

    def _create_root(self, query_activations: dict[str, float]) -> MCTSNode:
        """Create root node from query's activated anchors.

        Expands parent anchors to descendants so that a query at a higher
        level (e.g. 'east_asia') seeds with infons stored at lower levels
        (e.g. 'senkaku', 'japan').
        """
        sorted_anchors = sorted(query_activations.items(), key=lambda x: -x[1])
        top_anchors = [(a, s) for a, s in sorted_anchors[:8] if s > 0.1]

        # Retrieve infons for top anchors (with descendant expansion)
        root_infons = []
        seen_ids = set()
        for anchor_name, _ in top_anchors:
            role = self.schema.role_for_type(self.schema.types.get(anchor_name, ""))
            query_names = [anchor_name]
            if self.schema.get_children(anchor_name):
                query_names.extend(self.schema.get_descendants(anchor_name))
            for name in query_names:
                anchor_infons = self.store.get_infons_for_anchor(
                    name, role=role, limit=20,
                )
                for inf in anchor_infons:
                    if inf.infon_id not in seen_ids:
                        seen_ids.add(inf.infon_id)
                        root_infons.append(inf)

        root = MCTSNode(
            node_id="root",
            anchor_path=[a for a, _ in top_anchors[:4]],
            infons=root_infons,
            visit_count=1,
        )
        return root

    def _select(self, root: MCTSNode) -> MCTSNode | None:
        """SELECT: choose the best leaf node to expand using UCB1."""
        # Find all unexpanded leaves
        leaves = []
        self._collect_leaves(root, leaves)

        if not leaves:
            return None

        # UCB1 selection
        best = max(leaves, key=lambda n: n.ucb1_score)
        return best

    def _collect_leaves(self, node: MCTSNode, leaves: list):
        """Recursively collect unexpanded leaf nodes."""
        if not node.is_expanded and not node.is_terminal:
            # Check depth limit
            depth = len(node.anchor_path) - len(node.parent.anchor_path if node.parent else [])
            if self._get_depth(node) < self.max_depth:
                leaves.append(node)
        for child in node.children:
            self._collect_leaves(child, leaves)

    def _get_depth(self, node: MCTSNode) -> int:
        """Get depth of node in tree."""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def _expand(self, node: MCTSNode, query_activations: dict[str, float]) -> list[MCTSNode]:
        """EXPAND: follow NEXT edges and shared-anchor links."""
        node.is_expanded = True
        node.visit_count += 1
        new_children = []

        # Strategy 1: Follow NEXT edges from this node's infons
        next_targets = set()
        for infon in node.infons[:10]:  # top 10 by importance
            edges = self.store.get_edges(source=infon.infon_id, edge_type="NEXT", limit=5)
            for edge in edges:
                if edge.target not in next_targets:
                    next_targets.add(edge.target)
                    target_infon = self.store.get_infon(edge.target)
                    if target_infon:
                        # Group by shared anchor
                        shared = edge.metadata.get("anchor", "")
                        child_id = f"{node.node_id}→{shared}"
                        existing = next((c for c in new_children if c.node_id == child_id), None)
                        if existing:
                            existing.infons.append(target_infon)
                            existing.edges_followed.append(edge)
                        else:
                            child = MCTSNode(
                                node_id=child_id,
                                anchor_path=node.anchor_path + [shared] if shared else node.anchor_path,
                                infons=[target_infon],
                                edges_followed=[edge],
                                parent=node,
                            )
                            new_children.append(child)

        # Strategy 2: Follow shared anchors to find related infons
        # Get anchors present in current infons but not yet in path
        current_anchors = set()
        for infon in node.infons:
            current_anchors.update([infon.subject, infon.predicate, infon.object])
        explored_anchors = set(node.anchor_path)
        new_anchors = current_anchors - explored_anchors

        # For each new anchor, retrieve connected infons
        for anchor in list(new_anchors)[:3]:  # limit branching factor
            if not anchor or anchor in explored_anchors:
                continue
            related = self.store.get_infons_for_anchor(anchor, limit=10)
            # Filter: only infons not already in this node or ancestors
            existing_ids = {inf.infon_id for inf in node.infons}
            novel = [inf for inf in related if inf.infon_id not in existing_ids]

            if novel:
                child = MCTSNode(
                    node_id=f"{node.node_id}→{anchor}",
                    anchor_path=node.anchor_path + [anchor],
                    infons=novel[:8],
                    parent=node,
                )
                new_children.append(child)

        if not new_children:
            node.is_terminal = True

        node.children.extend(new_children)
        return new_children

    def _evaluate(self, node: MCTSNode, query_cls: torch.Tensor):
        """EVALUATE: NLI head scores infons against query."""
        if not node.infons:
            node.belief_mass = MassFunction(theta=1.0)
            return

        if self.heads is None:
            # Fallback: heuristic evaluation
            self._evaluate_heuristic(node)
            return

        # Encode infon sentences
        sentences = [inf.sentence for inf in node.infons if inf.sentence]
        if not sentences:
            node.belief_mass = MassFunction(theta=1.0)
            return

        infon_cls = self._encode_cls(sentences)
        query_expanded = query_cls.unsqueeze(0).expand(len(sentences), -1)

        # NLI: does each infon support/contradict/neutral to query?
        masses = self.heads.nli.predict_mass(infon_cls, query_expanded)
        if self.contrary:
            masses = [m.invert() for m in masses]
        node.per_infon_masses = masses

        # Combine: top-5 most directional (highest supports OR refutes signal)
        decisive = sorted(masses, key=lambda m: max(m.supports, m.refutes), reverse=True)[:5]
        node.belief_mass = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)
        node.visit_count += 1

    def _evaluate_heuristic(self, node: MCTSNode):
        """Fallback evaluation without trained heads."""
        from .dempster_shafer import mass_from_confidence, mass_from_polarity
        masses = []
        for infon in node.infons[:5]:
            m1 = mass_from_polarity(infon)
            m2 = mass_from_confidence(infon)
            combined = combine_dempster(m1, m2)
            if self.contrary:
                combined = combined.invert()
            masses.append(combined)
        node.belief_mass = combine_multiple(masses) if masses else MassFunction(theta=1.0)
        node.per_infon_masses = masses
        node.visit_count += 1

    def _backpropagate(self, node: MCTSNode):
        """BACKPROP: propagate beliefs up to root via DS combination.

        Each node's belief = its own NLI evaluation combined with its
        children's beliefs. Only children with stronger directional signal
        (higher max(S,R)) than uncertain mass contribute — otherwise they
        would dilute the parent's signal via Dempster combination.
        """
        current = node
        while current is not None:
            if current.children:
                # Only accept children with clear directional signal
                # (max(S,R) > uncertain mass means it's not just "I don't know")
                informative_children = [
                    c.belief_mass for c in current.children
                    if c.visit_count > 0
                    and max(c.belief_mass.supports, c.belief_mass.refutes) > c.belief_mass.uncertain * 0.3
                ]
                if informative_children:
                    # Combine own top evidence with informative child beliefs
                    own_masses = current.per_infon_masses if current.per_infon_masses else []
                    decisive_own = sorted(
                        own_masses,
                        key=lambda m: max(m.supports, m.refutes),
                        reverse=True,
                    )[:5]
                    decisive_children = sorted(
                        informative_children,
                        key=lambda m: max(m.supports, m.refutes),
                        reverse=True,
                    )[:3]
                    all_decisive = decisive_own + decisive_children
                    if all_decisive:
                        current.belief_mass = combine_multiple(all_decisive)
            current.visit_count += 1
            current = current.parent

    # ── HELPERS ────────────────────────────────────────────────────────

    def _encode_cls(self, texts: list[str]) -> torch.Tensor:
        """Get [CLS] embeddings, with caching."""
        uncached = [t for t in texts if t not in self._cls_cache]
        if uncached:
            # Use encoder's backbone directly
            all_cls = []
            device = self.encoder.splade.device if hasattr(self.encoder, 'splade') else 'cpu'
            encoder = self.encoder.splade if hasattr(self.encoder, 'splade') else self.encoder

            for i in range(0, len(uncached), 32):
                batch = uncached[i:i + 32]
                enc = encoder.tokenizer(
                    batch, max_length=encoder.max_length, padding=True,
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=False,
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                with torch.no_grad():
                    # base_model is generic across BERT / XLM-R backbones
                    outputs = encoder.model.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    cls_emb = outputs.last_hidden_state[:, 0, :]
                    for j, text in enumerate(batch):
                        self._cls_cache[text] = cls_emb[j].cpu()

        return torch.stack([self._cls_cache[t] for t in texts])

    def _collect_chains(self, root: MCTSNode) -> list[list[str]]:
        """Collect discovered anchor chains from the traversal tree."""
        chains = []
        self._walk_chains(root, [], chains)
        return chains

    def _walk_chains(self, node: MCTSNode, path: list[str], chains: list):
        current_path = path + [a for a in node.anchor_path if a not in path]
        if not node.children:
            if len(current_path) > 1:
                chains.append(current_path)
        else:
            for child in node.children:
                self._walk_chains(child, current_path, chains)

    def _count_nodes(self, node: MCTSNode) -> int:
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _count_infons(self, node: MCTSNode) -> int:
        count = len(node.infons)
        for child in node.children:
            count += self._count_infons(child)
        return count

    def _print_iteration(self, iteration: int, selected: MCTSNode,
                          new_children: list[MCTSNode], root: MCTSNode):
        """Print iteration summary."""
        print(f"\n  ── Iteration {iteration} ──")
        print(f"  SELECT: path={selected.anchor_path[-3:]}, "
              f"entropy={selected.entropy:.3f}, "
              f"infons={len(selected.infons)}")

        if new_children:
            print(f"  EXPAND: {len(new_children)} new branches")
            for child in new_children[:3]:
                label = "supports" if child.belief_mass.supports > child.belief_mass.refutes else (
                    "refutes" if child.belief_mass.refutes > 0.1 else "neutral")
                anchor_tail = child.anchor_path[-1] if child.anchor_path else "?"
                print(f"    → {anchor_tail}: {label} "
                      f"(S={child.belief_mass.supports:.2f} "
                      f"R={child.belief_mass.refutes:.2f} "
                      f"θ={child.belief_mass.theta:.2f})")
        else:
            print("  EXPAND: terminal (no new edges)")

        print(f"  ROOT: S={root.belief_mass.supports:.3f} "
              f"R={root.belief_mass.refutes:.3f} "
              f"θ={root.belief_mass.theta:.3f} "
              f"entropy={root.entropy:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ═══════════════════════════════════════════════════════════════════════

def format_mcts_result(result: MCTSResult) -> str:
    """Format MCTS result as a readable tree structure (AlphaGo-style)."""
    lines = []
    lines.append(f"Query: \"{result.query}\"")
    lines.append("")
    lines.append("MCTS Traversal over Hypergraph:")

    root = result.traversal_tree
    if not root:
        lines.append("  (no traversal)")
        return "\n".join(lines)

    # Root info
    lines.append(f"├── Root: query anchors → {{{', '.join(root.anchor_path)}}}")
    lines.append(f"│   → {len(root.infons)} infons, "
                 f"{sum(1 for e in result.chains_discovered)} NEXT chains")
    lines.append("│")

    # SELECT phase
    if result.iteration_log:
        first = result.iteration_log[0]
        path = first["selected_path"]
        # Find the cluster with most infons
        best_cluster = path[0] if path else "?"
        lines.append("├── SELECT: highest-entropy anchor cluster")
        lines.append(f"│   → \"{best_cluster} + {path[1] if len(path) > 1 else '?'}\" "
                     f"has {len(root.infons)} infons, "
                     f"{sum(1 for e in result.chains_discovered)} NEXT chains")
        lines.append("│")

    # EXPAND phase: show discovered chains grouped by direction
    supporting_chains = []
    refuting_chains = []
    neutral_chains = []

    for child in _all_children(root):
        if child.visit_count > 0 and child.infons:
            chain = child.anchor_path
            m = child.belief_mass
            if m.supports > m.refutes and m.supports > 0.08:
                supporting_chains.append((chain, m))
            elif m.refutes > m.supports and m.refutes > 0.08:
                refuting_chains.append((chain, m))
            else:
                neutral_chains.append((chain, m))

    lines.append("├── EXPAND: follow NEXT edges from those infons")
    for chain, m in supporting_chains[:3]:
        chain_str = "→".join(chain[-4:])
        lines.append(f"│   → discovers chain: {chain_str}")
    for chain, m in refuting_chains[:2]:
        chain_str = "→".join(chain[-4:])
        lines.append(f"│   → discovers branch: {chain_str}")
    if not supporting_chains and not refuting_chains:
        for chain, m in neutral_chains[:3]:
            chain_str = "→".join(chain[-4:])
            lines.append(f"│   → discovers chain: {chain_str}")
    lines.append("│")

    # EVALUATE phase
    lines.append("├── EVALUATE: NLI head scores each chain against query")
    for chain, m in supporting_chains[:3]:
        chain_str = "→".join(chain[-3:])
        lines.append(f"│   → {chain_str}: supports ({m.supports:.2f})")
    for chain, m in refuting_chains[:3]:
        chain_str = "→".join(chain[-3:])
        lines.append(f"│   → {chain_str}: refutes ({m.refutes:.2f})")
    for chain, m in neutral_chains[:2]:
        chain_str = "→".join(chain[-3:])
        lines.append(f"│   → {chain_str}: neutral ({m.uncertain:.2f})")
    lines.append("│")

    # BACKPROP phase
    m = result.combined_mass
    lines.append("├── BACKPROP: DS combination over discovered paths")
    lines.append(f"│   → combined belief: SUPPORTS {m.supports:.2f}, "
                 f"REFUTES {m.refutes:.2f}, θ {m.theta:.2f}")
    lines.append("│")

    # NEXT ITERATION hint
    if refuting_chains:
        chain_str = "→".join(refuting_chains[0][0][-3:])
        lines.append("└── NEXT ITERATION: drill into high-uncertainty branch")
        lines.append(f"    → follow the {chain_str} chain to verify...")
    elif neutral_chains:
        chain_str = "→".join(neutral_chains[0][0][-3:])
        lines.append("└── NEXT ITERATION: drill into uncertain branch")
        lines.append(f"    → follow the {chain_str} chain for more evidence...")
    else:
        lines.append("└── CONVERGED: all branches evaluated")

    # Final summary
    lines.append("")
    lines.append(f"═══ Verdict: {result.verdict} ═══")
    lines.append(f"  belief(SUPPORTS)={m.supports:.3f}  "
                 f"belief(REFUTES)={m.refutes:.3f}  "
                 f"uncertainty={m.uncertain + m.theta:.3f}")
    lines.append(f"  {result.nodes_explored} nodes explored, "
                 f"{result.infons_evaluated} infons evaluated, "
                 f"{result.iterations} iterations in {result.elapsed_s:.2f}s")

    return "\n".join(lines)


def _all_children(node: MCTSNode) -> list[MCTSNode]:
    """Recursively collect all children."""
    result = []
    for child in node.children:
        result.append(child)
        result.extend(_all_children(child))
    return result


def _mass_label(m: MassFunction) -> str:
    if m.supports > m.refutes and m.supports > 0.1:
        return "supports"
    elif m.refutes > m.supports and m.refutes > 0.1:
        return "refutes"
    else:
        return "neutral"
