"""Category-theoretic extensions for cognition.

Three constructions from category theory applied to knowledge graphs:

1. **Sheaf coherence** — a presheaf on the anchor graph measures local-to-global
   consistency of co-activation patterns. Replaces placeholder coherence=0.0
   in importance scoring with a real topological signal.

2. **Functorial data migration** — a functor F: Schema_old → Schema_new
   pushes forward all infons, constraints, and edges along the mapping.
   Schema evolution without re-ingesting documents.

3. **Left Kan extension (schema-free discovery)** — given SPLADE activations
   on a corpus with no schema, compute spectral clustering on the co-activation
   matrix to discover the "natural" anchor categories from data.
"""

from __future__ import annotations

import math
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from .infon import Infon, Edge, Constraint
from .schema import AnchorSchema
from .encoder import Encoder, SpladeEncoder


# ═══════════════════════════════════════════════════════════════════════
# 1. SHEAF COHERENCE
# ═══════════════════════════════════════════════════════════════════════

class SheafCoherence:
    """Compute sheaf coherence on the anchor co-activation graph.

    A sheaf assigns to each open set (subset of anchors) a vector space of
    consistent activations. Coherence measures whether local observations
    (per-sentence anchor scores) glue into global sections (corpus-level facts).

    High coherence = anchors that co-activate in a sentence genuinely belong
    together (the triple is structurally sound).

    Low coherence = anchors activated by coincidence, not structural co-occurrence.

    The construction:
    1. Build NPMI co-activation matrix from corpus (the site)
    2. For each infon, measure how well its (S, P, O) anchors are connected
       in the co-activation graph (local section consistency)
    3. Compare against the Laplacian spectrum to get a global coherence score
    """

    def __init__(self, anchor_names: list[str]):
        self.anchor_names = anchor_names
        self.name_to_idx = {n: i for i, n in enumerate(anchor_names)}
        self.n = len(anchor_names)
        # Co-occurrence counts
        self._pair_count = np.zeros((self.n, self.n), dtype=np.float64)
        self._anchor_count = np.zeros(self.n, dtype=np.float64)
        self._total_sentences = 0
        # Computed matrices
        self.npmi: np.ndarray | None = None
        self.laplacian: np.ndarray | None = None
        self.fiedler_value: float = 0.0

    def observe(self, activation_matrix: np.ndarray, threshold: float = 0.3):
        """Observe a batch of sentences' anchor activations.

        Args:
            activation_matrix: (n_sentences, n_anchors) from encoder.encode()
            threshold: minimum activation to count as "present"
        """
        binary = (activation_matrix > threshold).astype(np.float64)
        self._total_sentences += binary.shape[0]

        # Per-anchor counts
        self._anchor_count += binary.sum(axis=0)

        # Pair co-occurrence: binary^T @ binary gives co-occurrence matrix
        self._pair_count += binary.T @ binary

    def fit(self):
        """Compute NPMI matrix and Laplacian from observed co-occurrences."""
        N = max(self._total_sentences, 1)
        npmi = np.zeros((self.n, self.n), dtype=np.float64)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                p_ij = self._pair_count[i, j] / N
                p_i = self._anchor_count[i] / N
                p_j = self._anchor_count[j] / N

                if p_ij > 0 and p_i > 0 and p_j > 0:
                    pmi = math.log(p_ij / (p_i * p_j))
                    # Normalize: NPMI ∈ [-1, 1]
                    npmi_val = pmi / (-math.log(p_ij)) if p_ij < 1.0 else 0.0
                    npmi[i, j] = npmi_val
                    npmi[j, i] = npmi_val

        self.npmi = npmi

        # Build adjacency from positive NPMI values
        adj = np.maximum(npmi, 0)
        degree = adj.sum(axis=1)
        D = np.diag(degree)
        self.laplacian = D - adj

        # Fiedler value (second-smallest eigenvalue of Laplacian)
        # Measures algebraic connectivity of the co-activation graph
        eigenvalues = np.linalg.eigvalsh(self.laplacian)
        # Sort and take second smallest (first is always ~0)
        eigenvalues.sort()
        self.fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

        return self

    def score_infon(self, infon: Infon) -> float:
        """Compute sheaf coherence for a single infon.

        Measures how strongly the infon's (S, P, O) anchors are connected
        in the NPMI co-activation graph. Returns a score in [0, 1].

        High score = the triple's anchors genuinely co-occur across the corpus,
        not just in this one sentence.
        """
        if self.npmi is None:
            return 0.0

        anchors = [infon.subject, infon.predicate, infon.object]
        indices = [self.name_to_idx.get(a) for a in anchors if a in self.name_to_idx]

        if len(indices) < 2:
            return 0.0

        # Average pairwise NPMI among the triple's anchors
        npmi_sum = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                npmi_sum += self.npmi[indices[i], indices[j]]
                count += 1

        avg_npmi = npmi_sum / count if count > 0 else 0.0

        # Map from [-1, 1] to [0, 1]
        return float(max(0.0, min(1.0, (avg_npmi + 1.0) / 2.0)))

    def score_batch(self, infons: list[Infon]) -> list[float]:
        """Compute sheaf coherence for a batch of infons."""
        return [self.score_infon(inf) for inf in infons]

    def global_connectivity(self) -> float:
        """Global algebraic connectivity (Fiedler value).

        Higher = the anchor graph is well-connected (single global section).
        Zero = the graph has disconnected components (no global section exists).
        """
        return self.fiedler_value

    def anchor_centrality(self) -> dict[str, float]:
        """Per-anchor centrality in the co-activation graph.

        High centrality = this anchor connects many other anchors (a hub).
        """
        if self.npmi is None:
            return {}
        # Degree centrality on positive NPMI graph
        adj = np.maximum(self.npmi, 0)
        degree = adj.sum(axis=1)
        max_deg = degree.max() if degree.max() > 0 else 1.0
        return {
            name: float(degree[i] / max_deg)
            for i, name in enumerate(self.anchor_names)
        }

    def component_structure(self) -> list[set[str]]:
        """Find connected components in the positive NPMI graph.

        Each component is a "natural category" — anchors that co-occur.
        Disconnected components are independent categories.
        """
        if self.npmi is None:
            return []

        adj = np.maximum(self.npmi, 0) > 0.05  # threshold for edge existence
        visited = set()
        components = []

        for start in range(self.n):
            if start in visited:
                continue
            # BFS
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(self.anchor_names[node])
                for neighbor in range(self.n):
                    if adj[node, neighbor] and neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(component)

        return components


# ═══════════════════════════════════════════════════════════════════════
# 2. FUNCTORIAL DATA MIGRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SchemaFunctor:
    """A functor F: Schema_old → Schema_new.

    Defines how anchors map between schemas:
    - rename: old_name → new_name (1:1)
    - merge: {old_name1, old_name2} → new_name (many:1)
    - split: old_name → {new_name1, new_name2} (1:many, with predicate)
    - delete: old_name → ∅ (forget)
    - create: ∅ → new_name (new anchor, no migration)
    """
    rename: dict[str, str] = field(default_factory=dict)
    merge: dict[str, str] = field(default_factory=dict)  # old_name → merged_name
    delete: set[str] = field(default_factory=set)

    def map_anchor(self, name: str) -> str | None:
        """Apply the functor to a single anchor name.

        Returns the new name, or None if deleted.
        """
        if name in self.delete:
            return None
        if name in self.rename:
            return self.rename[name]
        if name in self.merge:
            return self.merge[name]
        return name  # identity — unchanged

    def map_triple(self, subj: str, pred: str, obj: str) -> tuple[str, str, str] | None:
        """Apply functor to a (S, P, O) triple.

        Returns mapped triple, or None if any component is deleted.
        """
        new_s = self.map_anchor(subj)
        new_p = self.map_anchor(pred)
        new_o = self.map_anchor(obj)

        if new_s is None or new_p is None or new_o is None:
            return None
        return (new_s, new_p, new_o)


class FunctorialMigration:
    """Pushforward along a schema functor.

    Given F: Schema_old → Schema_new, migrates:
    - Infons: remap (S, P, O), merge duplicates via reinforcement
    - Edges: remap source/target anchor references
    - Constraints: re-aggregate after infon migration
    """

    def __init__(self, functor: SchemaFunctor, source_schema: AnchorSchema,
                 target_schema: AnchorSchema):
        self.functor = functor
        self.source = source_schema
        self.target = target_schema

    def migrate_infon(self, infon: Infon) -> Infon | None:
        """Pushforward a single infon along the functor.

        Returns migrated infon or None if the triple is deleted.
        """
        mapped = self.functor.map_triple(infon.subject, infon.predicate, infon.object)
        if mapped is None:
            return None

        new_s, new_p, new_o = mapped

        # Update hierarchy metadata from target schema
        new_subject_meta = self.target.get_hierarchy(new_s)
        new_subject_meta["type"] = self.target.types.get(new_s, "")
        new_predicate_meta = self.target.get_hierarchy(new_p)
        new_predicate_meta["type"] = self.target.types.get(new_p, "")
        new_object_meta = self.target.get_hierarchy(new_o)
        new_object_meta["type"] = self.target.types.get(new_o, "")

        migrated = Infon(
            infon_id=infon.infon_id,
            subject=new_s,
            predicate=new_p,
            object=new_o,
            polarity=infon.polarity,
            direction=infon.direction,
            confidence=infon.confidence,
            sentence=infon.sentence,
            doc_id=infon.doc_id,
            sent_id=infon.sent_id,
            spans=infon.spans,
            support=infon.support,
            subject_meta=new_subject_meta,
            predicate_meta=new_predicate_meta,
            object_meta=new_object_meta,
            locations=infon.locations,
            timestamp=infon.timestamp,
            precision=infon.precision,
            temporal_refs=infon.temporal_refs,
            tense=infon.tense,
            aspect=infon.aspect,
            activation=infon.activation,
            coherence=infon.coherence,
            specificity=infon.specificity,
            novelty=infon.novelty,
            importance=infon.importance,
            reinforcement_count=infon.reinforcement_count,
            last_reinforced=infon.last_reinforced,
            decay_rate=infon.decay_rate,
        )
        return migrated

    def migrate_edge(self, edge: Edge) -> Edge | None:
        """Pushforward a single edge along the functor."""
        new_source = self.functor.map_anchor(edge.source)
        new_target = self.functor.map_anchor(edge.target)

        # For edges, source/target might be infon IDs (not anchor names)
        # Only remap if the name is in the functor's domain
        if new_source is None or new_target is None:
            return None

        return Edge(
            source=new_source if new_source != edge.source else edge.source,
            target=new_target if new_target != edge.target else edge.target,
            edge_type=edge.edge_type,
            weight=edge.weight,
            metadata=edge.metadata,
        )

    def migrate_all(self, infons: list[Infon], edges: list[Edge]
                    ) -> tuple[list[Infon], list[Edge]]:
        """Migrate all infons and edges. Merges duplicate triples."""
        migrated_infons = []
        seen_triples: dict[tuple, Infon] = {}

        for inf in infons:
            new_inf = self.migrate_infon(inf)
            if new_inf is None:
                continue

            key = new_inf.triple_key()
            if key in seen_triples:
                # Merge: average confidence, sum reinforcement
                existing = seen_triples[key]
                existing.reinforcement_count += 1
                n = existing.reinforcement_count + 1
                existing.confidence = (
                    existing.confidence * (n - 1) + new_inf.confidence
                ) / n
            else:
                seen_triples[key] = new_inf

        migrated_infons = list(seen_triples.values())

        # Migrate edges
        migrated_edges = []
        for edge in edges:
            new_edge = self.migrate_edge(edge)
            if new_edge is not None:
                migrated_edges.append(new_edge)

        return migrated_infons, migrated_edges

    def report(self, original_infons: list[Infon], migrated_infons: list[Infon]) -> dict:
        """Report migration statistics."""
        orig_triples = set(inf.triple_key() for inf in original_infons)
        new_triples = set(inf.triple_key() for inf in migrated_infons)

        deleted = len(original_infons) - len(migrated_infons)
        merged = len(original_infons) - deleted - len(migrated_infons)

        return {
            "original_infons": len(original_infons),
            "migrated_infons": len(migrated_infons),
            "original_triples": len(orig_triples),
            "migrated_triples": len(new_triples),
            "deleted": deleted,
            "merged_duplicates": sum(
                inf.reinforcement_count for inf in migrated_infons
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# 3. LEFT KAN EXTENSION — SCHEMA-FREE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveredAnchor:
    """An anchor discovered from data via spectral clustering."""
    name: str
    inferred_type: str
    tokens: list[str]
    centroid_indices: list[int]  # BERT vocab IDs at cluster center
    size: int  # number of vocab tokens in cluster
    mean_activation: float
    coherence: float  # internal cluster coherence


class SchemaDiscovery:
    """Discover categorical structure from raw SPLADE activations.

    The left Kan extension along the inclusion J: Observed → Full_vocab
    constructs the "best" schema that explains co-activation patterns.

    Algorithm:
    1. Encode corpus through SPLADE (no schema needed)
    2. Build vocab × vocab co-activation matrix
    3. Threshold to significant entries (document frequency filter)
    4. Spectral clustering on the co-activation graph
    5. Label clusters by top tokens → discovered anchors
    6. Infer types from token semantics (nouns=actor/feature, verbs=relation, etc.)
    """

    def __init__(self, encoder: SpladeEncoder | None = None):
        self.encoder = encoder or SpladeEncoder()
        self.tokenizer = self.encoder.tokenizer
        self.vocab_size = self.encoder.vocab_size

    def discover(self, texts: list[str],
                 n_anchors: int = 25,
                 min_doc_freq: int = 2,
                 activation_threshold: float = 0.3,
                 batch_size: int = 32) -> tuple[AnchorSchema, list[DiscoveredAnchor]]:
        """Discover a schema from raw text.

        Args:
            texts: list of sentences/documents
            n_anchors: target number of anchors to discover
            min_doc_freq: minimum document frequency for a vocab token to be considered
            activation_threshold: SPLADE activation threshold
            batch_size: encoding batch size

        Returns:
            (AnchorSchema, list[DiscoveredAnchor]) — the discovered schema and metadata
        """
        # 1. Encode all texts
        sparse_matrix = self.encoder.encode_sparse(texts, batch_size=batch_size)
        n_texts, vocab_size = sparse_matrix.shape

        # 2. Find frequently-activated tokens
        binary = (sparse_matrix > activation_threshold).astype(np.float32)
        doc_freq = binary.sum(axis=0)  # (vocab_size,)

        # Filter to tokens appearing in at least min_doc_freq documents
        active_mask = doc_freq >= min_doc_freq
        active_indices = np.where(active_mask)[0]

        if len(active_indices) < n_anchors:
            # Not enough data; lower threshold
            active_indices = np.argsort(doc_freq)[-max(n_anchors * 3, 50):]

        # 3. Build co-activation matrix for active tokens
        active_binary = binary[:, active_indices]  # (n_texts, n_active)
        coact = active_binary.T @ active_binary  # (n_active, n_active)

        # Normalize to NPMI-like measure
        df_active = doc_freq[active_indices]
        outer = np.outer(df_active, df_active)
        outer = np.where(outer > 0, outer, 1.0)
        coact_norm = coact * n_texts / outer

        # 4. Spectral clustering
        # Use the co-activation matrix as affinity
        affinity = np.maximum(coact_norm - 1.0, 0)  # PMI > 0 → positive association
        np.fill_diagonal(affinity, 0)

        # Compute Laplacian
        degree = affinity.sum(axis=1)
        degree_safe = np.where(degree > 0, degree, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_safe))
        L_norm = np.eye(len(active_indices)) - D_inv_sqrt @ affinity @ D_inv_sqrt

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Take first n_anchors eigenvectors (after the trivial one)
        k = min(n_anchors, len(active_indices) - 1, eigenvectors.shape[1] - 1)
        features = eigenvectors[:, 1:k+1]  # skip first (constant) eigenvector

        # K-means on spectral features
        labels = self._kmeans(features, k)

        # 5. Build discovered anchors
        discovered = []
        for cluster_id in range(k):
            member_mask = labels == cluster_id
            member_indices = active_indices[member_mask]

            if len(member_indices) == 0:
                continue

            # Top tokens by mean activation in the cluster
            mean_acts = sparse_matrix[:, member_indices].mean(axis=0)
            if hasattr(mean_acts, 'A1'):
                mean_acts = np.array(mean_acts).flatten()
            else:
                mean_acts = np.array(mean_acts).flatten()

            top_in_cluster = np.argsort(mean_acts)[-5:][::-1]
            top_vocab_ids = member_indices[top_in_cluster]

            tokens = []
            for vid in top_vocab_ids:
                tok = self.tokenizer.convert_ids_to_tokens(int(vid))
                if tok and not tok.startswith('[') and not tok.startswith('##'):
                    tokens.append(tok)

            if not tokens:
                # Fall back to all member tokens
                for vid in member_indices[:5]:
                    tok = self.tokenizer.convert_ids_to_tokens(int(vid))
                    if tok:
                        tokens.append(tok.lstrip('#'))

            if not tokens:
                continue

            # Name = most activated token
            name = tokens[0].lower().replace(' ', '_').replace('-', '_')

            # Infer type from token characteristics
            inferred_type = self._infer_type(tokens, name)

            # Internal coherence: mean pairwise co-activation within cluster
            if len(member_indices) > 1:
                cluster_coact = coact_norm[np.ix_(member_mask, member_mask)]
                np.fill_diagonal(cluster_coact, 0)
                coherence = float(cluster_coact.mean())
            else:
                coherence = 0.0

            discovered.append(DiscoveredAnchor(
                name=name,
                inferred_type=inferred_type,
                tokens=tokens[:5],
                centroid_indices=top_vocab_ids.tolist(),
                size=len(member_indices),
                mean_activation=float(mean_acts.mean()),
                coherence=coherence,
            ))

        # 6. Build AnchorSchema
        anchor_defs = {}
        # Deduplicate names
        seen_names = set()
        for da in discovered:
            name = da.name
            if name in seen_names:
                name = f"{name}_{da.inferred_type}"
            seen_names.add(name)
            da.name = name

            anchor_defs[name] = {
                "type": da.inferred_type,
                "tokens": da.tokens,
            }

        schema = AnchorSchema(anchor_defs)
        return schema, discovered

    def _kmeans(self, features: np.ndarray, k: int,
                max_iter: int = 50) -> np.ndarray:
        """Simple k-means clustering."""
        n = features.shape[0]
        if n == 0 or k == 0:
            return np.zeros(0, dtype=np.int32)

        # Initialize centroids with k-means++
        rng = np.random.RandomState(42)
        centroids = [features[rng.randint(n)]]
        for _ in range(k - 1):
            dists = np.array([
                min(np.sum((f - c) ** 2) for c in centroids)
                for f in features
            ])
            dists_safe = dists / (dists.sum() + 1e-10)
            next_idx = rng.choice(n, p=dists_safe)
            centroids.append(features[next_idx])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            # Assign
            dists = np.array([
                np.sum((features - centroids[j]) ** 2, axis=1)
                for j in range(k)
            ]).T  # (n, k)
            new_labels = dists.argmin(axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Update
            for j in range(k):
                members = features[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)

        return labels

    def _infer_type(self, tokens: list[str], name: str) -> str:
        """Infer anchor type from token semantics.

        Heuristic: verbs → relation, proper nouns/country codes → actor,
        geographic terms → market, everything else → feature.
        """
        # Common relation indicators (verbs and action nouns)
        relation_tokens = {
            "invest", "launch", "partner", "expand", "decline", "deploy",
            "sanction", "negotiate", "trade", "approve", "develop", "attack",
            "support", "oppose", "send", "build", "grow", "sign", "announce",
            "reject", "condemn", "urge", "demand", "agree", "cooperate",
            "conflict", "compete", "dominate", "enter", "exit", "delay",
            "ordered", "entered", "organized", "triggered", "rejected",
            "punished", "visited", "simulated", "developed", "sent",
            "called", "talk", "award", "condemned", "urged",
            "signed", "deployed", "imposed", "invaded", "seized",
            "resumed", "suspended", "withdrew", "allied", "opposed",
            "cooperation", "diplomatic", "military", "sanctions",
            "agreement", "treaty", "accord", "pact", "alliance",
        }

        # Actor indicators (entities that do things)
        actor_tokens = {
            "government", "president", "minister", "army", "navy",
            "forces", "troops", "authority", "council", "commission",
            "organization", "agency", "party", "group", "union",
            "nato", "un", "eu", "who", "imf",
        }

        # Market/location indicators
        market_tokens = {
            "china", "us", "europe", "asia", "pacific", "africa", "america",
            "india", "japan", "korea", "taiwan", "russia", "iran", "turkey",
            "beijing", "washington", "tokyo", "middle", "east", "west",
            "region", "global", "domestic", "international",
            "nations", "countries", "states",
        }

        # Check tokens
        tokens_lower = {t.lower() for t in tokens}
        name_lower = name.lower()

        # Score each type
        relation_overlap = len(tokens_lower & relation_tokens)
        actor_overlap = len(tokens_lower & actor_tokens)
        market_overlap = len(tokens_lower & market_tokens)

        if relation_overlap > max(actor_overlap, market_overlap):
            return "relation"
        if market_overlap > max(actor_overlap, relation_overlap):
            return "market"
        if actor_overlap > 0:
            return "actor"

        # Two-letter tokens are often country codes
        if any(len(t) == 2 and t.isupper() for t in tokens):
            return "actor"

        # Check for verb-like endings
        for t in tokens_lower:
            if t.endswith("ed") or t.endswith("ing") or t.endswith("tion"):
                return "relation"

        # Default: feature
        return "feature"
