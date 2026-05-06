"""Infon: document → infon → temporal knowledge graph.

One-stop library for building and querying situation-semantic knowledge
from raw documents. Uses SPLADE (Apache 2.0) for broad vocabulary coverage
with typed anchor projection — no model training required, just define
your schema and go. Works locally (SQLite + threads) or on AWS
(DynamoDB + S3 + Lambda containers).

Quick start:
    from infon import InfonEngine, InfonConfig

    cog = InfonEngine(InfonConfig(schema_path="data/schema.json"))
    cog.ingest([{"text": "Toyota invests in solid-state batteries.", "id": "doc1"}])
    result = cog.query("What is Toyota investing in?")
    for inf in result.infons:
        print(inf)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .atom import Infon, Edge, Constraint, Span, QueryResult
from .schema import AnchorSchema
from .config import InfonConfig
from .encoder import Encoder
from .extract import extract_infons, split_sentences
from .consolidate import (
    consolidate, aggregate_constraints, build_next_edges,
    reinforce, apply_decay,
)
from .query import query as _query_fn, detect_persona, compute_valence
from .category import (
    SheafCoherence, SchemaFunctor, FunctorialMigration, SchemaDiscovery,
)
from .structural import (
    StructuralAnalyzer, DriverTree, DriverNode,
    KanoClassifier, KanoResult,
    ConjointEstimator, ConjointResult,
    FeatureGapFunctor, FeatureGapResult,
    GhostDetector, GhostResult,
    PolarizationIndex, PolarizationResult,
    NarrativeAnalyzer, NarrativeResult,
    ContagionAnalyzer, ContagionResult,
    KanExtension, KanExtensionResult,
)
from .dempster_shafer import (
    MassFunction, combine_dempster, combine_multiple,
    mass_from_polarity, mass_from_triple_alignment,
    mass_from_anchor_distance, mass_from_confidence,
    verify_claim, VerificationVerdict,
)
from .heads import InfonHeads, NLIHead, RelevanceHead, PolarityHead, RelationTypeHead
from .graph_mcts import GraphMCTS, MCTSResult, format_mcts_result
from .logic import (
    HypergraphReasoner, HypergraphBuilder, HyperGraph,
    ReasoningResult, RefinementResult, CausalView,
    TypedMessagePassingLayer, MassReadout, NextAnchorHead,
    SubgraphPool, TimeToEventHead, RiskRankingHead, AnomalyLocalizationHead,
    RecommenderHead, DiversityHead, TemporalSuccessorHead,
    IKLThat, IKLAnd, IKLOr, IKLNot, IKLIf, IKLIff,
    IKLForall, IKLExists, IKLIst,
)


from infon.cassette.store import InfonStore
from infon.cassette.dsl import Query
from infon.cassette.analyst import Analyst


class InfonEngine:
    """Main entry point: ingest documents, query knowledge.

    Schema-only (no training needed):
        cog = InfonEngine(InfonConfig(schema_path="data/schema.json"))

    From saved config directory:
        cog = InfonEngine(InfonConfig.local("models/my-domain"))

    AWS:
        cog = InfonEngine(InfonConfig.aws(
            model_dir="models/my-domain",
            table="infon-prod",
            bucket="infon-prod-data",
        ))
    """

    def __init__(self, config: InfonConfig):
        self.config = config

        # 0. Pin global RNGs if the caller asked for determinism.
        #    See infon.random_state for the derived sub-seeds used
        #    by individual components (SSL, embedder training, etc.).
        if config.random_state is not None:
            from .random_state import set_global_seed
            set_global_seed(config.random_state)
        self.random_state = config.random_state

        # 1. Load schema first — needed for encoder projection
        if config.schema_path:
            self.schema = AnchorSchema.from_file(config.schema_path)
        elif config.model_dir:
            import json
            from pathlib import Path
            cfg_path = Path(config.model_dir) / "config.json"
            schema_path = Path(config.model_dir) / "schema.json"
            if schema_path.exists():
                self.schema = AnchorSchema.from_file(schema_path)
            elif cfg_path.exists():
                with open(cfg_path) as f:
                    model_config = json.load(f)
                anchor_defs = model_config.get("anchor_defs", {})
                if anchor_defs:
                    self.schema = AnchorSchema(anchor_defs)
                else:
                    names = model_config.get("anchor_names", [])
                    types = model_config.get("anchor_types", {})
                    self.schema = AnchorSchema({
                        name: {"type": types.get(name, "feature"), "tokens": [name]}
                        for name in names
                    })
            else:
                self.schema = AnchorSchema({})
        else:
            self.schema = AnchorSchema({})

        # 2. Load encoder (SPLADE + AnchorProjector)
        if config.model_dir:
            self.encoder = Encoder.from_dir(config.model_dir, device=config.device)
        else:
            # Empty model_name = use bundled splade-tiny
            kwargs = dict(
                schema=self.schema,
                max_length=config.max_length,
                device=config.device,
            )
            if config.model_name:
                kwargs["model_name"] = config.model_name
            self.encoder = Encoder(**kwargs)

        # Initialize store
        if config.backend == "aws":
            from .store.cloud import CloudStore
            self.store = CloudStore(
                table_name=config.aws_table,
                bucket=config.aws_bucket,
                region=config.aws_region,
            )
        else:
            from .store.local import LocalStore
            self.store = LocalStore(db_path=config.db_path)

        self.store.init()

        # Initialize compute
        if config.backend == "aws":
            from .compute.cloud import CloudCompute
            self.compute = CloudCompute(
                function_name=config.aws_lambda_function,
                bucket=config.aws_bucket,
                region=config.aws_region,
            )
        else:
            from .compute.local import LocalCompute
            self.compute = LocalCompute()

        # Optional trained SentenceEmbedder, lazily trained per-schema.
        self.embedder = None
        if getattr(config, "use_trained_embedder", False):
            from .embedder import get_or_train_embedder
            import os
            model_dir = config.embedder_model_dir
            if model_dir is None:
                # Sit next to the DB by default
                base = os.path.dirname(os.path.abspath(config.db_path or "."))
                model_dir = os.path.join(base, "embedder_cache")
            self.embedder = get_or_train_embedder(
                self.schema,
                self.encoder.splade,
                model_dir=model_dir,
                n_synth_examples=config.embedder_n_synth,
                trunk_dim=config.embedder_trunk_dim,
                epochs=config.embedder_epochs,
                verbose=False,
            )

        self._infon_count_since_consolidation = 0

        # Staleness tracking: every ingest bumps the generation counter.
        # A cached reasoner (see reasoner()) records the generation it
        # was last refitted against; refresh() triggers a rebuild
        # whenever the two diverge.
        self._generation = 0
        self._reasoner_cache = None   # (reasoner, generation_at_fit)

    def ingest(self, documents: list[dict], consolidate_now: bool = False) -> int:
        """Ingest documents → extract infons → store.

        Each document: {"text": str, "id": str, "timestamp": str (optional)}

        Args:
            documents: list of document dicts
            consolidate_now: force consolidation after ingestion

        Returns:
            Number of infons extracted.
        """
        infons, edges = extract_infons(
            documents, self.encoder, self.schema, self.config,
        )

        if not infons:
            return 0

        # Store infons and edges
        self.store.put_infons(infons)
        self.store.put_edges(edges)

        self._infon_count_since_consolidation += len(infons)
        # Mark the in-memory cache stale so refresh() will rebuild
        self._generation += 1

        # Auto-consolidate
        if consolidate_now or self._infon_count_since_consolidation >= self.config.consolidation_interval:
            self.consolidate()

        return len(infons)

    def consolidate(self) -> None:
        """Run consolidation: reinforce duplicates, build constraints and NEXT edges."""
        all_infons = self.store.query_infons(limit=50000)
        if not all_infons:
            return

        constraints = aggregate_constraints(all_infons)
        for c in constraints:
            self.store.put_constraint(c)

        next_edges = build_next_edges(all_infons)
        self.store.put_edges(next_edges)

        self._infon_count_since_consolidation = 0

    def query(self, text: str, persona: str | None = None,
              goal: str = "", top_k: int | None = None,
              min_importance: float = 0.0,
              include_chains: bool = True,
              contrary: bool = False) -> QueryResult:
        """Query the knowledge graph.

        Args:
            text: natural language query
            persona: override auto-detected persona
            goal: optional goal for valence tuning
            top_k: max results
            min_importance: filter threshold
            include_chains: walk NEXT edges for temporal prediction
            contrary: invert ranking to surface counter-evidence

        Returns:
            QueryResult with infons, constraints, edges, valence, timeline
        """
        return _query_fn(
            text=text,
            encoder=self.encoder,
            schema=self.schema,
            store=self.store,
            config=self.config,
            persona=persona,
            goal=goal,
            top_k=top_k,
            min_importance=min_importance,
            include_chains=include_chains,
            contrary=contrary,
        )

    def analyze(self, spec_anchors: set[str] | None = None,
                enrich: bool = True) -> dict:
        """Run all structural analysis engines and optionally enrich infon metrics.

        Returns dict of engine_name → results (kano, conjoint, feature_gap,
        ghosts, polarization, narrative, contagion). If enrich=True, also
        populates infon.metrics dicts in the store.
        """
        sa = StructuralAnalyzer(self.schema)
        all_infons = self.store.query_infons(limit=50000)
        edges = self.store.get_edges(edge_type="NEXT", limit=50000)
        results = sa.run_all(all_infons, edges, spec_anchors=spec_anchors)

        if enrich:
            sa.enrich(all_infons, results, edges)
            self.store.put_infons(all_infons)

        return results

    def driver_tree(self, results: dict | None = None) -> DriverNode:
        """Build the structural driver tree. Pass results from analyze() or compute fresh."""
        sa = StructuralAnalyzer(self.schema)
        all_infons = self.store.query_infons(limit=50000)
        if results is None:
            edges = self.store.get_edges(edge_type="NEXT", limit=50000)
            results = sa.run_all(all_infons, edges)
        return sa.driver_tree(all_infons, results)

    def decay(self, reference_date: str | None = None) -> None:
        """Apply importance decay to all infons."""
        all_infons = self.store.query_infons(limit=50000)
        decayed = apply_decay(all_infons, reference_date)
        self.store.put_infons(decayed)

    def prune(self, threshold: float | None = None) -> int:
        """Soft-delete infons below importance threshold."""
        t = threshold if threshold is not None else self.config.prune_threshold
        return self.store.prune(t)

    def reasoner(self, hidden_dim: int = 64, n_layers: int = 2,
                 force_rebuild: bool = False):
        """Return a cached HypergraphReasoner, refitting if stale.

        The first call builds and fits the reasoner against the current
        store state. Subsequent calls return the same instance unless
        ingest() has bumped the generation counter, in which case the
        reasoner is refitted against the new graph before being
        returned. Pass force_rebuild=True to force a rebuild regardless.
        """
        from .logic import HypergraphReasoner

        cache = self._reasoner_cache
        if (not force_rebuild and cache is not None
                and cache[1] == self._generation):
            return cache[0]

        reasoner = HypergraphReasoner(
            self.store, self.encoder, self.schema,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )
        if self.embedder is not None:
            reasoner.builder.embedder = self.embedder
        # Transductively fit against the current graph
        graph = reasoner.builder.build(feature_dim=hidden_dim)
        reasoner.fit(graph=graph, epochs=20, verbose=False)

        self._reasoner_cache = (reasoner, self._generation)
        return reasoner

    def refresh(self, retrain_heads: bool = False,
                hidden_dim: int = 64, n_layers: int = 2,
                verbose: bool = False) -> dict:
        """Rebuild the cached reasoner against the current store state.

        Call this after ingest() adds new documents if you want
        subsequent queries to reflect the new evidence. Without a
        refresh, cached reasoners use the graph they were last
        trained on — that's intentional so inference is fast by
        default, but it can silently become stale.

        Args:
            retrain_heads: if True, retrain all task heads (next-anchor,
                time-to-event, risk, anomaly, recommender) against the
                new graph. Otherwise only the core GNN+MassReadout
                refits. Defaults to False to keep refresh cheap.

        Returns:
            summary dict with keys: rebuilt (bool), generation (int),
            elapsed_s (float), n_infons (int), retrained_heads (list).
        """
        import time as _time
        t0 = _time.perf_counter()

        prev_generation = (self._reasoner_cache[1]
                            if self._reasoner_cache else -1)
        was_stale = prev_generation != self._generation

        # Force rebuild against current generation
        reasoner = self.reasoner(
            hidden_dim=hidden_dim, n_layers=n_layers,
            force_rebuild=True,
        )

        retrained: list[str] = []
        if retrain_heads:
            # Refit each task head that's available
            graph = reasoner.builder.build(feature_dim=hidden_dim)
            if hasattr(reasoner, "train_next_head") and \
                    getattr(reasoner, "_next_head_trained", False):
                try:
                    reasoner.train_next_head(
                        graph=graph, epochs=20, verbose=False,
                    )
                    retrained.append("next_anchor")
                except Exception:
                    pass
            if getattr(reasoner, "_risk_head_trained", False):
                try:
                    reasoner.train_risk_head(
                        graph=graph, epochs=20, verbose=False,
                    )
                    retrained.append("risk")
                except Exception:
                    pass
            if getattr(reasoner, "_anomaly_head_trained", False):
                try:
                    reasoner.train_anomaly_head(
                        graph=graph, epochs=40, verbose=False,
                    )
                    retrained.append("anomaly")
                except Exception:
                    pass

        elapsed = _time.perf_counter() - t0
        summary = {
            "rebuilt": was_stale,
            "generation": self._generation,
            "elapsed_s": elapsed,
            "n_infons": self.store.count_infons(),
            "retrained_heads": retrained,
        }
        if verbose:
            print(f"  refresh: generation {prev_generation} → "
                  f"{self._generation}, {elapsed:.2f}s, "
                  f"{summary['n_infons']} infons, "
                  f"retrained={retrained}")
        return summary

    def expand(self, query: str,
               max_docs: int = 5,
               theta_threshold: float = 0.4,
               source: str = "ddgs",
               search_fn=None,
               verbose: bool = False) -> dict:
        """Active exploration: if the system is uncertain about a
        query, fetch external documents and re-query.

        Flow:
          1. Run the query against the current corpus.
          2. If the returned θ mass < theta_threshold, return as-is.
          3. Otherwise fetch max_docs results from the configured
             search backend, ingest them, refresh(), and re-query.
          4. Return a dict with both masses side by side and the
             external snippets that were ingested.

        Args:
            query: natural-language claim to test
            max_docs: how many external results to fetch
            theta_threshold: θ above which we trigger exploration
            source: one of 'ddgs' (real web via ddgs library),
                    'mock' (for tests — uses search_fn callback)
            search_fn: required when source='mock'; takes the query
                      and returns a list of {'title','body','href'}

        Returns:
            {
              'query': str,
              'expanded': bool,            # did we fetch anything?
              'before': {'verdict', 'mass', 'theta'},
              'after':  {'verdict', 'mass', 'theta'} or None,
              'n_new_docs': int,
              'new_doc_ids': list[str],
              'sources': list[str],         # URLs of fetched results
            }
        """
        reasoner = self.reasoner()
        before = reasoner.reason(query)
        before_d = {
            "verdict": before.verdict,
            "supports": before.mass.supports,
            "refutes": before.mass.refutes,
            "uncertain": before.mass.uncertain,
            "theta": before.mass.theta,
        }

        if before.mass.theta < theta_threshold:
            if verbose:
                print(f"  [expand] θ={before.mass.theta:.3f} below "
                      f"{theta_threshold}; no expansion needed")
            return {
                "query": query,
                "expanded": False,
                "before": before_d,
                "after": None,
                "n_new_docs": 0,
                "new_doc_ids": [],
                "sources": [],
            }

        # Fetch external documents
        if verbose:
            print(f"  [expand] θ={before.mass.theta:.3f} above "
                  f"threshold; fetching from source={source!r}")
        snippets = []
        if source == "ddgs":
            try:
                from ddgs import DDGS
            except ImportError:
                if verbose:
                    print("  [expand] ddgs not installed; aborting")
                return {
                    "query": query,
                    "expanded": False,
                    "before": before_d,
                    "after": None,
                    "n_new_docs": 0,
                    "new_doc_ids": [],
                    "sources": [],
                }
            try:
                ddgs_client = DDGS()
                for r in ddgs_client.text(query, max_results=max_docs):
                    snippets.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    })
            except Exception as e:
                if verbose:
                    print(f"  [expand] ddgs error: {e}")
        elif source == "mock":
            if search_fn is None:
                raise ValueError("source='mock' requires search_fn callback")
            snippets = list(search_fn(query))[:max_docs]
        else:
            raise ValueError(f"unknown source {source!r}")

        if not snippets:
            if verbose:
                print("  [expand] no snippets fetched; returning before-mass")
            return {
                "query": query,
                "expanded": False,
                "before": before_d,
                "after": None,
                "n_new_docs": 0,
                "new_doc_ids": [],
                "sources": [],
            }

        # Ingest the snippets as new documents
        import hashlib
        new_docs = []
        for s in snippets:
            text = (s.get("title", "") + ". " + s.get("body", "")).strip()
            if not text or text == ".":
                continue
            doc_id = "expand_" + hashlib.md5(
                s.get("href", text).encode("utf-8")
            ).hexdigest()[:10]
            new_docs.append({"id": doc_id, "text": text})

        n_before_ingest = self.store.count_infons()
        for d in new_docs:
            self.ingest([d])
        self.consolidate()
        n_after_ingest = self.store.count_infons()
        new_infons_count = n_after_ingest - n_before_ingest

        # Refresh and re-query
        self.refresh(verbose=False)
        after = self.reasoner().reason(query)
        after_d = {
            "verdict": after.verdict,
            "supports": after.mass.supports,
            "refutes": after.mass.refutes,
            "uncertain": after.mass.uncertain,
            "theta": after.mass.theta,
        }

        result = {
            "query": query,
            "expanded": True,
            "before": before_d,
            "after": after_d,
            "n_new_docs": len(new_docs),
            "n_new_infons": new_infons_count,
            "new_doc_ids": [d["id"] for d in new_docs],
            "sources": [s.get("href", "") for s in snippets],
        }

        if verbose:
            print(f"  [expand] ingested {len(new_docs)} docs "
                  f"({new_infons_count} new infons)")
            print(f"  before: verdict={before_d['verdict']} "
                  f"S={before_d['supports']:.3f} "
                  f"θ={before_d['theta']:.3f}")
            print(f"  after:  verdict={after_d['verdict']} "
                  f"S={after_d['supports']:.3f} "
                  f"θ={after_d['theta']:.3f}")

        return result

    def stats(self) -> dict:
        """Basic statistics about the knowledge graph."""
        n_infons = self.store.count_infons()
        constraints = self.store.get_constraints(limit=1000)
        edges = self.store.get_edges(edge_type="NEXT", limit=1)
        return {
            "infon_count": n_infons,
            "constraint_count": len(constraints),
            "has_sequences": len(edges) > 0,
            "backend": self.config.backend,
            "model": self.config.model_name or "splade-tiny (bundled)",
            "anchors": len(self.schema.names),
        }

    def close(self) -> None:
        """Release resources."""
        if hasattr(self.store, "close"):
            self.store.close()
        if hasattr(self.compute, "shutdown"):
            self.compute.shutdown()
