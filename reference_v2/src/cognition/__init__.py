"""Cognition: document → infon → temporal knowledge graph.

One-stop library for building and querying situation-semantic knowledge
from raw documents. Uses SPLADE (Apache 2.0) for broad vocabulary coverage
with typed anchor projection — no model training required, just define
your schema and go. Works locally (SQLite + threads) or on AWS
(DynamoDB + S3 + Lambda containers).

Quick start:
    from cognition import Cognition, CognitionConfig

    cog = Cognition(CognitionConfig(schema_path="data/schema.json"))
    cog.ingest([{"text": "Toyota invests in solid-state batteries.", "id": "doc1"}])
    result = cog.query("What is Toyota investing in?")
    for inf in result.infons:
        print(inf)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .infon import Infon, Edge, Constraint, Span, QueryResult
from .schema import AnchorSchema
from .config import CognitionConfig
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
from .heads import CognitionHeads, NLIHead, RelevanceHead, PolarityHead, RelationTypeHead
from .graph_mcts import GraphMCTS, MCTSResult, format_mcts_result
from .logic import (
    HypergraphReasoner, HypergraphBuilder, HyperGraph,
    ReasoningResult, RefinementResult,
    TypedMessagePassingLayer, MassReadout,
    IKLThat, IKLAnd, IKLOr, IKLNot, IKLIf, IKLIff,
    IKLForall, IKLExists, IKLIst,
)


class Cognition:
    """Main entry point: ingest documents, query knowledge.

    Schema-only (no training needed):
        cog = Cognition(CognitionConfig(schema_path="data/schema.json"))

    From saved config directory:
        cog = Cognition(CognitionConfig.local("models/my-domain"))

    AWS:
        cog = Cognition(CognitionConfig.aws(
            model_dir="models/my-domain",
            table="cognition-prod",
            bucket="cognition-prod-data",
        ))
    """

    def __init__(self, config: CognitionConfig):
        self.config = config

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

        self._infon_count_since_consolidation = 0

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
