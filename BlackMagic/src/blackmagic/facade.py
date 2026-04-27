"""BlackMagic facade — the main entry point.

    bm = BlackMagic(BlackMagicConfig(schema_path="schemas/automotive.json"))
    bm.ingest([{"text": "...", "id": "d1", "timestamp": "2026-04-20"}])
    result = bm.search("query")
    imagined = bm.imagine("what-if query")
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import BlackMagicConfig
from .schema import AnchorSchema
from .infon import (
    Infon, Edge, Constraint, QueryResult, SearchResult,
    ImaginationResult,
)
from .encoder import Encoder
from .extract import extract_infons
from .consolidate import aggregate_constraints, build_next_edges, apply_decay
from .store import LocalStore
from .retrieve import query as _query_fn
from .dempster_shafer import verify_claim, VerificationVerdict
from .graph_mcts import GraphMCTS, MCTSResult


class BlackMagic:
    """Main entry point: ingest docs, search, verify, reason, imagine."""

    def __init__(self, config: BlackMagicConfig):
        self.config = config

        # Schema
        if config.schema_path:
            self.schema = AnchorSchema.from_file(config.schema_path)
        else:
            self.schema = AnchorSchema({})

        # Encoder (splade-tiny bundled)
        self.encoder = Encoder(
            schema=self.schema,
            model_name=config.model_name,
            max_length=config.max_length,
            device=config.device,
        )

        # Store (SQLite only)
        self.store = LocalStore(db_path=config.db_path)
        self.store.init()

        self._infon_count_since_consolidation = 0

    # ── Ingest ──────────────────────────────────────────────────────────

    def ingest(self, documents: list[dict],
               consolidate_now: bool = False) -> int:
        """Ingest documents → extract infons → store.

        Each document: {"text": str, "id": str, "timestamp": str (optional)}
        Returns number of infons extracted.
        """
        infons, edges = extract_infons(
            documents, self.encoder, self.schema, self.config,
        )

        # Store document text too, for snippet retrieval
        for doc in documents:
            doc_id = doc.get("id", doc.get("doc_id", ""))
            text = doc.get("text", "")
            timestamp = doc.get("timestamp", doc.get("ts"))
            if doc_id and text:
                self.store.put_document(doc_id, text, timestamp=timestamp)

        if not infons:
            return 0

        self.store.put_infons(infons)
        self.store.put_edges(edges)
        self._infon_count_since_consolidation += len(infons)

        auto = (self.config.consolidation_interval > 0
                and self._infon_count_since_consolidation
                >= self.config.consolidation_interval)
        if consolidate_now or auto:
            self.consolidate()

        return len(infons)

    def consolidate(self) -> None:
        """Rebuild constraints and NEXT edges from current observed infons."""
        all_infons = self.store.query_infons(limit=50000)
        if not all_infons:
            return

        constraints = aggregate_constraints(all_infons)
        for c in constraints:
            self.store.put_constraint(c)

        next_edges = build_next_edges(all_infons)
        self.store.put_edges(next_edges)

        self._infon_count_since_consolidation = 0

    # ── Search (retrieval) ──────────────────────────────────────────────

    def search(self, text: str, *,
               persona: str | None = None,
               goal: str = "",
               top_k: int | None = None,
               min_importance: float = 0.0,
               include_chains: bool = True,
               contrary: bool = False) -> SearchResult:
        """The primary API. Sparse retrieval with persona valence."""
        result = _query_fn(
            text=text,
            encoder=self.encoder,
            schema=self.schema,
            store=self.store,
            config=self.config,
            persona=persona or self.config.persona,
            goal=goal,
            top_k=top_k,
            min_importance=min_importance,
            include_chains=include_chains,
            contrary=contrary,
        )
        return result

    # Alias for backwards-compat with cognition callers
    query = search

    # ── Claim verification (Dempster-Shafer) ────────────────────────────

    def verify_claim(self, claim: str, top_k: int = 50) -> VerificationVerdict:
        """Fuse evidence for/against a claim using Dempster-Shafer."""
        claim_anchors = self.encoder.encode_single(claim)
        # Retrieve evidence likely to bear on the claim
        res = self.search(claim, top_k=top_k, include_chains=False)
        evidence = res.infons
        return verify_claim(
            evidence,
            claim_anchors=claim_anchors,
            schema_types=self.schema.types,
        )

    # ── Reasoning (MCTS) ────────────────────────────────────────────────

    def reason(self, query: str, *,
               max_iterations: int = 8,
               max_depth: int = 4,
               exploration_bias: float = 1.4,
               contrary: bool = False) -> MCTSResult:
        """AlphaGo-style tree search over the infon graph for multi-hop reasoning."""
        mcts = GraphMCTS(
            store=self.store, encoder=self.encoder, schema=self.schema,
            max_iterations=max_iterations, max_depth=max_depth,
            exploration_bias=exploration_bias, contrary=contrary,
        )
        return mcts.search(query, verbose=False)

    # ── Imagination (GA) ────────────────────────────────────────────────

    def imagine(self, query: str, *,
                n_generations: int | None = None,
                population_size: int | None = None,
                persona: str | None = None,
                cost_weights: dict | None = None,
                top_k: int = 20,
                store_imagined: bool = True) -> ImaginationResult:
        """Query-scoped GA imagination of plausible counterfactual infons.

        Output is isomorphic to MCTSResult — has traversal_tree, combined_mass,
        chains_discovered, iteration_log, plus imagination-native verdict
        (PLAUSIBLE/CONTRADICTED/SPECULATIVE) and MCTS-compatible mapping
        (SUPPORTS/REFUTES/UNCERTAIN) for renderer reuse.
        """
        from .imagine import Imagination  # lazy import to avoid cycle on bootstrap

        imag = Imagination(
            store=self.store, schema=self.schema, encoder=self.encoder,
            config=self.config,
        )
        result = imag.run(
            query=query,
            persona=persona or self.config.persona,
            n_generations=n_generations or self.config.imagine_generations,
            population_size=population_size or self.config.imagine_population,
            mutation_rate=self.config.imagine_mutation_rate,
            elitism=self.config.imagine_elitism,
            cost_weights=cost_weights or self.config.imagine_cost_weights,
            top_k=top_k,
        )
        if store_imagined and result.imagined_infons:
            self.store.put_infons(result.imagined_infons)
        return result

    # ── Maintenance ─────────────────────────────────────────────────────

    def decay(self, reference_date: str | None = None) -> None:
        """Apply importance decay."""
        all_infons = self.store.query_infons(limit=50000)
        decayed = apply_decay(all_infons, reference_date)
        self.store.put_infons(decayed)

    def prune(self, threshold: float | None = None) -> int:
        t = threshold if threshold is not None else self.config.prune_threshold
        return self.store.prune(t)

    def clear_imagined(self) -> int:
        """Delete all imagined infons. Returns count deleted."""
        return self.store.delete_imagined()

    def stats(self) -> dict:
        return {
            "infon_count": self.store.count_infons(),
            "imagined_count": self.store.count_infons(kind="imagined"),
            "document_count": self.store.count_documents(),
            "constraint_count": len(self.store.get_constraints(limit=10000)),
            "anchors": len(self.schema.names),
            "model": self.config.model_name or "splade-tiny (bundled)",
        }

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()
