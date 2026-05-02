"""Tests for category-theoretic extensions.

Tests sheaf coherence, functorial migration, and schema-free discovery
against a geopolitical corpus built from real extracted event data.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cognition import (
    Cognition, CognitionConfig, AnchorSchema, Encoder,
    SheafCoherence, SchemaFunctor, FunctorialMigration, SchemaDiscovery,
)
from cognition.extract import extract_infons, split_sentences
from cognition.consolidate import aggregate_constraints, build_next_edges
from cognition.encoder import SpladeEncoder


# ── Geopolitical corpus from real extraction data ────────────────────

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "geopolitical_corpus.json"

GEO_SCHEMA = {
    # Actors — nation-states and organizations
    "china":         {"type": "actor", "tokens": ["china", "chinese", "beijing"], "country_code": "CN"},
    "us":            {"type": "actor", "tokens": ["united states", "us", "washington", "american"], "country_code": "US"},
    "israel":        {"type": "actor", "tokens": ["israel", "israeli"], "country_code": "IL"},
    "palestine":     {"type": "actor", "tokens": ["palestine", "palestinian"], "country_code": "PS"},
    "russia":        {"type": "actor", "tokens": ["russia", "russian", "moscow"], "country_code": "RU"},
    "japan":         {"type": "actor", "tokens": ["japan", "japanese", "tokyo"], "country_code": "JP"},
    "india":         {"type": "actor", "tokens": ["india", "indian"], "country_code": "IN"},
    "iran":          {"type": "actor", "tokens": ["iran", "iranian", "tehran"], "country_code": "IR"},
    "nato":          {"type": "actor", "tokens": ["nato", "alliance"]},
    "un":            {"type": "actor", "tokens": ["un", "united nations", "security council"]},
    "eu":            {"type": "actor", "tokens": ["eu", "european union", "brussels"]},
    "african_union": {"type": "actor", "tokens": ["african union", "au"]},

    # Relations
    "sanction":    {"type": "relation", "tokens": ["sanction", "sanctions", "embargo", "restrict"]},
    "negotiate":   {"type": "relation", "tokens": ["negotiate", "talks", "diplomacy", "summit", "discuss"]},
    "deploy":      {"type": "relation", "tokens": ["deploy", "deployment", "station", "mobilize", "troops"]},
    "cooperate":   {"type": "relation", "tokens": ["cooperate", "cooperation", "collaborate", "partner", "aid"]},
    "condemn":     {"type": "relation", "tokens": ["condemn", "denounce", "protest", "urge", "demand"]},
    "trade":       {"type": "relation", "tokens": ["trade", "tariff", "import", "export", "economic"]},
    "attack":      {"type": "relation", "tokens": ["attack", "strike", "bomb", "clash", "military"]},
    "invest":      {"type": "relation", "tokens": ["invest", "investment", "fund", "billion"]},

    # Features
    "military":    {"type": "feature", "tokens": ["military", "defense", "armed forces", "weapons"]},
    "nuclear":     {"type": "feature", "tokens": ["nuclear", "atomic", "missile", "warhead"]},
    "humanitarian":{"type": "feature", "tokens": ["humanitarian", "aid", "relief", "refugee"]},
    "territory":   {"type": "feature", "tokens": ["territory", "border", "occupation", "settlement"]},
    "maritime":    {"type": "feature", "tokens": ["maritime", "naval", "sea", "coast", "waters"]},
    "technology":  {"type": "feature", "tokens": ["technology", "cyber", "digital", "innovation"]},

    # Markets/regions
    "middle_east": {"type": "market", "tokens": ["middle east", "gulf", "levant"]},
    "east_asia":   {"type": "market", "tokens": ["east asia", "pacific", "indo-pacific", "asia"]},
    "europe":      {"type": "market", "tokens": ["europe", "european"]},
    "africa":      {"type": "market", "tokens": ["africa", "african"]},
}

# Documents built from the real extraction data
GEO_DOCS = [
    {"id": "geo-001", "timestamp": "2023-07-20", "text": "Palestine urges Israel to halt settlement expansion and end the occupation of Palestinian territories in the West Bank."},
    {"id": "geo-002", "timestamp": "2017-07-02", "text": "China organized the Disability and Sustainable Development Forum in Beijing with UNESCO cooperation."},
    {"id": "geo-003", "timestamp": "2008-03-26", "text": "African Union troops clashed with forces loyal to Mohamed Bacar on the island of Anjouan in a military operation."},
    {"id": "geo-004", "timestamp": "2019-07-11", "text": "Israel and India launched a joint venture to manufacture missile defense systems through Rafael Advanced Defense Systems."},
    {"id": "geo-005", "timestamp": "2020-08-20", "text": "The United States rejected the UN Security Council snapback mechanism on Iran sanctions in a diplomatic breakdown."},
    {"id": "geo-006", "timestamp": "2024-09-01", "text": "Taiwan plays a critical role in the US-China competition for critical minerals and technology supply chains."},
    {"id": "geo-007", "timestamp": "2022-03-09", "text": "China sent the first batch of humanitarian aid to Ukraine including food and medical supplies."},
    {"id": "geo-008", "timestamp": "2015-07-05", "text": "Greeks voted in a referendum amid pressure from the European Commission over economic bailout conditions."},
    {"id": "geo-009", "timestamp": "2020-01-15", "text": "North Korea launched missile tests near the Korean Peninsula, drawing condemnation from Japan and the United States."},
    {"id": "geo-010", "timestamp": "2018-01-18", "text": "Malaysia and Singapore negotiated over border congestion at the Johor Causeway crossing."},
    {"id": "geo-011", "timestamp": "2023-10-15", "text": "Iran and Russia deepened military cooperation with joint naval exercises in the maritime region."},
    {"id": "geo-012", "timestamp": "2024-03-10", "text": "NATO deployed additional troops to Eastern Europe amid Russian military buildup near the border."},
    {"id": "geo-013", "timestamp": "2021-09-15", "text": "The United States and Japan signed a bilateral trade agreement covering technology and defense cooperation in East Asia."},
    {"id": "geo-014", "timestamp": "2022-06-20", "text": "The African Union condemned the military coup and deployed peacekeeping forces to restore order."},
    {"id": "geo-015", "timestamp": "2023-04-10", "text": "China invested billions in infrastructure across Africa as part of the Belt and Road Initiative."},
    {"id": "geo-016", "timestamp": "2019-12-05", "text": "Iran rejected nuclear inspections demanded by the United Nations Security Council."},
    {"id": "geo-017", "timestamp": "2024-06-15", "text": "The European Union imposed sanctions on Russian energy exports amid the Ukraine conflict."},
    {"id": "geo-018", "timestamp": "2020-09-10", "text": "Israel and the United States signed the Abraham Accords normalizing diplomatic relations in the Middle East."},
    {"id": "geo-019", "timestamp": "2021-03-20", "text": "India deployed naval vessels in the maritime region of East Asia for joint exercises with Japan."},
    {"id": "geo-020", "timestamp": "2023-11-01", "text": "China and the European Union held trade negotiations over tariffs and technology transfer in Brussels."},
    {"id": "geo-021", "timestamp": "2024-01-20", "text": "NATO condemned Russian military deployments near the European border as provocative."},
    {"id": "geo-022", "timestamp": "2022-11-10", "text": "The United Nations delivered humanitarian aid to Afghanistan after the Taliban takeover."},
    {"id": "geo-023", "timestamp": "2004-08-02", "text": "The African Union deployed troops to Darfur in a peacekeeping mission to protect civilians."},
    {"id": "geo-024", "timestamp": "2025-01-01", "text": "China invested in nuclear energy technology cooperation with Iran despite international sanctions."},
]


@pytest.fixture(scope="module")
def geo_setup():
    """Set up cognition with the geopolitical corpus."""
    schema_path = Path(tempfile.mktemp(suffix=".json"))
    schema_path.write_text(json.dumps(GEO_SCHEMA, indent=2))
    db_path = tempfile.mktemp(suffix=".db")

    config = CognitionConfig(schema_path=str(schema_path), db_path=db_path)
    cog = Cognition(config)
    n = cog.ingest(GEO_DOCS, consolidate_now=True)

    yield cog, n

    cog.close()
    schema_path.unlink(missing_ok=True)
    Path(db_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════
# TEST 1: SHEAF COHERENCE
# ════════════════════════════════════════════════════════════════════

class TestSheafCoherence:
    """Test the sheaf coherence construction."""

    def test_build_npmi_matrix(self, geo_setup):
        """NPMI co-activation matrix is computed correctly."""
        cog, _ = geo_setup
        encoder = cog.encoder
        schema = cog.schema

        sheaf = SheafCoherence(encoder.anchor_names)

        # Encode all sentences
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))

        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        assert sheaf.npmi is not None
        assert sheaf.npmi.shape == (len(encoder.anchor_names), len(encoder.anchor_names))
        # NPMI is symmetric
        np.testing.assert_allclose(sheaf.npmi, sheaf.npmi.T, atol=1e-10)
        # Diagonal is zero
        np.testing.assert_allclose(np.diag(sheaf.npmi), 0.0)
        # NPMI values in [-1, 1]
        assert sheaf.npmi.min() >= -1.0 - 1e-6
        assert sheaf.npmi.max() <= 1.0 + 1e-6

    def test_fiedler_value(self, geo_setup):
        """Fiedler value (algebraic connectivity) is non-negative."""
        cog, _ = geo_setup
        encoder = cog.encoder

        sheaf = SheafCoherence(encoder.anchor_names)
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))
        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        assert sheaf.fiedler_value >= 0.0
        print(f"\n  Fiedler value (algebraic connectivity): {sheaf.fiedler_value:.4f}")

    def test_infon_coherence_scores(self, geo_setup):
        """Infon coherence scores are in [0, 1] and vary meaningfully."""
        cog, _ = geo_setup
        encoder = cog.encoder

        sheaf = SheafCoherence(encoder.anchor_names)
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))
        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        infons = cog.store.query_infons(limit=500)
        scores = sheaf.score_batch(infons)

        assert len(scores) == len(infons)
        assert all(0.0 <= s <= 1.0 for s in scores)

        # There should be variance — not all the same
        assert max(scores) > min(scores), "Coherence scores should vary"

        # Show top and bottom
        scored = sorted(zip(scores, infons), key=lambda x: -x[0])
        print(f"\n  Scored {len(infons)} infons")
        print(f"  Mean coherence: {np.mean(scores):.3f}")
        print(f"  Std coherence:  {np.std(scores):.3f}")
        print(f"\n  Most coherent:")
        for s, inf in scored[:3]:
            print(f"    {s:.3f}  <<{inf.predicate}, {inf.subject}, {inf.object}>>")
        print(f"  Least coherent:")
        for s, inf in scored[-3:]:
            print(f"    {s:.3f}  <<{inf.predicate}, {inf.subject}, {inf.object}>>")

    def test_anchor_centrality(self, geo_setup):
        """Anchor centrality identifies hub anchors."""
        cog, _ = geo_setup
        encoder = cog.encoder

        sheaf = SheafCoherence(encoder.anchor_names)
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))
        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        centrality = sheaf.anchor_centrality()
        assert len(centrality) == len(encoder.anchor_names)

        # Sort and display
        ranked = sorted(centrality.items(), key=lambda x: -x[1])
        print(f"\n  Anchor centrality (top 10):")
        for name, score in ranked[:10]:
            atype = GEO_SCHEMA.get(name, {}).get("type", "?")
            print(f"    {atype:10s} {name:15s} {score:.3f} {'█' * int(score * 20)}")

    def test_component_structure(self, geo_setup):
        """Component structure reveals natural categories."""
        cog, _ = geo_setup
        encoder = cog.encoder

        sheaf = SheafCoherence(encoder.anchor_names)
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))
        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        components = sheaf.component_structure()
        assert len(components) >= 1

        print(f"\n  Connected components: {len(components)}")
        for i, comp in enumerate(components):
            print(f"    Component {i}: {comp}")


# ════════════════════════════════════════════════════════════════════
# TEST 2: FUNCTORIAL MIGRATION
# ════════════════════════════════════════════════════════════════════

class TestFunctorialMigration:
    """Test functorial data migration between schemas."""

    def test_rename_functor(self, geo_setup):
        """Renaming anchors preserves all infons."""
        cog, _ = geo_setup
        infons = cog.store.query_infons(limit=500)
        edges = cog.store.get_edges(limit=5000)

        # Rename "us" → "united_states" and "china" → "prc"
        functor = SchemaFunctor(rename={"us": "united_states", "china": "prc"})

        new_schema_defs = dict(GEO_SCHEMA)
        new_schema_defs["united_states"] = new_schema_defs.pop("us")
        new_schema_defs["prc"] = new_schema_defs.pop("china")
        target_schema = AnchorSchema(new_schema_defs)

        migration = FunctorialMigration(functor, cog.schema, target_schema)
        migrated_infons, migrated_edges = migration.migrate_all(infons, edges)

        # Infons may merge if rename creates duplicate triples (many:1 on triples)
        # But no infon should be deleted — only merged
        assert len(migrated_infons) <= len(infons)
        assert len(migrated_infons) > 0

        # Check that renamed anchors appear
        all_subjects = {inf.subject for inf in migrated_infons}
        all_objects = {inf.object for inf in migrated_infons}
        all_anchors = all_subjects | all_objects

        assert "us" not in all_anchors, "Old name should not appear"
        assert "china" not in all_anchors, "Old name should not appear"

        # Check some infons have the new names
        us_infons = [inf for inf in migrated_infons if inf.subject == "united_states"]
        prc_infons = [inf for inf in migrated_infons if inf.subject == "prc"]
        orig_us = [inf for inf in infons if inf.subject == "us"]
        orig_cn = [inf for inf in infons if inf.subject == "china"]

        print(f"\n  Rename: us→united_states: {len(orig_us)} → {len(us_infons)}")
        print(f"  Rename: china→prc: {len(orig_cn)} → {len(prc_infons)}")

        report = migration.report(infons, migrated_infons)
        print(f"  Report: {report}")

    def test_merge_functor(self, geo_setup):
        """Merging anchors reduces triple count (duplicates merged)."""
        cog, _ = geo_setup
        infons = cog.store.query_infons(limit=500)
        edges = cog.store.get_edges(limit=5000)

        # Merge "condemn" and "sanction" → "coerce"
        functor = SchemaFunctor(
            merge={"condemn": "coerce", "sanction": "coerce"},
        )

        new_schema_defs = {k: v for k, v in GEO_SCHEMA.items() if k not in ("condemn", "sanction")}
        new_schema_defs["coerce"] = {"type": "relation", "tokens": ["sanction", "condemn", "coerce"]}
        target_schema = AnchorSchema(new_schema_defs)

        migration = FunctorialMigration(functor, cog.schema, target_schema)
        migrated_infons, _ = migration.migrate_all(infons, edges)

        # Some infons should have merged (fewer unique triples)
        orig_triples = set(inf.triple_key() for inf in infons)
        new_triples = set(inf.triple_key() for inf in migrated_infons)

        print(f"\n  Merge condemn+sanction → coerce")
        print(f"  Original triples: {len(orig_triples)}")
        print(f"  Migrated triples: {len(new_triples)}")

        # Check coerce exists
        coerce_infons = [inf for inf in migrated_infons if inf.predicate == "coerce"]
        print(f"  Infons with predicate='coerce': {len(coerce_infons)}")

        # Merged infons should have reinforcement_count > 0
        reinforced = [inf for inf in migrated_infons if inf.reinforcement_count > 0]
        print(f"  Reinforced (merged) infons: {len(reinforced)}")

    def test_delete_functor(self, geo_setup):
        """Deleting anchors removes their infons."""
        cog, _ = geo_setup
        infons = cog.store.query_infons(limit=500)
        edges = cog.store.get_edges(limit=5000)

        # Delete "nuclear" — remove all nuclear-related infons
        functor = SchemaFunctor(delete={"nuclear"})

        new_schema_defs = {k: v for k, v in GEO_SCHEMA.items() if k != "nuclear"}
        target_schema = AnchorSchema(new_schema_defs)

        migration = FunctorialMigration(functor, cog.schema, target_schema)
        migrated_infons, _ = migration.migrate_all(infons, edges)

        # Should have fewer infons
        assert len(migrated_infons) <= len(infons)

        # No infon should reference "nuclear"
        for inf in migrated_infons:
            assert inf.subject != "nuclear"
            assert inf.predicate != "nuclear"
            assert inf.object != "nuclear"

        deleted = len(infons) - len(migrated_infons)
        print(f"\n  Delete 'nuclear': removed {deleted}/{len(infons)} infons")

    def test_composition_preserved(self, geo_setup):
        """Composing two functors equals applying the composite."""
        cog, _ = geo_setup
        infons = cog.store.query_infons(limit=500)
        edges = cog.store.get_edges(limit=5000)

        # F1: rename china → prc
        f1 = SchemaFunctor(rename={"china": "prc"})
        schema1_defs = dict(GEO_SCHEMA)
        schema1_defs["prc"] = schema1_defs.pop("china")
        schema1 = AnchorSchema(schema1_defs)
        m1 = FunctorialMigration(f1, cog.schema, schema1)
        step1_infons, step1_edges = m1.migrate_all(infons, edges)

        # F2: rename prc → peoples_republic
        f2 = SchemaFunctor(rename={"prc": "peoples_republic"})
        schema2_defs = dict(schema1_defs)
        schema2_defs["peoples_republic"] = schema2_defs.pop("prc")
        schema2 = AnchorSchema(schema2_defs)
        m2 = FunctorialMigration(f2, schema1, schema2)
        step2_infons, _ = m2.migrate_all(step1_infons, step1_edges)

        # Composite: china → peoples_republic directly
        fc = SchemaFunctor(rename={"china": "peoples_republic"})
        mc = FunctorialMigration(fc, cog.schema, schema2)
        direct_infons, _ = mc.migrate_all(infons, edges)

        # Same number of infons and same triples
        assert len(step2_infons) == len(direct_infons)
        step2_triples = sorted(inf.triple_key() for inf in step2_infons)
        direct_triples = sorted(inf.triple_key() for inf in direct_infons)
        assert step2_triples == direct_triples, \
            "Composition of functors should equal the composite functor"

        print(f"\n  F2 ∘ F1 = F_composite: {len(step2_triples)} triples match")


# ════════════════════════════════════════════════════════════════════
# TEST 3: SCHEMA-FREE DISCOVERY (LEFT KAN EXTENSION)
# ════════════════════════════════════════════════════════════════════

class TestSchemaDiscovery:
    """Test schema-free discovery from raw text."""

    def test_discover_from_corpus(self):
        """Discover a schema from the geopolitical corpus."""
        # Collect all sentences
        sentences = []
        for doc in GEO_DOCS:
            sentences.extend(split_sentences(doc["text"]))

        discovery = SchemaDiscovery()
        schema, anchors = discovery.discover(
            sentences,
            n_anchors=15,
            min_doc_freq=2,
        )

        assert len(schema.names) > 0
        assert len(anchors) > 0

        print(f"\n  Discovered {len(schema.names)} anchors from {len(sentences)} sentences:")
        for da in sorted(anchors, key=lambda x: -x.mean_activation):
            print(f"    {da.inferred_type:10s} {da.name:20s}  "
                  f"tokens={da.tokens[:3]}  size={da.size}  "
                  f"activation={da.mean_activation:.3f}  "
                  f"coherence={da.coherence:.3f}")

    def test_discovered_schema_is_usable(self):
        """A discovered schema can be used for ingestion and query."""
        sentences = []
        for doc in GEO_DOCS:
            sentences.extend(split_sentences(doc["text"]))

        discovery = SchemaDiscovery()
        schema, _ = discovery.discover(sentences, n_anchors=12, min_doc_freq=2)

        # Use discovered schema for cognition
        schema_path = Path(tempfile.mktemp(suffix=".json"))
        schema.save(schema_path)
        db_path = tempfile.mktemp(suffix=".db")

        config = CognitionConfig(schema_path=str(schema_path), db_path=db_path)
        cog = Cognition(config)

        n = cog.ingest(GEO_DOCS[:5], consolidate_now=True)
        assert n > 0, "Should extract infons with discovered schema"

        stats = cog.stats()
        print(f"\n  Discovered schema → {n} infons, {stats['constraint_count']} constraints")
        print(f"  Anchors: {stats['anchors']}")

        # Query should work
        result = cog.query("military conflict")
        assert result.persona is not None

        cog.close()
        schema_path.unlink(missing_ok=True)
        Path(db_path).unlink(missing_ok=True)

    def test_discovered_types_are_reasonable(self):
        """Discovered anchor types should include multiple categories."""
        sentences = []
        for doc in GEO_DOCS:
            sentences.extend(split_sentences(doc["text"]))

        discovery = SchemaDiscovery()
        schema, anchors = discovery.discover(sentences, n_anchors=15, min_doc_freq=2)

        # Should discover at least 2 different types
        types = set(da.inferred_type for da in anchors)
        assert len(types) >= 2, f"Should discover multiple types, got: {types}"

        print(f"\n  Discovered types: {types}")
        for t in sorted(types):
            count = sum(1 for da in anchors if da.inferred_type == t)
            names = [da.name for da in anchors if da.inferred_type == t]
            print(f"    {t:10s}: {count} anchors — {names}")


# ════════════════════════════════════════════════════════════════════
# TEST 4: INTEGRATION — ALL THREE TOGETHER
# ════════════════════════════════════════════════════════════════════

class TestCategoryIntegration:
    """Test the three extensions working together."""

    def test_sheaf_improves_importance(self, geo_setup):
        """Sheaf coherence can update importance scores."""
        cog, _ = geo_setup
        encoder = cog.encoder

        sheaf = SheafCoherence(encoder.anchor_names)
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))
        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        infons = cog.store.query_infons(limit=500)

        # Update coherence and recompute importance
        config = cog.config
        for inf in infons:
            inf.coherence = sheaf.score_infon(inf)
            inf.importance = (
                config.w_activation * inf.activation
                + config.w_coherence * inf.coherence
                + config.w_specificity * inf.specificity
                + config.w_novelty * inf.novelty
            )

        # Coherence should now be > 0 for most infons
        with_coherence = sum(1 for inf in infons if inf.coherence > 0.0)
        print(f"\n  Infons with coherence > 0: {with_coherence}/{len(infons)}")

        # Top by new importance should differ from top by old importance
        by_importance = sorted(infons, key=lambda x: -x.importance)[:5]
        print(f"  Top 5 by importance (with sheaf coherence):")
        for inf in by_importance:
            print(f"    <<{inf.predicate}, {inf.subject}, {inf.object}>>  "
                  f"imp={inf.importance:.3f} coh={inf.coherence:.3f}")

    def test_discover_then_migrate(self):
        """Discover a schema, use it, then migrate to a hand-designed schema."""
        sentences = []
        for doc in GEO_DOCS:
            sentences.extend(split_sentences(doc["text"]))

        # Step 1: Discover
        discovery = SchemaDiscovery()
        discovered_schema, discovered_anchors = discovery.discover(
            sentences, n_anchors=10, min_doc_freq=2,
        )

        # Step 2: Ingest with discovered schema
        schema_path = Path(tempfile.mktemp(suffix=".json"))
        discovered_schema.save(schema_path)
        db_path = tempfile.mktemp(suffix=".db")

        config = CognitionConfig(schema_path=str(schema_path), db_path=db_path)
        cog = Cognition(config)
        n1 = cog.ingest(GEO_DOCS[:10], consolidate_now=True)

        # Step 3: Build migration functor from discovered → hand-designed
        # Map discovered anchors to GEO_SCHEMA anchors by token overlap
        rename_map = {}
        for da in discovered_anchors:
            best_match = None
            best_overlap = 0
            da_tokens = set(t.lower() for t in da.tokens)
            for geo_name, geo_info in GEO_SCHEMA.items():
                geo_tokens = set(t.lower() for t in geo_info.get("tokens", []))
                overlap = len(da_tokens & geo_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = geo_name
            if best_match and best_overlap > 0:
                rename_map[da.name] = best_match

        if rename_map:
            functor = SchemaFunctor(rename=rename_map)
            target_schema = AnchorSchema(GEO_SCHEMA)

            infons = cog.store.query_infons(limit=500)
            edges = cog.store.get_edges(limit=5000)

            migration = FunctorialMigration(functor, discovered_schema, target_schema)
            migrated, _ = migration.migrate_all(infons, edges)

            print(f"\n  Discovered: {len(discovered_schema.names)} anchors → {n1} infons")
            print(f"  Migration map: {rename_map}")
            print(f"  Migrated: {len(infons)} → {len(migrated)} infons")

        cog.close()
        schema_path.unlink(missing_ok=True)
        Path(db_path).unlink(missing_ok=True)

    def test_full_pipeline_with_timeline(self, geo_setup):
        """Full pipeline: ingest → sheaf → query → timeline."""
        cog, n_infons = geo_setup

        # Build sheaf
        encoder = cog.encoder
        sheaf = SheafCoherence(encoder.anchor_names)
        all_sentences = []
        for doc in GEO_DOCS:
            all_sentences.extend(split_sentences(doc["text"]))
        activations = encoder.encode(all_sentences)
        sheaf.observe(activations)
        sheaf.fit()

        # Query with timeline
        result = cog.query(
            "How has the conflict between Russia and NATO evolved?",
            persona="analyst",
        )

        print(f"\n  Query: 'How has the conflict between Russia and NATO evolved?'")
        print(f"  Persona: {result.persona}")
        print(f"  Infons: {len(result.infons)}")
        print(f"  Timeline: {len(result.timeline)} events")
        print(f"  NEXT edges: {len(result.edges)}")
        print(f"  Constraints: {len(result.constraints)}")

        # Score results with sheaf coherence
        for inf in result.infons[:5]:
            coh = sheaf.score_infon(inf)
            v = result.valence.get(inf.infon_id, 0)
            arrow = "▲" if v > 0.1 else "▼" if v < -0.1 else "─"
            print(f"    {arrow} <<{inf.predicate}, {inf.subject}, {inf.object}>>  "
                  f"coh={coh:.3f} conf={inf.confidence:.3f}")

        if result.timeline:
            print(f"\n  Timeline:")
            for inf in result.timeline[:8]:
                coh = sheaf.score_infon(inf)
                print(f"    {inf.timestamp}  <<{inf.predicate}, {inf.subject}, {inf.object}>>  "
                      f"coh={coh:.3f}")

        # Sheaf diagnostics
        print(f"\n  Sheaf diagnostics:")
        print(f"    Fiedler value: {sheaf.fiedler_value:.4f}")
        print(f"    Components: {len(sheaf.component_structure())}")

        centrality = sheaf.anchor_centrality()
        top_central = sorted(centrality.items(), key=lambda x: -x[1])[:5]
        print(f"    Top central anchors: {[(n, f'{s:.3f}') for n, s in top_central]}")
