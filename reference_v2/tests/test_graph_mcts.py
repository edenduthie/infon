"""Tests for Graph MCTS traversal."""

import pytest
from pathlib import Path

from cognition import Cognition, CognitionConfig
from cognition.graph_mcts import GraphMCTS, MCTSResult, format_mcts_result


SCHEMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "automotive_schema.json")


@pytest.fixture
def populated_cog():
    """Build a small knowledge graph for testing."""
    config = CognitionConfig(
        schema_path=SCHEMA_PATH,
        db_path=":memory:",
        activation_threshold=0.15,
        min_confidence=0.02,
        top_k_per_role=5,
        default_top_k=50,
        consolidation_interval=3,
    )
    cog = Cognition(config)
    docs = [
        {"text": "Toyota invested 13 billion in battery technology for solid-state batteries.",
         "id": "d1", "timestamp": "2023-06-01"},
        {"text": "Toyota's EV market share grew from 3 percent to 5 percent in 2024.",
         "id": "d2", "timestamp": "2024-06-15"},
        {"text": "Toyota recalled vehicles due to battery cooling issues.",
         "id": "d3", "timestamp": "2024-04-12"},
        {"text": "Tesla maintains cost advantage over Toyota in battery production.",
         "id": "d4", "timestamp": "2024-10-01"},
        {"text": "Ford will license Toyota solid-state battery technology.",
         "id": "d5", "timestamp": "2024-11-20"},
    ]
    cog.ingest(docs, consolidate_now=True)
    yield cog
    cog.close()


class TestGraphMCTS:
    def test_search_returns_result(self, populated_cog):
        mcts = GraphMCTS(
            store=populated_cog.store,
            encoder=populated_cog.encoder,
            schema=populated_cog.schema,
            max_iterations=3,
        )
        result = mcts.search("Did Toyota invest in batteries?")
        assert isinstance(result, MCTSResult)
        assert result.verdict in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
        assert result.iterations > 0
        assert result.nodes_explored >= 1
        assert result.infons_evaluated > 0

    def test_search_explores_graph(self, populated_cog):
        mcts = GraphMCTS(
            store=populated_cog.store,
            encoder=populated_cog.encoder,
            schema=populated_cog.schema,
            max_iterations=5,
            max_depth=3,
        )
        result = mcts.search("What is Toyota doing with battery technology?")
        assert result.nodes_explored > 1
        assert len(result.chains_discovered) > 0

    def test_mass_is_normalized(self, populated_cog):
        mcts = GraphMCTS(
            store=populated_cog.store,
            encoder=populated_cog.encoder,
            schema=populated_cog.schema,
            max_iterations=3,
        )
        result = mcts.search("Toyota battery investment")
        m = result.combined_mass
        total = m.supports + m.refutes + m.uncertain + m.theta
        assert abs(total - 1.0) < 1e-5

    def test_format_output(self, populated_cog):
        mcts = GraphMCTS(
            store=populated_cog.store,
            encoder=populated_cog.encoder,
            schema=populated_cog.schema,
            max_iterations=3,
        )
        result = mcts.search("Did Toyota invest in batteries?")
        output = format_mcts_result(result)
        assert "Query:" in output
        assert "Verdict:" in output
        assert "MCTS Traversal" in output

    def test_empty_graph(self):
        """MCTS on empty graph should return NOT ENOUGH INFO."""
        config = CognitionConfig(
            schema_path=SCHEMA_PATH,
            db_path=":memory:",
        )
        cog = Cognition(config)
        cog.store.init()
        mcts = GraphMCTS(
            store=cog.store,
            encoder=cog.encoder,
            schema=cog.schema,
            max_iterations=3,
        )
        result = mcts.search("Anything at all")
        assert result.verdict == "NOT ENOUGH INFO"
        cog.close()
