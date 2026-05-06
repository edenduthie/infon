"""cog.expand(): θ-triggered active exploration with mock + ddgs backends."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


SCHEMA = {
    "toyota":    {"type": "actor",    "tokens": ["toyota"]},
    "honda":     {"type": "actor",    "tokens": ["honda"]},
    "tesla":     {"type": "actor",    "tokens": ["tesla"]},
    "catl":      {"type": "actor",    "tokens": ["catl"]},
    "invests":   {"type": "relation", "tokens":
                  ["invest", "invests"]},
    "partners":  {"type": "relation", "tokens":
                  ["partner", "partners"]},
    "acquires":  {"type": "relation", "tokens":
                  ["acquire", "acquires", "acquired"]},
    "produces":  {"type": "relation", "tokens":
                  ["produce", "produces"]},
    "battery":   {"type": "feature",  "tokens": ["battery", "batteries"]},
}

DOCS = [
    {"id": "d1", "text": "Toyota invests in battery technology."},
    {"id": "d2", "text": "Tesla produces batteries."},
]


def _make_cog(tmpdir):
    from infon import Cognition, CognitionConfig
    schema_path = os.path.join(tmpdir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f)
    return Cognition(CognitionConfig(
        schema_path=schema_path,
        db_path=os.path.join(tmpdir, "cog.db"),
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
        quality_threshold=0.04,
        max_triples_per_sentence=2,
    ))


def test_expand_no_action_below_threshold():
    """When θ is below the threshold, expand() returns without fetching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()

        def never_called(q):
            raise AssertionError("search should not have been called")

        result = cog.expand(
            "Does Toyota invest in batteries?",
            theta_threshold=0.99,   # extremely high → never triggers
            source="mock",
            search_fn=never_called,
        )
        assert result["expanded"] is False
        assert result["n_new_docs"] == 0
        assert result["after"] is None
        cog.close()


def test_expand_fetches_when_theta_high():
    """When θ is high, expand() fetches from the mock backend, ingests
    the results, and re-queries. Post-query θ should drop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()

        # The claim "Did Tesla acquire CATL?" is not in the corpus;
        # should trigger expand.
        # Mock backend returns a supporting snippet.
        def mock_search(query):
            return [{
                "title": "Tesla acquires CATL",
                "body": "Tesla acquires CATL battery business for $10B.",
                "href": "https://example.com/tesla-catl",
            }]

        result = cog.expand(
            "Did Tesla acquire CATL?",
            theta_threshold=0.3,
            source="mock",
            search_fn=mock_search,
            verbose=True,
        )
        print(f"\n  expanded={result['expanded']}")
        print(f"  n_new_docs={result['n_new_docs']}, "
              f"n_new_infons={result.get('n_new_infons')}")
        print(f"  before: {result['before']}")
        print(f"  after:  {result['after']}")

        assert result["expanded"] is True
        assert result["n_new_docs"] == 1
        # After ingesting the supporting doc, post-θ should be lower
        # than before-θ (the system has new evidence).
        assert result["after"]["theta"] <= result["before"]["theta"]
        # And the SUPPORTS mass should go up
        assert result["after"]["supports"] >= result["before"]["supports"]
        cog.close()


def test_expand_handles_empty_search_results():
    """When the mock backend returns no snippets, expand() should
    short-circuit cleanly and not bump the generation counter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()

        gen_before = cog._generation

        result = cog.expand(
            "Did Tesla acquire CATL?",
            theta_threshold=0.3,
            source="mock",
            search_fn=lambda q: [],  # nothing to fetch
        )
        assert result["expanded"] is False
        assert cog._generation == gen_before
        cog.close()


def test_expand_unknown_source_raises():
    """source='bogus' raises."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()
        with pytest.raises(ValueError):
            cog.expand("anything", theta_threshold=0.01, source="bogus")
        cog.close()


def test_expand_with_ddgs_backend_runs_if_installed():
    """If ddgs is installed, the 'ddgs' source should run without
    error (we don't assert on quality since results change over time)."""
    try:
        import ddgs  # noqa
    except ImportError:
        pytest.skip("ddgs not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()

        result = cog.expand(
            "Toyota battery investment",
            max_docs=2,
            theta_threshold=0.3,
            source="ddgs",
            verbose=True,
        )
        print(f"\n  ddgs returned expanded={result['expanded']}")
        print(f"  n_new_docs={result['n_new_docs']}")
        print(f"  sources={result['sources']}")
        assert "expanded" in result
        assert "before" in result
        cog.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
