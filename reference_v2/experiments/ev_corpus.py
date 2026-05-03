"""EV battery scenario corpus for Epic 01 experiments.

This module centralises the synthetic EV documents and schema used by:

- ``tests/test_logic.py`` (the 14-test regression baseline; previously the
  source of truth — kept importing the literal until A.6b moved it here),
- ``experiments/run.py`` (Stage A.6b runner — fits a reasoner and writes a
  JSON report per seed),
- the various test modules that follow A.2/A.3/A.4/A.5 (they import
  ``DOCUMENTS`` and ``setup_cognition`` via ``tests.test_logic`` for
  compatibility — see the re-export at the top of that test file).

Refactor rationale (see infon-6o3.11 / A.6b notes):

- Stage B.1 reproduces the audit's baseline numbers via the runner; the
  runner therefore must run on the same EV corpus the regression tests
  use, otherwise "the runner reproduces test_logic.py's numbers" cannot
  be a meaningful sanity check.
- Importing ``tests.test_logic`` from ``experiments.run`` would create a
  test ↔ production cycle (production code depending on the test layout,
  including breaking on collection if pytest is missing or the tests are
  excluded from a sdist).
- The cleanest fix is to put the corpus in production code (here) and
  have the test re-export it; the literal moves once and both call sites
  pick it up unchanged.

The shapes of ``DOCUMENTS`` and ``SCHEMA_DEFS`` are identical to the
literal A.6b found in ``tests/test_logic.py`` at the time of the move —
no semantics changed.

Schema overview
---------------
Five private-sector actors (Toyota, Honda, Tesla, Panasonic, CATL),
six relations (invests, partners, produces, expands, delays, acquires),
five features (battery, solid_state, ev, factory, supply_chain),
three markets (japan, north_america, china).

Documents overview
------------------
Five short paragraphs describing actor/relation/feature interactions
across the EV battery supply chain. Coverage is deliberately
heterogeneous:

- doc1: Toyota → invests/partners → solid-state batteries (JP)
- doc2: Tesla → expands/produces/acquires → battery factory (US)
- doc3: Honda → delays/partners/invests → EV/battery (CN partner: CATL)
- doc4: CATL/Panasonic → expands/produces/invests → batteries (CN/JP)
- doc5: Toyota → invests/produces → solid-state breakthrough

Together they yield the four diagnostic queries Epic 01 cares about:
Toyota, Honda, Tesla, CATL.
"""

from __future__ import annotations

# ── Synthetic schema: actors, relations, features, markets ────────────

SCHEMA_DEFS: dict[str, dict] = {
    # Actors
    "toyota": {"type": "actor", "tokens": ["toyota"], "country_code": "JP",
               "organisation_type": "private-sector"},
    "honda": {"type": "actor", "tokens": ["honda"], "country_code": "JP",
              "organisation_type": "private-sector"},
    "tesla": {"type": "actor", "tokens": ["tesla"], "country_code": "US",
              "organisation_type": "private-sector"},
    "panasonic": {"type": "actor", "tokens": ["panasonic"], "country_code": "JP",
                  "organisation_type": "private-sector"},
    "catl": {"type": "actor", "tokens": ["catl"], "country_code": "CN",
             "organisation_type": "private-sector"},

    # Relations
    "invests": {"type": "relation", "tokens": ["invest", "invests", "invested", "investment"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners", "partnered", "partnership"]},
    "produces": {"type": "relation", "tokens": ["produce", "produces", "produced", "production"]},
    "expands": {"type": "relation", "tokens": ["expand", "expands", "expanded", "expansion"]},
    "delays": {"type": "relation", "tokens": ["delay", "delays", "delayed"]},
    "acquires": {"type": "relation", "tokens": ["acquire", "acquires", "acquired", "acquisition"]},

    # Features
    "battery": {"type": "feature", "tokens": ["battery", "batteries"]},
    "solid_state": {"type": "feature", "tokens": ["solid-state", "solid state"],
                    "parent": "battery"},
    "ev": {"type": "feature", "tokens": ["ev", "electric vehicle", "electric vehicles"]},
    "factory": {"type": "feature", "tokens": ["factory", "plant", "facility"]},
    "supply_chain": {"type": "feature", "tokens": ["supply chain", "supply"]},

    # Markets
    "japan": {"type": "market", "tokens": ["japan", "japanese"], "country_code": "JP",
              "macro_region": "asia_pacific"},
    "north_america": {"type": "market", "tokens": ["north america", "us", "united states"],
                      "macro_region": "americas"},
    "china": {"type": "market", "tokens": ["china", "chinese"], "country_code": "CN",
              "macro_region": "asia_pacific"},
}

# ── Synthetic documents: a coherent EV battery scenario ───────────────

DOCUMENTS: list[dict] = [
    {
        "id": "doc1",
        "text": (
            "Toyota invests heavily in solid-state battery technology. "
            "The company announced a $13.6 billion investment in battery production. "
            "Toyota partners with Panasonic on battery development in Japan."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Tesla expands its battery factory in North America. "
            "Tesla produces batteries at its Gigafactory facility. "
            "Tesla acquires battery supply chain assets to reduce costs."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Honda delays its electric vehicle production timeline. "
            "Honda partners with CATL for battery supply in China. "
            "Honda invests in solid-state battery research but has not produced results."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "CATL expands battery production capacity in China. "
            "CATL produces batteries for multiple Japanese automakers. "
            "Panasonic invests in new battery factory in Japan."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "Toyota's solid-state battery investment leads to a breakthrough. "
            "Toyota produces prototype solid-state batteries ahead of schedule. "
            "If Toyota succeeds in solid-state batteries, it could reshape the EV market."
        ),
    },
]


# ── Diagnostic queries (Epic 01 acceptance gate + Stage B/C fixtures) ──
#
# Wording matches tests/test_logic.py::test_reasoner_end_to_end /
# test_full_pipeline. The Toyota and Honda probes are the load-bearing
# ones for the epic acceptance criterion (verdict polarity = SUPPORTS,
# m(Θ) ∈ [0.20, 0.40] under five pinned seeds); Tesla and CATL widen
# coverage so Stage B's sweep can detect cross-query polarity flips
# without re-running on a different fixture set.

DIAGNOSTIC_QUERIES: dict[str, str] = {
    "toyota": "Did Toyota invest in battery technology?",
    "honda": "Did Honda delay electric vehicles?",
    "tesla": "Is Tesla expanding production?",
    "catl": "Does CATL produce batteries?",
}


def setup_cognition(db_path: str):
    """Create a Cognition instance with the EV schema (does NOT ingest).

    The caller ingests ``DOCUMENTS`` themselves so that test fixtures
    can vary the ingestion order or filter to a subset without forking
    the schema. Mirrors the prior signature in ``tests/test_logic.py``.

    Parameters
    ----------
    db_path : str
        Path to the on-disk SQLite store (typically a tmp file).

    Returns
    -------
    cognition.Cognition
        A configured but un-ingested Cognition instance.
    """
    import json
    from cognition import Cognition, CognitionConfig

    # Schema is written next to the db so a tmpdir cleanup removes both.
    schema_path = db_path.replace(".db", "_schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA_DEFS, f)

    config = CognitionConfig(
        schema_path=schema_path,
        db_path=db_path,
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
    )
    return Cognition(config)
