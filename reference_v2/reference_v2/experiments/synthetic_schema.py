"""Synthetic schema builder for ablation_matrix experiments.

Generates a schema dict covering entity IDs found in the synthetic pilot
dataset. The synthetic text uses "Entity {N}" tokens where N is a global
entity ID (0-99) rather than a per-split positional index.
"""

from __future__ import annotations


def make_synthetic_schema(n_entities: int) -> dict:
    """Create a schema with one actor entry per entity ID (0..n_entities-1).

    Template sentences contain "Entity {i}" tokens so each scenario gets its
    own schema entry for precise retrieval during reason().

    Parameters
    ----------
    n_entities:
        Number of entity entries to create (should cover the max entity ID
        present in the corpus, e.g. 100 for entity IDs 0-99).

    Returns
    -------
    dict
        Schema mapping entity keys to type/token dicts, plus shared
        "confirms" and "fact" entries.
    """
    schema: dict = {}
    for i in range(n_entities):
        schema[f"entity{i}"] = {
            "type": "actor",
            "tokens": [f"entity {i}", f"Entity {i}"],
        }
    schema["confirms"] = {
        "type": "relation",
        "tokens": ["confirmed", "confirms", "confirmed true"],
    }
    schema["fact"] = {
        "type": "feature",
        "tokens": ["fact", "claim"],
    }
    return schema
