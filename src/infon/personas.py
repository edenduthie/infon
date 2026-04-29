"""Persona valence tables for retrieval ranking.

Each persona has a set of predicate anchors with associated valence weights:
- Positive predicates (+1): Valued highly by this persona
- Negative predicates (-1): Viewed negatively by this persona
- Neutral predicates (0): No special preference (default for unlisted predicates)

The valence weight is used in retrieval scoring to shift ranking based on persona context.
"""

from typing import Literal

PersonaType = Literal["investor", "engineer", "executive", "regulator", "analyst"]

# Persona valence tables mapping predicate -> valence weight
PERSONA_VALENCE: dict[PersonaType, dict[str, float]] = {
    "investor": {
        # Positive: Growth, revenue, market indicators
        "increases": 1.0,
        "generates": 1.0,
        "grows": 1.0,
        "expands": 1.0,
        "scales": 1.0,
        "monetizes": 1.0,
        "acquires": 1.0,
        # Negative: Risk, loss, decline indicators
        "decreases": -1.0,
        "loses": -1.0,
        "fails": -1.0,
        "declines": -1.0,
        "risks": -1.0,
        "threatens": -1.0,
    },
    "engineer": {
        # Positive: Quality, reliability, testing, modularity
        "tests": 1.0,
        "validates": 1.0,
        "refactors": 1.0,
        "optimizes": 1.0,
        "modularizes": 1.0,
        "documents": 1.0,
        "implements": 1.0,
        "encapsulates": 1.0,
        # Negative: Technical debt, coupling, brittleness
        "bypasses": -1.0,
        "couples": -1.0,
        "hardcodes": -1.0,
        "duplicates": -1.0,
        "skips": -1.0,
        "hacks": -1.0,
    },
    "executive": {
        # Positive: Strategy, leadership, vision, efficiency
        "leads": 1.0,
        "directs": 1.0,
        "aligns": 1.0,
        "streamlines": 1.0,
        "delegates": 1.0,
        "prioritizes": 1.0,
        "unifies": 1.0,
        # Negative: Fragmentation, inefficiency, conflict
        "fragments": -1.0,
        "conflicts": -1.0,
        "stalls": -1.0,
        "duplicates": -1.0,
        "blocks": -1.0,
    },
    "regulator": {
        # Positive: Compliance, security, validation, enforcement
        "validates": 1.0,
        "enforces": 1.0,
        "audits": 1.0,
        "certifies": 1.0,
        "authenticates": 1.0,
        "authorizes": 1.0,
        "encrypts": 1.0,
        "logs": 1.0,
        # Negative: Bypassing, circumventing, hiding
        "bypasses": -1.0,
        "circumvents": -1.0,
        "hides": -1.0,
        "obfuscates": -1.0,
        "skips": -1.0,
        "ignores": -1.0,
    },
    "analyst": {
        # Positive: Clarity, measurement, insight, data
        "measures": 1.0,
        "tracks": 1.0,
        "analyzes": 1.0,
        "reports": 1.0,
        "monitors": 1.0,
        "quantifies": 1.0,
        "correlates": 1.0,
        # Negative: Opacity, guessing, hiding
        "obscures": -1.0,
        "hides": -1.0,
        "guesses": -1.0,
        "assumes": -1.0,
        "approximates": -1.0,
    },
}


def get_valence(persona: PersonaType | None, predicate: str) -> float:
    """Get the valence weight for a predicate given a persona.
    
    Args:
        persona: The persona type (investor, engineer, executive, regulator, analyst)
                 or None for no persona
        predicate: The predicate anchor key
        
    Returns:
        float: Valence weight (-1.0, 0.0, or 1.0)
               Returns 0.0 if persona is None or predicate not in persona's table
    """
    if persona is None:
        return 0.0
    
    persona_table = PERSONA_VALENCE.get(persona, {})
    return persona_table.get(predicate, 0.0)
