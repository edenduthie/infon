"""
Core Infon data models.

This module implements the core data structures for the infon knowledge base:
- ImportanceScore: Multi-dimensional importance scoring with composite calculation
- Infon: The atomic unit of information as a typed triple with grounding

All models are frozen (immutable) Pydantic models with JSON serialization support.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from infon.grounding import Grounding


class ImportanceScore(BaseModel):
    """
    Multi-dimensional importance score for an Infon.
    
    Importance is calculated across five dimensions:
    - activation: Strength of anchor activation in the source text/code
    - coherence: Semantic coherence with surrounding context
    - specificity: How specific (vs. generic) the information is
    - novelty: How new this information is relative to existing knowledge
    - reinforcement: Strength from repeated observations
    
    All components are floats in the range [0, 1].
    
    The composite property provides a weighted average of all components,
    which is used for ranking and retrieval.
    
    Attributes:
        activation: Activation strength [0, 1]
        coherence: Contextual coherence [0, 1]
        specificity: Information specificity [0, 1]
        novelty: Novelty score [0, 1]
        reinforcement: Reinforcement strength [0, 1]
    """
    model_config = {"frozen": True}
    
    activation: float = Field(ge=0.0, le=1.0, description="Activation strength")
    coherence: float = Field(ge=0.0, le=1.0, description="Contextual coherence")
    specificity: float = Field(ge=0.0, le=1.0, description="Information specificity")
    novelty: float = Field(ge=0.0, le=1.0, description="Novelty score")
    reinforcement: float = Field(ge=0.0, le=1.0, description="Reinforcement strength")
    
    @property
    def composite(self) -> float:
        """
        Calculate the composite importance score as a weighted average.
        
        Currently uses equal weights for all components. Future versions
        may support configurable weights.
        
        Returns:
            float: The composite score in [0, 1]
        """
        return (
            self.activation +
            self.coherence +
            self.specificity +
            self.novelty +
            self.reinforcement
        ) / 5.0


class Infon(BaseModel):
    """
    Atomic unit of information as a typed triple with grounding.
    
    An Infon represents a single fact as a triple (subject, predicate, object)
    with polarity (affirmative or negated), grounded to a specific source location
    (text or AST), with importance scoring and provenance metadata.
    
    The model is frozen (immutable). To create a modified copy, use the replace()
    helper method.
    
    Attributes:
        id: UUID identifier (string)
        subject: Subject anchor key (entity or concept)
        predicate: Relation anchor key
        object: Object anchor key (entity or concept)
        polarity: True for affirmative, False for negated
        grounding: Source location (TextGrounding or ASTGrounding)
        confidence: Extraction confidence in [0, 1]
        timestamp: UTC timestamp when infon was created
        importance: Multi-dimensional importance score
        kind: One of "extracted", "consolidated", "imagined"
        reinforcement_count: Number of times this triple has been observed
    """
    model_config = {"frozen": True}
    
    id: str = Field(description="UUID identifier")
    subject: str = Field(description="Subject anchor key")
    predicate: str = Field(description="Relation anchor key")
    object: str = Field(description="Object anchor key")
    polarity: bool = Field(description="True=affirmative, False=negated")
    grounding: Grounding = Field(description="Source location grounding")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    timestamp: datetime = Field(description="UTC timestamp")
    importance: ImportanceScore = Field(description="Multi-dimensional importance")
    kind: Literal["extracted", "consolidated", "imagined"] = Field(
        description="Infon type"
    )
    reinforcement_count: int = Field(ge=1, description="Number of observations")
    
    def replace(self, **kwargs) -> "Infon":
        """
        Create a new Infon instance with updated fields.
        
        Since Infon is frozen (immutable), this method provides a convenient way
        to create a modified copy by specifying only the fields that should change.
        
        Args:
            **kwargs: Field names and their new values
            
        Returns:
            Infon: A new Infon instance with the updated fields
            
        Example:
            >>> original = Infon(id="123", subject="a", ...)
            >>> updated = original.replace(confidence=0.95)
            >>> assert original.confidence != updated.confidence
            >>> assert original.id == updated.id
        """
        # model_dump() returns a dict of all current field values
        current_values = self.model_dump()
        
        # Update with the provided kwargs
        current_values.update(kwargs)
        
        # Create and return a new instance
        return Infon(**current_values)
