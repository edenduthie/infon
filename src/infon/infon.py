"""
Core Infon data models.

This module implements the core data structures for the infon knowledge base:
- ImportanceScore: Multi-dimensional importance scoring with composite calculation
- Infon: The atomic unit of information (to be implemented in task 1.6)

All models are frozen (immutable) Pydantic models with JSON serialization support.
"""

from pydantic import BaseModel, Field, field_validator


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
