"""
Grounding models for anchoring Infons to specific locations.

Grounding provides the ability to anchor an Infon to a specific location
in source text (TextGrounding) or code AST (ASTGrounding). All grounding
models are frozen (immutable) Pydantic models with JSON serialization support.

The Grounding hierarchy uses Pydantic's discriminated union pattern with
grounding_type as the discriminator field.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, RootModel


class TextGrounding(BaseModel):
    """
    Grounding for text-based sources (documents, sentences).
    
    TextGrounding anchors an Infon to a specific character range within
    a sentence of a document. It includes the full sentence text for context.
    
    Attributes:
        grounding_type: Discriminator field, always "text"
        doc_id: Unique identifier for the source document
        sent_id: Sentence index within the document (0-indexed)
        char_start: Starting character offset within the sentence
        char_end: Ending character offset within the sentence (exclusive)
        sentence_text: Full text of the sentence for context
    """
    model_config = {"frozen": True}
    
    grounding_type: Literal["text"] = Field(default="text", description="Discriminator field")
    doc_id: str = Field(description="Document identifier")
    sent_id: int = Field(description="Sentence index in document")
    char_start: int = Field(description="Start character offset in sentence")
    char_end: int = Field(description="End character offset in sentence (exclusive)")
    sentence_text: str = Field(description="Full sentence text")


class ASTGrounding(BaseModel):
    """
    Grounding for AST (Abstract Syntax Tree) nodes in code.
    
    ASTGrounding anchors an Infon to a specific AST node in a source code file.
    It captures the file location, line number, and node type.
    
    Attributes:
        grounding_type: Discriminator field, always "ast"
        file_path: Path to the source file
        line_number: Line number where the AST node begins
        node_type: Type of AST node (e.g., "FunctionDef", "ClassDef", "Module")
    """
    model_config = {"frozen": True}
    
    grounding_type: Literal["ast"] = Field(default="ast", description="Discriminator field")
    file_path: str = Field(description="Source file path")
    line_number: int = Field(description="Line number in source file")
    node_type: str = Field(description="AST node type")


# Define the discriminated union type
GroundingUnion = Annotated[
    TextGrounding | ASTGrounding,
    Discriminator("grounding_type")
]


class Grounding(RootModel[GroundingUnion]):
    """
    Base class for all grounding types using Pydantic's discriminated union.
    
    This RootModel allows polymorphic deserialization based on the grounding_type
    discriminator field. It automatically routes to the correct subtype
    (TextGrounding or ASTGrounding) during deserialization.
    
    Example:
        # Deserialize JSON to the correct subtype
        json_str = '{"grounding_type": "text", "doc_id": "doc1", ...}'
        grounding = Grounding.model_validate_json(json_str)
        # grounding.root will be a TextGrounding instance
    """
    root: GroundingUnion
