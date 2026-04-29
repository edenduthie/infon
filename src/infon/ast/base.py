"""Base abstract class for AST extractors.

All language-specific extractors must inherit from BaseASTExtractor and implement
the extract() method to produce Infons from source code files.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from infon.grounding import ASTGrounding, Grounding
from infon.infon import ImportanceScore, Infon
from infon.schema import AnchorSchema


class BaseASTExtractor(ABC):
    """Abstract base class for language-specific AST extractors.
    
    Each concrete extractor implements the extract() method to parse source
    code files and produce Infons representing code relations (calls, imports,
    inherits, etc.) with ASTGrounding.
    
    Attributes:
        schema: The AnchorSchema containing relation anchors
    """
    
    def __init__(self, schema: AnchorSchema):
        """Initialize the extractor with a schema.
        
        Args:
            schema: AnchorSchema containing the relation anchors
        """
        self.schema = schema
    
    @abstractmethod
    def extract(self, file_path: Path) -> list[Infon]:
        """Extract infons from a source code file.
        
        This method must be implemented by concrete extractors. It should:
        1. Parse the file using tree-sitter
        2. Walk the AST to find relevant code relations
        3. Create Infons with ASTGrounding for each relation
        
        Args:
            file_path: Path to the source code file
            
        Returns:
            List of extracted Infons with ASTGrounding
        """
        pass
    
    def _create_infon(
        self,
        subject: str,
        predicate: str,
        obj: str,
        file_path: Path,
        line_number: int,
        node_type: str,
        confidence: float = 0.9,
    ) -> Infon:
        """Helper to create an Infon with ASTGrounding.
        
        Args:
            subject: Subject anchor key
            predicate: Relation anchor key
            obj: Object anchor key
            file_path: Source file path
            line_number: Line number in source file
            node_type: AST node type (e.g., "FunctionDef", "ImportDeclaration")
            confidence: Extraction confidence (default 0.9)
            
        Returns:
            Infon with ASTGrounding
        """
        grounding = Grounding(
            root=ASTGrounding(
                grounding_type="ast",
                file_path=str(file_path),
                line_number=line_number,
                node_type=node_type,
            )
        )
        
        # Default importance scores for AST extraction
        importance = ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.8,
            novelty=0.5,
            reinforcement=0.5,
        )
        
        return Infon(
            id=str(uuid.uuid4()),
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=True,
            grounding=grounding,
            confidence=confidence,
            timestamp=datetime.now(UTC),
            importance=importance,
            kind="extracted",
            reinforcement_count=1,
        )
