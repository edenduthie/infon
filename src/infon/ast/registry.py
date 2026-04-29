"""Extractor registry for mapping file extensions to AST extractors.

The ExtractorRegistry maintains a mapping from file extensions to extractor classes
and provides methods to get the appropriate extractor for a given file.
"""

from pathlib import Path
from typing import Type

from infon.ast.base import BaseASTExtractor
from infon.ast.python_extractor import PythonASTExtractor
from infon.ast.typescript_extractor import TypeScriptASTExtractor
from infon.schema import AnchorSchema


class ExtractorRegistry:
    """Registry for mapping file extensions to AST extractors.
    
    The registry maintains a mapping from file extensions to extractor classes.
    It can be used to get the appropriate extractor instance for a given file.
    
    Default mappings:
        .py -> PythonASTExtractor
        .ts, .js, .tsx, .jsx -> TypeScriptASTExtractor
    """
    
    def __init__(self, schema: AnchorSchema):
        """Initialize the registry with a schema.
        
        Args:
            schema: AnchorSchema to pass to extractors
        """
        self.schema = schema
        self._extractors: dict[str, Type[BaseASTExtractor]] = {}
        self._instances: dict[str, BaseASTExtractor] = {}
        
        # Register default extractors
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default extractor mappings."""
        # Python
        self._extractors[".py"] = PythonASTExtractor
        
        # TypeScript and JavaScript
        self._extractors[".ts"] = TypeScriptASTExtractor
        self._extractors[".js"] = TypeScriptASTExtractor
        self._extractors[".tsx"] = TypeScriptASTExtractor
        self._extractors[".jsx"] = TypeScriptASTExtractor
    
    def register(self, extension: str, extractor_class: Type[BaseASTExtractor]) -> None:
        """Register a new extractor for a file extension.
        
        Args:
            extension: File extension (e.g., ".py", ".rs")
            extractor_class: Extractor class to use for this extension
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        self._extractors[extension] = extractor_class
        # Clear cached instance if any
        if extension in self._instances:
            del self._instances[extension]
    
    def get_extractor(self, file_path: Path) -> BaseASTExtractor | None:
        """Get the appropriate extractor for a file.
        
        Returns a cached instance if available, otherwise creates a new one.
        Returns None if no extractor is registered for the file extension.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Extractor instance or None if no extractor is registered
        """
        extension = file_path.suffix.lower()
        
        if extension not in self._extractors:
            return None
        
        # Return cached instance if available
        if extension in self._instances:
            return self._instances[extension]
        
        # Create new instance and cache it
        extractor_class = self._extractors[extension]
        extractor = extractor_class(self.schema)
        self._instances[extension] = extractor
        
        return extractor
    
    def has_extractor(self, file_path: Path) -> bool:
        """Check if an extractor is registered for a file.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            True if an extractor is registered for this file extension
        """
        extension = file_path.suffix.lower()
        return extension in self._extractors
    
    def supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.
        
        Returns:
            List of file extensions (e.g., [".py", ".ts", ".js"])
        """
        return list(self._extractors.keys())
