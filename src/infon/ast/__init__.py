"""AST extraction pipeline for code repository ingestion.

This module provides AST-based extraction of code relations from source repositories.
It uses tree-sitter for parsing and extracts the eight built-in code relation anchors:
calls, imports, inherits, mutates, defines, returns, raises, decorates.

Main entry point:
    ingest_repo(repo_path, store, schema) -> list[Infon]
"""

from infon.ast.base import BaseASTExtractor
from infon.ast.ingest import ingest_repo
from infon.ast.python_extractor import PythonASTExtractor
from infon.ast.registry import ExtractorRegistry
from infon.ast.typescript_extractor import TypeScriptASTExtractor

__all__ = [
    "BaseASTExtractor",
    "PythonASTExtractor",
    "TypeScriptASTExtractor",
    "ExtractorRegistry",
    "ingest_repo",
]
