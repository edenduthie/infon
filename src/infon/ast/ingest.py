"""Repository ingestion using AST extraction.

Main entry point for extracting infons from source code repositories.
Walks the repository directory tree, identifies supported source files,
and extracts code relations using language-specific AST extractors.
"""

from pathlib import Path

from infon.ast.registry import ExtractorRegistry
from infon.infon import Infon
from infon.schema import AnchorSchema
from infon.store import InfonStore


def ingest_repo(
    repo_path: Path | str,
    store: InfonStore,
    schema: AnchorSchema,
) -> list[Infon]:
    """Ingest a source code repository and extract infons.
    
    This function walks the repository directory tree, identifies source files
    with supported extensions, and extracts code relations using the appropriate
    AST extractor for each language.
    
    Extracted infons are stored in the provided InfonStore and also returned.
    
    Args:
        repo_path: Path to the repository root directory
        store: InfonStore to persist extracted infons
        schema: AnchorSchema containing relation anchors (must be language=code)
        
    Returns:
        List of all extracted Infons
        
    Raises:
        ValueError: If schema language is not "code"
    """
    if schema.language != "code":
        raise ValueError(f"Schema language must be 'code', got '{schema.language}'")
    
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    # Create registry with schema
    registry = ExtractorRegistry(schema)
    
    # Collect all infons
    all_infons: list[Infon] = []
    
    # Walk the repository tree
    for file_path in _walk_source_files(repo_path):
        # Get extractor for this file
        extractor = registry.get_extractor(file_path)
        
        if extractor is None:
            # Skip files without registered extractors
            continue
        
        try:
            # Extract infons from file
            infons = extractor.extract(file_path)
            
            # Store infons
            for infon in infons:
                store.upsert(infon)
            
            all_infons.extend(infons)
            
        except Exception as e:
            # Log and continue on errors
            print(f"Warning: Failed to extract from {file_path}: {e}")
            continue
    
    return all_infons


def _walk_source_files(repo_path: Path) -> list[Path]:
    """Walk repository and collect all source files.
    
    This function recursively walks the directory tree and collects all
    regular files, excluding common directories that should be ignored
    (.git, __pycache__, node_modules, etc.).
    
    Args:
        repo_path: Root directory to walk
        
    Returns:
        List of paths to source files
    """
    # Directories to skip
    skip_dirs = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "coverage",
        ".coverage",
    }
    
    source_files = []
    
    def _walk(path: Path) -> None:
        """Recursive walk helper."""
        if not path.is_dir():
            return
        
        try:
            for item in path.iterdir():
                # Skip hidden files/directories and ignored directories
                if item.name.startswith(".") or item.name in skip_dirs:
                    continue
                
                if item.is_file():
                    source_files.append(item)
                elif item.is_dir():
                    _walk(item)
        except PermissionError:
            # Skip directories we can't read
            pass
    
    _walk(repo_path)
    return source_files
