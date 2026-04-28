"""
Integration test for repository structure (Task 1.1).

This test verifies that all required directories and files exist with correct content.
Following TDD: this test should FAIL first (red), then pass after structure creation (green).
"""

import pathlib


def test_repository_structure_exists():
    """Verify all required directories and files exist."""
    repo_root = pathlib.Path(__file__).parent.parent
    
    # Required directories
    required_dirs = [
        repo_root / "src" / "infon",
        repo_root / "tests",
        repo_root / "tests" / "fixtures",
        repo_root / "docs",
        repo_root / ".github" / "workflows",
    ]
    
    for directory in required_dirs:
        assert directory.exists(), f"Directory {directory} does not exist"
        assert directory.is_dir(), f"{directory} is not a directory"


def test_required_files_exist():
    """Verify all required files exist."""
    repo_root = pathlib.Path(__file__).parent.parent
    
    required_files = [
        repo_root / "CHANGELOG.md",
        repo_root / "CONTRIBUTING.md",
        repo_root / "LICENSE",
        repo_root / "README.md",
        repo_root / ".gitignore",
    ]
    
    for file_path in required_files:
        assert file_path.exists(), f"File {file_path} does not exist"
        assert file_path.is_file(), f"{file_path} is not a file"


def test_license_is_apache_2():
    """Verify LICENSE file contains Apache 2.0 license."""
    repo_root = pathlib.Path(__file__).parent.parent
    license_file = repo_root / "LICENSE"
    
    assert license_file.exists(), "LICENSE file does not exist"
    
    content = license_file.read_text(encoding="utf-8")
    assert "Apache License" in content, "LICENSE does not contain 'Apache License'"
    assert "Version 2.0" in content, "LICENSE does not contain 'Version 2.0'"


def test_gitignore_includes_required_patterns():
    """Verify .gitignore includes all required patterns."""
    repo_root = pathlib.Path(__file__).parent.parent
    gitignore_file = repo_root / ".gitignore"
    
    assert gitignore_file.exists(), ".gitignore file does not exist"
    
    content = gitignore_file.read_text(encoding="utf-8")
    
    required_patterns = [
        ".infon/",
        "*.ddb",
        "dist/",
        ".venv/",
    ]
    
    for pattern in required_patterns:
        assert pattern in content, f".gitignore does not contain pattern: {pattern}"
