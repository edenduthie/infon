"""
Integration test for pyproject.toml (Task 1.2).

This test verifies that pyproject.toml exists and contains all required configuration:
- Package name: infon
- License: Apache-2.0
- Python version: >=3.11
- Entry point: infon = 'infon.cli:cli'
- All runtime dependencies with correct versions

Following TDD: this test should FAIL first (red), then pass after file creation (green).
"""

import pathlib
import tomllib  # Python 3.11+ standard library


def test_pyproject_toml_exists():
    """Verify pyproject.toml file exists."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    assert pyproject_file.exists(), "pyproject.toml file does not exist"
    assert pyproject_file.is_file(), "pyproject.toml is not a file"


def test_pyproject_toml_is_valid_toml():
    """Verify pyproject.toml is valid TOML format."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)
    
    assert isinstance(data, dict), "pyproject.toml did not parse to a dictionary"


def test_pyproject_package_metadata():
    """Verify package metadata is correct."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)
    
    # Verify [project] section exists
    assert "project" in data, "pyproject.toml missing [project] section"
    project = data["project"]
    
    # Verify package name
    assert "name" in project, "pyproject.toml missing project.name"
    assert project["name"] == "infon", f"Expected package name 'infon', got '{project['name']}'"
    
    # Verify license
    assert "license" in project, "pyproject.toml missing project.license"
    assert "text" in project["license"], "pyproject.toml missing project.license.text"
    assert project["license"]["text"] == "Apache-2.0", \
        f"Expected license 'Apache-2.0', got '{project['license']['text']}'"
    
    # Verify Python version requirement
    assert "requires-python" in project, "pyproject.toml missing project.requires-python"
    assert project["requires-python"] == ">=3.11", \
        f"Expected requires-python '>=3.11', got '{project['requires-python']}'"


def test_pyproject_entry_points():
    """Verify entry points are correctly defined."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)
    
    project = data["project"]
    
    # Verify scripts section exists
    assert "scripts" in project, "pyproject.toml missing project.scripts"
    scripts = project["scripts"]
    
    # Verify infon entry point
    assert "infon" in scripts, "pyproject.toml missing 'infon' entry point"
    assert scripts["infon"] == "infon.cli:cli", \
        f"Expected entry point 'infon.cli:cli', got '{scripts['infon']}'"


def test_pyproject_runtime_dependencies():
    """Verify all runtime dependencies are present with correct version constraints."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)
    
    project = data["project"]
    
    # Verify dependencies section exists
    assert "dependencies" in project, "pyproject.toml missing project.dependencies"
    dependencies = project["dependencies"]
    
    # Expected runtime dependencies with version constraints
    expected_deps = {
        "duckdb": ">=0.10",
        "pydantic": ">=2.0",
        "click": ">=8.0",
        "fastmcp": ">=0.4",
        "tree-sitter": ">=0.21",
        "tree-sitter-python": ">=0.21",
        "tree-sitter-javascript": ">=0.21",
        "transformers": ">=4.35",
        "torch": ">=2.0",
        "numpy": ">=1.24",
        "scipy": ">=1.11",
    }
    
    # Build a dictionary of actual dependencies from list
    actual_deps = {}
    for dep in dependencies:
        if ">=" in dep:
            name, version = dep.split(">=")
            actual_deps[name.strip()] = f">={version.strip()}"
        else:
            # Handle other formats if needed
            actual_deps[dep.strip()] = dep.strip()
    
    # Verify each expected dependency is present with correct version
    for dep_name, expected_version in expected_deps.items():
        assert dep_name in actual_deps, \
            f"Missing runtime dependency: {dep_name}"
        assert actual_deps[dep_name] == expected_version, \
            f"Dependency {dep_name} has version '{actual_deps[dep_name]}', expected '{expected_version}'"


def test_pyproject_build_system():
    """Verify build system is properly configured."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)
    
    # Verify [build-system] section exists
    assert "build-system" in data, "pyproject.toml missing [build-system] section"
    build_system = data["build-system"]
    
    # Verify requires
    assert "requires" in build_system, "pyproject.toml missing build-system.requires"
    assert isinstance(build_system["requires"], list), "build-system.requires must be a list"
    
    # Verify build-backend
    assert "build-backend" in build_system, "pyproject.toml missing build-system.build-backend"


def test_pyproject_dev_dependencies():
    """Verify all dev dependencies are present with correct version constraints (Task 1.3)."""
    repo_root = pathlib.Path(__file__).parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    
    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)
    
    project = data["project"]
    
    # Verify optional-dependencies section exists
    assert "optional-dependencies" in project, \
        "pyproject.toml missing project.optional-dependencies"
    optional_deps = project["optional-dependencies"]
    
    # Verify dev dependencies section exists
    assert "dev" in optional_deps, \
        "pyproject.toml missing project.optional-dependencies.dev"
    dev_deps = optional_deps["dev"]
    
    # Expected dev dependencies with version constraints
    expected_dev_deps = {
        "pytest": ">=8.0",
        "pytest-asyncio": None,  # No specific version required beyond what's there
        "ruff": None,  # No specific version required beyond what's there
        "mypy": None,  # No specific version required beyond what's there
        "mkdocs-material": ">=9.0",
        "python-build": None,  # No version constraint specified
    }
    
    # Build a dictionary of actual dev dependencies from list
    actual_dev_deps = {}
    for dep in dev_deps:
        if ">=" in dep:
            name, version = dep.split(">=")
            actual_dev_deps[name.strip()] = f">={version.strip()}"
        else:
            # Handle packages without version constraints
            actual_dev_deps[dep.strip()] = None
    
    # Verify each expected dependency is present
    for dep_name, expected_version in expected_dev_deps.items():
        assert dep_name in actual_dev_deps, \
            f"Missing dev dependency: {dep_name}"
        
        # If a specific version is expected, verify it matches
        if expected_version is not None:
            assert actual_dev_deps[dep_name] == expected_version, \
                f"Dev dependency {dep_name} has version '{actual_dev_deps[dep_name]}', expected '{expected_version}'"
