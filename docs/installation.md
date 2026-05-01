# Installation

`infon` supports three installation paths to fit different workflows: zero-install with `uvx`, system-wide or venv installation with `pip`, or installation from source for development.

---

## Option 1: uvx (Zero-Install, Recommended)

**Best for**: Quick start, one-off usage, Claude Code integration

The `uvx` tool (part of the `uv` Python package installer) runs `infon` without requiring installation. It creates a temporary virtual environment, installs dependencies, and executes the command.

### Prerequisites

Install `uv` if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via Homebrew:

```bash
brew install uv
```

### Usage

Run any `infon` command directly with `uvx`:

```bash
# Initialize knowledge base
uvx infon init

# Search
uvx infon search "what calls process_data"

# Start MCP server
uvx infon serve
```

`uvx` automatically fetches the latest version from PyPI on first use and caches it for subsequent runs.

**Pros**:

- No installation or virtual environment management
- Always uses the latest version
- Perfect for Claude Code MCP integration (`.mcp.json` uses `uvx infon serve`)

**Cons**:

- First run is slower (downloads dependencies)
- Requires internet connection on first use

---

## Option 2: pip (System or Virtual Environment)

**Best for**: Teams with existing Python tooling, CI/CD pipelines, offline environments

### System-Wide Installation

Install `infon` globally (requires Python 3.11+):

```bash
pip install infon
```

Verify installation:

```bash
infon --version
```

### Virtual Environment Installation

Recommended for project isolation:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install infon
pip install infon
```

### Usage

After installation, use `infon` commands directly:

```bash
infon init
infon search "what calls DatabasePool"
infon stats
```

**Pros**:

- Fast execution (no download on each run)
- Works offline after installation
- Integrates with existing Python workflows

**Cons**:

- Requires manual environment management
- Must update manually to get new versions

### How `infon init` writes `.mcp.json`

`infon init` detects the interpreter it is running under and writes a
`.mcp.json` that matches your install path, so Claude Code launches the
MCP server the same way you launched the init command.

- **Outside a virtualenv** (system Python or `uvx`): the generated
  `.mcp.json` uses the `uvx` form. Claude Code will fetch the latest
  published `infon` on demand:

    ```json
    {
      "mcpServers": {
        "infon": {
          "type": "stdio",
          "command": "uvx",
          "args": ["--from", "infon", "infon", "serve"]
        }
      }
    }
    ```

- **Inside a virtualenv** (detected via `sys.prefix != sys.base_prefix`
  or the `VIRTUAL_ENV` environment variable): the generated `.mcp.json`
  pins to the absolute path of the venv's `infon` binary. The MCP server
  is then locked to whatever version is installed in that venv and is
  not silently upgraded by `uvx`:

    ```json
    {
      "mcpServers": {
        "infon": {
          "type": "stdio",
          "command": "/path/to/.venv/bin/infon",
          "args": ["serve"]
        }
      }
    }
    ```

If the detected venv does not actually contain an `infon` binary,
`infon init` falls back to the `uvx` form and prints a warning rather
than writing a config that cannot launch.

---

## Option 3: Install from Source

**Best for**: Development, contributing, customization

### Clone and Install

```bash
# Clone repository
git clone https://github.com/edenduthie/infon.git
cd infon

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
pytest -v

# Run linters
ruff check src/
mypy src/infon/

# Run infon
infon --version
```

**Pros**:

- Full control over source code
- Immediate reflection of code changes (editable install)
- Includes development tools (pytest, ruff, mypy, mkdocs)

**Cons**:

- Requires Git and Python development experience
- More setup overhead

---

## Dependencies

`infon` requires Python 3.11 or higher and installs the following dependencies:

### Core Dependencies

- `duckdb>=0.10` — Embedded analytical database
- `pydantic>=2.0` — Data validation and schema modeling
- `click>=8.0` — CLI framework
- `fastmcp>=0.4` — MCP server implementation
- `tree-sitter>=0.21` — AST parsing
- `tree-sitter-python>=0.21` — Python grammar
- `tree-sitter-javascript>=0.21` — TypeScript/JavaScript grammar
- `transformers>=4.35` — SPLADE model loading
- `torch>=2.0` — PyTorch for SPLADE encoder
- `numpy>=1.24` — Numerical computation
- `scipy>=1.11` — Spectral clustering

### Development Dependencies

Installed with `pip install infon[dev]`:

- `pytest>=8.0` — Testing framework
- `pytest-asyncio>=0.21` — Async test support
- `pytest-cov>=4.0` — Test coverage
- `ruff>=0.1` — Fast linter and formatter
- `mypy>=1.5` — Static type checker
- `mkdocs-material>=9.0` — Documentation generator
- `python-build` — Build tool

---

## Platform Support

`infon` is tested on:

- **macOS** (Intel and Apple Silicon)
- **Linux** (x86_64)
- **Windows** (x86_64)

PyTorch (for SPLADE encoding) may require additional setup on some platforms. See [PyTorch installation guide](https://pytorch.org/get-started/locally/) if you encounter issues.

---

## Verifying Installation

After installation, verify everything works:

```bash
# Check version
infon --version

# Initialize a test knowledge base
mkdir test-infon
cd test-infon
infon init

# Verify files were created
ls -la .infon/
# Should show kb.ddb and schema.json

# Search (empty results expected for empty KB)
infon search "test"

# Check stats
infon stats
```

---

## Next Steps

- [Concepts](concepts.md) — understand infons, anchors, and consolidation
- [CLI Reference](cli.md) — explore all five commands
- [MCP Server](mcp.md) — integrate with Claude Code
