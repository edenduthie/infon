# CLI Reference

The `infon` command-line interface provides five commands for managing your knowledge base: `init`, `search`, `stats`, `ingest`, and `serve`.

---

## infon init

Initialize a new infon knowledge base in the current directory.

### Usage

```bash
infon init [OPTIONS]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--schema PATH` | Path | Path to custom schema JSON file (default: auto-discover) |
| `--db PATH` | Path | Database path (default: `.infon/kb.ddb`) |

### Behavior

The `init` command:

1. Creates `.infon/` directory in the current directory
2. Auto-discovers anchor schema from the codebase (or loads from `--schema` if provided)
3. Writes schema to `.infon/schema.json`
4. Creates DuckDB database at `.infon/kb.ddb`
5. Ingests the current repository (extracts AST infons from Python/TypeScript files)
6. Writes `.mcp.json` configuration for Claude Code integration
7. Updates `.gitignore` to exclude `.infon/` directory

### Examples

**Basic initialization (auto-discover schema):**

```bash
cd my-project
infon init
```

Output:

```
Initializing infon knowledge base...
Creating default code schema...
Schema written to .infon/schema.json
Creating database at .infon/kb.ddb
Ingesting repository...
Extracted 1234 infons

Knowledge base initialized!
  Infons: 1234
  Edges: 0
  Documents: 0
```

**Initialize with custom schema:**

```bash
infon init --schema my-schema.json
```

**Initialize with custom database path:**

```bash
infon init --db /path/to/custom.ddb
```

### Created Files

```
.infon/
├── kb.ddb            # DuckDB knowledge base
└── schema.json       # Anchor schema
.mcp.json             # Claude Code MCP config (added to project root)
```

### .gitignore Update

The `init` command adds the following to `.gitignore` (if not already present):

```
.infon/
```

This prevents the knowledge base from being committed to version control.

---

## infon search

Search the knowledge base for infons matching a query.

### Usage

```bash
infon search QUERY [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `QUERY` | Natural language query or keyword |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--db PATH` | Path | `.infon/kb.ddb` | Database path |
| `--limit N` | Integer | 10 | Maximum number of results |
| `--persona NAME` | String | None | Persona for retrieval context |

### Behavior

The `search` command:

1. Encodes the query to anchor space using SPLADE
2. Matches infons containing activated anchors
3. Scores by relevance, reinforcement, and importance
4. Returns top-K results ranked by score

### Examples

**Basic search:**

```bash
infon search "what calls DatabasePool"
```

Output:

```
Found 3 results:

Subject                        Predicate            Object
--------------------------------------------------------------------------------
process_data                   calls                DatabasePool
update_user                    calls                DatabasePool
migrate_schema                 calls                DatabasePool
```

**Limit results:**

```bash
infon search "validation" --limit 5
```

**Search with persona context:**

```bash
infon search "performance issues" --persona engineer
```

Persona options: `investor`, `engineer`, `executive`, `regulator`, `analyst`

**Search with custom database:**

```bash
infon search "imports numpy" --db /path/to/custom.ddb
```

### Output Format

Results are displayed as a table:

```
Subject                        Predicate            Object
--------------------------------------------------------------------------------
module_name                    imports              numpy
process_array                  calls                numpy.array
calculate_stats                calls                numpy.mean
```

Each row represents an infon triple. Use `infon query` (future feature) to see full grounding details (file path, line number, confidence, etc.).

---

## infon stats

Display knowledge base statistics.

### Usage

```bash
infon stats [OPTIONS]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--db PATH` | Path | Database path (default: `.infon/kb.ddb`) |

### Behavior

The `stats` command displays:

- Total count of infons
- Total count of edges (NEXT-edge temporal chains)
- Total count of constraints
- Total count of documents
- Top 10 most frequent anchors

### Examples

**Basic stats:**

```bash
infon stats
```

Output:

```
Knowledge Base Statistics
==================================================
Infons:       1234
Edges:        56
Constraints:  12
Documents:    3

Top Anchors:
  calls                                      423
  imports                                    298
  defines                                    187
  inherits                                    64
  returns                                     52
  raises                                      38
  decorates                                   27
  mutates                                     15
  validation                                  12
  database                                    10
```

**Custom database:**

```bash
infon stats --db /path/to/custom.ddb
```

---

## infon ingest

Ingest repository files into the knowledge base.

### Usage

```bash
infon ingest [OPTIONS]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--incremental` | Flag | Only process files changed since last ingest (requires git) |
| `--db PATH` | Path | Database path (default: `.infon/kb.ddb`) |

### Behavior

The `ingest` command:

1. Loads schema from `.infon/schema.json`
2. Scans the current directory for Python and TypeScript/JavaScript files
3. Extracts AST infons from each file
4. Inserts infons into the database
5. Displays extraction statistics

With `--incremental`, only processes files changed since the last git commit.

### Examples

**Full ingest:**

```bash
infon ingest
```

Output:

```
Ingesting repository...
Extracted 1234 infons

Total infons: 1234
```

**Incremental ingest (git-tracked changes only):**

```bash
infon ingest --incremental
```

Output:

```
Processing 5 changed files...
Extracted 42 infons

Total infons: 1276
```

**Custom database:**

```bash
infon ingest --db /path/to/custom.ddb
```

### Supported Languages

- Python (`.py`)
- TypeScript (`.ts`, `.tsx`)
- JavaScript (`.js`, `.jsx`)

See [AST Extraction](ast-extraction.md) for details on extracted relations.

---

## infon serve

Start the MCP server for Claude Code integration.

### Usage

```bash
infon serve
```

### Behavior

The `serve` command:

1. Starts a FastMCP server on stdio (standard input/output)
2. Loads the knowledge base from `.infon/kb.ddb`
3. Loads the schema from `.infon/schema.json`
4. Exposes three MCP tools: `search`, `store_observation`, `query_ast`
5. Exposes three MCP resources: `infon://stats`, `infon://schema`, `infon://recent`

This command is intended to be run by Claude Desktop via the `.mcp.json` configuration file (written by `infon init`).

### .mcp.json Configuration

The `infon init` command writes the following to `.mcp.json`:

```json
{
  "mcpServers": {
    "infon": {
      "command": "uvx",
      "args": ["infon", "serve"]
    }
  }
}
```

Claude Desktop reads this file and launches the MCP server automatically.

### MCP Tools

The server exposes three tools:

#### 1. search

Search the knowledge base with natural language queries.

**Parameters:**

- `query` (string): Natural language query
- `limit` (integer, optional): Maximum results (default: 10)

**Returns:**

List of infons with scores and grounding details.

**Example:**

```json
{
  "query": "what calls process_data",
  "limit": 5
}
```

#### 2. store_observation

Store an agent observation as an infon.

**Parameters:**

- `text` (string): Natural language observation
- `source` (string, optional): Source identifier (e.g., session ID)

**Returns:**

Confirmation with inserted infon count.

**Example:**

```json
{
  "text": "UserService has circular dependency with AuthService",
  "source": "session-123"
}
```

#### 3. query_ast

Query AST-derived code relationships.

**Parameters:**

- `symbol` (string, optional): Symbol name (subject or object)
- `relation` (string, optional): Relation type (e.g., `calls`, `imports`)
- `limit` (integer, optional): Maximum results (default: 10)

**Returns:**

List of AST infons matching the query.

**Example:**

```json
{
  "symbol": "DatabasePool",
  "relation": "calls",
  "limit": 10
}
```

### MCP Resources

The server exposes three resources:

#### 1. infon://stats

Returns knowledge base statistics (same as `infon stats` command).

#### 2. infon://schema

Returns the active anchor schema (contents of `.infon/schema.json`).

#### 3. infon://recent

Returns the 20 most recently created infons.

### Examples

**Start the server manually:**

```bash
infon serve
```

This starts the server on stdio. It will block until terminated (Ctrl+C).

**Use with Claude Desktop:**

1. Run `infon init` to create `.mcp.json`
2. Restart Claude Desktop
3. Claude will automatically connect to the infon MCP server
4. Use the tools via prompts:

```
Claude, search my codebase for functions that call process_data
```

Claude will use the `search` or `query_ast` tool under the hood.

---

## Global Options

All commands support:

- `--help`: Display help for the command
- `--version`: Display infon version

### Examples

```bash
infon --help
infon --version
infon init --help
```

---

## Next Steps

- [MCP Server](mcp.md) — detailed MCP integration guide
- [API Reference](api-reference.md) — Python API for programmatic usage
- [Contributing](contributing.md) — add new CLI commands or tools
