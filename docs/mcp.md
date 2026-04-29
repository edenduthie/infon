# MCP Server

The `infon` MCP (Model Context Protocol) server integrates with Claude Desktop and other MCP-compatible clients, providing structured memory and code graph capabilities to AI agents.

---

## What is MCP?

MCP (Model Context Protocol) is a protocol developed by Anthropic that allows AI assistants to interact with external tools and data sources. The `infon` MCP server exposes three tools and three resources that Claude can use to search, store, and query knowledge.

---

## Quick Start

### 1. Initialize infon

Run `infon init` in your project directory:

```bash
cd my-project
infon init
```

This creates:

- `.infon/kb.ddb` — DuckDB knowledge base
- `.infon/schema.json` — Anchor schema
- `.mcp.json` — MCP server configuration

### 2. Restart Claude Desktop

After `infon init` writes `.mcp.json`, restart Claude Desktop. It will automatically detect and connect to the infon MCP server.

### 3. Use the Tools

Ask Claude to search your codebase:

```
Claude, search my codebase for functions that call process_data
```

Claude will use the `search` or `query_ast` tool under the hood and return grounded results with file paths and line numbers.

---

## MCP Configuration

The `infon init` command writes the following to `.mcp.json` in your project root:

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

This tells Claude Desktop to launch the infon MCP server using `uvx infon serve` (zero-install mode).

### Alternative: Use pip-installed infon

If you installed `infon` with `pip` instead of using `uvx`, update `.mcp.json`:

```json
{
  "mcpServers": {
    "infon": {
      "command": "infon",
      "args": ["serve"]
    }
  }
}
```

### Custom Database Path

If you initialized infon with a custom database path, pass it as an environment variable:

```json
{
  "mcpServers": {
    "infon": {
      "command": "uvx",
      "args": ["infon", "serve"],
      "env": {
        "INFON_DB_PATH": "/path/to/custom.ddb"
      }
    }
  }
}
```

---

## MCP Tools

The infon MCP server exposes three tools that Claude can invoke.

### 1. search

Search the knowledge base with natural language queries.

**Parameters:**

- `query` (string, required): Natural language query
- `limit` (integer, optional): Maximum results (default: 10)

**Returns:**

List of infons with:

- `subject`, `predicate`, `object`
- `polarity`: `+` or `-`
- `confidence`: 0.0 to 1.0
- `grounding`: File path + line number (AST) or doc ID + char span (text)
- `score`: Relevance score (for search results)

**Example Usage:**

```
Claude, search for functions that validate email addresses
```

Claude invokes:

```json
{
  "query": "validate email addresses",
  "limit": 10
}
```

Response:

```json
[
  {
    "subject": "UserService",
    "predicate": "calls",
    "object": "validate_email",
    "polarity": "+",
    "confidence": 1.0,
    "score": 0.87,
    "grounding": {
      "type": "ast",
      "file_path": "src/services/user.py",
      "line_number": 42,
      "node_type": "call_expression"
    }
  }
]
```

---

### 2. store_observation

Store an agent observation as infons.

**Parameters:**

- `text` (string, required): Natural language observation
- `source` (string, optional): Source identifier (default: "agent")

**Returns:**

Summary with:

- `infons_added`: Count of new infons
- `infons_reinforced`: Count of reinforced existing infons

**Example Usage:**

```
Claude, remember that UserService has a circular dependency with AuthService
```

Claude invokes:

```json
{
  "text": "UserService has circular dependency with AuthService",
  "source": "session-123"
}
```

Response:

```json
{
  "infons_added": 1,
  "infons_reinforced": 0
}
```

The observation is:

1. Extracted as infons (e.g., `<<has_dependency, "UserService", "AuthService"; +>>`)
2. Stored in the knowledge base
3. Consolidated with existing infons (duplicates merged, reinforcement counts incremented)

---

### 3. query_ast

Query AST-derived code relationships by symbol.

**Parameters:**

- `symbol` (string, required): Symbol/anchor to query for
- `relation` (string, optional): Predicate filter (e.g., `calls`, `imports`)
- `limit` (integer, optional): Maximum results (default: 20)

**Returns:**

List of infons where `symbol` appears as subject or object, sorted by reinforcement count.

**Example Usage:**

```
Claude, show me all functions that call DatabasePool
```

Claude invokes:

```json
{
  "symbol": "DatabasePool",
  "relation": "calls",
  "limit": 20
}
```

Response:

```json
[
  {
    "subject": "process_data",
    "predicate": "calls",
    "object": "DatabasePool",
    "polarity": "+",
    "confidence": 1.0,
    "grounding": {
      "type": "ast",
      "file_path": "src/services/data.py",
      "line_number": 15,
      "node_type": "call_expression"
    }
  },
  {
    "subject": "update_user",
    "predicate": "calls",
    "object": "DatabasePool",
    "polarity": "+",
    "confidence": 1.0,
    "grounding": {
      "type": "ast",
      "file_path": "src/services/user.py",
      "line_number": 87,
      "node_type": "call_expression"
    }
  }
]
```

---

## MCP Resources

The infon MCP server exposes three resources that Claude can read.

### 1. infon://stats

Returns knowledge base statistics (same as `infon stats` command).

**Example Usage:**

```
Claude, show me stats for the knowledge base
```

Response (Markdown):

```markdown
# Knowledge Base Statistics

- Infons: 1234
- Edges: 56
- Constraints: 12
- Documents: 3

## Top Anchors

| Anchor | Count |
|--------|-------|
| calls | 423 |
| imports | 298 |
| defines | 187 |
| inherits | 64 |
```

---

### 2. infon://schema

Returns the active anchor schema (contents of `.infon/schema.json`).

**Example Usage:**

```
Claude, what anchors are defined in the schema?
```

Response (JSON):

```json
{
  "version": "0.1.0",
  "language": "code",
  "anchors": [
    {"name": "calls", "type": "relation"},
    {"name": "imports", "type": "relation"},
    {"name": "inherits", "type": "relation"},
    {"name": "defines", "type": "relation"},
    {"name": "returns", "type": "relation"},
    {"name": "raises", "type": "relation"},
    {"name": "decorates", "type": "relation"},
    {"name": "mutates", "type": "relation"}
  ]
}
```

---

### 3. infon://recent

Returns the 20 most recently created infons.

**Example Usage:**

```
Claude, what are the most recent observations?
```

Response (Markdown):

```markdown
# Recent Infons

1. `<<calls, "process_data", "validate_input"; +>>` (src/services/data.py:15)
2. `<<imports, "user_service", "auth_service"; +>>` (src/services/user.py:3)
3. `<<defines, "UserService", "validate_email"; +>>` (src/services/user.py:42)
...
```

---

## Integration with Claude Desktop

### Step 1: Create .mcp.json

Run `infon init` in your project directory:

```bash
cd my-project
infon init
```

This writes `.mcp.json` to the project root.

### Step 2: Open Project in Claude Desktop

Open Claude Desktop and navigate to your project directory. Claude will detect `.mcp.json` and display a notification:

```
infon MCP server connected
```

### Step 3: Use the Tools

Ask Claude to use the tools:

**Search:**

```
Claude, search for functions that call process_data
```

**Store observation:**

```
Claude, remember that the UserService class needs refactoring due to circular dependencies
```

**Query AST:**

```
Claude, show me all classes that inherit from BaseService
```

Claude will invoke the appropriate tools and return grounded results with file paths and line numbers.

---

## Example Workflow

### 1. Initialize Knowledge Base

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

### 2. Restart Claude Desktop

Claude detects `.mcp.json` and connects to the infon MCP server.

### 3. Search for Code Relationships

**Prompt:**

```
Claude, what functions call validate_email?
```

**Claude's Action:**

Invokes `query_ast` tool:

```json
{
  "symbol": "validate_email",
  "relation": "calls"
}
```

**Response:**

```
I found 3 functions that call validate_email:

1. UserService.create_user (src/services/user.py:87)
2. UserService.update_email (src/services/user.py:124)
3. AuthService.verify_identity (src/services/auth.py:42)
```

### 4. Store an Observation

**Prompt:**

```
Claude, remember that UserService.create_user has a bug where it doesn't validate the domain
```

**Claude's Action:**

Invokes `store_observation` tool:

```json
{
  "text": "UserService.create_user has a bug where it doesn't validate the domain",
  "source": "session-456"
}
```

**Response:**

```
Observation stored. Added 2 infons:
- <<has_bug, "UserService.create_user", "domain_validation"; +>>
- <<needs_validation, "create_user", "domain"; +>>
```

### 5. Retrieve the Observation Later

**Prompt:**

```
Claude, search for bugs in UserService
```

**Claude's Action:**

Invokes `search` tool:

```json
{
  "query": "bugs in UserService",
  "limit": 10
}
```

**Response:**

```
I found 1 bug in UserService:

- UserService.create_user has a bug where it doesn't validate the domain
  (session-456, stored on 2026-04-29T14:23:45Z)
```

---

## Advanced Configuration

### Multiple Databases

You can run multiple infon instances with separate databases by using custom paths:

```json
{
  "mcpServers": {
    "infon-project-a": {
      "command": "uvx",
      "args": ["infon", "serve"],
      "env": {
        "INFON_DB_PATH": "/path/to/project-a/.infon/kb.ddb"
      }
    },
    "infon-project-b": {
      "command": "uvx",
      "args": ["infon", "serve"],
      "env": {
        "INFON_DB_PATH": "/path/to/project-b/.infon/kb.ddb"
      }
    }
  }
}
```

### Custom Schema

If you want to use a custom schema instead of the auto-discovered one:

```bash
infon init --schema my-schema.json
```

Then restart Claude Desktop to reload the schema.

---

## Troubleshooting

### Claude doesn't detect .mcp.json

- Ensure `.mcp.json` is in the project root (same directory as `.git/`)
- Restart Claude Desktop after creating `.mcp.json`
- Check Claude Desktop logs for errors

### MCP server fails to start

- Verify `uvx` is installed: `uvx --version`
- Try running manually: `uvx infon serve`
- Check that `.infon/kb.ddb` and `.infon/schema.json` exist

### Empty search results

- Run `infon stats` to verify infons were ingested
- Try `infon search "calls"` to test basic keyword matching
- Re-run `infon init` to re-index the repository

---

## Next Steps

- [CLI Reference](cli.md) — all five commands with examples
- [API Reference](api-reference.md) — Python API for programmatic usage
- [AST Extraction](ast-extraction.md) — how code is parsed into infons
