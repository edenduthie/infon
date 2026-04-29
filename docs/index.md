# infon

**Structured, interpretable knowledge-base memory for AI agents — backed by DuckDB, tree-sitter AST extraction, and zero-training SPLADE projection.**

---

## What is infon?

`infon` gives AI agents (Claude Code, Cursor, etc.) **structured memory** backed by a local DuckDB knowledge graph. It transforms a code repository into typed, queryable triples called **infons** — each grounded to a source file and line number — so an agent can reason about code structure, remember observations, and trace every fact back to its origin.

### Key Features

- **Infon data model** — typed triples `<<predicate, subject, object; polarity>>` grounded to source file + line number, drawn from situation semantics (Barwise & Perry 1983).
- **Multi-language AST extraction** — Python and TypeScript/JavaScript via tree-sitter. Extracts calls, imports, inheritance, mutations, definitions, returns, raises, and decorators into grounded infons.
- **SPLADE-based text extraction** — Converts docstrings, comments, and natural-language observations into infons via change-of-basis projection from BERT token space to anchor concept space. Zero training, bundled model.
- **Schema auto-discovery** — `infon init` runs spectral clustering on a co-activation matrix to derive anchors from your codebase. No hand-authored schema required.
- **Consolidation** — Duplicate merging, NEXT-edge temporal chains, constraint aggregation, and importance decay.
- **Persona-aware retrieval** — Persona-specific valence scoring (`investor`, `engineer`, `executive`, `regulator`, `analyst`) shifts ranking based on who's asking.
- **MCP server** — stdio-based FastMCP server exposing `search`, `store_observation`, and `query_ast` tools, plus `infon://stats`, `infon://schema`, `infon://recent` resources. Connects to Claude Code via `.mcp.json`.
- **DuckDB storage** — Embedded, columnar, zero-configuration. Analytical queries 5-50x faster than row-based alternatives.

---

## Quick Start

```bash
# 1. Index the current repository (auto-discovers schema, extracts AST infons)
uvx infon init

# 2. Search the knowledge base
uvx infon search "what calls DatabasePool"

# 3. Start the MCP server (stdio) for Claude Code
uvx infon serve
```

That's it. `.infon/schema.json` and `.infon/kb.ddb` are created in your project root (auto-ignored by `.gitignore`). A `.mcp.json` is written so Claude Code picks up the three tools automatically.

---

## Conceptual Overview

### Infons

An **infon** is a typed, grounded triple:

```
<<predicate, subject, object; polarity>>
```

For example:

```
<<calls, "process_data", "validate_input"; +>>
```

This means "process_data calls validate_input" (positive polarity). Each infon is **grounded** to a source location:

- **AST grounding**: file path, line number, node type
- **Text grounding**: document ID, sentence ID, character span, sentence text

### Anchors

**Anchors** are the vocabulary of concepts used in infons. They come from:

1. **Code relation anchors** — built-in: `calls`, `imports`, `inherits`, `defines`, `returns`, `raises`, `decorates`, `mutates`
2. **Auto-discovered anchors** — derived from your codebase via spectral clustering on SPLADE co-activation matrix

### Change of Basis

Natural language text is projected from **BERT token space** to **anchor concept space** using SPLADE (Sparse Lexical and Expansion). This allows text (docstrings, comments, observations) to be encoded as anchor activations, which become infon predicates, subjects, and objects.

### Consolidation

Multiple infons expressing the same fact are **consolidated**:

- Duplicate merging with reinforcement count
- NEXT-edge temporal chains for event sequences
- Constraint aggregation (min/max/count/sum)
- Importance decay over time

---

## Use Cases

### 1. Code Structure Queries

Query code relationships across your repository:

```bash
infon search "what calls DatabasePool"
infon search "which functions raise ValueError"
```

### 2. Agent Memory

Store observations mid-session so the agent remembers decisions:

```python
from infon.store import InfonStore
from infon.infon import Infon
from infon.grounding import TextGrounding

store = InfonStore(".infon/kb.ddb")
observation = Infon(
    subject="UserService",
    predicate="needs_refactoring",
    object="circular_dependency",
    polarity=True,
    grounding=TextGrounding(
        doc_id="session-123",
        sent_id=0,
        char_start=0,
        char_end=50,
        sentence_text="UserService has circular dependency with AuthService"
    ),
    confidence=0.9
)
store.insert([observation])
```

### 3. Domain Vocabulary Discovery

Automatically discover the concepts that define your codebase:

```bash
infon init
cat .infon/schema.json
```

The schema lists anchors derived from your codebase, ranked by frequency and semantic coherence.

---

## Next Steps

- [Installation](installation.md) — three install paths (uvx, pip, source)
- [Concepts](concepts.md) — deep dive into infons, anchors, and consolidation
- [CLI Reference](cli.md) — all five commands with examples
- [MCP Server](mcp.md) — integrate with Claude Code
- [API Reference](api-reference.md) — Python API with type signatures

---

## Project Structure

```
.infon/
├── kb.ddb            # DuckDB knowledge base
└── schema.json       # Discovered anchor schema
.mcp.json             # Claude Code MCP config
```

---

## License

Apache 2.0
