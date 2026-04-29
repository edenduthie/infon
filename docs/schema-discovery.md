# Schema Discovery

`infon` automatically discovers an anchor schema from your codebase using spectral clustering on a SPLADE co-activation matrix. This page explains the algorithm, the output format, and how to customize or override the discovered schema.

---

## Overview

When you run `infon init` without providing a custom schema, `infon` analyzes your codebase and derives a set of **anchors** (conceptual vocabulary) that represent the key concepts in your code.

This process is called **schema discovery**, and it uses unsupervised machine learning to cluster semantically related tokens into coherent concepts.

### Why Auto-Discovery?

Hand-authoring a schema for every codebase is tedious and error-prone. Auto-discovery:

- **Adapts to your domain** — discovers concepts specific to your codebase (e.g., `authentication`, `database`, `validation`)
- **Zero manual effort** — no schema file to write or maintain
- **Extensible** — you can override or refine the discovered schema later

### What's Discovered?

The discovery process produces:

- **Entity anchors** — nouns and concepts (e.g., `user`, `database`, `service`)
- **Relation anchors** — verbs and relationships (always includes the eight built-in code relations: `calls`, `imports`, etc.)
- **Property anchors** — attributes and characteristics (e.g., `email`, `password`, `status`)
- **Constraint anchors** — rules and invariants (e.g., `required`, `unique`, `valid`)
- **Event anchors** — actions and state changes (e.g., `created`, `updated`, `deleted`)

---

## Algorithm

The schema discovery algorithm follows these steps:

### 1. Collect Corpus

Scan the current directory for source files:

- **Code mode**: `.py`, `.ts`, `.tsx`, `.js`, `.jsx`
- **Text mode**: `.txt`, `.md`

Excludes:

- `__pycache__/`, `node_modules/`
- `.venv/`, `venv/`, `.env/`
- `.git/`, `dist/`, `build/`, `out/`

### 2. Build Co-Activation Matrix

For each source file:

1. **Encode with SPLADE** — convert text to sparse token activations (30,000+ vocab tokens)
2. **Filter by threshold** — keep only tokens with activation ≥ 0.1
3. **Compute co-occurrences** — track which tokens activate together in the same code unit (line)

After processing all files, compute **NPMI (Normalized Pointwise Mutual Information)** for each token pair:

```
NPMI(t1, t2) = PMI(t1, t2) / (-log P(t1, t2))
```

Where:

```
PMI(t1, t2) = log(P(t1, t2) / (P(t1) * P(t2)))
```

NPMI ranges from -1 (negative correlation) to +1 (perfect correlation). Only positive NPMI values are retained (tokens that co-activate more than expected by chance).

### 3. Filter to Top-F Frequent Tokens

Reduce the matrix to the top-F most frequent tokens (default F=2000). This:

- Removes rare tokens (noise)
- Reduces computational cost of spectral clustering
- Focuses on domain-central concepts

### 4. Compute Normalized Graph Laplacian

Build a graph where:

- **Nodes** = tokens
- **Edges** = positive NPMI values (weighted)

Compute the **normalized graph Laplacian**:

```
L = I - D^(-1/2) * W * D^(-1/2)
```

Where:

- `W` = adjacency matrix (NPMI values)
- `D` = degree matrix (diagonal, sum of edge weights per node)
- `I` = identity matrix

### 5. Extract Bottom-K Eigenvectors

Compute the bottom-K eigenvectors of the Laplacian (eigenvalues closest to 0). These eigenvectors form an embedding space where semantically similar tokens cluster together.

Default K = 50 (configurable via `n_clusters` parameter).

### 6. Run K-Means Clustering

Project tokens into the K-dimensional eigenvector space and run k-means clustering to partition them into K clusters.

Each cluster represents a semantic concept (e.g., one cluster might contain `user`, `account`, `profile`, `identity`).

### 7. Label Clusters and Infer Types

For each cluster:

1. **Select representative tokens** — top 3-5 most frequent tokens in the cluster
2. **Label the anchor** — use the most frequent token as the canonical name
3. **Infer anchor type** — classify as `entity`, `relation`, `property`, `constraint`, or `event` based on linguistic patterns

Example cluster:

```
Cluster 12:
  Tokens: user, account, profile, identity, authentication
  Canonical label: "user"
  Type: entity
  Frequency: 0.15 (appears in 15% of code units)
```

### 8. Add Built-In Code Relations

For code mode, ensure the eight built-in code relation anchors are included:

- `calls`
- `imports`
- `inherits`
- `defines`
- `returns`
- `raises`
- `decorates`
- `mutates`

These anchors are **always present** in code mode schemas, regardless of what's discovered.

---

## Schema Output

The discovered schema is written to `.infon/schema.json`:

```json
{
  "version": "auto-1.0",
  "language": "code",
  "anchors": [
    {
      "name": "calls",
      "type": "relation",
      "frequency": 0.0
    },
    {
      "name": "imports",
      "type": "relation",
      "frequency": 0.0
    },
    {
      "name": "user",
      "type": "entity",
      "frequency": 0.15
    },
    {
      "name": "database",
      "type": "entity",
      "frequency": 0.12
    },
    {
      "name": "validation",
      "type": "entity",
      "frequency": 0.08
    },
    {
      "name": "authentication",
      "type": "entity",
      "frequency": 0.07
    },
    {
      "name": "email",
      "type": "property",
      "frequency": 0.05
    },
    {
      "name": "required",
      "type": "constraint",
      "frequency": 0.03
    }
  ]
}
```

### Schema Fields

Each anchor has:

- **name** (string): Canonical token or phrase (e.g., `"user"`, `"validation"`)
- **type** (string): Anchor type (`entity`, `relation`, `property`, `constraint`, `event`)
- **frequency** (float): Proportion of code units containing this anchor (0.0 to 1.0)
- **aliases** (list of strings, optional): Synonyms or variations (e.g., `["users", "account"]`)

---

## Manual Override

If the auto-discovered schema is noisy or missing key concepts, you can manually edit `.infon/schema.json` or provide a custom schema with `--schema`:

### Option 1: Edit .infon/schema.json

After running `infon init`, edit `.infon/schema.json`:

```json
{
  "version": "custom-1.0",
  "language": "code",
  "anchors": [
    {"name": "calls", "type": "relation"},
    {"name": "imports", "type": "relation"},
    {"name": "my_custom_concept", "type": "entity", "frequency": 0.0}
  ]
}
```

Then re-run ingestion:

```bash
infon ingest
```

### Option 2: Provide Custom Schema

Create a custom schema file `my-schema.json`:

```json
{
  "version": "1.0",
  "language": "code",
  "anchors": [
    {"name": "calls", "type": "relation"},
    {"name": "imports", "type": "relation"},
    {"name": "user", "type": "entity"},
    {"name": "database", "type": "entity"},
    {"name": "api_endpoint", "type": "entity"}
  ]
}
```

Initialize with:

```bash
infon init --schema my-schema.json
```

---

## Tuning Discovery

You can tune the discovery algorithm by modifying `src/infon/discovery.py`:

### Parameters

```python
class SchemaDiscovery:
    def __init__(
        self,
        n_clusters: int = 50,        # Number of anchor clusters
        top_tokens: int = 2000,      # Top frequent tokens to retain
        min_activation: float = 0.1  # SPLADE activation threshold
    ):
```

**n_clusters** — Controls how many anchor concepts are discovered:

- **Smaller** (e.g., 20): Produces coarse-grained, broad concepts
- **Larger** (e.g., 100): Produces fine-grained, specific concepts
- **Default** (50): Balanced for most codebases

**top_tokens** — Controls vocabulary size:

- **Smaller** (e.g., 500): Faster, but may miss rare concepts
- **Larger** (e.g., 5000): Slower, captures more concepts
- **Default** (2000): Balanced for most codebases

**min_activation** — Controls SPLADE filtering:

- **Higher** (e.g., 0.3): Only strong activations (reduces noise)
- **Lower** (e.g., 0.05): More tokens (increases recall)
- **Default** (0.1): Balanced signal-to-noise ratio

### Example: High-Resolution Schema

To discover more fine-grained concepts:

```python
from infon.discovery import SchemaDiscovery

discovery = SchemaDiscovery(
    n_clusters=100,     # More clusters
    top_tokens=5000,    # Larger vocabulary
    min_activation=0.05 # Lower threshold
)

schema = discovery.discover(corpus_path=".", mode="code")
```

---

## Warnings

### Small Corpus Warning

If the corpus has fewer than 50 files, you'll see:

```
Warning: Corpus has fewer than 50 files (12 found); auto-discovered schema may be noisy.
Consider providing a manual schema with --schema.
```

Small corpora produce less reliable clustering. For codebases with < 50 files, consider:

- Using a manual schema
- Combining with a related codebase for discovery
- Accepting the noisy schema and refining it manually

### Empty Corpus

If no source files are found:

```
Error: No source files found in current directory.
```

Ensure you're running `infon init` from the repository root.

---

## Algorithm Details

### Why Spectral Clustering?

Spectral clustering is ideal for discovering semantic clusters because:

- **Graph-based** — captures token co-activation patterns naturally
- **Non-convex** — handles overlapping or irregularly shaped clusters
- **Dimensionality reduction** — eigenvectors embed tokens in low-dimensional space where similar tokens are near each other

Alternative clustering methods (k-means on raw NPMI, hierarchical clustering) tend to produce less coherent clusters.

### Why NPMI?

NPMI (Normalized Pointwise Mutual Information) measures correlation between tokens while:

- **Normalizing by co-occurrence probability** — avoids bias toward frequent tokens
- **Bounded [-1, 1]** — interpretable scale
- **Symmetric** — treats token pairs equally

### Why SPLADE?

SPLADE (Sparse Lexical and Expansion) is a learned sparse encoder that:

- **Expands semantics** — activates related tokens (e.g., "validate" activates "check", "verify")
- **Pre-trained on general text** — requires no fine-tuning on your codebase
- **Sparse** — activates ~100-300 tokens per text unit (efficient for co-activation matrix)

---

## Example Workflow

### 1. Run Schema Discovery

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

### 2. Inspect Schema

```bash
cat .infon/schema.json
```

Output:

```json
{
  "version": "auto-1.0",
  "language": "code",
  "anchors": [
    {"name": "calls", "type": "relation", "frequency": 0.0},
    {"name": "imports", "type": "relation", "frequency": 0.0},
    {"name": "user", "type": "entity", "frequency": 0.15},
    {"name": "database", "type": "entity", "frequency": 0.12},
    {"name": "validation", "type": "entity", "frequency": 0.08}
  ]
}
```

### 3. Refine Schema (Optional)

Edit `.infon/schema.json` to add missing concepts or remove noise:

```json
{
  "version": "custom-1.0",
  "language": "code",
  "anchors": [
    {"name": "calls", "type": "relation"},
    {"name": "imports", "type": "relation"},
    {"name": "user", "type": "entity"},
    {"name": "database", "type": "entity"},
    {"name": "api_endpoint", "type": "entity"}
  ]
}
```

### 4. Re-Ingest

```bash
infon ingest
```

This re-extracts infons using the updated schema.

---

## Next Steps

- [Concepts](concepts.md) — understand anchors and change of basis
- [CLI Reference](cli.md) — `infon init` command details
- [API Reference](api-reference.md) — use schema discovery programmatically
