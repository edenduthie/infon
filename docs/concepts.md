# Concepts

This page explains the core concepts behind `infon`: what infons are, how anchors work, how text is projected into anchor space, and how consolidation maintains the knowledge base.

---

## Infons

An **infon** is a typed, grounded triple representing a single fact or observation. It follows the structure from situation semantics (Barwise & Perry 1983):

```
<<predicate, subject, object; polarity>>
```

### Structure

- **subject**: The entity the fact is about (e.g., `"process_data"`, `"UserService"`)
- **predicate**: The relationship type (e.g., `"calls"`, `"imports"`, `"needs_refactoring"`)
- **object**: The target entity (e.g., `"validate_input"`, `"circular_dependency"`)
- **polarity**: `+` (positive) or `-` (negative), representing whether the fact holds or is negated

### Example

```python
<<calls, "process_data", "validate_input"; +>>
```

This means: "process_data calls validate_input" (positive polarity).

A negative example:

```python
<<calls, "process_data", "deprecated_function"; ->>
```

This means: "process_data does NOT call deprecated_function".

### Additional Fields

Each infon includes:

- **grounding**: Source location (file path + line number for AST, or doc ID + char span for text)
- **confidence**: Strength of evidence (0.0 to 1.0)
- **reinforcement_count**: Number of times this fact has been observed (starts at 1, incremented by consolidation)
- **created_at**: Timestamp when the infon was created
- **importance**: Decayed importance score (used for ranking and retrieval)
- **context**: Optional linked infons (e.g., NEXT-edge for temporal sequences)

---

## Grounding

Every infon is **grounded** to a source location, making it traceable and verifiable.

### AST Grounding

For infons extracted from code (via tree-sitter):

```python
ASTGrounding(
    file_path="src/services/user.py",
    line_number=42,
    node_type="call_expression"
)
```

This allows queries like "show me the code that proves `UserService` calls `validate_email`" to return the exact file and line number.

### Text Grounding

For infons extracted from natural language (docstrings, comments, observations):

```python
TextGrounding(
    doc_id="session-123",
    sent_id=0,
    char_start=0,
    char_end=50,
    sentence_text="UserService has circular dependency with AuthService"
)
```

This allows tracing observations back to the original sentence and session.

---

## Anchors

**Anchors** are the vocabulary of concepts used in infons. They serve as the semantic basis for subjects, predicates, and objects.

### Code Relation Anchors

`infon` ships with eight built-in code relation anchors:

| Anchor | Type | Description |
|--------|------|-------------|
| `calls` | relation | Function/method calls |
| `imports` | relation | Module/package imports |
| `inherits` | relation | Class inheritance |
| `defines` | relation | Function/class definitions |
| `returns` | relation | Return statements |
| `raises` | relation | Exception raising |
| `decorates` | relation | Decorator applications |
| `mutates` | relation | Attribute mutations (e.g., `self.x = y`) |

These are used for AST-extracted infons.

### Auto-Discovered Anchors

For text and domain-specific concepts, `infon init` automatically discovers anchors from your codebase using **spectral clustering** on a SPLADE co-activation matrix.

Example discovered anchors:

```json
{
  "anchors": [
    {"name": "database", "type": "entity", "frequency": 0.12},
    {"name": "authentication", "type": "entity", "frequency": 0.08},
    {"name": "validation", "type": "entity", "frequency": 0.06},
    {"name": "refactoring", "type": "entity", "frequency": 0.03}
  ]
}
```

Each anchor has:

- **name**: The canonical token or phrase
- **type**: `entity`, `relation`, `property`, `constraint`, or `event`
- **frequency**: Proportion of documents containing this concept
- **aliases**: Synonyms and variations (optional)

### Anchor Schema

The complete set of anchors is stored in `.infon/schema.json`:

```json
{
  "version": "0.1.0",
  "language": "code",
  "anchors": [
    {"name": "calls", "type": "relation"},
    {"name": "imports", "type": "relation"},
    ...
  ]
}
```

You can manually edit this file to add custom anchors or override auto-discovered ones.

---

## Change of Basis

Natural language text (docstrings, comments, observations) must be converted into **anchor space** before becoming infons. This is done via **change-of-basis projection** using SPLADE.

### How It Works

1. **SPLADE encoding**: Text is encoded into BERT token space, producing activations for 30,000+ vocabulary tokens.
2. **Projection to anchor space**: Token activations are projected to anchor activations using the anchor schema.
3. **Triple extraction**: Top-K activated anchors become candidates for subject, predicate, and object.

### Example

Input text:

```
UserService validates email addresses using regex patterns.
```

SPLADE encoding produces activations like:

```python
{
  "user": 0.82,
  "service": 0.75,
  "validates": 0.68,
  "email": 0.91,
  "regex": 0.53,
  ...
}
```

Projected to anchor space (assuming anchors `UserService`, `validates`, `email`, `regex_pattern` exist):

```python
{
  "UserService": 0.85,
  "validates": 0.70,
  "email": 0.90,
  "regex_pattern": 0.55
}
```

Extracted infon:

```python
<<validates, "UserService", "email"; +>>
```

### Zero Training

The SPLADE model is pre-trained on general text and requires **no fine-tuning** on your codebase. It provides robust semantic representations out of the box.

---

## Consolidation

Over time, the knowledge base accumulates duplicate or related infons. **Consolidation** merges and organizes these infons to maintain coherence.

### Duplicate Merging

Infons with identical `(subject, predicate, object, polarity)` are merged by:

- Incrementing `reinforcement_count`
- Averaging `confidence` (or taking max, depending on policy)
- Retaining all groundings as a list (optional)

Example:

```python
# Before
<<calls, "process_data", "validate_input"; +>> (reinforcement_count=1)
<<calls, "process_data", "validate_input"; +>> (reinforcement_count=1)

# After consolidation
<<calls, "process_data", "validate_input"; +>> (reinforcement_count=2)
```

### NEXT-Edge Temporal Chains

Infons that describe events or state changes can be linked with **NEXT edges** to form temporal sequences:

```python
<<state, "UserSession", "created"; +>>
  -> NEXT -> <<state, "UserSession", "authenticated"; +>>
  -> NEXT -> <<state, "UserSession", "expired"; +>>
```

This allows queries like "what happens after UserSession is authenticated?"

### Constraint Aggregation

Infons describing constraints (e.g., "UserService must have at least 3 replicas") can be aggregated:

```python
<<replicas, "UserService", ">=3"; +>>
<<replicas, "UserService", "<=10"; +>>

# Consolidated to range constraint
<<replicas, "UserService", "3-10"; +>>
```

### Importance Decay

Infons have an `importance` score that decays over time. Older observations become less relevant unless reinforced:

```python
importance = base_importance * exp(-lambda * age_in_days)
```

This ensures recent and frequently reinforced facts rank higher in retrieval.

---

## Retrieval

Retrieval ranks infons by relevance to a query. The process is:

1. **Encode query** to anchor space using SPLADE
2. **Match infons** containing activated anchors (in subject, predicate, or object)
3. **Score** by anchor activation strength, reinforcement count, and importance
4. **Rank** and return top-K results

### Scoring Formula

```python
score = anchor_activation * (1 + reinforcement_count / 10) * importance
```

This balances semantic relevance, evidence strength, and temporal recency.

### Persona-Aware Retrieval

Retrieval can be adjusted based on **persona** (investor, engineer, executive, regulator, analyst). Each persona has a valence vector that shifts anchor weights:

- **investor**: Prioritizes `revenue`, `growth`, `risk`
- **engineer**: Prioritizes `performance`, `reliability`, `maintainability`
- **executive**: Prioritizes `strategy`, `alignment`, `outcomes`
- **regulator**: Prioritizes `compliance`, `security`, `audit`
- **analyst**: Prioritizes `metrics`, `trends`, `anomalies`

This allows the same query to return different results depending on who's asking.

---

## Example Workflow

### 1. Index a Repository

```bash
infon init
```

This:

1. Auto-discovers anchors from the codebase
2. Extracts AST infons from Python/TypeScript files
3. Stores infons in `.infon/kb.ddb`
4. Writes schema to `.infon/schema.json`

### 2. Query the Knowledge Base

```bash
infon search "what calls validate_input"
```

Returns:

```
<<calls, "process_data", "validate_input"; +>>
  grounding: src/services/user.py:42
  confidence: 1.0
  reinforcement_count: 1
```

### 3. Store an Observation

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
        doc_id="session-456",
        sent_id=0,
        char_start=0,
        char_end=60,
        sentence_text="UserService has circular dependency with AuthService"
    ),
    confidence=0.9
)
store.insert([observation])
```

### 4. Consolidate

```bash
infon consolidate
```

This merges duplicates, builds NEXT chains, and applies importance decay.

---

## Next Steps

- [CLI Reference](cli.md) — explore all five commands
- [MCP Server](mcp.md) — integrate with Claude Code
- [AST Extraction](ast-extraction.md) — how code is parsed into infons
- [Schema Discovery](schema-discovery.md) — how anchors are auto-discovered
