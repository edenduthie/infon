---
name: spec-writer
description: "Write openspec specs with TDD, no-mock, and review-after-each-task directives."
---

# Spec Writer

Writes openspec change specifications that enforce TDD, real integrations, and post-task review.

**Trigger:** When user asks for a new feature, plan, or openspec change.

**Input:** Feature description or problem statement.

## Openspec Structure

An openspec change consists of four files:

```
openspec/
  proposal.md      # Why the change, what changes, phased scope, impact
  design.md        # Context, decisions, trade-offs, risks
  specs/           # Detailed requirements per feature area
    <feature>-spec.md
  tasks.md         # Numbered, phased, checkmarked tasks
```

## Writing Each File

### proposal.md

```markdown
# Change: <name> — <Short description>

## Why

<Problem statement. What gap exists? Who is affected?>

## What Changes

<Bullet list of concrete changes. Each bullet is a tangible artifact or behavior.>

## Phased Scope

- **v1 (this spec):** <Core, shippable functionality>
- **v2 (follow-up):** <Nice-to-have, dependent on v1>
- **v3 (follow-up):** <Future direction>

## Impact

- <New files, modified files, dependencies added>
- <Breaking changes, if any>
- <Runtime dependency requirements>
```

### design.md

Section by section:

1. **Context** — constraints, target users, operating environment
2. **Goals / Non-Goals** — what is and isn't in scope
3. **Decisions** — each decision in its own subsection:
   - **Decision: <title>** — one sentence
   - **Why:** — rationale, alternatives considered, trade-offs
   - Multiple decisions, one per logical area
4. **Risks / Trade-offs** — what could go wrong, mitigations
5. **Open Questions** — unresolved items that need user input

### specs/<feature>-spec.md

Each spec file covers one feature area. Format:

```markdown
# <Feature> Spec

## Requirements

### REQ-1: <Description>

<Shall statement — unambiguous, testable.>

**Testability:** <How the requirement is verified. Must be a real integration test, not a unit test.>

### REQ-2: <Description>

...

## Testing Rules (Mandatory)

- **TDD only:** Every requirement must have a test written before implementation
- **No mocks, no stubs, no test doubles:** All tests exercise real dependencies
- **No unit tests:** Tests must exercise full system behavior end-to-end
- **Provision and tear down:** Tests provision real infrastructure before running and tear down after
- **Red-green cycle:** Each test must fail against the baseline before implementation
```

### tasks.md

Phase-based, numbered tasks. Each task is a single unit of work that one agent can complete.

```markdown
## Phase 1 — <Phase name>

- [ ] 1.1 <Action> — <description of what to do>
- [ ] 1.2 <Action> — <description, reference other tasks for dependencies>
- [ ] 1.N Write integration tests covering: <list assertions>

## Phase 2 — <Phase name>

- [ ] 2.1 ...
```

**Task ordering rules:**
- Phase N depends on Phase N-1 completion
- Tests go at the end of each phase to verify the phase is working
- Independent tasks within a phase can run in parallel
- Each task must reference what it modifies (files, APIs)

## TDD Directives in Specs

Every spec must enforce TDD at the specification level:

### In each spec requirement

Add a "**Testability**" sub-bullet that describes the integration test needed. This forces the spec to be concrete:

```markdown
### REQ-3: InfonStore persists triples

Shall store an (subject, predicate, object) triple to disk and retrieve it by primary key.

**Testability:** A test provisions a real DuckDB store in a temp directory, upserts a triple, opens the store again in a fresh process, and asserts the triple is returned. No mocked database.
```

### In tasks.md

Every implementation task must be paired with a test task. Pattern:

```markdown
- [ ] 4.1 Implement `extract_text` in `src/infon/extract.py` — <details>
- [ ] 4.2 Write integration tests in `tests/test_extract.py` covering: <real assertions>
```

The test task always follows the implementation task in the dependency graph, enforcing the "write test first, verify red" cycle.

## No-Mock Directives

The following must appear verbatim in every spec file and tasks.md:

```markdown
## Hard Rules

- **No `MagicMock`, `mock`, `stub`, `patch`, or any form of test double.** Tests that exercise a mock are testing the mock, not the system.
- **No isolated unit tests.** Every test must exercise the full stack: real database, real file system, real network, real APIs.
- **Provision real dependencies.** If the feature needs a database, the test creates a real database. If it needs an external service, the test provisions a real instance (Docker, sandbox, etc.).
- **Tear down after the test.** No leftover state between test runs.
- **No `pytest.mark.skip`, `pytest.mark.xfail`, or equivalent bypasses.** If a test is hard to write, the difficulty is a signal that the real integration needs to be built.
- **No shortcuts.** A partially working system with skipped tests is not done.
```

This is not a style preference — it is a hard rule. The spec writer must ensure every requirement, as written, can only be verified by a real integration test.

## Review-After-Each-Task Directives

The spec must support the post-task review cycle from AGENTS.md Rule 4. This means:

### Each task must be independently verifiable

A task must produce a state that the full test suite can verify. If a task leaves the codebase in a partially-working state that would cause the regression suite to fail, the task is too large and must be split.

### Task boundaries aligned with regression safety

- Tasks that add new functionality without modifying existing code → safe for parallel execution
- Tasks that modify shared interfaces → must serialize or use feature flags
- Tasks that only add new files → safe for parallel execution
- Tasks that modify test configuration → must run sequentially

### Spec-level dependency notes

Add dependency notes to tasks that must run sequentially:

```markdown
- [ ] 5.3 Implement `PythonASTExtractor` — depends on 5.2 (BaseASTExtractor)
- [ ] 5.4 Implement `TypeScriptASTExtractor` — depends on 5.2 (BaseASTExtractor); can run with 5.3
```

This enables the epic-executor to launch independent tasks in parallel while serializing dependent ones.

## Checklist Before Delivering the Spec

- [ ] Proposal clearly states the problem and the changes
- [ ] Design has explicit decisions with rationale and alternatives
- [ ] Each requirement has a Testability sub-bullet
- [ ] Tasks are phased with dependencies noted
- [ ] Every implementation task has a paired test task
- [ ] No-mock directives are present in every spec file
- [ ] Tasks are independently verifiable (regression-safe boundaries)
- [ ] Spec can only be implemented correctly via TDD + real integrations
- [ ] Risk and trade-off section is honest about known failure modes
