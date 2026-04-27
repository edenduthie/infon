---
name: plan-to-beads
description: "Convert openspec plan to Beads epic with tasks and dependencies."
---

# Plan to Beads

Converts an openspec plan into a Beads epic with tasks and dependencies.

**Trigger:** After user approves an openspec plan, use this skill automatically.

**Input:** Openspec plan file path (e.g., `openspec/tasks.md` inside an openspec change, or a standalone implementation plan)

## Process

### 1. Parse Plan

Read the openspec files and extract:
- **Title** from `openspec/proposal.md` — the `# Change: <title>` heading
- **Goal** from `openspec/proposal.md` — the `## Why` section
- **Architecture** from `openspec/design.md` — key decisions and constraints
- **Requirements** from `openspec/specs/` — all spec files in the specs directory
- **Tasks** from `openspec/tasks.md` — numbered, phased tasks (e.g., `1.1`, `2.3`, `3.8`)

If `tasks.md` does not exist, derive tasks from the spec files by identifying distinct implementation units.

### 2. Create Epic

```bash
bd create --type=epic --priority=0 --json \
  --title="<title from proposal>" --description="<goal from Why section>"
```

Add two sections to the epic:

- **Section: Timeline** — to be updated by agents when they start/stop/pause/resume tasks from this epic.
- **Section: Learnings** — agents append a short summary of learnings (happy path, errors, findings, shortcuts) upon task completion. Reference the task ID.

### 3. Infer Dependencies

Apply these rules to determine task dependencies:

- **Phase ordering:** Tasks in Phase N depend on all Phase N-1 tasks being complete.
- **File overlap:** Task N and Task M modify the same file → M depends on N.
- **Explicit references:** "after Task N", "requires Task N", "defined in Phase X task X.Y" → hard dependency.
- **Cross-phase references:** If Task 3.8 references output from Task 1.4, add that dependency.

### 4. Create Tasks

For each task in the plan:

```bash
bd create --type=task --priority=0 --json \
  --parent=<epic-id> \
  --title="<task title>" \
  --description="<full implementation steps - DO NOT summarize>" \
  --design "<architecture context from design.md>" \
  --notes "Plan: openspec/tasks.md (Task X.Y)" \
  --acceptance="<acceptance criteria from specs/>"
```

**Rules:**
- **Never summarize** implementation steps — copy the full detail verbatim.
- **Separate concerns:** Description = what to build, Design = architectural context.
- **Always `--json`** for all `bd` commands.
- **Link to plan:** Include the plan file reference in task notes.
- **Tag by phase:** Include the phase number in the title prefix (e.g., `Phase 1: 1.1 Create repository structure`).

### 5. Wire Dependencies

```bash
bd dep add <dependent-task-id> <dependency-task-id> --json
```

Run this for every inferred dependency from step 3.

### 6. Output Summary

```
Epic: <id> "<title>"
Tasks: N total (M across K phases)
Dependencies: D edges wired
Ready: bd ready --json
```

## Key Rules

- **Never summarize** implementation steps — copy full detail from the plan
- **Separate concerns:** Description = what to build, Design = why and how
- **Always `--json`** for all `bd` commands
- **Link to plan:** Include plan file reference in task notes
- **Phase ordering:** Enforce phase boundaries as dependencies
- **File overlap:** Detect shared files and create dependencies
- **Epic sections:** Always add Timeline and Learnings sections to the epic

## Conventional Commits (REQUIRED)

When tasks are executed, all commits MUST follow https://www.conventionalcommits.org/en/v1.0.0/

Format: `<type>(<scope>): <description> (<task-id>)`

Types:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `refactor:` — code refactoring
- `test:` — adding tests
- `chore:` — maintenance

Include this in task acceptance criteria where commits are expected.
