---
name: epic-executor
description: "Execute Beads epic tasks in parallel using opencode subagents."
---

# Epic Executor

Executes tasks from a Beads epic using opencode subagents, running independent tasks in parallel within each phase, with full review at phase boundaries.

**Trigger:** Run when user provides an epic ID or asks to execute tasks from an epic.

**Input:** Epic ID (e.g., `bd-123`), or reference to `openspec/change-name/`

## Core Principle — Phased MVP

Tasks are organized into phases. Within each phase, independent tasks run in parallel. At the phase boundary — when all tasks in a phase are complete — a full review and regression suite runs. This catches integration problems early without wasting cycles on regression checks after trivial tasks.

**Per-task:** Subagent confirms its own tests pass. No full regression needed.
**Per-phase:** Full regression, spec compliance, openspec update, beads update. **This is mandatory.**

## Process

### 1. Assess Ready Tasks

```bash
bd ready --json
```

Filter for tasks belonging to the target epic. These are tasks with no unmet dependencies — they can run in parallel. Note which phase they belong to (tasks prefixed with `Phase N:`).

### 2. Classify Tasks by Subagent Type

Group ready tasks by their subagent specialization:

| Task character | Subagent | Reasoning |
|---|---|---|
| Research, codebase exploration, reading existing patterns | `explore` | Fast, read-only, parallel-safe |
| Code implementation, tests, file creation, full TDD cycles | `general` | Has full tool access, handles multi-step work |
| Documentation only | `general` | Writes files, follows conventions |

### 3. Launch Parallel Subagents

For each ready task, dispatch a subagent. **Maximum parallelism:** launch all independent tasks simultaneously. Each subagent gets a self-contained prompt:

```
Execute task <bd-id> "<task title>"

Context:
- Epic: <epic-id>
- Phase: <phase number>
- Task description: <full task description from bd>
- Design context: <design field from bd>
- Acceptance criteria: <acceptance field from bd>
- Plan reference: <notes field from bd>

Rules (from AGENTS.md):
1. STRICT TDD — write tests first, verify they fail red, implement minimally, verify green
2. NO MOCKS, NO STUBS, NO UNIT TESTS — real dependencies, real integrations only
3. ITERATE UNTIL GREEN — no skips, no xfails, no shortcuts

Commit format: <type>(<scope>): <description> (<bd-id>)
```

Dispatch using the Task tool:

```
Task(subagent_type="explore", prompt="<above prompt for research tasks>")
Task(subagent_type="general", prompt="<above prompt for implementation tasks>")
```

**Parallel execution rules:**
- All tasks with no mutual dependencies launch at once
- If Task A and Task B are both ready (no deps on each other), launch both in the same message
- If Task B depends on Task A, wait for Task A to complete in Step 4, then proceed to Step 5

### 4. Collect Task Results

Wait for all subagents to return. For each completed task:

a. Verify the subagent's own tests pass (the subagent should have already done this).
b. If a subagent's tests fail, send it back with the failure output and iterate until green.
c. Once green, close the task:

```bash
bd close <bd-id> --reason "Completed" --json
```

Update the epic's Timeline and Learnings sections:
```bash
bd update <epic-id> --json
```

**Per-task note:** This is a lightweight check. Do NOT run the full regression suite here. The full review happens at the phase boundary (Step 5).

### 5. Check Phase Completion

After closing all currently-ready tasks, determine if the current phase is complete:

```bash
bd ready --json
```

If no tasks from the current phase remain open, **the phase is complete**. Proceed to Step 6 (Phase-Boundary Review).

If new tasks from a later phase are now unblocked, hold — do not start the next phase until the phase-boundary review passes.

### 6. Phase-Boundary Review

**This is mandatory before starting the next phase.**

a. **Full regression suite** — run `pytest -v`. Every test must pass. If any test fails, fix the issue before proceeding.

b. **Spec compliance** — re-read the relevant `openspec/specs/` sections and `tasks.md` to verify all requirements for this phase and prior phases are still met and functioning.

c. **MVP checkpoint** — does the system work end-to-end at this phase? Even a minimal flow through all built layers should succeed. If not, the phase is incomplete.

d. **Inconsistency Resolution** — if any inconsistencies are found between the spec, implementation, or tasks, fix them immediately.

e. **Update openspec plan** — modify the openspec plan to reflect the completed work and any discovered changes.

f. **Update beads tasks** — update any pending beads tasks to reflect the current state, dependencies, and blockers.

Only after the phase-boundary review passes may you begin the next phase. Return to Step 1.

### 7. Epic Completion

When all tasks are closed and all phases have passed their boundary review:

a. **Final regression:** Run `pytest -v` one last time.
b. **Final spec compliance:** Re-read all `openspec/specs/` files and verify full compliance.
c. **Quality gates:** Run `ruff check src/`, `ruff format --check src/`, type checks.
d. **Update epic Learnings:** Append final summary to the epic's Learnings section.

### 8. Session Landing

Follow the beads skill landing procedure:

```bash
git pull --rebase
bd dolt push
git push
git status  # Must show "up to date with origin"
```

## Parallel Execution Pattern

```
Phase 1 tasks: [1.1, 1.2, 1.3] — all independent → launch 3 subagents
  → all 3 complete → close tasks → PHASE 1 REVIEW (full regression, spec check)
  → review passes → proceed

Phase 2 tasks: [2.1, 2.2]      — depends on Phase 1 → now launchable
  → both complete → close tasks → PHASE 2 REVIEW
  → review passes → proceed

Phase 3 tasks: [3.1, 3.2, 3.3] — 3.1 and 3.2 independent, 3.3 depends on 3.1
                                    → launch 3.1 + 3.2 in parallel, then 3.3
  → all complete → PHASE 3 REVIEW
```

## Key Rules

- **Maximum parallelism within a phase** — launch all independent tasks simultaneously
- **Claim before dispatch** — `bd update <id> --claim --json` before sending to subagent
- **No sequential bottlenecks** — never run tasks one at a time when they could run in parallel
- **Phase-gate discipline** — do not start the next phase until the phase-boundary review passes
- **Per-task: lightweight check only** — confirm the subagent's own tests pass
- **Per-phase: full review** — this is where regressions are caught
- **Always `--json`** for all `bd` commands
- **Conventional commits** — all commits must follow `<type>(<scope>): <description> (<bd-id>)`
- **TDD is non-negotiable** — every subagent must follow red-green-refactor
- **No mocks** — this is the hardest rule; if a subagent tries to use mocks, correct it immediately

## Conventional Commits (REQUIRED)

All commits follow https://www.conventionalcommits.org/en/v1.0.0/

Format: `<type>(<scope>): <description> (<bd-id>)`

Types:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `refactor:` — code refactoring
- `test:` — adding tests
- `chore:` — maintenance
