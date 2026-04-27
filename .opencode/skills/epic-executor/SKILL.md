---
name: epic-executor
description: "Execute Beads epic tasks in parallel using opencode subagents."
---

# Epic Executor

Executes tasks from a Beads epic using opencode subagents, running independent tasks in parallel.

**Trigger:** Run when user provides an epic ID or asks to execute tasks from an epic.

**Input:** Epic ID (e.g., `bd-123`), or reference to `openspec/change-name/`

## Process

### 1. Assess Ready Tasks

```bash
bd ready --json
```

Filter for tasks belonging to the target epic. These are tasks with no unmet dependencies — they can run in parallel.

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
- Task description: <full task description from bd>
- Design context: <design field from bd>
- Acceptance criteria: <acceptance field from bd>
- Plan reference: <notes field from bd>

Rules (from AGENTS.md):
1. STRICT TDD — write tests first, verify they fail red, implement minimally, verify green
2. NO MOCKS, NO STUBS, NO UNIT TESTS — real dependencies, real integrations only
3. ITERATE UNTIL GREEN — no skips, no xfails, no shortcuts
4. POST-TASK REVIEW — verify spec compliance, run full regression, update openspec plan

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

### 4. Post-Task Review

After each subagent completes, run the full review **before** closing the task. This mirrors AGENTS.md Rule 4.

For each completed task:

a. **Spec compliance** — re-read the relevant `openspec/specs/` sections and `tasks.md` to verify the task's requirements are fully met.

b. **Full regression suite** — run `pytest -v` (or the project's full test command). **Every test must pass.** If any test fails, send the task back to the subagent with the failure output and iterate until green.

c. **Inconsistency resolution** — check for conflicts between the spec, the implementation, and `tasks.md`. Fix immediately.

d. **Update openspec plan** — modify the openspec plan to reflect the completed work.

e. **Update beads tasks** — update any pending beads tasks whose dependencies or blockers have changed.

Only after the review passes, close the task:
```bash
bd close <bd-id> --reason "Completed" --json
```

Update the epic's Timeline and Learnings sections:
```bash
bd update <epic-id> --json
```

**Important:** A task is NOT complete until the full regression suite passes. Do not skip this step to maintain parallelism — parallelism is between independent tasks, not between work and review.

### 5. Check for New Ready Tasks

After closing completed tasks, re-check for newly unblocked work:

```bash
bd ready --json
```

If tasks are now ready (their dependencies just completed), return to Step 2. If no ready tasks remain and no open tasks remain, the epic is complete.

### 6. Epic Completion

When all tasks are closed:

a. **Final regression:** Run the full test suite (`pytest -v`) to confirm zero regressions.
b. **Spec compliance pass:** Re-read all relevant `openspec/specs/` files and verify implementation matches every requirement.
c. **Update openspec plan:** Modify the openspec change plan to reflect completed work.
d. **Update epic Learnings:** Append final summary to the epic's Learnings section.
e. **Quality gates:** Run `ruff check src/`, `ruff format --check src/`, type checks.

### 7. Session Landing

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
Phase 2 tasks: [2.1, 2.2]      — depends on Phase 1 → wait, then launch 2 subagents
Phase 3 tasks: [3.1, 3.2, 3.3] — 3.1 and 3.2 independent, 3.3 depends on 3.1
                                    → launch 3.1 + 3.2 in parallel, then 3.3
```

## Key Rules

- **Maximum parallelism** — launch all independent tasks simultaneously
- **Claim before dispatch** — `bd update <id> --claim --json` before sending to subagent
- **No sequential bottlenecks** — never run tasks one at a time when they could run in parallel
- **Verify each subagent** — check that each subagent completed its task and ran its tests green
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
