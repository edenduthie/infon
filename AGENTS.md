# Project Instructions for AI Agents

This file provides instructions and context for the **infon** project — an open-source Python project described by the openspec specification.

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:d4f96305 -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- END BEADS INTEGRATION -->

## Development Approach — Phased MVP

Development proceeds in **phases**, each producing a working vertical slice of the full system. The goal is to reach a minimal end-to-end system (MVP) **as quickly as possible**, so design errors are caught early.

### Why Phased MVP?

Building each module to completion before moving on hides integration problems until the final assembly phase — that's when you discover your data model doesn't fit the store, or the encoder's output doesn't match the retrieval pipeline. Phased MVP surfaces these issues in Phase 1 or 2, not Phase 12.

**Each phase must:**

1. **Be independently testable** — the system works end-to-end at the phase boundary, even if minimally.
2. **Add one new capability layer** — Phase 1 is the data model, Phase 2 adds persistence, Phase 3 adds encoding, etc. Each phase builds on the previous one and verifies the integration works.
3. **Ship as little as necessary to prove the concept** — the first end-to-end flow through all layers should happen in the earliest possible phase, even if it's a stripped-down flow.

## Development Rules

### 1. Test-Driven Development (TDD) — Mandatory

All development **must** follow strict TDD:

1. **Write the test first** — before any implementation code.
2. **Verify the test fails (red)** — run the test and confirm it fails. This proves the test is valid and not vacuously passing.
3. **Write the minimal implementation** — write only the code needed to make the test pass.
4. **Run the test again (green)** — confirm the test passes.
5. **Iterate** — repeat the red-green cycle for each piece of functionality until all tests pass.

Do not skip the red step. A test that passes before implementation is writing meaningless coverage.

### 2. No Mocks, No Stubs, No Unit Tests — Ever

This is a **hard rule**. All tests must exercise the **full, real functionality** of the system:

- **No `MagicMock`, `mock`, `stub`, `patch`, or any form of test double.**
- **No isolated unit tests** that test a single function in vacuum.
- Tests must **provision real dependencies** before running (real databases, real services, real APIs, real sandboxes, etc.).
- Tests must **tear down all provisions** after completion.
- The goal is to verify that the entire system works end-to-end, not that a function behaves correctly against a fake.

Testing against mocks is testing the mock, not the system. This rule exists to ensure every test validates real behavior.

### 3. Iterate Until Green — No Shortcuts

Do not accept incomplete implementations or partial test coverage:

- Keep iterating through the red-green cycle until **all tests pass**.
- Do not bypass failing tests with mocks, skips, or `pytest.mark.xfail`.
- Do not accept code that does not fully meet the openspec specification.
- If a test is difficult to write, the difficulty is a signal that the real integration needs to be built — do not circumvent it.

### 4. Phase-Boundary Review — Mandatory at the End of Each Phase

After completing all tasks in a phase, perform a thorough review **before moving to the next phase**. This is where the heavy lifting happens — not after every task, but at each phase boundary, where integration risk is highest.

a. **Full regression suite** — run `pytest -v`. Every test must pass. Phase boundaries are where regressions actually surface, so this check matters.

b. **Spec compliance** — re-read the relevant openspec spec and `tasks.md` sections to verify all requirements for this phase and prior phases are still met and functioning.

c. **MVP checkpoint** — does the system work end-to-end at this phase? Even a minimal flow through all built layers should succeed. If not, the phase is incomplete.

d. **Inconsistency Resolution** — if any inconsistencies are found between the spec, implementation, or tasks, fix them immediately.

e. **Update openspec plan** — modify the openspec plan to reflect the completed work and any discovered changes.

f. **Update beads tasks** — update any pending beads tasks to reflect the current state, dependencies, and blockers.

**After each individual task** within a phase, the agent only needs to confirm the task's own tests pass. The full review is reserved for the phase boundary unless something goes wrong mid-phase.

This balance prevents wasteful regression checks after trivial tasks (like creating a directory or writing a `.toml` file) while still catching regressions at the points that matter.

## Build & Test

```bash
# Install dependencies
pip install -e ".[dev]"

# Run full test suite
pytest

# Run tests with verbose output
pytest -v
```

## Architecture Overview

_open-source Python project — refer to openspec/ for full specification_

## Conventions & Patterns

- Follow TDD religiously: red → green → refactor → repeat
- Integration tests only — no mocks, no stubs, no unit tests
- Provision and tear down real dependencies in every test
- Run full regression and review spec compliance at each phase boundary (not after every task)
- Update openspec and beads tasks to stay in sync with progress
