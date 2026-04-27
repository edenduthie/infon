# Project Instructions for AI Agents

This file provides instructions and context for the **infon** project — an open-source Python project described by the openspec specification.

<!-- BEGIN BEADS INTEGRATION v:2 profile:skill hash:d4f96305 -->
## Issue Tracking

This project uses **bd (beads)** for issue tracking. Load the `beads` skill via the `skill` tool for full workflow instructions.

**Short reminder:** always use `bd` (not TODOs), use `--json`, check `bd ready` for unblocked work, `bd update <id> --claim` to claim, `bd close <id>` to complete. On session end: push everything.

<!-- END BEADS INTEGRATION -->

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

### 4. Post-Task Review — Mandatory

After completing each task, perform a thorough review:

a. **Spec compliance** — re-read the relevant openspec spec and `tasks.md` sections to verify all prior requirements are still met and functioning.

b. **Regression tests** — run the **full test suite** to confirm no regressions. Every test must pass before moving on.

c. **Next-step validation** — verify the planned next steps make sense given the current state.

d. **Inconsistency Resolution** — if any inconsistencies are found between the spec, implementation, or tasks, fix them immediately.

e. **Update openspec plan** — modify the openspec plan to reflect the completed work and any discovered changes.

f. **Update beads tasks** — update any pending beads tasks to reflect the current state,Dependencies, and blockers.

This review is not optional. It ensures the project remains consistent, spec-compliant, and regression-free at every step.

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
- Review spec compliance and run full regressions after every task
- Update openspec and beads tasks to stay in sync with progress
