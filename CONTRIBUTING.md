# Contributing to infon

Thank you for your interest in contributing to infon!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/infon.git
   cd infon
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest -v
   ```

## Development Guidelines

### Test-Driven Development (TDD)

All development must follow strict TDD:

1. Write the test first
2. Verify the test fails (red)
3. Write minimal implementation
4. Verify the test passes (green)
5. Refactor if needed
6. Repeat

### No Mocks Policy

- All tests must use real dependencies (real DuckDB, real files, real models)
- No `mock`, `MagicMock`, `stub`, or `patch` allowed
- Tests should provision and tear down real resources

### Code Quality

Run quality checks before submitting:

```bash
pytest tests/ -v
ruff check src/
ruff format --check src/
mypy src/infon/
```

## Pull Request Process

1. Ensure all tests pass
2. Follow conventional commit format: `type(scope): description`
3. Update CHANGELOG.md with your changes
4. Request review from maintainers

## Questions?

Open an issue for any questions or concerns.
