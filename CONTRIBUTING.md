# Contributing to hazy-frames

Thank you for considering contributing to hazy-frames! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Littie28/hazy-frames.git
cd hazy-frames

# Install with development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/hazy --cov-report=html

# Run specific test markers
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m numpy         # NumPy integration tests

# Watch mode for development
pytest-watcher
```

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Format code
ruff format

# Check linting
ruff check

# Auto-fix linting issues
ruff check --fix
```

Pre-commit hooks will automatically run these checks before each commit.

## Testing Guidelines

We use pytest with three test categories:

- `@pytest.mark.unit` - Unit tests for isolated components
- `@pytest.mark.integration` - Tests for component interplay and realistic scenarios
- `@pytest.mark.numpy` - Tests for NumPy integration and compatibility

Focus on testing behavior, not implementation details. Include tests for error cases with meaningful error messages.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(frame): add support for quaternion rotation
fix(vector): correct normalization for zero-length vectors
docs: add tutorial for batch transformations
test(numpy): add tests for array stacking operations
```

## Documentation

Documentation is built with Sphinx and hosted on ReadTheDocs:

```bash
# Build documentation locally
cd docs
make html

# View in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and coverage is maintained
4. Update documentation if needed
5. Submit a pull request with a clear description

## Questions?

Feel free to open an issue for questions, bug reports, or feature requests.
