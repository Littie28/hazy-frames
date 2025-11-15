# Documentation

This directory contains the Sphinx documentation for hazy-frames.

## Building locally

Install dependencies:

```bash
uv sync --group dev
```

Build the documentation:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

View the documentation:

```bash
open docs/_build/html/index.html
```

## ReadTheDocs

The documentation is automatically built and deployed to ReadTheDocs on every push to `main`.

Configuration: `.readthedocs.yaml`

## Notebooks

Example notebooks from `examples/` are copied to `docs/examples/` during the build process. The notebooks are ignored in git to avoid duplication.
