# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-15

### Added

- **Comprehensive NumPy Integration**: Native support for NumPy arrays and batch operations
  - `from_array()` factory methods for `Point` and `Vector` classes
  - `list_from_array()` for batch creation from NumPy arrays
  - Support for array-like inputs in `Frame.point()` and `Frame.vector()`
  - NumPy stack operations (`np.stack()`, `np.hstack()`, `np.vstack()`) for primitives
  - Compatibility with NumPy scalar types (`np.float64`, `np.int32`, etc.)
  - Dedicated test suite for NumPy integration (`tests/test_numpy_integration.py`)

- **Custom Formatters**: `__format__()` method for `GeometricPrimitive` with multiple output styles
  - Standard format specifiers (e.g., `:6.2f`, `:.3f`, `:10.4e`) format all three components
  - `'a'`: Array-only format without class name or frame info
  - `'n'`: No-frame format showing only class name and coordinates
  - Default: Full format with class name and frame

- **Multi-Index Support**: `__getitem__()` now supports advanced NumPy indexing
  - Integer indexing: `primitive[0]` returns float
  - Slicing: `primitive[0:2]` returns NDArray
  - Fancy indexing: `primitive[[0, 2]]` returns NDArray

- **Quaternion Rotation Support**: `Frame.rotate_quat()` method for quaternion-based rotations

- **Child Frame Creation**: `Frame.make_child()` method to create child frames directly

- **Batch Transformations**: Explicit methods for batch operations
  - `batch_transform_points()` for transforming multiple points
  - `batch_transform_vectors()` for transforming multiple vectors

- **Parent-Child Tracking**: Frames now maintain bidirectional parent-child relationships
  - Each frame tracks its direct children
  - Cache invalidation propagates recursively through the hierarchy

- **Documentation**: Complete Sphinx documentation with ReadTheDocs integration
  - Installation guide, quick start tutorial, and API reference
  - Jupyter notebook examples integrated via nbsphinx
  - Available at [hazy-frames.readthedocs.io](https://hazy-frames.readthedocs.io/)

- **Examples**: Two interactive Jupyter notebooks demonstrating core concepts
  - `01-surface_intersections.ipynb`: Ray-plane intersection with coordinate transformations
  - `02-frame-hierarchies.ipynb`: Parent-child relationships and dynamic transformations

- **CI/CD**: GitHub Actions workflow with pytest, ruff linting, and codecov integration

- **Pre-commit Hooks**: Automated code quality checks with ruff linter and formatter

### Changed

- **API Naming Improvements**: More consistent and pythonic naming conventions
  - Removed `coords` property (use `np.array(primitive)` instead)
  - Renamed `unit_x/y/z` to `axis_x/y/z` for frame axes
  - Removed `make_` prefix from `Frame.point()` and `Frame.vector()` methods
  - Simplified `Frame.scale()` API: uniform scaling with single argument, per-axis with three

- **Root Frame Handling**: Removed implicit singleton pattern
  - Frames can now be orphans (no parent) instead of defaulting to global singleton
  - More explicit frame hierarchy management

- **API Documentation**: Comprehensive docstrings with examples for all public methods

### Fixed

- **Cache Invalidation**: Parent frame modifications now properly invalidate child frame caches
  - Previously, modifying a parent would cause stale transformations in child frames
  - Cache invalidation now propagates recursively through the entire hierarchy

- **Vector Batch Transform**: Fixed critical bug in `batch_transform_vectors_global()` using incorrect homogeneous coordinate (was `w=1`, now correctly `w=0`)

- **Parent Immutability**: Frame parent is now immutable after creation to maintain hierarchy consistency

### Removed

- **Legacy Methods**: Removed redundant `Vector.unit_x/y/z()` methods (use `frame.axis_x/y/z` instead)

## [0.1.0] - 2025-11-09

Initial release of hazy-frames.

### Added

- Hierarchical frame system with parent-child relationships
- Frame-aware geometric primitives (`Point` and `Vector`)
- Automatic coordinate transformations between frames
- Transformation caching for performance
- Type-safe geometric arithmetic (e.g., Point - Point = Vector)
- Support for rotation (Euler angles), translation, and scaling
- NumPy array conversion via `__array__()` protocol
- Comprehensive test suite with pytest
- MIT License

[Unreleased]: https://github.com/Littie28/hazy-frames/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Littie28/hazy-frames/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Littie28/hazy-frames/releases/tag/v0.1.0
