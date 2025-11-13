# hazy-frames

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
![Coverage](./.github/badges/coverage.svg)
![Tests](https://img.shields.io/badge/tests-163%20passed-success)

**Hierarchical coordinate frames - crystal clear transforms**

A Python library for managing hierarchical reference frames and frame-aware geometric primitives with efficient transformation caching, inspired by [optiland](https://github.com/HarrisonKramer/optiland).

`hazy-frames` fills the gap between heavyweight robotics frameworks like `ROS` and specialized libraries like `pytransform3d` or `astropy`. It provides a lightweight, Pythonic API for tracking geometric objects across multiple coordinate systems - perfect for raytracing, computer graphics, robotics simulations, CFD mesh generation (OpenFOAM `blockMesh`), or any application requiring coordinate transformations. Born from the need of managing objects across different frames in a non-sequential raytracing project, it offers a clean, intuitive interface that just works.

**Philosophy:** Do one thing well with minimal dependencies (only numpy and scipy). No visualization, no physics simulation, no file I/O - just clean, efficient coordinate transformations. The library enforces geometric semantics (e.g., Point + Point is forbidden, but Point - Point = Vector is allowed) as an opt-out contract - preventing common bugs while staying Pythonic. Designed as a lightweight building block that integrates seamlessly into your existing toolchain.

## Features

- **Hierarchical Frame System**: Build complex frame hierarchies with parent-child relationships
- **Frame-Aware Primitives**: `Point` and `Vector` classes that carry frame information
- **NumPy Integration**: Native array support with batch operations, factory methods, and stack functions
- **Automatic Transformations**: Transform primitives between any frames in the hierarchy
- **Efficient Caching**: Transformation matrices are cached for performance
- **Proper Geometric Semantics**: Type-safe arithmetic operations (e.g., Point - Point = Vector)
- **Flexible Transformations**: Support for rotation (Euler angles, quaternions, matrices), translation, and scaling

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/Littie28/hazy-frames.git
```

Or download the wheel from [releases](https://github.com/Littie28/hazy-frames/releases/latest):

```bash
pip install hazy_frames-0.1.0-py3-none-any.whl
```

For development:

```bash
# Clone the repository
git clone https://github.com/Littie28/hazy-frames.git
cd hazy-frames

# Install with development dependencies
uv sync --group dev
```

## Quick Start

```python
from hazy import Frame, Point, Vector

# Create a frame hierarchy
world = Frame.make_root("world")
robot = world.make_child(name="robot")
robot.translate(x=5, y=0, z=0).rotate_euler(z=90, degrees=True)

camera = robot.make_child(name="camera")  # different initialization methods
camera.translate(x=0, y=0, z=1)

# Create frame-aware primitives
point_in_camera = camera.point(1, 0, 0)

# Transform between frames
point_in_world = point_in_camera.to_frame(world)
point_in_robot = point_in_camera.to_frame(robot)

print(f"Camera: {point_in_camera}")
print(f"World: {point_in_world}")
print(f"Robot: {point_in_robot}")
```

## Key Concepts

Frames represent coordinate systems organized hierarchically. Points and Vectors are frame-aware primitives that can be transformed between any frames in the hierarchy.

```python
import numpy as np
from hazy import Frame, Point, Vector

# Build frame hierarchy
root = Frame.make_root(name="root")
child = root.make_child("child")
child.translate(x=10, y=5, z=0).rotate_euler(z=90, degrees=True)

# Create primitives (with or without numpy)
point = child.point(1, 2, 3)
point_from_array = Point.from_array([1, 2, 3], frame=child)
vector = child.vector(0, 0, 1)

# Geometric operations
displacement = point_b - point_a  # Point - Point = Vector
new_point = point + displacement  # Point + Vector = Point
magnitude = vector.magnitude
vector.normalize()

# Transform between frames
point_in_root = point.to_frame(root)
point_global = point.to_global()

# Batch operations with numpy
coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
points = Point.list_from_array(coords, frame=child)
transformed = child.batch_transform_points_global(coords)
```

All classes are fully documented with type hints and docstrings. Use your IDE's autocomplete or `help()` to explore the API.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/hazy --cov-report=html

# Run specific test markers
pytest -m unit
pytest -m integration

# Watch mode for development
pytest-watcher
```
## License

 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
