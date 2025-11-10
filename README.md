# hazy-frames

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
![Coverage](./.github/badges/coverage.svg)
![Tests](https://img.shields.io/badge/tests-117%20passed-success)

**Hierarchical coordinate frames - crystal clear transforms**

A Python library for managing hierarchical reference frames and frame-aware geometric primitives with efficient transformation caching, inspired by [optiland](https://github.com/HarrisonKramer/optiland).

## Features

- **Hierarchical Frame System**: Build complex frame hierarchies with parent-child relationships
- **Frame-Aware Primitives**: `Point` and `Vector` classes that carry frame information
- **Automatic Transformations**: Transform primitives between any frames in the hierarchy
- **Efficient Caching**: Transformation matrices are cached for performance
- **Proper Geometric Semantics**: Type-safe arithmetic operations (e.g., Point - Point = Vector)
- **Flexible Transformations**: Support for rotation (Euler angles, matrices), translation, and scaling
- **Global Reference Frame**: Singleton global frame for world coordinates

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
world = Frame.get_global()
robot = Frame(parent=world, name="robot")
robot.translate(x=5, y=0, z=0).rotate_euler(z=90, degrees=True)

camera = robot.make_child(name="camera")  # different initialization methods
camera.translate(x=0, y=0, z=1)

# Create frame-aware primitives
point_in_camera = Point(1, 0, 0, frame=camera)

# Transform between frames
point_in_world = point_in_camera.to_frame(world)
point_in_robot = point_in_camera.to_frame(robot)

print(f"Camera: {point_in_camera}")
print(f"World: {point_in_world}")
print(f"Robot: {point_in_robot}")
```

## Core Concepts

### Frames

Frames represent coordinate systems and can be organized hierarchically:

```python
# Create frames
parent = Frame.global_frame().make_child("parent")
child = Frame(parent=parent, name="child")

# Apply transformations (chainable)
child.translate(x=10, y=5, z=0)
child.rotate_euler(x=45, y=0, z=90, degrees=True)
child.scale(2.0)  # uniform scaling
child.scale((1.0, 2.0, 1.5))  # non-uniform scaling

# Freeze frames to prevent modifications
child.freeze()
```

Transformations are applied in **S->R->T order** (Scale, Rotation, Translation) when converting from local to parent coordinates.

### Points and Vectors

Points represent positions, vectors represent directions/displacements:

```python
# Create primitives
origin = Point(0, 0, 0, frame=child)
direction = Vector(1, 0, 0, frame=child)

# Proper geometric arithmetic
point_a = Point(1, 2, 3, frame=child)
point_b = Point(4, 5, 6, frame=child)

displacement = point_b - point_a  # Returns Vector
new_point = point_a + displacement  # Returns Point

# Vectors support additional operations
vec = Vector(1, 0, 0, frame=child)
vec.normalize()  # Normalize in-place
magnitude = vec.magnitude

# Cross product
vec1 = Vector(1, 0, 0, frame=child)
vec2 = Vector(0, 1, 0, frame=child)
perpendicular = vec1.cross(vec2)  # Vector(0, 0, 1)
```

### Frame Transformations

```python
# Get transformation matrices
T_to_parent = frame.transform_to_parent  # 4x4 matrix
T_from_parent = frame.transform_from_parent  # Inverse
T_to_global = frame.transform_to_global  # To world frame
T_to_target = frame.transform_to(target_frame)  # To any frame

# Transform primitives
point_global = point_local.to_global()
point_target = point_local.to_frame(target_frame)

# Batch transformations for efficiency
points_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
transformed = frame.batch_transform_global(points_array)
```

### Unit Vectors and Origin

Frames provide convenient accessors for unit vectors and origin:

```python
# Unit vectors in frame's local coordinates
x_axis = frame.unit_x  # Vector(1, 0, 0) scaled by frame
y_axis = frame.unit_y
z_axis = frame.unit_z

# Unit vectors in global coordinates
x_global = frame.unit_x_global
y_global = frame.unit_y_global
z_global = frame.unit_z_global

# Frame origin
origin_local = frame.origin  # Point(0, 0, 0) in frame
origin_global = frame.origin_global  # Origin in global coords
```

## API Overview

### Frame Class

| Method/Property | Description |
|----------------|-------------|
| `translate(x, y, z)` | Add translation to frame |
| `rotate_euler(x, y, z, seq, degrees)` | Add Euler angle rotation |
| `rotate(matrix)` | Add rotation from 3x3 matrix |
| `scale(factor)` | Add scaling (uniform or non-uniform) |
| `freeze()`/`unfreeze()` | Prevent/allow modifications |
| `transform_to_parent` | 4x4 transformation matrix to parent |
| `transform_to_global` | 4x4 transformation matrix to global |
| `transform_to(target)` | 4x4 transformation matrix to target frame |
| `create_point(x, y, z)` | Create Point in this frame |
| `create_vector(x, y, z)` | Create Vector in this frame |

### Point Class

| Method/Property | Description |
|----------------|-------------|
| `to_frame(target)` | Transform to target frame |
| `to_global()` | Transform to global frame |
| `x`, `y`, `z` | Coordinate components |
| `coords` | NumPy array of coordinates |
| `Point + Vector` | Returns Point (displacement) |
| `Point - Point` | Returns Vector (difference) |
| `Point - Vector` | Returns Point (reverse displacement) |

### Vector Class

| Method/Property | Description |
|----------------|-------------|
| `to_frame(target)` | Transform to target frame |
| `to_global()` | Transform to global frame |
| `x`, `y`, `z` | Vector components |
| `coords` | NumPy array of components |
| `magnitude` | Vector length |
| `normalize()` | Normalize to unit length (in-place) |
| `cross(other)` | Cross product with another vector |
| `Vector + Vector` | Returns Vector (sum) |
| `Vector - Vector` | Returns Vector (difference) |
| `Vector + Point` | Returns Point (displacement) |

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
