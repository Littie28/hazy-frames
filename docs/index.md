# hazy-frames

Crystal clear coordinate transformations for Python.

**hazy-frames** is a lightweight library for managing hierarchical coordinate frames and transforming points and vectors between them. Built for the NumPy and SciPy ecosystems with a pythonic, chainable API.

```python
import numpy as np
from hazy import Frame

# Create coordinate frames with chaining
world = Frame.make_root(name="world")
camera = (
    world.make_child("camera")
    .translate(z=5)
    .rotate_euler(y=90, degrees=True)
    .freeze()
)

# Transform points between frames
point_world = world.point(1, 2, 3)
point_camera = point_world.to_frame(camera)

# Transform many points at once
cloud_camera = world.points(np.random.rand(1000, 3)).to_frame(camera)
```

## Features

- **Type Safety**: Points and Vectors are distinct types
- **Mathematical Correctness**: Only formally valid operations allowed
- **Chainable API**: Fluent method chaining for frame transformations
- **Batched Primitives**: `Points` and `Vectors` transform millions of
  coordinates at numpy speed with the same type rules
- **Zero Runtime Overhead**: Uses views where possible, avoids unnecessary copies
- **Easy Opt-Out**: Convert to numpy arrays when you need raw manipulation

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
examples/index
api/index
```

## Installation

```bash
pip install hazy-frames
```

Or with uv:

```bash
uv add hazy-frames
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
