# Quick Start

## Basic Usage

### Creating Frames

Every coordinate transformation starts with a root frame:

```python
from hazy import Frame

world = Frame.make_root("world")
```

Create child frames with transformations:

```python
camera = (
    world.make_child("camera")
    .translate(x=1, y=2, z=5)
    .rotate_euler(y=90, degrees=True)
)
```

### Working with Points

Create points in a frame:

```python
point = world.point(1, 2, 3)
```

Transform points between frames:

```python
point_in_camera = point.to_frame(camera)
```

### Working with Vectors

Vectors are direction quantities (unaffected by translation):

```python
direction = world.vector(0, 0, 1)  # Looking down Z-axis
direction_in_camera = direction.to_frame(camera)
```

### Working with Many Points

For large data, use the batched types `Points` and `Vectors`. They store
coordinates of shape `(..., 3)` in one frame and transform all of them in a
single vectorized operation, about as fast as plain numpy:

```python
import numpy as np
from hazy import Points

cloud = camera.points(np.random.rand(10_000, 3))  # Points, shape (10000,)
cloud_world = cloud.to_frame(world)               # one matrix product

offsets = cloud - camera.origin         # Vectors (broadcasts with a Point)
distances = offsets.magnitude           # ndarray, shape (10000,)
directions = offsets.copy().normalize() # normalize() works in-place
```

Indexing follows numpy semantics:

```python
cloud[0]                 # Point (independent copy)
cloud[100:200]           # Points (view, shares memory with cloud)
cloud[distances < 0.5]   # Points (copy)
cloud.to_list()          # list[Point]
Points.from_primitives([p1, p2])  # back from single points
```

## Type Safety

Points and vectors are distinct types with mathematically valid operations:

```python
# Valid operations
v1 + v2          # Vector addition
point + vector   # Point displacement
point - point    # Vector between points
vector * 2       # Scalar multiplication

# Invalid operations (raise TypeError)
point + point    # Can't add points
point * 2        # Can't scale points
```

The same rules apply to `Points` and `Vectors`, also when mixed with single
primitives (`points - point` gives `Vectors`). `vectors * factors` accepts a
scalar or one factor per element.

## Convert to NumPy

Opt out to raw arrays when needed:

```python
import numpy as np

coords = np.array(point)  # [x, y, z]
point * 2  # TypeError
np.array(point) * 2  # Works: [2x, 2y, 2z]

raw = np.asarray(cloud)  # (N, 3) view without copy, writes affect cloud
raw = np.array(cloud)    # (N, 3) independent copy
```

## Next Steps

- See {doc}`examples/index` for detailed examples
- Read the {doc}`api/index` for complete API reference
