# Geometric Primitives

## Point

```{eval-rst}
.. autoclass:: hazy.Point
   :members:
   :undoc-members:
   :show-inheritance:
```

## Vector

```{eval-rst}
.. autoclass:: hazy.Vector
   :members:
   :undoc-members:
   :show-inheritance:
```

## Batched Primitives

`Points` and `Vectors` hold many coordinates of shape `(..., 3)` in one frame.
They follow the same type rules as `Point` and `Vector`, but run every operation
vectorized, which makes them the fast path for large data.

### Points

```{eval-rst}
.. autoclass:: hazy.Points
   :members:
   :inherited-members:
   :show-inheritance:
```

### Vectors

```{eval-rst}
.. autoclass:: hazy.Vectors
   :members:
   :inherited-members:
   :show-inheritance:
```
