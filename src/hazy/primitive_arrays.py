"""Batched geometric primitives: many points or vectors sharing one frame.

`Points` and `Vectors` store coordinates as a float array of shape (..., 3) and
follow the same type rules as `Point` and `Vector`, but every operation runs
vectorized over all elements. Frame bookkeeping happens once per batch.

Indexing follows numpy semantics on the leading axes:
    - A single element (`cloud[i]` for 1-D batches) is returned as an
      independent `Point`/`Vector` copy.
    - Slices return a batch that shares memory with the original (view),
      masks and integer arrays return a copy, exactly like numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from hazy.constants import VSMALL, VVSMALL
from hazy.primitives import GeometricPrimitive, Point, Vector
from hazy.utils import check_same_frame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray

    from hazy.frame import Frame


HANDLED_ARRAY_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(numpy_function: Callable[..., Any]):
    """Register a function as implementation of a numpy array function."""

    def decorator(func):
        HANDLED_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


type Geometric = GeometricPrimitive | PrimitiveArray


def _coordinates(obj: Geometric) -> NDArray[np.floating]:
    if isinstance(obj, PrimitiveArray):
        return obj._coords
    return obj._homogeneous[:3]


def _is_point_like(obj: Geometric) -> bool:
    return isinstance(obj, Point | Points)


def _leading_axis(axis: int) -> int:
    """Map an axis over the leading (element) dimensions to the storage axis."""
    return axis - 1 if axis < 0 else axis


class PrimitiveArray:
    """Base class for batches of geometric primitives sharing one frame.

    Coordinates are stored as a float array of shape (..., 3). The leading
    dimensions index the elements, the last dimension holds (x, y, z).
    """

    __slots__ = ("_coords", "frame")

    _element_type: ClassVar[type[Point | Vector]]
    _is_point: ClassVar[bool]

    def __init__(self, coords: ArrayLike, frame: Frame, *, copy: bool = True):
        """Initialize batch from coordinates of shape (..., 3).

        Args:
            coords: Array-like with shape (..., 3), at least 2-D
            frame: Reference frame shared by all elements
            copy: If False, reuse the memory of `coords` when possible

        Raises:
            ValueError: If coords is not at least 2-D with last dimension 3
        """
        coords = np.array(coords, dtype=float, copy=True if copy else None)
        name = type(self).__name__
        if coords.ndim < 2 or coords.shape[-1] != 3:
            raise ValueError(
                f"{name} expects coordinates of shape (..., 3) with at least "
                f"2 dimensions, got shape {coords.shape}.\n"
                "Use:\n"
                f"  frame.{name.lower()}([[x1, y1, z1], [x2, y2, z2]])  # Batch\n"
                f"  frame.{self._element_type.__name__.lower()}(x, y, z)"
                "  # Single element"
            )
        self._coords = coords
        self.frame = frame

    @classmethod
    def _wrap(cls, coords: NDArray[np.floating], frame: Frame) -> Self:
        """Wrap a float array of shape (..., 3) without validation (internal)."""
        batch = object.__new__(cls)
        batch._coords = coords
        batch.frame = frame
        return batch

    @classmethod
    def from_array(cls, array: ArrayLike, frame: Frame, *, copy: bool = True) -> Self:
        """Create batch from array with Cartesian coordinates of shape (..., 3).

        Args:
            array: Array-like with shape (..., 3)
            frame: Reference frame
            copy: If False, reuse the memory of `array` when possible

        Returns:
            New batch

        Examples:
            >>> cloud = Points.from_array(np.zeros((100, 3)), frame=world)
        """
        return cls(array, frame, copy=copy)

    @classmethod
    def from_primitives(cls, primitives: Sequence[Point] | Sequence[Vector]) -> Self:
        """Collect single primitives of one frame into a 1-D batch.

        Args:
            primitives: Non-empty sequence of Point (for Points) or
                Vector (for Vectors), all in the same frame

        Returns:
            New batch with shape (len(primitives),)

        Raises:
            ValueError: If the sequence is empty
            TypeError: If an element has the wrong type
            RuntimeError: If elements are in different frames

        Examples:
            >>> cloud = Points.from_primitives([world.point(0, 0, 0), p])
        """
        if len(primitives) == 0:
            raise ValueError(
                f"Cannot create {cls.__name__} from an empty sequence, the frame "
                "is unknown.\n"
                f"Use: {cls.__name__}.from_array(np.empty((0, 3)), frame=frame)"
            )
        element_type = cls._element_type
        wrong = [type(p).__name__ for p in primitives if type(p) is not element_type]
        if wrong:
            raise TypeError(
                f"{cls.__name__}.from_primitives expects only "
                f"{element_type.__name__} instances, got {sorted(set(wrong))}."
            )
        check_same_frame(*primitives)
        coords = np.stack([p._homogeneous[:3] for p in primitives])
        return cls._wrap(coords, primitives[0].frame)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the batch (leading dimensions, without coordinates)."""
        return self._coords.shape[:-1]

    @property
    def x(self) -> NDArray[np.floating]:
        """X coordinates as read-only view."""
        return self._readonly(self._coords[..., 0])

    @property
    def y(self) -> NDArray[np.floating]:
        """Y coordinates as read-only view."""
        return self._readonly(self._coords[..., 1])

    @property
    def z(self) -> NDArray[np.floating]:
        """Z coordinates as read-only view."""
        return self._readonly(self._coords[..., 2])

    @staticmethod
    def _readonly(view: NDArray[np.floating]) -> NDArray[np.floating]:
        view.flags.writeable = False
        return view

    def __len__(self) -> int:
        return self._coords.shape[0]

    def __array__(self, dtype=None, copy=None) -> NDArray:
        """Return coordinates of shape (..., 3), a view unless a copy is required.

        `np.asarray(batch)` returns a writable view (zero-cost opt-out),
        `np.array(batch)` returns a copy.
        """
        if dtype is not None and not np.issubdtype(dtype, np.floating):
            raise TypeError(
                f"Geometric primitives require floating-point dtype, got {dtype}.\n"
                "Use dtype=float, dtype=np.float32, or dtype=np.float64."
            )
        if copy:
            return np.array(self._coords, dtype=dtype, copy=True)
        return np.asarray(self._coords, dtype=dtype)

    def _storage_key(self, key: Any) -> tuple[Any, ...]:
        """Extend an element index so it never addresses the coordinate axis."""
        return (*(key if isinstance(key, tuple) else (key,)), slice(None))

    def __getitem__(self, key: Any) -> Self | GeometricPrimitive:
        """Index elements like numpy indexes the leading axes.

        Returns:
            Independent Point/Vector copy for a single element, otherwise a
            batch (a view for slices, a copy for masks and integer arrays)

        Examples:
            >>> cloud[0]          # Point (copy)
            >>> cloud[10:20]      # Points (view)
            >>> cloud[cloud.z > 0]  # Points (copy)
        """
        selected = self._coords[self._storage_key(key)]
        if selected.ndim == 1:
            homogeneous = np.append(selected, 1.0 if self._is_point else 0.0)
            return self._element_type._from_homogeneous(homogeneous, self.frame)
        return self._wrap(selected, self.frame)

    def __setitem__(self, key: Any, value: Geometric) -> None:
        """Assign elements from a primitive or batch of the same kind and frame.

        Raises:
            TypeError: If value is not of the same kind (point vs. vector)
            RuntimeError: If value is in a different frame
        """
        if not isinstance(value, GeometricPrimitive | PrimitiveArray) or (
            _is_point_like(value) != self._is_point
        ):
            name = type(self).__name__
            raise TypeError(
                f"Can only assign {self._element_type.__name__} or {name} to "
                f"{name}, got {type(value).__name__}.\n"
                "Use:\n"
                f"  batch[i] = frame.{self._element_type.__name__.lower()}(x, y, z)\n"
                "Or write raw coordinates through the array view:\n"
                "  np.asarray(batch)[i] = [x, y, z]"
            )
        if value.frame is not self.frame:
            check_same_frame(self, value)
        self._coords[self._storage_key(key)] = _coordinates(value)

    def __iter__(self) -> Iterator[Self | GeometricPrimitive]:
        """Iterate over the first axis (slow path, creates one object per item)."""
        return (self[index] for index in range(len(self)))

    def to_list(self) -> list[GeometricPrimitive]:
        """Return all elements as a flat list of independent Point/Vector objects.

        Examples:
            >>> points = cloud.to_list()  # list[Point]
        """
        flat = self._wrap(self._coords.reshape(-1, 3), self.frame)
        return list(flat)

    def copy(self) -> Self:
        """Create an independent copy in the same frame."""
        return self._wrap(self._coords.copy(), self.frame)

    def to_frame(self, target_frame: Frame) -> Self:
        """Return copy of batch transformed to a different reference frame.

        Args:
            target_frame: Target reference frame

        Returns:
            New batch of same type in target frame

        Examples:
            >>> cloud_world = camera.points(coords).to_frame(world)
        """
        matrix = self.frame._matrix_to(target_frame)
        coords = self._coords @ matrix[:3, :3].T
        if self._is_point:
            coords += matrix[:3, 3]
        return self._wrap(coords, target_frame)

    def to_global(self) -> Self:
        """Return copy of batch transformed to the root frame."""
        return self.to_frame(self.frame.root)

    def isclose(self, other: Geometric, atol: float = VVSMALL) -> NDArray[np.bool_]:
        """Element-wise comparison of global positions/directions.

        Args:
            other: Primitive or batch of the same kind (broadcastable shape)
            atol: Absolute tolerance per coordinate

        Returns:
            Boolean array with the broadcast batch shape

        Raises:
            TypeError: If kinds differ (point vs. vector)
            RuntimeError: If frames belong to different hierarchies

        Examples:
            >>> cloud.isclose(cloud.to_frame(camera)).all()  # True
        """
        if not isinstance(other, GeometricPrimitive | PrimitiveArray) or (
            _is_point_like(other) != self._is_point
        ):
            raise TypeError(
                f"Cannot compare {type(self).__name__} with {type(other).__name__}."
            )
        own = self.to_global()
        other_coords = _coordinates(other.to_frame(own.frame))
        return np.isclose(own._coords, other_coords, rtol=0, atol=atol).all(axis=-1)

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            f"'==' is ambiguous for {type(self).__name__}.\n"
            "Use:\n"
            "  batch.isclose(other)  # Element-wise comparison in global frame\n"
            "  batch.isclose(other).all()  # Single boolean\n"
            "Or compare raw coordinates:\n"
            "  np.asarray(batch) == np.asarray(other)"
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        prefix = f"{type(self).__name__}("
        body = np.array2string(
            self._coords, separator=", ", prefix=prefix, threshold=12, edgeitems=2
        )
        return f"{prefix}{body}, frame={self.frame.name})"

    def _operand(self, other: object) -> tuple[NDArray[np.floating], bool]:
        """Coordinates of a geometric operand and whether it is point-like."""
        if not isinstance(other, GeometricPrimitive | PrimitiveArray):
            raise TypeError(
                f"Cannot combine {type(self).__name__} with {type(other).__name__}.\n"
                "Wrap raw coordinates in a frame first:\n"
                "  batch + frame.vectors(array)  # Displace by vectors\n"
                "Or convert to array:\n"
                "  np.asarray(batch) + array  # Coordinate manipulation"
            )
        if other.frame is not self.frame:
            check_same_frame(self, other)
        return _coordinates(other), _is_point_like(other)

    def __add__(self, other: Geometric) -> Points | Vectors:
        other_coords, other_is_point = self._operand(other)
        if self._is_point and other_is_point:
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(other).__name__} "
                "(geometrically undefined).\n"
                "Use:\n"
                "  points + vectors  # Points (displace points)\n"
                "  points - points  # Vectors (displacements)"
            )
        result_type = Points if self._is_point or other_is_point else Vectors
        return result_type._wrap(self._coords + other_coords, self.frame)

    __radd__ = __add__

    def __sub__(self, other: Geometric) -> Points | Vectors:
        other_coords, other_is_point = self._operand(other)
        if other_is_point and not self._is_point:
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from {type(self).__name__} "
                "(geometrically undefined).\n"
                "Did you mean:\n"
                "  points - vectors  # Points (displace backwards)"
            )
        result_type = Vectors if self._is_point == other_is_point else Points
        return result_type._wrap(self._coords - other_coords, self.frame)

    def __rsub__(self, other: Geometric) -> Points | Vectors:
        other_coords, other_is_point = self._operand(other)
        if self._is_point and not other_is_point:
            raise TypeError(
                f"Cannot subtract {type(self).__name__} from {type(other).__name__} "
                "(geometrically undefined).\n"
                "Did you mean:\n"
                "  points - vectors  # Points (displace backwards)"
            )
        result_type = Vectors if self._is_point == other_is_point else Points
        return result_type._wrap(other_coords - self._coords, self.frame)

    def __rtruediv__(self, other: Any) -> None:
        raise TypeError(
            f"Division by {type(self).__name__} is undefined.\n"
            "Convert to array for coordinate manipulation:\n"
            "  other / np.asarray(batch)"
        )

    def _apply(self, func: Callable[..., Any], **kwargs: Any) -> Self:
        return self._wrap(func(self._coords, **kwargs), self.frame)

    def _apply_(self, func: Callable[..., Any], **kwargs: Any) -> Self:
        func(self._coords, out=self._coords, **kwargs)
        return self

    def round(self, decimals: int = 0) -> Self:
        """Return batch with coordinates rounded to given number of decimals."""
        return self._apply(np.round, decimals=decimals)

    def round_(self, decimals: int = 0) -> Self:
        """Round coordinates in-place. Returns self for method chaining."""
        return self._apply_(np.round, decimals=decimals)

    def clip(self, a_min: float = -np.inf, a_max: float = np.inf) -> Self:
        """Return batch with coordinates clipped to [a_min, a_max]."""
        return self._apply(np.clip, a_min=a_min, a_max=a_max)

    def clip_(self, a_min: float = -np.inf, a_max: float = np.inf) -> Self:
        """Clip coordinates in-place. Returns self for method chaining."""
        return self._apply_(np.clip, a_min=a_min, a_max=a_max)

    def floor(self) -> Self:
        """Return batch with coordinates rounded down."""
        return self._apply(np.floor)

    def floor_(self) -> Self:
        """Round coordinates down in-place. Returns self for method chaining."""
        return self._apply_(np.floor)

    def ceil(self) -> Self:
        """Return batch with coordinates rounded up."""
        return self._apply(np.ceil)

    def ceil_(self) -> Self:
        """Round coordinates up in-place. Returns self for method chaining."""
        return self._apply_(np.ceil)

    def trunc(self) -> Self:
        """Return batch with coordinates truncated toward zero."""
        return self._apply(np.trunc)

    def trunc_(self) -> Self:
        """Truncate coordinates in-place. Returns self for method chaining."""
        return self._apply_(np.trunc)

    def rint(self) -> Self:
        """Return batch with coordinates rounded to nearest integer."""
        return self._apply(np.rint)

    def rint_(self) -> Self:
        """Round coordinates to nearest integer in-place. Returns self."""
        return self._apply_(np.rint)

    def fix(self) -> Self:
        """Return batch with coordinates rounded toward zero."""
        return self._apply(np.fix)

    def fix_(self) -> Self:
        """Round coordinates toward zero in-place. Returns self for chaining."""
        return self._apply_(np.fix)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support only ufuncs with geometric meaning, see module docs."""
        name = type(self).__name__
        if method != "__call__" or "out" in kwargs:
            raise TypeError(
                f"ufunc '{ufunc.__name__}' with method '{method}' or out= is not "
                f"supported for {name}.\n"
                "Use in-place methods like batch.round_() or convert explicitly:\n"
                f"  np.{ufunc.__name__}(np.asarray(batch))"
            )

        if ufunc is np.negative:
            return -inputs[0]

        operators = {
            np.add: ("__add__", "__radd__"),
            np.subtract: ("__sub__", "__rsub__"),
            np.multiply: ("__mul__", "__rmul__"),
            np.divide: ("__truediv__", "__rtruediv__"),
        }
        if ufunc in operators and len(inputs) == 2:
            forward, reflected = operators[ufunc]
            left, right = inputs
            if isinstance(left, PrimitiveArray):
                return getattr(left, forward)(right)
            return getattr(right, reflected)(left)

        if ufunc in (np.floor, np.ceil, np.trunc, np.rint):
            return inputs[0]._apply(ufunc)

        if ufunc in (np.isnan, np.isinf, np.isfinite, np.signbit):
            return ufunc(inputs[0]._coords)

        if ufunc is np.absolute:
            if isinstance(inputs[0], Vectors):
                return inputs[0].magnitude
            raise TypeError(
                f"abs({name}) is geometrically undefined.\n"
                "Use np.abs(np.asarray(points)) for coordinate-wise absolute values."
            )

        if ufunc is np.vecdot and len(inputs) == 2 and not kwargs:
            left, right = inputs
            if isinstance(left, Vectors):
                return left.dot(right)
            return right.dot(left)

        raise TypeError(
            f"ufunc '{ufunc.__name__}' is not supported for {name}.\n"
            f"Convert explicitly: np.{ufunc.__name__}(np.asarray(batch))"
        )

    def __array_function__(self, func, types, args, kwargs):
        """Dispatch to registered implementations, see HANDLED_ARRAY_FUNCTIONS."""
        if not all(
            issubclass(t, PrimitiveArray | GeometricPrimitive | np.ndarray)
            for t in types
        ):
            return NotImplemented
        implementation = HANDLED_ARRAY_FUNCTIONS.get(func)
        if implementation is None:
            raise TypeError(
                f"np.{func.__name__} is not supported for {type(self).__name__}.\n"
                f"Convert explicitly: np.{func.__name__}(np.asarray(batch), ...)"
            )
        return implementation(*args, **kwargs)


class Points(PrimitiveArray):
    """Batch of points sharing one frame, stored as array of shape (..., 3).

    Arithmetic semantics (with Point/Points/Vector/Vectors, broadcasting):
        - Points - Points = Vectors
        - Points + Vectors = Points
        - Points - Vectors = Points
        - Points + Points = ERROR

    Examples:
        >>> cloud = camera.points(np.random.rand(1000, 3))
        >>> cloud_world = cloud.to_frame(world)
        >>> offsets = cloud - camera.origin  # Vectors
    """

    __slots__ = ()
    _element_type = Point
    _is_point = True

    def __mul__(self, other: Any) -> None:
        """Multiplication is undefined for points.

        Raises:
            TypeError: Always, with explanation of alternatives
        """
        raise TypeError(
            "Scalar multiplication of Points is undefined.\n"
            "Points can only be scaled relative to an origin.\n"
            "Use:\n"
            "  (points - origin) * scalar + origin  # Scale relative to origin\n"
            "Or convert to array:\n"
            "  np.asarray(points) * scalar  # Coordinate manipulation"
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> None:
        """Division is undefined for points.

        Raises:
            TypeError: Always, with explanation of alternatives
        """
        raise TypeError(
            "Scalar division of Points is undefined.\n"
            "Points can only be scaled relative to an origin.\n"
            "Use:\n"
            "  (points - origin) / scalar + origin  # Scale relative to origin\n"
            "Or convert to array:\n"
            "  np.asarray(points) / scalar  # Coordinate manipulation"
        )


class Vectors(PrimitiveArray):
    """Batch of vectors sharing one frame, stored as array of shape (..., 3).

    Arithmetic semantics (with Point/Points/Vector/Vectors, broadcasting):
        - Vectors + Vectors = Vectors
        - Vectors - Vectors = Vectors
        - Vectors + Points = Points
        - Vectors * factors = Vectors (scalar or one factor per element)
        - Vectors - Points = ERROR

    Examples:
        >>> directions = camera.vectors(np.random.rand(1000, 3)).normalize()
        >>> lengths = (cloud_b - cloud_a).magnitude
    """

    __slots__ = ()
    _element_type = Vector
    _is_point = False

    def _factors(self, other: Any, operation: str) -> NDArray[np.floating]:
        """Validate scale factors and shape them to broadcast per element."""
        if isinstance(other, GeometricPrimitive | PrimitiveArray):
            raise TypeError(
                f"Cannot {operation} Vectors by {type(other).__name__}.\n"
                "Use:\n"
                "  vectors.dot(other)  # Element-wise dot product\n"
                "  vectors.cross(other)  # Element-wise cross product"
            )
        factors = np.asarray(other)
        if factors.dtype.kind not in "iuf":
            raise TypeError(
                f"Can only {operation} Vectors by real numbers, got dtype "
                f"{factors.dtype}.\n"
                "Use np.asarray(vectors) for other operations."
            )
        try:
            broadcast_shape = np.broadcast_shapes(factors.shape, self.shape)
        except ValueError:
            broadcast_shape = None
        if broadcast_shape != self.shape:
            raise ValueError(
                f"Cannot {operation} Vectors of shape {self.shape} by factors of "
                f"shape {factors.shape}.\n"
                "Provide a scalar or factors broadcastable to the batch shape "
                "(one factor per element).\n"
                "For per-component scaling, convert to array:\n"
                "  frame.vectors(np.asarray(vectors) * [sx, sy, sz])"
            )
        return factors[..., np.newaxis]

    def __mul__(self, other: ArrayLike) -> Vectors:
        """Scale vectors by a scalar or one factor per element.

        Examples:
            >>> vectors * 2
            >>> vectors * lengths  # lengths.shape == vectors.shape
        """
        return self._wrap(self._coords * self._factors(other, "multiply"), self.frame)

    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> Vectors:
        """Divide vectors by a scalar or one factor per element.

        Raises:
            ZeroDivisionError: If any factor is zero
        """
        factors = self._factors(other, "divide")
        if np.any(factors == 0):
            raise ZeroDivisionError("Cannot divide Vectors by zero.")
        return self._wrap(self._coords / factors, self.frame)

    def __neg__(self) -> Vectors:
        """Negate all vectors."""
        return self._wrap(-self._coords, self.frame)

    @property
    def magnitude(self) -> NDArray[np.floating]:
        """Euclidean length of each vector, shape equals batch shape."""
        return np.linalg.norm(self._coords, axis=-1)

    @property
    def is_zero(self) -> NDArray[np.bool_]:
        """Mask of vectors with near-zero magnitude."""
        return self.magnitude < VSMALL

    def normalize(self) -> Self:
        """Normalize all vectors to unit length in-place.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If any vector has zero length

        Examples:
            >>> directions = (targets - origin).normalize()
        """
        magnitude = self.magnitude
        zero_count = int(np.count_nonzero(magnitude < VSMALL))
        if zero_count:
            raise RuntimeError(
                f"Cannot normalize {zero_count} zero-length vector(s).\n"
                "Filter them first:\n"
                "  nonzero = vectors[~vectors.is_zero].normalize()"
            )
        self._coords /= magnitude[..., np.newaxis]
        return self

    def _vector_operand(self, other: object, operation: str) -> NDArray[np.floating]:
        if not isinstance(other, Vector | Vectors):
            raise TypeError(
                f"{operation} requires Vector or Vectors, got {type(other).__name__}."
            )
        if other.frame is not self.frame:
            check_same_frame(self, other)
        return _coordinates(other)

    def dot(self, other: Vector | Vectors) -> NDArray[np.floating]:
        """Element-wise dot product (broadcasting).

        Examples:
            >>> cos_angles = directions.dot(frame.z_axis)
        """
        return np.vecdot(self._coords, self._vector_operand(other, "dot"))

    def cross(self, other: Vector | Vectors) -> Vectors:
        """Element-wise cross product (broadcasting).

        Examples:
            >>> normals = edges_a.cross(edges_b)
        """
        other_coords = self._vector_operand(other, "cross")
        return self._wrap(np.cross(self._coords, other_coords), self.frame)


def _reject_out(name: str, out: Any) -> None:
    if out is not None:
        raise TypeError(
            f"out parameter not supported for {name}.\n"
            "Use the in-place methods instead, e.g. batch.round_()."
        )


@implements(np.round)
def _round(a: PrimitiveArray, decimals: int = 0, out: None = None) -> PrimitiveArray:
    _reject_out("np.round", out)
    return a.round(decimals=decimals)


@implements(np.clip)
def _clip(
    a: PrimitiveArray,
    a_min: float | None = None,
    a_max: float | None = None,
    out: None = None,
    *,
    min: float | None = None,  # noqa: A002 - mirrors numpy signature
    max: float | None = None,  # noqa: A002 - mirrors numpy signature
) -> PrimitiveArray:
    _reject_out("np.clip", out)
    lower = a_min if a_min is not None else min
    upper = a_max if a_max is not None else max
    return a.clip(
        a_min=-np.inf if lower is None else lower,
        a_max=np.inf if upper is None else upper,
    )


@implements(np.fix)
def _fix(a: PrimitiveArray, out: None = None) -> PrimitiveArray:
    _reject_out("np.fix", out)
    return a.fix()


@implements(np.copy)
def _copy(a: PrimitiveArray, order: str = "K", subok: bool = False) -> PrimitiveArray:
    return a.copy()


@implements(np.linalg.norm)
def _norm(
    x: PrimitiveArray, ord: Any = None, axis: Any = None, keepdims: bool = False
) -> NDArray[np.floating]:
    if not isinstance(x, Vectors):
        raise TypeError(
            "np.linalg.norm is only defined for Vectors.\n"
            "For distances between points use (points - origin).magnitude"
        )
    if ord not in (None, 2) or axis not in (None, -1) or keepdims:
        raise TypeError(
            "np.linalg.norm(vectors) returns the magnitude of each vector; "
            "ord, axis and keepdims are not supported.\n"
            "Convert explicitly: np.linalg.norm(np.asarray(vectors), ...)"
        )
    return x.magnitude


@implements(np.cross)
def _cross(a: Vector | Vectors, b: Vector | Vectors, **kwargs: Any) -> Vectors:
    if kwargs or not isinstance(a, Vector | Vectors):
        raise TypeError(
            "np.cross on hazy objects requires two Vector/Vectors without "
            "axis arguments.\nUse: vectors.cross(other)"
        )
    other_coords = Vectors._vector_operand(a, b, "cross")
    return Vectors._wrap(np.cross(_coordinates(a), other_coords), a.frame)


@implements(np.dot)
def _dot(*args: Any, **kwargs: Any) -> None:
    raise TypeError(
        "np.dot is ambiguous for batches (matrix product vs. element-wise).\n"
        "Use:\n"
        "  vectors.dot(other)  # Element-wise dot product\n"
        "  np.vecdot(vectors, other)  # Same, numpy style"
    )


def _same_kind_batches(arrays: Sequence[Any], function: str) -> type[PrimitiveArray]:
    batch_types = {type(array) for array in arrays}
    if len(batch_types) != 1 or not issubclass(
        batch_type := batch_types.pop(), PrimitiveArray
    ):
        raise TypeError(
            f"np.{function} requires batches of a single kind (all Points or all "
            f"Vectors), got {sorted(t.__name__ for t in {type(a) for a in arrays})}.\n"
            "Collect single primitives first:\n"
            "  Points.from_primitives([p1, p2])"
        )
    check_same_frame(*arrays)
    return batch_type


@implements(np.concatenate)
def _concatenate(arrays: Sequence[PrimitiveArray], axis: int | None = 0, **kwargs):
    batch_type = _same_kind_batches(arrays, "concatenate")
    if axis is None:
        coords = [array._coords.reshape(-1, 3) for array in arrays]
        axis = 0
    else:
        coords = [array._coords for array in arrays]
        axis = _leading_axis(axis)
    return batch_type._wrap(
        np.concatenate(coords, axis=axis, **kwargs), arrays[0].frame
    )


@implements(np.stack)
def _stack(arrays: Sequence[PrimitiveArray], axis: int = 0, **kwargs):
    batch_type = _same_kind_batches(arrays, "stack")
    coords = [array._coords for array in arrays]
    return batch_type._wrap(
        np.stack(coords, axis=_leading_axis(axis), **kwargs), arrays[0].frame
    )
