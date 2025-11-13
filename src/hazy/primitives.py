"""Geometric primitives with frame awareness for coordinate system transformations.

This module provides Point and Vector classes that carry frame information and
support arithmetic operations with proper type semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self, TypeVar, overload

import numpy as np

from hazy.constants import VSMALL, VVSMALL
from hazy.utils import all_same_type, check_same_frame

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from hazy import Frame


F = TypeVar("F", bound=Callable[..., Any])
HANDLED_ARRAY_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(numpy_function: Callable[..., Any]) -> Callable[[F], F]:
    """Register a function as implementation for a NumPy array function."""

    def decorator(func: F) -> F:
        HANDLED_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


class GeometricPrimitive:
    """Base class for geometric primitives (Point and Vector) with frame awareness.

    Uses homogeneous coordinates (x, y, z, w) for unified transformation handling.
    Points have w=1, Vectors have w=0.
    """

    def __init__(self, x: float, y: float, z: float, w: float, frame: Frame):
        """Initialize geometric primitive in homogeneous coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            w: Homogeneous coordinate (1 for Point, 0 for Vector)
            frame: Reference frame for this primitive
        """
        self._homogeneous = np.array([x, y, z, w], dtype=float)
        self.frame = frame

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return Cartesian coordinates for numpy operations.

        Enables usage like: np.array(point) or np.add(point1, point2)

        Args:
            dtype: Desired array data type
            copy: Whether to copy the data

        Returns:
            Numpy array of Cartesian coordinates
        """
        coords = self._homogeneous[:3]
        if dtype is not None:
            coords = coords.astype(dtype)
        return coords if copy is False else coords.copy()

    @property
    def x(self) -> float:
        """X coordinate."""
        return self._homogeneous[0]

    @property
    def y(self) -> float:
        """Y coordinate."""
        return self._homogeneous[1]

    @property
    def z(self) -> float:
        """Z coordinate."""
        return self._homogeneous[2]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, GeometricPrimitive):
            self_global = self.to_global()
            value_global = value.to_global()
            return np.allclose(
                self_global._homogeneous, value_global._homogeneous, atol=VVSMALL
            )
        else:
            raise ValueError(
                f"Can not compare {self} of type {self.__class__.__qualname__} "
                f"with onject of type {type(value)}"
            )

    def __getitem__(self, index: int) -> float:
        """Access coordinates by index: primitive[0] for x, primitive[1] for y, etc."""
        return np.array(self)[index]

    def __iter__(self):
        """Iterate over Cartesian coordinates."""
        return iter(np.array(self))

    def to_frame(self, target_frame: Frame) -> Self:
        """Transform this primitive to a different reference frame.

        Args:
            target_frame: Target reference frame

        Returns:
            New primitive of same type in target frame
        """
        transformation = self.frame.transform_to(target=target_frame)
        x, y, z, w = transformation @ self._homogeneous

        if isinstance(self, Point):
            # Points (w=1): normalize by w after transformation
            return type(self)(x=x / w, y=y / w, z=z / w, w=1.0, frame=target_frame)
        else:
            # Vectors (w=0): do not normalize, w stays 0
            return type(self)(x=x, y=y, z=z, w=0.0, frame=target_frame)

    def to_global(self) -> Self:
        """Transform to the root frame of this primitive's hierarchy.

        For frames with parents, this transforms to the top-most parent.
        For orphan frames, this returns coordinates in the orphan frame itself.

        Returns:
            Primitive in root frame coordinates
        """
        return self.to_frame(target_frame=self.frame.root)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"x={self.x}, "
            f"y={self.y}, "
            f"z={self.z}, "
            f"frame={self.frame.name})"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain type consistency.

        This prevents numpy from converting our custom types to plain arrays
        when using operators like *, +, -, etc.
        """

        # For binary operations, handle scalar multiplication/division for Vectors only
        if method == "__call__" and len(inputs) == 2:
            # Identify which input is the geometric primitive
            geom_idx = 0 if isinstance(inputs[0], GeometricPrimitive) else 1
            other_idx = 1 - geom_idx
            geom_primitive = inputs[geom_idx]
            other = inputs[other_idx]

            # Handle multiplication/division with scalars (only for Vector)
            if (
                ufunc in (np.multiply, np.divide, np.true_divide)
                and np.isscalar(other)
                and isinstance(geom_primitive, Vector)
            ):
                if ufunc == np.multiply:
                    return geom_primitive.__mul__(other)
                elif ufunc in (np.divide, np.true_divide) and geom_idx == 0:
                    return geom_primitive.__truediv__(other)

        # delegate arithmetic to __dunder__ methods
        if ufunc in (
            np.add,
            np.subtract,
            np.multiply,
            np.divide,
            np.true_divide,
            np.negative,
        ):
            return NotImplemented

        # delegate comparison to __dunder__ methods
        if ufunc in (
            np.equal,
            np.not_equal,
            np.less,
            np.less_equal,
            np.greater,
            np.greater_equal,
        ):
            return NotImplemented

        # Rounding operations - preserve geometric type
        if (
            ufunc in (np.floor, np.ceil, np.trunc, np.rint, np.fix)
            and method == "__call__"
        ):
            coords = getattr(ufunc, method)(np.array(inputs[0]), **kwargs)
            result = inputs[0].copy()
            result._homogeneous[:3] = coords
            return result

        # Type queries - return boolean arrays
        if (
            ufunc in (np.isnan, np.isinf, np.isfinite, np.signbit)
            and method == "__call__"
        ):
            return getattr(ufunc, method)(np.array(inputs[0]), **kwargs)

        # Absolute value - special handling
        if ufunc == np.absolute and method == "__call__":
            if isinstance(inputs[0], Vector):
                return inputs[0].magnitude
            raise TypeError(
                f"abs({inputs[0].__class__.__qualname__}) is geometrically undefined. "
                "Use np.abs(np.array(point)) for coordinate-wise absolute values."
            )

        # Everything else explicit error:
        # For other ufuncs, convert to array and return array result
        # This handles operations where maintaining custom type doesn't make sense
        raise TypeError(
            f"ufunc {ufunc.__name__} not supported for {self.__class__.__qualname__}.\n"
            f"Convert explicitly: np.{ufunc.__name__}(np.array(obj))"
        )

    def __array_function__(self, func, types, inputs, kwargs):
        if func not in HANDLED_ARRAY_FUNCTIONS:
            return NotImplemented

        if not all(
            issubclass(t, (GeometricPrimitive, np.ndarray)) or t is type(None)
            for t in types
        ):
            return NotImplemented

        return HANDLED_ARRAY_FUNCTIONS[func](*inputs, **kwargs)

    def copy(self) -> Self:
        """Create a copy of this geometric primitive in the same frame."""
        return type(self)(
            self.x, self.y, self.z, frame=self.frame, w=self._homogeneous[3]
        )

    def round(self, decimals: int = 0) -> Self:
        out = self.copy()
        out._homogeneous[:3] = np.round(self._homogeneous[:3], decimals=decimals)
        return out

    def round_(self, decimals: int = 0) -> Self:
        self._homogeneous[:3] = np.round(self._homogeneous[:3], decimals=decimals)
        return self

    def clip(self, a_min: float = -np.inf, a_max: float = np.inf) -> Self:
        out = self.copy()
        out._homogeneous[:3] = np.clip(self._homogeneous[:3], a_min=a_min, a_max=a_max)
        return out

    def clip_(self, a_min: float = -np.inf, a_max: float = np.inf) -> Self:
        self._homogeneous[:3] = np.clip(self._homogeneous[:3], a_min=a_min, a_max=a_max)
        return self

    def floor(self) -> Self:
        copy = self.copy()
        copy._homogeneous[:3] = np.floor(copy._homogeneous[:3])
        return copy

    def floor_(self) -> Self:
        self._homogeneous[:3] = np.floor(self._homogeneous[:3])
        return self

    def fix(self) -> Self:
        copy = self.copy()
        copy._homogeneous[:3] = np.fix(copy._homogeneous[:3])
        return copy

    def fix_(self) -> Self:
        self._homogeneous[:3] = np.fix(self._homogeneous[:3])
        return self

    def ceil(self) -> Self:
        copy = self.copy()
        copy._homogeneous[:3] = np.ceil(copy._homogeneous[:3])
        return copy

    def ceil_(self) -> Self:
        self._homogeneous[:3] = np.ceil(self._homogeneous[:3])
        return self

    def trunc(self) -> Self:
        copy = self.copy()
        copy._homogeneous[:3] = np.trunc(copy._homogeneous[:3])
        return copy

    def trunc_(self) -> Self:
        self._homogeneous[:3] = np.trunc(self._homogeneous[:3])
        return self

    def rint(self) -> Self:
        copy = self.copy()
        copy._homogeneous[:3] = np.rint(copy._homogeneous[:3])
        return copy

    def rint_(self) -> Self:
        self._homogeneous[:3] = np.rint(self._homogeneous[:3])
        return self

    @classmethod
    def from_array(cls, array: ArrayLike, frame: Frame) -> Self:
        array = np.asarray(array).flatten()
        if array.shape != (4,):
            raise ValueError(f"Expected 4 coordinates, got {array.shape}")
        return cls(x=array[0], y=array[1], z=array[2], w=array[3], frame=frame)


class Vector(GeometricPrimitive):
    """Geometric vector representing direction and magnitude.

    Vectors have homogeneous coordinate w=0, making them invariant to translation.

    Arithmetic semantics:
        - Vector + Vector = Vector (combine displacements)
        - Vector - Vector = Vector (difference of displacements)
        - Vector + Point = Point (displace position)
        - Vector - Point = ERROR (undefined operation)
    """

    def __init__(self, x: float, y: float, z: float, frame: Frame, *, w=0.0):
        """Initialize vector in given frame.

        Args:
            x: X component
            y: Y component
            z: Z component
            frame: Reference frame
            w: Homogeneous coordinate (should be 0 for vectors)
        """
        super().__init__(x=x, y=y, z=z, w=w, frame=frame)

    @overload
    def __add__(self, other: Point) -> Point: ...

    @overload
    def __add__(self, other: Vector) -> Vector: ...

    @overload
    def __add__(self, other: NDArray[np.floating]) -> NDArray[np.floating]: ...

    def __add__(
        self, other: Point | Vector | NDArray[np.floating]
    ) -> Point | Vector | NDArray[np.floating]:
        """Add vector to another vector or point.

        Args:
            other: Vector, Point, or numpy array

        Returns:
            Vector if adding to Vector, Point if adding to Point

        Raises:
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            check_same_frame(self, other)
            x, y, z = np.array(self) + np.array(other)
            return Point(x, y, z, frame=self.frame)
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.array(self) + np.array(other)
            return Vector(x, y, z, frame=self.frame)
        else:
            return other.__add__(self)

    @overload
    def __sub__(self, other: Vector) -> Vector: ...

    @overload
    def __sub__(self, other: NDArray[np.floating]) -> NDArray[np.floating]: ...

    def __sub__(
        self, other: Vector | NDArray[np.floating]
    ) -> Vector | NDArray[np.floating]:
        """Subtract vector from this vector.

        Args:
            other: Vector or numpy array

        Returns:
            Resulting vector

        Raises:
            TypeError: If attempting to subtract Point from Vector
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            raise TypeError("Cannot subtract Point from Vector")
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.array(self) - np.array(other)
            return Vector(x, y, z, frame=self.frame)
        else:
            return other.__rsub__(self)

    def __mul__(self, other: float | np.generic) -> Vector:
        if not np.isscalar(other):
            raise TypeError(
                f"Can only multiply with Vector by scalar, got {type(other)}"
            )
        if isinstance(other, complex | np.complexfloating):
            raise TypeError(
                f"unsupported operand __mul__ between {self.__class__.__qualname__} "
                "and complex numbers"
            )
        copy = self.copy()
        copy._homogeneous[:3] *= other
        return copy

    def __rmul__(self, other: float | np.generic) -> Vector:
        return self.__mul__(other)

    def __truediv__(self, other: float | np.generic) -> Vector:
        """Divide vector by scalar."""
        if not np.isscalar(other):
            raise TypeError(f"Can only divide Vector by scalar, got {type(other)}")
        if other == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        copy = self.copy()
        copy._homogeneous[:3] /= other
        return copy

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self._homogeneous[:3]))

    @property
    def is_zero(self) -> bool:
        return self.magnitude < VSMALL

    def normalize(self) -> Self:
        if self.is_zero:
            raise RuntimeError(f"Can not normalize vector with zero length {self}")
        self._homogeneous[:3] /= self.magnitude
        return self

    @classmethod
    def unit_x(cls, frame: Frame) -> Vector:
        return cls(x=1.0, y=0.0, z=0.0, frame=frame)

    @classmethod
    def unit_y(cls, frame: Frame) -> Vector:
        return cls(x=0.0, y=1.0, z=0.0, frame=frame)

    @classmethod
    def unit_z(cls, frame: Frame) -> Vector:
        return cls(x=0.0, y=0.0, z=1.0, frame=frame)

    @classmethod
    def nan(cls, frame: Frame) -> Vector:
        """Return a point at the origin of the specified coordinate system"""
        return cls(x=np.nan, y=np.nan, z=np.nan, frame=frame)

    def __neg__(self) -> Vector:
        """Negate the vector, inverting its direction.

        Returns:
            Vector with inverted x, y, z components in the same frame
        """
        return Vector(-self.x, -self.y, -self.z, frame=self.frame)

    def cross(self, other: Vector) -> Vector:
        """Compute cross product with another vector.

        Args:
            other: Vector to cross with

        Returns:
            Vector perpendicular to both input vectors

        Raises:
            RuntimeError: If frames don't match
        """
        check_same_frame(self, other)
        x, y, z = np.cross(np.array(self), np.array(other))
        return Vector(x, y, z, frame=self.frame)

    def dot(self, other: Vector) -> float:
        """Compute dot product with another vector.

        Args:
            other: Vector to dot with

        Returns:
            scalar dot product between both vectors

        Raises:
            RuntimeError: If frames don't match
        """
        check_same_frame(self, other)
        return np.dot(np.array(self), np.array(other))

    @classmethod
    def from_array(cls, array: ArrayLike, frame: Frame) -> Self:
        array = np.asarray(array).flatten()
        if array.shape != (3,):
            raise ValueError(f"Expected 3 coordinates, got {array.shape}")
        return cls(x=array[0], y=array[1], z=array[2], w=0.0, frame=frame)


class Point(GeometricPrimitive):
    """Geometric point representing position in space.

    Points have homogeneous coordinate w=1, making them affected by translation.

    Arithmetic semantics:
        - Point - Point = Vector (displacement between positions)
        - Point + Vector = Point (displace position)
        - Point - Vector = Point (displace position backwards)
        - Point + Point = ERROR (undefined operation)
    """

    def __init__(self, x: float, y: float, z: float, frame: Frame, *, w=1.0):
        """Initialize point in given frame.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            frame: Reference frame
            w: Homogeneous coordinate (should be 1 for points)
        """
        super().__init__(x=x, y=y, z=z, w=w, frame=frame)

    @overload
    def __sub__(self, other: Point) -> Vector: ...

    @overload
    def __sub__(self, other: Vector) -> Point: ...

    @overload
    def __sub__(self, other: NDArray[np.floating]) -> NDArray[np.floating]: ...

    def __sub__(
        self, other: Point | Vector | NDArray[np.floating]
    ) -> Point | Vector | NDArray[np.floating]:
        """Subtract point or vector from this point.

        Args:
            other: Point, Vector, or numpy array

        Returns:
            Vector if subtracting Point, Point if subtracting Vector

        Raises:
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            check_same_frame(self, other)
            x, y, z = np.array(self) - np.array(other)
            return Vector(x, y, z, frame=self.frame)
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.array(self) - np.array(other)
            return Point(x, y, z, frame=self.frame)
        else:
            return other.__rsub__(self)

    @overload
    def __add__(self, other: Vector) -> Point: ...

    @overload
    def __add__(self, other: NDArray[np.floating]) -> NDArray[np.floating]: ...

    def __add__(
        self, other: Vector | NDArray[np.floating]
    ) -> Point | NDArray[np.floating]:
        """Add vector to this point.

        Args:
            other: Vector or numpy array

        Returns:
            Resulting point

        Raises:
            TypeError: If attempting to add two Points
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            raise TypeError("Can not add 2 Points.")
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.array(self) + np.array(other)
            return Point(x, y, z, frame=self.frame)
        else:
            return other.__add__(self)

    def __mul__(self, other):
        raise TypeError(
            f"Scalar multiplication of {self.__class__.__qualname__} is undefined. "
            "Points can only be scaled relative to an origin. Use:\n"
            "  (point - origin) * scalar + origin  # Scale relative to origin\n"
            "Or convert to array:\n"
            "  np.array(point) * scalar  # Coordinate manipulation"
        )

    def __truediv__(self, other):
        raise TypeError(
            f"Scalar division of {self.__class__.__qualname__} is undefined. "
            "Points can only be scaled relative to an origin. Use:\n"
            "  (point - origin) * scalar + origin  # Scale relative to origin\n"
            "Or convert to array:\n"
            "  np.array(point) * scalar  # Coordinate manipulation"
        )

    @classmethod
    def create_origin(cls, frame: Frame) -> Point:
        """Return a point at the origin of the specified coordinate system"""
        return cls(x=0.0, y=0.0, z=0.0, frame=frame)

    @classmethod
    def create_nan(cls, frame: Frame) -> Point:
        """Return a point at the origin of the specified coordinate system"""
        return cls(x=np.nan, y=np.nan, z=np.nan, frame=frame)

    @classmethod
    def from_array(cls, array: ArrayLike, frame: Frame) -> Self:
        array = np.asarray(array).flatten()
        if array.shape != (3,):
            raise ValueError(f"Expected 3 coordinates, got {array.shape}")
        return cls(x=array[0], y=array[1], z=array[2], w=1.0, frame=frame)

    @classmethod
    def list_from_array(cls, points: ArrayLike, frame) -> list[Point]:
        """Creates a list of Point instances from an array of points."""
        points = np.asarray(points).reshape((-1, 3))
        return [cls.from_array(arr, frame=frame) for arr in points]


@implements(np.copy)
def _copy_geometric(
    a: GeometricPrimitive,
    order: str = "K",  # Ignored for geometric primitives
    subok: bool = False,  # Ignored for geometric primitives
) -> GeometricPrimitive:
    return a.copy()


@implements(np.asarray)
def _asarray_geometric(
    a: GeometricPrimitive, dtype=None, order=None, *, like=None
) -> np.ndarray:
    """Convert geometric primitive to array (returns coordinates)."""
    if dtype is not None and not np.issubdtype(dtype, np.floating):
        raise TypeError("Geometric primitives require floating-point dtype")
    return a.__array__(dtype=dtype)


@implements(np.round)
def _round_geometric[T: GeometricPrimitive](
    a: T, decimals: int = 0, out: None = None
) -> T:
    if out is not None:
        raise TypeError(
            "out parameter not supported for GeometricPrimitive. "
            "Use .round_() for in-place modification."
        )
    return a.round(decimals=decimals)


@implements(np.clip)
def _clip_geometric[T: GeometricPrimitive](
    a: T,
    a_min: float = -np.inf,
    a_max: float = np.inf,
    out: None = None,
) -> T:
    if out is not None:
        raise TypeError(
            "out parameter not supported for GeometricPrimitive. "
            "Use .clip_() for in-place modification."
        )
    return a.clip(a_max=a_max, a_min=a_min)


@implements(np.floor)
def _floor_geometric[T: GeometricPrimitive](a: T, out=None) -> T:
    return a.floor()


@implements(np.ceil)
def _ceil_geometric[T: GeometricPrimitive](a: T, out=None) -> T:
    return a.ceil()


@implements(np.trunc)
def _trunc_geometric[T: GeometricPrimitive](a: T, out=None) -> T:
    return a.trunc()


@implements(np.rint)
def _rint_geometric[T: GeometricPrimitive](a: T, out=None) -> T:
    return a.rint()


@implements(np.fix)
def _fix_geometric[T: GeometricPrimitive](a: T, out=None) -> T:
    return a.fix()


@implements(np.isnan)
def _isnan_geometric(a: GeometricPrimitive) -> NDArray[np.bool_]:
    return np.isnan(a._homogeneous[:3])


@implements(np.isinf)
def _isinf_geometric(a: GeometricPrimitive) -> NDArray[np.bool_]:
    return np.isinf(a._homogeneous[:3])


@implements(np.isfinite)
def _isfinite_geometric(a: GeometricPrimitive) -> NDArray[np.bool_]:
    return np.isfinite(a._homogeneous[:3])


@implements(np.absolute)
def _absolute_geometric(a: GeometricPrimitive) -> float:
    """Absolute value: magnitude for Vector, error for Point."""
    if isinstance(a, Vector):
        return a.magnitude
    raise TypeError(
        f"abs({a.__class__.__qualname__}) is geometrically undefined. "
        "Use np.abs(np.array(point)) for coordinate-wise absolute values."
    )


@implements(np.cross)
def _cross_geometric(a: Vector, b: Vector, **kwargs) -> Vector:
    """Cross product for Vectors (frame-aware).

    Args:
        a: First vector (must be Vector for frame-aware operation)
        b: Second vector (must be Vector and same frame as a)
        **kwargs: Additional arguments passed to np.cross on arrays

    Returns:
        Vector perpendicular to both inputs (if both are Vectors)
        ndarray otherwise

    Raises:
        TypeError: If a is not a Vector
        RuntimeError: If frames don't match
    """
    if not isinstance(a, Vector) or not isinstance(b, Vector):
        raise TypeError(
            f"np.cross requires both Vector arguments, "
            f"got {type(a).__name__} and {type(b).__name__}"
        )
    return a.cross(b)


@implements(np.dot)
def _dot_geometric(a: Vector, b: Vector, out=None) -> float:
    """Dot product for Vectors (frame-aware).

    Args:
        a: First vector (must be Vector for frame-aware operation)
        b: Second vector (must be Vector and same frame as a)
        out: Output array (not supported for GeometricPrimitives)

    Returns:
        Scalar dot product (if both are Vectors)
        ndarray otherwise

    Raises:
        TypeError: If a is not a Vector or out is specified
        RuntimeError: If frames don't match
    """
    if out is not None:
        raise TypeError(
            "out parameter not supported for GeometricPrimitive dot product"
        )

    if not isinstance(a, Vector) or not isinstance(b, Vector):
        raise TypeError(
            f"np.dot requires both Vector arguments, "
            f"got {type(a).__name__} and {type(b).__name__}"
        )
    return a.dot(b)


@implements(np.linalg.norm)
def _norm_geometric(
    x: GeometricPrimitive,
    ord=None,
    axis=None,
    keepdims=False,
) -> float:
    """Compute norm for geometric primitives.

    Args:
        x: Vector to compute norm of
        ord: Order of the norm (only default supported)
        axis: Axis parameter (not supported for GeometricPrimitives)
        keepdims: Keep dimensions parameter (not supported for GeometricPrimitives)

    Returns:
        Magnitude of the vector

    Raises:
        TypeError: If x is not a Vector or unsupported parameters are used
    """
    if not isinstance(x, Vector):
        raise TypeError(
            f"np.linalg.norm for {x.__class__.__qualname__} is undefined. "
            "Use np.linalg.norm(np.array(obj)) for coordinate-wise norm."
        )

    if ord is not None:
        raise TypeError(
            "ord parameter not supported for Vector. "
            "Use np.linalg.norm(np.array(vector), ord=...) for custom norms."
        )

    if axis is not None:
        raise TypeError("axis parameter not supported for Vector.")

    if keepdims:
        raise TypeError("keepdims parameter not supported for Vector.")

    return x.magnitude


@implements(np.stack)
def _stack_geometric(
    arrays: list[GeometricPrimitive] | tuple[GeometricPrimitive, ...],
    axis: int = 0,
    out=None,
    **kwargs,
) -> NDArray[np.floating]:
    """Stack geometric primitives into array.

    All primitives must be of same type and in the same frame.

    Args:
        arrays: Sequence of Points or Vectors
        axis: Axis along which to stack (default: 0)
        out: Output array (not supported)
        **kwargs: Additional arguments passed to np.stack

    Returns:
        Stacked array of coordinates with shape determined by axis

    Raises:
        TypeError: If out is specified or items are mixed types
        RuntimeError: If items are in different frames

    Examples:
        >>> v1, v2, v3 = [frame.vector(i, i+1, i+2) for i in range(3)]
        >>> np.stack([v1, v2, v3])  # (3, 3) array
        >>> np.stack([v1, v2, v3], axis=1)  # (3, 3) array
    """
    if out is not None:
        raise TypeError("out parameter not supported for GeometricPrimitive stacking")

    if not arrays:
        raise ValueError("Cannot stack empty sequence")

    arrays_list = list(arrays)

    if not all_same_type(arrays_list):
        types = {type(a).__name__ for a in arrays_list}
        raise TypeError(f"All items must be same type, got: {types}")

    check_same_frame(*arrays_list)

    coord_arrays = [np.array(a) for a in arrays_list]
    return np.stack(coord_arrays, axis=axis, **kwargs)


@implements(np.vstack)
def _vstack_geometric(
    tup: list[GeometricPrimitive] | tuple[GeometricPrimitive, ...],
    **kwargs,
) -> NDArray[np.floating]:
    """Vertically stack geometric primitives.

    All primitives must be of same type and in the same frame.
    Equivalent to np.stack(arrays, axis=0).

    Args:
        tup: Sequence of Points or Vectors
        **kwargs: Additional arguments passed to np.vstack

    Returns:
        Vertically stacked array of coordinates (N, 3)

    Raises:
        TypeError: If items are mixed types
        RuntimeError: If items are in different frames

    Examples:
        >>> p1, p2, p3 = [frame.point(i, i+1, i+2) for i in range(3)]
        >>> np.vstack([p1, p2, p3])  # (3, 3) array
    """
    if not tup:
        raise ValueError("Cannot vstack empty sequence")

    tup_list = list(tup)

    if not all_same_type(tup_list):
        types = {type(t).__name__ for t in tup_list}
        raise TypeError(f"All items must be same type, got: {types}")

    check_same_frame(*tup_list)

    coord_arrays = [np.array(t) for t in tup_list]
    return np.vstack(coord_arrays, **kwargs)


@implements(np.hstack)
def _hstack_geometric(
    tup: list[GeometricPrimitive] | tuple[GeometricPrimitive, ...],
    **kwargs,
) -> NDArray[np.floating]:
    """Horizontally stack geometric primitives.

    All primitives must be of same type and in the same frame.
    Creates array with shape (3, N) where N is number of primitives.

    Args:
        tup: Sequence of Points or Vectors
        **kwargs: Additional arguments passed to np.hstack

    Returns:
        Horizontally stacked array of coordinates (3, N)

    Raises:
        TypeError: If items are mixed types
        RuntimeError: If items are in different frames

    Examples:
        >>> v1, v2, v3 = [frame.vector(i, i+1, i+2) for i in range(3)]
        >>> np.hstack([v1, v2, v3])  # (3, 3) array - coordinates as columns
    """
    if not tup:
        raise ValueError("Cannot hstack empty sequence")

    tup_list = list(tup)

    if not all_same_type(tup_list):
        types = {type(t).__name__ for t in tup_list}
        raise TypeError(f"All items must be same type, got: {types}")

    check_same_frame(*tup_list)

    coord_arrays = [np.array(t) for t in tup_list]
    return np.hstack(coord_arrays, **kwargs)
