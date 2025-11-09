"""Geometric primitives with frame awareness for coordinate system transformations.

This module provides Point and Vector classes that carry frame information and
support arithmetic operations with proper type semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, overload

import numpy as np

from hazy.constants import VSMALL, VVSMALL
from hazy.utils import check_same_frame

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from hazy import Frame


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

    @property
    def coords(self) -> NDArray[np.floating]:
        """Return Cartesian coordinates (x, y, z) as numpy array."""
        return self._homogeneous[:3]

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
        return self.coords[index]

    def __iter__(self):
        """Iterate over Cartesian coordinates."""
        return iter(self.coords)

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

    def __mul__(self, other: ArrayLike):
        copy = self.copy()
        copy._homogeneous[:3] = copy.coords * np.asarray(other)
        return copy

    def __rmul__(self, other: ArrayLike):
        copy = self.copy()
        copy._homogeneous[:3] = copy.coords * np.asarray(other)
        return copy

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain type consistency.

        This prevents numpy from converting our custom types to plain arrays
        when using operators like *, +, -, etc.
        """
        if ufunc == np.multiply and method == "__call__":
            # Handle scalar * Vector or Vector * scalar
            if isinstance(inputs[0], (int, float, np.number)):
                return inputs[1].__rmul__(inputs[0])
            elif isinstance(inputs[1], (int, float, np.number)):
                return inputs[0].__mul__(inputs[1])

        # For other ufuncs, convert to array and return array result
        # This handles operations where maintaining custom type doesn't make sense
        args = [x.coords if isinstance(x, GeometricPrimitive) else x for x in inputs]
        return getattr(ufunc, method)(*args, **kwargs)

    def copy(self) -> Self:
        """Create a copy of this geometric primitive in the same frame."""
        return type(self)(
            self.x, self.y, self.z, frame=self.frame, w=self._homogeneous[3]
        )


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
            x, y, z = np.add(self, other)
            return Point(x, y, z, frame=self.frame)
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.add(self, other)
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
            x, y, z = np.subtract(self, other)
            return Vector(x, y, z, frame=self.frame)
        else:
            return other.__rsub__(self)

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
    def create_unit_x(cls, frame: Frame) -> Vector:
        return cls(x=1.0, y=0.0, z=0.0, frame=frame)

    @classmethod
    def create_unit_y(cls, frame: Frame) -> Vector:
        return cls(x=0.0, y=1.0, z=0.0, frame=frame)

    @classmethod
    def create_unit_z(cls, frame: Frame) -> Vector:
        return cls(x=0.0, y=0.0, z=1.0, frame=frame)

    @classmethod
    def create_nan(cls, frame: Frame) -> Vector:
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
        x, y, z = np.cross(self.coords, other.coords)
        return Vector(x, y, z, frame=self.frame)


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
            x, y, z = np.subtract(self, other)
            return Vector(x, y, z, frame=self.frame)
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.subtract(self, other)
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
            x, y, z = np.add(self, other)
            return Point(x, y, z, frame=self.frame)
        else:
            return other.__add__(self)

    @classmethod
    def create_origin(cls, frame: Frame) -> Point:
        """Return a point at the origin of the specified coordinate system"""
        return cls(x=0.0, y=0.0, z=0.0, frame=frame)

    @classmethod
    def create_nan(cls, frame: Frame) -> Point:
        """Return a point at the origin of the specified coordinate system"""
        return cls(x=np.nan, y=np.nan, z=np.nan, frame=frame)

    @classmethod
    def from_array(cls, points: NDArray, frame) -> list[Point]:
        """Creates a list of Point instances from an array of points."""
        points = np.asarray(points)
        return [cls(x=x, y=y, z=z, frame=frame) for x, y, z in points.T]
