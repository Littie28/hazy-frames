"""Hierarchical reference frames with transformation tracking and caching.

This module provides a Frame class for managing coordinate system transformations
using homogeneous transformation matrices. Frames can be hierarchically organized
with parent-child relationships, and transformations are cached for performance.
"""

from __future__ import annotations

import copy
from functools import reduce, wraps
from operator import add, mul
from typing import TYPE_CHECKING, Literal, Self, overload

import numpy as np
from scipy.spatial.transform import Rotation

from hazy.constants import IDENTITY_ROTATION, IDENTITY_SCALE, IDENTITY_TRANSLATION
from hazy.primitives import Point, Vector

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    # from laser_cross_calibration.coordinate_system.primitives import Point, Vector


def invalidate_transform_cache(method):
    """Decorator to invalidate cached transforms when frame is modified.

    Args:
        method: Method that modifies the frame

    Returns:
        Wrapped method that clears caches before execution

    Raises:
        RuntimeError: If frame is frozen
    """

    @wraps(method)
    def wrapper(self: Frame, *args, **kwargs):
        if self._is_frozen:
            raise RuntimeError("Can not modify frozen frame.")
        self._cached_transform = None
        self._cached_transform_global = None
        return method(self, *args, **kwargs)

    return wrapper


class Frame:
    """Hierarchical reference frame with transformation tracking.

    Frames support accumulation of transformations (translation, rotation, scaling)
    and provide cached transformation matrices for efficient repeated calculations.

    Transformations are applied in S→R→T order (Scale, Rotation, Translation)
    when converting from local to parent coordinates.

    Attributes:
        parent: Parent frame in hierarchy (None for root frames)
        name: Human-readable frame identifier
    """

    def __init__(
        self,
        parent: Frame | None = None,
        name: str | None = None,
    ):
        """Initialize a new reference frame.

        Args:
            parent: Parent frame in hierarchy (None for root frames)
            name: Frame identifier (auto-generated if not provided)
        """
        self._parent: Frame | None = parent
        self._name = name or f"Frame-{id(self)}"

        self._rotations: list[Rotation] = [IDENTITY_ROTATION]
        self._translations: list[NDArray[np.floating]] = [IDENTITY_TRANSLATION]
        self._scalings: list[NDArray[np.floating]] = [IDENTITY_SCALE]

        self._cached_transform: NDArray[np.floating] | None = None
        self._cached_transform_global: NDArray[np.floating] | None = None

        self._is_frozen = False

    def __deepcopy__(self, memo):
        """Create a deep copy of the frame and its hierarchy."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    @property
    def parent(self) -> Frame | None:
        return self._parent

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> Frame:
        """Get the root frame of this hierarchy.

        Returns:
            Root frame (self if no parent, otherwise traverses up to root)
        """
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    @property
    def combined_rotation(self) -> Rotation:
        """Combined rotation from all accumulated rotations."""
        return reduce(mul, self._rotations)

    @property
    def combined_scale(self) -> NDArray[np.floating]:
        """Combined scaling matrix from all accumulated scalings."""
        return np.diag(np.append(reduce(mul, self._scalings), 1))

    @property
    def combined_translation(self) -> NDArray[np.floating]:
        """Combined translation vector from all accumulated translations."""
        return reduce(add, self._translations)

    @property
    def transform_to_parent(self) -> NDArray[np.floating]:
        """4x4 homogeneous transformation matrix from local to parent frame.

        Transformations are applied in S→R→T order (Scale, Rotation, Translation).
        Results are cached for performance.

        Returns:
            4x4 transformation matrix (copy to prevent modification)
        """
        if self._cached_transform is not None:
            return self._cached_transform.copy()

        transform = np.eye(4)

        transform = self.combined_scale @ transform
        transform[:3, :3] = self.combined_rotation.as_matrix() @ transform[:3, :3]
        transform[:3, 3] += self.combined_translation

        self._cached_transform = transform.copy()

        return transform.copy()

    @property
    def transform_from_parent(self) -> NDArray[np.floating]:
        """4x4 homogeneous transformation matrix from parent to local frame.

        Returns:
            Inverse of transform_to_parent
        """
        return np.linalg.inv(self.transform_to_parent)

    def freeze(self) -> Self:
        """Freeze frame to prevent further modifications.

        Returns:
            Self for method chaining
        """
        self._is_frozen = True
        return self

    def unfreeze(self) -> Self:
        """Unfreeze frame to allow modifications.

        Returns:
            Self for method chaining
        """
        self._is_frozen = False
        return self

    @invalidate_transform_cache
    def rotate_euler(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        seq: Literal["xyz", "xzy", "yzx", "yxz", "zxy", "zyx"] = "xyz",
        degrees: bool = False,
    ) -> Self:
        """Add Euler angle rotation to frame.

        Args:
            x: Rotation around x-axis
            y: Rotation around y-axis
            z: Rotation around z-axis
            seq: Rotation sequence (default: xyz)
            degrees: If True, angles are in degrees, otherwise radians

        Returns:
            Self for method chaining
        """
        R = Rotation.from_euler(seq=seq, angles=(x, y, z), degrees=degrees)
        self._rotations.append(R)
        return self

    @invalidate_transform_cache
    def rotate_quaternion(
        self, quaternion: ArrayLike, *, scalar_first: bool = False
    ) -> Self:
        """Add Quaternion rotation to frame:

        Args:
            quaternion: a (4, ) or (N, 4) array describing a rotation with a quaternion
            scalar_first: Wether the the scaling is the first or last element of the
                quaternion

        Returns:
            Self for method chaining
        """
        R = Rotation.from_quaternion(quaternion, scalar_first=scalar_first)
        self._rotations.append(R)
        return self

    @invalidate_transform_cache
    def rotate(self, rotation) -> Self:
        """Add rotation matrix to frame.

        Args:
            rotation: 3x3 rotation matrix

        Returns:
            Self for method chaining
        """
        R = Rotation.from_matrix(rotation)
        self._rotations.append(R)
        return self

    @invalidate_transform_cache
    def translate(self, *, x=0.0, y=0.0, z=0.0) -> Self:
        """Add translation to frame.

        Args:
            x: Translation along x-axis
            y: Translation along y-axis
            z: Translation along z-axis

        Returns:
            Self for method chaining
        """
        translation = np.array([x, y, z], dtype=float)
        self._translations.append(translation)
        return self

    @invalidate_transform_cache
    def scale(self, scale: float | tuple[float, float, float] | ArrayLike) -> Self:
        """Add scaling to frame.

        Args:
            scale: Uniform scale factor or (sx, sy, sz) tuple

        Returns:
            Self for method chaining

        Raises:
            ValueError: If tuple doesn't have exactly 3 elements
        """
        if isinstance(scale, float | int):
            scaling = np.ones(3, dtype=float) * scale
        else:
            scaling = np.asarray(scale, dtype=float).flatten()
            if scaling.shape != (3,):
                raise ValueError()

        self._scalings.append(scaling)
        return self

    @property
    def transform_to_global(self) -> NDArray[np.floating]:
        """4x4 transformation matrix from this frame to global frame.

        Recursively composes transformations through parent hierarchy.
        Results are cached for performance.

        Returns:
            4x4 transformation matrix
        """
        if self._cached_transform_global is not None:
            return self._cached_transform_global.copy()

        if self.parent is None:
            self._cached_transform_global = np.eye(4, dtype=float)
        else:
            self._cached_transform_global = (
                self.parent.transform_to_global @ self.transform_to_parent
            )
        return self._cached_transform_global

    @property
    def transform_from_global(self) -> NDArray[np.floating]:
        """4x4 transformation matrix from global frame to this frame.

        Returns:
            Inverse of transform_to_global
        """
        return np.linalg.inv(self.transform_to_global)

    def transform_to(self, target: Frame) -> NDArray[np.floating]:
        """Compute transformation matrix from this frame to target frame.

        Args:
            target: Target reference frame

        Returns:
            4x4 transformation matrix from self to target

        Raises:
            RuntimeError: If frames belong to different hierarchies (different roots)
        """
        if self == target:
            return np.eye(4)

        if self.root is not target.root:
            raise RuntimeError(
                f"Cannot transform between frames from different hierarchies. "
                f"Frame '{self.name}' has root '{self.root.name}', "
                f"but frame '{target.name}' has root '{target.root.name}'."
            )

        return target.transform_from_global @ self.transform_to_global

    @property
    def x_axis(self) -> Vector:
        return Vector(x=1.0, y=0.0, z=0.0, frame=self)

    @property
    def x_axis_global(self) -> Vector:
        return self.x_axis.to_frame(target_frame=self.root)

    @property
    def y_axis(self) -> Vector:
        return Vector(x=0.0, y=1.0, z=0.0, frame=self)

    @property
    def y_axis_global(self) -> Vector:
        return self.y_axis.to_frame(target_frame=self.root)

    @property
    def z_axis(self) -> Vector:
        return Vector(x=0.0, y=0.0, z=1.0, frame=self)

    @property
    def z_axis_global(self) -> Vector:
        return self.z_axis.to_frame(target_frame=self.root)

    @property
    def origin(self) -> Point:
        return Point(x=0.0, y=0.0, z=0.0, frame=self)

    @property
    def origin_global(self):
        return self.origin.to_frame(target_frame=self.root)

    @overload
    def vector(self, x: float, y: float, z: float) -> Vector: ...

    @overload
    def vector(self, x: ArrayLike) -> Vector: ...

    def vector(
        self, x: float | ArrayLike, y: float | None = None, z: float | None = None
    ) -> Vector:
        """Create vector in this frame.

        Args:
            x: X-coordinate or array-like [x, y, z]
            y: Y-coordinate (required if x is scalar)
            z: Z-coordinate (required if x is scalar)

        Returns:
            Vector in this frame

        Examples:
            >>> frame.vector(1.0, 2.0, 3.0)
            >>> frame.vector([1, 2, 3])
            >>> frame.vector(np.array([1, 2, 3]))
        """
        if y is None and z is None:
            return Vector.from_array(x, frame=self)
        elif y is not None and z is not None:
            return Vector(x=x, y=y, z=z, frame=self)
        else:
            raise ValueError("Provide either (x, y, z) or single array-like")

    @overload
    def point(self, x: float, y: float, z: float) -> Point: ...

    @overload
    def point(self, x: ArrayLike) -> Point: ...

    def point(
        self, x: float | ArrayLike, y: float | None = None, z: float | None = None
    ) -> Point:
        """Create point in this frame.

        Args:
            x: X-coordinate or array-like [x, y, z]
            y: Y-coordinate (required if x is scalar)
            z: Z-coordinate (required if x is scalar)

        Returns:
            Point in this frame

        Examples:
            >>> frame.point(1.0, 2.0, 3.0)
            >>> frame.point([1, 2, 3])
            >>> frame.point(np.array([1, 2, 3]))
        """
        if y is None and z is None:
            return Point.from_array(x, frame=self)
        elif y is not None and z is not None:
            return Point(x=x, y=y, z=z, frame=self)
        else:
            raise ValueError("Provide either (x, y, z) or single array-like")

    def batch_transform_points_global(
        self, points: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Batch transform an array of points from this coordinate system to global.
        Homogenous coordinate (w=1) will be added automatically.

        Args
            points: array of 3d points, will be reshaped to (N, 3)

        Returns:
            points transformed to global space
        """
        points = np.asarray(points)
        original_shape = points.shape
        points_homogenous = np.hstack(
            [points.reshape(-1, 3), np.ones((points.size // 3, 1))]
        )
        transformed = points_homogenous @ self.transform_to_global.T
        return transformed[:, :3].reshape(original_shape)

    def batch_transform_vectors_global(
        self, vectors: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Batch transform an array of vectors from this coordinate system to global.
        Homogenous coordinate (w=0) will be added automatically.

        Args
            vectors: array of 3d vectors, will be reshaped to (N, 3)

        Returns:
            vectors transformed to global space
        """
        vectors = np.asarray(vectors)
        original_shape = vectors.shape
        vectors_homogenous = np.hstack(
            [vectors.reshape(-1, 3), np.ones((vectors.size // 3, 1))]
        )
        transformed = vectors_homogenous @ self.transform_to_global.T
        return transformed[:, :3].reshape(original_shape)

    def __repr__(self) -> str:
        parent_name = self.parent.name if self.parent else "None"
        # Subtract 1 because we always have identity elements
        n_rot = len(self._rotations) - 1
        n_trans = len(self._translations) - 1
        n_scale = len(self._scalings) - 1
        transforms = f"{n_rot}R+{n_trans}T+{n_scale}S"
        frozen = " [FROZEN]" if self._is_frozen else ""
        return (
            f"Frame('{self.name}', "
            f"parent='{parent_name}', "
            f"transforms={transforms}{frozen})"
        )

    @classmethod
    def make_root(cls, name: str | None = None) -> Frame:
        """Create a root frame (frame without parent).

        Args:
            name: Optional name for the root frame

        Returns:
            New root frame

        Example:
            >>> root = Frame.make_root(name="world")
            >>> robot = root.make_child(name="robot")
        """
        return cls(parent=None, name=name)

    def make_child(self, name: str | None = None) -> Frame:
        """Creates a frame with this frame as its parent.

        Args:
            name: Optional name for the child frame

        Returns:
            New child frame

        Example:
            >>> root = Frame.make_root(name="world")
            >>> child = root.make_child(name="child")
        """
        return Frame(parent=self, name=name)
