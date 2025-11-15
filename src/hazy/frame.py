"""Hierarchical reference frames with transformation tracking and caching.

This module provides a Frame class for managing coordinate system transformations
using homogeneous transformation matrices. Frames can be hierarchically organized
with parent-child relationships, and transformations are cached for performance.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Literal, Self, overload

import numpy as np
from scipy.spatial.transform import Rotation

from hazy.constants import IDENTITY_ROTATION, IDENTITY_SCALE, IDENTITY_TRANSLATION
from hazy.primitives import Point, Vector

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def invalidate_transform_cache(method):
    """Decorator to invalidate cached transforms when frame is modified.

    Invalidates both local and global transform caches of this frame and
    recursively invalidates global transform caches of all descendant frames.

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
            raise RuntimeError(
                "Cannot modify frozen frame.\n"
                "Use frame.unfreeze() to allow modifications, "
                "or create a child frame with frame.make_child()."
            )
        self._cached_transform = None
        self._cached_transform_global = None
        self._invalidate_children_cache()
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

        Examples:
            >>> root = Frame(name="world")
            >>> child = Frame(parent=root, name="camera")
        """
        self._parent: Frame | None = parent
        self._name = name or f"Frame-{id(self)}"
        self._children: set[Frame] = set()

        self._rotations: list[Rotation] = [IDENTITY_ROTATION]
        self._translations: list[NDArray[np.floating]] = [IDENTITY_TRANSLATION]
        self._scalings: list[NDArray[np.floating]] = [IDENTITY_SCALE]

        self._cached_transform: NDArray[np.floating] | None = None
        self._cached_transform_global: NDArray[np.floating] | None = None

        self._is_frozen = False

        if parent is not None:
            parent._add_child(self)

    def _add_child(self, child: Frame) -> None:
        """Register a child frame.

        Args:
            child: Child frame to register
        """
        self._children.add(child)

    def _invalidate_children_cache(self) -> None:
        """Recursively invalidate global transform cache of all children."""
        for child in self._children:
            # Defensive check in case child was deleted but still in set
            if isinstance(child, Frame):
                child._cached_transform_global = None
                child._invalidate_children_cache()

    @property
    def parent(self) -> Frame | None:
        """Reference to the parent of this frame.
        If this is a root frame parent is None.
        """
        return self._parent

    @parent.setter
    def parent(self, value: Frame | None) -> None:
        """Prevent parent modification after frame creation.

        Raises:
            RuntimeError: Always, as reparenting would break children set consistency
        """
        raise RuntimeError(
            "Cannot change parent after frame creation.\n"
            "The parent-child relationship is immutable to maintain consistency.\n"
            "Create a new frame instead:\n"
            "  new_frame = new_parent.make_child(name='...')\n"
            "  new_frame.translate(...).rotate(...)"
        )

    @property
    def name(self) -> str:
        """Name of this frame."""
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
        return np.multiply.reduce(self._rotations)

    @property
    def combined_scale(self) -> NDArray[np.floating]:
        """Combined scaling matrix from all accumulated scalings."""
        return np.diag(np.append(np.multiply.reduce(self._scalings), 1))

    @property
    def combined_translation(self) -> NDArray[np.floating]:
        """Combined translation vector from all accumulated translations."""
        return np.add.reduce(self._translations)

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

    @property
    def x_axis(self) -> Vector:
        """Unit vector along the x-axis in this frame."""
        return Vector(x=1.0, y=0.0, z=0.0, frame=self)

    @property
    def x_axis_global(self) -> Vector:
        """Unit vector along the x-axis transformed to global frame."""
        return self.x_axis.to_frame(target_frame=self.root)

    @property
    def y_axis(self) -> Vector:
        """Unit vector along the y-axis in this frame."""
        return Vector(x=0.0, y=1.0, z=0.0, frame=self)

    @property
    def y_axis_global(self) -> Vector:
        """Unit vector along the y-axis transformed to global frame."""
        return self.y_axis.to_frame(target_frame=self.root)

    @property
    def z_axis(self) -> Vector:
        """Unit vector along the z-axis in this frame."""
        return Vector(x=0.0, y=0.0, z=1.0, frame=self)

    @property
    def z_axis_global(self) -> Vector:
        """Unit vector along the z-axis transformed to global frame."""
        return self.z_axis.to_frame(target_frame=self.root)

    @property
    def origin(self) -> Point:
        """Origin point (0, 0, 0) in this frame."""
        return Point(x=0.0, y=0.0, z=0.0, frame=self)

    @property
    def origin_global(self) -> Point:
        """Origin point transformed to global frame."""
        return self.origin.to_frame(target_frame=self.root)

    def freeze(self) -> Self:
        """Freeze frame to prevent further modifications.

        Returns:
            Self for method chaining

        Examples:
            >>> frame = Frame().translate(x=1.0).freeze()
            >>> frame.translate(x=2.0)  # Raises RuntimeError
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

        Examples:
            >>> frame = Frame()
            >>> frame.rotate_euler(z=90, degrees=True)
            >>> frame.rotate_euler(y=np.pi) # default radians
            >>> frame.rotate_euler(x=30, y=45, z=60, seq="zyx", degrees=True)
        """
        R = Rotation.from_euler(seq=seq, angles=(x, y, z), degrees=degrees)
        self._rotations.append(R)
        return self

    @invalidate_transform_cache
    def rotate_quaternion(
        self, quaternion: ArrayLike, *, scalar_first: bool = False
    ) -> Self:
        """Add quaternion rotation to frame.

        Args:
            quaternion: (4,) or (N, 4) array describing rotation with quaternion
            scalar_first: Whether the scalar is the first or last element of the
                quaternion

        Returns:
            Self for method chaining

        Examples:
            >>> frame = Frame()
            >>> frame.rotate_quaternion([0, 0, 0, 1])  # Identity, scalar last
            >>> frame.rotate_quaternion(
                    [1, 0, 0, 0], scalar_first=True
                )  # Identity, scalar first
        """
        R = Rotation.from_quaternion(quaternion, scalar_first=scalar_first)
        self._rotations.append(R)
        return self

    @invalidate_transform_cache
    def rotate(self, rotation: ArrayLike) -> Self:
        """Add rotation matrix to frame.

        Args:
            rotation: 3x3 rotation matrix

        Returns:
            Self for method chaining

        Examples:
            >>> frame = Frame()
            >>> R = np.eye(3)  # Identity rotation
            >>> frame.rotate(R)
        """
        R = Rotation.from_matrix(rotation)
        self._rotations.append(R)
        return self

    @invalidate_transform_cache
    def translate(self, *, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Self:
        """Add translation to frame.

        Args:
            x: Translation along x-axis
            y: Translation along y-axis
            z: Translation along z-axis

        Returns:
            Self for method chaining

        Examples:
            >>> frame = Frame()
            >>> frame.translate(x=1.0, y=2.0, z=3.0)
        """
        translation = np.array([x, y, z], dtype=float)
        self._translations.append(translation)
        return self

    @overload
    def scale(self, x: float) -> Self: ...

    @overload
    def scale(self, x: float, y: float, z: float) -> Self: ...

    @invalidate_transform_cache
    def scale(self, x: float, y: float | None = None, z: float | None = None) -> Self:
        """Add scaling to frame.

        Args:
            x: Uniform scale factor or x-axis scale
            y: Y-axis scale (if provided, x/y/z are per-axis scales)
            z: Z-axis scale (if provided, x/y/z are per-axis scales)

        Returns:
            Self for method chaining

        Examples:
            >>> frame = Frame()
            >>> frame.scale(2.0)  # Uniform scaling
            >>> frame.scale(1.0, 2.0, 3.0)  # Per-axis scaling
        """
        if y is None and z is None:
            scaling = np.ones(3, dtype=float) * x
        elif y is not None and z is not None:
            scaling = np.array([x, y, z], dtype=float)
        else:
            raise ValueError(
                "Provide either uniform scale or (x, y, z).\n"
                "Use:\n"
                "  frame.scale(2.0)  # Uniform scaling\n"
                "  frame.scale(1.0, 2.0, 3.0)  # Per-axis scaling"
            )

        self._scalings.append(scaling)
        return self

    def transform_to(self, target: Frame) -> NDArray[np.floating]:
        """Compute transformation matrix from this frame to target frame.

        Args:
            target: Target reference frame

        Returns:
            4x4 transformation matrix from self to target

        Raises:
            RuntimeError: If frames belong to different hierarchies (different roots)

        Examples:
            >>> world = Frame.make_root("world")
            >>> camera = world.make_child("camera").translate(z=5.0)
            >>> T = camera.transform_to(world)
        """
        if self == target:
            return np.eye(4)

        if self.root is not target.root:
            raise RuntimeError(
                f"Cannot transform between frames from different hierarchies.\n"
                f"Frame '{self.name}' has root '{self.root.name}', "
                f"but frame '{target.name}' has root '{target.root.name}'."
            )

        return target.transform_from_global @ self.transform_to_global

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
            raise ValueError(
                "Provide either (x, y, z) or single array-like.\n"
                "Use:\n"
                "  frame.vector(1.0, 2.0, 3.0)  # Three scalars\n"
                "  frame.vector([1, 2, 3])  # Array-like"
            )

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
            raise ValueError(
                "Provide either (x, y, z) or single array-like.\n"
                "Use:\n"
                "  frame.point(1.0, 2.0, 3.0)  # Three scalars\n"
                "  frame.point([1, 2, 3])  # Array-like"
            )

    def batch_transform_points_global(
        self, points: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Batch transform an array of points from this coordinate system to global.

        Homogeneous coordinate (w=1) will be added automatically.

        Args:
            points: Array of 3D points, will be reshaped to (N, 3)

        Returns:
            Points transformed to global space

        Examples:
            >>> frame = Frame().translate(x=1.0)
            >>> points = np.array([[0, 0, 0], [1, 0, 0]])
            >>> frame.batch_transform_points_global(points)
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

        Homogeneous coordinate (w=0) will be added automatically.

        Args:
            vectors: Array of 3D vectors, will be reshaped to (N, 3)

        Returns:
            Vectors transformed to global space

        Examples:
            >>> frame = Frame().translate(x=1.0)
            >>> vectors = np.array([[1, 0, 0], [0, 1, 0]])
            >>> frame.batch_transform_vectors_global(vectors)
        """
        vectors = np.asarray(vectors)
        original_shape = vectors.shape
        vectors_homogenous = np.hstack(
            [vectors.reshape(-1, 3), np.zeros((vectors.size // 3, 1))]
        )
        transformed = vectors_homogenous @ self.transform_to_global.T
        return transformed[:, :3].reshape(original_shape)

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
