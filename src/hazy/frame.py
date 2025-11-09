"""Hierarchical reference frames with transformation tracking and caching.

This module provides a Frame class for managing coordinate system transformations
using homogeneous transformation matrices. Frames can be hierarchically organized
with parent-child relationships, and transformations are cached for performance.
"""

from __future__ import annotations

from functools import reduce, wraps
from operator import add, mul
from typing import TYPE_CHECKING, Literal, Self

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

    def __new__(
        cls,
        parent: Frame | None = None,
        name: str | None = None,
        _allow_orphan: bool = False,
    ) -> Self | Frame:
        if parent is None and not _allow_orphan:
            return cls.global_frame()
        else:
            instance = super().__new__(cls)
            return instance

    def __init__(
        self,
        parent: Frame | None = None,
        name: str | None = None,
        *,
        _allow_orphan: bool = False,
    ):
        """Initialize a new reference frame.

        Note:
            Calling Frame() without parent returns the global singleton frame.
            Providing a name in this case will raise a ValueError.

        Args:
            parent: Parent frame (None for root frames)
            name: Frame identifier (auto-generated if not provided)
            _allow_orphan: Internal flag to allow creation without parent

        Raises:
            ValueError: If name is provided but global singleton is returned
        """
        if hasattr(self, "parent"):
            if name is not None:
                raise ValueError(
                    f"Cannot specify name '{name}' when creating global frame "
                    f"singleton. Use Frame(parent=..., name='{name}') or "
                    "Frame.global_frame() instead."
                )
            return

        self._parent: Frame | None = parent
        self._name = name or f"Frame-{id(self)}"

        self._rotations: list[Rotation] = [IDENTITY_ROTATION]
        self._translations: list[NDArray[np.floating]] = [IDENTITY_TRANSLATION]
        self._scalings: list[NDArray[np.floating]] = [IDENTITY_SCALE]

        self._cached_transform: NDArray[np.floating] | None = None
        self._cached_transform_global: NDArray[np.floating] | None = None

        self._is_frozen = False

    @classmethod
    def create_orphan(cls, name: str | None = None) -> Frame:
        return cls(parent=None, name=name, _allow_orphan=True)

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

    _global_frame = None

    @classmethod
    def global_frame(cls) -> Frame:
        """Get or create the singleton global reference frame.

        Returns:
            Global frame instance
        """
        if cls._global_frame is None:
            cls._global_frame = Frame(parent=None, name="global", _allow_orphan=True)
        return cls._global_frame

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
    def unit_x(self) -> Vector:
        return Vector(x=1.0, y=0.0, z=0.0, frame=self)

    @property
    def unit_x_global(self) -> Vector:
        return self.unit_x.to_frame(target_frame=Frame.global_frame())

    @property
    def unit_y(self) -> Vector:
        return Vector(x=0.0, y=1.0, z=0.0, frame=self)

    @property
    def unit_y_global(self) -> Vector:
        return self.unit_y.to_frame(target_frame=Frame.global_frame())

    @property
    def unit_z(self) -> Vector:
        return Vector(x=0.0, y=0.0, z=1.0, frame=self)

    @property
    def unit_z_global(self) -> Vector:
        return self.unit_z.to_frame(target_frame=Frame.global_frame())

    @property
    def origin(self) -> Point:
        return Point(x=0.0, y=0.0, z=0.0, frame=self)

    @property
    def origin_global(self):
        return self.origin.to_frame(target_frame=Frame.global_frame())

    def create_vector(self, x: float, y: float, z: float) -> Vector:
        return Vector(x=x, y=y, z=z, frame=self)

    def create_point(self, x: float, y: float, z: float) -> Point:
        return Point(x=x, y=y, z=z, frame=self)

    def batch_transform_global(
        self, points: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        points = np.asarray(points)
        original_shape = points.shape
        points_homogenous = np.hstack(
            [points.reshape(-1, 3), np.ones((points.size // 3, 1))]
        )
        transformed = points_homogenous @ self.transform_to_global.T
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
