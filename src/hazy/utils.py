from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazy.primitives import GeometricPrimitive


def check_same_frame(*objects: GeometricPrimitive):
    """Verify that two geometric primitives are in the same reference frame.

    Args:
        object1: First geometric primitive
        object2: Second geometric primitive

    Raises:
        RuntimeError: If objects are in different frames
    """
    if len(objects) <= 1:
        return

    frames = []
    for obj in objects:
        if not hasattr(obj, "frame"):
            raise RuntimeError(
                f"Expected object with frame attribute, got {type(obj)}: {obj}"
            )
        else:
            frames.append(obj.frame)

    if not all(frames[0] == frame for frame in frames[1:]):
        mixed_systems = set([obj.frame for obj in objects])
        raise RuntimeError(
            "Expected all objects to be in the same coordinate system, "
            f"got {mixed_systems}"
        )
