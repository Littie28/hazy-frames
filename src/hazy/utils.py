from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

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
                "Expected GeometricPrimitive with frame attribute, "
                f"got {type(obj).__name__}: {obj}.\n"
                "This is likely a developer error - ensure all arguments are "
                "Point or Vector instances."
            )
        else:
            frames.append(obj.frame)

    if not all(frames[0] == frame for frame in frames[1:]):
        mixed_systems = set([obj.frame.name for obj in objects])
        raise RuntimeError(
            "Expected all objects to be in the same coordinate system, "
            f"got frames: {mixed_systems}.\n"
            "Use obj.to_frame(target_frame) to transform objects to "
            "a common frame first."
        )


def all_same_type(objects: Iterable) -> bool:
    """Check if all objects in iterable have the exact same type.

    Args:
        objects: Iterable of objects to check

    Returns:
        True if all objects have identical type, False otherwise

    Raises:
        ValueError: If iterable is empty
    """
    iterator = iter(objects)

    try:
        first = next(iterator)
    except StopIteration as err:
        raise ValueError("Cannot check type consistency of empty iterable.") from err

    return all(type(first) is type(obj) for obj in iterator)
