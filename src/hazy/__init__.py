"""Coordinate system module for frame-aware geometric transformations.

This module provides hierarchical reference frames and geometric primitives (Point, Vector)
that carry frame information and support coordinate transformations.

Key classes:
    - Frame: Hierarchical reference frame with transformation tracking
    - Point: Geometric point with frame awareness (w=1)
    - Vector: Geometric vector with frame awareness (w=0)

Example:
    >>> from hazy import Frame, Point, Vector
    >>>
    >>> # Create frames
    >>> parent = Frame(name="parent")
    >>> child = Frame(parent=parent, name="child")
    >>> child.translate(x=5).rotate_euler(z=90, degrees=True)
    >>>
    >>> # Create points in child frame
    >>> p_local = Point(1, 0, 0, frame=child)
    >>> p_parent = p_local.to_frame(parent)
    >>> print(p_parent)  # Point at (5, 1, 0) in parent frame
"""

from hazy.frame import Frame
from hazy.primitives import Point, Vector

__all__ = ["Frame", "Point", "Vector"]
