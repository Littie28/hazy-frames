from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hazy.utils import all_same_type, check_same_frame

if TYPE_CHECKING:
    from hazy import Frame


def test_check_same_frame_no_primitive_raises_runtime_error(frame: Frame):
    with pytest.raises(RuntimeError, match="Expected GeometricPrimitive"):
        check_same_frame(int, float)

    with pytest.raises(RuntimeError, match="Expected GeometricPrimitive"):
        check_same_frame(frame.vector(1, 1, 1), float)


def test_check_same_frame(frame: Frame):
    # check frame raises a RuntimeError
    check_same_frame(frame.vector(1, 1, 1), frame.vector(5, 5, 5))
    check_same_frame(frame.point(1, 1, 1), frame.vector(5, 5, 5))
    check_same_frame(frame.point(1, 1, 1), frame.point(5, 5, 5))


def test_all_same_type_empty_raises_value_error():
    with pytest.raises(ValueError, match="Cannot check"):
        all_same_type([])


def test_all_same(frame: Frame):
    vectors = [frame.vector(i, 0, 0) for i in range(10)]
    assert all_same_type(vectors)

    points = [frame.point(i, 0, 0) for i in range(10)]
    assert all_same_type(points)

    assert not all_same_type(vectors + points)
