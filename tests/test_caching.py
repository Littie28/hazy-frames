"""Regression tests for transform caching.

Caches must never return stale matrices, no matter in which order frames of a
hierarchy are queried and modified.
"""

from __future__ import annotations

import gc
import pickle
import weakref
from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from hazy import Frame, Point

finite = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
positive_scale = st.floats(min_value=0.1, max_value=5)


def uncached_local_matrix(frame: Frame) -> np.ndarray:
    matrix = frame.combined_scale
    matrix[:3, :3] = frame.combined_rotation.as_matrix() @ matrix[:3, :3]
    matrix[:3, 3] = frame.combined_translation
    return matrix


def uncached_global_matrix(frame: Frame) -> np.ndarray:
    matrix = np.eye(4)
    current: Frame | None = frame
    while current is not None:
        matrix = uncached_local_matrix(current) @ matrix
        current = current.parent
    return matrix


def mutate(frame: Frame, operation: str, values: tuple[float, float, float]) -> None:
    x, y, z = values
    if operation == "translate":
        frame.translate(x=x, y=y, z=z)
    elif operation == "rotate":
        frame.rotate_euler(x=x, y=y, z=z, degrees=True)
    elif operation == "scale":
        frame.scale(abs(x) + 0.1, abs(y) + 0.1, abs(z) + 0.1)
    else:
        frame.clear_all_transforms()


@pytest.mark.integration
class TestCacheConsistency:
    @settings(max_examples=60, deadline=None)
    @given(data=st.data())
    def test_interleaved_mutations_and_queries_never_see_stale_caches(self, data):
        root = Frame.make_root("root")
        frames = [root]
        for index in range(1, 6):
            parent = frames[data.draw(st.integers(0, index - 1))]
            frames.append(parent.make_child(f"frame-{index}"))

        frame_index = st.integers(0, len(frames) - 1)
        values = st.tuples(finite, finite, finite)
        operations = st.sampled_from(["translate", "rotate", "scale", "clear"])

        for _ in range(data.draw(st.integers(1, 15))):
            if data.draw(st.booleans()):
                frame = frames[data.draw(frame_index)]
                mutate(frame, data.draw(operations), data.draw(values))
                continue

            source = frames[data.draw(frame_index)]
            target = frames[data.draw(frame_index)]
            expected = np.linalg.inv(
                uncached_global_matrix(target)
            ) @ uncached_global_matrix(source)
            assert_allclose(source.transform_to(target), expected, atol=1e-9)

            point = source.point(1.0, -2.0, 3.0)
            expected_point = (expected @ [1.0, -2.0, 3.0, 1.0])[:3]
            assert_allclose(point.to_frame(target), expected_point, atol=1e-9)

    def test_ancestor_change_updates_inverse_of_descendant(self, root_frame: Frame):
        arm = root_frame.make_child("arm")
        tool = arm.make_child("tool").translate(x=1)
        sibling = root_frame.make_child("sibling")
        _ = tool.transform_from_global, sibling.transform_to(tool)

        arm.translate(y=5)

        assert_allclose(tool.transform_from_global @ [0, 5, 0, 1], [-1, 0, 0, 1])
        assert_allclose(sibling.origin.to_frame(tool), [-1, -5, 0])

    def test_mutating_returned_matrices_does_not_corrupt_cache(self, root_frame: Frame):
        frame = root_frame.make_child("child").translate(x=1)
        expected = frame.transform_to_global

        for matrix in (
            frame.transform_to_parent,
            frame.transform_to_global,
            frame.transform_from_global,
            frame.transform_to(root_frame),
            frame.transform_to(frame),
        ):
            matrix[:] = 0.0

        assert_allclose(frame.transform_to_global, expected)
        assert_allclose(frame.transform_to(frame), np.eye(4))
        assert_allclose(frame.origin.to_global(), [1, 0, 0])

    def test_frozen_child_follows_modified_parent(self, root_frame: Frame):
        parent = root_frame.make_child("parent")
        child = parent.make_child("child").translate(x=1).freeze()
        _ = child.origin_global

        parent.translate(x=2)

        assert_allclose(child.origin_global, [3, 0, 0])


@pytest.mark.unit
class TestPrimitiveInternals:
    def test_nan_point_keeps_point_semantics_after_transform(self, root_frame: Frame):
        frame = root_frame.make_child("child").translate(x=1)
        point = Point.create_nan(frame).to_global()
        displaced = point + root_frame.vector(1, 0, 0)

        assert isinstance(point, Point)
        assert isinstance(displaced, Point)
        assert np.isnan(np.asarray(displaced)).all()

    def test_primitives_reject_arbitrary_attributes(self, frame: Frame):
        with pytest.raises(AttributeError):
            frame.point(1, 2, 3).color = "red"  # type: ignore

    def test_arithmetic_result_types_from_homogeneous_coordinates(self, frame: Frame):
        p, q, v = frame.point(1, 2, 3), frame.point(0, 1, 1), frame.vector(1, 1, 1)

        assert type(p - q).__name__ == "Vector"
        assert type(p + v).__name__ == "Point"
        assert type(v + p).__name__ == "Point"
        assert type(p - v).__name__ == "Point"
        assert type(v - v).__name__ == "Vector"
        assert type(-v).__name__ == "Vector"
        assert_allclose((p - q).to_global(), [1, 1, 2])


@pytest.mark.unit
class TestFrameLifetime:
    def test_unreferenced_child_is_garbage_collected(self, root_frame: Frame):
        child = root_frame.make_child("temporary").translate(x=1)
        child_ref = weakref.ref(child)

        del child
        gc.collect()

        assert child_ref() is None
        assert len(root_frame._children) == 0

    def test_primitive_keeps_its_frame_and_ancestors_alive(self):
        point = Frame.make_root("world").make_child("camera").point(1, 2, 3)
        frame_ref = weakref.ref(point.frame)
        gc.collect()

        assert frame_ref() is point.frame
        assert point.frame.root.name == "world"
        assert_allclose(point.to_global(), [1, 2, 3])

    def test_surviving_children_still_see_parent_changes(self, root_frame: Frame):
        parent = root_frame.make_child("parent")
        kept = parent.make_child("kept")
        points = kept.points([[0.0, 0.0, 0.0]])
        for index in range(10):
            parent.make_child(f"dropped-{index}").origin_global  # noqa: B018
        _ = kept.origin_global
        gc.collect()

        parent.translate(x=2)

        assert_allclose(kept.origin_global, [2, 0, 0])
        assert_allclose(points.to_global(), [[2, 0, 0]])
        assert list(parent._children) == [kept]

    def test_deepcopy_keeps_copied_hierarchy_consistent(self, root_frame: Frame):
        parent = root_frame.make_child("parent")
        child = parent.make_child("child").translate(y=1)
        _ = child.origin_global

        copied = deepcopy(child)
        assert copied.parent is not None
        copied.parent.translate(x=5)

        assert list(copied.parent._children) == [copied]
        assert_allclose(copied.origin_global, [5, 1, 0])
        assert_allclose(child.origin_global, [0, 1, 0])

        parent.translate(z=2)
        assert_allclose(child.origin_global, [0, 1, 2])
        assert_allclose(copied.origin_global, [5, 1, 0])

    def test_pickle_round_trip_keeps_hierarchy(self, root_frame: Frame):
        parent = root_frame.make_child("parent")
        point = parent.make_child("child").translate(x=1).point(1, 0, 0)
        _ = point.to_global()

        restored = pickle.loads(pickle.dumps(point))
        assert restored.frame.parent is not None
        restored.frame.parent.translate(y=2)

        assert_allclose(restored.to_global(), [2, 2, 0])
