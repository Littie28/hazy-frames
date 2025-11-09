from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from hazy import Frame
from hazy.constants import (
    IDENTITY_ROTATION,
    IDENTITY_SCALE,
    IDENTITY_TRANSLATION,
    VVSMALL,
)

coords = floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
small_coords = floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
angles = floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
scales = floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False)


@pytest.mark.unit
class TestFrameCreation:
    def test_creation_without_arguments_returns_global_singleton(self):
        frame = Frame()

        assert frame.name == "global"
        assert frame.parent is None

    def test_creation_with_parent_none_returns_global_singleton(self):
        frame = Frame(parent=None)

        assert frame.name == "global"
        assert frame.parent is None

    def test_creation_with_name_only_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot specify name"):
            Frame(name="some name")

    def test_creation_with_parent_none_and_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot specify name"):
            Frame(parent=None, name="some name")

    def test_creation_with_parent_and_name(self):
        frame = Frame(parent=Frame.global_frame(), name="some sub-frame")

        assert frame.parent is Frame.global_frame()
        assert frame.name == "some sub-frame"

    def test_auto_generated_name(self):
        parent = Frame.global_frame()
        frame = Frame(parent=parent)

        assert frame.name.startswith("Frame-")
        assert frame.parent is parent

    def test_singleton_behavior(self):
        frame1 = Frame()
        frame2 = Frame()
        frame3 = Frame.global_frame()

        assert frame1 is frame2
        assert frame2 is frame3

    def test_create_orphan(self):
        orphan = Frame.create_orphan(name="orphan")

        assert orphan.name == "orphan"
        assert orphan.parent is None
        assert orphan is not Frame.global_frame()


@pytest.mark.unit
class TestFrameHierarchy:
    def test_hierarchical_frames(self):
        global_frame = Frame.global_frame()
        parent_frame = Frame(parent=global_frame, name="parent")
        child_frame = Frame(parent=parent_frame, name="child")

        assert child_frame.parent is parent_frame
        assert parent_frame.parent is global_frame
        assert global_frame.parent is None


@pytest.mark.unit
class TestFrameInitialState:
    def test_initial_transform_state(self):
        frame = Frame(parent=Frame.global_frame())

        assert len(frame._rotations) == 1
        assert len(frame._translations) == 1
        assert len(frame._scalings) == 1
        assert frame._cached_transform is None
        assert frame._cached_transform_global is None
        assert frame._is_frozen is False

    def test_identity_transform_initialization(self):
        frame = Frame()

        assert_allclose(frame._rotations[0].as_matrix(), IDENTITY_ROTATION.as_matrix())
        assert_allclose(frame._translations[0], IDENTITY_TRANSLATION)
        assert_allclose(frame._scalings[0], IDENTITY_SCALE)

    def test_identity_transform_combination(self):
        frame = Frame()

        assert_allclose(
            frame.combined_rotation.as_matrix(), IDENTITY_ROTATION.as_matrix()
        )
        assert_allclose(frame.combined_translation, IDENTITY_TRANSLATION)
        assert_allclose(np.diagonal(frame.combined_scale)[:3], IDENTITY_SCALE)

        frame.rotate(IDENTITY_ROTATION.as_matrix())
        frame.translate(x=0.0, y=0.0, z=0.0)
        frame.scale(1.0).scale((1.0, 1.0, 1.0))

        assert_allclose(
            frame.combined_rotation.as_matrix(), IDENTITY_ROTATION.as_matrix()
        )
        assert_allclose(frame.combined_translation, IDENTITY_TRANSLATION)
        assert_allclose(np.diagonal(frame.combined_scale)[:3], IDENTITY_SCALE)


@pytest.mark.unit
class TestFrameUnitVectors:
    def test_global_frame_unit_vectors(self):
        frame = Frame()

        assert frame.unit_x == frame.unit_x_global
        assert frame.unit_y == frame.unit_y_global
        assert frame.unit_z == frame.unit_z_global

    def test_frame_unit_vectors_rotation(self):
        parent = Frame(parent=Frame.global_frame(), name="parent").rotate_euler(
            x=90, degrees=True
        )

        assert parent.unit_x.frame == parent
        assert_allclose(parent.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert parent.unit_y.frame == parent
        assert_allclose(parent.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert parent.unit_z.frame == parent
        assert_allclose(parent.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert parent.unit_x_global.frame == Frame.global_frame()
        assert_allclose(parent.unit_x_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert parent.unit_y_global.frame == Frame.global_frame()
        assert_allclose(parent.unit_y_global, [0.0, 0.0, 1.0], atol=VVSMALL)
        assert parent.unit_z_global.frame == Frame.global_frame()
        assert_allclose(parent.unit_z_global, [0.0, -1.0, 0.0], atol=VVSMALL)

        child = Frame(parent=parent, name="child").rotate_euler(y=180, degrees=True)

        assert child.unit_x.frame == child
        assert_allclose(child.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert child.unit_y.frame == child
        assert_allclose(child.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert child.unit_z.frame == child
        assert_allclose(child.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert child.unit_x_global.frame == Frame.global_frame()
        assert_allclose(child.unit_x_global, [-1.0, 0.0, 0.0], atol=VVSMALL)
        assert child.unit_y_global.frame == Frame.global_frame()
        assert_allclose(child.unit_y_global, [0.0, 0.0, 1.0], atol=VVSMALL)
        assert child.unit_z_global.frame == Frame.global_frame()
        assert_allclose(child.unit_z_global, [0.0, 1.0, 0.0], atol=VVSMALL)

    def test_frame_unit_vectors_translation(self):
        parent = Frame(parent=Frame.global_frame(), name="parent").translate(x=1)

        assert parent.origin.frame == parent
        assert_allclose(parent.origin, [0.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert parent.origin_global.frame == Frame.global_frame()
        assert_allclose(parent.origin_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.unit_x_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.unit_y_global, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.unit_z_global, [0.0, 0.0, 1.0], atol=VVSMALL)

        child = Frame(parent=parent, name="child").translate(y=2)

        assert child.origin.frame == child
        assert_allclose(child.origin, [0.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(child.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(child.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(child.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert child.origin_global.frame == Frame.global_frame()
        assert_allclose(child.origin_global, [1.0, 2.0, 0.0], atol=VVSMALL)
        assert_allclose(child.unit_x_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(child.unit_y_global, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(child.unit_z_global, [0.0, 0.0, 1.0], atol=VVSMALL)

    def test_frame_unit_vectors_uniform_scale(self):
        frame = Frame(parent=Frame.global_frame(), name="test")
        frame.scale(2.0)

        assert_allclose(frame.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert_allclose(frame.unit_x_global, [2.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_y_global, [0.0, 2.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_z_global, [0.0, 0.0, 2.0], atol=VVSMALL)

    def test_frame_unit_vectors_non_uniform_scale(self):
        frame = Frame(parent=Frame.global_frame(), name="test")
        frame.scale((2.0, 3.0, 4.0))

        assert_allclose(frame.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert_allclose(frame.unit_x_global, [2.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_y_global, [0.0, 3.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_z_global, [0.0, 0.0, 4.0], atol=VVSMALL)

    def test_frame_unit_vectors_scale_and_rotation(self):
        frame = Frame(parent=Frame.global_frame(), name="test")
        frame.scale(2.0)
        frame.rotate_euler(z=90, degrees=True)

        assert_allclose(frame.unit_x, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_y, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_z, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert_allclose(frame.unit_x_global, [0.0, 2.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_y_global, [-2.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.unit_z_global, [0.0, 0.0, 2.0], atol=VVSMALL)


@pytest.mark.unit
class TestFreeze:
    def test_freeze_frame_rotation(self):
        frame = Frame(parent=Frame.global_frame())
        frame.freeze()
        with pytest.raises(RuntimeError, match="Can not modify frozen frame."):
            frame.rotate_euler(x=0.0)
        frame.unfreeze()
        frame.rotate_euler(x=0.0)

    def test_freeze_frame_translation(self):
        frame = Frame(parent=Frame.global_frame())
        frame.freeze()
        with pytest.raises(RuntimeError, match="Can not modify frozen frame."):
            frame.translate(x=0.0)
        frame.unfreeze()
        frame.translate(x=0.0)

    def test_freeze_frame_scale(self):
        frame = Frame(parent=Frame.global_frame())
        frame.freeze()
        with pytest.raises(RuntimeError, match="Can not modify frozen frame."):
            frame.scale(1.0)
        frame.unfreeze()
        frame.scale(1.0)


@pytest.mark.unit
class TestTransformations:
    def test_identity_to_global(self):
        frame = Frame(parent=Frame.global_frame())

        assert isinstance(frame.transform_to_global, np.ndarray)
        assert frame.transform_from_global.shape == (4, 4)
        expected = np.eye(4)
        assert_allclose(frame.transform_to_global, expected)

    def test_identity_to_parent(self):
        frame = Frame(parent=Frame.global_frame())

        assert isinstance(frame.transform_to_parent, np.ndarray)
        assert frame.transform_to_parent.shape == (4, 4)
        expected = np.eye(4)
        assert_allclose(frame.transform_to_parent, expected)

    def test_identity_to_frame(self):
        global_frame = Frame.global_frame()
        frame = Frame(parent=global_frame)

        assert isinstance(global_frame.transform_to(frame), np.ndarray)
        assert global_frame.transform_to(frame).shape == (4, 4)
        expected = np.eye(4)
        assert_allclose(global_frame.transform_to(frame), expected)

    def test_translation_to_parent(self):
        frame = Frame(parent=Frame.global_frame()).translate(x=1, y=2, z=3)
        expected = np.eye(4)
        expected[:3, 3] = [1, 2, 3]
        assert_allclose(frame.transform_to_parent, expected)

    def test_rotation_to_parent(self):
        frame = Frame(parent=Frame.global_frame()).rotate_euler(
            x=np.pi / 2, y=np.pi / 2, z=np.pi / 2
        )

        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler(
            "xyz", [np.pi / 2, np.pi / 2, np.pi / 2]
        ).as_matrix()
        assert_allclose(frame.transform_to_parent, expected)

    def test_scale_to_parent(self):
        frame = Frame(parent=Frame.global_frame()).scale(3)
        expected = np.diag([3, 3, 3, 1])
        assert_allclose(frame.transform_to_parent, expected)

        frame = Frame(parent=Frame.global_frame()).scale((2, 3, 4))
        expected = np.diag([2, 3, 4, 1])
        assert_allclose(frame.transform_to_parent, expected)

    def test_transform_from_parent_is_inverse(self):
        frame = (
            Frame(parent=Frame.global_frame())
            .translate(x=1, y=2, z=3)
            .rotate_euler(z=45, degrees=True)
            .scale(2.0)
        )

        product = frame.transform_to_parent @ frame.transform_from_parent
        assert_allclose(product, np.eye(4), atol=1e-10)

    def test_transform_from_global_is_inverse(self):
        parent = Frame(parent=Frame.global_frame()).translate(x=5)
        child = Frame(parent=parent).rotate_euler(y=90, degrees=True)

        product = child.transform_to_global @ child.transform_from_global
        assert_allclose(product, np.eye(4), atol=1e-10)

    def test_transform_to_self_is_identity(self):
        frame = Frame(parent=Frame.global_frame()).translate(x=1).scale(2.0)

        assert_allclose(frame.transform_to(frame), np.eye(4))


@pytest.mark.unit
class TestScaling:
    def test_scale_with_invalid_tuple_raises_value_error(self):
        frame = Frame(parent=Frame.global_frame())

        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0))

        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0, 3.0, 4.0))


@pytest.mark.unit
class TestFactoryMethods:
    def test_create_vector(self):
        frame = Frame(parent=Frame.global_frame())
        vector = frame.create_vector(1.0, 2.0, 3.0)

        assert vector.x == 1.0
        assert vector.y == 2.0
        assert vector.z == 3.0
        assert vector.frame is frame

    def test_create_point(self):
        frame = Frame(parent=Frame.global_frame())
        point = frame.create_point(4.0, 5.0, 6.0)

        assert point.x == 4.0
        assert point.y == 5.0
        assert point.z == 6.0
        assert point.frame is frame


@pytest.mark.unit
class TestBatchTransform:
    def test_batch_transform_global_single_point(self):
        frame = Frame(parent=Frame.global_frame()).translate(x=1, y=2, z=3)
        points = np.array([[0, 0, 0]])

        transformed = frame.batch_transform_global(points)

        assert_allclose(transformed, [[1, 2, 3]], atol=VVSMALL)

    def test_batch_transform_global_multiple_points(self):
        frame = Frame(parent=Frame.global_frame()).translate(x=1, y=0, z=0)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        transformed = frame.batch_transform_global(points)

        expected = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]])
        assert_allclose(transformed, expected, atol=VVSMALL)

    def test_batch_transform_global_with_rotation(self):
        frame = Frame(parent=Frame.global_frame()).rotate_euler(z=90, degrees=True)
        points = np.array([[1, 0, 0], [0, 1, 0]])

        transformed = frame.batch_transform_global(points)

        expected = np.array([[0, 1, 0], [-1, 0, 0]])
        assert_allclose(transformed, expected, atol=VVSMALL)


@pytest.mark.unit
class TestTransformationInverses:
    @given(tx=small_coords, ty=small_coords, tz=small_coords, angle=angles, scale=scales)
    def test_transform_to_parent_inverse(self, tx, ty, tz, angle, scale):
        """transform_to_parent @ transform_from_parent = Identity"""
        frame = (
            Frame(parent=Frame.global_frame())
            .translate(x=tx, y=ty, z=tz)
            .rotate_euler(z=angle, degrees=True)
            .scale(scale)
        )

        product = frame.transform_to_parent @ frame.transform_from_parent
        assert_allclose(product, np.eye(4), atol=1e-10)

    @given(tx=small_coords, ty=small_coords, tz=small_coords, angle=angles)
    def test_transform_to_global_inverse(self, tx, ty, tz, angle):
        """transform_to_global @ transform_from_global = Identity"""
        parent = Frame(parent=Frame.global_frame()).translate(x=tx, y=ty, z=tz)
        child = Frame(parent=parent).rotate_euler(z=angle, degrees=True)

        product = child.transform_to_global @ child.transform_from_global
        assert_allclose(product, np.eye(4), atol=1e-10)

    @given(
        tx1=small_coords,
        ty1=small_coords,
        tx2=small_coords,
        ty2=small_coords,
        angle=angles,
    )
    def test_transform_to_is_inverse_of_transform_from(self, tx1, ty1, tx2, ty2, angle):
        """frame1.transform_to(frame2) is inverse of frame2.transform_to(frame1)"""
        global_frame = Frame.global_frame()
        frame1 = Frame(parent=global_frame).translate(x=tx1, y=ty1)
        frame2 = Frame(parent=global_frame).translate(x=tx2, y=ty2).rotate_euler(
            z=angle, degrees=True
        )

        t_1to2 = frame1.transform_to(frame2)
        t_2to1 = frame2.transform_to(frame1)

        assert_allclose(t_1to2 @ t_2to1, np.eye(4), atol=1e-10)


@pytest.mark.unit
class TestTranslationProperties:
    @given(tx1=small_coords, ty1=small_coords, tx2=small_coords, ty2=small_coords)
    def test_translation_composition(self, tx1, ty1, tx2, ty2):
        """Multiple translations should accumulate"""
        frame = (
            Frame(parent=Frame.global_frame())
            .translate(x=tx1, y=ty1)
            .translate(x=tx2, y=ty2)
        )

        expected_translation = np.array([tx1 + tx2, ty1 + ty2, 0.0])
        assert_allclose(frame.combined_translation, expected_translation, atol=1e-10)

    @given(tx=small_coords, ty=small_coords, tz=small_coords)
    def test_translation_only_affects_position(self, tx, ty, tz):
        """Translation should not affect rotation or scale"""
        frame = Frame(parent=Frame.global_frame()).translate(x=tx, y=ty, z=tz)

        assert_allclose(
            frame.combined_rotation.as_matrix(), IDENTITY_ROTATION.as_matrix(), atol=1e-10
        )
        assert_allclose(np.diagonal(frame.combined_scale)[:3], IDENTITY_SCALE, atol=1e-10)


@pytest.mark.unit
class TestRotationProperties:
    @given(angle1=angles, angle2=angles)
    def test_rotation_composition(self, angle1, angle2):
        """Multiple rotations should compose"""
        frame = (
            Frame(parent=Frame.global_frame())
            .rotate_euler(z=angle1, degrees=True)
            .rotate_euler(z=angle2, degrees=True)
        )

        expected_angle = angle1 + angle2
        expected_rotation = Rotation.from_euler("z", expected_angle, degrees=True)

        assert_allclose(
            frame.combined_rotation.as_matrix(),
            expected_rotation.as_matrix(),
            atol=1e-10,
        )

    @given(angle=angles)
    def test_rotation_inverse(self, angle):
        """Rotation followed by inverse rotation should be identity"""
        frame = (
            Frame(parent=Frame.global_frame())
            .rotate_euler(z=angle, degrees=True)
            .rotate_euler(z=-angle, degrees=True)
        )

        assert_allclose(
            frame.combined_rotation.as_matrix(), IDENTITY_ROTATION.as_matrix(), atol=1e-10
        )


@pytest.mark.unit
class TestScalingProperties:
    @given(s1=scales, s2=scales)
    def test_uniform_scale_composition(self, s1, s2):
        """Multiple uniform scales should multiply"""
        frame = Frame(parent=Frame.global_frame()).scale(s1).scale(s2)

        expected = s1 * s2
        assert_allclose(np.diagonal(frame.combined_scale)[:3], [expected] * 3, atol=1e-10)

    @given(
        sx1=scales,
        sy1=scales,
        sz1=scales,
        sx2=scales,
        sy2=scales,
        sz2=scales,
    )
    def test_non_uniform_scale_composition(self, sx1, sy1, sz1, sx2, sy2, sz2):
        """Multiple non-uniform scales should multiply component-wise"""
        frame = Frame(parent=Frame.global_frame()).scale((sx1, sy1, sz1)).scale(
            (sx2, sy2, sz2)
        )

        expected = [sx1 * sx2, sy1 * sy2, sz1 * sz2]
        assert_allclose(np.diagonal(frame.combined_scale)[:3], expected, atol=1e-10)

    @given(scale=scales)
    def test_scale_inverse(self, scale):
        """Scale followed by inverse scale should be identity"""
        frame = Frame(parent=Frame.global_frame()).scale(scale).scale(1.0 / scale)

        assert_allclose(np.diagonal(frame.combined_scale)[:3], [1, 1, 1], atol=1e-10)


@pytest.mark.unit
class TestCacheInvalidation:
    @given(tx=small_coords, ty=small_coords)
    def test_cache_invalidated_after_translate(self, tx, ty):
        """Cache should be invalidated after translation"""
        frame = Frame(parent=Frame.global_frame())

        _ = frame.transform_to_parent
        assert frame._cached_transform is not None

        frame.translate(x=tx, y=ty)
        assert frame._cached_transform is None

    @given(angle=angles)
    def test_cache_invalidated_after_rotate(self, angle):
        """Cache should be invalidated after rotation"""
        frame = Frame(parent=Frame.global_frame())

        _ = frame.transform_to_parent
        assert frame._cached_transform is not None

        frame.rotate_euler(z=angle, degrees=True)
        assert frame._cached_transform is None

    @given(scale=scales)
    def test_cache_invalidated_after_scale(self, scale):
        """Cache should be invalidated after scaling"""
        frame = Frame(parent=Frame.global_frame())

        _ = frame.transform_to_parent
        assert frame._cached_transform is not None

        frame.scale(scale)
        assert frame._cached_transform is None


@pytest.mark.unit
class TestHierarchyDepth:
    @given(depth=integers(min_value=1, max_value=10), offset=small_coords)
    def test_deep_hierarchy_accumulation(self, depth, offset):
        """Transformations should accumulate correctly through deep hierarchies"""
        current = Frame.global_frame()

        for _ in range(depth):
            current = Frame(parent=current).translate(x=offset)

        expected = np.eye(4)
        expected[0, 3] = depth * offset

        assert_allclose(current.transform_to_global, expected, atol=1e-8)

    @given(depth=integers(min_value=1, max_value=10))
    def test_deep_hierarchy_root_finding(self, depth):
        """Root should be found correctly regardless of depth"""
        root = Frame.create_orphan(name="root")
        current = root

        for i in range(depth):
            current = Frame(parent=current, name=f"child_{i}")

        assert current.root is root


@pytest.mark.unit
class TestTransformationOrder:
    @given(tx=small_coords, angle=angles, scale=scales)
    def test_srt_order(self, tx, angle, scale):
        """Transformations should be applied in S->R->T order"""
        frame = (
            Frame(parent=Frame.global_frame())
            .scale(scale)
            .rotate_euler(z=angle, degrees=True)
            .translate(x=tx)
        )

        expected = np.eye(4)
        expected = np.diag([scale, scale, scale, 1]) @ expected
        expected[:3, :3] = (
            Rotation.from_euler("z", angle, degrees=True).as_matrix() @ expected[:3, :3]
        )
        expected[:3, 3] += [tx, 0, 0]

        assert_allclose(frame.transform_to_parent, expected, atol=1e-10)


@pytest.mark.unit
class TestTransformCopies:
    @given(tx=small_coords)
    def test_transform_to_parent_returns_copy(self, tx):
        """transform_to_parent should return a copy, not a reference"""
        frame = Frame(parent=Frame.global_frame()).translate(x=tx)

        t1 = frame.transform_to_parent
        t2 = frame.transform_to_parent

        assert t1 is not t2
        assert_allclose(t1, t2)

    @given(tx=small_coords)
    def test_transform_to_global_returns_copy(self, tx):
        """transform_to_global should return a copy, not a reference"""
        frame = Frame(parent=Frame.global_frame()).translate(x=tx)

        t1 = frame.transform_to_global
        t2 = frame.transform_to_global

        assert t1 is not t2
        assert_allclose(t1, t2)
