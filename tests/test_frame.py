from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import composite, floats, integers
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from hazy import Frame
from hazy.constants import (
    IDENTITY_ROTATION,
    IDENTITY_SCALE,
    IDENTITY_TRANSLATION,
    VVSMALL,
)
from tests.conftest import (
    rotation_lists,
    scaling_lists,
    translation_lists,
    vector_3d_lists,
)

coords = floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
small_coords = floats(
    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
)
angles = floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
scales = floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False)


@pytest.mark.unit
class TestFrameCreation:
    def test_return_global_singleton(self):
        frame = Frame.make_root(name="global")

        assert frame.name == "global"
        assert frame.parent is None

    def test_creation_with_parent_and_name(self):
        root = Frame.make_root(name="root")
        frame = root.make_child(name="some sub-frame")

        assert frame.parent is root
        assert frame.name == "some sub-frame"

    def test_auto_generated_name(self):
        parent = Frame.make_root(name="global")
        frame = parent.make_child()

        assert frame.name.startswith("Frame-")
        assert frame.parent is parent


@pytest.mark.unit
class TestFrameHierarchy:
    def test_hierarchical_frames(self):
        global_frame = Frame.make_root(name="global")
        parent_frame = global_frame.make_child(name="parent")
        child_frame = parent_frame.make_child(name="child")

        assert child_frame.parent is parent_frame
        assert parent_frame.parent is global_frame
        assert global_frame.parent is None

    def test_separate_hierarchies_raises_RuntimeError(self):
        root1 = Frame.make_root("root1")
        root2 = Frame.make_root("root2")

        child1 = root1.make_child("child1")
        child2 = root2.make_child("child2")

        with pytest.raises(
            RuntimeError,
            match="Cannot transform between frames from different hierarchies",
        ):
            child1.transform_to(child2)


@pytest.mark.unit
class TestFrameInitialState:
    def test_initial_transform_state(self, root_frame: Frame):
        frame = root_frame.make_child()

        assert len(frame._rotations) == 1
        assert len(frame._translations) == 1
        assert len(frame._scalings) == 1
        assert frame._cached_transform is None
        assert frame._cached_transform_global is None
        assert frame._is_frozen is False

    def test_identity_transform_initialization(self, root_frame: Frame):
        frame = root_frame.make_child()

        assert_allclose(frame._rotations[0].as_matrix(), IDENTITY_ROTATION.as_matrix())
        assert_allclose(frame._translations[0], IDENTITY_TRANSLATION)
        assert_allclose(frame._scalings[0], IDENTITY_SCALE)

    def test_identity_transform_combination(self, root_frame: Frame):
        frame = root_frame.make_child()

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
    def test_global_frame_unit_vectors(self, root_frame: Frame):
        frame = root_frame.make_child()

        assert frame.x_axis == frame.x_axis_global
        assert frame.y_axis == frame.y_axis_global
        assert frame.z_axis == frame.z_axis_global

    def test_frame_unit_vectors_rotation(self, root_frame: Frame):
        parent = root_frame.make_child(name="parent").rotate_euler(x=90, degrees=True)

        assert parent.x_axis.frame == parent
        assert_allclose(parent.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert parent.y_axis.frame == parent
        assert_allclose(parent.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert parent.z_axis.frame == parent
        assert_allclose(parent.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert parent.x_axis_global.frame is root_frame
        assert_allclose(parent.x_axis_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert parent.y_axis_global.frame is root_frame
        assert_allclose(parent.y_axis_global, [0.0, 0.0, 1.0], atol=VVSMALL)
        assert parent.z_axis_global.frame is root_frame
        assert_allclose(parent.z_axis_global, [0.0, -1.0, 0.0], atol=VVSMALL)

        child = parent.make_child(name="child").rotate_euler(y=180, degrees=True)

        assert child.x_axis.frame == child
        assert_allclose(child.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert child.y_axis.frame == child
        assert_allclose(child.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert child.z_axis.frame == child
        assert_allclose(child.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert child.x_axis_global.frame is root_frame
        assert_allclose(child.x_axis_global, [-1.0, 0.0, 0.0], atol=VVSMALL)
        assert child.y_axis_global.frame is root_frame
        assert_allclose(child.y_axis_global, [0.0, 0.0, 1.0], atol=VVSMALL)
        assert child.z_axis_global.frame is root_frame
        assert_allclose(child.z_axis_global, [0.0, 1.0, 0.0], atol=VVSMALL)

    def test_frame_unit_vectors_translation(self, root_frame: Frame):
        parent = root_frame.make_child(name="parent").translate(x=1)

        assert parent.origin.frame == parent
        assert_allclose(parent.origin, [0.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert parent.origin_global.frame is root_frame
        assert_allclose(parent.origin_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.x_axis_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.y_axis_global, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(parent.z_axis_global, [0.0, 0.0, 1.0], atol=VVSMALL)

        child = parent.make_child(name="child").translate(y=2)

        assert child.origin.frame == child
        assert_allclose(child.origin, [0.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(child.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(child.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(child.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert child.origin_global.frame is root_frame
        assert_allclose(child.origin_global, [1.0, 2.0, 0.0], atol=VVSMALL)
        assert_allclose(child.x_axis_global, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(child.y_axis_global, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(child.z_axis_global, [0.0, 0.0, 1.0], atol=VVSMALL)

    def test_frame_unit_vectors_uniform_scale(self):
        frame = Frame(parent=Frame(parent=None, name="global"), name="test")
        frame.scale(2.0)

        assert_allclose(frame.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert_allclose(frame.x_axis_global, [2.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.y_axis_global, [0.0, 2.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.z_axis_global, [0.0, 0.0, 2.0], atol=VVSMALL)

    def test_frame_unit_vectors_non_uniform_scale(self):
        frame = Frame(parent=Frame(parent=None, name="global"), name="test")
        frame.scale((2.0, 3.0, 4.0))

        assert_allclose(frame.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert_allclose(frame.x_axis_global, [2.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.y_axis_global, [0.0, 3.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.z_axis_global, [0.0, 0.0, 4.0], atol=VVSMALL)

    def test_frame_unit_vectors_scale_and_rotation(self):
        frame = Frame(parent=Frame(parent=None, name="global"), name="test")
        frame.scale(2.0)
        frame.rotate_euler(z=90, degrees=True)

        assert_allclose(frame.x_axis, [1.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.y_axis, [0.0, 1.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.z_axis, [0.0, 0.0, 1.0], atol=VVSMALL)

        assert_allclose(frame.x_axis_global, [0.0, 2.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.y_axis_global, [-2.0, 0.0, 0.0], atol=VVSMALL)
        assert_allclose(frame.z_axis_global, [0.0, 0.0, 2.0], atol=VVSMALL)


@pytest.mark.unit
class TestFreeze:
    def test_freeze_frame_rotation(self):
        frame = Frame(parent=Frame(parent=None, name="global"))
        frame.freeze()
        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.rotate_euler(x=0.0)
        frame.unfreeze()
        frame.rotate_euler(x=0.0)

    def test_freeze_frame_translation(self):
        frame = Frame(parent=Frame(parent=None, name="global"))
        frame.freeze()
        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.translate(x=0.0)
        frame.unfreeze()
        frame.translate(x=0.0)

    def test_freeze_frame_scale(self):
        frame = Frame(parent=Frame(parent=None, name="global"))
        frame.freeze()
        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.scale(1.0)
        frame.unfreeze()
        frame.scale(1.0)


@pytest.mark.unit
class TestTransformations:
    def test_identity_to_global(self, frame: Frame):
        assert isinstance(frame.transform_to_global, np.ndarray)
        assert frame.transform_from_global.shape == (4, 4)
        expected = np.eye(4)
        assert_allclose(frame.transform_to_global, expected)

    def test_identity_to_parent(self, frame: Frame):
        assert isinstance(frame.transform_to_parent, np.ndarray)
        assert frame.transform_to_parent.shape == (4, 4)
        expected = np.eye(4)
        assert_allclose(frame.transform_to_parent, expected)

    def test_identity_to_frame(self):
        global_frame = Frame.make_root(name="global")
        frame = global_frame.make_child()

        assert isinstance(global_frame.transform_to(frame), np.ndarray)
        assert global_frame.transform_to(frame).shape == (4, 4)
        expected = np.eye(4)
        assert_allclose(global_frame.transform_to(frame), expected)

    def test_translation_to_parent(self, frame: Frame):
        frame.translate(x=1, y=2, z=3)
        expected = np.eye(4)
        expected[:3, 3] = [1, 2, 3]
        assert_allclose(frame.transform_to_parent, expected)

    def test_rotation_to_parent(self, frame: Frame):
        frame.rotate_euler(x=np.pi / 2, y=np.pi / 2, z=np.pi / 2)

        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler(
            "xyz", [np.pi / 2, np.pi / 2, np.pi / 2]
        ).as_matrix()
        assert_allclose(frame.transform_to_parent, expected)

    def test_scale_to_parent(self, frame: Frame):
        frame.scale(3)
        expected = np.diag([3, 3, 3, 1])
        assert_allclose(frame.transform_to_parent, expected)

        frame.clear_all_transforms()
        frame.scale((2, 3, 4))
        expected = np.diag([2, 3, 4, 1])
        assert_allclose(frame.transform_to_parent, expected)

    def test_transform_from_parent_is_inverse(self, frame: Frame):
        frame.translate(x=1, y=2, z=3).rotate_euler(z=45, degrees=True).scale(2.0)

        product = frame.transform_to_parent @ frame.transform_from_parent
        assert_allclose(product, np.eye(4), atol=1e-10)

    def test_transform_from_global_is_inverse(self, frame: Frame):
        frame.rotate_euler(y=90, degrees=True)

        product = frame.transform_to_global @ frame.transform_from_global
        assert_allclose(product, np.eye(4), atol=1e-10)

    def test_transform_to_self_is_identity(self, frame: Frame):
        frame.translate(x=1).scale(2.0)

        assert_allclose(frame.transform_to(frame), np.eye(4))

    def test_clear_all_transforms(self, frame: Frame):
        frame.rotate_euler(x=45, degrees=True).rotate_euler(z=-45, degrees=True).scale(
            5
        ).scale(1, 4, 5).translate(5, 5, 5).translate((4, 3, 2))

        # Verify transformations exist (not identity)
        assert not np.allclose(frame.transform_to_global, np.eye(4))

        frame.clear_all_transforms()

        # Verify frame is back to identity transformation
        assert_allclose(frame.transform_to_global, np.eye(4))


@pytest.mark.unit
class TestScaling:
    def test_scale_with_invalid_tuple_raises_value_error(self, frame: Frame):
        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0))

        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0, 3.0, 4.0))

    def test_scale_invalid_sequence_scalar_raises_value_error(self, frame: Frame):
        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0, 3.0), 1.0)

        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0, 3.0), z=5.0)

    def test_scale_three_scalars(self, frame: Frame):
        frame.scale(1, 2, 3)

        assert_allclose(np.diag([1, 2, 3, 1]), frame.combined_scale)

    def test_scale_two_scalars_raises_value_error(self, frame: Frame):
        with pytest.raises(ValueError, match="Provide either"):
            frame.scale(1.0, 2.0)

    def test_clear_scaling_only_scaling(self, frame: Frame):
        frame.scale(5)

        # Verify scaling was applied
        assert_allclose(frame.combined_scale, np.diag([5, 5, 5, 1]))

        frame.clear_scalings()

        # Verify scaling is back to identity
        assert_allclose(frame.combined_scale, np.diag([1, 1, 1, 1]))

    def test_clear_scaling_mixed_transforms(self, frame: Frame):
        frame.scale(5).rotate_euler(x=45, degrees=True).translate(5, 4, 3)

        # Store transformation before clearing scaling
        transform_before = frame.transform_to_global.copy()

        frame.clear_scalings()

        # Verify scaling is identity but other transforms remain
        assert_allclose(frame.combined_scale, np.diag([1, 1, 1, 1]))
        assert not np.allclose(frame.combined_rotation.as_matrix(), np.eye(3))
        assert not np.allclose(frame.combined_translation, np.zeros(3))

        # Verify overall transformation changed
        assert not np.allclose(frame.transform_to_global, transform_before)


@pytest.mark.unit
class TestRotations:
    def test_rotate_with_invalid_args_raises_value_error(self, frame: Frame):
        with pytest.raises(ValueError, match="Expected `matrix` to have shape"):
            frame.rotate(5)

        with pytest.raises(ValueError, match="Expected `matrix` to have shape"):
            frame.rotate((5, 5, 5))

        with pytest.raises(ValueError, match="Expected `matrix` to have shape"):
            frame.rotate(np.ones((4, 4)))

    def test_rotate_euler_invalid_args_raises_value_error(self, frame: Frame):
        with pytest.raises(ValueError, match="setting an array element"):
            frame.rotate_euler(x=(1, 2, 3))

        with pytest.raises(ValueError, match="setting an array element"):
            frame.rotate_euler(x=5, y=(1, 2, 3), degrees=True)

        with pytest.raises(
            TypeError, match="Frame\\.rotate_euler\\(\\) takes 1 positional"
        ):
            frame.rotate_euler(5, degrees=True)

    def test_rotate_quaternion_invalid_args_raises_value_error(self, frame: Frame):
        with pytest.raises(
            ValueError, match="Quaternion must have shape \\(4,\\), got .*"
        ):
            frame.rotate_quaternion((1, 2, 3))

        with pytest.raises(
            ValueError, match="Quaternion must have shape \\(4,\\), got .*"
        ):
            frame.rotate_quaternion((1, 2, 3, 5, 6))

        with pytest.raises(
            ValueError, match="Quaternion must have shape \\(4,\\), got .*"
        ):
            frame.rotate_quaternion(1)

    def test_rotate_quaternion(self, frame: Frame):
        quat = [0, 0, 0, 1]
        frame.rotate_quaternion(quat)
        np.allclose(frame.combined_rotation.as_matrix(), np.eye(3))

    def test_clear_rotation_only_rotation(self):
        root = Frame.make_root("root")
        frame = (
            root.make_child("child")
            .rotate_euler(x=5, degrees=True)
            .rotate_euler(y=45, degrees=True)
        )

        # Verify rotation was applied
        assert not np.allclose(frame.combined_rotation.as_matrix(), np.eye(3))

        frame.clear_rotations()

        # Verify rotation is back to identity
        assert_allclose(frame.combined_rotation.as_matrix(), np.eye(3))


@pytest.mark.unit
class TestTranslation:
    def test_translate_with_invalid_args_raises_value_error(self, frame: Frame):
        with pytest.raises(
            ValueError, match="y and z parameter are not supported if x is a Sequence"
        ):
            frame.translate((3, 5, 1), 1.0)

        with pytest.raises(
            ValueError, match="y and z parameter are not supported if x is a Sequence"
        ):
            frame.translate((3, 5, 1), 1.0, 5.0)

        with pytest.raises(ValueError, match="Can not translate by x"):
            frame.translate((5, 5, 5, 5))

        with pytest.raises(ValueError, match="Can not translate by x"):
            frame.translate(np.ones((4, 4)))

    def test_clear_translations_only_translation(self, frame: Frame):
        frame.translate(1, 1, 1).translate((-3, 3, -3))

        # Verify translation was applied
        assert not np.allclose(frame.combined_translation, np.zeros(3))

        frame.clear_translations()

        # Verify translation is back to zero
        assert_allclose(frame.combined_translation, np.zeros(3))


@pytest.mark.unit
class TestClearEdgeCases:
    def test_clear_on_frozen_frame_raises_error(self, frame: Frame):
        frame.scale(2).translate(x=1).freeze()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_scalings()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_rotations()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_translations()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_all_transforms()

    def test_clear_already_empty_transforms(self, frame: Frame):
        # Clear without any transformations applied
        frame.clear_scalings()
        frame.clear_rotations()
        frame.clear_translations()

        # Should remain identity
        assert_allclose(frame.transform_to_global, np.eye(4))

    def test_multiple_clears_in_sequence(self, frame: Frame):
        frame.scale(2).translate(x=5).rotate_euler(z=45, degrees=True)

        # First clear
        frame.clear_all_transforms()
        assert_allclose(frame.transform_to_global, np.eye(4))

        # Apply new transforms
        frame.translate(y=10)
        assert not np.allclose(frame.transform_to_global, np.eye(4))

        # Second clear
        frame.clear_all_transforms()
        assert_allclose(frame.transform_to_global, np.eye(4))

    def test_clear_preserves_parent_child_relationship(self, frame: Frame):
        frame.translate(x=10)
        child = frame.make_child("child").translate(y=5)

        # Clear parent transforms
        frame.clear_all_transforms()

        # Verify hierarchy is intact
        assert child.parent is frame
        assert frame.parent is child.root
        assert child in frame._children

    def test_selective_clears_independent(self, frame: Frame):
        frame.scale(2).translate(x=5).rotate_euler(z=45, degrees=True)

        # Clear only scaling
        frame.clear_scalings()
        assert_allclose(frame.combined_scale, np.diag([1, 1, 1, 1]))
        assert not np.allclose(frame.combined_rotation.as_matrix(), np.eye(3))
        assert not np.allclose(frame.combined_translation, np.zeros(3))

        # Clear only rotation (translation should still remain)
        frame.clear_rotations()
        assert_allclose(frame.combined_rotation.as_matrix(), np.eye(3))
        assert not np.allclose(frame.combined_translation, np.zeros(3))

        # Clear translation
        frame.clear_translations()
        assert_allclose(frame.transform_to_global, np.eye(4))


@pytest.mark.unit
class TestFactoryMethods:
    def test_create_vector(self, frame: Frame):
        vector = frame.vector(1.0, 2.0, 3.0)

        assert vector.x == 1.0
        assert vector.y == 2.0
        assert vector.z == 3.0
        assert vector.frame is frame

    def test_create_point(self, frame: Frame):
        point = frame.point(4.0, 5.0, 6.0)

        assert point.x == 4.0
        assert point.y == 5.0
        assert point.z == 6.0
        assert point.frame is frame

    def test_create_point_raises_value_error(self, frame: Frame):
        with pytest.raises(ValueError, match="x must be a scalar"):
            frame.point((1, 2, 3), 4, 5)

        with pytest.raises(ValueError, match="Provide either"):
            frame.point((1, 2, 3), 5)


@pytest.mark.unit
class TestBatchTransform:
    @given(vector_3d_lists(min_size=1, max_size=10))
    def test_batch_transform_point_global(self, vector_3d_list):
        root = Frame.make_root("root")
        frame = root.make_child("frame")
        frame.translate(x=1, y=2, z=3)
        points = np.vstack(vector_3d_list)
        expected = points.copy() + [1, 2, 3]
        transformed = frame.batch_transform_points_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

    @given(vector_3d_lists(min_size=1, max_size=10))
    def test_batch_transform_point_global_with_rotation(self, vector_3d_list):
        # rotate around x
        root = Frame.make_root("root")
        frame = root.make_child("child").rotate_euler(x=90, degrees=True)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        expected = expected[:, [0, 2, 1]] * [1, -1, 1]
        transformed = frame.batch_transform_points_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

        # rotate around y
        root = Frame.make_root("root")
        frame = root.make_child("child").rotate_euler(y=90, degrees=True)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        expected = expected[:, [2, 1, 0]] * [1, 1, -1]
        transformed = frame.batch_transform_points_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

        # rotate around z
        root = Frame.make_root("root")
        frame = root.make_child("child").rotate_euler(z=90, degrees=True)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        expected = expected[:, [1, 0, 2]] * [-1, 1, 1]
        transformed = frame.batch_transform_points_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

    @given(vector_3d_lists(min_size=1, max_size=10))
    def test_batch_transform_vector_global_with_rotation(self, vector_3d_list):
        # rotate around x
        root = Frame.make_root("root")
        frame = root.make_child("child").rotate_euler(x=90, degrees=True)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        expected = expected[:, [0, 2, 1]] * [1, -1, 1]
        transformed = frame.batch_transform_vectors_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

        # rotate around y
        root = Frame.make_root("root")
        frame = root.make_child("child").rotate_euler(y=90, degrees=True)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        expected = expected[:, [2, 1, 0]] * [1, 1, -1]
        transformed = frame.batch_transform_vectors_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

        # rotate around z
        root = Frame.make_root("root")
        frame = root.make_child("child").rotate_euler(z=90, degrees=True)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        expected = expected[:, [1, 0, 2]] * [-1, 1, 1]
        transformed = frame.batch_transform_vectors_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

    @given(vector_3d_lists(min_size=1, max_size=10))
    def test_batch_transform_vector_global_with_translation(self, vector_3d_list):
        """Vectors not affected by translation."""
        # rotate around x
        root = Frame.make_root("root")
        frame = root.make_child("child").translate(x=1)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        transformed = frame.batch_transform_vectors_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

        # rotate around y
        root = Frame.make_root("root")
        frame = root.make_child("child").translate(y=1)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        transformed = frame.batch_transform_vectors_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)

        # rotate around z
        root = Frame.make_root("root")
        frame = root.make_child("child").translate(z=1)
        points = np.vstack(vector_3d_list)
        expected = points.copy()
        transformed = frame.batch_transform_vectors_global(points)
        assert_allclose(transformed, expected, atol=VVSMALL)


@pytest.mark.unit
class TestTransformationInverses:
    @given(
        tx=small_coords, ty=small_coords, tz=small_coords, angle=angles, scale=scales
    )
    def test_transform_to_parent_inverse(self, tx, ty, tz, angle, scale):
        """transform_to_parent @ transform_from_parent = Identity"""
        frame = (
            Frame(parent=Frame(parent=None, name="global"))
            .translate(x=tx, y=ty, z=tz)
            .rotate_euler(z=angle, degrees=True)
            .scale(scale)
        )  # explicit frame creation since fixtures may cause
        # unexpected behavior of hypothesis

        product = frame.transform_to_parent @ frame.transform_from_parent
        assert_allclose(product, np.eye(4), atol=1e-10)

    @given(tx=small_coords, ty=small_coords, tz=small_coords, angle=angles)
    def test_transform_to_global_inverse(self, tx, ty, tz, angle):
        """transform_to_global @ transform_from_global = Identity"""
        parent = Frame(parent=Frame(parent=None, name="global")).translate(
            x=tx, y=ty, z=tz
        )
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
        global_frame = Frame(parent=None, name="global")
        frame1 = Frame(parent=global_frame).translate(x=tx1, y=ty1)
        frame2 = (
            Frame(parent=global_frame)
            .translate(x=tx2, y=ty2)
            .rotate_euler(z=angle, degrees=True)
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
            Frame(parent=Frame(parent=None, name="global"))
            .translate(x=tx1, y=ty1)
            .translate(x=tx2, y=ty2)
        )

        expected_translation = np.array([tx1 + tx2, ty1 + ty2, 0.0])
        assert_allclose(frame.combined_translation, expected_translation, atol=1e-10)

    @given(tx=small_coords, ty=small_coords, tz=small_coords)
    def test_translation_only_affects_position(self, tx, ty, tz):
        """Translation should not affect rotation or scale"""
        frame = Frame(parent=Frame(parent=None, name="global")).translate(
            x=tx, y=ty, z=tz
        )

        assert_allclose(
            frame.combined_rotation.as_matrix(),
            IDENTITY_ROTATION.as_matrix(),
            atol=1e-10,
        )
        assert_allclose(
            np.diagonal(frame.combined_scale)[:3], IDENTITY_SCALE, atol=1e-10
        )


@pytest.mark.unit
class TestRotationProperties:
    @given(angle1=angles, angle2=angles)
    def test_rotation_composition(self, angle1, angle2):
        """Multiple rotations should compose"""
        frame = (
            Frame(parent=Frame(parent=None, name="global"))
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
            Frame(parent=Frame(parent=None, name="global"))
            .rotate_euler(z=angle, degrees=True)
            .rotate_euler(z=-angle, degrees=True)
        )

        assert_allclose(
            frame.combined_rotation.as_matrix(),
            IDENTITY_ROTATION.as_matrix(),
            atol=1e-10,
        )


@pytest.mark.unit
class TestScalingProperties:
    @given(s1=scales, s2=scales)
    def test_uniform_scale_composition(self, s1, s2):
        """Multiple uniform scales should multiply"""
        frame = Frame(parent=Frame(parent=None, name="global")).scale(s1).scale(s2)

        expected = s1 * s2
        assert_allclose(
            np.diagonal(frame.combined_scale)[:3], [expected] * 3, atol=1e-10
        )

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
        frame = (
            Frame(parent=Frame(parent=None, name="global"))
            .scale((sx1, sy1, sz1))
            .scale((sx2, sy2, sz2))
        )

        expected = [sx1 * sx2, sy1 * sy2, sz1 * sz2]
        assert_allclose(np.diagonal(frame.combined_scale)[:3], expected, atol=1e-10)

    @given(scale=scales)
    def test_scale_inverse(self, scale):
        """Scale followed by inverse scale should be identity"""
        frame = (
            Frame(parent=Frame(parent=None, name="global"))
            .scale(scale)
            .scale(1.0 / scale)
        )

        assert_allclose(np.diagonal(frame.combined_scale)[:3], [1, 1, 1], atol=1e-10)


@pytest.mark.unit
class TestCacheInvalidation:
    @given(tx=small_coords, ty=small_coords)
    def test_cache_invalidated_after_translate(self, tx, ty):
        """Cache should be invalidated after translation"""
        frame = Frame(parent=Frame(parent=None, name="global"))

        _ = frame.transform_to_parent
        assert frame._cached_transform is not None

        frame.translate(x=tx, y=ty)
        assert frame._cached_transform is None

    @given(angle=angles)
    def test_cache_invalidated_after_rotate(self, angle):
        """Cache should be invalidated after rotation"""
        frame = Frame(parent=Frame(parent=None, name="global"))

        _ = frame.transform_to_parent
        assert frame._cached_transform is not None

        frame.rotate_euler(z=angle, degrees=True)
        assert frame._cached_transform is None

    @given(scale=scales)
    def test_cache_invalidated_after_scale(self, scale):
        """Cache should be invalidated after scaling"""
        frame = Frame(parent=Frame(parent=None, name="global"))

        _ = frame.transform_to_parent
        assert frame._cached_transform is not None

        frame.scale(scale)
        assert frame._cached_transform is None

    def test_child_cache_invalidated_when_parent_modified(self):
        """Child frame caches should be invalidated when parent is modified"""
        root = Frame.make_root("root")
        child = root.make_child("child")
        grandchild = child.make_child("grandchild")

        # Populate caches
        _ = child.transform_to_global
        _ = grandchild.transform_to_global
        assert child._cached_transform_global is not None
        assert grandchild._cached_transform_global is not None

        # Modify root - should invalidate all descendant caches
        root.rotate_euler(z=45, degrees=True)
        assert child._cached_transform_global is None
        assert grandchild._cached_transform_global is None

    def test_sibling_caches_invalidated_independently(self):
        """Sibling frames should have independent cache invalidation"""
        root = Frame.make_root("root")
        child1 = root.make_child("child1")
        child2 = root.make_child("child2")

        # Populate caches
        _ = child1.transform_to_global
        _ = child2.transform_to_global
        assert child1._cached_transform_global is not None
        assert child2._cached_transform_global is not None

        # Modify child1 - should not affect child2
        child1.translate(x=1.0)
        assert child1._cached_transform_global is None
        assert child2._cached_transform_global is not None

    def test_persistent_primitives_updated_after_parent_rotation(self):
        """Points/Vectors reflect parent transformations via cache invalidation"""
        root = Frame.make_root("root")
        disk = root.make_child("disk")
        laser_frame = disk.make_child("laser").translate(y=1)

        # Create persistent primitives
        origin = laser_frame.point(0, 0, 0)
        direction = laser_frame.vector(1, 0, 0)

        # Initial state
        origin_global = origin.to_global()
        assert_allclose(np.array(origin_global), [0, 1, 0], atol=VVSMALL)

        # Rotate parent
        disk.rotate_euler(z=90, degrees=True)

        # Verify transformation is updated
        origin_global = origin.to_global()
        assert_allclose(np.array(origin_global), [-1, 0, 0], atol=1e-10)

        direction_global = direction.to_global()
        assert_allclose(np.array(direction_global), [0, 1, 0], atol=1e-10)


@pytest.mark.unit
class TestFrameImmutability:
    def test_parent_cannot_be_changed_after_creation(self):
        """Parent property should be immutable to maintain consistency"""
        root = Frame.make_root("root")
        child = root.make_child("child")
        other_parent = Frame.make_root("other")

        with pytest.raises(RuntimeError, match="Cannot change parent"):
            child.parent = other_parent

    def test_parent_setter_provides_helpful_error(self):
        """Parent setter should provide clear alternatives"""
        root = Frame.make_root("root")
        child = root.make_child("child")
        other_parent = Frame.make_root("other")

        try:
            child.parent = other_parent
        except RuntimeError as e:
            error_msg = str(e)
            assert "consistency" in error_msg
            assert "make_child" in error_msg
        else:
            pytest.fail("Expected RuntimeError")


@pytest.mark.unit
class TestChildTracking:
    def test_children_registered_on_creation(self):
        """Children should be automatically registered in parent's _children set"""
        root = Frame.make_root("root")
        child1 = root.make_child("child1")
        child2 = root.make_child("child2")

        assert len(root._children) == 2
        assert child1 in root._children
        assert child2 in root._children

    def test_children_set_prevents_duplicates(self):
        """Using a set should prevent duplicate children"""
        root = Frame.make_root("root")
        child = Frame(parent=root, name="child")

        # Manually try to add duplicate (shouldn't happen in normal usage)
        root._add_child(child)
        root._add_child(child)

        assert len(root._children) == 1


@pytest.mark.unit
class TestHierarchyDepth:
    @given(depth=integers(min_value=1, max_value=10), offset=small_coords)
    def test_deep_hierarchy_accumulation(self, depth, offset):
        """Transformations should accumulate correctly through deep hierarchies"""
        current = Frame(parent=None, name="global")

        for _ in range(depth):
            current = Frame(parent=current).translate(x=offset)

        expected = np.eye(4)
        expected[0, 3] = depth * offset

        assert_allclose(current.transform_to_global, expected, atol=1e-8)

    @given(depth=integers(min_value=1, max_value=10))
    def test_deep_hierarchy_root_finding(self, depth):
        """Root should be found correctly regardless of depth"""
        root = Frame(parent=None, name="root")
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
            Frame(parent=Frame(parent=None, name="global"))
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
        frame = Frame(parent=Frame(parent=None, name="global")).translate(x=tx)

        t1 = frame.transform_to_parent
        t2 = frame.transform_to_parent

        assert t1 is not t2
        assert_allclose(t1, t2)

    @given(tx=small_coords)
    def test_transform_to_global_returns_copy(self, tx):
        """transform_to_global should return a copy, not a reference"""
        frame = Frame(parent=Frame(parent=None, name="global")).translate(x=tx)

        t1 = frame.transform_to_global
        t2 = frame.transform_to_global

        assert t1 is not t2
        assert_allclose(t1, t2)


@pytest.mark.unit
class TestDeepCopy:
    def test_deepcopy_points_with_shared_frame(self):
        """Deepcopy of Points should share the same copied frame"""
        from copy import deepcopy

        root = Frame(parent=None, name="root")
        points = [root.point(x=i, y=0, z=0) for i in range(10)]

        copied_points = deepcopy(points)

        # All copied points should share the same (copied) frame
        assert all(p.frame is copied_points[0].frame for p in copied_points)
        # But the copied frame should be different from the original
        assert copied_points[0].frame is not root

    def test_deepcopy_vectors_with_shared_frame(self):
        """Deepcopy of Vectors should share the same copied frame"""
        from copy import deepcopy

        root = Frame(parent=None, name="root")
        vectors = [root.vector(x=i, y=1, z=0) for i in range(5)]

        copied_vectors = deepcopy(vectors)

        # All copied vectors should share the same (copied) frame
        assert all(v.frame is copied_vectors[0].frame for v in copied_vectors)
        # But the copied frame should be different from the original
        assert copied_vectors[0].frame is not root

    def test_deepcopy_frame_creates_independent_copy(self):
        """Deepcopy of a regular frame should create independent copy"""
        from copy import deepcopy

        root = Frame(parent=None, name="root")
        frame = Frame(parent=root, name="test").translate(x=1, y=2)

        copied_frame = deepcopy(frame)

        assert copied_frame is not frame
        assert copied_frame.parent is not root
        assert_allclose(copied_frame.combined_translation, frame.combined_translation)

        frame.translate(x=5)
        assert not np.allclose(
            copied_frame.combined_translation, frame.combined_translation
        )

    def test_deepcopy_frame_hierarchy(self):
        """Deepcopy of frame hierarchy should preserve structure"""
        from copy import deepcopy

        root = Frame(parent=None, name="root")
        parent = Frame(parent=root, name="parent").translate(x=1)
        child = Frame(parent=parent, name="child").translate(y=2)

        copied_child = deepcopy(child)

        assert copied_child is not child
        assert copied_child.parent is not parent
        assert copied_child.parent.parent.name == "root"

        expected_global = np.eye(4)
        expected_global[:3, 3] = [1, 2, 0]
        assert_allclose(copied_child.transform_to_global, expected_global)

    def test_deepcopy_preserves_transformations(self):
        """Deepcopy should preserve all transformations"""
        from copy import deepcopy

        root = Frame(parent=None, name="root")
        frame = (
            Frame(parent=root, name="test")
            .translate(x=1, y=2, z=3)
            .rotate_euler(z=45, degrees=True)
            .scale(2.0)
        )

        copied_frame = deepcopy(frame)

        assert_allclose(
            copied_frame.transform_to_parent, frame.transform_to_parent, atol=1e-10
        )
        assert_allclose(
            copied_frame.combined_translation, frame.combined_translation, atol=1e-10
        )
        assert_allclose(
            copied_frame.combined_rotation.as_matrix(),
            frame.combined_rotation.as_matrix(),
            atol=1e-10,
        )
        assert_allclose(copied_frame.combined_scale, frame.combined_scale, atol=1e-10)

    def test_deepcopy_frozen_state_preserved(self):
        """Deepcopy should preserve frozen state"""
        from copy import deepcopy

        root = Frame(parent=None, name="frozen_root")
        root.freeze()

        copied_root = deepcopy(root)

        assert copied_root._is_frozen is True
        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            copied_root.translate(x=1)


@pytest.mark.unit
class TestRepr:
    def test_frame_repr_root(self):
        root = Frame.make_root("root")
        r = repr(root)

        assert "root" in r
        assert "parent='None'" in r
        assert "0R+0T+0S" in r
        assert "FROZEN" not in r

    def test_frame_repr_child(self):
        root = Frame.make_root("root")
        child = root.make_child("child")
        r = repr(child)

        assert "child" in r
        assert "parent='root'" in r
        assert "0R+0T+0S" in r

    def test_frame_repr_transformed(self):
        root = Frame.make_root("root")
        child = (
            root.make_child("child").scale(1).translate(1, 1, 1).rotate_euler(x=3.141)
        )
        r = repr(child)

        assert "child" in r
        assert "parent='root'" in r
        assert "1R+1T+1S" in r
        assert "FROZEN" not in r

    def test_frame_repr_transformed_frozen(self):
        root = Frame.make_root("root")
        child = (
            root.make_child("child")
            .scale(1)
            .translate(1, 1, 1)
            .rotate_euler(x=3.141)
            .freeze()
        )
        r = repr(child)

        assert "child" in r
        assert "parent='root'" in r
        assert "1R+1T+1S" in r
        assert "FROZEN" in r


@composite
def positive_scalars(draw):
    """Generate positive scalar values for scaling."""
    return draw(floats(min_value=0.1, max_value=10))


@pytest.mark.unit
class TestClearMethodsProperties:
    """Property-based tests for clear methods using Hypothesis."""

    @given(
        scalings=scaling_lists(min_size=0, max_size=10),
        rotations=rotation_lists(min_size=0, max_size=10),
        translations=translation_lists(min_size=0, max_size=10),
    )
    def test_clear_all_always_returns_to_identity(
        self, scalings, rotations, translations
    ):
        """Property: clear_all always results in identity transform."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for kwargs in rotations:
            frame.rotate_euler(**kwargs)
        for x, y, z in translations:
            frame.translate(x, y, z)

        frame.clear_all_transforms()

        assert_allclose(frame.transform_to_global, np.eye(4), atol=1e-10)

    @given(
        scalings=scaling_lists(min_size=1, max_size=5),
        rotations=rotation_lists(min_size=1, max_size=5),
        translations=translation_lists(min_size=1, max_size=5),
    )
    def test_clear_is_idempotent(self, scalings, rotations, translations):
        """Property: clear_all is idempotent (calling twice = calling once)."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        frame.clear_all_transforms()
        first_clear = frame.transform_to_global.copy()

        frame.clear_all_transforms()
        second_clear = frame.transform_to_global

        assert_allclose(first_clear, second_clear, atol=1e-10)

    @given(
        scalings=scaling_lists(min_size=1, max_size=5),
        rotations=rotation_lists(min_size=1, max_size=5),
        translations=translation_lists(min_size=1, max_size=5),
    )
    def test_clear_rotation_preserves_scale_and_translation(
        self, scalings, rotations, translations
    ):
        """Property: clearing rotations never affects scale or translation."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        expected_scale = frame.combined_scale.copy()
        expected_translation = frame.combined_translation.copy()

        frame.clear_rotations()

        assert_allclose(frame.combined_scale, expected_scale, atol=1e-10)
        assert_allclose(frame.combined_translation, expected_translation, atol=1e-10)
        assert_allclose(frame.combined_rotation.as_matrix(), np.eye(3), atol=1e-10)

    @given(
        scalings=scaling_lists(min_size=1, max_size=5),
        rotations=rotation_lists(min_size=1, max_size=5),
        translations=translation_lists(min_size=1, max_size=5),
    )
    def test_clear_scaling_preserves_rotation_and_translation(
        self, scalings, rotations, translations
    ):
        """Property: clearing scalings never affects rotation or translation."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        expected_rotation = frame.combined_rotation.as_matrix().copy()
        expected_translation = frame.combined_translation.copy()

        frame.clear_scalings()

        assert_allclose(
            frame.combined_rotation.as_matrix(), expected_rotation, atol=1e-10
        )
        assert_allclose(frame.combined_translation, expected_translation, atol=1e-10)
        assert_allclose(frame.combined_scale, np.diag([1, 1, 1, 1]), atol=1e-10)

    @given(
        scalings=scaling_lists(min_size=1, max_size=5),
        rotations=rotation_lists(min_size=1, max_size=5),
        translations=translation_lists(min_size=1, max_size=5),
    )
    def test_clear_translation_preserves_rotation_and_scaling(
        self, scalings, rotations, translations
    ):
        """Property: clearing translations never affects rotation or scaling."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        expected_rotation = frame.combined_rotation.as_matrix().copy()
        expected_scale = frame.combined_scale.copy()

        frame.clear_translations()

        assert_allclose(
            frame.combined_rotation.as_matrix(), expected_rotation, atol=1e-10
        )
        assert_allclose(frame.combined_scale, expected_scale, atol=1e-10)
        assert_allclose(frame.combined_translation, np.zeros(3), atol=1e-10)

    @given(
        scalings=scaling_lists(min_size=1, max_size=5),
        rotations=rotation_lists(min_size=1, max_size=5),
        translations=translation_lists(min_size=1, max_size=5),
    )
    def test_frozen_frame_rejects_clear_operations(
        self, scalings, rotations, translations
    ):
        """Property: frozen frames always reject clear operations."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        frame.freeze()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_all_transforms()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_scalings()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_rotations()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_translations()

    @given(
        depth=integers(min_value=2, max_value=5),
        num_transforms=integers(min_value=1, max_value=3),
    )
    def test_clear_preserves_hierarchy(self, depth, num_transforms):
        """Property: clearing frames never breaks parent-child relationships."""
        frames = []
        current = Frame.make_root("root")
        frames.append(current)

        for i in range(depth - 1):
            child = current.make_child(f"child_{i}")
            for _ in range(num_transforms):
                child.translate(x=1).rotate_euler(z=45, degrees=True).scale(2)
            frames.append(child)
            current = child

        for frame in frames:
            frame.clear_all_transforms()

        for i, frame in enumerate(frames[1:], 1):
            assert frame.parent is frames[i - 1]
            assert frame in frames[i - 1]._children

    @given(
        scalings=scaling_lists(min_size=1, max_size=5),
        rotations=rotation_lists(min_size=1, max_size=5),
        translations=translation_lists(min_size=1, max_size=5),
    )
    def test_clear_and_reapply_gives_same_result(
        self, scalings, rotations, translations
    ):
        """Property: clearing and reapplying same transforms gives same result."""
        frame = Frame.make_root("test")

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        original_transform = frame.transform_to_global.copy()

        frame.clear_all_transforms()

        for s in scalings:
            frame.scale(s)
        for rot_args in rotations:
            frame.rotate_euler(**rot_args)
        for x, y, z in translations:
            frame.translate(x, y, z)

        reapplied_transform = frame.transform_to_global

        assert_allclose(original_transform, reapplied_transform, atol=1e-9)
