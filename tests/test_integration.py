from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

from hazy import Frame
from hazy.constants import VVSMALL
from hazy.primitives import Point, Vector
from tests.conftest import (
    rotation_lists,
    scaling_lists,
    scalings,
    translation_lists,
    translations,
)


@pytest.mark.integration
class TestMultiFrameTransformations:
    def test_point_transformation_through_hierarchy(self, root_frame: Frame):
        """Test point transformation through multi-level frame hierarchy"""
        # Use root_frame fixture
        robot_base = Frame(parent=root_frame, name="robot_base").translate(
            x=1, y=0, z=0
        )
        robot_arm = Frame(parent=robot_base, name="robot_arm").rotate_euler(
            z=90, degrees=True
        )
        end_effector = Frame(parent=robot_arm, name="end_effector").translate(
            x=0.5, y=0, z=0
        )

        point_local = Point(0.1, 0, 0, frame=end_effector)
        point_global = point_local.to_global()

        expected_x = 1.0
        expected_y = 0.5 + 0.1
        expected_z = 0.0
        assert_allclose(
            point_global, [expected_x, expected_y, expected_z], atol=VVSMALL
        )

    def test_vector_transformation_preserves_direction(self, root_frame: Frame):
        """Test that vectors transform correctly through rotated frames"""
        rotated = Frame(parent=root_frame, name="rotated").rotate_euler(
            z=90, degrees=True
        )

        vector_local = Vector(1, 0, 0, frame=rotated)
        vector_global = vector_local.to_global()

        assert_allclose(vector_global, [0, 1, 0], atol=VVSMALL)

    def test_point_arithmetic_across_frame_hierarchy(self, root_frame: Frame):
        """Test point arithmetic when transforming between frames"""
        frame_a = Frame(parent=root_frame, name="frame_a").translate(x=1)
        frame_b = Frame(parent=root_frame, name="frame_b").translate(y=1)

        point_a = Point(0, 0, 0, frame=frame_a)
        point_b = Point(0, 0, 0, frame=frame_b)

        point_a_global = point_a.to_global()
        point_b_global = point_b.to_global()

        displacement = point_b_global - point_a_global

        assert isinstance(displacement, Vector)
        assert_allclose(displacement, [-1, 1, 0], atol=VVSMALL)


@pytest.mark.integration
class TestComplexTransformations:
    def test_combined_rotation_translation_scale(self, root_frame: Frame):
        """Test complex transformation combining all types"""
        complex_frame = (
            Frame(parent=root_frame, name="complex")
            .scale(2.0)
            .rotate_euler(z=45, degrees=True)
            .translate(x=1, y=1, z=0)
        )

        point = Point(1, 0, 0, frame=complex_frame)
        point_global = point.to_global()

        cos45 = np.cos(np.radians(45))
        sin45 = np.sin(np.radians(45))
        expected_x = 1 + 2 * 1 * cos45
        expected_y = 1 + 2 * 1 * sin45
        expected_z = 0

        assert_allclose(point_global, [expected_x, expected_y, expected_z], atol=1e-10)

    def test_deep_hierarchy_accumulation(self, root_frame: Frame):
        """Test transformation accumulation through deep hierarchy"""

        current_frame = root_frame
        for i in range(5):
            current_frame = Frame(parent=current_frame, name=f"level_{i}").translate(
                x=1, y=0, z=0
            )

        point = Point(0, 0, 0, frame=current_frame)
        point_global = point.to_global()

        assert_allclose(point_global, [5, 0, 0], atol=VVSMALL)

    def test_non_uniform_scale_with_rotation(self, root_frame: Frame):
        """Test that non-uniform scaling works correctly with rotation"""
        frame = (
            Frame(parent=root_frame, name="test")
            .scale((2, 3, 1))
            .rotate_euler(z=90, degrees=True)
        )

        vector = Vector(1, 0, 0, frame=frame)
        vector_global = vector.to_global()

        assert_allclose(vector_global, [0, 2, 0], atol=VVSMALL)


@pytest.mark.integration
class TestRobotKinematics:
    def test_simple_robot_arm_forward_kinematics(self, root_frame: Frame):
        """Simulate simple 2-link robot arm forward kinematics"""

        base = Frame(parent=root_frame, name="base")

        shoulder = Frame(parent=base, name="shoulder").rotate_euler(z=90, degrees=True)

        elbow = (
            Frame(parent=shoulder, name="elbow")
            .translate(x=1, y=0, z=0)
            .rotate_euler(z=-45, degrees=True)
        )

        end_effector = Frame(parent=elbow, name="end_effector").translate(x=1, y=0, z=0)

        tcp = Point(0, 0, 0, frame=end_effector)
        tcp_global = tcp.to_global()

        cos45 = np.cos(np.radians(45))
        sin45 = np.sin(np.radians(45))
        expected_x = cos45
        expected_y = 1 + sin45
        expected_z = 0

        assert_allclose(tcp_global, [expected_x, expected_y, expected_z], atol=1e-10)

    def test_camera_calibration_scenario(self, root_frame: Frame):
        """Simulate camera mounted on robot scenario"""

        robot_base = Frame(parent=root_frame, name="robot").translate(x=0.5, y=0, z=0)

        camera_frame = (
            Frame(parent=robot_base, name="camera")
            .translate(x=0, y=0.2, z=0.3)
            .rotate_euler(x=90, degrees=True)
        )

        observed_point_camera = Point(0, 0, 1, frame=camera_frame)
        observed_point_global = observed_point_camera.to_global()

        expected_x = 0.5
        expected_y = 0.2 - 1.0
        expected_z = 0.3

        assert_allclose(
            observed_point_global,
            [expected_x, expected_y, expected_z],
            atol=VVSMALL,
        )


@pytest.mark.integration
class TestBatchOperations:
    def test_batch_transform_with_frame_hierarchy(self, root_frame: Frame):
        """Test batch transformation through frame hierarchy"""
        frame = (
            Frame(parent=root_frame, name="test")
            .translate(x=1, y=2, z=3)
            .rotate_euler(z=90, degrees=True)
        )

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        transformed = frame.batch_transform_points_global(points)

        expected = np.array([[1, 2, 3], [1, 3, 3], [0, 2, 3], [1, 2, 4]])

        assert_allclose(transformed, expected, atol=VVSMALL)

    def test_point_list_transformation(self, root_frame: Frame):
        """Test transforming multiple Point objects"""
        source_frame = Frame(parent=root_frame, name="source").translate(x=1)
        target_frame = Frame(parent=root_frame, name="target").translate(y=1)

        points = [
            Point(0, 0, 0, frame=source_frame),
            Point(1, 0, 0, frame=source_frame),
            Point(0, 1, 0, frame=source_frame),
        ]

        points_in_target = [p.to_frame(target_frame) for p in points]

        assert_allclose(points_in_target[0], [1, -1, 0], atol=VVSMALL)
        assert_allclose(points_in_target[1], [2, -1, 0], atol=VVSMALL)
        assert_allclose(points_in_target[2], [1, 0, 0], atol=VVSMALL)


@pytest.mark.integration
class TestFrameFreezingInHierarchy:
    def test_freeze_propagation_through_children(self, root_frame: Frame):
        """Test that freezing parent doesn't affect child creation"""
        parent = Frame(parent=root_frame, name="parent").translate(x=1)
        parent.freeze()

        child = Frame(parent=parent, name="child").translate(y=1)

        point = Point(0, 0, 0, frame=child)
        point_global = point.to_global()

        assert_allclose(point_global, [1, 1, 0], atol=VVSMALL)

    def test_frozen_frame_prevents_modifications(self, root_frame: Frame):
        """Test that frozen frames in hierarchy still transform correctly"""
        frozen = Frame(parent=root_frame, name="frozen").translate(x=1).freeze()

        child = Frame(parent=frozen, name="child").translate(y=1)

        point = Point(1, 1, 1, frame=child)
        point_global = point.to_global()

        assert_allclose(point_global, [2, 2, 1], atol=VVSMALL)


@pytest.mark.integration
class TestVectorOperations:
    def test_cross_product_in_different_frames(self, root_frame: Frame):
        """Test cross product after frame transformations"""
        frame = Frame(parent=root_frame, name="test").rotate_euler(z=90, degrees=True)

        v1_local = Vector(1, 0, 0, frame=frame)
        v2_local = Vector(0, 1, 0, frame=frame)

        v1_global = v1_local.to_global()
        v2_global = v2_local.to_global()

        cross_global = v1_global.cross(v2_global)

        assert_allclose(cross_global, [0, 0, 1], atol=VVSMALL)

    def test_vector_magnitude_invariant_under_translation(self, root_frame: Frame):
        """Test that vector magnitude is invariant under translation"""
        translated = Frame(parent=root_frame, name="translated").translate(
            x=100, y=200, z=300
        )

        vector_local = Vector(3, 4, 0, frame=translated)
        vector_global = vector_local.to_global()

        assert abs(vector_local.magnitude - vector_global.magnitude) < VVSMALL
        assert abs(vector_local.magnitude - 5.0) < VVSMALL


class TestBuildinIntegration:
    """Test operations between Python native types and Vector/Point."""

    def test_add_scalar_point_raises(self, root_frame: Frame):
        point = root_frame.point(0.0, 0.0, 0.0)

        # integer
        with pytest.raises(TypeError, match="unsupported operand"):
            point = point + 5

        # float
        with pytest.raises(TypeError, match="unsupported operand"):
            point = point + 5.0

        # complex
        with pytest.raises(TypeError, match="unsupported operand"):
            z = 5.0 + 1j
            point = point + z

    def test_add_scalar_vector_raises(self, root_frame: Frame):
        vector = root_frame.vector(1.0, 0.0, 0.0)

        # integer
        with pytest.raises(TypeError, match="unsupported operand"):
            vector = vector + 5

        # float
        with pytest.raises(TypeError, match="unsupported operand"):
            vector = vector + 5.0

        # complex
        with pytest.raises(TypeError, match="unsupported operand"):
            z = 5.0 + 1j
            vector = vector + z

    def test_subtract_scalar_point(self, root_frame: Frame):
        point = root_frame.point(0.0, 0.0, 0.0)

        # integer
        with pytest.raises(TypeError, match="unsupported operand"):
            point = point - 5

        # float
        with pytest.raises(TypeError, match="unsupported operand"):
            point = point - 5.0

        # complex
        with pytest.raises(TypeError, match="unsupported operand"):
            z = 5.0 + 1j
            point = point - z

    def test_subtract_scalar_vector(self, root_frame: Frame):
        vector = root_frame.vector(1.0, 0.0, 0.0)

        # integer
        with pytest.raises(TypeError, match="unsupported operand"):
            vector = vector - 5

        # float
        with pytest.raises(TypeError, match="unsupported operand"):
            vector = vector - 5.0

        # complex
        with pytest.raises(TypeError, match="unsupported operand"):
            z = 5.0 + 1j
            vector = vector - z

    def test_multiply_scalar_point(self, root_frame: Frame):
        # integer
        point = root_frame.point(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Scalar multiplication of Point is undefined"
        ):
            point = point * 5

        # float
        point = root_frame.point(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Scalar multiplication of Point is undefined"
        ):
            point = point * 5.0

        # complex
        point = root_frame.point(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Scalar multiplication of Point is undefined"
        ):
            z = 5.0 + 1j
            point = point * z

    def test_multiply_scalar_vector(self, root_frame: Frame):
        # integer
        vector = root_frame.vector(1.0, 1.0, 1.0)
        vector = vector * 5
        expected = root_frame.vector(5.0, 5.0, 5.0)
        assert isinstance(vector, Vector)
        assert_allclose(vector, expected)

        # float
        vector = root_frame.vector(1.0, 1.0, 1.0)
        vector = vector * 5.0
        expected = root_frame.vector(5.0, 5.0, 5.0)
        assert isinstance(vector, Vector)
        assert_allclose(vector, expected)

        # complex
        vector = root_frame.vector(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Complex number multiplication not supported"
        ):
            z = 5.0 + 1j
            vector = vector * z


@pytest.mark.integration
class TestClearTransformations:
    def test_clear_rotations_resets_to_identity(self, root_frame: Frame):
        """Test that clear_rotations resets frame to no rotation"""
        frame = (
            Frame(parent=root_frame, name="test")
            .rotate_euler(x=45, degrees=True)
            .rotate_euler(y=90, degrees=True)
            .rotate_euler(z=30, degrees=True)
        )

        frame.clear_rotations()

        vector_after = Vector(1, 0, 0, frame=frame).to_global()
        assert_allclose(vector_after, [1, 0, 0], atol=VVSMALL)

    def test_clear_translations_resets_to_origin(self, root_frame: Frame):
        """Test that clear_translations resets frame to no translation"""
        frame = (
            Frame(parent=root_frame, name="test")
            .translate(x=5, y=10, z=15)
            .translate(x=1, y=2, z=3)
        )

        point = Point(0, 0, 0, frame=frame)
        point_before = point.to_global()

        frame.clear_translations()

        point_after = Point(0, 0, 0, frame=frame).to_global()

        assert not np.allclose(
            np.array(point_before), np.array(point_after), atol=VVSMALL
        )
        assert_allclose(point_after, [0, 0, 0], atol=VVSMALL)

    def test_clear_scalings_resets_to_unit_scale(self, root_frame: Frame):
        """Test that clear_scalings resets frame to unit scale"""
        frame = Frame(parent=root_frame, name="test").scale(2.0).scale([3.0, 4.0, 5.0])

        vector = Vector(1, 1, 1, frame=frame)
        vector_before = vector.to_global()

        frame.clear_scalings()

        vector_after = Vector(1, 1, 1, frame=frame).to_global()

        assert not np.allclose(
            np.array(vector_before), np.array(vector_after), atol=VVSMALL
        )
        assert_allclose(np.array(vector_after), [1, 1, 1], atol=VVSMALL)

    def test_clear_all_transforms_resets_completely(self, root_frame: Frame):
        """Test that clear_all_transforms resets all transformations"""
        frame = (
            Frame(parent=root_frame, name="test")
            .scale(2.0)
            .rotate_euler(z=90, degrees=True)
            .translate(x=10, y=20, z=30)
        )

        point = Point(1, 2, 3, frame=frame)
        point_before = point.to_global()

        frame.clear_all_transforms()

        point_after = Point(1, 2, 3, frame=frame).to_global()

        assert not np.allclose(
            np.array(point_before), np.array(point_after), atol=VVSMALL
        )
        assert_allclose(np.array(point_after), [1, 2, 3], atol=VVSMALL)

    def test_clear_preserves_parent_transformations(self, root_frame: Frame):
        """Test that clearing child transforms does not affect parent"""
        parent = Frame(parent=root_frame, name="parent").translate(x=5, y=5, z=5)

        child = (
            Frame(parent=parent, name="child")
            .translate(x=1, y=2, z=3)
            .rotate_euler(z=45, degrees=True)
        )

        child.clear_all_transforms()

        point = Point(0, 0, 0, frame=child)
        point_global = point.to_global()

        assert_allclose(point_global, [5, 5, 5], atol=VVSMALL)

    def test_clear_invalidates_cache(self, root_frame: Frame):
        """Test that clearing transformations invalidates caches"""
        frame = Frame(parent=root_frame, name="test").translate(x=10)

        point_before = Point(0, 0, 0, frame=frame).to_global()
        assert_allclose(point_before, [10, 0, 0], atol=VVSMALL)

        frame.clear_translations()

        point_after = Point(0, 0, 0, frame=frame).to_global()
        assert_allclose(point_after, [0, 0, 0], atol=VVSMALL)

    def test_clear_rotations_preserves_other_transforms(self, root_frame: Frame):
        """Test that clear_rotations keeps scale and translation"""
        frame = (
            Frame(parent=root_frame, name="test")
            .scale(2.0)
            .rotate_euler(z=90, degrees=True)
            .translate(x=5, y=10, z=15)
        )

        frame.clear_rotations()

        vector = Vector(1, 0, 0, frame=frame)
        vector_global = vector.to_global()
        assert_allclose(vector_global, [2, 0, 0], atol=VVSMALL)

        point = Point(0, 0, 0, frame=frame)
        point_global = point.to_global()
        assert_allclose(point_global, [5, 10, 15], atol=VVSMALL)

    def test_clear_translations_preserves_other_transforms(self, root_frame: Frame):
        """Test that clear_translations keeps scale and rotation"""
        frame = (
            Frame(parent=root_frame, name="test")
            .scale(2.0)
            .rotate_euler(z=90, degrees=True)
            .translate(x=5, y=10, z=15)
        )

        frame.clear_translations()

        vector = Vector(1, 0, 0, frame=frame)
        vector_global = vector.to_global()
        assert_allclose(vector_global, [0, 2, 0], atol=VVSMALL)

    def test_clear_scalings_preserves_other_transforms(self, root_frame: Frame):
        """Test that clear_scalings keeps rotation and translation"""
        frame = (
            Frame(parent=root_frame, name="test")
            .scale(2.0)
            .rotate_euler(z=90, degrees=True)
            .translate(x=5, y=10, z=15)
        )

        frame.clear_scalings()

        vector = Vector(1, 0, 0, frame=frame)
        vector_global = vector.to_global()
        assert_allclose(vector_global, [0, 1, 0], atol=VVSMALL)

        point = Point(0, 0, 0, frame=frame)
        point_global = point.to_global()
        assert_allclose(point_global, [5, 10, 15], atol=VVSMALL)

    def test_clear_methods_return_self(self, root_frame: Frame):
        """Test that all clear methods return self for chaining"""
        frame = Frame(parent=root_frame, name="test")

        result1 = frame.clear_rotations()
        assert result1 is frame

        result2 = frame.clear_translations()
        assert result2 is frame

        result3 = frame.clear_scalings()
        assert result3 is frame

        result4 = frame.clear_all_transforms()
        assert result4 is frame

    def test_clear_methods_can_be_chained(self, root_frame: Frame):
        """Test that clear methods support method chaining"""
        frame = (
            Frame(parent=root_frame, name="test")
            .scale(2.0)
            .rotate_euler(z=90, degrees=True)
            .translate(x=5)
            .clear_rotations()
            .clear_scalings()
            .translate(y=10)
        )

        point = Point(0, 0, 0, frame=frame)
        point_global = point.to_global()
        assert_allclose(point_global, [5, 10, 0], atol=VVSMALL)

    def test_clear_on_frozen_frame_raises(self, root_frame: Frame):
        """Test that clearing frozen frame raises error"""
        frame = (
            Frame(parent=root_frame, name="test")
            .translate(x=5)
            .rotate_euler(z=45, degrees=True)
            .freeze()
        )

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_rotations()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_translations()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_scalings()

        with pytest.raises(RuntimeError, match="Cannot modify frozen frame"):
            frame.clear_all_transforms()

    def test_clear_affects_children_cache(self, root_frame: Frame):
        """Test that clearing parent invalidates child caches"""
        parent = Frame(parent=root_frame, name="parent").translate(x=5)
        child = Frame(parent=parent, name="child").translate(y=3)

        point_before = Point(0, 0, 0, frame=child).to_global()
        assert_allclose(point_before, [5, 3, 0], atol=VVSMALL)

        parent.clear_translations()

        point_after = Point(0, 0, 0, frame=child).to_global()
        assert_allclose(point_after, [0, 3, 0], atol=VVSMALL)


@pytest.mark.integration
class TestClearTransformationsHypothesis:
    """Property-based tests for clear transformation methods using Hypothesis"""

    @pytest.mark.hypothesis
    def test_clear_rotations_with_arbitrary_rotation_lists(self, root_frame: Frame):
        """Test clear_rotations with arbitrary lists of rotations"""

        @given(rotation_lists(min_size=1, max_size=5))
        def property_test(rotations_list):
            frame = Frame(parent=root_frame, name="test")

            for rot_args in rotations_list:
                frame.rotate_euler(**rot_args)

            frame.clear_rotations()

            vector_after = Vector(1, 0, 0, frame=frame).to_global()

            assert_allclose(vector_after, [1, 0, 0], atol=VVSMALL)

        property_test()

    @pytest.mark.hypothesis
    def test_clear_translations_with_arbitrary_translation_lists(
        self, root_frame: Frame
    ):
        """Test clear_translations with arbitrary lists of translations"""

        @given(translation_lists(min_size=1, max_size=5))
        def property_test(translations_list):
            frame = root_frame.make_child(name="test")

            for trans in translations_list:
                frame.translate(trans)

            frame.clear_translations()

            point_after = np.array(frame.point(0, 0, 0).to_global())

            assert_allclose(point_after, [0, 0, 0], atol=VVSMALL)

        property_test()

    @pytest.mark.hypothesis
    def test_clear_scalings_with_arbitrary_scaling_lists(self, root_frame: Frame):
        """Test clear_scalings with arbitrary lists of scalings"""

        @given(scaling_lists(min_size=1, max_size=5))
        def property_test(scalings_list):
            frame = Frame(parent=root_frame, name="test")

            for scale in scalings_list:
                frame.scale(scale)

            frame.clear_scalings()

            vector_after = frame.vector(1, 1, 1).to_global()

            assert_allclose(vector_after, [1, 1, 1], atol=VVSMALL)

        property_test()

    @pytest.mark.hypothesis
    def test_clear_all_with_arbitrary_transforms(self, root_frame: Frame):
        """Test clear_all_transforms with arbitrary transformation lists"""

        @given(
            translation_lists(min_size=0, max_size=3),
            rotation_lists(min_size=0, max_size=3),
            scaling_lists(min_size=0, max_size=3),
        )
        def property_test(translations_list, rotations_list, scalings_list):
            frame = Frame(parent=root_frame, name="test")

            for trans in translations_list:
                frame.translate(trans)
            for rot_args in rotations_list:
                frame.rotate_euler(**rot_args)
            for scale in scalings_list:
                frame.scale(scale)

            frame._cached_transform = None

            frame.clear_all_transforms()

            point = Point(1, 2, 3, frame=frame).to_global()
            assert_allclose(point, [1, 2, 3], atol=VVSMALL)

        property_test()

    @pytest.mark.hypothesis
    def test_clear_rotations_preserves_translations_and_scales(self, root_frame: Frame):
        """Test that clear_rotations preserves translations and scalings"""

        @given(
            translations(),
            scalings(),
            rotation_lists(min_size=1, max_size=3),
        )
        def property_test(translation, scaling, rotations_list):
            frame = Frame(parent=root_frame, name="test")

            frame.translate(translation)
            frame.scale(scaling)
            for rot_args in rotations_list:
                frame.rotate_euler(**rot_args)

            frame.clear_rotations()

            point = Point(0, 0, 0, frame=frame).to_global()
            assert_allclose(point, translation, atol=VVSMALL)

        property_test()

    @pytest.mark.hypothesis
    def test_clear_translations_preserves_rotations_and_scales(self, root_frame: Frame):
        """Test that clear_translations preserves rotations and scalings"""

        @given(
            scalings(),
            rotation_lists(min_size=1, max_size=3),
            translation_lists(min_size=1, max_size=3),
        )
        def property_test(scaling, rotations_list, translations_list):
            frame = Frame(parent=root_frame, name="test")

            for trans in translations_list:
                frame.translate(trans)
            for rot_args in rotations_list:
                frame.rotate_euler(**rot_args)
            frame.scale(scaling)
            frame.clear_translations()

            point = Point(0, 0, 0, frame=frame).to_global()
            assert_allclose(point, [0, 0, 0], atol=VVSMALL)

        property_test()

    @pytest.mark.hypothesis
    def test_clear_scalings_preserves_rotations_and_translations(
        self, root_frame: Frame
    ):
        """Test that clear_scalings preserves rotations and translations"""

        @given(
            translations(),
            rotation_lists(min_size=1, max_size=3),
            scaling_lists(min_size=1, max_size=3),
        )
        def property_test(translation, rotations_list, scalings_list):
            frame = Frame(parent=root_frame, name="test")

            frame._translations.append(translation)
            for rot_args in rotations_list:
                frame.rotate_euler(**rot_args)
            for scale in scalings_list:
                frame.scale(scale)

            frame.clear_scalings()

            point = Point(0, 0, 0, frame=frame).to_global()
            assert_allclose(point, translation, atol=VVSMALL)

        property_test()
