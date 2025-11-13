from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazy import Frame
from hazy.constants import VVSMALL
from hazy.primitives import Point, Vector


@pytest.mark.integration
class TestMultiFrameTransformations:
    def test_point_transformation_through_hierarchy(self):
        """Test point transformation through multi-level frame hierarchy"""
        global_frame = Frame(parent=None, name="global")
        robot_base = Frame(parent=global_frame, name="robot_base").translate(
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

    def test_vector_transformation_preserves_direction(self):
        """Test that vectors transform correctly through rotated frames"""
        global_frame = Frame(parent=None, name="global")
        rotated = Frame(parent=global_frame, name="rotated").rotate_euler(
            z=90, degrees=True
        )

        vector_local = Vector(1, 0, 0, frame=rotated)
        vector_global = vector_local.to_global()

        assert_allclose(vector_global, [0, 1, 0], atol=VVSMALL)

    def test_point_arithmetic_across_frame_hierarchy(self):
        """Test point arithmetic when transforming between frames"""
        global_frame = Frame(parent=None, name="global")
        frame_a = Frame(parent=global_frame, name="frame_a").translate(x=1)
        frame_b = Frame(parent=global_frame, name="frame_b").translate(y=1)

        point_a = Point(0, 0, 0, frame=frame_a)
        point_b = Point(0, 0, 0, frame=frame_b)

        point_a_global = point_a.to_global()
        point_b_global = point_b.to_global()

        displacement = point_b_global - point_a_global

        assert isinstance(displacement, Vector)
        assert_allclose(displacement, [-1, 1, 0], atol=VVSMALL)


@pytest.mark.integration
class TestComplexTransformations:
    def test_combined_rotation_translation_scale(self):
        """Test complex transformation combining all types"""
        global_frame = Frame(parent=None, name="global")
        complex_frame = (
            Frame(parent=global_frame, name="complex")
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

    def test_deep_hierarchy_accumulation(self):
        """Test transformation accumulation through deep hierarchy"""
        global_frame = Frame(parent=None, name="global")

        current_frame = global_frame
        for i in range(5):
            current_frame = Frame(parent=current_frame, name=f"level_{i}").translate(
                x=1, y=0, z=0
            )

        point = Point(0, 0, 0, frame=current_frame)
        point_global = point.to_global()

        assert_allclose(point_global, [5, 0, 0], atol=VVSMALL)

    def test_non_uniform_scale_with_rotation(self):
        """Test that non-uniform scaling works correctly with rotation"""
        global_frame = Frame(parent=None, name="global")
        frame = (
            Frame(parent=global_frame, name="test")
            .scale((2, 3, 1))
            .rotate_euler(z=90, degrees=True)
        )

        vector = Vector(1, 0, 0, frame=frame)
        vector_global = vector.to_global()

        assert_allclose(vector_global, [0, 2, 0], atol=VVSMALL)


@pytest.mark.integration
class TestRobotKinematics:
    def test_simple_robot_arm_forward_kinematics(self):
        """Simulate simple 2-link robot arm forward kinematics"""
        global_frame = Frame(parent=None, name="global")

        base = Frame(parent=global_frame, name="base")

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

    def test_camera_calibration_scenario(self):
        """Simulate camera mounted on robot scenario"""
        global_frame = Frame(parent=None, name="global")

        robot_base = Frame(parent=global_frame, name="robot").translate(x=0.5, y=0, z=0)

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
    def test_batch_transform_with_frame_hierarchy(self):
        """Test batch transformation through frame hierarchy"""
        global_frame = Frame(parent=None, name="global")
        frame = (
            Frame(parent=global_frame, name="test")
            .translate(x=1, y=2, z=3)
            .rotate_euler(z=90, degrees=True)
        )

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        transformed = frame.batch_transform_points_global(points)

        expected = np.array([[1, 2, 3], [1, 3, 3], [0, 2, 3], [1, 2, 4]])

        assert_allclose(transformed, expected, atol=VVSMALL)

    def test_point_list_transformation(self):
        """Test transforming multiple Point objects"""
        global_frame = Frame(parent=None, name="global")
        source_frame = Frame(parent=global_frame, name="source").translate(x=1)
        target_frame = Frame(parent=global_frame, name="target").translate(y=1)

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
    def test_freeze_propagation_through_children(self):
        """Test that freezing parent doesn't affect child creation"""
        global_frame = Frame(parent=None, name="global")
        parent = Frame(parent=global_frame, name="parent").translate(x=1)
        parent.freeze()

        child = Frame(parent=parent, name="child").translate(y=1)

        point = Point(0, 0, 0, frame=child)
        point_global = point.to_global()

        assert_allclose(point_global, [1, 1, 0], atol=VVSMALL)

    def test_frozen_frame_prevents_modifications(self):
        """Test that frozen frames in hierarchy still transform correctly"""
        global_frame = Frame(parent=None, name="global")
        frozen = Frame(parent=global_frame, name="frozen").translate(x=1).freeze()

        child = Frame(parent=frozen, name="child").translate(y=1)

        point = Point(1, 1, 1, frame=child)
        point_global = point.to_global()

        assert_allclose(point_global, [2, 2, 1], atol=VVSMALL)


@pytest.mark.integration
class TestVectorOperations:
    def test_cross_product_in_different_frames(self):
        """Test cross product after frame transformations"""
        global_frame = Frame(parent=None, name="global")
        frame = Frame(parent=global_frame, name="test").rotate_euler(z=90, degrees=True)

        v1_local = Vector(1, 0, 0, frame=frame)
        v2_local = Vector(0, 1, 0, frame=frame)

        v1_global = v1_local.to_global()
        v2_global = v2_local.to_global()

        cross_global = v1_global.cross(v2_global)

        assert_allclose(cross_global, [0, 0, 1], atol=VVSMALL)

    def test_vector_magnitude_invariant_under_translation(self):
        """Test that vector magnitude is invariant under translation"""
        global_frame = Frame(parent=None, name="global")
        translated = Frame(parent=global_frame, name="translated").translate(
            x=100, y=200, z=300
        )

        vector_local = Vector(3, 4, 0, frame=translated)
        vector_global = vector_local.to_global()

        assert abs(vector_local.magnitude - vector_global.magnitude) < VVSMALL
        assert abs(vector_local.magnitude - 5.0) < VVSMALL


class TestBuildinIntegration:
    """Test operations between Python native types and Vector/Point."""

    def test_add_scalar_point_raises(self):
        point = Frame(parent=None, name="global").point(0.0, 0.0, 0.0)

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

    def test_add_scalar_vector_raises(self):
        vector = Frame(parent=None, name="global").vector(1.0, 0.0, 0.0)

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

    def test_subtract_scalar_point(self):
        point = Frame(parent=None, name="global").point(0.0, 0.0, 0.0)

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

    def test_subtract_scalar_vector(self):
        vector = Frame(parent=None, name="global").vector(1.0, 0.0, 0.0)

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

    def test_multiply_scalar_point(self):
        # integer
        point = Frame(parent=None, name="global").point(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Scalar multiplication of Point is undefined"
        ):
            point = point * 5

        # float
        point = Frame(parent=None, name="global").point(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Scalar multiplication of Point is undefined"
        ):
            point = point * 5.0

        # complex
        point = Frame(parent=None, name="global").point(1.0, 1.0, 1.0)
        with pytest.raises(
            TypeError, match="Scalar multiplication of Point is undefined"
        ):
            z = 5.0 + 1j
            point = point * z

    def test_multiply_scalar_vector(self):
        # integer
        vector = Frame(parent=None, name="global").vector(1.0, 1.0, 1.0)
        vector = vector * 5
        expected = Frame(parent=None, name="global").vector(5.0, 5.0, 5.0)
        assert isinstance(vector, Vector)
        assert_allclose(vector, expected)

        # float
        vector = Frame(parent=None, name="global").vector(1.0, 1.0, 1.0)
        vector = vector * 5.0
        expected = Frame(parent=None, name="global").vector(5.0, 5.0, 5.0)
        assert isinstance(vector, Vector)
        assert_allclose(vector, expected)

        # complex
        vector = Frame(parent=None, name="global").vector(1.0, 1.0, 1.0)
        with pytest.raises(TypeError, match="unsupported operand"):
            z = 5.0 + 1j
            vector = vector * z


@pytest.mark.integration
class TestNumpyIntegration:
    def test_numpy_add_point(self):
        """Test if"""
        point = Frame(parent=None, name="global").point(0.0, 0.0, 0.0)
        array = np.array((1.0, 2.0, 3.0))

        with pytest.raises(TypeError):
            point = point + array
