from __future__ import annotations

from math import isclose

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazy import Frame
from hazy.primitives import Point, Vector


@pytest.mark.numpy
class TestNumpyScalarOperations:
    """Test NumPy scalar (np.float64, etc.) operations with geometric primitives"""

    def test_vector_multiply_numpy_scalar(self):
        """Vector * np.float64 should work"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        scalar = np.float64(2.0)

        result = v * scalar
        assert isinstance(result, Vector)
        assert_allclose(result, [2, 4, 6])
        assert result.frame is frame

    def test_numpy_scalar_multiply_vector(self):
        """np.float64 * Vector should work (commutative)"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        scalar = np.float64(0.5)

        result = scalar * v
        assert isinstance(result, Vector)
        assert_allclose(result, [0.5, 1.0, 1.5])
        assert result.frame is frame

    def test_vector_divide_numpy_scalar(self):
        """Vector / np.float64 should work"""
        frame = Frame(parent=None, name="global")
        v = Vector(2, 4, 6, frame=frame)
        scalar = np.float64(2.0)

        result = v / scalar
        assert isinstance(result, Vector)
        assert_allclose(result, [1, 2, 3])
        assert result.frame is frame

    def test_vector_multiply_int32(self):
        """Vector * np.int32 should work"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        scalar = np.int32(3)

        result = v * scalar
        assert isinstance(result, Vector)
        assert_allclose(result, [3, 6, 9])

    def test_point_multiply_numpy_scalar_raises(self):
        """Point * np.float64 should raise TypeError"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)
        scalar = np.float64(2.0)

        with pytest.raises(TypeError, match="Scalar multiplication"):
            p * scalar

    def test_numpy_scalar_multiply_point_raises(self):
        """np.float64 * Point should raise TypeError"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)
        scalar = np.float64(2.0)

        with pytest.raises(TypeError):
            scalar * p


@pytest.mark.numpy
class TestNumpyArrayConstruction:
    """Test creating geometric primitives from NumPy arrays"""

    def test_vector_from_array_list(self):
        """Vector.from_array should work with list"""
        frame = Frame(parent=None, name="global")
        v = Vector.from_array([1, 2, 3], frame)

        assert isinstance(v, Vector)
        assert_allclose(v, [1, 2, 3])
        assert v.frame is frame

    def test_vector_from_array_numpy(self):
        """Vector.from_array should work with numpy array"""
        frame = Frame(parent=None, name="global")
        arr = np.array([4, 5, 6])
        v = Vector.from_array(arr, frame)

        assert isinstance(v, Vector)
        assert_allclose(v, [4, 5, 6])
        assert v.frame is frame

    def test_point_from_array_list(self):
        """Point.from_array should work with list"""
        frame = Frame(parent=None, name="global")
        p = Point.from_array([1, 2, 3], frame)

        assert isinstance(p, Point)
        assert_allclose(p, [1, 2, 3])
        assert p.frame is frame

    def test_point_from_array_numpy(self):
        """Point.from_array should work with numpy array"""
        frame = Frame(parent=None, name="global")
        arr = np.array([7, 8, 9])
        p = Point.from_array(arr, frame)

        assert isinstance(p, Point)
        assert_allclose(p, [7, 8, 9])
        assert p.frame is frame

    def test_from_array_wrong_shape_raises(self):
        """from_array with wrong shape should raise ValueError"""
        frame = Frame(parent=None, name="global")

        with pytest.raises(ValueError, match="Expected 3 coordinates"):
            Vector.from_array([1, 2], frame)

        with pytest.raises(ValueError, match="Expected 3 coordinates"):
            Point.from_array([1, 2, 3, 4], frame)

    def test_from_array_flattens_input(self):
        """from_array should flatten multi-dimensional input"""
        frame = Frame(parent=None, name="global")
        arr = np.array([[1, 2, 3]])

        v = Vector.from_array(arr, frame)
        assert_allclose(v, [1, 2, 3])

    def test_frame_vector_from_scalars(self):
        """frame.vector(x, y, z) should work"""
        frame = Frame(parent=None, name="global")
        v = frame.vector(1, 2, 3)

        assert isinstance(v, Vector)
        assert_allclose(v, [1, 2, 3])
        assert v.frame is frame

    def test_frame_vector_from_array(self):
        """frame.vector([x, y, z]) should work"""
        frame = Frame(parent=None, name="global")
        v = frame.vector([1, 2, 3])

        assert isinstance(v, Vector)
        assert_allclose(v, [1, 2, 3])
        assert v.frame is frame

    def test_frame_vector_from_numpy_array(self):
        """frame.vector(np.array([x, y, z])) should work"""
        frame = Frame(parent=None, name="global")
        v = frame.vector(np.array([4, 5, 6]))

        assert isinstance(v, Vector)
        assert_allclose(v, [4, 5, 6])
        assert v.frame is frame

    def test_frame_point_from_scalars(self):
        """frame.point(x, y, z) should work"""
        frame = Frame(parent=None, name="global")
        p = frame.point(1, 2, 3)

        assert isinstance(p, Point)
        assert_allclose(p, [1, 2, 3])
        assert p.frame is frame

    def test_frame_point_from_array(self):
        """frame.point([x, y, z]) should work"""
        frame = Frame(parent=None, name="global")
        p = frame.point([7, 8, 9])

        assert isinstance(p, Point)
        assert_allclose(p, [7, 8, 9])
        assert p.frame is frame

    def test_frame_vector_mixed_args_raises(self):
        """frame.vector with mixed args should raise"""
        frame = Frame(parent=None, name="global")

        with pytest.raises(ValueError):
            frame.vector(1, 2)

        with pytest.raises(ValueError):
            frame.vector(1)


@pytest.mark.numpy
class TestNumpyStackOperations:
    """Test np.stack, np.vstack, np.hstack operations"""

    def test_vstack_vectors(self):
        """np.vstack should stack vectors vertically"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = Vector(4, 5, 6, frame=frame)
        v3 = Vector(7, 8, 9, frame=frame)

        result = np.vstack([v1, v2, v3])

        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_allclose(result, expected)
        assert result.shape == (3, 3)

    def test_vstack_points(self):
        """np.vstack should stack points vertically"""
        frame = Frame(parent=None, name="global")
        p1 = Point(1, 2, 3, frame=frame)
        p2 = Point(4, 5, 6, frame=frame)

        result = np.vstack([p1, p2])

        expected = np.array([[1, 2, 3], [4, 5, 6]])
        assert_allclose(result, expected)
        assert result.shape == (2, 3)

    def test_hstack_vectors(self):
        """np.hstack should stack vectors horizontally"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = Vector(4, 5, 6, frame=frame)

        result = np.hstack([v1, v2])

        expected = np.array([1, 2, 3, 4, 5, 6])
        assert_allclose(result, expected)
        assert result.shape == (6,)

    def test_hstack_points(self):
        """np.hstack should stack points horizontally"""
        frame = Frame(parent=None, name="global")
        p1 = Point(1, 2, 3, frame=frame)
        p2 = Point(4, 5, 6, frame=frame)
        p3 = Point(7, 8, 9, frame=frame)

        result = np.hstack([p1, p2, p3])

        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert_allclose(result, expected)

    def test_stack_vectors_axis0(self):
        """np.stack with axis=0 should stack as rows"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = Vector(4, 5, 6, frame=frame)

        result = np.stack([v1, v2], axis=0)

        expected = np.array([[1, 2, 3], [4, 5, 6]])
        assert_allclose(result, expected)

    def test_stack_vectors_axis1(self):
        """np.stack with axis=1 should stack as columns"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = Vector(4, 5, 6, frame=frame)

        result = np.stack([v1, v2], axis=1)

        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert_allclose(result, expected)

    def test_stack_single_element(self):
        """np.vstack with single element should work"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        result = np.vstack([v])

        expected = np.array([[1, 2, 3]])
        assert_allclose(result, expected)

    def test_vstack_different_frames_raises(self):
        """np.vstack with different frames should raise"""
        frame1 = Frame(parent=None, name="f1")
        frame2 = Frame(parent=None, name="f2")
        v1 = Vector(1, 2, 3, frame=frame1)
        v2 = Vector(4, 5, 6, frame=frame2)

        with pytest.raises(RuntimeError, match="coordinate system"):
            np.vstack([v1, v2])

    def test_vstack_mixed_types_raises(self):
        """np.vstack with mixed Point/Vector should raise"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        p = Point(4, 5, 6, frame=frame)

        with pytest.raises(TypeError, match="same type"):
            np.vstack([v, p])

    def test_hstack_mixed_types_raises(self):
        """np.hstack with mixed Point/Vector should raise"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        p = Point(4, 5, 6, frame=frame)

        with pytest.raises(TypeError, match="same type"):
            np.hstack([v, p])

    def test_stack_empty_raises(self):
        """np.vstack with empty list should raise"""
        with pytest.raises(ValueError):
            np.vstack([])


@pytest.mark.numpy
class TestNumpyLinalgFunctions:
    """Test np.linalg.* functions"""

    def test_norm_vector(self):
        """np.linalg.norm should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(3, 4, 0, frame=frame)

        result = np.linalg.norm(v)

        assert isclose(result, 5.0)

    def test_norm_vector_equals_magnitude(self):
        """np.linalg.norm(v) should equal v.magnitude"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        assert isclose(np.linalg.norm(v), v.magnitude)

    def test_norm_zero_vector(self):
        """np.linalg.norm of zero vector should be 0"""
        frame = Frame(parent=None, name="global")
        v = Vector(0, 0, 0, frame=frame)

        result = np.linalg.norm(v)

        assert isclose(result, 0.0)

    def test_norm_point_raises(self):
        """np.linalg.norm should raise for Point"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)

        with pytest.raises(TypeError, match="undefined"):
            np.linalg.norm(p)

    def test_norm_with_ord_parameter_raises(self):
        """np.linalg.norm with ord parameter should raise"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        with pytest.raises(TypeError, match="ord parameter not supported"):
            np.linalg.norm(v, ord=1)

    def test_norm_with_axis_parameter_raises(self):
        """np.linalg.norm with axis parameter should raise"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        with pytest.raises(TypeError, match="axis parameter not supported"):
            np.linalg.norm(v, axis=0)

    def test_norm_with_keepdims_parameter_raises(self):
        """np.linalg.norm with keepdims parameter should raise"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        with pytest.raises(TypeError, match="keepdims parameter not supported"):
            np.linalg.norm(v, keepdims=True)

    def test_dot_vectors(self):
        """np.dot should work with two Vectors"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = Vector(4, 5, 6, frame=frame)

        result = np.dot(v1, v2)

        expected = 1 * 4 + 2 * 5 + 3 * 6
        assert isclose(result, expected)

    def test_dot_orthogonal_vectors(self):
        """np.dot of orthogonal vectors should be 0"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 0, 0, frame=frame)
        v2 = Vector(0, 1, 0, frame=frame)

        result = np.dot(v1, v2)

        assert isclose(result, 0.0)

    def test_cross_vectors(self):
        """np.cross should work with two Vectors"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 0, 0, frame=frame)
        v2 = Vector(0, 1, 0, frame=frame)

        result = np.cross(v1, v2)

        assert isinstance(result, Vector)
        assert_allclose(result, [0, 0, 1])
        assert result.frame is frame

    def test_cross_parallel_vectors(self):
        """np.cross of parallel vectors should be zero"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = Vector(2, 4, 6, frame=frame)

        result = np.cross(v1, v2)

        assert isinstance(result, Vector)
        assert_allclose(result, [0, 0, 0], atol=1e-10)


@pytest.mark.numpy
class TestNumpyUniversalFunctions:
    """Test NumPy ufunc behavior"""

    def test_floor_vector(self):
        """np.floor should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(1.7, 2.3, 3.9, frame=frame)

        result = np.floor(v)

        assert isinstance(result, Vector)
        assert_allclose(result, [1, 2, 3])
        assert result.frame is frame

    def test_ceil_vector(self):
        """np.ceil should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(1.1, 2.5, 3.9, frame=frame)

        result = np.ceil(v)

        assert isinstance(result, Vector)
        assert_allclose(result, [2, 3, 4])
        assert result.frame is frame

    def test_round_vector(self):
        """np.round should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(1.234, 2.567, 3.891, frame=frame)

        result = np.round(v, decimals=1)

        assert isinstance(result, Vector)
        assert_allclose(result, [1.2, 2.6, 3.9])
        assert result.frame is frame

    def test_absolute_vector(self):
        """np.absolute should return magnitude for Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(3, 4, 0, frame=frame)

        result = np.absolute(v)

        assert isclose(result, 5.0)

    def test_absolute_point_raises(self):
        """np.absolute should raise for Point"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)

        with pytest.raises(TypeError, match="geometrically undefined"):
            np.absolute(p)

    def test_isnan_vector(self):
        """np.isnan should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, np.nan, 3, frame=frame)

        result = np.isnan(v)

        assert_allclose(result, [False, True, False])

    def test_isfinite_vector(self):
        """np.isfinite should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, np.inf, 3, frame=frame)

        result = np.isfinite(v)

        assert_allclose(result, [True, False, True])


@pytest.mark.numpy
class TestNumpyArrayConversion:
    """Test __array__ protocol and conversions"""

    def test_vector_to_array(self):
        """np.array(vector) should return coordinates"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        arr = np.array(v)

        assert isinstance(arr, np.ndarray)
        assert_allclose(arr, [1, 2, 3])
        assert arr.shape == (3,)

    def test_point_to_array(self):
        """np.array(point) should return coordinates"""
        frame = Frame(parent=None, name="global")
        p = Point(4, 5, 6, frame=frame)

        arr = np.array(p)

        assert isinstance(arr, np.ndarray)
        assert_allclose(arr, [4, 5, 6])
        assert arr.shape == (3,)

    def test_asarray_vector(self):
        """np.asarray should work with Vector"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        arr = np.asarray(v)

        assert_allclose(arr, [1, 2, 3])

    def test_copy_vector(self):
        """np.copy should create independent copy"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        v_copy = np.copy(v)

        assert isinstance(v_copy, Vector)
        assert v_copy == v
        assert v_copy is not v
        assert v_copy.frame is v.frame
