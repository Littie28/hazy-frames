from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats
from numpy.testing import assert_allclose

from hazy import Frame
from hazy.primitives import Point, Vector

coords = floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
small_coords = floats(
    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
)
angles = floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)


@pytest.mark.unit
class TestPointTransformations:
    @given(x=coords, y=coords, z=coords)
    def test_point_transform_to_same_frame_is_identity(self, x, y, z):
        """Transform point to its own frame should not change it"""
        frame = Frame(parent=Frame(parent=None, name="global"), name="test")
        point = Point(x, y, z, frame=frame)
        transformed = point.to_frame(frame)

        assert_allclose(point, transformed, atol=1e-10)
        assert transformed.frame is frame

    @given(
        x=coords, y=coords, z=coords, dx=small_coords, dy=small_coords, dz=small_coords
    )
    def test_point_transform_roundtrip(self, x, y, z, dx, dy, dz):
        """Transform point to different frame and back equals original"""
        root = Frame(parent=None, name="root")
        frame1 = Frame(parent=root, name="f1")
        frame2 = (
            Frame(parent=root, name="f2")
            .translate(x=dx, y=dy, z=dz)
            .rotate_euler(z=30, degrees=True)
        )

        point = Point(x, y, z, frame=frame1)
        roundtrip = point.to_frame(frame2).to_frame(frame1)

        assert_allclose(point, roundtrip, atol=1e-8)

    @given(x=coords, y=coords, z=coords, tx=small_coords)
    def test_point_translation_invariance_through_frames(self, x, y, z, tx):
        """Point should have same global coords regardless of frame definition"""
        global_frame = Frame(parent=None, name="global")
        translated_frame = Frame(parent=global_frame, name="translated").translate(x=tx)

        point_in_global = Point(x, y, z, frame=global_frame)
        point_in_local = Point(x - tx, y, z, frame=translated_frame)

        assert_allclose(
            point_in_global.to_global(),
            point_in_local.to_global(),
            atol=1e-10,
        )


@pytest.mark.unit
class TestVectorTransformations:
    @given(x=coords, y=coords, z=coords)
    def test_vector_transform_to_same_frame_is_identity(self, x, y, z):
        """Transform vector to its own frame should not change it"""
        frame = Frame(parent=Frame(parent=None, name="global"), name="test")
        vector = Vector(x, y, z, frame=frame)
        transformed = vector.to_frame(frame)

        assert_allclose(vector, transformed, atol=1e-10)
        assert transformed.frame is frame

    @given(
        x=coords,
        y=coords,
        z=coords,
        angle=angles,
        scale=floats(min_value=0.1, max_value=10),
    )
    def test_vector_transform_roundtrip(self, x, y, z, angle, scale):
        """Transform vector to different frame and back equals original"""
        root = Frame(parent=None, name="root")
        frame1 = Frame(parent=root, name="f1")
        frame2 = (
            Frame(parent=root, name="f2")
            .scale(scale)
            .rotate_euler(z=angle, degrees=True)
        )

        vector = Vector(x, y, z, frame=frame1)
        roundtrip = vector.to_frame(frame2).to_frame(frame1)

        assert_allclose(vector, roundtrip, atol=1e-8)

    @given(x=coords, y=coords, z=coords, tx=small_coords)
    def test_vector_translation_invariant(self, x, y, z, tx):
        """Vectors should be invariant to translation (w=0)"""
        global_frame = Frame(parent=None, name="global")
        translated_frame = Frame(parent=global_frame, name="translated").translate(x=tx)

        vector = Vector(x, y, z, frame=translated_frame)
        vector_global = vector.to_global()

        assert_allclose(vector, vector_global, atol=1e-10)


@pytest.mark.unit
class TestPointArithmetic:
    @given(
        x1=small_coords,
        y1=small_coords,
        z1=small_coords,
        x2=small_coords,
        y2=small_coords,
        z2=small_coords,
    )
    def test_point_minus_point_plus_point_identity(self, x1, y1, z1, x2, y2, z2):
        """(P - Q) + Q == P"""
        frame = Frame(parent=None, name="global")
        p = Point(x1, y1, z1, frame=frame)
        q = Point(x2, y2, z2, frame=frame)

        v = p - q
        result = v + q

        assert isinstance(v, Vector)
        assert isinstance(result, Point)
        assert_allclose(result, p, atol=1e-10)

    @given(
        x1=small_coords,
        y1=small_coords,
        z1=small_coords,
        vx=small_coords,
        vy=small_coords,
        vz=small_coords,
    )
    def test_point_plus_vector_minus_vector_identity(self, x1, y1, z1, vx, vy, vz):
        """(P + V) - V == P"""
        frame = Frame(parent=None, name="global")
        p = Point(x1, y1, z1, frame=frame)
        v = Vector(vx, vy, vz, frame=frame)

        result = (p + v) - v

        assert isinstance(result, Point)
        assert_allclose(result, p, atol=1e-10)

    def test_point_plus_point_raises_type_error(self):
        """Adding two points should raise TypeError"""
        frame = Frame(parent=None, name="global")
        p1 = Point(1, 2, 3, frame=frame)
        p2 = Point(4, 5, 6, frame=frame)

        with pytest.raises(TypeError, match="Can not add 2 Points"):
            p1 + p2


@pytest.mark.unit
class TestVectorArithmetic:
    @given(
        x1=small_coords,
        y1=small_coords,
        z1=small_coords,
        x2=small_coords,
        y2=small_coords,
        z2=small_coords,
    )
    def test_vector_addition_commutative(self, x1, y1, z1, x2, y2, z2):
        """V1 + V2 == V2 + V1"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(x1, y1, z1, frame=frame)
        v2 = Vector(x2, y2, z2, frame=frame)

        result1 = v1 + v2
        result2 = v2 + v1

        assert_allclose(result1, result2, atol=1e-10)

    @given(x=small_coords, y=small_coords, z=small_coords, scale=small_coords)
    def test_vector_scalar_multiplication_commutative(self, x, y, z, scale):
        """v * s == s * v"""
        frame = Frame(parent=None, name="global")
        v = Vector(x, y, z, frame=frame)

        result1 = v * scale
        result2 = scale * v

        assert_allclose(result1, result2, atol=1e-10)

    @given(
        x=small_coords,
        y=small_coords,
        z=small_coords,
        vx=small_coords,
        vy=small_coords,
        vz=small_coords,
    )
    def test_vector_subtraction_inverse_of_addition(self, x, y, z, vx, vy, vz):
        """(V1 + V2) - V2 == V1"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(x, y, z, frame=frame)
        v2 = Vector(vx, vy, vz, frame=frame)

        result = (v1 + v2) - v2

        assert_allclose(result, v1, atol=1e-10)

    def test_vector_minus_point_raises_type_error(self):
        """Subtracting point from vector should raise TypeError"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        p = Point(4, 5, 6, frame=frame)

        with pytest.raises(TypeError, match="Cannot subtract Point from Vector"):
            v - p


@pytest.mark.unit
class TestErrorHandling:
    def test_normalize_zero_vector_raises_runtime_error(self):
        """Normalizing zero vector should raise RuntimeError"""
        frame = Frame(parent=None, name="global")
        zero_vector = Vector(0, 0, 0, frame=frame)

        with pytest.raises(
            RuntimeError, match="Can not normalize vector with zero length"
        ):
            zero_vector.normalize()

    def test_equality_with_non_geometric_type_raises(self):
        """Comparing with incompatible type should raise ValueError"""
        frame = Frame(parent=None, name="global")
        point = Point(1, 2, 3, frame=frame)

        with pytest.raises(ValueError, match="Can not compare"):
            point == "not a point"

        with pytest.raises(ValueError, match="Can not compare"):
            point == 42


@pytest.mark.unit
class TestVectorProperties:
    @given(
        x=floats(min_value=-100, max_value=100, allow_nan=False),
        y=floats(min_value=-100, max_value=100, allow_nan=False),
        z=floats(min_value=-100, max_value=100, allow_nan=False),
    )
    def test_normalized_vector_has_unit_magnitude(self, x, y, z):
        """Normalized vector should have magnitude 1"""
        frame = Frame(parent=None, name="global")
        v = Vector(x, y, z, frame=frame)

        if not v.is_zero:
            normalized = v.copy().normalize()
            assert abs(normalized.magnitude - 1.0) < 1e-10

    @given(x=small_coords, y=small_coords, z=small_coords)
    def test_magnitude_is_non_negative(self, x, y, z):
        """Vector magnitude should always be >= 0"""
        frame = Frame(parent=None, name="global")
        v = Vector(x, y, z, frame=frame)

        assert v.magnitude >= 0

    @given(x=small_coords, y=small_coords, z=small_coords, scale=small_coords)
    def test_magnitude_scales_linearly(self, x, y, z, scale):
        """Scaling vector should scale magnitude by same factor"""
        frame = Frame(parent=None, name="global")
        v = Vector(x, y, z, frame=frame)

        if not v.is_zero:
            scaled = v * scale
            assert abs(scaled.magnitude - abs(scale) * v.magnitude) < 1e-8

    @given(
        x1=small_coords,
        y1=small_coords,
        z1=small_coords,
        x2=small_coords,
        y2=small_coords,
        z2=small_coords,
    )
    def test_cross_product_perpendicular(self, x1, y1, z1, x2, y2, z2):
        """Cross product should be perpendicular to both input vectors"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(x1, y1, z1, frame=frame)
        v2 = Vector(x2, y2, z2, frame=frame)

        cross = v1.cross(v2)

        if not v1.is_zero and not v2.is_zero and not cross.is_zero:
            dot1 = np.dot(cross, v1)
            dot2 = np.dot(cross, v2)
            assert abs(dot1) < 1e-8
            assert abs(dot2) < 1e-8

    @given(x=small_coords, y=small_coords, z=small_coords)
    def test_vector_negation(self, x, y, z):
        """Negating vector should invert all components"""
        frame = Frame(parent=None, name="global")
        v = Vector(x, y, z, frame=frame)
        neg_v = -v

        assert_allclose(neg_v, -v, atol=1e-10)
        assert neg_v.frame is v.frame


@pytest.mark.unit
class TestEquality:
    @given(x=coords, y=coords, z=coords)
    def test_point_equality_is_reflexive(self, x, y, z):
        """Point should equal itself"""
        frame = Frame(parent=None, name="global")
        p = Point(x, y, z, frame=frame)

        assert p == p

    @given(x=coords, y=coords, z=coords)
    def test_vector_equality_is_reflexive(self, x, y, z):
        """Vector should equal itself"""
        frame = Frame(parent=None, name="global")
        v = Vector(x, y, z, frame=frame)

        assert v == v

    @given(x=coords, y=coords, z=coords, tx=small_coords)
    def test_same_point_in_different_frames(self, x, y, z, tx):
        """Same global point expressed in different frames should be equal"""
        global_frame = Frame(parent=None, name="global")
        translated = Frame(parent=global_frame, name="translated").translate(x=tx)

        p1 = Point(x, y, z, frame=global_frame)
        p2 = Point(x - tx, y, z, frame=translated)

        assert p1 == p2


@pytest.mark.unit
class TestFrameChecking:
    def test_point_addition_different_frames_raises(self):
        """Operations on primitives from different frames should raise"""
        frame1 = Frame(parent=Frame(parent=None, name="global"), name="f1")
        frame2 = Frame(parent=Frame(parent=None, name="global"), name="f2")

        p = Point(1, 2, 3, frame=frame1)
        v = Vector(1, 0, 0, frame=frame2)

        with pytest.raises(RuntimeError, match="coordinate system"):
            p + v

    def test_vector_cross_different_frames_raises(self):
        """Cross product on vectors from different frames should raise"""
        frame1 = Frame(parent=Frame(parent=None, name="global"), name="f1")
        frame2 = Frame(parent=Frame(parent=None, name="global"), name="f2")

        v1 = Vector(1, 0, 0, frame=frame1)
        v2 = Vector(0, 1, 0, frame=frame2)

        with pytest.raises(RuntimeError, match="coordinate system"):
            v1.cross(v2)


@pytest.mark.unit
class TestSpecialMethods:
    def test_point_indexing(self):
        """Points should be indexable"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)

        assert p[0] == 1
        assert p[1] == 2
        assert p[2] == 3

    def test_vector_iteration(self):
        """Vectors should be iterable"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)

        coords = list(v)
        assert coords == [1, 2, 3]

    def test_point_array_conversion(self):
        """Points should convert to numpy arrays"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)

        arr = np.array(p)
        assert_allclose(arr, [1, 2, 3])

    def test_vector_copy(self):
        """Copying should create independent instance"""
        frame = Frame(parent=None, name="global")
        v1 = Vector(1, 2, 3, frame=frame)
        v2 = v1.copy()

        assert v1 == v2
        assert v1 is not v2
        assert_allclose(v1, v2)


@pytest.mark.unit
class TestFactoryMethods:
    def test_vector_create_unit_x(self):
        frame = Frame(parent=None, name="global")
        v = Vector.unit_x(frame)

        assert_allclose(v, [1, 0, 0])
        assert v.frame is frame

    def test_vector_create_unit_y(self):
        frame = Frame(parent=None, name="global")
        v = Vector.unit_y(frame)

        assert_allclose(v, [0, 1, 0])
        assert v.frame is frame

    def test_vector_create_unit_z(self):
        frame = Frame(parent=None, name="global")
        v = Vector.unit_z(frame)

        assert_allclose(v, [0, 0, 1])
        assert v.frame is frame

    def test_vector_create_nan(self):
        frame = Frame(parent=None, name="global")
        v = Vector.nan(frame)

        assert np.isnan(v.x)
        assert np.isnan(v.y)
        assert np.isnan(v.z)
        assert v.frame is frame

    def test_point_create_origin(self):
        frame = Frame(parent=None, name="global")
        p = Point.create_origin(frame)

        assert_allclose(p, [0, 0, 0])
        assert p.frame is frame

    def test_point_create_nan(self):
        frame = Frame(parent=None, name="global")
        p = Point.create_nan(frame)

        assert np.isnan(p.x)
        assert np.isnan(p.y)
        assert np.isnan(p.z)
        assert p.frame is frame

    def test_point_from_array(self):
        frame = Frame(parent=None, name="global")
        points_array = np.array([[1, 3, 5], [2, 4, 6]])

        points = Point.list_from_array(points_array, frame)

        assert len(points) == 2
        assert_allclose(points[0], [1, 3, 5])
        assert_allclose(points[1], [2, 4, 6])
        assert all(p.frame is frame for p in points)


@pytest.mark.unit
class TestNumpyInteroperability:
    def test_vector_add_numpy_array(self):
        """Vector + numpy array should work"""
        frame = Frame(parent=None, name="global")
        v = Vector(1, 2, 3, frame=frame)
        arr = np.array([1, 0, 0])
        with pytest.raises(TypeError):
            v + arr

    def test_point_add_numpy_array(self):
        """Point + numpy array should work"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)
        arr = np.array([1, 0, 0])

        with pytest.raises(TypeError):
            p + arr

    def test_point_subtract_numpy_array(self):
        """Point - numpy array should work"""
        frame = Frame(parent=None, name="global")
        p = Point(1, 2, 3, frame=frame)
        arr = np.array([1, 0, 0])

        with pytest.raises(TypeError):
            p - arr
